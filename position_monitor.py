#!/usr/bin/env python3
"""
Position monitor — runs every 30s to manage open positions.
Handles TP1 partials, trailing stops, reconciliation with exchange, PnL tracking.
"""

import time
from datetime import datetime, timezone
from typing import Dict, Optional
from pybit.unified_trading import HTTP
from config import Config, logger, trade_logger
from trading_state import TradingState
from risk_validator import RiskValidator
from telegram_notifier import TelegramNotifier


class PositionMonitor:
    """Monitors and manages open positions."""

    def __init__(self, session: HTTP, state: TradingState,
                 risk_validator: RiskValidator, telegram: TelegramNotifier):
        self.session = session
        self.state = state
        self.risk = risk_validator
        self.telegram = telegram

    def check_positions(self):
        """Main monitoring loop — reconcile with exchange and manage positions."""
        # Reconcile: detect orphaned exchange positions not in local state
        if not self.state.open_positions:
            self._check_orphaned_positions()
            return

        for pos in self.state.open_positions[:]:
            symbol = pos['symbol']
            current_price = self._get_current_price(symbol)
            if not current_price:
                continue

            # Reconcile with Bybit
            bybit_size = self._get_exchange_position_size(symbol)

            # Position closed by exchange (SL/TP hit)
            if bybit_size == 0:
                self._handle_closed_position(pos)
                continue

            # Update qty if exchange shows different size
            if bybit_size < pos['qty'] and bybit_size > 0:
                pos['qty'] = bybit_size

            # TP1 partial management
            if not pos.get('partial_filled', False):
                self._check_tp1_partial(pos, current_price)

            # Trailing stop after TP1
            if pos.get('partial_filled', False):
                self._update_trailing_stop(pos, current_price)

            # Max hold time check
            ts = pos.get('timestamp', '')
            if ts:
                entry_time = datetime.fromisoformat(ts)
                elapsed_h = (datetime.now(timezone.utc) - entry_time).total_seconds() / 3600
                if elapsed_h >= Config.MAX_HOLD_HOURS:
                    logger.warning(f"[{symbol}] Max hold time ({Config.MAX_HOLD_HOURS}h) reached")
                    self._force_close(pos, 'max_hold_time')

    def _check_tp1_partial(self, pos: Dict, current_price: float):
        """Check if price hit TP1 and execute 50% partial close."""
        tp1 = pos.get('take_profit_1', 0)
        if tp1 <= 0:
            return

        hit_tp1 = False
        if pos['side'] == 'Buy' and current_price >= tp1:
            hit_tp1 = True
        elif pos['side'] == 'Sell' and current_price <= tp1:
            hit_tp1 = True

        if not hit_tp1:
            return

        close_qty = pos['original_qty'] * Config.PARTIAL_CLOSE_PCT

        # Round to lot step
        try:
            instrument = self.session.get_instruments_info(category="linear", symbol=pos['symbol'])
            if instrument['result']['list']:
                lot_step = float(instrument['result']['list'][0]['lotSizeFilter']['qtyStep'])
                min_qty = float(instrument['result']['list'][0]['lotSizeFilter']['minOrderQty'])
                close_qty = max(min_qty, round(close_qty / lot_step) * lot_step)
                close_qty = float(f"{close_qty:g}")
        except Exception:
            pass

        close_side = 'Sell' if pos['side'] == 'Buy' else 'Buy'

        try:
            order = self.session.place_order(
                category="linear",
                symbol=pos['symbol'],
                side=close_side,
                orderType="Market",
                qty=f"{close_qty:g}",
                reduceOnly=True
            )

            if order['retCode'] == 0:
                pos['partial_filled'] = True
                pos['qty'] = pos['original_qty'] - close_qty

                # Calculate partial PnL
                if pos['side'] == 'Buy':
                    partial_pnl = (current_price - pos['entry_price']) * close_qty
                else:
                    partial_pnl = (pos['entry_price'] - current_price) * close_qty

                logger.info(
                    f"TP1 PARTIAL: Closed {close_qty} of {pos['original_qty']} {pos['symbol']} | "
                    f"PnL: ${partial_pnl:+.2f}"
                )
                trade_logger.info(
                    f"TP1_PARTIAL | symbol={pos['symbol']} | side={pos['side']} | "
                    f"qty_closed={close_qty} | pnl={partial_pnl:+.4f} | price={current_price:.4f}"
                )

                self.telegram.notify_partial_close(pos['symbol'], pos['side'], close_qty, partial_pnl)

                # Move SL to breakeven
                self._move_sl_to_breakeven(pos)

            else:
                logger.error(f"TP1 partial order failed: {order['retMsg']}")

        except Exception as e:
            logger.error(f"Error executing TP1 partial: {e}")

    def _move_sl_to_breakeven(self, pos: Dict):
        """Move stop loss to entry price."""
        try:
            be_price = round(pos['entry_price'], 4)
            self.session.set_trading_stop(
                category="linear",
                symbol=pos['symbol'],
                stopLoss=str(be_price),
                positionIdx=0
            )
            pos['stop_loss'] = be_price
            pos['breakeven_activated'] = True
            logger.info(f"SL moved to break-even: {pos['symbol']} SL=${be_price:.4f}")
        except Exception as e:
            logger.error(f"Error moving SL to BE: {e}")

    def _update_trailing_stop(self, pos: Dict, current_price: float):
        """Trail stop loss by ATR after TP1 partial."""
        atr = pos.get('atr', 0)
        if atr <= 0:
            return

        trail_distance = atr * Config.TRAILING_ATR_MULT

        if pos['side'] == 'Buy':
            new_sl = current_price - trail_distance
            if new_sl > pos.get('stop_loss', 0):
                try:
                    self.session.set_trading_stop(
                        category="linear",
                        symbol=pos['symbol'],
                        stopLoss=str(round(new_sl, 4)),
                        positionIdx=0
                    )
                    pos['stop_loss'] = new_sl
                    pos['trailing_activated'] = True
                    logger.info(f"Trailing SL: {pos['symbol']} SL=${new_sl:.4f}")
                except Exception as e:
                    logger.error(f"Error updating trailing: {e}")
        else:
            new_sl = current_price + trail_distance
            if new_sl < pos.get('stop_loss', float('inf')):
                try:
                    self.session.set_trading_stop(
                        category="linear",
                        symbol=pos['symbol'],
                        stopLoss=str(round(new_sl, 4)),
                        positionIdx=0
                    )
                    pos['stop_loss'] = new_sl
                    pos['trailing_activated'] = True
                    logger.info(f"Trailing SL: {pos['symbol']} SL=${new_sl:.4f}")
                except Exception as e:
                    logger.error(f"Error updating trailing: {e}")

    def _handle_closed_position(self, pos: Dict):
        """Handle a position closed by exchange (SL/TP hit or manual close)."""
        symbol = pos['symbol']
        pnl = 0.0
        exit_type = 'UNKNOWN'

        # Get actual PnL and exit type from closed trades
        try:
            closed = self.session.get_closed_pnl(category="linear", symbol=symbol, limit=10)
            if closed['result']['list']:
                entry_time = datetime.fromisoformat(pos.get('timestamp', ''))
                for trade in closed['result']['list']:
                    trade_time = datetime.fromtimestamp(
                        int(trade['createdTime']) / 1000, tz=timezone.utc
                    )
                    if trade_time >= entry_time:
                        pnl = float(trade['closedPnl'])
                        break
        except Exception as e:
            logger.error(f"Error fetching closed PnL: {e}")

        # Fallback: estimate
        if pnl == 0:
            price = self._get_current_price(symbol) or pos.get('entry_price', 0)
            if pos['side'] == 'Buy':
                pnl = (price - pos['entry_price']) * pos.get('qty', 0)
            else:
                pnl = (pos['entry_price'] - price) * pos.get('qty', 0)

        # Determine exit type: SL, TP, or MANUAL
        exit_type = self._detect_exit_type(pos, pnl)

        # Record in state
        cb_triggered = self.state.record_trade(pnl, symbol, pos['side'], exit_type)

        ts = pos.get('timestamp', '')
        duration_h = 0
        if ts:
            entry_time = datetime.fromisoformat(ts)
            duration_h = (datetime.now(timezone.utc) - entry_time).total_seconds() / 3600

        logger.info(
            f"EXIT: {symbol} {pos['side']} | PnL=${pnl:+.2f} | "
            f"Type={exit_type} | Duration={duration_h:.1f}h"
        )
        trade_logger.info(
            f"EXIT | symbol={symbol} | side={pos['side']} | pnl={pnl:+.4f} | "
            f"exit_type={exit_type} | duration_h={duration_h:.1f} | "
            f"partial={pos.get('partial_filled', False)} | trailing={pos.get('trailing_activated', False)} | "
            f"daily_pnl={self.state.daily_pnl:.2f} | total_pnl={self.state.total_pnl:.2f}"
        )

        balance = self.risk.get_balance()
        self.telegram.notify_exit(symbol, pos['side'], pnl, exit_type,
                                  self.state.daily_pnl, balance)

        # Extra notification for manual close
        if exit_type == 'MANUAL':
            self.telegram.send(
                f"<b>MANUAL CLOSE DETECTED</b>\n"
                f"{symbol} {pos['side']} | PnL: ${pnl:+.2f}\n"
                f"Position was closed manually on exchange.\n"
                f"Bot is now free to open new trades."
            )

        if cb_triggered:
            self.state.activate_circuit_breaker()
            self.telegram.notify_circuit_breaker(
                self.state.consecutive_losses, Config.CIRCUIT_BREAKER_PAUSE_H
            )

        self.state.open_positions.remove(pos)
        self.state.save_to_file()

    def _detect_exit_type(self, pos: Dict, pnl: float) -> str:
        """Detect if position was closed by SL, TP, or manually."""
        symbol = pos['symbol']
        try:
            # Check recent execution history for the trigger
            executions = self.session.get_executions(
                category="linear", symbol=symbol, limit=20
            )
            if executions['result']['list']:
                entry_time = datetime.fromisoformat(pos.get('timestamp', ''))
                for ex in executions['result']['list']:
                    ex_time = datetime.fromtimestamp(
                        int(ex['execTime']) / 1000, tz=timezone.utc
                    )
                    if ex_time < entry_time:
                        continue
                    # closedSize > 0 means it's a close execution
                    closed_size = float(ex.get('closedSize', 0))
                    if closed_size <= 0:
                        continue
                    order_type = ex.get('orderType', '')
                    stop_order_type = ex.get('stopOrderType', '')
                    # SL/TP triggers have specific stopOrderType values
                    if stop_order_type in ('StopLoss', 'Stop'):
                        return 'SL'
                    elif stop_order_type in ('TakeProfit', 'PartialTakeProfit'):
                        return 'TP'
                    elif order_type == 'Market' and stop_order_type == '':
                        # Market order with no stop type = manual close
                        return 'MANUAL'
        except Exception as e:
            logger.error(f"Error detecting exit type: {e}")

        # Fallback: guess from PnL
        if pnl > 0:
            return 'TP'
        return 'SL'

    def _force_close(self, pos: Dict, reason: str):
        """Force close a position."""
        try:
            close_side = 'Sell' if pos['side'] == 'Buy' else 'Buy'
            order = self.session.place_order(
                category="linear",
                symbol=pos['symbol'],
                side=close_side,
                orderType="Market",
                qty=f"{pos['qty']:g}",
                reduceOnly=True
            )
            if order['retCode'] == 0:
                logger.info(f"Force closed {pos['symbol']} ({reason})")
            else:
                logger.error(f"Force close failed: {order['retMsg']}")
        except Exception as e:
            logger.error(f"Error force closing: {e}")

    def _get_current_price(self, symbol: str) -> Optional[float]:
        try:
            ticker = self.session.get_tickers(category="linear", symbol=symbol)
            return float(ticker['result']['list'][0]['lastPrice'])
        except Exception:
            return None

    def _get_exchange_position_size(self, symbol: str) -> float:
        """Get actual position size from Bybit."""
        try:
            positions = self.session.get_positions(category="linear", symbol=symbol)
            for p in positions['result']['list']:
                if float(p['size']) > 0:
                    return float(p['size'])
        except Exception as e:
            logger.error(f"Error checking position {symbol}: {e}")
        return 0

    def _check_orphaned_positions(self):
        """Detect positions on Bybit that aren't tracked locally.
        Logs a warning so the bot doesn't accidentally open a second position."""
        try:
            positions = self.session.get_positions(category="linear", settleCoin="USDT")
            for p in positions['result']['list']:
                size = float(p.get('size', 0))
                if size > 0:
                    symbol = p['symbol']
                    side = p['side']
                    logger.warning(
                        f"ORPHANED position detected on exchange: {symbol} {side} size={size} — "
                        f"adding to local state for tracking"
                    )
                    # Add minimal record so the bot tracks it and won't open new trades
                    orphan_record = {
                        'order_id': 'orphan_reconciled',
                        'symbol': symbol,
                        'side': side,
                        'qty': size,
                        'original_qty': size,
                        'entry_price': float(p.get('avgPrice', 0)),
                        'stop_loss': float(p.get('stopLoss', 0)),
                        'take_profit_1': float(p.get('takeProfit', 0)),
                        'take_profit_2': float(p.get('takeProfit', 0)),
                        'confidence': 0,
                        'reasoning': 'Orphan reconciled from exchange',
                        'risk_reward_ratio': 0,
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                        'partial_filled': False,
                        'trailing_activated': False,
                        'breakeven_activated': False,
                        'atr': 0,
                    }
                    self.state.open_positions.append(orphan_record)
                    self.state.save_to_file()
                    self.telegram.send(
                        f"<b>⚠ ORPHAN DETECTED</b>\n"
                        f"{symbol} {side} size={size}\n"
                        f"Added to tracking — no new trades until closed"
                    )
        except Exception as e:
            logger.error(f"Error checking orphaned positions: {e}")
