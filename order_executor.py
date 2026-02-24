#!/usr/bin/env python3
"""
Order executor — places trades on Bybit with SL/TP and verifies placement.
"""

import time
from typing import Dict, Optional
from datetime import datetime, timezone
from pybit.unified_trading import HTTP
from config import Config, logger, trade_logger
from trading_state import TradingState
from risk_validator import RiskValidator
from telegram_notifier import TelegramNotifier


class OrderExecutor:
    """Executes validated trades on Bybit."""

    def __init__(self, session: HTTP, state: TradingState,
                 risk_validator: RiskValidator, telegram: TelegramNotifier):
        self.session = session
        self.state = state
        self.risk = risk_validator
        self.telegram = telegram

    def execute_trade(self, signal: Dict) -> Optional[Dict]:
        """Full trade execution pipeline: validate → size → order → verify SL → notify."""
        symbol = signal.get('symbol', '')
        action = signal.get('action', 'HOLD')

        if action == 'HOLD':
            return None

        side = 'Buy' if action == 'BUY' else 'Sell'
        entry = signal.get('entry_price', 0)
        sl = signal.get('stop_loss', 0)
        tp1 = signal.get('take_profit_1', 0)
        tp2 = signal.get('take_profit_2', 0)
        confidence = signal.get('confidence', 0)

        # TP2 is optional — default to TP1 if missing/zero
        if tp2 <= 0 and tp1 > 0:
            tp2 = tp1
            signal['take_profit_2'] = tp2

        if entry <= 0 or sl <= 0 or tp1 <= 0:
            logger.error(f"Invalid signal prices: entry={entry}, sl={sl}, tp1={tp1}, tp2={tp2}")
            return None

        # Step 0: Ensure no open positions (local + exchange)
        if self.state.open_positions:
            logger.warning(f"Cannot open trade: already have {len(self.state.open_positions)} position(s)")
            return None
        try:
            bybit_positions = self.session.get_positions(category="linear", settleCoin="USDT")
            for p in bybit_positions['result']['list']:
                if float(p.get('size', 0)) > 0:
                    logger.warning(f"Cannot open trade: exchange has position in {p['symbol']}")
                    return None
        except Exception as e:
            logger.error(f"Error checking exchange positions before trade: {e}")
            return None  # Fail safe: don't open if can't verify

        # Step 1: Validate
        balance = self.risk.get_balance()
        allowed, reason = self.risk.validate_trade(signal, balance)
        if not allowed:
            logger.warning(f"Trade rejected: {reason}")
            self.telegram.notify_risk_rejection(symbol, side, reason)
            return None

        # Step 2: Calculate position size using CURRENT market price (not Gemini's predicted entry).
        # Gemini's entry_price is a prediction; the actual fill will be at the live market price.
        # Using the predicted entry causes the real SL risk to exceed $100 USDT when the fill
        # price differs from the prediction.
        current_price = self._get_current_price(symbol)
        sizing_price = current_price if (current_price and current_price > 0) else entry
        if sizing_price != entry:
            logger.info(f"Position sizing: using current price ${sizing_price:.4f} (signal entry was ${entry:.4f})")
        qty = self.risk.calculate_position_size(symbol, sizing_price, sl, confidence, balance)
        if qty <= 0:
            logger.warning(f"Position size is 0 for {symbol}")
            return None

        # Step 3: Place market order with SL/TP1
        position = self._place_market_order(symbol, side, qty, sl, tp1)
        if not position:
            return None

        # Step 4: Verify SL placement
        sl_verified = self._verify_sl_placement(symbol, side)
        if not sl_verified:
            logger.error(f"SL verification failed for {symbol}, emergency closing")
            self._emergency_close(symbol, side, qty)
            self.telegram.notify_error(f"SL verification failed for {symbol} — position closed")
            return None

        # Step 5: Record in state
        position_record = {
            'order_id': position['order_id'],
            'symbol': symbol,
            'side': side,
            'qty': qty,
            'original_qty': qty,
            'entry_price': position['entry_price'],
            'stop_loss': sl,
            'take_profit_1': tp1,
            'take_profit_2': tp2,
            'confidence': confidence,
            'reasoning': signal.get('reasoning', ''),
            'risk_reward_ratio': signal.get('risk_reward_ratio', 0),
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'partial_filled': False,
            'trailing_activated': False,
            'breakeven_activated': False,
            'atr': signal.get('atr', 0),
        }
        self.state.open_positions.append(position_record)
        self.state.save_to_file()

        # Step 6: Log and notify
        trade_logger.info(
            f"ENTRY | symbol={symbol} | side={side} | qty={qty} | "
            f"entry={position['entry_price']:.4f} | sl={sl:.4f} | tp1={tp1:.4f} | tp2={tp2:.4f} | "
            f"confidence={confidence} | rr={signal.get('risk_reward_ratio', 0):.2f} | "
            f"reasoning={signal.get('reasoning', '')[:200]}"
        )

        self.telegram.notify_entry(
            symbol, side, qty, position['entry_price'],
            sl, tp1, tp2, confidence, signal.get('reasoning', '')
        )

        logger.info(
            f"ENTRY: {side} {qty} {symbol} @ ${position['entry_price']:.2f} | "
            f"SL=${sl:.2f} | TP1=${tp1:.2f} | TP2=${tp2:.2f} | "
            f"Confidence: {confidence}/10"
        )

        return position_record

    def _place_market_order(self, symbol: str, side: str, qty: float,
                             sl: float, tp: float) -> Optional[Dict]:
        """Place market order with SL and TP on Bybit."""
        try:
            order = self.session.place_order(
                category="linear",
                symbol=symbol,
                side=side,
                orderType="Market",
                qty=f"{qty:g}",
                stopLoss=str(round(sl, 4)),
                takeProfit=str(round(tp, 4)),
                slTriggerBy="LastPrice",
                tpTriggerBy="LastPrice"
            )

            if order['retCode'] != 0:
                logger.error(f"Order failed: {order['retMsg']}")
                return None

            order_id = order['result']['orderId']

            # Get actual fill price
            entry_price = self._get_current_price(symbol)
            if not entry_price:
                entry_price = 0  # Will be updated on next monitor cycle

            logger.info(f"Order placed: {order_id} | {side} {qty} {symbol}")
            return {'order_id': order_id, 'entry_price': entry_price}

        except Exception as e:
            logger.error(f"Error placing order {symbol}: {e}")
            return None

    def _verify_sl_placement(self, symbol: str, side: str) -> bool:
        """Verify SL is set within 30 seconds. Required by HyroTrader rules."""
        deadline = time.time() + Config.SL_PLACEMENT_TIMEOUT
        while time.time() < deadline:
            try:
                positions = self.session.get_positions(category="linear", symbol=symbol)
                for p in positions['result']['list']:
                    if float(p['size']) > 0:
                        sl = float(p.get('stopLoss', 0))
                        if sl > 0:
                            logger.info(f"SL verified for {symbol}: ${sl:.4f}")
                            return True
            except Exception as e:
                logger.error(f"Error verifying SL: {e}")
            time.sleep(2)

        logger.error(f"SL NOT VERIFIED within {Config.SL_PLACEMENT_TIMEOUT}s for {symbol}")
        return False

    def _emergency_close(self, symbol: str, side: str, qty: float):
        """Emergency close position if SL verification fails."""
        try:
            close_side = 'Sell' if side == 'Buy' else 'Buy'
            self.session.place_order(
                category="linear",
                symbol=symbol,
                side=close_side,
                orderType="Market",
                qty=f"{qty:g}",
                reduceOnly=True
            )
            logger.warning(f"Emergency close executed for {symbol}")
        except Exception as e:
            logger.error(f"CRITICAL: Emergency close failed for {symbol}: {e}")
            self.telegram.notify_error(f"CRITICAL: Emergency close FAILED for {symbol}: {e}")

    def _get_current_price(self, symbol: str) -> Optional[float]:
        try:
            ticker = self.session.get_tickers(category="linear", symbol=symbol)
            return float(ticker['result']['list'][0]['lastPrice'])
        except Exception:
            return None
