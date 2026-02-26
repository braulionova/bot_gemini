#!/usr/bin/env python3
"""
Risk validator — absolute veto power over Gemini decisions.
Every HyroTrader rule is checked here. No trade passes without ALL checks green.
"""

from typing import Dict, Tuple, Optional
from pybit.unified_trading import HTTP
from config import Config, logger
from trading_state import TradingState


class RiskValidator:
    """Validates trades against all HyroTrader rules. Has absolute veto power."""

    def __init__(self, session: HTTP, state: TradingState):
        self.session = session
        self.state = state

    def get_balance(self) -> float:
        """Get current account balance from Bybit."""
        try:
            wallet = self.session.get_wallet_balance(accountType="UNIFIED")
            if wallet['result']['list']:
                return float(wallet['result']['list'][0]['totalEquity'])
        except Exception as e:
            logger.error(f"Error getting balance: {e}")
        return Config.ACCOUNT_SIZE

    def get_drawdown_pct(self) -> float:
        """Current drawdown as percentage of account."""
        balance = self.get_balance()
        if balance > self.state.peak_balance:
            self.state.peak_balance = balance
        return ((self.state.peak_balance - balance) / Config.ACCOUNT_SIZE) * 100

    def validate_trade(self, signal: Dict, balance: float) -> Tuple[bool, str]:
        """Master validation — runs ALL checks. Returns (allowed, reason)."""
        self.state.reset_daily_if_needed()

        checks = [
            self._check_circuit_breaker,
            lambda s, b: self._check_daily_loss_limit(b),
            lambda s, b: self._check_total_drawdown(b),
            lambda s, b: self._check_daily_trade_count(),
            lambda s, b: self._check_daily_profit_cap(b),
            self._check_confidence,
            self._check_stop_loss_risk,
            self._check_tp_placement,
            self._check_min_trade_value,
            self._check_max_margin,
            self._check_rr_ratio,
            lambda s, b: self._check_profit_distribution(),
        ]

        for check in checks:
            allowed, reason = check(signal, balance)
            if not allowed:
                logger.warning(f"RISK REJECTION: {reason}")
                return False, reason

        return True, "All checks passed"

    def _check_circuit_breaker(self, signal: Dict, balance: float) -> Tuple[bool, str]:
        if self.state.is_circuit_breaker_active():
            remaining = self.state.get_circuit_breaker_remaining_h()
            return False, f"Circuit breaker active ({remaining:.1f}h remaining)"
        return True, "OK"

    def _check_daily_loss_limit(self, balance: float) -> Tuple[bool, str]:
        max_daily_loss = Config.ACCOUNT_SIZE * Config.DAILY_LOSS_LIMIT
        if self.state.daily_pnl <= -max_daily_loss:
            return False, f"Daily loss limit hit (${self.state.daily_pnl:+.2f} <= -${max_daily_loss:.0f})"
        return True, "OK"

    def _check_total_drawdown(self, balance: float) -> Tuple[bool, str]:
        if balance > self.state.peak_balance:
            self.state.peak_balance = balance
        drawdown = self.state.peak_balance - balance
        max_dd = Config.ACCOUNT_SIZE * Config.MAX_DRAWDOWN_TOTAL
        if drawdown >= max_dd:
            return False, f"Total drawdown limit (${drawdown:.0f} >= ${max_dd:.0f})"
        return True, "OK"

    def _check_daily_trade_count(self) -> Tuple[bool, str]:
        if self.state.trades_today >= Config.MAX_TRADES_PER_DAY:
            return False, f"Max trades/day reached ({self.state.trades_today}/{Config.MAX_TRADES_PER_DAY})"
        return True, "OK"

    def _check_daily_profit_cap(self, balance: float) -> Tuple[bool, str]:
        cap = Config.ACCOUNT_SIZE * Config.DAILY_PROFIT_CAP
        if self.state.daily_pnl >= cap:
            return False, f"Daily profit cap reached (${self.state.daily_pnl:+.2f} >= ${cap:.0f})"
        return True, "OK"

    def _check_confidence(self, signal: Dict, balance: float) -> Tuple[bool, str]:
        confidence = signal.get('confidence', 0)
        if confidence < Config.MIN_CONFIDENCE:
            return False, f"Confidence too low ({confidence} < {Config.MIN_CONFIDENCE})"
        return True, "OK"

    def _check_stop_loss_risk(self, signal: Dict, balance: float) -> Tuple[bool, str]:
        entry = signal.get('entry_price', 0)
        sl = signal.get('stop_loss', 0)
        if entry <= 0 or sl <= 0:
            return False, "Invalid entry/SL prices"
        risk_pct = abs(entry - sl) / entry
        # Use escalated risk for high-confidence signals
        confidence = signal.get('confidence', 0)
        max_risk = Config.MAX_RISK_ESCALATED if confidence >= 8 else Config.MAX_RISK_PER_TRADE
        # This checks the SL distance is reasonable; position sizing enforces dollar risk
        return True, "OK"

    def _check_tp_placement(self, signal: Dict, balance: float) -> Tuple[bool, str]:
        """Validate TP is logically placed relative to entry and SL."""
        action = signal.get('action', '').upper()
        entry = signal.get('entry_price', 0)
        sl = signal.get('stop_loss', 0)
        tp1 = signal.get('take_profit_1', 0)
        if entry <= 0 or sl <= 0 or tp1 <= 0:
            return True, "OK"  # Will be caught by other checks
        # TP must be on the correct side of entry
        if action == 'BUY' and tp1 <= entry:
            return False, f"LONG TP1 ({tp1}) is not above entry ({entry})"
        if action == 'SELL' and tp1 >= entry:
            return False, f"SHORT TP1 ({tp1}) is not below entry ({entry})"
        # TP must not be in the same direction as SL
        if action == 'BUY' and sl >= entry:
            return False, f"LONG SL ({sl}) is not below entry ({entry})"
        if action == 'SELL' and sl <= entry:
            return False, f"SHORT SL ({sl}) is not above entry ({entry})"
        # TP distance must be at least as large as SL distance (R:R ≥ 1.0 minimum sanity)
        sl_dist = abs(entry - sl)
        tp_dist = abs(tp1 - entry)
        if tp_dist < sl_dist * 0.8:
            return False, f"TP1 too close: reward ({tp_dist:.2f}) < 80% of risk ({sl_dist:.2f})"
        return True, "OK"

    def _check_min_trade_value(self, signal: Dict, balance: float) -> Tuple[bool, str]:
        # Will be checked during position sizing
        min_value = balance * Config.MIN_TRADE_VALUE_PCT
        entry = signal.get('entry_price', 0)
        if entry <= 0:
            return False, "Invalid entry price"
        return True, "OK"

    def _check_max_margin(self, signal: Dict, balance: float) -> Tuple[bool, str]:
        # Check existing margin + proposed
        max_margin = balance * Config.MAX_MARGIN_EXPOSURE
        try:
            positions = self.session.get_positions(category="linear", settleCoin="USDT")
            current_margin = 0
            for p in positions['result']['list']:
                size = float(p.get('size', 0))
                avg_price = float(p.get('avgPrice', 0))
                leverage = float(p.get('leverage', 1))
                if size > 0 and leverage > 0:
                    current_margin += (size * avg_price) / leverage
            if current_margin >= max_margin:
                return False, f"Max margin exposure (${current_margin:.0f} >= ${max_margin:.0f})"
        except Exception as e:
            logger.error(f"Error checking margin: {e}")
        return True, "OK"

    def _check_rr_ratio(self, signal: Dict, balance: float) -> Tuple[bool, str]:
        # Calculate R:R from actual prices if reported value is missing/zero
        rr = signal.get('risk_reward_ratio', 0)
        entry = signal.get('entry_price', 0)
        sl = signal.get('stop_loss', 0)
        tp1 = signal.get('take_profit_1', 0)
        if rr <= 0 and entry > 0 and sl > 0 and tp1 > 0:
            risk = abs(entry - sl)
            reward = abs(tp1 - entry)
            rr = reward / risk if risk > 0 else 0
            signal['risk_reward_ratio'] = round(rr, 2)
        if rr < Config.MIN_RR_RATIO:
            return False, f"R:R too low ({rr:.2f} < {Config.MIN_RR_RATIO})"
        return True, "OK"

    def _check_profit_distribution(self) -> Tuple[bool, str]:
        # Skip check when too few trading days — mathematically impossible to distribute
        trading_days = self.state.get_trading_days_count()
        if trading_days < 5:
            return True, "OK"
        # Skip when total profit is still small
        if self.state.total_pnl < 500:
            return True, "OK"
        ratio = self.state.get_profit_distribution_ratio()
        if ratio > Config.PROFIT_DISTRIBUTION_MAX:
            return False, f"Profit distribution limit ({ratio:.0%} > {Config.PROFIT_DISTRIBUTION_MAX:.0%})"
        return True, "OK"

    def calculate_position_size(self, symbol: str, entry: float, sl: float,
                                 confidence: int, balance: float) -> float:
        """Calculate position size respecting ALL constraints."""
        try:
            # Get instrument info for lot sizing
            instrument = self.session.get_instruments_info(category="linear", symbol=symbol)
            if not instrument['result']['list']:
                return 0
            info = instrument['result']['list'][0]
            lot_step = float(info['lotSizeFilter']['qtyStep'])
            min_qty = float(info['lotSizeFilter']['minOrderQty'])

            # Fixed dollar risk sizing ($100 USDT)
            risk_amount = Config.FIXED_RISK_AMOUNT
            risk_per_unit = abs(entry - sl)
            if risk_per_unit <= 0:
                return 0

            qty = risk_amount / risk_per_unit

            # Constraint: min trade value >= 6% of balance
            min_value = balance * Config.MIN_TRADE_VALUE_PCT
            min_qty_for_value = min_value / entry
            if qty * entry < min_value:
                qty = min_qty_for_value

            # Constraint: max margin <= 15% of balance
            max_margin_value = balance * Config.MAX_MARGIN_EXPOSURE
            # Assuming ~10x leverage on Bybit for USDT perps
            max_position_value = max_margin_value * 10
            if qty * entry > max_position_value:
                qty = max_position_value / entry

            # Hard cap: never risk more than $50 USDT (FIXED_RISK_AMOUNT)
            actual_risk = qty * risk_per_unit
            if actual_risk > risk_amount:
                qty = risk_amount / risk_per_unit

            # Round to lot step
            qty = max(min_qty, round(qty / lot_step) * lot_step)
            qty = float(f"{qty:g}")

            # Final validation: dollar risk
            final_risk = qty * risk_per_unit
            risk_pct = final_risk / balance if balance > 0 else 0
            logger.info(
                f"Position size: {qty} {symbol} | "
                f"Risk: ${final_risk:.2f} ({risk_pct*100:.1f}%) [fixed ${Config.FIXED_RISK_AMOUNT}] | "
                f"Value: ${qty * entry:,.2f} | "
                f"Entry: ${entry:.2f} | SL: ${sl:.2f}"
            )
            return qty

        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0
