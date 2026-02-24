#!/usr/bin/env python3
"""
Trading state management with crash recovery.
Tracks PnL, trades, circuit breaker, profit distribution.
"""

import json
import os
import tempfile
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
from config import Config, logger


class TradingState:
    """Persistent trading state with atomic file saves."""

    def __init__(self):
        self.daily_pnl: float = 0.0
        self.total_pnl: float = 0.0
        self.peak_balance: float = Config.ACCOUNT_SIZE
        self.trades_today: int = 0
        self.wins_today: int = 0
        self.losses_today: int = 0
        self.consecutive_losses: int = 0
        self.winning_streak: int = 0
        self.circuit_breaker_until: Optional[str] = None  # ISO format
        self.last_daily_reset: str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        self.open_positions: List[Dict] = []
        self.trade_history: List[Dict] = []
        self.daily_profit_by_date: Dict[str, float] = {}
        self.start_date: str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        self.trading_days: List[str] = []

    def reset_daily_if_needed(self):
        """Reset daily counters at UTC midnight."""
        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        if today != self.last_daily_reset:
            # Save yesterday's profit
            if self.daily_pnl != 0:
                self.daily_profit_by_date[self.last_daily_reset] = self.daily_pnl
            if self.trades_today > 0 and self.last_daily_reset not in self.trading_days:
                self.trading_days.append(self.last_daily_reset)

            logger.info(f"Daily reset: PnL was ${self.daily_pnl:+.2f}, trades={self.trades_today}")
            self.daily_pnl = 0.0
            self.trades_today = 0
            self.wins_today = 0
            self.losses_today = 0
            self.last_daily_reset = today

    def record_trade(self, pnl: float, symbol: str, side: str, exit_type: str):
        """Record a completed trade."""
        self.reset_daily_if_needed()
        self.daily_pnl += pnl
        self.total_pnl += pnl
        self.trades_today += 1

        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        if today not in self.trading_days:
            self.trading_days.append(today)

        if pnl > 0:
            self.consecutive_losses = 0
            self.winning_streak += 1
            self.wins_today += 1
        elif pnl < 0:
            self.consecutive_losses += 1
            self.winning_streak = 0
            self.losses_today += 1

        self.trade_history.append({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'symbol': symbol,
            'side': side,
            'pnl': pnl,
            'exit_type': exit_type,
            'daily_pnl': self.daily_pnl,
            'total_pnl': self.total_pnl,
        })

        # Keep last 100 trades
        if len(self.trade_history) > 100:
            self.trade_history = self.trade_history[-100:]

        return self.is_circuit_breaker_triggered()

    def is_circuit_breaker_triggered(self) -> bool:
        """Check if circuit breaker should activate."""
        return self.consecutive_losses >= Config.CIRCUIT_BREAKER_LOSSES

    def activate_circuit_breaker(self):
        """Activate circuit breaker pause."""
        until = datetime.now(timezone.utc) + timedelta(hours=Config.CIRCUIT_BREAKER_PAUSE_H)
        self.circuit_breaker_until = until.isoformat()
        logger.warning(
            f"CIRCUIT BREAKER: {self.consecutive_losses} consecutive losses, "
            f"pausing until {self.circuit_breaker_until}"
        )

    def is_circuit_breaker_active(self) -> bool:
        """Check if circuit breaker is currently active."""
        if not self.circuit_breaker_until:
            return False
        now = datetime.now(timezone.utc)
        until = datetime.fromisoformat(self.circuit_breaker_until)
        if now < until:
            return True
        # Expired
        self.circuit_breaker_until = None
        logger.info("Circuit breaker expired, resuming trading")
        return False

    def get_circuit_breaker_remaining_h(self) -> float:
        """Get remaining circuit breaker hours."""
        if not self.circuit_breaker_until:
            return 0.0
        now = datetime.now(timezone.utc)
        until = datetime.fromisoformat(self.circuit_breaker_until)
        remaining = (until - now).total_seconds() / 3600
        return max(0.0, remaining)

    def get_profit_distribution_ratio(self) -> float:
        """Check if today's profit exceeds 30% of total profit (HyroTrader rule)."""
        if self.total_pnl <= 0:
            return 0.0
        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        today_profit = self.daily_profit_by_date.get(today, 0.0) + self.daily_pnl
        if today_profit <= 0:
            return 0.0
        return today_profit / self.total_pnl

    def get_trading_days_count(self) -> int:
        """Number of unique trading days."""
        return len(self.trading_days)

    def save_to_file(self, filepath: str = None):
        """Atomic save to JSON file."""
        filepath = filepath or Config.STATE_FILE
        data = {
            'daily_pnl': self.daily_pnl,
            'total_pnl': self.total_pnl,
            'peak_balance': self.peak_balance,
            'trades_today': self.trades_today,
            'wins_today': self.wins_today,
            'losses_today': self.losses_today,
            'consecutive_losses': self.consecutive_losses,
            'winning_streak': self.winning_streak,
            'circuit_breaker_until': self.circuit_breaker_until,
            'last_daily_reset': self.last_daily_reset,
            'open_positions': self.open_positions,
            'trade_history': self.trade_history,
            'daily_profit_by_date': self.daily_profit_by_date,
            'start_date': self.start_date,
            'trading_days': self.trading_days,
        }
        try:
            dir_name = os.path.dirname(filepath) or '.'
            fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix='.tmp')
            with os.fdopen(fd, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            os.replace(tmp_path, filepath)
        except Exception as e:
            logger.error(f"Error saving state: {e}")
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    @classmethod
    def load_from_file(cls, filepath: str = None) -> 'TradingState':
        """Load state from file, or create fresh if missing/corrupt."""
        filepath = filepath or Config.STATE_FILE
        state = cls()
        if not os.path.exists(filepath):
            logger.info("No state file found, starting fresh")
            return state
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            state.daily_pnl = data.get('daily_pnl', 0.0)
            state.total_pnl = data.get('total_pnl', 0.0)
            state.peak_balance = data.get('peak_balance', Config.ACCOUNT_SIZE)
            state.trades_today = data.get('trades_today', 0)
            state.wins_today = data.get('wins_today', 0)
            state.losses_today = data.get('losses_today', 0)
            state.consecutive_losses = data.get('consecutive_losses', 0)
            state.winning_streak = data.get('winning_streak', 0)
            state.circuit_breaker_until = data.get('circuit_breaker_until')
            state.last_daily_reset = data.get('last_daily_reset', datetime.now(timezone.utc).strftime('%Y-%m-%d'))
            state.open_positions = data.get('open_positions', [])
            state.trade_history = data.get('trade_history', [])
            state.daily_profit_by_date = data.get('daily_profit_by_date', {})
            state.start_date = data.get('start_date', datetime.now(timezone.utc).strftime('%Y-%m-%d'))
            state.trading_days = data.get('trading_days', [])
            state.reset_daily_if_needed()
            logger.info(f"State restored: total_pnl=${state.total_pnl:+.2f}, trades_today={state.trades_today}")
            return state
        except Exception as e:
            logger.error(f"Error loading state: {e}, starting fresh")
            return cls()
