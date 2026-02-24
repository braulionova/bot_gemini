#!/usr/bin/env python3
"""
Gemini Flash Trading Bot — Main orchestrator.
Dual-frequency loop: 30s position monitoring + 15min analysis cycles.
"""

import time
import signal
import sys
from datetime import datetime, timezone
from config import Config, logger
from trading_state import TradingState
from telegram_notifier import TelegramNotifier
from data_collector import DataCollector
from gemini_analyzer import GeminiAnalyzer
from risk_validator import RiskValidator
from order_executor import OrderExecutor
from position_monitor import PositionMonitor


class GeminiTradingBot:
    """Main bot orchestrator."""

    def __init__(self):
        logger.info("=" * 60)
        logger.info("GEMINI FLASH TRADING BOT — INITIALIZING")
        logger.info("=" * 60)

        # Create Bybit session
        self.session = Config.create_bybit_session()

        # Load persistent state
        self.state = TradingState.load_from_file()

        # Initialize components
        self.telegram = TelegramNotifier(Config.TELEGRAM_TOKEN, Config.TELEGRAM_CHAT_ID)
        self.data_collector = DataCollector(self.session)
        self.gemini = GeminiAnalyzer()
        self.risk_validator = RiskValidator(self.session, self.state)
        self.order_executor = OrderExecutor(self.session, self.state, self.risk_validator, self.telegram)
        self.position_monitor = PositionMonitor(self.session, self.state, self.risk_validator, self.telegram)

        # Timing
        self.last_analysis_time = 0
        self.last_daily_summary = None
        self.running = True

        # Signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        logger.info(f"Signal {signum} received, shutting down gracefully...")
        self.running = False

    def _get_balance(self) -> float:
        return self.risk_validator.get_balance()

    def _send_daily_summary(self):
        """Send daily summary at 00:00 UTC."""
        now = datetime.now(timezone.utc)
        today = now.date()
        if self.last_daily_summary == today:
            return
        if now.hour == 0 and now.minute < 5:
            self.last_daily_summary = today
            balance = self._get_balance()
            stats = {
                'trades_today': self.state.trades_today,
                'wins': self.state.wins_today,
                'losses': self.state.losses_today,
                'daily_pnl': self.state.daily_pnl,
                'total_pnl': self.state.total_pnl,
                'balance': balance,
                'drawdown_pct': self.risk_validator.get_drawdown_pct(),
                'trading_days': self.state.get_trading_days_count(),
                'open_positions': len(self.state.open_positions),
            }
            self.telegram.notify_daily_summary(stats)

    def _should_run_analysis(self) -> bool:
        """Check if 15 minutes have passed since last analysis."""
        return (time.time() - self.last_analysis_time) >= Config.ANALYSIS_INTERVAL_SEC

    def _run_analysis_cycle(self):
        """Collect data → Gemini analyze → execute if BUY/SELL."""
        self.last_analysis_time = time.time()
        self.state.reset_daily_if_needed()

        # Global limit: only 1 position at a time
        if self.state.open_positions:
            logger.info(
                f"Position open ({self.state.open_positions[0]['symbol']} "
                f"{self.state.open_positions[0]['side']}), skipping analysis"
            )
            return

        # Double-check with exchange — no positions on Bybit
        try:
            bybit_positions = self.session.get_positions(category="linear", settleCoin="USDT")
            for p in bybit_positions['result']['list']:
                if float(p.get('size', 0)) > 0:
                    logger.info(
                        f"Exchange has open position ({p['symbol']}), skipping analysis"
                    )
                    return
        except Exception as e:
            logger.error(f"Error checking exchange positions: {e}")

        for symbol in Config.SYMBOLS:
            # Pre-checks
            balance = self._get_balance()
            allowed, reason = self.risk_validator.validate_trade(
                {'confidence': 10, 'risk_reward_ratio': 10, 'entry_price': 1, 'stop_loss': 0.99},
                balance
            )
            if not allowed and 'confidence' not in reason.lower() and 'R:R' not in reason:
                logger.info(f"Cannot trade: {reason}")
                continue

            try:
                # Step 1: Collect data
                data_package = self.data_collector.collect_all(symbol)

                # Add ATR from 4h for position management
                atr = 0
                indicators_4h = data_package.get('indicators', {}).get('4h', {})
                if indicators_4h:
                    atr = indicators_4h.get('atr', 0) or 0

                # Step 2: Gemini analysis
                result = self.gemini.analyze(symbol, data_package)
                if not result:
                    logger.info(f"[{symbol}] Gemini returned no result")
                    continue

                action = result.get('action', 'HOLD').upper()
                confidence = result.get('confidence', 0)

                if action == 'HOLD':
                    logger.info(f"[{symbol}] Gemini says HOLD (confidence={confidence})")
                    continue

                if confidence < Config.MIN_CONFIDENCE:
                    logger.info(f"[{symbol}] Confidence {confidence} < {Config.MIN_CONFIDENCE}, skipping")
                    continue

                # Add symbol and ATR to signal for executor
                result['symbol'] = symbol
                result['atr'] = atr

                # Step 3: Execute trade (executor does its own validation)
                logger.info(
                    f"[{symbol}] Signal: {action} | Confidence: {confidence}/10 | "
                    f"RR: {result.get('risk_reward_ratio', 0):.2f}"
                )
                self.order_executor.execute_trade(result)

            except Exception as e:
                logger.error(f"Analysis cycle error for {symbol}: {e}", exc_info=True)

    def run(self):
        """Main dual-frequency loop."""
        balance = self._get_balance()
        logger.info(f"Symbols: {Config.SYMBOLS}")
        logger.info(f"Balance: ${balance:,.2f}")
        logger.info(f"Mode: {'TESTNET' if Config.BYBIT_TESTNET else 'LIVE'}")
        logger.info(f"Gemini model: {Config.GEMINI_MODEL}")
        logger.info(f"Analysis interval: {Config.ANALYSIS_INTERVAL_SEC}s")
        logger.info(f"Monitor interval: {Config.MONITOR_INTERVAL_SEC}s")
        logger.info(f"State: total_pnl=${self.state.total_pnl:+.2f}, trades_today={self.state.trades_today}")
        logger.info("=" * 60)

        self.telegram.notify_startup(balance, Config.BYBIT_TESTNET, Config.SYMBOLS)

        iteration = 0
        while self.running:
            try:
                loop_start = time.time()
                iteration += 1

                # Always: monitor positions (every 30s)
                self.position_monitor.check_positions()

                # Daily summary check
                self._send_daily_summary()

                # Analysis cycle (every 15 min)
                if self._should_run_analysis():
                    logger.info(f"--- Analysis cycle #{iteration} ---")
                    self._run_analysis_cycle()

                # Save state each iteration
                self.state.save_to_file()

                # Sleep for monitor interval
                elapsed = time.time() - loop_start
                sleep_time = max(5, Config.MONITOR_INTERVAL_SEC - elapsed)
                time.sleep(sleep_time)

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Main loop error: {e}", exc_info=True)
                self.telegram.notify_error(f"Main loop error: {str(e)[:200]}")
                time.sleep(30)

        # Graceful shutdown
        logger.info("Shutting down...")
        self.state.save_to_file()
        self.telegram.send("<b>GEMINI BOT STOPPED</b>")
        logger.info("Bot stopped.")


if __name__ == '__main__':
    bot = GeminiTradingBot()
    bot.run()
