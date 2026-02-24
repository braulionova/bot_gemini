#!/usr/bin/env python3
"""
Telegram notification system for trading alerts.
"""

import requests
from config import logger


class TelegramNotifier:
    """Sends formatted notifications via Telegram bot."""

    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.enabled = bool(token and chat_id)
        if not self.enabled:
            logger.warning("Telegram notifications disabled (missing token/chat_id)")

    def send(self, message: str) -> bool:
        """Send HTML-formatted message to Telegram."""
        if not self.enabled:
            return False
        try:
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            payload = {"chat_id": self.chat_id, "text": message, "parse_mode": "HTML"}
            resp = requests.post(url, json=payload, timeout=10)
            return resp.status_code == 200
        except Exception as e:
            logger.error(f"Telegram error: {e}")
            return False

    def notify_startup(self, balance: float, testnet: bool, symbols: list):
        msg = (
            f"<b>GEMINI BOT STARTED</b>\n\n"
            f"Mode: {'TESTNET' if testnet else 'LIVE'}\n"
            f"Symbols: {', '.join(symbols)}\n"
            f"Balance: ${balance:,.2f}\n"
            f"AI: Gemini Flash"
        )
        self.send(msg)

    def notify_entry(self, symbol: str, side: str, qty: float, entry: float,
                     sl: float, tp1: float, tp2: float, confidence: int, reasoning: str):
        msg = (
            f"<b>GEMINI BOT - NEW ENTRY</b>\n\n"
            f"Symbol: {symbol}\n"
            f"Side: {side}\n"
            f"Qty: {qty}\n"
            f"Entry: ${entry:,.2f}\n"
            f"SL: ${sl:,.2f}\n"
            f"TP1: ${tp1:,.2f}\n"
            f"TP2: ${tp2:,.2f}\n"
            f"Confidence: {confidence}/10\n\n"
            f"Reasoning: {reasoning}"
        )
        self.send(msg)

    def notify_exit(self, symbol: str, side: str, pnl: float, exit_type: str,
                    daily_pnl: float, balance: float):
        result = "WIN" if pnl > 0 else "LOSS"
        msg = (
            f"<b>GEMINI BOT - {result}</b>\n\n"
            f"Symbol: {symbol} {side}\n"
            f"PnL: <b>${pnl:+.2f}</b>\n"
            f"Exit: {exit_type}\n"
            f"Daily PnL: ${daily_pnl:+.2f}\n"
            f"Balance: ${balance:,.2f}"
        )
        self.send(msg)

    def notify_partial_close(self, symbol: str, side: str, qty_closed: float, pnl_locked: float):
        msg = (
            f"<b>GEMINI BOT - TP1 PARTIAL</b>\n\n"
            f"Symbol: {symbol} {side}\n"
            f"Closed: {qty_closed} (50%)\n"
            f"PnL locked: ${pnl_locked:+.2f}\n"
            f"SL moved to break-even\n"
            f"Remaining rides to TP2"
        )
        self.send(msg)

    def notify_risk_rejection(self, symbol: str, side: str, reason: str):
        msg = (
            f"<b>GEMINI BOT - TRADE REJECTED</b>\n\n"
            f"Symbol: {symbol} {side}\n"
            f"Reason: {reason}"
        )
        self.send(msg)

    def notify_circuit_breaker(self, losses: int, pause_hours: int):
        msg = (
            f"<b>GEMINI BOT - CIRCUIT BREAKER</b>\n\n"
            f"Consecutive losses: {losses}\n"
            f"Pausing for {pause_hours}h\n"
            f"Capital protection is priority #1"
        )
        self.send(msg)

    def notify_daily_summary(self, stats: dict):
        msg = (
            f"<b>GEMINI BOT - DAILY SUMMARY</b>\n\n"
            f"Trades: {stats['trades_today']}\n"
            f"Wins: {stats['wins']} | Losses: {stats['losses']}\n"
            f"Daily PnL: <b>${stats['daily_pnl']:+.2f}</b>\n"
            f"Total PnL: ${stats['total_pnl']:+.2f}\n"
            f"Balance: ${stats['balance']:,.2f}\n"
            f"Drawdown: {stats['drawdown_pct']:.1f}%\n"
            f"Trading days: {stats['trading_days']}\n"
            f"Open positions: {stats['open_positions']}"
        )
        self.send(msg)

    def notify_error(self, error: str):
        msg = f"<b>GEMINI BOT - ERROR</b>\n\n{error}"
        self.send(msg)
