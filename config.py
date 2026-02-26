#!/usr/bin/env python3
"""
Configuration for Gemini Flash Trading Bot.
Safety constants are HARDCODED — never loaded from .env.
"""

import os
import logging
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv
from pybit.unified_trading import HTTP

load_dotenv()

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler('gemini_bot.log', maxBytes=10*1024*1024, backupCount=5),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('gemini_bot')

trade_logger = logging.getLogger('gemini_trades')
trade_logger.setLevel(logging.INFO)
_th = RotatingFileHandler('gemini_trades.log', maxBytes=20*1024*1024, backupCount=3)
_th.setFormatter(logging.Formatter('%(asctime)s | %(message)s'))
trade_logger.addHandler(_th)


class Config:
    """Bot configuration. Safety limits are hardcoded constants."""

    # ── API keys (from .env) ──
    BYBIT_API_KEY = os.getenv('BYBIT_API_KEY', '')
    BYBIT_API_SECRET = os.getenv('BYBIT_API_SECRET', '')
    BYBIT_TESTNET = os.getenv('BYBIT_TESTNET', 'true').lower() == 'true'

    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
    GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')
    GEMINI_BASE_URL = os.getenv('GEMINI_BASE_URL', 'https://generativelanguage.googleapis.com/v1beta/openai/')

    TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN', '')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')

    COINGLASS_API_KEY = os.getenv('COINGLASS_API_KEY', '')

    # ── Trading config ──
    SYMBOLS = ['SOLUSDT']
    ACCOUNT_SIZE = 10_000

    # ── HARDCODED SAFETY CONSTANTS — DO NOT CHANGE ──
    FIXED_RISK_AMOUNT = 50             # Fixed $50 USDT risk per trade
    MAX_RISK_PER_TRADE = 0.01          # 1% ($100) — fallback cap
    MAX_RISK_ESCALATED = 0.015         # 1.5% ($150) — fallback cap
    DAILY_LOSS_LIMIT = 0.0275          # 2.75% ($275)
    MAX_DRAWDOWN_TOTAL = 0.045         # 4.5% ($450)
    MIN_TRADE_VALUE_PCT = 0.06         # 6% ($600) of balance
    MAX_MARGIN_EXPOSURE = 0.15         # 15% ($1500) of balance
    MAX_TRADES_PER_DAY = 2
    DAILY_PROFIT_CAP = 0.03            # 3% ($300)
    MIN_RR_RATIO = 1.5
    MIN_CONFIDENCE = 7                 # Gemini signal threshold
    SL_PLACEMENT_TIMEOUT = 30          # seconds
    CIRCUIT_BREAKER_LOSSES = 3
    CIRCUIT_BREAKER_PAUSE_H = 24
    PROFIT_DISTRIBUTION_MAX = 0.30     # No single day > 30% of total profit

    # ── Timing ──
    ANALYSIS_INTERVAL_SEC = 900        # 15 minutes
    MONITOR_INTERVAL_SEC = 30          # Position check
    STATE_FILE = 'trading_state.json'

    # ── Partial / Trailing ──
    PARTIAL_CLOSE_PCT = 0.50           # Close 50% at TP1
    TRAILING_ATR_MULT = 1.0
    MAX_HOLD_HOURS = 48

    @classmethod
    def create_bybit_session(cls) -> HTTP:
        """Create authenticated Bybit session."""
        return HTTP(
            demo=cls.BYBIT_TESTNET,
            api_key=cls.BYBIT_API_KEY,
            api_secret=cls.BYBIT_API_SECRET
        )
