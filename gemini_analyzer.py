#!/usr/bin/env python3
"""
Gemini Flash AI analyzer — the analytical brain.
Sends market data to Gemini via OpenAI-compatible API, gets trading decisions.
"""

import json
import re
from typing import Dict, Optional
from openai import OpenAI
from config import Config, logger


class GeminiAnalyzer:
    """Interfaces with Gemini Flash for trade analysis."""

    def __init__(self):
        self.client = OpenAI(
            api_key=Config.GEMINI_API_KEY,
            base_url=Config.GEMINI_BASE_URL,
        )
        self.model = Config.GEMINI_MODEL

    def _build_system_prompt(self) -> str:
        return """You are an elite crypto futures trading analyst for a funded account challenge.

ROLE: Analyze market data and return ONLY a valid JSON trading decision. No extra text.
CAPITAL PRESERVATION is priority #1, but you MUST take trades when setups align.

CHALLENGE RULES:
- Max risk per trade: 1% of account
- Daily loss limit: 2.5% | Max drawdown: 4.5%
- Minimum R:R ratio: 1.5:1 | Max 2 trades/day
- Target: 0.8-1.2% daily profit

ANALYSIS FRAMEWORK:
1. Determine TREND from Daily/4H (EMA alignment, structure, ADX)
2. Find ENTRY ZONE on 1H (pullback to EMA, order block, support/resistance)
3. TRIGGER on 15m (momentum shift, MACD histogram turn, StochRSI cross)
4. CONFLUENCE: Need 3+ factors agreeing (trend, structure, momentum, volume, orderbook)

TRADING WITH THE TREND:
- If Daily+4H are bearish → look for SHORT entries on 1H/15m bounces
- If Daily+4H are bullish → look for LONG entries on 1H/15m pullbacks
- Ranging market → trade from extremes of range with tight stops
- Do NOT require all timeframes to agree — lower TFs are for timing, not direction

ORDER BLOCKS: Strong institutional zones — ideal entries when price retests them with trend.

CONFIDENCE SCALE:
- 1-3: No clear setup, HOLD
- 4-5: Setup forming but not ready, HOLD
- 6: Decent setup but missing 1 factor, borderline
- 7: Good setup with 3+ confluence factors → TRADE
- 8-9: Strong setup with clear trend + structure + momentum → TRADE
- 10: Textbook setup, rare

RESPOND WITH ONLY THIS JSON (no markdown, no extra text). Keep reasoning under 50 words:
{"confidence":7,"action":"SELL","entry_price":170.50,"stop_loss":172.00,"take_profit_1":167.00,"take_profit_2":164.00,"risk_reward_ratio":2.3,"reasoning":"short reason here"}

RULES:
- Trade WITH the trend. Shorts in downtrends, longs in uptrends.
- Stop loss at logical structure level (swing high/low + small buffer)
- If you see a valid trend-following setup with 3+ factors, confidence should be 7+
- Do NOT default to HOLD — actively look for setups. Only HOLD if truly no setup exists.
- RSI extremes in trending markets are continuation signals, not reversal signals
- Volume/OBV confirming trend = extra confidence"""

    def _build_user_prompt(self, symbol: str, data_package: Dict) -> str:
        parts = [f"Analyze {symbol} for a potential trade entry.\n"]

        # Ticker info
        ticker = data_package.get('ticker', {})
        if ticker:
            parts.append(f"Current price: ${ticker.get('last_price', 'N/A')}")
            parts.append(f"24h change: {ticker.get('price_24h_pct', 0):.2f}%")
            parts.append(f"24h range: ${ticker.get('low_24h', 0):.2f} - ${ticker.get('high_24h', 0):.2f}")
            parts.append(f"24h volume: ${ticker.get('turnover_24h', 0):,.0f}")
            parts.append(f"Open interest: {ticker.get('open_interest', 0):,.0f}\n")

        # Multi-timeframe indicators
        indicators = data_package.get('indicators', {})
        for tf in ['daily', '4h', '1h', '15m']:
            if tf in indicators:
                ind = indicators[tf]
                parts.append(f"--- {tf.upper()} Timeframe ---")
                parts.append(f"Price: {ind.get('price')} | EMA50: {ind.get('ema_50')} | EMA200: {ind.get('ema_200')}")
                parts.append(f"RSI: {ind.get('rsi')} | MACD hist: {ind.get('macd_hist')} (prev: {ind.get('macd_hist_prev')})")
                parts.append(f"ADX: {ind.get('adx')} | +DI: {ind.get('plus_di')} | -DI: {ind.get('minus_di')}")
                parts.append(f"BB: {ind.get('bb_lower')} - {ind.get('bb_upper')} | ATR: {ind.get('atr')}")
                parts.append(f"StochRSI K: {ind.get('stoch_rsi_k')} D: {ind.get('stoch_rsi_d')}")
                parts.append(f"OBV: {ind.get('obv')} (SMA: {ind.get('obv_sma')}) | Vol ratio: {ind.get('volume_ratio')}")
                parts.append("")

        # Order blocks
        obs = data_package.get('order_blocks', [])
        if obs:
            parts.append("--- ORDER BLOCKS ---")
            for ob in obs[:3]:
                parts.append(f"  {ob['type'].upper()}: ${ob['zone_low']:.2f} - ${ob['zone_high']:.2f} (strength: {ob['strength']})")
            parts.append("")

        # Orderbook
        ob_data = data_package.get('orderbook', {})
        if ob_data.get('available'):
            parts.append(f"Orderbook: bid/ask imbalance={ob_data['bid_ask_imbalance']:.3f}, near pressure={ob_data['near_pressure']:.3f}")

        # Funding rate
        funding = data_package.get('funding_rate', {})
        if funding.get('available'):
            parts.append(f"Funding rate: {funding['funding_rate']:.6f}")

        # Fear & Greed
        fg = data_package.get('fear_greed', {})
        if fg.get('available'):
            parts.append(f"Fear & Greed: {fg['value']} ({fg['classification']})")

        # Open Interest
        oi = data_package.get('open_interest', {})
        if oi.get('available'):
            parts.append(f"OI: ${oi.get('open_interest_usd', 0):,.0f} (24h change: {oi.get('oi_change_24h', 0):.2f}%)")

        # Liquidations
        liqs = data_package.get('liquidations', {})
        if liqs.get('available'):
            parts.append(f"Liquidations 1h: longs=${liqs.get('long_liquidations', 0):,.0f} shorts=${liqs.get('short_liquidations', 0):,.0f}")

        parts.append("\nRespond with ONLY the JSON object. No markdown, no explanation outside JSON.")
        return '\n'.join(parts)

    def _extract_json(self, content: str) -> Optional[Dict]:
        """Try multiple strategies to extract valid JSON from Gemini response."""
        # Clean markdown wrappers
        cleaned = content
        if '```json' in cleaned:
            cleaned = cleaned.split('```json')[1].split('```')[0]
        elif '```' in cleaned:
            cleaned = cleaned.split('```')[1].split('```')[0]
        cleaned = cleaned.strip()

        # Strategy 1: Direct parse
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Strategy 2: Find complete JSON object with regex
        match = re.search(r'\{[^{}]*"action"\s*:\s*"[^"]*"[^{}]*\}', content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # Strategy 3: Fix truncated JSON — find last complete key:value pair
        for candidate in [cleaned, content]:
            # Find the start of JSON
            start = candidate.find('{')
            if start == -1:
                continue
            text = candidate[start:]

            # Find the last complete "key": value pattern
            # Match patterns like "key": "value", or "key": number,
            pairs = list(re.finditer(
                r'"(\w+)"\s*:\s*(?:"(?:[^"\\]|\\.)*"|[\d.]+(?:e[+-]?\d+)?|null|true|false)',
                text
            ))
            if not pairs:
                continue

            # Rebuild JSON from complete pairs
            last_complete = pairs[-1].end()
            truncated = text[:last_complete].rstrip().rstrip(',')
            truncated += '}'

            try:
                result = json.loads(truncated)
                # Only accept if it has at least action field
                if 'action' in result:
                    return result
            except json.JSONDecodeError:
                pass

        return None

    def analyze(self, symbol: str, data_package: Dict, _retry: int = 0) -> Optional[Dict]:
        """Call Gemini Flash and return parsed trading decision."""
        MAX_RETRIES = 2
        try:
            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(symbol, data_package)

            logger.info(f"Calling Gemini Flash for {symbol} analysis...")

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=1024,
            )

            content = response.choices[0].message.content
            result = self._extract_json(content)

            if result is None:
                logger.warning(f"Gemini JSON parse failed (attempt {_retry+1}/{MAX_RETRIES+1}): {content[:300]}")
                if _retry < MAX_RETRIES:
                    return self.analyze(symbol, data_package, _retry + 1)
                logger.error(f"Gemini JSON parse failed after {MAX_RETRIES+1} attempts")
                return None

            # Strictly required: action and confidence
            if 'action' not in result or 'confidence' not in result:
                logger.warning(f"Gemini missing action/confidence (attempt {_retry+1}). Got: {json.dumps(result)[:300]}")
                if _retry < MAX_RETRIES:
                    return self.analyze(symbol, data_package, _retry + 1)
                return None

            # Fill defaults for optional fields that may be truncated
            result.setdefault('reasoning', 'Truncated response')
            result.setdefault('risk_reward_ratio', 0)
            result.setdefault('entry_price', 0)
            result.setdefault('stop_loss', 0)
            result.setdefault('take_profit_1', 0)
            result.setdefault('take_profit_2', 0)

            action = result['action'].upper()
            if action not in ('BUY', 'SELL', 'HOLD'):
                logger.error(f"Invalid action from Gemini: {action}")
                return None

            logger.info(
                f"Gemini analysis: {action} | confidence={result['confidence']}/10 | "
                f"RR={result.get('risk_reward_ratio', 'N/A')} | {result['reasoning'][:100]}"
            )

            return result

        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            if _retry < MAX_RETRIES:
                return self.analyze(symbol, data_package, _retry + 1)
            return None
