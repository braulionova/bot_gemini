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
        return """You are a professional crypto futures trader managing a $10,000 funded account.
You think like an institutional trader: confluence of macro + structure + momentum + positioning.

ROLE: Analyze all data sources and return ONLY a valid JSON trading decision. No extra text.

CHALLENGE RULES:
- Max risk per trade: 1% ($100) | Daily loss limit: 2.5% | Max drawdown: 4.5%
- Minimum R:R: 1.5:1 | Max 2 trades/day | Target: 0.8-1.2% daily

═══ PROFESSIONAL ANALYSIS FRAMEWORK ═══

STEP 1 — MACRO CONTEXT (BTC + Dominance):
- BTC is the market leader. SOL follows BTC 70-80% of the time.
- If BTC is bearish (negative 24h, below VWAP), bias SHORT for SOL unless SOL shows extreme relative strength.
- If BTC is bullish, bias LONG. If BTC is neutral, rely entirely on SOL structure.
- BTC Dominance >55% = BTC season, altcoins lag → be cautious on LONG SOL.
- BTC Dominance <48% = Altcoin season → SOL can outperform, LONG bias gets extra weight.

STEP 2 — TREND STRUCTURE (Daily + 4H):
- EMA50/EMA200 alignment: price above both = bullish, below both = bearish.
- ADX >25 = trending market (trade with trend). ADX <20 = ranging (trade extremes only).
- Market structure: higher highs/lows = uptrend. Lower highs/lows = downtrend.

STEP 3 — INSTITUTIONAL POSITIONING (Long/Short Ratio + Funding):
- Long/Short Ratio: if >65% accounts are long → CONTRARIAN BEARISH signal (crowded longs get liquidated).
- If >65% accounts are short → CONTRARIAN BULLISH signal.
- Funding Rate: positive = longs paying shorts (bullish sentiment), negative = shorts paying longs (bearish).
- Funding trend rising = sentiment escalating. High positive funding (>0.05%) = warning sign for longs.
- Funding history: 5 consecutive positive rates = crowded long positioning → fade longs.

STEP 4 — VWAP ANALYSIS (Institutional Benchmark):
- Price above VWAP = institutional buyers in control → bullish intraday bias.
- Price below VWAP = sellers in control → bearish intraday bias.
- Price far above VWAP (+1% or more) = stretched, avoid chasing longs.
- Price retesting VWAP from above = potential long entry in uptrend.

STEP 5 — CVD (Cumulative Volume Delta):
- CVD rising = more buying aggression than selling (bullish confirmation).
- CVD falling while price rises = DIVERGENCE → price likely to reverse (bearish signal).
- CVD falling = selling pressure dominant (bearish).
- CVD rising while price falls = divergence → potential reversal (bullish).

STEP 6 — KEY LEVELS (Pivot Points + Order Blocks):
- Pivot Points (PP, R1/S1, R2/S2) are levels ALL traders watch → high probability reaction zones.
- Price near S1/S2 = support zone, look for longs if trend is up.
- Price near R1/R2 = resistance zone, look for shorts if trend is down.
- Order Blocks (institutional supply/demand zones) near pivot levels = high-confluence entry.

STEP 7 — MOMENTUM TIMING (1H + 15m):
- Enter on 1H pullback to EMA/VWAP/Order Block in trend direction.
- Trigger: 15m MACD histogram turning in trade direction + StochRSI crossing.
- Volume ratio >1.2 on entry candle = volume confirmation.
- OBV trend must match trade direction.

CONFIDENCE SCALE:
- 1-4: No setup or conflicting signals → HOLD
- 5-6: Partial setup, missing key confluences → HOLD
- 7: 3-4 confluences aligned (trend + structure + momentum + one positioning signal) → TRADE
- 8: 5+ confluences (adds VWAP, CVD, or BTC alignment) → TRADE
- 9: Institutional-grade setup (all signals aligned + contrarian positioning extreme) → TRADE
- 10: Textbook, almost never happens

CONFLUENCE CHECKLIST (need 3+ for confidence 7, 5+ for confidence 8+):
✓ BTC aligned with trade direction
✓ Trend (EMA alignment Daily+4H)
✓ ADX >25 (trending market)
✓ Price on correct side of VWAP
✓ CVD confirming direction
✓ Long/Short ratio contrarian signal
✓ Funding rate supports direction
✓ Price near Pivot support/resistance
✓ Order Block as entry zone
✓ 1H momentum (MACD, StochRSI)
✓ Volume confirmation

STOP LOSS PLACEMENT:
- Place SL at nearest significant structure: swing high/low, order block boundary, or pivot level.
- Add small buffer (0.1-0.2 ATR) beyond the level.
- SL should be tight enough for R:R ≥ 1.5 but logical enough that normal price action won't hit it.

TAKE PROFIT PLACEMENT (CRITICAL — use the pre-calculated levels provided):
- You will receive "CALCULATED TP TARGETS" with levels anchored to real structure.
- TP1: pick the NEAREST meaningful level with R:R ≥ 1.5 (first natural resistance/support).
- TP2: pick the NEXT significant level beyond TP1 (second structural target).
- NEVER place TP at an arbitrary price unrelated to structure (e.g. round numbers only).
- NEVER place TP closer than 1 ATR from entry.
- TP must sit BELOW a resistance (for longs) or ABOVE a support (for shorts) — not at the level itself.

RESPOND WITH ONLY THIS JSON (no markdown, no extra text). Keep reasoning under 60 words:
{"confidence":7,"action":"SELL","entry_price":170.50,"stop_loss":172.00,"take_profit_1":167.00,"take_profit_2":164.00,"risk_reward_ratio":2.3,"reasoning":"short reason here"}

CRITICAL RULES:
- Trade WITH the macro trend (BTC + Daily structure).
- NEVER fight a strong trend. RSI extremes in trending markets = continuation.
- Crowded positioning (65%+ one side) is your edge — institutions fade the crowd.
- Only HOLD if truly no setup exists or signals conflict sharply."""

    def _build_user_prompt(self, symbol: str, data_package: Dict) -> str:
        parts = [f"=== {symbol} PROFESSIONAL ANALYSIS ===\n"]

        # ── MACRO CONTEXT ──
        parts.append("── MACRO CONTEXT ──")
        btc = data_package.get('btc_context', {})
        if btc.get('available'):
            parts.append(
                f"BTC: ${btc['price']:,.2f} | 24h: {btc['change_24h_pct']:+.2f}% | "
                f"Range position: {btc['position_in_range_pct']:.0f}% | Bias: {btc['bias'].upper()}"
            )
        btc_dom = data_package.get('btc_dominance', {})
        if btc_dom.get('available'):
            parts.append(
                f"BTC Dominance: {btc_dom['btc_dominance']:.1f}% | ETH: {btc_dom['eth_dominance']:.1f}% | "
                f"Regime: {btc_dom['market_regime'].upper()}"
            )
        fg = data_package.get('fear_greed', {})
        if fg.get('available'):
            parts.append(f"Fear & Greed: {fg['value']}/100 ({fg['classification']})")
        parts.append("")

        # ── SOL TICKER ──
        ticker = data_package.get('ticker', {})
        if ticker:
            parts.append("── SOL MARKET ──")
            parts.append(
                f"Price: ${ticker.get('last_price', 'N/A')} | "
                f"24h: {ticker.get('price_24h_pct', 0):+.2f}% | "
                f"Range: ${ticker.get('low_24h', 0):.2f}-${ticker.get('high_24h', 0):.2f}"
            )
            parts.append(
                f"Volume 24h: ${ticker.get('turnover_24h', 0):,.0f} | "
                f"Open Interest: {ticker.get('open_interest', 0):,.0f}"
            )
            parts.append("")

        # ── POSITIONING & SENTIMENT ──
        parts.append("── POSITIONING & SENTIMENT ──")
        ls = data_package.get('long_short_ratio', {})
        if ls.get('available'):
            buy_pct = ls.get('buy_ratio', 0) * 100
            sell_pct = ls.get('sell_ratio', 0) * 100
            crowded = "CROWDED LONGS ⚠" if buy_pct > 65 else ("CROWDED SHORTS ⚠" if sell_pct > 65 else "balanced")
            parts.append(
                f"Long/Short Ratio: {buy_pct:.1f}% long / {sell_pct:.1f}% short | {crowded} | "
                f"Trend: {ls.get('buy_ratio_trend', 'N/A')}"
            )
        fh = data_package.get('funding_history', {})
        if fh.get('available'):
            rates_str = ', '.join(f"{r:.5f}" for r in fh.get('rates', []))
            parts.append(
                f"Funding history (newest→oldest): [{rates_str}] | "
                f"Avg: {fh['average']:.5f} ({fh['trend']}, {fh['direction']})"
            )
        funding = data_package.get('funding_rate', {})
        if funding.get('available'):
            parts.append(f"Current funding: {funding['funding_rate']:.6f}")
        oi = data_package.get('open_interest', {})
        if oi.get('available'):
            parts.append(f"OI: ${oi.get('open_interest_usd', 0):,.0f} | 24h change: {oi.get('oi_change_24h', 0):+.2f}%")
        liqs = data_package.get('liquidations', {})
        if liqs.get('available'):
            parts.append(
                f"Liquidations 1h: longs=${liqs.get('long_liquidations', 0):,.0f} | "
                f"shorts=${liqs.get('short_liquidations', 0):,.0f}"
            )
        parts.append("")

        # ── TECHNICAL INDICATORS (multi-timeframe) ──
        parts.append("── TECHNICAL ANALYSIS ──")
        indicators = data_package.get('indicators', {})
        for tf in ['daily', '4h', '1h', '15m']:
            if tf in indicators:
                ind = indicators[tf]
                vwap = ind.get('vwap')
                vwap_bias = ind.get('vwap_bias_pct')
                cvd_trend = ind.get('cvd_trend', 'N/A')
                parts.append(f"[{tf.upper()}]")
                parts.append(
                    f"  Price: {ind.get('price')} | EMA50: {ind.get('ema_50')} | EMA200: {ind.get('ema_200')} | "
                    f"VWAP: {vwap} ({vwap_bias:+.2f}% from VWAP)" if vwap and vwap_bias is not None
                    else f"  Price: {ind.get('price')} | EMA50: {ind.get('ema_50')} | EMA200: {ind.get('ema_200')}"
                )
                parts.append(
                    f"  RSI: {ind.get('rsi')} | MACD hist: {ind.get('macd_hist')} (prev: {ind.get('macd_hist_prev')}) | "
                    f"ADX: {ind.get('adx')} (+DI:{ind.get('plus_di')} -DI:{ind.get('minus_di')})"
                )
                parts.append(
                    f"  StochRSI K:{ind.get('stoch_rsi_k')} D:{ind.get('stoch_rsi_d')} | "
                    f"BB: {ind.get('bb_lower')}-{ind.get('bb_upper')} | ATR: {ind.get('atr')}"
                )
                parts.append(
                    f"  OBV: {ind.get('obv')} (SMA:{ind.get('obv_sma')}) | "
                    f"Vol ratio: {ind.get('volume_ratio')} | CVD trend: {cvd_trend}"
                )
                parts.append("")

        # ── KEY LEVELS ──
        parts.append("── KEY LEVELS ──")
        piv = data_package.get('pivot_points', {})
        if piv:
            parts.append(
                f"Daily Pivots: S3={piv.get('s3')} S2={piv.get('s2')} S1={piv.get('s1')} | "
                f"PP={piv.get('pp')} | R1={piv.get('r1')} R2={piv.get('r2')} R3={piv.get('r3')}"
            )
        obs = data_package.get('order_blocks', [])
        if obs:
            parts.append("Order Blocks (4H):")
            for ob in obs[:3]:
                parts.append(
                    f"  {ob['type'].upper()} OB: ${ob['zone_low']:.2f}-${ob['zone_high']:.2f} "
                    f"(strength: {ob['strength']})"
                )
        parts.append("")

        # ── ORDERBOOK ──
        ob_data = data_package.get('orderbook', {})
        if ob_data.get('available'):
            parts.append(
                f"── ORDERBOOK ──\n"
                f"Bid/Ask imbalance: {ob_data['bid_ask_imbalance']:.3f} | "
                f"Near pressure: {ob_data['near_pressure']:.3f} | "
                f"Spread: {ob_data['spread']}"
            )

        # ── CALCULATED TP TARGETS (pre-calculated from structure) ──
        tp_data = data_package.get('tp_levels', {})
        if tp_data:
            parts.append("\n── CALCULATED TP TARGETS (anchored to real structure) ──")
            parts.append(f"Price: ${tp_data.get('current_price')} | ATR(4H): {tp_data.get('atr')}")
            long_tps = tp_data.get('long_tp_candidates', [])
            if long_tps:
                parts.append("LONG TP candidates (nearest → farthest):")
                for t in long_tps[:5]:
                    parts.append(f"  ${t['tp']} [{t['source']}] indicative RR~{t['rr']}")
            short_tps = tp_data.get('short_tp_candidates', [])
            if short_tps:
                parts.append("SHORT TP candidates (nearest → farthest):")
                for t in short_tps[:5]:
                    parts.append(f"  ${t['tp']} [{t['source']}] indicative RR~{t['rr']}")
            parts.append("→ Use these levels for take_profit_1 and take_profit_2. Pick the nearest valid R:R ≥ 1.5.")

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
