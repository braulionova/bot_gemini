#!/usr/bin/env python3
"""
Technical indicator calculations.
EMA, RSI, MACD, Bollinger, ATR, ADX, Stochastic RSI, OBV, Order Blocks.
"""

from typing import Dict, List
import pandas as pd
import numpy as np
from config import logger


class IndicatorEngine:
    """Calculates technical indicators for multi-timeframe analysis."""

    @staticmethod
    def calculate_all(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all indicators on a DataFrame of OHLCV candles."""
        if df.empty or len(df) < 50:
            return df

        # ── EMAs ──
        df['ema_50'] = df['close'].ewm(span=50).mean()
        df['ema_200'] = df['close'].ewm(span=200).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()

        # ── RSI (14) ──
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0.0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # ── MACD (12, 26, 9) ──
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # ── Bollinger Bands (20, 2) ──
        df['bb_middle'] = df['sma_20']
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)

        # ── ATR (14) ──
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['atr'] = true_range.rolling(14).mean()

        # ── ADX (14) ──
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
        smooth_tr = true_range.rolling(14).sum()
        smooth_plus = plus_dm.rolling(14).sum()
        smooth_minus = minus_dm.rolling(14).sum()
        df['plus_di'] = 100 * (smooth_plus / smooth_tr)
        df['minus_di'] = 100 * (smooth_minus / smooth_tr)
        dx = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
        df['adx'] = dx.rolling(14).mean()

        # ── Stochastic RSI ──
        rsi_series = df['rsi']
        rsi_min = rsi_series.rolling(14).min()
        rsi_max = rsi_series.rolling(14).max()
        stoch_rsi = (rsi_series - rsi_min) / (rsi_max - rsi_min)
        df['stoch_rsi_k'] = stoch_rsi.rolling(3).mean() * 100
        df['stoch_rsi_d'] = df['stoch_rsi_k'].rolling(3).mean()

        # ── OBV ──
        obv = [0]
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv.append(obv[-1] + df['volume'].iloc[i])
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv.append(obv[-1] - df['volume'].iloc[i])
            else:
                obv.append(obv[-1])
        df['obv'] = obv
        df['obv_sma'] = pd.Series(obv).rolling(20).mean().values

        # ── Volume ratio ──
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']

        # ── VWAP (anchored to daily sessions) ──
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['tp_vol'] = df['typical_price'] * df['volume']
        if 'timestamp' in df.columns:
            df['_date'] = df['timestamp'].dt.date
            df['cum_tp_vol'] = df.groupby('_date')['tp_vol'].cumsum()
            df['cum_vol_grp'] = df.groupby('_date')['volume'].cumsum()
            df['vwap'] = df['cum_tp_vol'] / df['cum_vol_grp']
            df.drop(columns=['_date', 'cum_tp_vol', 'cum_vol_grp'], inplace=True)
        else:
            rolling_tp_vol = df['tp_vol'].rolling(window=24).sum()
            rolling_vol = df['volume'].rolling(window=24).sum()
            df['vwap'] = rolling_tp_vol / rolling_vol
        df.drop(columns=['tp_vol'], inplace=True)

        # ── CVD (Cumulative Volume Delta) ──
        candle_delta = df.apply(
            lambda r: r['volume'] if r['close'] >= r['open'] else -r['volume'], axis=1
        )
        df['cvd'] = candle_delta.cumsum()
        df['cvd_sma'] = df['cvd'].rolling(window=20).mean()

        return df

    @staticmethod
    def calculate_pivot_points(daily_df: pd.DataFrame) -> Dict:
        """Calculate Classic Pivot Points from the previous daily candle."""
        if daily_df.empty or len(daily_df) < 2:
            return {}
        prev = daily_df.iloc[-2]  # Previous completed day
        h = float(prev['high'])
        l = float(prev['low'])
        c = float(prev['close'])
        pp = (h + l + c) / 3
        r1 = 2 * pp - l
        s1 = 2 * pp - h
        r2 = pp + (h - l)
        s2 = pp - (h - l)
        r3 = h + 2 * (pp - l)
        s3 = l - 2 * (h - pp)
        return {
            'pp': round(pp, 4),
            'r1': round(r1, 4), 'r2': round(r2, 4), 'r3': round(r3, 4),
            's1': round(s1, 4), 's2': round(s2, 4), 's3': round(s3, 4),
        }

    @staticmethod
    def detect_order_blocks(df: pd.DataFrame) -> List[Dict]:
        """Detect ICT Order Blocks (unmitigated supply/demand zones)."""
        if df.empty or len(df) < 20:
            return []

        order_blocks = []
        atr = df['atr'].iloc[-1] if 'atr' in df.columns else None
        if atr is None or pd.isna(atr) or atr <= 0:
            return []

        avg_volume = df['volume'].rolling(20).mean()
        current_price = df['close'].iloc[-1]

        for i in range(5, len(df) - 1):
            atr_i = df['atr'].iloc[i] if not pd.isna(df['atr'].iloc[i]) else atr

            # Bullish OB: strong move up from a bearish candle
            move_up = df['close'].iloc[i] - df['low'].iloc[max(0, i-3)]
            if move_up > atr_i * 1.5:
                vol_avg = avg_volume.iloc[i] if not pd.isna(avg_volume.iloc[i]) else 0
                if vol_avg > 0 and df['volume'].iloc[i] >= vol_avg:
                    # Find the last bearish candle before the move
                    for j in range(i-1, max(i-5, 0), -1):
                        if df['close'].iloc[j] < df['open'].iloc[j]:
                            zone_low = df['low'].iloc[j]
                            zone_high = df['high'].iloc[j]
                            # Check not mitigated
                            mitigated = any(df['low'].iloc[k] < zone_low for k in range(j+1, len(df)))
                            if not mitigated and current_price > zone_low:
                                order_blocks.append({
                                    'type': 'bullish', 'zone_high': zone_high,
                                    'zone_low': zone_low, 'strength': round(move_up / atr_i, 2),
                                })
                            break

            # Bearish OB: strong move down from a bullish candle
            move_down = df['high'].iloc[max(0, i-3)] - df['close'].iloc[i]
            if move_down > atr_i * 1.5:
                vol_avg = avg_volume.iloc[i] if not pd.isna(avg_volume.iloc[i]) else 0
                if vol_avg > 0 and df['volume'].iloc[i] >= vol_avg:
                    for j in range(i-1, max(i-5, 0), -1):
                        if df['close'].iloc[j] > df['open'].iloc[j]:
                            zone_low = df['low'].iloc[j]
                            zone_high = df['high'].iloc[j]
                            mitigated = any(df['high'].iloc[k] > zone_high for k in range(j+1, len(df)))
                            if not mitigated and current_price < zone_high:
                                order_blocks.append({
                                    'type': 'bearish', 'zone_high': zone_high,
                                    'zone_low': zone_low, 'strength': round(move_down / atr_i, 2),
                                })
                            break

        order_blocks.sort(key=lambda x: x['strength'], reverse=True)
        return order_blocks[:5]

    @staticmethod
    def calculate_tp_levels(df: pd.DataFrame, pivot_points: Dict, atr: float) -> Dict:
        """
        Calculate professional TP target levels anchored to real technical structure.
        Returns ordered TP candidates for LONG and SHORT setups.
        """
        if df.empty or atr <= 0:
            return {}

        current_price = float(df['close'].iloc[-1])

        # Collect key structural levels
        resistances = []  # levels above price (for LONG TPs)
        supports = []     # levels below price (for SHORT TPs)

        # 1. Pivot Points
        for key, val in pivot_points.items():
            if val and val != current_price:
                if val > current_price:
                    resistances.append({'level': val, 'source': f'Pivot {key.upper()}'})
                else:
                    supports.append({'level': val, 'source': f'Pivot {key.upper()}'})

        # 2. Recent swing highs (last 50 candles, 4H-level structure)
        recent = df.tail(50)
        for i in range(2, len(recent) - 2):
            h = float(recent['high'].iloc[i])
            if h > float(recent['high'].iloc[i-1]) and h > float(recent['high'].iloc[i+1]):
                if h > current_price * 1.003:  # min 0.3% away
                    resistances.append({'level': round(h, 2), 'source': 'Swing High'})

        # 3. Recent swing lows
        for i in range(2, len(recent) - 2):
            l = float(recent['low'].iloc[i])
            if l < float(recent['low'].iloc[i-1]) and l < float(recent['low'].iloc[i+1]):
                if l < current_price * 0.997:  # min 0.3% away
                    supports.append({'level': round(l, 2), 'source': 'Swing Low'})

        # 4. Bollinger Band levels
        if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
            bb_upper = float(df['bb_upper'].iloc[-1])
            bb_lower = float(df['bb_lower'].iloc[-1])
            if not pd.isna(bb_upper) and bb_upper > current_price:
                resistances.append({'level': round(bb_upper, 2), 'source': 'BB Upper'})
            if not pd.isna(bb_lower) and bb_lower < current_price:
                supports.append({'level': round(bb_lower, 2), 'source': 'BB Lower'})

        # 5. ATR-based minimum targets (1.5x, 2.5x, 4x ATR from price)
        for mult, label in [(1.5, '1.5xATR'), (2.5, '2.5xATR'), (4.0, '4xATR')]:
            resistances.append({'level': round(current_price + mult * atr, 2), 'source': label})
            supports.append({'level': round(current_price - mult * atr, 2), 'source': label})

        # 6. VWAP
        if 'vwap' in df.columns:
            vwap_val = float(df['vwap'].iloc[-1])
            if not pd.isna(vwap_val) and vwap_val > 0:
                if vwap_val > current_price * 1.003:
                    resistances.append({'level': round(vwap_val, 2), 'source': 'VWAP'})
                elif vwap_val < current_price * 0.997:
                    supports.append({'level': round(vwap_val, 2), 'source': 'VWAP'})

        # Sort: resistances ascending (nearest first), supports descending (nearest first)
        resistances = sorted(resistances, key=lambda x: x['level'])
        supports = sorted(supports, key=lambda x: x['level'], reverse=True)

        # Deduplicate levels within 0.5% of each other (keep first/nearest)
        def dedup(levels: list, tol_pct: float = 0.005) -> list:
            out = []
            for lv in levels:
                if not out or abs(lv['level'] - out[-1]['level']) / out[-1]['level'] > tol_pct:
                    out.append(lv)
            return out

        resistances = dedup(resistances)
        supports = dedup(supports)

        # Build structured output: pick the best 3 TPs for each direction
        long_tps = [{'tp': r['level'], 'source': r['source'], 'rr': round((r['level'] - current_price) / (atr * 0.8), 2)} for r in resistances[:5]]
        short_tps = [{'tp': s['level'], 'source': s['source'], 'rr': round((current_price - s['level']) / (atr * 0.8), 2)} for s in supports[:5]]

        return {
            'current_price': round(current_price, 2),
            'atr': round(atr, 2),
            'long_tp_candidates': long_tps,    # for BUY setups
            'short_tp_candidates': short_tps,  # for SELL setups
        }

    @staticmethod
    def get_indicator_summary(df: pd.DataFrame) -> Dict:
        """Extract latest indicator values as a dict for Gemini prompt."""
        if df.empty:
            return {}

        c = df.iloc[-1]
        prev = df.iloc[-2] if len(df) >= 2 else c

        def safe(val):
            if pd.isna(val):
                return None
            return round(float(val), 4)

        summary = {
            'price': safe(c['close']),
            'open': safe(c['open']),
            'high': safe(c['high']),
            'low': safe(c['low']),
            'volume': safe(c['volume']),
            'ema_50': safe(c.get('ema_50')),
            'ema_200': safe(c.get('ema_200')),
            'sma_20': safe(c.get('sma_20')),
            'rsi': safe(c.get('rsi')),
            'macd': safe(c.get('macd')),
            'macd_signal': safe(c.get('macd_signal')),
            'macd_hist': safe(c.get('macd_hist')),
            'macd_hist_prev': safe(prev.get('macd_hist')),
            'bb_upper': safe(c.get('bb_upper')),
            'bb_lower': safe(c.get('bb_lower')),
            'bb_middle': safe(c.get('bb_middle')),
            'atr': safe(c.get('atr')),
            'adx': safe(c.get('adx')),
            'plus_di': safe(c.get('plus_di')),
            'minus_di': safe(c.get('minus_di')),
            'stoch_rsi_k': safe(c.get('stoch_rsi_k')),
            'stoch_rsi_d': safe(c.get('stoch_rsi_d')),
            'obv': safe(c.get('obv')),
            'obv_sma': safe(c.get('obv_sma')),
            'volume_ratio': safe(c.get('volume_ratio')),
            'vwap': safe(c.get('vwap')),
            'cvd': safe(c.get('cvd')),
            'cvd_sma': safe(c.get('cvd_sma')),
        }
        # VWAP bias: above = bullish, below = bearish
        vwap = c.get('vwap')
        price = c.get('close')
        if vwap and price and not pd.isna(vwap) and not pd.isna(price) and float(vwap) > 0:
            vwap_pct = ((float(price) - float(vwap)) / float(vwap)) * 100
            summary['vwap_bias_pct'] = round(vwap_pct, 3)
        # CVD trend: rising = buying pressure, falling = selling
        if len(df) >= 5:
            cvd_now = c.get('cvd')
            cvd_5ago = df.iloc[-5].get('cvd') if 'cvd' in df.columns else None
            if cvd_now is not None and cvd_5ago is not None and not pd.isna(cvd_now) and not pd.isna(cvd_5ago):
                summary['cvd_trend'] = 'rising' if float(cvd_now) > float(cvd_5ago) else 'falling'

        # Price action context (last 5 candles)
        recent = []
        for i in range(-5, 0):
            if abs(i) <= len(df):
                row = df.iloc[i]
                recent.append({
                    'open': safe(row['open']), 'high': safe(row['high']),
                    'low': safe(row['low']), 'close': safe(row['close']),
                    'volume': safe(row['volume']),
                })
        summary['recent_candles'] = recent

        return summary
