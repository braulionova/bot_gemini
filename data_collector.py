#!/usr/bin/env python3
"""
Market data collector: Bybit klines, orderbook, funding rate + external sources.
"""

import requests
import pandas as pd
from typing import Dict, Optional
from pybit.unified_trading import HTTP
from config import Config, logger
from indicators import IndicatorEngine


class DataCollector:
    """Collects and packages market data for Gemini analysis."""

    def __init__(self, session: HTTP):
        self.session = session
        self.indicator_engine = IndicatorEngine()

    def get_klines(self, symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
        """Fetch klines from Bybit and return as DataFrame."""
        try:
            klines = self.session.get_kline(
                category="linear", symbol=symbol, interval=interval, limit=str(limit)
            )
            if not klines['result']['list']:
                return pd.DataFrame()
            data = []
            for k in reversed(klines['result']['list']):
                data.append({
                    'timestamp': int(k[0]), 'open': float(k[1]), 'high': float(k[2]),
                    'low': float(k[3]), 'close': float(k[4]), 'volume': float(k[5]),
                    'turnover': float(k[6])
                })
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            logger.error(f"Error fetching klines {symbol} {interval}: {e}")
            return pd.DataFrame()

    def get_multi_timeframe_klines(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """Fetch klines for all analysis timeframes: 15m, 1H, 4H, Daily."""
        timeframes = {
            '15m': '15',
            '1h': '60',
            '4h': '240',
            'daily': 'D',
        }
        result = {}
        for label, interval in timeframes.items():
            df = self.get_klines(symbol, interval, limit=200)
            if not df.empty and len(df) >= 50:
                df = self.indicator_engine.calculate_all(df)
            result[label] = df
        return result

    def get_orderbook(self, symbol: str) -> Dict:
        """Get orderbook pressure analysis."""
        try:
            ob = self.session.get_orderbook(category="linear", symbol=symbol, limit=50)
            if ob['retCode'] != 0:
                return {'available': False}
            result = ob['result']
            bids = [[float(p), float(q)] for p, q in result['b']]
            asks = [[float(p), float(q)] for p, q in result['a']]
            if not bids or not asks:
                return {'available': False}
            total_bid = sum(q for _, q in bids)
            total_ask = sum(q for _, q in asks)
            imbalance = total_bid / total_ask if total_ask > 0 else 1.0
            near_bid = sum(q for _, q in bids[:5])
            near_ask = sum(q for _, q in asks[:5])
            near_pressure = near_bid / near_ask if near_ask > 0 else 1.0
            spread = asks[0][0] - bids[0][0]
            return {
                'available': True,
                'bid_ask_imbalance': round(imbalance, 3),
                'near_pressure': round(near_pressure, 3),
                'spread': round(spread, 6),
                'top_bid': bids[0][0],
                'top_ask': asks[0][0],
            }
        except Exception as e:
            logger.error(f"Orderbook error {symbol}: {e}")
            return {'available': False}

    def get_funding_rate(self, symbol: str) -> Dict:
        """Get current funding rate."""
        try:
            ticker = self.session.get_tickers(category="linear", symbol=symbol)
            if ticker['result']['list']:
                info = ticker['result']['list'][0]
                return {
                    'available': True,
                    'funding_rate': float(info.get('fundingRate', 0)),
                    'next_funding_time': info.get('nextFundingTime', ''),
                }
        except Exception as e:
            logger.error(f"Funding rate error {symbol}: {e}")
        return {'available': False}

    def get_ticker(self, symbol: str) -> Dict:
        """Get current price info."""
        try:
            ticker = self.session.get_tickers(category="linear", symbol=symbol)
            if ticker['result']['list']:
                info = ticker['result']['list'][0]
                return {
                    'last_price': float(info['lastPrice']),
                    'price_24h_pct': float(info.get('price24hPcnt', 0)) * 100,
                    'high_24h': float(info.get('highPrice24h', 0)),
                    'low_24h': float(info.get('lowPrice24h', 0)),
                    'volume_24h': float(info.get('volume24h', 0)),
                    'turnover_24h': float(info.get('turnover24h', 0)),
                    'open_interest': float(info.get('openInterest', 0)),
                }
        except Exception as e:
            logger.error(f"Ticker error {symbol}: {e}")
        return {}

    def get_fear_greed_index(self) -> Dict:
        """Get Fear & Greed Index from alternative.me."""
        try:
            resp = requests.get('https://api.alternative.me/fng/?limit=1', timeout=10)
            if resp.status_code == 200:
                data = resp.json()['data'][0]
                return {
                    'available': True,
                    'value': int(data['value']),
                    'classification': data['value_classification'],
                }
        except Exception as e:
            logger.error(f"Fear & Greed error: {e}")
        return {'available': False, 'value': 50, 'classification': 'Neutral'}

    def get_open_interest(self, symbol: str) -> Dict:
        """Get Open Interest data from CoinGlass."""
        if not Config.COINGLASS_API_KEY:
            return {'available': False}
        try:
            headers = {'coinglassSecret': Config.COINGLASS_API_KEY}
            # Map symbol: SOLUSDT -> SOL
            coin = symbol.replace('USDT', '')
            resp = requests.get(
                f'https://open-api.coinglass.com/public/v2/open_interest?symbol={coin}&time_type=all',
                headers=headers, timeout=10
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get('data'):
                    return {
                        'available': True,
                        'open_interest_usd': data['data'].get('openInterest', 0),
                        'oi_change_24h': data['data'].get('oiChange24h', 0),
                    }
        except Exception as e:
            logger.error(f"CoinGlass OI error: {e}")
        return {'available': False}

    def get_long_short_ratio(self, symbol: str) -> Dict:
        """Get long/short ratio from Bybit (top traders + all accounts)."""
        try:
            result = {}
            # All accounts ratio
            resp = self.session.get_long_short_ratio(
                category="linear", symbol=symbol, period="1h", limit=5
            )
            if resp.get('result', {}).get('list'):
                latest = resp['result']['list'][0]
                result['buy_ratio'] = float(latest.get('buyRatio', 0))
                result['sell_ratio'] = float(latest.get('sellRatio', 0))
                result['available'] = True
                # Trend: compare first vs last in the 5-bar window
                if len(resp['result']['list']) >= 3:
                    oldest = resp['result']['list'][-1]
                    old_buy = float(oldest.get('buyRatio', 0.5))
                    result['buy_ratio_trend'] = 'rising' if result['buy_ratio'] > old_buy else 'falling'
            return result if result else {'available': False}
        except Exception as e:
            logger.error(f"Long/short ratio error {symbol}: {e}")
            return {'available': False}

    def get_funding_rate_history(self, symbol: str) -> Dict:
        """Get last 5 funding rates to assess sentiment trend."""
        try:
            resp = self.session.get_funding_rate_history(
                category="linear", symbol=symbol, limit=5
            )
            if resp.get('result', {}).get('list'):
                rates = [float(r['fundingRate']) for r in resp['result']['list']]
                avg = sum(rates) / len(rates)
                trend = 'positive' if avg > 0 else 'negative'
                # Check if rates are escalating
                direction = 'stable'
                if len(rates) >= 3:
                    if rates[0] > rates[-1] * 1.5:
                        direction = 'rising'
                    elif rates[0] < rates[-1] * 0.5:
                        direction = 'falling'
                return {
                    'available': True,
                    'rates': [round(r, 7) for r in rates],
                    'average': round(avg, 7),
                    'trend': trend,
                    'direction': direction,
                }
        except Exception as e:
            logger.error(f"Funding history error {symbol}: {e}")
        return {'available': False}

    def get_btc_context(self) -> Dict:
        """Get BTC market context (macro reference for altcoin analysis)."""
        try:
            ticker = self.session.get_tickers(category="linear", symbol="BTCUSDT")
            if ticker['result']['list']:
                info = ticker['result']['list'][0]
                price = float(info['lastPrice'])
                change_24h = float(info.get('price24hPcnt', 0)) * 100
                high_24h = float(info.get('highPrice24h', 0))
                low_24h = float(info.get('lowPrice24h', 0))
                # Simple trend: 24h position in range
                range_24h = high_24h - low_24h
                position_in_range = ((price - low_24h) / range_24h * 100) if range_24h > 0 else 50
                return {
                    'available': True,
                    'price': price,
                    'change_24h_pct': round(change_24h, 2),
                    'high_24h': high_24h,
                    'low_24h': low_24h,
                    'position_in_range_pct': round(position_in_range, 1),
                    'bias': 'bullish' if change_24h > 0.5 else ('bearish' if change_24h < -0.5 else 'neutral'),
                }
        except Exception as e:
            logger.error(f"BTC context error: {e}")
        return {'available': False}

    def get_btc_dominance(self) -> Dict:
        """Get BTC dominance from CoinGecko (free, no key required)."""
        try:
            resp = requests.get(
                'https://api.coingecko.com/api/v3/global',
                timeout=10,
                headers={'Accept': 'application/json'}
            )
            if resp.status_code == 200:
                data = resp.json().get('data', {})
                btc_dom = data.get('market_cap_percentage', {}).get('btc', 0)
                eth_dom = data.get('market_cap_percentage', {}).get('eth', 0)
                # High BTC dominance (>55%) = altcoins underperform
                # Low BTC dominance (<45%) = altcoin season
                alt_season = btc_dom < 48
                return {
                    'available': True,
                    'btc_dominance': round(btc_dom, 2),
                    'eth_dominance': round(eth_dom, 2),
                    'altcoin_season': alt_season,
                    'market_regime': 'altcoin_season' if alt_season else ('btc_season' if btc_dom > 55 else 'mixed'),
                }
        except Exception as e:
            logger.error(f"BTC dominance error: {e}")
        return {'available': False}

    def get_liquidations(self, symbol: str) -> Dict:
        """Get recent liquidation data from CoinGlass."""
        if not Config.COINGLASS_API_KEY:
            return {'available': False}
        try:
            headers = {'coinglassSecret': Config.COINGLASS_API_KEY}
            coin = symbol.replace('USDT', '')
            resp = requests.get(
                f'https://open-api.coinglass.com/public/v2/liquidation_history?symbol={coin}&time_type=h1',
                headers=headers, timeout=10
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get('data'):
                    return {
                        'available': True,
                        'long_liquidations': data['data'].get('longLiquidationUsd', 0),
                        'short_liquidations': data['data'].get('shortLiquidationUsd', 0),
                    }
        except Exception as e:
            logger.error(f"CoinGlass liquidations error: {e}")
        return {'available': False}

    def collect_all(self, symbol: str) -> Dict:
        """Orchestrate all data collection into a unified package for Gemini."""
        logger.info(f"Collecting data for {symbol}...")

        # Multi-timeframe klines with indicators
        klines = self.get_multi_timeframe_klines(symbol)

        # Extract indicator summaries for each timeframe
        indicator_summaries = {}
        for tf, df in klines.items():
            if not df.empty:
                indicator_summaries[tf] = self.indicator_engine.get_indicator_summary(df)

        # Order blocks from 4H
        order_blocks = []
        if not klines.get('4h', pd.DataFrame()).empty:
            order_blocks = self.indicator_engine.detect_order_blocks(klines['4h'])

        # Pivot Points from Daily klines
        pivot_points = {}
        if not klines.get('daily', pd.DataFrame()).empty:
            pivot_points = self.indicator_engine.calculate_pivot_points(klines['daily'])

        # TP level suggestions (anchored to real technical structure)
        tp_levels = {}
        df_4h = klines.get('4h', pd.DataFrame())
        if not df_4h.empty and 'atr' in df_4h.columns:
            atr_4h = float(df_4h['atr'].iloc[-1])
            tp_levels = self.indicator_engine.calculate_tp_levels(df_4h, pivot_points, atr_4h)

        # Real-time data
        orderbook = self.get_orderbook(symbol)
        funding = self.get_funding_rate(symbol)
        funding_history = self.get_funding_rate_history(symbol)
        ticker = self.get_ticker(symbol)
        fear_greed = self.get_fear_greed_index()
        oi = self.get_open_interest(symbol)
        liquidations = self.get_liquidations(symbol)
        long_short = self.get_long_short_ratio(symbol)
        btc_context = self.get_btc_context()
        btc_dominance = self.get_btc_dominance()

        package = {
            'symbol': symbol,
            'indicators': indicator_summaries,
            'order_blocks': order_blocks,
            'pivot_points': pivot_points,
            'tp_levels': tp_levels,
            'orderbook': orderbook,
            'funding_rate': funding,
            'funding_history': funding_history,
            'ticker': ticker,
            'fear_greed': fear_greed,
            'open_interest': oi,
            'liquidations': liquidations,
            'long_short_ratio': long_short,
            'btc_context': btc_context,
            'btc_dominance': btc_dominance,
        }

        logger.info(
            f"Data collected: {len(indicator_summaries)} timeframes, "
            f"{len(order_blocks)} OBs, "
            f"L/S={long_short.get('buy_ratio', 'N/A')}, "
            f"BTC={btc_context.get('bias', 'N/A')}, "
            f"BTCdom={btc_dominance.get('btc_dominance', 'N/A')}%"
        )
        return package
