"""
Universal Trading Analyzer
Supports all Yahoo Finance markets and tickers
"""
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import warnings
import asyncio
import time
from datetime import datetime

warnings.filterwarnings('ignore')

import ta
from config import CONFIG

class SimpleCache:
    """Simple in-memory cache for all tickers"""
    def __init__(self, ttl: int = 300, max_size: int = 100):
        self.ttl = ttl
        self.max_size = max_size
        self.cache = {}
        self.timestamps = {}
    
    def get(self, key: str):
        if key in self.cache:
            if time.time() - self.timestamps[key] < self.ttl:
                return self.cache[key]
            else:
                del self.cache[key]
                del self.timestamps[key]
        return None
    
    def set(self, key: str, value):
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.timestamps, key=self.timestamps.get)
            del self.cache[oldest_key]
            del self.timestamps[oldest_key]
        self.cache[key] = value
        self.timestamps[key] = time.time()

class TradingAnalyzer:
    """Universal analyzer for ALL Yahoo Finance tickers"""
    
    def __init__(self):
        self.config = CONFIG
        self._last_request_time = 0
        self.cache = SimpleCache(
            ttl=self.config.CACHE_TTL,
            max_size=self.config.MAX_CACHE_SIZE
        )
        
        print("✅ Universal Analyzer initialized")
        print("🌍 Supports all markets and tickers")
        print(f"⏱️ Delay: {self.config.YAHOO_DELAY_SECONDS}s")
    
    def _rate_limit(self):
        """Rate limiting for Yahoo Finance"""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        
        if time_since_last < self.config.YAHOO_DELAY_SECONDS:
            sleep_time = self.config.YAHOO_DELAY_SECONDS - time_since_last
            time.sleep(sleep_time)
        
        self._last_request_time = time.time()
    
    async def analyze_ticker(self, ticker_symbol: str, period: str = '1y') -> Dict:
        """Universal analysis method for any ticker"""
        try:
            print(f"🔍 Analyzing {ticker_symbol} ({period})")
            
            # Map period
            period_map = {
                '3m': '3mo', '3months': '3mo', '3month': '3mo',
                '6m': '6mo', '6months': '6mo', '6month': '6mo',
                '1y': '1y', '1year': '1y',
                '2y': '2y', '2years': '2y',
                '3y': '3y', '3years': '3y',
                '5y': '5y', '5years': '5y',
                'max': 'max', 'maximum': 'max'
            }
            yf_period = period_map.get(period.lower(), '1y')
            
            # Format ticker
            formatted_ticker = self._format_ticker_for_yfinance(ticker_symbol)
            print(f"Formatted: {ticker_symbol} -> {formatted_ticker}")
            
            # Get exchange info
            exchange_info = self.config.get_exchange_info(formatted_ticker)
            print(f"Exchange: {exchange_info['exchange']}")
            
            # Check cache
            cache_key = f"{formatted_ticker}_{yf_period}"
            cached_data = self.cache.get(cache_key)
            
            if cached_data is not None:
                print("✅ Using cached data")
                data = cached_data
            else:
                # Fetch data with multiple attempts
                data = await self._fetch_data_universal(formatted_ticker, yf_period, exchange_info)
                
                if data is None or data.empty:
                    return {
                        'success': False,
                        'error': f"No data found for {ticker_symbol}. Check if ticker exists on Yahoo Finance."
                    }
                
                # Cache it
                self.cache.set(cache_key, data)
                print(f"✅ Data fetched: {len(data)} rows")
            
            # Clean data before indicator calculation
            data = self._clean_data(data)
            
            # Calculate technical indicators
            data = self._calculate_complete_indicators(data)
            
            # Get ticker info
            info = await self._get_ticker_info(formatted_ticker, data, ticker_symbol)
            
            # Generate all signals
            signals = self._generate_all_signals(data)
            
            # Detect reversal patterns
            reversal_patterns = self._detect_reversal_patterns(data)
            
            # Detect divergences
            divergences = self._detect_divergences(data)
            
            # Calculate performance statistics
            stats = self._calculate_statistics(data)
            
            # Prepare summary
            summary = self._create_comprehensive_summary(
                ticker_symbol, data, info, signals, 
                reversal_patterns, divergences, stats, exchange_info
            )
            compact = self._create_compact_summary(ticker_symbol, data, info, signals, divergences)
            
            # Debug: Print signal counts
            bull_count = len([s for s in signals if s['direction'] == 'BULLISH'])
            bear_count = len([s for s in signals if s['direction'] == 'BEARISH'])
            print(f"✅ Analysis complete for {ticker_symbol}")
            print(f"📊 Signals: {bull_count} Bullish, {bear_count} Bearish, {len(signals)} Total")
            
            return {
                'success': True,
                'data': data,
                'info': info,
                'signals': signals,
                'reversal_patterns': reversal_patterns,
                'divergences': divergences,
                'stats': stats,
                'summary': summary,
                'compact_summary': compact,
                'period': period,
                'exchange': exchange_info['exchange']
            }
            
        except Exception as e:
            print(f"❌ Error analyzing {ticker_symbol}: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': f"Analysis failed: {str(e)[:150]}"
            }
    
    def _format_ticker_for_yfinance(self, ticker: str) -> str:
        """Format ticker for yfinance - universal"""
        ticker = ticker.upper().strip().replace('$', '')
        
        # Common mappings for indices and commodities
        special_mapping = {
            # US Indices
            'SPX': '^GSPC',
            'SPY': 'SPY',
            'DJI': '^DJI',
            'DIA': 'DIA',
            'IXIC': '^IXIC',
            'QQQ': 'QQQ',
            'VIX': '^VIX',
            
            # European Indices
            'DAX': '^GDAXI',
            'CAC': '^FCHI',
            'FTSE': '^FTSE',
            'IBEX': '^IBEX',
            'STOXX50': '^STOXX50E',
            
            # Asian Indices
            'N225': '^N225',
            'HSI': '^HSI',
            'SSEC': '000001.SS',
            'SENSEX': '^BSESN',
            
            # Commodities
            'GOLD': 'GC=F',
            'SILVER': 'SI=F',
            'OIL': 'CL=F',
            'BRENT': 'BZ=F',
            'NATGAS': 'NG=F',
            'COPPER': 'HG=F',
            
            # Currency pairs
            'EURUSD': 'EURUSD=X',
            'GBPUSD': 'GBPUSD=X',
            'USDJPY': 'JPY=X',
            'USDCHF': 'CHF=X',
            'USDCAD': 'CAD=X',
            'AUDUSD': 'AUDUSD=X',
            
            # Crypto
            'BTC': 'BTC-USD',
            'ETH': 'ETH-USD',
            'XRP': 'XRP-USD',
            'DOGE': 'DOGE-USD',
            'ADA': 'ADA-USD',
            'SOL': 'SOL-USD',
            
            # Popular ETFs
            'VOO': 'VOO',
            'VTI': 'VTI',
            'BND': 'BND',
            'GLD': 'GLD',
            'SLV': 'SLV',
            'IWM': 'IWM',
            
            # Futures
            'ES': 'ES=F',
            'NQ': 'NQ=F',
            'YM': 'YM=F',
            'RTY': 'RTY=F',
        }
        
        # Return mapped ticker or original
        return special_mapping.get(ticker, ticker)
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare data for analysis"""
        if data.empty:
            return data
        
        df = data.copy()
        
        # Ensure single-level columns (handle multi-index from yfinance)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Rename columns if needed
        if 'Adj Close' in df.columns:
            df = df.rename(columns={'Adj Close': 'Close'})
        
        # Ensure we have required columns
        required_columns = ['Close', 'High', 'Low', 'Open', 'Volume']
        for col in required_columns:
            if col not in df.columns:
                if col == 'Volume':
                    df[col] = 0
                elif col == 'Close':
                    df[col] = df.iloc[:, -1] if len(df.columns) > 0 else 0
        
        # Ensure all data is numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop rows with NaN in critical columns
        df = df.dropna(subset=['Close', 'High', 'Low', 'Open'])
        
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        return df
    
    async def _fetch_data_universal(self, ticker: str, period: str, exchange_info: Dict) -> Optional[pd.DataFrame]:
        """Universal data fetching for any ticker"""
        attempts = [
            lambda: self._fetch_direct(ticker, period),
            lambda: self._fetch_ticker_object(ticker, period),
            lambda: self._fetch_with_alternatives(ticker, period, exchange_info),
        ]
        
        for i, fetch_method in enumerate(attempts):
            if i > 0:
                wait_time = self.config.YAHOO_DELAY_SECONDS * i
                print(f"⏳ Waiting {wait_time}s before attempt {i+1}...")
                await asyncio.sleep(wait_time)
            
            try:
                print(f"📥 Attempt {i+1}/{len(attempts)}: {fetch_method.__name__}")
                self._rate_limit()
                
                data = fetch_method()
                
                if data is not None and not data.empty:
                    print(f"✅ Success with {fetch_method.__name__}: {len(data)} rows")
                    return data
                
            except Exception as e:
                error_msg = str(e)
                print(f"⚠️ Attempt {i+1} failed: {error_msg[:100]}")
                
                if "429" in error_msg or "rate limit" in error_msg.lower():
                    wait = self.config.YAHOO_DELAY_SECONDS * 3
                    print(f"⏸️ Rate limited, waiting {wait}s...")
                    await asyncio.sleep(wait)
        
        print(f"❌ All fetch attempts failed for {ticker}")
        return None
    
    def _fetch_direct(self, ticker: str, period: str) -> Optional[pd.DataFrame]:
        """Direct download using yf.download"""
        try:
            data = yf.download(
                tickers=ticker,
                period=period,
                interval="1d",
                progress=False,
                threads=False,
                timeout=self.config.YAHOO_TIMEOUT,
                auto_adjust=True,
                prepost=False
            )
            return data if not data.empty else None
        except Exception as e:
            print(f"Direct download error: {e}")
            return None
    
    def _fetch_ticker_object(self, ticker: str, period: str) -> Optional[pd.DataFrame]:
        """Fetch using yf.Ticker object"""
        try:
            ticker_obj = yf.Ticker(ticker)
            data = ticker_obj.history(
                period=period,
                interval="1d",
                timeout=self.config.YAHOO_TIMEOUT
            )
            return data if not data.empty else None
        except Exception as e:
            print(f"Ticker object error: {e}")
            return None
    
    def _fetch_with_alternatives(self, ticker: str, period: str, exchange_info: Dict) -> Optional[pd.DataFrame]:
        """Try alternative ticker formats for international stocks"""
        if exchange_info['suffix']:
            base_ticker = exchange_info['base_ticker']
            print(f"Trying US format: {base_ticker}")
            data = self._fetch_direct(base_ticker, period)
            if data is not None and not data.empty:
                return data
        
        # Try common European suffixes
        european_suffixes = ['.MI', '.PA', '.DE', '.AS', '.BR', '.IR', '.MC', '.SW', '.L']
        
        for suffix in european_suffixes:
            if not ticker.endswith(suffix):
                alternative = f"{ticker}{suffix}"
                print(f"Trying alternative: {alternative}")
                data = self._fetch_direct(alternative, period)
                if data is not None and not data.empty:
                    return data
        
        return None
    
    async def _get_ticker_info(self, ticker: str, data: pd.DataFrame, original_ticker: str) -> Dict:
        """Get ticker information"""
        try:
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.info
            
            if not info or 'symbol' not in info:
                info = {
                    'symbol': original_ticker,
                    'longName': original_ticker,
                    'shortName': original_ticker,
                    'currency': self._detect_currency(ticker),
                    'marketCap': None,
                    'sector': None,
                    'industry': None,
                    'country': self._detect_country(ticker),
                }
            
            if 'regularMarketPrice' not in info and len(data) > 0:
                info['regularMarketPrice'] = float(data['Close'].iloc[-1])
            
            if 'currency' not in info:
                info['currency'] = self._detect_currency(ticker)
            
            if len(data) > 0:
                latest = data.iloc[-1]
                info['last_price'] = float(latest['Close'])
                info['last_volume'] = float(latest.get('Volume', 0))
                info['last_open'] = float(latest.get('Open', latest['Close']))
                info['last_high'] = float(latest.get('High', latest['Close']))
                info['last_low'] = float(latest.get('Low', latest['Close']))
            
            return info
            
        except Exception as e:
            print(f"Warning: Could not get ticker info: {e}")
            return {
                'symbol': original_ticker,
                'longName': original_ticker,
                'shortName': original_ticker,
                'currency': self._detect_currency(ticker),
                'last_price': float(data['Close'].iloc[-1]) if len(data) > 0 else 0,
                'country': self._detect_country(ticker),
            }
    
    def _detect_currency(self, ticker: str) -> str:
        """Detect currency based on ticker"""
        ticker_upper = ticker.upper()
        
        currency_map = {
            '.MI': 'EUR', '.PA': 'EUR', '.DE': 'EUR', '.AS': 'EUR',
            '.BR': 'EUR', '.IR': 'EUR', '.MC': 'EUR', '.SW': 'CHF',
            '.L': 'GBP', '.T': 'JPY', '.HK': 'HKD', '.SS': 'CNY',
            '.SZ': 'CNY', '.KS': 'KRW', '.AX': 'AUD', '.V': 'CAD',
            '.TO': 'CAD', '.CN': 'CAD', '.OL': 'NOK', '.ST': 'SEK',
            '.CO': 'DKK', '.HE': 'EUR', '.VI': 'EUR',
        }
        
        for suffix, currency in currency_map.items():
            if ticker_upper.endswith(suffix):
                return currency
        
        return 'USD'
    
    def _detect_country(self, ticker: str) -> str:
        """Detect country based on ticker"""
        ticker_upper = ticker.upper()
        
        country_map = {
            '.MI': 'Italy', '.PA': 'France', '.DE': 'Germany',
            '.AS': 'Netherlands', '.BR': 'Belgium', '.IR': 'Ireland',
            '.MC': 'Spain', '.SW': 'Switzerland', '.L': 'UK',
            '.T': 'Japan', '.HK': 'Hong Kong', '.SS': 'China',
            '.SZ': 'China', '.KS': 'South Korea', '.AX': 'Australia',
            '.V': 'Canada', '.TO': 'Canada', '.CN': 'Canada',
            '.OL': 'Norway', '.ST': 'Sweden', '.CO': 'Denmark',
            '.HE': 'Finland', '.VI': 'Austria',
        }
        
        for suffix, country in country_map.items():
            if ticker_upper.endswith(suffix):
                return country
        
        return 'USA'
    
    def _calculate_complete_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate ALL technical indicators safely"""
        df = data.copy()
        
        if len(df) < 10:
            return df
        
        try:
            # Ensure all columns are 1D Series
            for col in ['Close', 'High', 'Low', 'Open', 'Volume']:
                if col in df.columns:
                    # Convert to numpy array and ensure 1D
                    values = df[col].values
                    if values.ndim > 1:
                        values = values.flatten()
                    # Create new Series with proper index
                    df[col] = pd.Series(values, index=df.index)
            
            # Ensure numeric
            for col in ['Close', 'High', 'Low', 'Open', 'Volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Debug: Print data info
            print(f"📊 Data shape: {df.shape}")
            print(f"📊 Columns: {list(df.columns[:10])}")
            
            # Price transformations
            df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
            df['Weighted_Close'] = (df['Close'] * 2 + df['High'] + df['Low']) / 4
            
            # Moving Averages
            for period in [5, 9, 20, 50, 100, 200]:
                if len(df) >= period:
                    df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
                    df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
                    # Also calculate for typical price
                    df[f'TP_SMA_{period}'] = df['Typical_Price'].rolling(window=period).mean()
            
            # MACD
            if len(df) >= 26:
                try:
                    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
                    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
                    df['MACD'] = exp1 - exp2
                    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
                    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
                    df['MACD_Hist_Color'] = np.where(df['MACD_Hist'] >= 0, 'green', 'red')
                    print(f"✅ MACD calculated: {df['MACD'].iloc[-1]:.4f}")
                except Exception as e:
                    print(f"MACD calculation error: {e}")
                    df['MACD'] = 0
                    df['MACD_Signal'] = 0
                    df['MACD_Hist'] = 0
                    df['MACD_Hist_Color'] = 'gray'
            
            # RSI
            if len(df) >= 14:
                try:
                    delta = df['Close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    df['RSI'] = 100 - (100 / (1 + rs))
                    df['RSI'] = df['RSI'].fillna(50)
                    print(f"✅ RSI calculated: {df['RSI'].iloc[-1]:.1f}")
                except Exception as e:
                    print(f"RSI calculation error: {e}")
                    df['RSI'] = 50
            
            # Stochastic
            if len(df) >= 14:
                try:
                    low_14 = df['Low'].rolling(window=14).min()
                    high_14 = df['High'].rolling(window=14).max()
                    df['Stoch_%K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
                    df['Stoch_%D'] = df['Stoch_%K'].rolling(window=3).mean()
                    df['Stoch_%K'] = df['Stoch_%K'].fillna(50)
                    df['Stoch_%D'] = df['Stoch_%D'].fillna(50)
                    print(f"✅ Stochastic calculated: K={df['Stoch_%K'].iloc[-1]:.1f}, D={df['Stoch_%D'].iloc[-1]:.1f}")
                except Exception as e:
                    print(f"Stochastic calculation error: {e}")
                    df['Stoch_%K'] = 50
                    df['Stoch_%D'] = 50
            
            # Williams %R
            if len(df) >= 14:
                try:
                    highest_high = df['High'].rolling(window=14).max()
                    lowest_low = df['Low'].rolling(window=14).min()
                    df['Williams_%R'] = -100 * ((highest_high - df['Close']) / (highest_high - lowest_low))
                    df['Williams_%R'] = df['Williams_%R'].fillna(-50)
                    print(f"✅ Williams %R calculated: {df['Williams_%R'].iloc[-1]:.1f}")
                except Exception as e:
                    print(f"Williams %R calculation error: {e}")
                    df['Williams_%R'] = -50
            
            # ROC
            for period in [12, 14, 26]:
                if len(df) >= period:
                    df[f'ROC_{period}'] = ((df['Close'] - df['Close'].shift(period)) / df['Close'].shift(period)) * 100
                    df[f'ROC_{period}'] = df[f'ROC_{period}'].fillna(0)
            
            # CCI
            if len(df) >= 20:
                try:
                    typical_price = df['Typical_Price']
                    sma = typical_price.rolling(window=20).mean()
                    mad = typical_price.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
                    df['CCI'] = (typical_price - sma) / (0.015 * mad)
                    df['CCI'] = df['CCI'].fillna(0)
                except Exception as e:
                    print(f"CCI calculation error: {e}")
                    df['CCI'] = 0
            
            # Money Flow Index
            if len(df) >= 14 and 'Volume' in df.columns:
                try:
                    typical_price = df['Typical_Price']
                    money_flow = typical_price * df['Volume']
                    
                    # Positive and negative money flow
                    positive_flow = np.where(typical_price > typical_price.shift(1), money_flow, 0)
                    negative_flow = np.where(typical_price < typical_price.shift(1), money_flow, 0)
                    
                    positive_mf = pd.Series(positive_flow, index=df.index).rolling(window=14).sum()
                    negative_mf = pd.Series(negative_flow, index=df.index).rolling(window=14).sum()
                    
                    money_ratio = positive_mf / negative_mf
                    df['MFI'] = 100 - (100 / (1 + money_ratio))
                    df['MFI'] = df['MFI'].fillna(50)
                except Exception as e:
                    print(f"MFI calculation error: {e}")
                    df['MFI'] = 50
            
            # Volume indicators
            if 'Volume' in df.columns:
                try:
                    # OBV
                    obv = np.zeros(len(df))
                    for i in range(1, len(df)):
                        if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                            obv[i] = obv[i-1] + df['Volume'].iloc[i]
                        elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                            obv[i] = obv[i-1] - df['Volume'].iloc[i]
                        else:
                            obv[i] = obv[i-1]
                    df['OBV'] = obv
                    
                    # A/D Line (Accumulation/Distribution)
                    clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
                    clv = clv.fillna(0)
                    df['AD_Line'] = (clv * df['Volume']).cumsum()
                    
                    # Volume MA and ratio
                    df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
                    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_20'].replace(0, 1)
                    df['Volume_Ratio'] = df['Volume_Ratio'].fillna(1)
                    
                    # Volume Price Trend
                    df['VPT'] = df['Volume'] * ((df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1))
                    df['VPT'] = df['VPT'].fillna(0).cumsum()
                    
                    print(f"✅ Volume indicators calculated: OBV={df['OBV'].iloc[-1]:.0f}, AD={df['AD_Line'].iloc[-1]:.0f}")
                except Exception as e:
                    print(f"Volume indicators error: {e}")
                    df['OBV'] = 0
                    df['AD_Line'] = 0
                    df['VPT'] = 0
                    df['Volume_MA_20'] = df['Volume']
                    df['Volume_Ratio'] = 1
            
            # Bollinger Bands
            if len(df) >= 20:
                try:
                    df['BB_Middle'] = df['Close'].rolling(20).mean()
                    bb_std = df['Close'].rolling(20).std()
                    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
                    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
                    df['BB_Width'] = ((df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']) * 100
                    df['BB_%B'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
                    
                    # Handle division by zero
                    df['BB_%B'] = df['BB_%B'].fillna(0.5)
                    df['BB_Width'] = df['BB_Width'].fillna(0)
                except Exception as e:
                    print(f"Bollinger Bands error: {e}")
                    df['BB_Middle'] = df['Close']
                    df['BB_Upper'] = df['Close']
                    df['BB_Lower'] = df['Close']
                    df['BB_Width'] = 0
                    df['BB_%B'] = 0.5
            
            # ATR
            if len(df) >= 14:
                try:
                    high_low = df['High'] - df['Low']
                    high_close = np.abs(df['High'] - df['Close'].shift())
                    low_close = np.abs(df['Low'] - df['Close'].shift())
                    
                    ranges = pd.DataFrame({
                        'high_low': high_low,
                        'high_close': high_close,
                        'low_close': low_close
                    })
                    
                    true_range = ranges.max(axis=1)
                    df['ATR'] = true_range.rolling(window=14).mean()
                    df['ATR_Pct'] = (df['ATR'] / df['Close']) * 100
                    df['ATR_Pct'] = df['ATR_Pct'].fillna(1)
                except Exception as e:
                    print(f"ATR calculation error: {e}")
                    df['ATR'] = 0
                    df['ATR_Pct'] = 1
            
            # Performance metrics
            df['Daily_Return'] = df['Close'].pct_change().fillna(0)
            df['Cumulative_Return'] = (1 + df['Daily_Return']).cumprod() - 1
            
            df['Volatility_20d'] = df['Daily_Return'].rolling(20).std() * np.sqrt(252)
            df['Volatility_60d'] = df['Daily_Return'].rolling(60).std() * np.sqrt(252)
            
            # Parabolic SAR (simplified)
            if len(df) >= 10:
                try:
                    df['SAR'] = df['Close'].rolling(10).mean()
                except:
                    df['SAR'] = df['Close']
            
            # Fill NaN values
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            # Debug: Print some indicator values
            if len(df) > 0:
                last_row = df.iloc[-1]
                print(f"📊 Last row indicators:")
                print(f"  Close: {last_row.get('Close', 0):.2f}")
                print(f"  RSI: {last_row.get('RSI', 0):.1f}")
                print(f"  MACD: {last_row.get('MACD', 0):.4f}")
                print(f"  Volume Ratio: {last_row.get('Volume_Ratio', 0):.2f}")
                print(f"  A/D Line: {last_row.get('AD_Line', 0):.0f}")
            
            indicator_count = len([col for col in df.columns if 'Unnamed' not in str(col)])
            print(f"✅ Calculated {indicator_count} indicators")
            
        except Exception as e:
            print(f"⚠️ Indicator calculation error: {e}")
            import traceback
            traceback.print_exc()
        
        return df
    
    def _generate_all_signals(self, data: pd.DataFrame) -> List[Dict]:
        """Generate signals from all indicators"""
        signals = []
        
        if len(data) < 10:
            print("⚠️ Not enough data for signals")
            return signals
        
        try:
            # Get scalar values from last row
            last_idx = len(data) - 1
            prev_idx = last_idx - 1 if last_idx > 0 else last_idx
            
            # Helper function to safely get scalar values
            def get_scalar(column, idx):
                try:
                    val = data[column].iloc[idx]
                    if pd.isna(val):
                        return None
                    return float(val)
                except:
                    return None
            
            # Get current and previous values
            close = get_scalar('Close', last_idx)
            
            # === TREND SIGNALS ===
            # Golden/Death Cross
            sma50 = get_scalar('SMA_50', last_idx)
            sma200 = get_scalar('SMA_200', last_idx)
            prev_sma50 = get_scalar('SMA_50', prev_idx)
            prev_sma200 = get_scalar('SMA_200', prev_idx)
            
            if all(v is not None for v in [sma50, sma200, prev_sma50, prev_sma200]):
                if prev_sma50 <= prev_sma200 and sma50 > sma200:
                    signals.append({'type': 'GOLDEN_CROSS', 'strength': 'STRONG', 'direction': 'BULLISH'})
                    print("✅ Signal: GOLDEN_CROSS")
                elif prev_sma50 >= prev_sma200 and sma50 < sma200:
                    signals.append({'type': 'DEATH_CROSS', 'strength': 'STRONG', 'direction': 'BEARISH'})
                    print("✅ Signal: DEATH_CROSS")
            
            # Price vs Moving Averages
            if close is not None:
                for period in [20, 50, 200]:
                    sma = get_scalar(f'SMA_{period}', last_idx)
                    ema = get_scalar(f'EMA_{period}', last_idx)
                    if sma is not None:
                        if close > sma:
                            signals.append({'type': f'ABOVE_{period}SMA', 'strength': 'MODERATE', 'direction': 'BULLISH'})
                        else:
                            signals.append({'type': f'BELOW_{period}SMA', 'strength': 'MODERATE', 'direction': 'BEARISH'})
                    if ema is not None:
                        if close > ema:
                            signals.append({'type': f'ABOVE_{period}EMA', 'strength': 'MODERATE', 'direction': 'BULLISH'})
                        else:
                            signals.append({'type': f'BELOW_{period}EMA', 'strength': 'MODERATE', 'direction': 'BEARISH'})
            
            # === MOMENTUM SIGNALS ===
            # RSI
            rsi = get_scalar('RSI', last_idx)
            if rsi is not None:
                if rsi < 30:
                    signals.append({'type': 'RSI_OVERSOLD', 'strength': 'STRONG', 'direction': 'BULLISH'})
                    print("✅ Signal: RSI_OVERSOLD")
                elif rsi > 70:
                    signals.append({'type': 'RSI_OVERBOUGHT', 'strength': 'STRONG', 'direction': 'BEARISH'})
                    print("✅ Signal: RSI_OVERBOUGHT")
                elif rsi < 35:
                    signals.append({'type': 'RSI_NEAR_OVERSOLD', 'strength': 'MODERATE', 'direction': 'BULLISH'})
                elif rsi > 65:
                    signals.append({'type': 'RSI_NEAR_OVERBOUGHT', 'strength': 'MODERATE', 'direction': 'BEARISH'})
            
            # MACD
            macd = get_scalar('MACD', last_idx)
            signal = get_scalar('MACD_Signal', last_idx)
            prev_macd = get_scalar('MACD', prev_idx)
            prev_signal = get_scalar('MACD_Signal', prev_idx)
            
            if all(v is not None for v in [macd, signal, prev_macd, prev_signal]):
                if prev_macd <= prev_signal and macd > signal:
                    signals.append({'type': 'MACD_BULLISH_CROSS', 'strength': 'MODERATE', 'direction': 'BULLISH'})
                    print("✅ Signal: MACD_BULLISH_CROSS")
                elif prev_macd >= prev_signal and macd < signal:
                    signals.append({'type': 'MACD_BEARISH_CROSS', 'strength': 'MODERATE', 'direction': 'BEARISH'})
                    print("✅ Signal: MACD_BEARISH_CROSS")
                
                if macd > signal:
                    signals.append({'type': 'MACD_ABOVE_SIGNAL', 'strength': 'MODERATE', 'direction': 'BULLISH'})
                else:
                    signals.append({'type': 'MACD_BELOW_SIGNAL', 'strength': 'MODERATE', 'direction': 'BEARISH'})
            
            # Stochastic
            stoch_k = get_scalar('Stoch_%K', last_idx)
            stoch_d = get_scalar('Stoch_%D', last_idx)
            if stoch_k is not None and stoch_d is not None:
                if stoch_k < 20 and stoch_d < 20:
                    signals.append({'type': 'STOCH_OVERSOLD', 'strength': 'MODERATE', 'direction': 'BULLISH'})
                    print("✅ Signal: STOCH_OVERSOLD")
                elif stoch_k > 80 and stoch_d > 80:
                    signals.append({'type': 'STOCH_OVERBOUGHT', 'strength': 'MODERATE', 'direction': 'BEARISH'})
                    print("✅ Signal: STOCH_OVERBOUGHT")
            
            # Williams %R
            williams = get_scalar('Williams_%R', last_idx)
            if williams is not None:
                if williams < -80:
                    signals.append({'type': 'WILLIAMS_OVERSOLD', 'strength': 'MODERATE', 'direction': 'BULLISH'})
                    print("✅ Signal: WILLIAMS_OVERSOLD")
                elif williams > -20:
                    signals.append({'type': 'WILLIAMS_OVERBOUGHT', 'strength': 'MODERATE', 'direction': 'BEARISH'})
                    print("✅ Signal: WILLIAMS_OVERBOUGHT")
            
            # CCI
            cci = get_scalar('CCI', last_idx)
            if cci is not None:
                if cci < -100:
                    signals.append({'type': 'CCI_OVERSOLD', 'strength': 'MODERATE', 'direction': 'BULLISH'})
                    print("✅ Signal: CCI_OVERSOLD")
                elif cci > 100:
                    signals.append({'type': 'CCI_OVERBOUGHT', 'strength': 'MODERATE', 'direction': 'BEARISH'})
                    print("✅ Signal: CCI_OVERBOUGHT")
            
            # MFI
            mfi = get_scalar('MFI', last_idx)
            if mfi is not None:
                if mfi < 20:
                    signals.append({'type': 'MFI_OVERSOLD', 'strength': 'MODERATE', 'direction': 'BULLISH'})
                elif mfi > 80:
                    signals.append({'type': 'MFI_OVERBOUGHT', 'strength': 'MODERATE', 'direction': 'BEARISH'})
            
            # === VOLUME SIGNALS ===
            vol_ratio = get_scalar('Volume_Ratio', last_idx)
            if vol_ratio is not None:
                if vol_ratio > 2.0:
                    signals.append({'type': 'HIGH_VOLUME', 'strength': 'STRONG', 'direction': 'NEUTRAL'})
                    print("✅ Signal: HIGH_VOLUME")
                elif vol_ratio > 1.5:
                    signals.append({'type': 'ABOVE_AVG_VOLUME', 'strength': 'MODERATE', 'direction': 'NEUTRAL'})
            
            # A/D Line trending
            ad_line = get_scalar('AD_Line', last_idx)
            if ad_line is not None:
                prev_ad = get_scalar('AD_Line', prev_idx)
                if prev_ad is not None and ad_line > prev_ad:
                    signals.append({'type': 'AD_BULLISH', 'strength': 'MODERATE', 'direction': 'BULLISH'})
                elif prev_ad is not None and ad_line < prev_ad:
                    signals.append({'type': 'AD_BEARISH', 'strength': 'MODERATE', 'direction': 'BEARISH'})
            
            # === VOLATILITY SIGNALS ===
            bb_percent = get_scalar('BB_%B', last_idx)
            if bb_percent is not None:
                if bb_percent < 0.2:
                    signals.append({'type': 'BB_OVERSOLD', 'strength': 'STRONG', 'direction': 'BULLISH'})
                    print("✅ Signal: BB_OVERSOLD")
                elif bb_percent > 0.8:
                    signals.append({'type': 'BB_OVERBOUGHT', 'strength': 'STRONG', 'direction': 'BEARISH'})
                    print("✅ Signal: BB_OVERBOUGHT")
            
            bb_width = get_scalar('BB_Width', last_idx)
            if bb_width is not None:
                if bb_width < 5:
                    signals.append({'type': 'BB_SQUEEZE', 'strength': 'MODERATE', 'direction': 'NEUTRAL'})
                    print("✅ Signal: BB_SQUEEZE")
            
            atr_pct = get_scalar('ATR_Pct', last_idx)
            if atr_pct is not None:
                if atr_pct > 3:
                    signals.append({'type': 'HIGH_VOLATILITY', 'strength': 'MODERATE', 'direction': 'NEUTRAL'})
                    print("✅ Signal: HIGH_VOLATILITY")
            
            print(f"✅ Generated {len(signals)} signals")
            
        except Exception as e:
            print(f"⚠️ Signal generation error: {e}")
            import traceback
            traceback.print_exc()
        
        return signals
    
    def _detect_reversal_patterns(self, data: pd.DataFrame) -> Dict:
        """Detect reversal patterns"""
        patterns = {
            'bullish_reversal': False,
            'bearish_reversal': False,
            'confidence': 0,
            'signals': [],
            'details': {}
        }
        
        if len(data) < 20:
            return patterns
        
        try:
            # Get scalar values
            last_idx = len(data) - 1
            
            # Helper function
            def get_scalar(column, idx):
                try:
                    val = data[column].iloc[idx]
                    if pd.isna(val):
                        return None
                    return float(val)
                except:
                    return None
            
            # === BULLISH REVERSAL PATTERNS ===
            bullish_count = 0
            bullish_details = []
            
            # RSI oversold
            rsi = get_scalar('RSI', last_idx)
            if rsi is not None and rsi < 30:
                bullish_count += 1
                bullish_details.append(f"RSI oversold ({rsi:.1f})")
            
            # Stochastic oversold
            stoch_k = get_scalar('Stoch_%K', last_idx)
            stoch_d = get_scalar('Stoch_%D', last_idx)
            if stoch_k is not None and stoch_d is not None and stoch_k < 20 and stoch_d < 20:
                bullish_count += 1
                bullish_details.append(f"Stochastic oversold")
            
            # Bollinger Bands oversold
            bb_percent = get_scalar('BB_%B', last_idx)
            if bb_percent is not None and bb_percent < 0.2:
                bullish_count += 1
                bullish_details.append(f"BB position: {bb_percent*100:.1f}% (oversold)")
            
            # Williams %R oversold
            williams = get_scalar('Williams_%R', last_idx)
            if williams is not None and williams < -80:
                bullish_count += 1
                bullish_details.append(f"Williams %R oversold ({williams:.1f})")
            
            # MFI oversold
            mfi = get_scalar('MFI', last_idx)
            if mfi is not None and mfi < 20:
                bullish_count += 1
                bullish_details.append(f"MFI oversold ({mfi:.1f})")
            
            # Determine bullish reversal
            if bullish_count >= 2:
                patterns['bullish_reversal'] = True
                patterns['confidence'] = min(95, bullish_count * 25)
                patterns['signals'] = bullish_details
                patterns['details']['bullish_count'] = bullish_count
                print(f"✅ Bullish reversal detected ({bullish_count} signals)")
            
            # === BEARISH REVERSAL PATTERNS ===
            bearish_count = 0
            bearish_details = []
            
            # RSI overbought
            if rsi is not None and rsi > 70:
                bearish_count += 1
                bearish_details.append(f"RSI overbought ({rsi:.1f})")
            
            # Stochastic overbought
            if stoch_k is not None and stoch_d is not None and stoch_k > 80 and stoch_d > 80:
                bearish_count += 1
                bearish_details.append(f"Stochastic overbought")
            
            # Bollinger Bands overbought
            if bb_percent is not None and bb_percent > 0.8:
                bearish_count += 1
                bearish_details.append(f"BB position: {bb_percent*100:.1f}% (overbought)")
            
            # Williams %R overbought
            if williams is not None and williams > -20:
                bearish_count += 1
                bearish_details.append(f"Williams %R overbought ({williams:.1f})")
            
            # MFI overbought
            if mfi is not None and mfi > 80:
                bearish_count += 1
                bearish_details.append(f"MFI overbought ({mfi:.1f})")
            
            # Determine bearish reversal
            if bearish_count >= 2:
                patterns['bearish_reversal'] = True
                patterns['confidence'] = max(patterns['confidence'], min(95, bearish_count * 25))
                patterns['signals'].extend(bearish_details)
                patterns['details']['bearish_count'] = bearish_count
                print(f"✅ Bearish reversal detected ({bearish_count} signals)")
            
        except Exception as e:
            print(f"⚠️ Reversal pattern error: {e}")
            import traceback
            traceback.print_exc()
        
        return patterns
    
    def _detect_divergences(self, data: pd.DataFrame, lookback: int = 30) -> Dict:
        """Detect divergences between price and indicators"""
        divergences = {
            'bullish_rsi': False,
            'bearish_rsi': False,
            'bullish_macd': False,
            'bearish_macd': False,
            'bullish_ad': False,
            'bearish_ad': False,
            'bullish_obv': False,
            'bearish_obv': False,
            'details': []
        }
        
        if len(data) < lookback + 5:
            return divergences
        
        try:
            recent = data.iloc[-lookback:].copy()
            
            # Get arrays
            price = recent['Close'].values
            rsi = recent['RSI'].values if 'RSI' in recent.columns else None
            macd = recent['MACD'].values if 'MACD' in recent.columns else None
            ad_line = recent['AD_Line'].values if 'AD_Line' in recent.columns else None
            obv = recent['OBV'].values if 'OBV' in recent.columns else None
            
            # Simple divergence detection (last 20 periods)
            if rsi is not None and len(price) >= 20:
                # Find peaks and troughs
                price_slice = price[-20:]
                rsi_slice = rsi[-20:]
                
                price_min_idx = np.argmin(price_slice) + len(price) - 20
                price_max_idx = np.argmax(price_slice) + len(price) - 20
                
                rsi_min_idx = np.argmin(rsi_slice) + len(rsi) - 20
                rsi_max_idx = np.argmax(rsi_slice) + len(rsi) - 20
                
                # Bullish divergence: price lower low, RSI higher low
                if price_min_idx > len(price) - 10 and rsi_min_idx > len(rsi) - 10:
                    if price_slice.min() < price_slice[:10].min() and rsi_slice.min() > rsi_slice[:10].min():
                        divergences['bullish_rsi'] = True
                        divergences['details'].append("RSI Bullish Divergence")
                        print("✅ RSI Bullish Divergence")
                
                # Bearish divergence: price higher high, RSI lower high
                if price_max_idx > len(price) - 10 and rsi_max_idx > len(rsi) - 10:
                    if price_slice.max() > price_slice[:10].max() and rsi_slice.max() < rsi_slice[:10].max():
                        divergences['bearish_rsi'] = True
                        divergences['details'].append("RSI Bearish Divergence")
                        print("✅ RSI Bearish Divergence")
            
            if ad_line is not None and len(price) >= 20:
                ad_slice = ad_line[-20:]
                price_slice = price[-20:]
                
                # Simple A/D divergence
                if ad_slice[-1] > ad_slice[-5] and price_slice[-1] < price_slice[-5]:
                    divergences['bullish_ad'] = True
                    divergences['details'].append("A/D Bullish Divergence")
                    print("✅ A/D Bullish Divergence")
                elif ad_slice[-1] < ad_slice[-5] and price_slice[-1] > price_slice[-5]:
                    divergences['bearish_ad'] = True
                    divergences['details'].append("A/D Bearish Divergence")
                    print("✅ A/D Bearish Divergence")
            
        except Exception as e:
            print(f"⚠️ Divergence detection error: {e}")
            import traceback
            traceback.print_exc()
        
        return divergences
    
    def _calculate_statistics(self, data: pd.DataFrame) -> Dict:
        """Calculate performance statistics"""
        if len(data) < 5:
            return {}
        
        try:
            # Ensure Close is 1D
            close_series = data['Close']
            if hasattr(close_series, 'values') and close_series.values.ndim > 1:
                close_series = pd.Series(close_series.values.flatten(), index=data.index)
            
            returns = close_series.pct_change().dropna()
            
            if len(returns) == 0:
                return {
                    'total_return': 0,
                    'avg_daily_return': 0,
                    'volatility': 0,
                    'sharpe_ratio': 0,
                    'max_drawdown': 0,
                    'win_rate': 0,
                    'best_day': 0,
                    'worst_day': 0,
                }
            
            # Calculate win rate
            win_count = (returns > 0).sum()
            if isinstance(win_count, pd.Series):
                win_count = win_count.iloc[0] if len(win_count) > 0 else 0
            
            stats = {
                'total_return': float(((close_series.iloc[-1] / close_series.iloc[0]) - 1) * 100) if len(close_series) > 0 else 0,
                'avg_daily_return': float(returns.mean() * 100) if len(returns) > 0 else 0,
                'volatility': float(returns.std() * np.sqrt(252) * 100) if len(returns) > 0 else 0,
                'sharpe_ratio': float(returns.mean() / returns.std() * np.sqrt(252)) if len(returns) > 0 and returns.std() != 0 else 0,
                'max_drawdown': float(self._calculate_max_drawdown(close_series)),
                'win_rate': float(win_count / len(returns) * 100) if len(returns) > 0 else 0,
                'best_day': float(returns.max() * 100) if len(returns) > 0 else 0,
                'worst_day': float(returns.min() * 100) if len(returns) > 0 else 0,
            }
            
            if len(data) >= 20:
                try:
                    monthly_returns = close_series.resample('M').last().pct_change().dropna()
                    stats['best_month'] = float(monthly_returns.max() * 100) if len(monthly_returns) > 0 else 0
                    stats['worst_month'] = float(monthly_returns.min() * 100) if len(monthly_returns) > 0 else 0
                except:
                    pass
            
            print(f"📈 Statistics calculated: {stats.get('total_return', 0):.1f}% return")
            return stats
        except Exception as e:
            print(f"Statistics calculation error: {e}")
            return {}
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown"""
        try:
            cumulative_returns = (1 + prices.pct_change()).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            return float(drawdown.min() * 100)
        except:
            return 0.0
    
    def _create_comprehensive_summary(self, ticker: str, data: pd.DataFrame, info: Dict, 
                                     signals: List[Dict], reversal_patterns: Dict,
                                     divergences: Dict, stats: Dict, exchange_info: Dict) -> str:
        """Create comprehensive analysis summary"""
        try:
            # Get scalar values
            latest_close = float(data['Close'].iloc[-1])
            if len(data) > 1:
                prev_close_val = float(data['Close'].iloc[-2])
            else:
                prev_close_val = latest_close
            
            # Calculate daily change safely
            if prev_close_val != 0:
                daily_change = ((latest_close - prev_close_val) / prev_close_val * 100)
            else:
                daily_change = 0
            
            # Get latest values as scalar
            latest_open = float(data['Open'].iloc[-1]) if 'Open' in data.columns else latest_close
            latest_high = float(data['High'].iloc[-1]) if 'High' in data.columns else latest_close
            latest_low = float(data['Low'].iloc[-1]) if 'Low' in data.columns else latest_close
            latest_volume = float(data['Volume'].iloc[-1]) if 'Volume' in data.columns else 0
            
            bull_signals = [s for s in signals if s['direction'] == 'BULLISH']
            bear_signals = [s for s in signals if s['direction'] == 'BEARISH']
            neutral_signals = [s for s in signals if s['direction'] == 'NEUTRAL']
            
            currency = info.get('currency', 'USD')
            currency_symbol = '$' if currency == 'USD' else '€' if currency == 'EUR' else '£' if currency == 'GBP' else f'{currency} '
            
            # Color codes for signals
            green_circle = "🟢"
            red_circle = "🔴"
            yellow_circle = "🟡"
            white_circle = "⚪"
            blue_circle = "🔵"
            
            summary = f"""
📊 {green_circle} COMPREHENSIVE TECHNICAL ANALYSIS: {ticker.upper()} {green_circle}

📈 MARKET INFORMATION
• Exchange: {exchange_info['exchange']}
• Currency: {currency}
• Country: {info.get('country', 'Unknown')}
• Sector: {info.get('sector', 'N/A')}
• Industry: {info.get('industry', 'N/A')}

💰 PRICE ACTION ({data.index[-1].strftime('%Y-%m-%d')})
• Current: {currency_symbol}{latest_close:.2f} ({daily_change:+.2f}%)
• Open: {currency_symbol}{latest_open:.2f}
• High: {currency_symbol}{latest_high:.2f}
• Low: {currency_symbol}{latest_low:.2f}
• Volume: {self._format_number(latest_volume)} shares

📊 PERFORMANCE STATISTICS ({len(data)} trading days)
• Total Return: {stats.get('total_return', 0):.2f}%
• Avg Daily Return: {stats.get('avg_daily_return', 0):.3f}%
• Win Rate: {stats.get('win_rate', 0):.1f}% days positive
• Volatility: {stats.get('volatility', 0):.1f}%
• Sharpe Ratio: {stats.get('sharpe_ratio', 0):.2f}
• Max Drawdown: {stats.get('max_drawdown', 0):.1f}%
• Best Day: {stats.get('best_day', 0):+.2f}%
• Worst Day: {stats.get('worst_day', 0):+.2f}%

{green_circle} KEY INDICATORS {green_circle}
"""
            
            # Helper function to safely get indicator values
            def get_indicator(col):
                try:
                    val = data[col].iloc[-1]
                    if pd.isna(val):
                        return None
                    return float(val)
                except:
                    return None
            
            # RSI
            rsi = get_indicator('RSI')
            if rsi is not None:
                if rsi < 30:
                    status = f"{red_circle} OVERSOLD"
                elif rsi > 70:
                    status = f"{green_circle} OVERBOUGHT"
                else:
                    status = f"{white_circle} NEUTRAL"
                summary += f"• {status} RSI (14): {rsi:.1f}\n"
            
            # MACD
            macd = get_indicator('MACD')
            signal = get_indicator('MACD_Signal')
            if macd is not None and signal is not None:
                if macd > signal:
                    trend = f"{green_circle} BULLISH"
                else:
                    trend = f"{red_circle} BEARISH"
                summary += f"• {trend} MACD: {macd:.4f} | Signal: {signal:.4f}\n"
            
            # Stochastic
            stoch_k = get_indicator('Stoch_%K')
            stoch_d = get_indicator('Stoch_%D')
            if stoch_k is not None and stoch_d is not None:
                if stoch_k < 20 and stoch_d < 20:
                    stoch_status = f"{red_circle} OVERSOLD"
                elif stoch_k > 80 and stoch_d > 80:
                    stoch_status = f"{green_circle} OVERBOUGHT"
                else:
                    stoch_status = f"{white_circle} NEUTRAL"
                summary += f"• {stoch_status} Stochastic: K={stoch_k:.1f}, D={stoch_d:.1f}\n"
            
            # Moving averages
            for period in [20, 50, 200]:
                sma = get_indicator(f'SMA_{period}')
                if sma is not None:
                    if latest_close > sma:
                        position = f"{green_circle} ABOVE"
                    else:
                        position = f"{red_circle} BELOW"
                    distance = abs(latest_close - sma) / sma * 100 if sma != 0 else 0
                    summary += f"• {position} {period}-day MA: {currency_symbol}{sma:.2f} ({distance:.1f}% away)\n"
            
            # Volume
            vol_ratio = get_indicator('Volume_Ratio')
            if vol_ratio is not None:
                if vol_ratio > 1.5:
                    vol_status = f"{green_circle} HIGH VOLUME"
                elif vol_ratio < 0.5:
                    vol_status = f"{red_circle} LOW VOLUME"
                else:
                    vol_status = f"{white_circle} NORMAL VOLUME"
                summary += f"• {vol_status} ({vol_ratio:.1f}x average)\n"
            
            # A/D Line
            ad_line = get_indicator('AD_Line')
            if ad_line is not None:
                prev_ad = get_indicator('AD_Line') if len(data) > 1 else None
                if prev_ad is not None and ad_line > prev_ad:
                    ad_status = f"{green_circle} BULLISH"
                elif prev_ad is not None and ad_line < prev_ad:
                    ad_status = f"{red_circle} BEARISH"
                else:
                    ad_status = f"{white_circle} NEUTRAL"
                summary += f"• {ad_status} A/D Line: {ad_line:.0f}\n"
            
            # Bollinger Bands
            bb_percent = get_indicator('BB_%B')
            if bb_percent is not None:
                if bb_percent < 0.2:
                    bb_status = f"{red_circle} NEAR LOWER BAND"
                elif bb_percent > 0.8:
                    bb_status = f"{green_circle} NEAR UPPER BAND"
                else:
                    bb_status = f"{white_circle} MID RANGE"
                summary += f"• {bb_status} BB Position: {bb_percent*100:.1f}%\n"
            
            # CCI
            cci = get_indicator('CCI')
            if cci is not None:
                if cci < -100:
                    cci_status = f"{red_circle} OVERSOLD"
                elif cci > 100:
                    cci_status = f"{green_circle} OVERBOUGHT"
                else:
                    cci_status = f"{white_circle} NEUTRAL"
                summary += f"• {cci_status} CCI: {cci:.1f}\n"
            
            # Divergences
            if divergences['details']:
                summary += f"\n{blue_circle} DIVERGENCE ANALYSIS {blue_circle}\n"
                for detail in divergences['details'][:3]:
                    if 'Bullish' in detail:
                        summary += f"• {green_circle} {detail}\n"
                    elif 'Bearish' in detail:
                        summary += f"• {red_circle} {detail}\n"
                    else:
                        summary += f"• {white_circle} {detail}\n"
            else:
                summary += f"\n{blue_circle} DIVERGENCE ANALYSIS {blue_circle}\n• {white_circle} No significant divergences detected\n"
            
            # Reversal patterns
            if reversal_patterns['bullish_reversal']:
                summary += f"\n{green_circle} BULLISH REVERSAL PATTERN DETECTED {green_circle}\n"
                summary += f"• Confidence: {reversal_patterns['confidence']}%\n"
                for signal in reversal_patterns['signals'][:3]:
                    summary += f"• {green_circle} {signal}\n"
            
            if reversal_patterns['bearish_reversal']:
                summary += f"\n{red_circle} BEARISH REVERSAL PATTERN DETECTED {red_circle}\n"
                summary += f"• Confidence: {reversal_patterns['confidence']}%\n"
                for signal in reversal_patterns['signals'][:3]:
                    summary += f"• {red_circle} {signal}\n"
            
            summary += f"\n{yellow_circle} TECHNICAL SIGNALS: {len(signals)} total {yellow_circle}\n"
            summary += f"• {green_circle} Bullish: {len(bull_signals)}\n"
            summary += f"• {red_circle} Bearish: {len(bear_signals)}\n"
            summary += f"• {white_circle} Neutral: {len(neutral_signals)}\n"
            
            # Show top 5 strong signals
            strong_signals = [s for s in signals if s['strength'] == 'STRONG']
            if strong_signals:
                summary += f"\n{yellow_circle} STRONG SIGNALS {yellow_circle}\n"
                for signal in strong_signals[:5]:
                    if signal['direction'] == 'BULLISH':
                        summary += f"• {green_circle} {signal['type']}\n"
                    elif signal['direction'] == 'BEARISH':
                        summary += f"• {red_circle} {signal['type']}\n"
                    else:
                        summary += f"• {white_circle} {signal['type']}\n"
            
            # Overall recommendation
            if reversal_patterns['bullish_reversal'] and divergences.get('bullish_rsi', False):
                recommendation = f"{green_circle} STRONG BULLISH - High probability reversal upward"
                recommendation_emoji = green_circle
            elif reversal_patterns['bearish_reversal'] and divergences.get('bearish_rsi', False):
                recommendation = f"{red_circle} STRONG BEARISH - High probability reversal downward"
                recommendation_emoji = red_circle
            elif len(bull_signals) > len(bear_signals) + 5:
                recommendation = f"{green_circle} BULLISH - Strong buying pressure"
                recommendation_emoji = green_circle
            elif len(bear_signals) > len(bull_signals) + 5:
                recommendation = f"{red_circle} BEARISH - Strong selling pressure"
                recommendation_emoji = red_circle
            elif len(bull_signals) > len(bear_signals):
                recommendation = f"{green_circle} MILD BULLISH - Slight edge to bulls"
                recommendation_emoji = green_circle
            elif len(bear_signals) > len(bull_signals):
                recommendation = f"{red_circle} MILD BEARISH - Slight edge to bears"
                recommendation_emoji = red_circle
            else:
                recommendation = f"{white_circle} NEUTRAL - Balanced market"
                recommendation_emoji = white_circle
            
            summary += f"\n{yellow_circle} OVERALL RECOMMENDATION {yellow_circle}\n{recommendation_emoji} {recommendation}"
            
            summary += f"\n\n⏰ Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            return summary
            
        except Exception as e:
            print(f"Summary creation error: {e}")
            import traceback
            traceback.print_exc()
            return f"📊 Analysis for {ticker}\n\nComplete technical analysis generated."
    
    def _create_compact_summary(self, ticker: str, data: pd.DataFrame, info: Dict, 
                               signals: List[Dict], divergences: Dict) -> str:
        """Create compact summary for photo captions"""
        try:
            # Get scalar values
            latest_close = float(data['Close'].iloc[-1])
            if len(data) > 1:
                prev_close_val = float(data['Close'].iloc[-2])
            else:
                prev_close_val = latest_close
            
            # Calculate daily change safely
            if prev_close_val != 0:
                daily_change = ((latest_close - prev_close_val) / prev_close_val * 100)
            else:
                daily_change = 0
            
            bull_signals = len([s for s in signals if s['direction'] == 'BULLISH'])
            bear_signals = len([s for s in signals if s['direction'] == 'BEARISH'])
            
            # Get RSI value
            rsi_value = None
            if 'RSI' in data.columns:
                try:
                    rsi_value = float(data['RSI'].iloc[-1])
                except:
                    rsi_value = None
            
            # Get SMA50 position
            sma50_position = ""
            if 'SMA_50' in data.columns:
                try:
                    sma50 = float(data['SMA_50'].iloc[-1])
                    if latest_close > sma50:
                        sma50_position = "🟢 Above"
                    else:
                        sma50_position = "🔴 Below"
                except:
                    sma50_position = ""
            
            # Get volume ratio
            vol_ratio = ""
            if 'Volume_Ratio' in data.columns:
                try:
                    vol_ratio_val = float(data['Volume_Ratio'].iloc[-1])
                    if vol_ratio_val > 1.5:
                        vol_ratio = f"🟢 {vol_ratio_val:.1f}x avg"
                    elif vol_ratio_val < 0.5:
                        vol_ratio = f"🔴 {vol_ratio_val:.1f}x avg"
                    else:
                        vol_ratio = f"⚪ {vol_ratio_val:.1f}x avg"
                except:
                    vol_ratio = ""
            
            # Get A/D Line
            ad_line = ""
            if 'AD_Line' in data.columns:
                try:
                    ad_value = float(data['AD_Line'].iloc[-1])
                    if len(data) > 1:
                        prev_ad = float(data['AD_Line'].iloc[-2])
                        if ad_value > prev_ad:
                            ad_line = "🟢 A/D Up"
                        elif ad_value < prev_ad:
                            ad_line = "🔴 A/D Down"
                        else:
                            ad_line = "⚪ A/D Flat"
                except:
                    ad_line = ""
            
            currency = info.get('currency', 'USD')
            currency_symbol = '$' if currency == 'USD' else '€' if currency == 'EUR' else '£' if currency == 'GBP' else f'{currency} '
            
            # Determine overall sentiment with emoji
            if bull_signals > bear_signals + 5:
                sentiment = '🟢 BULLISH'
            elif bear_signals > bull_signals + 5:
                sentiment = '🔴 BEARISH'
            elif bull_signals > bear_signals:
                sentiment = '🟢 MILD BULLISH'
            elif bear_signals > bull_signals:
                sentiment = '🔴 MILD BEARISH'
            else:
                sentiment = '⚪ NEUTRAL'
            
            summary = f"📊 {ticker.upper()}\n"
            summary += f"💰 Price: {currency_symbol}{latest_close:.2f} ({daily_change:+.2f}%)\n"
            
            if rsi_value is not None:
                if rsi_value < 30:
                    summary += f"📈 RSI: {rsi_value:.1f} 🔴\n"
                elif rsi_value > 70:
                    summary += f"📈 RSI: {rsi_value:.1f} 🟢\n"
                else:
                    summary += f"📈 RSI: {rsi_value:.1f} ⚪\n"
            
            if sma50_position:
                summary += f"📏 50MA: {sma50_position}\n"
            
            if vol_ratio:
                summary += f"📊 {vol_ratio}\n"
            
            if ad_line:
                summary += f"📉 {ad_line}\n"
            
            if divergences['details']:
                div_count = len(divergences['details'])
                summary += f"🔀 Divergences: {div_count}\n"
            
            summary += f"📶 Signals: 🟢{bull_signals} | 🔴{bear_signals}\n"
            summary += f"🎯 Overall: {sentiment}"
            
            return summary[:1020] + "..." if len(summary) > 1020 else summary
            
        except Exception as e:
            print(f"Compact summary error: {e}")
            return f"📊 {ticker.upper()} - Technical Analysis"
    
    def _format_number(self, num: float) -> str:
        """Format large numbers"""
        try:
            num = float(num)
            if num >= 1_000_000_000:
                return f"{num/1_000_000_000:.1f}B"
            elif num >= 1_000_000:
                return f"{num/1_000_000:.1f}M"
            elif num >= 1_000:
                return f"{num/1_000:.1f}K"
            else:
                return f"{num:.0f}"
        except:
            return "N/A"