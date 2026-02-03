"""
Universal Trading Analyzer - Local Version
Complete technical analysis with all indicators
Optimized for Telegram and local usage
"""
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import warnings
import time
from datetime import datetime
import re

warnings.filterwarnings('ignore')

from config import CONFIG

class SimpleCache:
    """Simple in-memory cache for all tickers"""
    
    def __init__(self, ttl: int = 300, max_size: int = 100):
        self.ttl = ttl
        self.max_size = max_size
        self.cache = {}
        self.timestamps = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired"""
        if key in self.cache:
            if time.time() - self.timestamps[key] < self.ttl:
                return self.cache[key]
            else:
                del self.cache[key]
                del self.timestamps[key]
        return None
    
    def set(self, key: str, value: Any) -> None:
        """Set value in cache with timestamp"""
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.timestamps, key=self.timestamps.get)
            del self.cache[oldest_key]
            del self.timestamps[oldest_key]
        self.cache[key] = value
        self.timestamps[key] = time.time()


class TradingAnalyzer:
    """Universal analyzer for ALL Yahoo Finance tickers with complete technical analysis"""
    
    def __init__(self):
        self.config = CONFIG
        self.cache = SimpleCache(
            ttl=self.config.CACHE_TTL,
            max_size=self.config.MAX_CACHE_SIZE
        )
        
        print("✅ Universal Analyzer initialized (Complete Technical Analysis)")
        print("🌍 Supports all markets and tickers")
        print("📊 Includes: Trend, Momentum, Volume, Volatility, Oscillators")
    
    async def analyze_ticker(self, ticker_symbol: str, period: str = '1y') -> Dict:
        """Complete technical analysis method for any ticker"""
        try:
            print(f"🔍 Analyzing {ticker_symbol} ({period})")
            
            # Map period to Yahoo Finance format
            period_map = {
                '3m': '3mo', '6m': '6mo', '1y': '1y',
                '2y': '2y', '3y': '3y', '5y': '5y',
                'max': 'max'
            }
            yf_period = period_map.get(period.lower(), '1y')
            
            # Format ticker for Yahoo Finance
            formatted_ticker = self._format_ticker_for_yfinance(ticker_symbol)
            print(f"Formatted: {ticker_symbol} -> {formatted_ticker}")
            
            # Get exchange information
            exchange_info = self.config.get_exchange_info(formatted_ticker)
            print(f"Exchange: {exchange_info['exchange']}")
            
            # Check cache first
            cache_key = f"{formatted_ticker}_{yf_period}"
            cached_data = self.cache.get(cache_key)
            
            if cached_data is not None:
                print("✅ Using cached data")
                data = cached_data
            else:
                # Fetch fresh data from Yahoo Finance
                data = self._fetch_data_simple(formatted_ticker, yf_period)
                
                if data is None or data.empty:
                    return {
                        'success': False,
                        'error': f"No data found for {ticker_symbol}."
                    }
                
                # Cache the data
                self.cache.set(cache_key, data)
                print(f"✅ Data fetched: {len(data)} rows")
            
            # Clean and prepare data
            data = self._clean_data(data)
            
            # Calculate complete set of technical indicators
            data = self._calculate_complete_indicators(data)
            
            # Get ticker information
            info = await self._get_ticker_info(formatted_ticker, data, ticker_symbol)
            
            # Generate trading signals from all indicators
            signals = self._generate_all_signals(data)
            
            # Detect reversal patterns
            reversal_patterns = self._detect_reversal_patterns(data)
            
            # Detect divergences
            divergences = self._detect_divergences(data)
            
            # Calculate performance statistics
            stats = self._calculate_statistics(data)
            
            # Prepare comprehensive summary
            summary = self._create_comprehensive_summary(
                ticker_symbol, data, info, signals, 
                reversal_patterns, divergences, stats, exchange_info
            )
            
            # Prepare compact summary for chart caption
            compact = self._create_compact_summary(ticker_symbol, data, info, signals, divergences)
            
            # Debug information
            indicator_count = len([col for col in data.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']])
            bull_count = len([s for s in signals if s['direction'] == 'BULLISH'])
            bear_count = len([s for s in signals if s['direction'] == 'BEARISH'])
            
            print(f"✅ Analysis complete for {ticker_symbol}")
            print(f"📊 Indicators calculated: {indicator_count}")
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
                'error': f"Analysis failed: {str(e)}"
            }
    
    def _format_ticker_for_yfinance(self, ticker: str) -> str:
        """Format ticker for Yahoo Finance API"""
        ticker = ticker.upper().strip().replace('$', '')
        return ticker
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare data for analysis"""
        if data.empty:
            return data
        
        df = data.copy()
        
        # Handle multi-index columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Rename columns if needed
        if 'Adj Close' in df.columns:
            df = df.rename(columns={'Adj Close': 'Close'})
        
        # Ensure required columns exist
        required_columns = ['Close', 'High', 'Low', 'Open', 'Volume']
        for col in required_columns:
            if col not in df.columns:
                if col == 'Volume':
                    df[col] = 0
                elif col == 'Close':
                    df[col] = df.iloc[:, -1] if len(df.columns) > 0 else 0
        
        # Convert all columns to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with NaN in critical columns
        df = df.dropna(subset=['Close', 'High', 'Low', 'Open'])
        
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        return df
    
    def _fetch_data_simple(self, ticker: str, period: str) -> Optional[pd.DataFrame]:
        """Fetch data from Yahoo Finance with simple retry logic"""
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
            
            # Small delay to respect Yahoo's rate limits
            time.sleep(0.2)
            
            return data if not data.empty else None
            
        except Exception as e:
            print(f"Download error for {ticker}: {e}")
            
            # Try alternative method
            try:
                ticker_obj = yf.Ticker(ticker)
                data = ticker_obj.history(
                    period=period,
                    interval="1d",
                    timeout=self.config.YAHOO_TIMEOUT
                )
                return data if not data.empty else None
            except Exception as e2:
                print(f"Alternative fetch also failed: {e2}")
                return None
    
    async def _get_ticker_info(self, ticker: str, data: pd.DataFrame, original_ticker: str) -> Dict:
        """Get ticker information from Yahoo Finance"""
        try:
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.info
            
            if not info or 'symbol' not in info:
                info = {
                    'symbol': original_ticker,
                    'longName': original_ticker,
                    'shortName': original_ticker,
                    'currency': 'USD',
                    'marketCap': None,
                    'sector': None,
                    'industry': None,
                    'country': 'USA',
                }
            
            # Add price data from our analysis
            if len(data) > 0:
                latest = data.iloc[-1]
                info['last_price'] = float(latest['Close'])
                info['last_volume'] = float(latest.get('Volume', 0))
                info['last_open'] = float(latest.get('Open', latest['Close']))
                info['last_high'] = float(latest.get('High', latest['Close']))
                info['last_low'] = float(latest.get('Low', latest['Close']))
            
            # Ensure currency is set
            if 'currency' not in info:
                info['currency'] = 'USD'
            
            return info
            
        except Exception as e:
            print(f"Warning: Could not get ticker info: {e}")
            return {
                'symbol': original_ticker,
                'longName': original_ticker,
                'shortName': original_ticker,
                'currency': 'USD',
                'last_price': float(data['Close'].iloc[-1]) if len(data) > 0 else 0,
                'country': 'USA',
            }
    
    def _calculate_complete_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate complete set of technical indicators"""
        df = data.copy()
        
        if len(df) < 20:
            return df
        
        try:
            # Prepare price data
            for col in ['Close', 'High', 'Low', 'Open', 'Volume']:
                if col in df.columns:
                    values = df[col].values
                    if values.ndim > 1:
                        values = values.flatten()
                    df[col] = pd.Series(values, index=df.index)
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            print(f"📊 Data shape: {df.shape}")
            print(f"📊 Calculating complete technical indicators...")
            
            # === TREND INDICATORS ===
            print("📈 Calculating trend indicators...")
            
            # Moving Averages
            for period in [5, 10, 20, 50, 100, 200]:
                if len(df) >= period:
                    df[f'SMA_{period}'] = df['Close'].rolling(window=period, min_periods=1).mean()
                    df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
                else:
                    df[f'SMA_{period}'] = df['Close'].expanding().mean()
                    df[f'EMA_{period}'] = df['Close'].ewm(span=min(len(df), period), adjust=False).mean()
            
            # === MOMENTUM INDICATORS ===
            print("📊 Calculating momentum indicators...")
            
            # MACD
            if len(df) >= 26:
                exp1 = df['Close'].ewm(span=12, adjust=False).mean()
                exp2 = df['Close'].ewm(span=26, adjust=False).mean()
                df['MACD'] = exp1 - exp2
                df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
                df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
            
            # RSI
            if len(df) >= 14:
                delta = df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
                rs = gain / loss
                df['RSI'] = 100 - (100 / (1 + rs))
                df['RSI'] = df['RSI'].fillna(50)
            
            # Stochastic
            if len(df) >= 14:
                low_14 = df['Low'].rolling(window=14, min_periods=1).min()
                high_14 = df['High'].rolling(window=14, min_periods=1).max()
                df['Stoch_%K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
                df['Stoch_%D'] = df['Stoch_%K'].rolling(window=3, min_periods=1).mean()
            
            # Williams %R
            if len(df) >= 14:
                highest_high = df['High'].rolling(window=14, min_periods=1).max()
                lowest_low = df['Low'].rolling(window=14, min_periods=1).min()
                df['Williams_%R'] = -100 * ((highest_high - df['Close']) / (highest_high - lowest_low))
            
            # CCI (Commodity Channel Index)
            if len(df) >= 20:
                tp = (df['High'] + df['Low'] + df['Close']) / 3
                sma = tp.rolling(window=20, min_periods=1).mean()
                mad = tp.rolling(window=20, min_periods=1).apply(
                    lambda x: np.abs(x - x.mean()).mean(), raw=True
                )
                df['CCI'] = (tp - sma) / (0.015 * mad)
            
            # ROC (Rate of Change)
            for period in [12, 14, 26]:
                if len(df) >= period:
                    df[f'ROC_{period}'] = ((df['Close'] - df['Close'].shift(period)) / df['Close'].shift(period)) * 100
            
            # === VOLUME INDICATORS ===
            print("📈 Calculating volume indicators...")
            
            if 'Volume' in df.columns:
                # OBV (On Balance Volume)
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
                
                # Volume Moving Average
                df['Volume_MA_20'] = df['Volume'].rolling(window=20, min_periods=1).mean()
                df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_20'].replace(0, 1)
                
                # Money Flow Index
                if len(df) >= 14:
                    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
                    money_flow = typical_price * df['Volume']
                    
                    positive_flow = np.where(typical_price > typical_price.shift(1), money_flow, 0)
                    negative_flow = np.where(typical_price < typical_price.shift(1), money_flow, 0)
                    
                    positive_mf = pd.Series(positive_flow, index=df.index).rolling(window=14, min_periods=1).sum()
                    negative_mf = pd.Series(negative_flow, index=df.index).rolling(window=14, min_periods=1).sum()
                    
                    money_ratio = positive_mf / negative_mf.replace(0, 1)
                    df['MFI'] = 100 - (100 / (1 + money_ratio))
                    df['MFI'] = df['MFI'].fillna(50)
            
            # === VOLATILITY INDICATORS ===
            print("📉 Calculating volatility indicators...")
            
            # Bollinger Bands
            if len(df) >= 20:
                df['BB_Middle'] = df['Close'].rolling(window=20, min_periods=1).mean()
                bb_std = df['Close'].rolling(window=20, min_periods=1).std()
                df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
                df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
                df['BB_Width'] = ((df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']) * 100
                df['BB_%B'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
                df['BB_%B'] = df['BB_%B'].fillna(0.5)
            
            # ATR (Average True Range)
            if len(df) >= 14:
                high_low = df['High'] - df['Low']
                high_close = np.abs(df['High'] - df['Close'].shift())
                low_close = np.abs(df['Low'] - df['Close'].shift())
                
                ranges = pd.DataFrame({
                    'high_low': high_low,
                    'high_close': high_close,
                    'low_close': low_close
                })
                
                true_range = ranges.max(axis=1)
                df['ATR'] = true_range.rolling(window=14, min_periods=1).mean()
                df['ATR_Pct'] = (df['ATR'] / df['Close']) * 100
            
            # === OSCILLATORS ===
            print("📊 Calculating oscillators...")
            
            # Awesome Oscillator
            if len(df) >= 34:
                median_price = (df['High'] + df['Low']) / 2
                df['AO'] = median_price.rolling(window=5, min_periods=1).mean() - median_price.rolling(window=34, min_periods=1).mean()
            
            # Chaikin Oscillator
            if 'AD_Line' in df.columns and len(df) >= 10:
                df['Chaikin_3EMA'] = df['AD_Line'].ewm(span=3, adjust=False).mean()
                df['Chaikin_10EMA'] = df['AD_Line'].ewm(span=10, adjust=False).mean()
                df['Chaikin_Osc'] = df['Chaikin_3EMA'] - df['Chaikin_10EMA']
            
            # === SUPPORT/RESISTANCE ===
            print("📈 Calculating support/resistance levels...")
            
            if len(df) >= 20:
                df['Resistance_20'] = df['High'].rolling(window=20, min_periods=1).max()
                df['Support_20'] = df['Low'].rolling(window=20, min_periods=1).min()
            
            # === PERFORMANCE METRICS ===
            print("📊 Calculating performance metrics...")
            
            df['Daily_Return'] = df['Close'].pct_change()
            df['Cumulative_Return'] = (1 + df['Daily_Return']).cumprod() - 1
            
            # Annualized volatility
            df['Volatility_20d'] = df['Daily_Return'].rolling(window=20, min_periods=1).std() * np.sqrt(252)
            df['Volatility_60d'] = df['Daily_Return'].rolling(window=60, min_periods=1).std() * np.sqrt(252)
            
            # Fill NaN values
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            # Display key indicators
            if len(df) > 0:
                last_row = df.iloc[-1]
                print(f"\n📊 KEY INDICATORS (Latest):")
                print(f"  Price: ${last_row.get('Close', 0):.2f}")
                print(f"  RSI: {last_row.get('RSI', 0):.1f}")
                print(f"  MACD: {last_row.get('MACD', 0):.4f}")
                print(f"  BB %B: {last_row.get('BB_%B', 0):.2f}")
                print(f"  Volume Ratio: {last_row.get('Volume_Ratio', 0):.1f}x")
                print(f"  A/D Line: {last_row.get('AD_Line', 0):.0f}")
                print(f"  Stochastic %K: {last_row.get('Stoch_%K', 0):.1f}")
                print(f"  CCI: {last_row.get('CCI', 0):.1f}")
            
            indicator_count = len([col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']])
            print(f"\n✅ Calculated {indicator_count} technical indicators")
            
        except Exception as e:
            print(f"⚠️ Indicator calculation error: {e}")
            import traceback
            traceback.print_exc()
        
        return df
    
    def _generate_all_signals(self, data: pd.DataFrame) -> List[Dict]:
        """Generate trading signals from all technical indicators"""
        signals = []
        
        if len(data) < 20:
            print("⚠️ Not enough data for signals")
            return signals
        
        try:
            # Get current and previous index
            last_idx = len(data) - 1
            prev_idx = last_idx - 1 if last_idx > 0 else last_idx
            
            # Helper function to get scalar values safely
            def get_scalar(column, idx):
                try:
                    val = data[column].iloc[idx]
                    if pd.isna(val):
                        return None
                    return float(val)
                except:
                    return None
            
            # Current price
            close = get_scalar('Close', last_idx)
            
            # === TREND SIGNALS ===
            print("📈 Generating trend signals...")
            
            # Moving Average Crossovers
            for fast, slow in [(20, 50), (50, 200), (20, 200)]:
                fast_ma = get_scalar(f'SMA_{fast}', last_idx)
                slow_ma = get_scalar(f'SMA_{slow}', last_idx)
                prev_fast = get_scalar(f'SMA_{fast}', prev_idx)
                prev_slow = get_scalar(f'SMA_{slow}', prev_idx)
                
                if all(v is not None for v in [fast_ma, slow_ma, prev_fast, prev_slow]):
                    if prev_fast <= prev_slow and fast_ma > slow_ma:
                        signals.append({'type': f'MA_CROSS_{fast}_{slow}_BULLISH', 'strength': 'STRONG', 'direction': 'BULLISH'})
                    elif prev_fast >= prev_slow and fast_ma < slow_ma:
                        signals.append({'type': f'MA_CROSS_{fast}_{slow}_BEARISH', 'strength': 'STRONG', 'direction': 'BEARISH'})
            
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
            print("📊 Generating momentum signals...")
            
            # RSI Signals
            rsi = get_scalar('RSI', last_idx)
            if rsi is not None:
                if rsi < 30:
                    signals.append({'type': 'RSI_OVERSOLD', 'strength': 'STRONG', 'direction': 'BULLISH'})
                elif rsi > 70:
                    signals.append({'type': 'RSI_OVERBOUGHT', 'strength': 'STRONG', 'direction': 'BEARISH'})
                elif rsi < 40:
                    signals.append({'type': 'RSI_NEAR_OVERSOLD', 'strength': 'MODERATE', 'direction': 'BULLISH'})
                elif rsi > 60:
                    signals.append({'type': 'RSI_NEAR_OVERBOUGHT', 'strength': 'MODERATE', 'direction': 'BEARISH'})
            
            # MACD Signals
            macd = get_scalar('MACD', last_idx)
            macd_signal = get_scalar('MACD_Signal', last_idx)
            prev_macd = get_scalar('MACD', prev_idx)
            prev_signal = get_scalar('MACD_Signal', prev_idx)
            
            if all(v is not None for v in [macd, macd_signal, prev_macd, prev_signal]):
                if prev_macd <= prev_signal and macd > macd_signal:
                    signals.append({'type': 'MACD_BULLISH_CROSS', 'strength': 'STRONG', 'direction': 'BULLISH'})
                elif prev_macd >= prev_signal and macd < macd_signal:
                    signals.append({'type': 'MACD_BEARISH_CROSS', 'strength': 'STRONG', 'direction': 'BEARISH'})
                
                if macd > macd_signal:
                    signals.append({'type': 'MACD_ABOVE_SIGNAL', 'strength': 'MODERATE', 'direction': 'BULLISH'})
                else:
                    signals.append({'type': 'MACD_BELOW_SIGNAL', 'strength': 'MODERATE', 'direction': 'BEARISH'})
            
            # Stochastic Signals
            stoch_k = get_scalar('Stoch_%K', last_idx)
            stoch_d = get_scalar('Stoch_%D', last_idx)
            if stoch_k is not None and stoch_d is not None:
                if stoch_k < 20 and stoch_d < 20:
                    signals.append({'type': 'STOCH_OVERSOLD', 'strength': 'MODERATE', 'direction': 'BULLISH'})
                elif stoch_k > 80 and stoch_d > 80:
                    signals.append({'type': 'STOCH_OVERBOUGHT', 'strength': 'MODERATE', 'direction': 'BEARISH'})
            
            # Williams %R Signals
            williams = get_scalar('Williams_%R', last_idx)
            if williams is not None:
                if williams < -80:
                    signals.append({'type': 'WILLIAMS_OVERSOLD', 'strength': 'MODERATE', 'direction': 'BULLISH'})
                elif williams > -20:
                    signals.append({'type': 'WILLIAMS_OVERBOUGHT', 'strength': 'MODERATE', 'direction': 'BEARISH'})
            
            # CCI Signals
            cci = get_scalar('CCI', last_idx)
            if cci is not None:
                if cci < -100:
                    signals.append({'type': 'CCI_OVERSOLD', 'strength': 'MODERATE', 'direction': 'BULLISH'})
                elif cci > 100:
                    signals.append({'type': 'CCI_OVERBOUGHT', 'strength': 'MODERATE', 'direction': 'BEARISH'})
            
            # === VOLUME SIGNALS ===
            print("📈 Generating volume signals...")
            
            # Volume Ratio
            vol_ratio = get_scalar('Volume_Ratio', last_idx)
            if vol_ratio is not None:
                if vol_ratio > 2.0:
                    signals.append({'type': 'HIGH_VOLUME', 'strength': 'STRONG', 'direction': 'NEUTRAL'})
                elif vol_ratio > 1.5:
                    signals.append({'type': 'ABOVE_AVG_VOLUME', 'strength': 'MODERATE', 'direction': 'NEUTRAL'})
                elif vol_ratio < 0.5:
                    signals.append({'type': 'LOW_VOLUME', 'strength': 'MODERATE', 'direction': 'NEUTRAL'})
            
            # A/D Line Trend
            ad_line = get_scalar('AD_Line', last_idx)
            if ad_line is not None:
                prev_ad = get_scalar('AD_Line', prev_idx)
                if prev_ad is not None:
                    if ad_line > prev_ad:
                        signals.append({'type': 'AD_LINE_RISING', 'strength': 'MODERATE', 'direction': 'BULLISH'})
                    elif ad_line < prev_ad:
                        signals.append({'type': 'AD_LINE_FALLING', 'strength': 'MODERATE', 'direction': 'BEARISH'})
            
            # MFI Signals
            mfi = get_scalar('MFI', last_idx)
            if mfi is not None:
                if mfi < 20:
                    signals.append({'type': 'MFI_OVERSOLD', 'strength': 'MODERATE', 'direction': 'BULLISH'})
                elif mfi > 80:
                    signals.append({'type': 'MFI_OVERBOUGHT', 'strength': 'MODERATE', 'direction': 'BEARISH'})
            
            # === VOLATILITY SIGNALS ===
            print("📉 Generating volatility signals...")
            
            # Bollinger Bands Signals
            bb_percent = get_scalar('BB_%B', last_idx)
            if bb_percent is not None:
                if bb_percent < 0.2:
                    signals.append({'type': 'BB_OVERSOLD', 'strength': 'STRONG', 'direction': 'BULLISH'})
                elif bb_percent > 0.8:
                    signals.append({'type': 'BB_OVERBOUGHT', 'strength': 'STRONG', 'direction': 'BEARISH'})
                elif 0.4 < bb_percent < 0.6:
                    signals.append({'type': 'BB_MIDDLE', 'strength': 'NEUTRAL', 'direction': 'NEUTRAL'})
            
            bb_width = get_scalar('BB_Width', last_idx)
            if bb_width is not None:
                if bb_width < 5:
                    signals.append({'type': 'BB_SQUEEZE', 'strength': 'MODERATE', 'direction': 'NEUTRAL'})
                elif bb_width > 15:
                    signals.append({'type': 'BB_EXPANSION', 'strength': 'MODERATE', 'direction': 'NEUTRAL'})
            
            # ATR Signals
            atr_pct = get_scalar('ATR_Pct', last_idx)
            if atr_pct is not None:
                if atr_pct > 3:
                    signals.append({'type': 'HIGH_VOLATILITY', 'strength': 'MODERATE', 'direction': 'NEUTRAL'})
                elif atr_pct < 1:
                    signals.append({'type': 'LOW_VOLATILITY', 'strength': 'MODERATE', 'direction': 'NEUTRAL'})
            
            # === OSCILLATOR SIGNALS ===
            print("📊 Generating oscillator signals...")
            
            # Awesome Oscillator
            ao = get_scalar('AO', last_idx)
            prev_ao = get_scalar('AO', prev_idx)
            if ao is not None and prev_ao is not None:
                if ao > 0 and prev_ao <= 0:
                    signals.append({'type': 'AO_BULLISH_CROSS', 'strength': 'MODERATE', 'direction': 'BULLISH'})
                elif ao < 0 and prev_ao >= 0:
                    signals.append({'type': 'AO_BEARISH_CROSS', 'strength': 'MODERATE', 'direction': 'BEARISH'})
            
            # Chaikin Oscillator
            chaikin = get_scalar('Chaikin_Osc', last_idx)
            if chaikin is not None:
                if chaikin > 0:
                    signals.append({'type': 'CHAIKIN_BULLISH', 'strength': 'MODERATE', 'direction': 'BULLISH'})
                else:
                    signals.append({'type': 'CHAIKIN_BEARISH', 'strength': 'MODERATE', 'direction': 'BEARISH'})
            
            print(f"✅ Generated {len(signals)} total signals")
            
        except Exception as e:
            print(f"⚠️ Signal generation error: {e}")
            import traceback
            traceback.print_exc()
        
        return signals
    
    def _detect_reversal_patterns(self, data: pd.DataFrame) -> Dict:
        """Detect reversal patterns from multiple indicators"""
        patterns = {
            'bullish_reversal': False,
            'bearish_reversal': False,
            'confidence': 0,
            'signals': [],
            'details': {}
        }
        
        if len(data) < 30:
            return patterns
        
        try:
            last_idx = len(data) - 1
            
            def get_scalar(column, idx):
                try:
                    val = data[column].iloc[idx]
                    if pd.isna(val):
                        return None
                    return float(val)
                except:
                    return None
            
            # === BULLISH REVERSAL INDICATORS ===
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
                bullish_details.append(f"Stochastic oversold (K={stoch_k:.1f}, D={stoch_d:.1f})")
            
            # Williams %R oversold
            williams = get_scalar('Williams_%R', last_idx)
            if williams is not None and williams < -80:
                bullish_count += 1
                bullish_details.append(f"Williams %R oversold ({williams:.1f})")
            
            # Bollinger Bands oversold
            bb_percent = get_scalar('BB_%B', last_idx)
            if bb_percent is not None and bb_percent < 0.2:
                bullish_count += 1
                bullish_details.append(f"BB position: {bb_percent*100:.1f}% (oversold)")
            
            # MFI oversold
            mfi = get_scalar('MFI', last_idx)
            if mfi is not None and mfi < 20:
                bullish_count += 1
                bullish_details.append(f"MFI oversold ({mfi:.1f})")
            
            # CCI oversold
            cci = get_scalar('CCI', last_idx)
            if cci is not None and cci < -100:
                bullish_count += 1
                bullish_details.append(f"CCI oversold ({cci:.1f})")
            
            # Determine bullish reversal
            if bullish_count >= 3:
                patterns['bullish_reversal'] = True
                patterns['confidence'] = min(95, bullish_count * 20)
                patterns['signals'] = bullish_details
                patterns['details']['bullish_count'] = bullish_count
            
            # === BEARISH REVERSAL INDICATORS ===
            bearish_count = 0
            bearish_details = []
            
            # RSI overbought
            if rsi is not None and rsi > 70:
                bearish_count += 1
                bearish_details.append(f"RSI overbought ({rsi:.1f})")
            
            # Stochastic overbought
            if stoch_k is not None and stoch_d is not None and stoch_k > 80 and stoch_d > 80:
                bearish_count += 1
                bearish_details.append(f"Stochastic overbought (K={stoch_k:.1f}, D={stoch_d:.1f})")
            
            # Williams %R overbought
            if williams is not None and williams > -20:
                bearish_count += 1
                bearish_details.append(f"Williams %R overbought ({williams:.1f})")
            
            # Bollinger Bands overbought
            if bb_percent is not None and bb_percent > 0.8:
                bearish_count += 1
                bearish_details.append(f"BB position: {bb_percent*100:.1f}% (overbought)")
            
            # MFI overbought
            if mfi is not None and mfi > 80:
                bearish_count += 1
                bearish_details.append(f"MFI overbought ({mfi:.1f})")
            
            # CCI overbought
            if cci is not None and cci > 100:
                bearish_count += 1
                bearish_details.append(f"CCI overbought ({cci:.1f})")
            
            # Determine bearish reversal
            if bearish_count >= 3:
                patterns['bearish_reversal'] = True
                patterns['confidence'] = max(patterns['confidence'], min(95, bearish_count * 20))
                patterns['signals'].extend(bearish_details)
                patterns['details']['bearish_count'] = bearish_count
            
            if patterns['bullish_reversal'] or patterns['bearish_reversal']:
                print(f"✅ Reversal pattern detected: {bullish_count} bullish, {bearish_count} bearish signals")
            
        except Exception as e:
            print(f"⚠️ Reversal pattern error: {e}")
        
        return patterns
    
    def _detect_divergences(self, data: pd.DataFrame, lookback: int = 30) -> Dict:
        """Detect divergences between price and indicators"""
        divergences = {
            'bullish_rsi': False,
            'bearish_rsi': False,
            'bullish_macd': False,
            'bearish_macd': False,
            'bullish_stoch': False,
            'bearish_stoch': False,
            'details': []
        }
        
        if len(data) < lookback + 10:
            return divergences
        
        try:
            recent = data.iloc[-lookback:].copy()
            
            # Get price and indicator arrays
            price = recent['Close'].values
            rsi = recent['RSI'].values if 'RSI' in recent.columns else None
            macd = recent['MACD'].values if 'MACD' in recent.columns else None
            stoch_k = recent['Stoch_%K'].values if 'Stoch_%K' in recent.columns else None
            
            # Simple divergence detection (last 20 periods)
            slice_len = 20
            
            if len(price) >= slice_len and rsi is not None and len(rsi) >= slice_len:
                price_slice = price[-slice_len:]
                rsi_slice = rsi[-slice_len:]
                
                # Bullish divergence: price makes lower low, RSI makes higher low
                if price_slice[-1] < price_slice[-10] and rsi_slice[-1] > rsi_slice[-10]:
                    divergences['bullish_rsi'] = True
                    divergences['details'].append("RSI Bullish Divergence")
                
                # Bearish divergence: price makes higher high, RSI makes lower high
                if price_slice[-1] > price_slice[-10] and rsi_slice[-1] < rsi_slice[-10]:
                    divergences['bearish_rsi'] = True
                    divergences['details'].append("RSI Bearish Divergence")
            
            if len(price) >= slice_len and macd is not None and len(macd) >= slice_len:
                price_slice = price[-slice_len:]
                macd_slice = macd[-slice_len:]
                
                # MACD divergence detection
                if price_slice[-1] < price_slice[-10] and macd_slice[-1] > macd_slice[-10]:
                    divergences['bullish_macd'] = True
                    divergences['details'].append("MACD Bullish Divergence")
                
                if price_slice[-1] > price_slice[-10] and macd_slice[-1] < macd_slice[-10]:
                    divergences['bearish_macd'] = True
                    divergences['details'].append("MACD Bearish Divergence")
            
            if len(price) >= slice_len and stoch_k is not None and len(stoch_k) >= slice_len:
                price_slice = price[-slice_len:]
                stoch_slice = stoch_k[-slice_len:]
                
                # Stochastic divergence detection
                if price_slice[-1] < price_slice[-10] and stoch_slice[-1] > stoch_slice[-10]:
                    divergences['bullish_stoch'] = True
                    divergences['details'].append("Stochastic Bullish Divergence")
                
                if price_slice[-1] > price_slice[-10] and stoch_slice[-1] < stoch_slice[-10]:
                    divergences['bearish_stoch'] = True
                    divergences['details'].append("Stochastic Bearish Divergence")
            
            if divergences['details']:
                print(f"✅ Divergences detected: {len(divergences['details'])}")
            
        except Exception as e:
            print(f"⚠️ Divergence detection error: {e}")
        
        return divergences
    
    def _calculate_statistics(self, data: pd.DataFrame) -> Dict:
        """Calculate performance statistics"""
        if len(data) < 5:
            return {}
        
        try:
            close_series = data['Close']
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
            total_trades = len(returns)
            
            stats = {
                'total_return': float(((close_series.iloc[-1] / close_series.iloc[0]) - 1) * 100),
                'avg_daily_return': float(returns.mean() * 100),
                'volatility': float(returns.std() * np.sqrt(252) * 100),
                'sharpe_ratio': float(returns.mean() / returns.std() * np.sqrt(252)) if returns.std() != 0 else 0,
                'max_drawdown': float(self._calculate_max_drawdown(close_series)),
                'win_rate': float(win_count / total_trades * 100) if total_trades > 0 else 0,
                'best_day': float(returns.max() * 100),
                'worst_day': float(returns.min() * 100),
                'avg_volume': float(data['Volume'].mean()) if 'Volume' in data.columns else 0,
                'avg_true_range': float(data['ATR'].mean()) if 'ATR' in data.columns else 0,
            }
            
            print(f"📊 Statistics calculated: Return={stats['total_return']:.1f}%, Volatility={stats['volatility']:.1f}%")
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
    
    def _clean_text_for_telegram(self, text: str) -> str:
        """Clean text to avoid Telegram HTML parsing errors"""
        if not text:
            return text
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Replace problematic characters
        text = text.replace('<', '(').replace('>', ')')
        text = text.replace('&', 'and')
        
        # Clean up any remaining HTML entities
        text = text.replace('&lt;', '<').replace('&gt;', '>')
        text = text.replace('&amp;', '&').replace('&quot;', '"').replace('&apos;', "'")
        
        return text
    
    def _create_comprehensive_summary(self, ticker: str, data: pd.DataFrame, info: Dict, 
                                     signals: List[Dict], reversal_patterns: Dict,
                                     divergences: Dict, stats: Dict, exchange_info: Dict) -> str:
        """Create comprehensive technical analysis summary"""
        try:
            # Get latest values
            latest_close = float(data['Close'].iloc[-1])
            if len(data) > 1:
                prev_close_val = float(data['Close'].iloc[-2])
            else:
                prev_close_val = latest_close
            
            # Calculate daily change
            if prev_close_val != 0:
                daily_change = ((latest_close - prev_close_val) / prev_close_val * 100)
            else:
                daily_change = 0
            
            latest_volume = float(data['Volume'].iloc[-1]) if 'Volume' in data.columns else 0
            
            # Count signals by direction
            bull_signals = [s for s in signals if s['direction'] == 'BULLISH']
            bear_signals = [s for s in signals if s['direction'] == 'BEARISH']
            neutral_signals = [s for s in signals if s['direction'] == 'NEUTRAL']
            
            currency = info.get('currency', 'USD')
            currency_symbol = '$' if currency == 'USD' else '€' if currency == 'EUR' else '£' if currency == 'GBP' else f'{currency} '
            
            # Helper function to get indicator values safely
            def get_indicator(col):
                try:
                    val = data[col].iloc[-1]
                    if pd.isna(val):
                        return None
                    return float(val)
                except:
                    return None
            
            # === CREATE SUMMARY ===
            summary = f"""
📊 COMPLETE TECHNICAL ANALYSIS: {ticker.upper()}

📈 MARKET OVERVIEW
• Exchange: {exchange_info['exchange']}
• Currency: {currency}
• Last Price: {currency_symbol}{latest_close:.2f} ({daily_change:+.2f}%)
• Volume: {self._format_number(latest_volume)} shares

📊 PERFORMANCE STATISTICS
• Total Return: {stats.get('total_return', 0):.2f}%
• Avg Daily Return: {stats.get('avg_daily_return', 0):.3f}%
• Volatility: {stats.get('volatility', 0):.1f}%
• Sharpe Ratio: {stats.get('sharpe_ratio', 0):.2f}
• Max Drawdown: {stats.get('max_drawdown', 0):.1f}%
• Win Rate: {stats.get('win_rate', 0):.1f}%
• Best Day: {stats.get('best_day', 0):+.2f}%
• Worst Day: {stats.get('worst_day', 0):+.2f}%

📈 TREND ANALYSIS
"""
            
            # Moving Averages
            for period in [20, 50, 200]:
                sma = get_indicator(f'SMA_{period}')
                ema = get_indicator(f'EMA_{period}')
                if sma is not None:
                    if latest_close > sma:
                        summary += f"• 🟢 Above {period}-day SMA: {currency_symbol}{sma:.2f}\n"
                    else:
                        summary += f"• 🔴 Below {period}-day SMA: {currency_symbol}{sma:.2f}\n"
            
            summary += f"""
📊 MOMENTUM INDICATORS
"""
            
            # RSI
            rsi = get_indicator('RSI')
            if rsi is not None:
                if rsi < 30:
                    summary += f"• 🟢 RSI OVERSOLD: {rsi:.1f}\n"
                elif rsi > 70:
                    summary += f"• 🔴 RSI OVERBOUGHT: {rsi:.1f}\n"
                elif rsi < 50:
                    summary += f"• 🔴 RSI Bearish: {rsi:.1f}\n"
                else:
                    summary += f"• 🟢 RSI Bullish: {rsi:.1f}\n"
            
            # MACD
            macd = get_indicator('MACD')
            signal = get_indicator('MACD_Signal')
            if macd is not None and signal is not None:
                if macd > signal:
                    summary += f"• 🟢 MACD Bullish: {macd:.4f} > Signal: {signal:.4f}\n"
                else:
                    summary += f"• 🔴 MACD Bearish: {macd:.4f} < Signal: {signal:.4f}\n"
            
            # Stochastic
            stoch_k = get_indicator('Stoch_%K')
            stoch_d = get_indicator('Stoch_%D')
            if stoch_k is not None and stoch_d is not None:
                if stoch_k < 20 and stoch_d < 20:
                    summary += f"• 🟢 Stochastic OVERSOLD: K={stoch_k:.1f}, D={stoch_d:.1f}\n"
                elif stoch_k > 80 and stoch_d > 80:
                    summary += f"• 🔴 Stochastic OVERBOUGHT: K={stoch_k:.1f}, D={stoch_d:.1f}\n"
                else:
                    summary += f"• ⚪ Stochastic Neutral: K={stoch_k:.1f}, D={stoch_d:.1f}\n"
            
            # Williams %R
            williams = get_indicator('Williams_%R')
            if williams is not None:
                if williams < -80:
                    summary += f"• 🟢 Williams %R OVERSOLD: {williams:.1f}\n"
                elif williams > -20:
                    summary += f"• 🔴 Williams %R OVERBOUGHT: {williams:.1f}\n"
            
            # CCI
            cci = get_indicator('CCI')
            if cci is not None:
                if cci < -100:
                    summary += f"• 🟢 CCI OVERSOLD: {cci:.1f}\n"
                elif cci > 100:
                    summary += f"• 🔴 CCI OVERBOUGHT: {cci:.1f}\n"
            
            summary += f"""
📈 VOLUME ANALYSIS
"""
            
            # Volume Ratio
            vol_ratio = get_indicator('Volume_Ratio')
            if vol_ratio is not None:
                if vol_ratio > 2.0:
                    summary += f"• 🔥 High Volume: {vol_ratio:.1f}x average\n"
                elif vol_ratio > 1.5:
                    summary += f"• 📈 Above Avg Volume: {vol_ratio:.1f}x average\n"
                elif vol_ratio < 0.5:
                    summary += f"• 📉 Low Volume: {vol_ratio:.1f}x average\n"
                else:
                    summary += f"• ⚪ Normal Volume: {vol_ratio:.1f}x average\n"
            
            # A/D Line
            ad_line = get_indicator('AD_Line')
            if ad_line is not None:
                summary += f"• A/D Line: {ad_line:.0f}\n"
            
            # MFI
            mfi = get_indicator('MFI')
            if mfi is not None:
                if mfi < 20:
                    summary += f"• 🟢 MFI OVERSOLD: {mfi:.1f}\n"
                elif mfi > 80:
                    summary += f"• 🔴 MFI OVERBOUGHT: {mfi:.1f}\n"
            
            summary += f"""
📉 VOLATILITY & BOLLINGER BANDS
"""
            
            # Bollinger Bands
            bb_percent = get_indicator('BB_%B')
            if bb_percent is not None:
                if bb_percent < 0.2:
                    summary += f"• 🟢 BB OVERSOLD: {bb_percent*100:.1f}%\n"
                elif bb_percent > 0.8:
                    summary += f"• 🔴 BB OVERBOUGHT: {bb_percent*100:.1f}%\n"
                else:
                    summary += f"• ⚪ BB Position: {bb_percent*100:.1f}%\n"
            
            bb_width = get_indicator('BB_Width')
            if bb_width is not None:
                if bb_width < 5:
                    summary += f"• 📏 BB Squeeze: Width={bb_width:.1f}%\n"
                elif bb_width > 15:
                    summary += f"• 📈 BB Expansion: Width={bb_width:.1f}%\n"
            
            # ATR
            atr_pct = get_indicator('ATR_Pct')
            if atr_pct is not None:
                if atr_pct > 3:
                    summary += f"• ⚡ High Volatility: ATR={atr_pct:.1f}%\n"
                elif atr_pct < 1:
                    summary += f"• 🐌 Low Volatility: ATR={atr_pct:.1f}%\n"
            
            summary += f"""
📊 OSCILLATORS
"""
            
            # Awesome Oscillator
            ao = get_indicator('AO')
            if ao is not None:
                if ao > 0:
                    summary += f"• 🟢 Awesome Oscillator Bullish: {ao:.2f}\n"
                else:
                    summary += f"• 🔴 Awesome Oscillator Bearish: {ao:.2f}\n"
            
            # Chaikin Oscillator
            chaikin = get_indicator('Chaikin_Osc')
            if chaikin is not None:
                if chaikin > 0:
                    summary += f"• 🟢 Chaikin Oscillator Bullish: {chaikin:.0f}\n"
                else:
                    summary += f"• 🔴 Chaikin Oscillator Bearish: {chaikin:.0f}\n"
            
            # Support/Resistance
            resistance = get_indicator('Resistance_20')
            support = get_indicator('Support_20')
            if resistance is not None and support is not None:
                summary += f"""
📈 SUPPORT & RESISTANCE
• Resistance (20-day): {currency_symbol}{resistance:.2f}
• Support (20-day): {currency_symbol}{support:.2f}
• Current: {currency_symbol}{latest_close:.2f}
"""
            
            # Divergences
            if divergences['details']:
                summary += f"\n🔀 DIVERGENCES DETECTED\n"
                for detail in divergences['details']:
                    if 'Bullish' in detail:
                        summary += f"• 🟢 {detail}\n"
                    elif 'Bearish' in detail:
                        summary += f"• 🔴 {detail}\n"
                    else:
                        summary += f"• ⚪ {detail}\n"
            
            # Reversal patterns
            if reversal_patterns['bullish_reversal']:
                summary += f"\n🔄 BULLISH REVERSAL PATTERN\n"
                summary += f"• Confidence: {reversal_patterns['confidence']}%\n"
                for signal in reversal_patterns['signals']:
                    summary += f"• 🟢 {signal}\n"
            
            if reversal_patterns['bearish_reversal']:
                summary += f"\n🔄 BEARISH REVERSAL PATTERN\n"
                summary += f"• Confidence: {reversal_patterns['confidence']}%\n"
                for signal in reversal_patterns['signals']:
                    summary += f"• 🔴 {signal}\n"
            
            summary += f"""
📊 SIGNAL SUMMARY
• 🟢 Bullish Signals: {len(bull_signals)}
• 🔴 Bearish Signals: {len(bear_signals)}
• ⚪ Neutral Signals: {len(neutral_signals)}
"""
            
            # Overall recommendation
            strong_bull = len([s for s in signals if s['direction'] == 'BULLISH' and s['strength'] == 'STRONG'])
            strong_bear = len([s for s in signals if s['direction'] == 'BEARISH' and s['strength'] == 'STRONG'])
            
            if strong_bull > strong_bear + 2:
                recommendation = "🟢 STRONG BULLISH - Multiple strong bullish signals"
            elif strong_bear > strong_bull + 2:
                recommendation = "🔴 STRONG BEARISH - Multiple strong bearish signals"
            elif len(bull_signals) > len(bear_signals) + 5:
                recommendation = "🟢 BULLISH - Strong buying pressure"
            elif len(bear_signals) > len(bull_signals) + 5:
                recommendation = "🔴 BEARISH - Strong selling pressure"
            elif len(bull_signals) > len(bear_signals):
                recommendation = "🟢 MILD BULLISH - Slight edge to bulls"
            elif len(bear_signals) > len(bull_signals):
                recommendation = "🔴 MILD BEARISH - Slight edge to bears"
            else:
                recommendation = "⚪ NEUTRAL - Balanced market"
            
            summary += f"\n🎯 OVERALL RECOMMENDATION: {recommendation}"
            summary += f"\n\n⏰ Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            # Clean text for Telegram
            return self._clean_text_for_telegram(summary)
            
        except Exception as e:
            print(f"Summary creation error: {e}")
            import traceback
            traceback.print_exc()
            return f"📊 Complete Technical Analysis for {ticker}\n\nComprehensive analysis with all indicators."
    
    def _create_compact_summary(self, ticker: str, data: pd.DataFrame, info: Dict, 
                               signals: List[Dict], divergences: Dict) -> str:
        """Create compact summary for chart caption"""
        try:
            latest_close = float(data['Close'].iloc[-1])
            if len(data) > 1:
                prev_close_val = float(data['Close'].iloc[-2])
            else:
                prev_close_val = latest_close
            
            # Calculate daily change
            if prev_close_val != 0:
                daily_change = ((latest_close - prev_close_val) / prev_close_val * 100)
            else:
                daily_change = 0
            
            bull_signals = len([s for s in signals if s['direction'] == 'BULLISH'])
            bear_signals = len([s for s in signals if s['direction'] == 'BEARISH'])
            
            # Get key indicators
            rsi_value = None
            macd_value = None
            bb_percent = None
            vol_ratio = None
            
            if 'RSI' in data.columns:
                try:
                    rsi_value = float(data['RSI'].iloc[-1])
                except:
                    pass
            
            if 'MACD' in data.columns and 'MACD_Signal' in data.columns:
                try:
                    macd_value = float(data['MACD'].iloc[-1])
                    signal_value = float(data['MACD_Signal'].iloc[-1])
                    macd_status = "🟢" if macd_value > signal_value else "🔴"
                except:
                    macd_status = ""
            
            if 'BB_%B' in data.columns:
                try:
                    bb_percent = float(data['BB_%B'].iloc[-1])
                except:
                    pass
            
            if 'Volume_Ratio' in data.columns:
                try:
                    vol_ratio = float(data['Volume_Ratio'].iloc[-1])
                except:
                    pass
            
            # Determine overall sentiment
            if bull_signals > bear_signals + 5:
                sentiment = '🟢 STRONG BULLISH'
            elif bear_signals > bull_signals + 5:
                sentiment = '🔴 STRONG BEARISH'
            elif bull_signals > bear_signals:
                sentiment = '🟢 BULLISH'
            elif bear_signals > bull_signals:
                sentiment = '🔴 BEARISH'
            else:
                sentiment = '⚪ NEUTRAL'
            
            currency = info.get('currency', 'USD')
            currency_symbol = '$' if currency == 'USD' else '€' if currency == 'EUR' else '£' if currency == 'GBP' else f'{currency} '
            
            summary = f"📊 {ticker.upper()}\n"
            summary += f"💰 {currency_symbol}{latest_close:.2f} ({daily_change:+.2f}%)\n"
            
            if rsi_value is not None:
                if rsi_value < 30:
                    summary += f"📈 RSI: {rsi_value:.1f} 🟢\n"
                elif rsi_value > 70:
                    summary += f"📈 RSI: {rsi_value:.1f} 🔴\n"
                else:
                    summary += f"📈 RSI: {rsi_value:.1f} ⚪\n"
            
            if macd_status and macd_value is not None:
                summary += f"{macd_status} MACD: {macd_value:.4f}\n"
            elif macd_status:
                summary += f"{macd_status} MACD\n"
            
            if bb_percent is not None:
                if bb_percent < 0.2:
                    summary += f"📉 BB: {bb_percent*100:.1f}% 🟢\n"
                elif bb_percent > 0.8:
                    summary += f"📉 BB: {bb_percent*100:.1f}% 🔴\n"
                else:
                    summary += f"📉 BB: {bb_percent*100:.1f}% ⚪\n"
            
            if vol_ratio is not None:
                if vol_ratio > 1.5:
                    summary += f"📊 Vol: {vol_ratio:.1f}x 🔥\n"
                else:
                    summary += f"📊 Vol: {vol_ratio:.1f}x\n"
            
            if divergences['details']:
                summary += f"🔀 Div: {len(divergences['details'])}\n"
            
            summary += f"📶 🟢{bull_signals} | 🔴{bear_signals}\n"
            summary += f"🎯 {sentiment}"
            
            # Clean text for Telegram
            return self._clean_text_for_telegram(summary)
            
        except Exception as e:
            print(f"Compact summary error: {e}")
            return f"📊 {ticker.upper()} - Complete Technical Analysis"
    
    def _format_number(self, num: float) -> str:
        """Format large numbers for display"""
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