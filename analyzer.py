"""
Universal Trading Analyzer - Local Version
Simple yfinance data fetching without rate limiting
"""
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import warnings
import time
from datetime import datetime

warnings.filterwarnings('ignore')

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
    """Universal analyzer for ALL Yahoo Finance tickers - Local version"""
    
    def __init__(self):
        self.config = CONFIG
        self.cache = SimpleCache(
            ttl=self.config.CACHE_TTL,
            max_size=self.config.MAX_CACHE_SIZE
        )
        
        print("✅ Universal Analyzer initialized (Local Version)")
        print("🌍 Supports all markets and tickers")
    
    async def analyze_ticker(self, ticker_symbol: str, period: str = '1y') -> Dict:
        """Universal analysis method for any ticker"""
        try:
            print(f"🔍 Analyzing {ticker_symbol} ({period})")
            
            # Map period
            period_map = {
                '3m': '3mo', '6m': '6mo', '1y': '1y',
                '2y': '2y', '3y': '3y', '5y': '5y',
                'max': 'max'
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
                # Simple fetch for local use
                data = self._fetch_data_simple(formatted_ticker, yf_period)
                
                if data is None or data.empty:
                    return {
                        'success': False,
                        'error': f"No data found for {ticker_symbol}."
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
                'error': f"Analysis failed: {str(e)}"
            }
    
    def _format_ticker_for_yfinance(self, ticker: str) -> str:
        """Simple ticker formatting for yfinance - local version"""
        ticker = ticker.upper().strip().replace('$', '')
        return ticker  # Keep original, yfinance handles most
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare data for analysis"""
        if data.empty:
            return data
        
        df = data.copy()
        
        # Ensure single-level columns
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
    
    def _fetch_data_simple(self, ticker: str, period: str) -> Optional[pd.DataFrame]:
        """Simple data fetching for local use"""
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
            
            # Small delay to be polite
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
        """Get ticker information - simplified"""
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
            
            if len(data) > 0:
                latest = data.iloc[-1]
                info['last_price'] = float(latest['Close'])
                info['last_volume'] = float(latest.get('Volume', 0))
                info['last_open'] = float(latest.get('Open', latest['Close']))
                info['last_high'] = float(latest.get('High', latest['Close']))
                info['last_low'] = float(latest.get('Low', latest['Close']))
            
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
        """Calculate ALL technical indicators safely - Keep all indicators"""
        df = data.copy()
        
        if len(df) < 10:
            return df
        
        try:
            # Ensure all columns are 1D Series
            for col in ['Close', 'High', 'Low', 'Open', 'Volume']:
                if col in df.columns:
                    values = df[col].values
                    if values.ndim > 1:
                        values = values.flatten()
                    df[col] = pd.Series(values, index=df.index)
            
            # Ensure numeric
            for col in ['Close', 'High', 'Low', 'Open', 'Volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            print(f"📊 Data shape: {df.shape}")
            
            # Price transformations
            df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
            
            # Moving Averages
            for period in [20, 50, 200]:
                if len(df) >= period:
                    df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
                    df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
            
            # MACD
            if len(df) >= 26:
                exp1 = df['Close'].ewm(span=12, adjust=False).mean()
                exp2 = df['Close'].ewm(span=26, adjust=False).mean()
                df['MACD'] = exp1 - exp2
                df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
                df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
                df['MACD_Hist_Color'] = np.where(df['MACD_Hist'] >= 0, 'green', 'red')
            
            # RSI
            if len(df) >= 14:
                delta = df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                df['RSI'] = 100 - (100 / (1 + rs))
                df['RSI'] = df['RSI'].fillna(50)
            
            # Volume indicators
            if 'Volume' in df.columns:
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
                
                # A/D Line
                clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
                clv = clv.fillna(0)
                df['AD_Line'] = (clv * df['Volume']).cumsum()
                
                df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
                df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_20'].replace(0, 1)
                df['Volume_Ratio'] = df['Volume_Ratio'].fillna(1)
            
            # Bollinger Bands
            if len(df) >= 20:
                df['BB_Middle'] = df['Close'].rolling(20).mean()
                bb_std = df['Close'].rolling(20).std()
                df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
                df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
                df['BB_%B'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
                df['BB_%B'] = df['BB_%B'].fillna(0.5)
            
            # Performance metrics
            df['Daily_Return'] = df['Close'].pct_change().fillna(0)
            
            # Fill NaN values
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            if len(df) > 0:
                last_row = df.iloc[-1]
                print(f"📊 Last row indicators:")
                print(f"  Close: {last_row.get('Close', 0):.2f}")
                print(f"  RSI: {last_row.get('RSI', 0):.1f}")
                print(f"  MACD: {last_row.get('MACD', 0):.4f}")
                print(f"  Volume Ratio: {last_row.get('Volume_Ratio', 0):.2f}")
            
            indicator_count = len([col for col in df.columns if 'Unnamed' not in str(col)])
            print(f"✅ Calculated {indicator_count} indicators")
            
        except Exception as e:
            print(f"⚠️ Indicator calculation error: {e}")
        
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
            # Price vs Moving Averages
            if close is not None:
                for period in [20, 50, 200]:
                    sma = get_scalar(f'SMA_{period}', last_idx)
                    if sma is not None:
                        if close > sma:
                            signals.append({'type': f'ABOVE_{period}SMA', 'strength': 'MODERATE', 'direction': 'BULLISH'})
                        else:
                            signals.append({'type': f'BELOW_{period}SMA', 'strength': 'MODERATE', 'direction': 'BEARISH'})
            
            # === MOMENTUM SIGNALS ===
            # RSI
            rsi = get_scalar('RSI', last_idx)
            if rsi is not None:
                if rsi < 30:
                    signals.append({'type': 'RSI_OVERSOLD', 'strength': 'STRONG', 'direction': 'BULLISH'})
                elif rsi > 70:
                    signals.append({'type': 'RSI_OVERBOUGHT', 'strength': 'STRONG', 'direction': 'BEARISH'})
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
                elif prev_macd >= prev_signal and macd < signal:
                    signals.append({'type': 'MACD_BEARISH_CROSS', 'strength': 'MODERATE', 'direction': 'BEARISH'})
            
            # === VOLUME SIGNALS ===
            vol_ratio = get_scalar('Volume_Ratio', last_idx)
            if vol_ratio is not None:
                if vol_ratio > 2.0:
                    signals.append({'type': 'HIGH_VOLUME', 'strength': 'STRONG', 'direction': 'NEUTRAL'})
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
            
            print(f"✅ Generated {len(signals)} signals")
            
        except Exception as e:
            print(f"⚠️ Signal generation error: {e}")
        
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
            
            # MACD bullish crossover
            macd = get_scalar('MACD', last_idx)
            signal = get_scalar('MACD_Signal', last_idx)
            if macd is not None and signal is not None and macd > signal:
                if len(data) > 1:
                    prev_macd = get_scalar('MACD', last_idx-1)
                    prev_signal = get_scalar('MACD_Signal', last_idx-1)
                    if prev_macd is not None and prev_signal is not None and prev_macd <= prev_signal:
                        bullish_count += 1
                        bullish_details.append("MACD bullish crossover")
            
            # Determine bullish reversal
            if bullish_count >= 2:
                patterns['bullish_reversal'] = True
                patterns['confidence'] = min(95, bullish_count * 25)
                patterns['signals'] = bullish_details
                patterns['details']['bullish_count'] = bullish_count
            
            # === BEARISH REVERSAL PATTERNS ===
            bearish_count = 0
            bearish_details = []
            
            # RSI overbought
            if rsi is not None and rsi > 70:
                bearish_count += 1
                bearish_details.append(f"RSI overbought ({rsi:.1f})")
            
            # MACD bearish crossover
            if macd is not None and signal is not None and macd < signal:
                if len(data) > 1:
                    prev_macd = get_scalar('MACD', last_idx-1)
                    prev_signal = get_scalar('MACD_Signal', last_idx-1)
                    if prev_macd is not None and prev_signal is not None and prev_macd >= prev_signal:
                        bearish_count += 1
                        bearish_details.append("MACD bearish crossover")
            
            # Determine bearish reversal
            if bearish_count >= 2:
                patterns['bearish_reversal'] = True
                patterns['confidence'] = max(patterns['confidence'], min(95, bearish_count * 25))
                patterns['signals'].extend(bearish_details)
                patterns['details']['bearish_count'] = bearish_count
            
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
            'details': []
        }
        
        if len(data) < lookback + 5:
            return divergences
        
        try:
            recent = data.iloc[-lookback:].copy()
            
            # Get arrays
            price = recent['Close'].values
            rsi = recent['RSI'].values if 'RSI' in recent.columns else None
            
            # Simple RSI divergence detection
            if rsi is not None and len(price) >= 20:
                price_slice = price[-20:]
                rsi_slice = rsi[-20:]
                
                # Find minima and maxima
                price_min = price_slice.min()
                price_max = price_slice.max()
                rsi_min = rsi_slice.min()
                rsi_max = rsi_slice.max()
                
                # Bullish divergence: price makes lower low, RSI makes higher low
                if price_slice[-1] < price_slice[-10] and rsi_slice[-1] > rsi_slice[-10]:
                    divergences['bullish_rsi'] = True
                    divergences['details'].append("RSI Bullish Divergence")
                
                # Bearish divergence: price makes higher high, RSI makes lower high
                if price_slice[-1] > price_slice[-10] and rsi_slice[-1] < rsi_slice[-10]:
                    divergences['bearish_rsi'] = True
                    divergences['details'].append("RSI Bearish Divergence")
            
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
                    'max_drawdown': 0,
                    'win_rate': 0,
                }
            
            # Calculate win rate
            win_count = (returns > 0).sum()
            if isinstance(win_count, pd.Series):
                win_count = win_count.iloc[0] if len(win_count) > 0 else 0
            
            stats = {
                'total_return': float(((close_series.iloc[-1] / close_series.iloc[0]) - 1) * 100),
                'avg_daily_return': float(returns.mean() * 100),
                'volatility': float(returns.std() * np.sqrt(252) * 100),
                'max_drawdown': float(self._calculate_max_drawdown(close_series)),
                'win_rate': float(win_count / len(returns) * 100),
            }
            
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
            
            # Calculate daily change
            if prev_close_val != 0:
                daily_change = ((latest_close - prev_close_val) / prev_close_val * 100)
            else:
                daily_change = 0
            
            latest_volume = float(data['Volume'].iloc[-1]) if 'Volume' in data.columns else 0
            
            bull_signals = [s for s in signals if s['direction'] == 'BULLISH']
            bear_signals = [s for s in signals if s['direction'] == 'BEARISH']
            neutral_signals = [s for s in signals if s['direction'] == 'NEUTRAL']
            
            currency = info.get('currency', 'USD')
            currency_symbol = '$' if currency == 'USD' else '€' if currency == 'EUR' else '£' if currency == 'GBP' else f'{currency} '
            
            # Color codes - simplified
            green_dot = "🟢"
            red_dot = "🔴"
            yellow_dot = "🟡"
            white_dot = "⚪"
            
            summary = f"""
📊 {green_dot} TECHNICAL ANALYSIS: {ticker.upper()} {green_dot}

📈 MARKET INFORMATION
• Exchange: {exchange_info['exchange']}
• Currency: {currency}
• Last Price: {currency_symbol}{latest_close:.2f} ({daily_change:+.2f}%)
• Volume: {self._format_number(latest_volume)} shares

📊 PERFORMANCE STATISTICS
• Total Return: {stats.get('total_return', 0):.2f}%
• Avg Daily Return: {stats.get('avg_daily_return', 0):.3f}%
• Win Rate: {stats.get('win_rate', 0):.1f}%
• Volatility: {stats.get('volatility', 0):.1f}%
• Max Drawdown: {stats.get('max_drawdown', 0):.1f}%

{yellow_dot} KEY INDICATORS {yellow_dot}
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
                    status = f"{green_dot} OVERSOLD"
                elif rsi > 70:
                    status = f"{red_dot} OVERBOUGHT"
                elif rsi < 50:
                    status = f"{yellow_dot} BEARISH"
                else:
                    status = f"{white_dot} BULLISH"
                summary += f"• {status} RSI: {rsi:.1f}\n"
            
            # MACD
            macd = get_indicator('MACD')
            signal = get_indicator('MACD_Signal')
            if macd is not None and signal is not None:
                if macd > signal:
                    trend = f"{green_dot} BULLISH"
                else:
                    trend = f"{red_dot} BEARISH"
                summary += f"• {trend} MACD: {macd:.4f} | Signal: {signal:.4f}\n"
            
            # Moving averages
            for period in [20, 50, 200]:
                sma = get_indicator(f'SMA_{period}')
                if sma is not None:
                    if latest_close > sma:
                        position = f"{green_dot} ABOVE"
                    else:
                        position = f"{red_dot} BELOW"
                    distance = abs(latest_close - sma) / sma * 100 if sma != 0 else 0
                    summary += f"• {position} {period}-day MA: {currency_symbol}{sma:.2f} ({distance:.1f}% away)\n"
            
            # Volume
            vol_ratio = get_indicator('Volume_Ratio')
            if vol_ratio is not None:
                if vol_ratio > 1.5:
                    vol_status = f"{yellow_dot} HIGH"
                elif vol_ratio < 0.5:
                    vol_status = f"{yellow_dot} LOW"
                else:
                    vol_status = f"{white_dot} NORMAL"
                summary += f"• {vol_status} Volume: {vol_ratio:.1f}x average\n"
            
            # A/D Line
            ad_line = get_indicator('AD_Line')
            if ad_line is not None and len(data) > 1:
                prev_ad = get_indicator('AD_Line')
                if prev_ad is not None and ad_line > prev_ad:
                    ad_status = f"{green_dot} RISING"
                elif prev_ad is not None and ad_line < prev_ad:
                    ad_status = f"{red_dot} FALLING"
                else:
                    ad_status = f"{white_dot} FLAT"
                summary += f"• {ad_status} A/D Line: {ad_line:.0f}\n"
            
            # Divergences
            if divergences['details']:
                summary += f"\n{yellow_dot} DIVERGENCES DETECTED {yellow_dot}\n"
                for detail in divergences['details'][:3]:
                    if 'Bullish' in detail:
                        summary += f"• {green_dot} {detail}\n"
                    elif 'Bearish' in detail:
                        summary += f"• {red_dot} {detail}\n"
            
            # Reversal patterns
            if reversal_patterns['bullish_reversal']:
                summary += f"\n{green_dot} BULLISH REVERSAL PATTERN {green_dot}\n"
                summary += f"• Confidence: {reversal_patterns['confidence']}%\n"
                for signal in reversal_patterns['signals'][:3]:
                    summary += f"• {green_dot} {signal}\n"
            
            if reversal_patterns['bearish_reversal']:
                summary += f"\n{red_dot} BEARISH REVERSAL PATTERN {red_dot}\n"
                summary += f"• Confidence: {reversal_patterns['confidence']}%\n"
                for signal in reversal_patterns['signals'][:3]:
                    summary += f"• {red_dot} {signal}\n"
            
            summary += f"\n{yellow_dot} SIGNAL SUMMARY {yellow_dot}\n"
            summary += f"• {green_dot} Bullish: {len(bull_signals)}\n"
            summary += f"• {red_dot} Bearish: {len(bear_signals)}\n"
            summary += f"• {white_dot} Neutral: {len(neutral_signals)}\n"
            
            # Overall recommendation
            if len(bull_signals) > len(bear_signals) + 3:
                recommendation = f"{green_dot} BULLISH"
            elif len(bear_signals) > len(bull_signals) + 3:
                recommendation = f"{red_dot} BEARISH"
            elif len(bull_signals) > len(bear_signals):
                recommendation = f"{green_dot} MILD BULLISH"
            elif len(bear_signals) > len(bull_signals):
                recommendation = f"{red_dot} MILD BEARISH"
            else:
                recommendation = f"{white_dot} NEUTRAL"
            
            summary += f"\n{yellow_dot} OVERALL: {recommendation}"
            summary += f"\n\n⏰ Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            return summary
            
        except Exception as e:
            print(f"Summary creation error: {e}")
            return f"📊 Analysis for {ticker}\n\nComplete technical analysis generated."
    
    def _create_compact_summary(self, ticker: str, data: pd.DataFrame, info: Dict, 
                               signals: List[Dict], divergences: Dict) -> str:
        """Create compact summary for photo captions"""
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
            
            # Get RSI value
            rsi_value = None
            if 'RSI' in data.columns:
                try:
                    rsi_value = float(data['RSI'].iloc[-1])
                except:
                    rsi_value = None
            
            # Get MACD position
            macd_status = ""
            if 'MACD' in data.columns and 'MACD_Signal' in data.columns:
                try:
                    macd = float(data['MACD'].iloc[-1])
                    signal = float(data['MACD_Signal'].iloc[-1])
                    if macd > signal:
                        macd_status = "🟢 MACD ↑"
                    else:
                        macd_status = "🔴 MACD ↓"
                except:
                    macd_status = ""
            
            # Get volume ratio
            vol_ratio = ""
            if 'Volume_Ratio' in data.columns:
                try:
                    vol_ratio_val = float(data['Volume_Ratio'].iloc[-1])
                    vol_ratio = f"📊 {vol_ratio_val:.1f}x vol"
                except:
                    vol_ratio = ""
            
            # Determine overall sentiment
            if bull_signals > bear_signals + 2:
                sentiment = '🟢 BULLISH'
            elif bear_signals > bull_signals + 2:
                sentiment = '🔴 BEARISH'
            elif bull_signals > bear_signals:
                sentiment = '🟢 MILD BULLISH'
            elif bear_signals > bull_signals:
                sentiment = '🔴 MILD BEARISH'
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
            
            if macd_status:
                summary += f"{macd_status}\n"
            
            if vol_ratio:
                summary += f"{vol_ratio}\n"
            
            summary += f"📶 🟢{bull_signals} | 🔴{bear_signals}\n"
            summary += f"🎯 {sentiment}"
            
            return summary
            
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