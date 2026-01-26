import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List
import warnings
import time
import asyncio
import os

warnings.filterwarnings('ignore')

# Apply yfinance fixes before importing other modules
import yfinance_fix
yfinance_fix.apply_yfinance_fix()

# Using pure Python TA library instead of compiled talib
import ta

from config import CONFIG

class TradingAnalyzer:
    """Comprehensive analyzer for technical analysis with reversal detection"""
    
    def __init__(self):
        self.config = CONFIG
    
    async def analyze_ticker(self, ticker_symbol: str, period: str = '1y') -> Dict:
        """Perform comprehensive analysis of a ticker for specified period"""
        try:
            print(f"🔍 Starting analysis for {ticker_symbol} ({period})")
            
            # Clear cache at the start
            cache_dir = "/tmp/yfinance_cache"
            if os.path.exists(cache_dir):
                import shutil
                shutil.rmtree(cache_dir, ignore_errors=True)
            
            yf.set_tz_cache_location(cache_dir)
            
            # Map period
            yf_period = self.config.TIME_PERIODS.get(period, period)
            
            # Try multiple methods to get data
            data = await self._fetch_data_with_retry(ticker_symbol, yf_period)
            
            if data is None or data.empty:
                return {
                    'success': False, 
                    'error': f'No data found for {ticker_symbol} (period={period})'
                }
            
            print(f"✅ Data fetched: {len(data)} rows")
            
            # Flatten MultiIndex columns
            data = self._flatten_dataframe(data)
            
            # Get ticker info
            info = await self._get_ticker_info(ticker_symbol, data)
            
            # Calculate technical indicators
            data = self._calculate_indicators(data)
            
            # Get fundamental analysis
            fundamental = self._analyze_fundamentals(info)
            
            # Generate signals
            signals = self._generate_signals(data)
            
            # Detect reversal patterns
            reversal_patterns = self._detect_reversal_patterns(data)
            
            # Prepare analysis summary
            summary = self._prepare_summary(ticker_symbol, data, info, fundamental, signals, reversal_patterns)
            
            # Prepare technical overview
            tech_overview = self._prepare_technical_overview(data, reversal_patterns)
            
            print(f"✅ Analysis complete for {ticker_symbol}")
            
            return {
                'success': True,
                'data': data,
                'info': info,
                'fundamental': fundamental,
                'signals': signals,
                'reversal_patterns': reversal_patterns,
                'summary': summary,
                'technical_overview': tech_overview,
                'requested_period': period
            }
            
        except Exception as e:
            print(f"❌ Error in analyze_ticker: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}
    
    async def _fetch_data_with_retry(self, ticker_symbol: str, yf_period: str) -> pd.DataFrame:
        """Fetch data with multiple retries and fallbacks"""
        max_retries = 5
        
        for attempt in range(max_retries):
            try:
                print(f"Attempt {attempt + 1} to fetch {ticker_symbol}")
                
                # Method 1: Use Ticker.history (most reliable)
                try:
                    ticker = yf.Ticker(ticker_symbol)
                    
                    # Small delay before request
                    if attempt > 0:
                        time.sleep(1 + attempt)
                    
                    # Get historical data
                    data = ticker.history(period=yf_period, interval="1d")
                    
                    if data is not None and not data.empty:
                        print(f"✅ Data obtained via Ticker.history ({len(data)} rows)")
                        return data
                except Exception as e:
                    print(f"Ticker.history failed: {str(e)}")
                
                # Method 2: Use yf.download as fallback
                if attempt >= 2:  # Try download after 2 failures
                    print(f"Trying yf.download for {ticker_symbol}")
                    time.sleep(2)
                    
                    data = yf.download(
                        ticker_symbol,
                        period=yf_period,
                        progress=False,
                        threads=False,
                        ignore_tz=True
                    )
                    
                    if data is not None and not data.empty:
                        print(f"✅ Data obtained via yf.download ({len(data)} rows)")
                        return data
                
            except Exception as e:
                print(f"Fetch attempt {attempt + 1} failed: {str(e)}")
                
                if attempt == max_retries - 1:
                    print("All data fetch methods failed")
                    raise e
                
                # Exponential backoff
                time.sleep(2 ** attempt)
        
        return None
    
    async def _get_ticker_info(self, ticker_symbol: str, data: pd.DataFrame) -> Dict:
        """Get ticker information with fallback"""
        try:
            ticker = yf.Ticker(ticker_symbol)
            info = ticker.info
            
            # Validate info
            if not info or 'symbol' not in info:
                raise ValueError("Invalid ticker info")
            
            return info
            
        except Exception as e:
            print(f"Warning: Could not get ticker info: {e}")
            # Return basic info
            return {
                'symbol': ticker_symbol,
                'longName': ticker_symbol,
                'shortName': ticker_symbol,
                'regularMarketPrice': data['Close'].iloc[-1] if len(data) > 0 else 0,
                'currency': 'USD'
            }
        def _fetch_with_proxy_fallback(self, ticker_symbol: str, period: str):
            """Try to fetch data with proxy if direct connection fails"""
            import requests
            import json
            
            # List of free proxy servers (rotating)
            proxies_list = [
                None,  # Try direct first
                # Add more proxies if needed
            ]
            
            for proxy in proxies_list:
                try:
                    print(f"Trying with proxy: {proxy}")
                    
                    # Create session
                    session = requests.Session()
                    if proxy:
                        session.proxies = {"http": proxy, "https": proxy}
                    
                    # Custom headers
                    session.headers.update({
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                        'Accept': 'application/json',
                    })
                    
                    # Fetch using yfinance with custom session
                    import yfinance as yf
                    data = yf.download(
                        ticker_symbol,
                        period=period,
                        progress=False,
                        threads=False,
                        session=session
                    )
                    
                    if not data.empty:
                        return data
                        
                except Exception as e:
                    print(f"Proxy attempt failed: {e}")
                    continue
            
            return None        
    
    def _flatten_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flatten MultiIndex columns to simple column names"""
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    
    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators using pure Python TA library"""
        df = data.copy()
        
        # Ensure required columns
        required_columns = ['Close', 'High', 'Low', 'Open', 'Volume']
        for col in required_columns:
            if col not in df.columns:
                print(f"Warning: Missing required column: {col}")
                return df
        
        # Calculate moving averages
        for ma in self.config.MOVING_AVERAGES:
            if len(df) >= ma:
                # SMA (Simple Moving Average)
                try:
                    sma_indicator = ta.trend.SMAIndicator(df['Close'], window=ma)
                    df[f'SMA_{ma}'] = sma_indicator.sma_indicator()
                except Exception as e:
                    print(f"Error calculating SMA_{ma}: {e}")
                    df[f'SMA_{ma}'] = np.nan
                
                # EMA for 9-period only
                if ma == 9:
                    try:
                        ema_indicator = ta.trend.EMAIndicator(df['Close'], window=ma)
                        df['EMA_9'] = ema_indicator.ema_indicator()
                    except Exception as e:
                        print(f"Error calculating EMA_9: {e}")
                        df['EMA_9'] = np.nan
        
        # Calculate RSI (Relative Strength Index)
        if len(df) >= self.config.RSI_PERIOD:
            try:
                rsi_indicator = ta.momentum.RSIIndicator(df['Close'], window=self.config.RSI_PERIOD)
                df['RSI'] = rsi_indicator.rsi()
            except Exception as e:
                print(f"Error calculating RSI: {e}")
                df['RSI'] = np.nan
        
        # Calculate MACD
        if len(df) >= self.config.MACD_SLOW:
            try:
                macd_indicator = ta.trend.MACD(
                    df['Close'],
                    window_slow=self.config.MACD_SLOW,
                    window_fast=self.config.MACD_FAST,
                    window_sign=self.config.MACD_SIGNAL
                )
                df['MACD'] = macd_indicator.macd()
                df['MACD_Signal'] = macd_indicator.macd_signal()
                df['MACD_Hist'] = macd_indicator.macd_diff()
            except Exception as e:
                print(f"Error calculating MACD: {e}")
                df['MACD'] = np.nan
                df['MACD_Signal'] = np.nan
                df['MACD_Hist'] = np.nan
        
        # Calculate Bollinger Bands
        if len(df) >= self.config.BB_PERIOD:
            try:
                bb_indicator = ta.volatility.BollingerBands(
                    df['Close'],
                    window=self.config.BB_PERIOD,
                    window_dev=self.config.BB_STD
                )
                df['BB_Upper'] = bb_indicator.bollinger_hband()
                df['BB_Middle'] = bb_indicator.bollinger_mavg()
                df['BB_Lower'] = bb_indicator.bollinger_lband()
            except Exception as e:
                print(f"Error calculating Bollinger Bands: {e}")
                df['BB_Upper'] = np.nan
                df['BB_Middle'] = np.nan
                df['BB_Lower'] = np.nan
        
        # Calculate A/D Line
        try:
            hl_range = df['High'] - df['Low']
            hl_range = hl_range.replace(0, 0.000001)
            mfm = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / hl_range
            mfv = mfm * df['Volume']
            df['AD_Line'] = mfv.cumsum()
        except Exception as e:
            print(f"Error calculating A/D Line: {e}")
            df['AD_Line'] = 0
        
        # Calculate volume moving average
        try:
            df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        except Exception as e:
            print(f"Error calculating volume indicators: {e}")
            df['Volume_MA'] = np.nan
            df['Volume_Ratio'] = np.nan
        
        return df
    
    def _detect_reversal_patterns(self, data: pd.DataFrame) -> Dict:
        """Detect specific reversal patterns in the data"""
        patterns = {
            'bullish_reversal': False,
            'bearish_reversal': False,
            'confidence': 0,
            'signals': [],
            'details': {}
        }
        
        if len(data) < 10:
            return patterns
        
        try:
            # Get recent data (last 10 periods)
            recent = data.tail(10).copy()
            latest = recent.iloc[-1]
            
            # Check for bullish reversal pattern
            bullish_signals = []
            
            # 1. Price touching lower Bollinger Band
            if 'BB_Lower' in recent.columns and 'BB_Upper' in recent.columns:
                try:
                    bb_position = (latest['Close'] - latest['BB_Lower']) / (latest['BB_Upper'] - latest['BB_Lower'])
                    if bb_position < 0.1:  # Price in lower 10% of BB
                        bullish_signals.append(f"Price near lower Bollinger Band ({bb_position*100:.1f}% position)")
                        patterns['details']['bb_position'] = bb_position
                except:
                    pass
            
            # 2. A/D Line bullish divergence
            if 'AD_Line' in recent.columns and len(recent) >= 5:
                try:
                    price_min_idx = recent['Close'].tail(5).idxmin()
                    ad_at_price_min = recent.loc[price_min_idx, 'AD_Line']
                    ad_current = latest['AD_Line']
                    
                    if ad_current > ad_at_price_min:
                        bullish_signals.append("A/D Line showing bullish divergence")
                        patterns['details']['ad_divergence'] = True
                except:
                    pass
            
            # 3. RSI oversold with bullish divergence
            if 'RSI' in recent.columns and len(recent) >= 5:
                try:
                    rsi_current = latest['RSI']
                    if pd.isna(rsi_current):
                        pass
                    else:
                        rsi_oversold = rsi_current < self.config.REVERSAL_SETTINGS['rsi_oversold']
                        
                        price_min_idx = recent['Close'].tail(5).idxmin()
                        rsi_at_price_min = recent.loc[price_min_idx, 'RSI']
                        
                        if rsi_oversold and rsi_current > rsi_at_price_min:
                            bullish_signals.append(f"RSI oversold ({rsi_current:.1f}) with bullish divergence")
                            patterns['details']['rsi_divergence'] = True
                            patterns['details']['rsi_value'] = rsi_current
                except:
                    pass
            
            # 4. MACD histogram showing shrinking negative bars
            if 'MACD_Hist' in recent.columns and len(recent) >= 3:
                try:
                    hist_values = recent['MACD_Hist'].tail(3).values
                    if all(not pd.isna(h) and h < 0 for h in hist_values) and hist_values[-1] > hist_values[0]:
                        bullish_signals.append("MACD histogram showing shrinking negative bars")
                        patterns['details']['macd_hist_shrinking'] = True
                except:
                    pass
            
            # 5. Volume spike on low day
            if 'Volume_Ratio' in recent.columns:
                try:
                    min_price_idx = recent['Close'].idxmin()
                    volume_ratio_at_low = recent.loc[min_price_idx, 'Volume_Ratio']
                    
                    if not pd.isna(volume_ratio_at_low) and volume_ratio_at_low > self.config.REVERSAL_SETTINGS['volume_spike_threshold']:
                        bullish_signals.append(f"Volume spike ({volume_ratio_at_low:.1f}x) on low day")
                        patterns['details']['volume_spike'] = volume_ratio_at_low
                except:
                    pass
            
            # Determine if bullish reversal pattern is present
            if len(bullish_signals) >= 3:
                patterns['bullish_reversal'] = True
                patterns['confidence'] = min(100, len(bullish_signals) * 20)
                patterns['signals'] = bullish_signals
            
            # Check for bearish reversal pattern
            bearish_signals = []
            
            if 'RSI' in recent.columns:
                try:
                    rsi_value = latest['RSI']
                    if not pd.isna(rsi_value) and rsi_value > self.config.REVERSAL_SETTINGS['rsi_overbought']:
                        bearish_signals.append(f"RSI overbought ({rsi_value:.1f})")
                except:
                    pass
            
            if 'BB_Upper' in recent.columns:
                try:
                    bb_position = (latest['Close'] - latest['BB_Lower']) / (latest['BB_Upper'] - latest['BB_Lower'])
                    if bb_position > 0.9:
                        bearish_signals.append(f"Price near upper Bollinger Band ({bb_position*100:.1f}% position)")
                except:
                    pass
            
            if 'MACD' in recent.columns and 'MACD_Signal' in recent.columns:
                try:
                    macd_val = latest['MACD']
                    signal_val = latest['MACD_Signal']
                    if not pd.isna(macd_val) and not pd.isna(signal_val) and macd_val < signal_val:
                        bearish_signals.append("MACD bearish crossover")
                except:
                    pass
            
            if len(bearish_signals) >= 2:
                patterns['bearish_reversal'] = True
                patterns['confidence'] = max(patterns['confidence'], len(bearish_signals) * 25)
                patterns['signals'].extend(bearish_signals)
        
        except Exception as e:
            print(f"Error detecting reversal patterns: {e}")
        
        return patterns
    
    def _analyze_fundamentals(self, info: Dict) -> Dict:
        """Analyze fundamental data"""
        fundamental = {
            'valuation': {},
            'profitability': {},
            'growth': {},
            'financial_health': {},
            'dividends': {},
            'score': 50
        }
        
        try:
            # Valuation metrics
            fundamental['valuation'] = {
                'pe_ratio': info.get('trailingPE', np.nan),
                'forward_pe': info.get('forwardPE', np.nan),
                'price_to_book': info.get('priceToBook', np.nan),
                'price_to_sales': info.get('priceToSalesTrailing12Months', np.nan),
                'peg_ratio': info.get('pegRatio', np.nan)
            }
            
            # Profitability
            fundamental['profitability'] = {
                'roe': info.get('returnOnEquity', np.nan),
                'roa': info.get('returnOnAssets', np.nan),
                'profit_margin': info.get('profitMargins', np.nan),
                'operating_margin': info.get('operatingMargins', np.nan)
            }
            
            # Growth
            fundamental['growth'] = {
                'revenue_growth': info.get('revenueGrowth', np.nan),
                'eps_growth': info.get('earningsQuarterlyGrowth', np.nan),
                'five_year_growth': info.get('earningsGrowth', np.nan)
            }
            
            # Financial Health
            fundamental['financial_health'] = {
                'debt_to_equity': info.get('debtToEquity', np.nan),
                'current_ratio': info.get('currentRatio', np.nan),
                'quick_ratio': info.get('quickRatio', np.nan)
            }
            
            # Dividends
            fundamental['dividends'] = {
                'dividend_yield': info.get('dividendYield', np.nan),
                'dividend_rate': info.get('discountRate', np.nan),  # Changed from dividendRate
                'payout_ratio': info.get('payoutRatio', np.nan)
            }
            
            # Calculate fundamental score
            fundamental['score'] = self._calculate_fundamental_score(fundamental)
            
        except Exception as e:
            print(f"Error in fundamental analysis: {e}")
        
        return fundamental
    
    def _calculate_fundamental_score(self, fundamental: Dict) -> int:
        """Calculate fundamental score (0-100)"""
        score = 50
        
        try:
            # Valuation (max 25 points)
            pe = fundamental['valuation'].get('pe_ratio')
            if pe and not np.isnan(pe) and 0 < pe < 20:
                score += 15
            elif pe and not np.isnan(pe) and 0 < pe < 30:
                score += 10
            
            # Profitability (max 25 points)
            roe = fundamental['profitability'].get('roe')
            if roe and not np.isnan(roe) and roe > 0.15:
                score += 15
            elif roe and not np.isnan(roe) and roe > 0.10:
                score += 10
            
            # Financial Health (max 20 points)
            debt_equity = fundamental['financial_health'].get('debt_to_equity')
            if debt_equity and not np.isnan(debt_equity) and debt_equity < 1.0:
                score += 10
            elif debt_equity and not np.isnan(debt_equity) and debt_equity < 2.0:
                score += 5
            
            # Growth (max 15 points)
            revenue_growth = fundamental['growth'].get('revenue_growth')
            if revenue_growth and not np.isnan(revenue_growth) and revenue_growth > 0.1:
                score += 10
            elif revenue_growth and not np.isnan(revenue_growth) and revenue_growth > 0.05:
                score += 5
            
            # Dividends (max 15 points)
            dividend_yield = fundamental['dividends'].get('dividend_yield')
            if dividend_yield and not np.isnan(dividend_yield) and dividend_yield > 0.03:
                score += 10
            elif dividend_yield and not np.isnan(dividend_yield) and dividend_yield > 0.01:
                score += 5
            
        except Exception as e:
            print(f"Error calculating fundamental score: {e}")
        
        return max(0, min(100, score))
    
    def _generate_signals(self, data: pd.DataFrame) -> List[Dict]:
        """Generate trading signals"""
        signals = []
        
        if len(data) < 10:
            return signals
        
        try:
            latest = data.iloc[-1]
            
            # RSI signals
            if 'RSI' in data.columns:
                rsi = latest['RSI']
                if not pd.isna(rsi):
                    if rsi < 30:
                        signals.append({'type': 'RSI_OVERSOLD', 'strength': 'STRONG', 'direction': 'BULLISH'})
                    elif rsi > 70:
                        signals.append({'type': 'RSI_OVERBOUGHT', 'strength': 'STRONG', 'direction': 'BEARISH'})
                    elif rsi < 35:
                        signals.append({'type': 'RSI_NEAR_OVERSOLD', 'strength': 'MODERATE', 'direction': 'BULLISH'})
                    elif rsi > 65:
                        signals.append({'type': 'RSI_NEAR_OVERBOUGHT', 'strength': 'MODERATE', 'direction': 'BEARISH'})
            
            # MACD signals
            if 'MACD' in data.columns and 'MACD_Signal' in data.columns:
                macd = latest['MACD']
                macd_signal = latest['MACD_Signal']
                
                if not pd.isna(macd) and not pd.isna(macd_signal):
                    if macd > macd_signal:
                        signals.append({'type': 'MACD_BULLISH', 'strength': 'MODERATE', 'direction': 'BULLISH'})
                    else:
                        signals.append({'type': 'MACD_BEARISH', 'strength': 'MODERATE', 'direction': 'BEARISH'})
            
            # Moving average signals
            if 'Close' in data.columns:
                close = latest['Close']
                
                # Check MA crossovers
                if 'SMA_20' in data.columns and 'SMA_50' in data.columns:
                    sma20 = latest['SMA_20']
                    sma50 = latest['SMA_50']
                    
                    if not pd.isna(sma20) and not pd.isna(sma50):
                        if sma20 > sma50:
                            signals.append({'type': 'GOLDEN_CROSS', 'strength': 'STRONG', 'direction': 'BULLISH'})
                        elif sma20 < sma50:
                            signals.append({'type': 'DEATH_CROSS', 'strength': 'STRONG', 'direction': 'BEARISH'})
                
                # Price position relative to 50 MA
                if 'SMA_50' in data.columns:
                    sma50 = latest['SMA_50']
                    if not pd.isna(sma50):
                        if close > sma50:
                            signals.append({'type': 'ABOVE_50MA', 'strength': 'MODERATE', 'direction': 'BULLISH'})
                        elif close < sma50:
                            signals.append({'type': 'BELOW_50MA', 'strength': 'MODERATE', 'direction': 'BEARISH'})
            
            # Bollinger Bands signals
            if 'Close' in data.columns and 'BB_Upper' in data.columns and 'BB_Lower' in data.columns:
                close = latest['Close']
                bb_upper = latest['BB_Upper']
                bb_lower = latest['BB_Lower']
                
                if not pd.isna(bb_upper) and not pd.isna(bb_lower):
                    if close < bb_lower:
                        signals.append({'type': 'BB_OVERSOLD', 'strength': 'STRONG', 'direction': 'BULLISH'})
                    elif close > bb_upper:
                        signals.append({'type': 'BB_OVERBOUGHT', 'strength': 'STRONG', 'direction': 'BEARISH'})
            
            # Volume signals
            if 'Volume_Ratio' in data.columns:
                volume_ratio = latest['Volume_Ratio']
                if not pd.isna(volume_ratio) and volume_ratio > 1.5:
                    signals.append({'type': 'HIGH_VOLUME', 'strength': 'MODERATE', 'direction': 'NEUTRAL'})
        
        except Exception as e:
            print(f"Error generating signals: {e}")
        
        return signals
    
    def _prepare_summary(self, ticker: str, data: pd.DataFrame, info: Dict, 
                        fundamental: Dict, signals: List, reversal_patterns: Dict) -> str:
        """Prepare analysis summary"""
        try:
            latest = data.iloc[-1]
            
            # Price information
            current_price = latest['Close']
            prev_close = data['Close'].iloc[-2] if len(data) > 1 else current_price
            price_change = ((current_price - prev_close) / prev_close * 100) if prev_close != 0 else 0
            
            # Volume
            volume = latest['Volume']
            avg_volume = data['Volume'].tail(20).mean()
            volume_ratio = volume / avg_volume if avg_volume > 0 else 1
            
            # Signal summary
            bull_signals = len([s for s in signals if s['direction'] == 'BULLISH'])
            bear_signals = len([s for s in signals if s['direction'] == 'BEARISH'])
            
            summary = f"""
📊 **ANALYSIS: {ticker}**

**Price Information:**
• Current: ${current_price:.2f}
• Change: {price_change:+.2f}%
• Volume: {self._format_number(volume)} ({volume_ratio:.1f}x avg)

**Key Technical Levels:**
"""
            
            # Add moving averages
            for ma in self.config.MOVING_AVERAGES:
                ma_col = f'SMA_{ma}'
                if ma_col in data.columns:
                    ma_val = latest[ma_col]
                    if not pd.isna(ma_val):
                        relation = "ABOVE" if current_price > ma_val else "BELOW"
                        distance_pct = abs(current_price - ma_val) / ma_val * 100 if ma_val != 0 else 0
                        summary += f"• {ma}MA: ${ma_val:.2f} ({relation}, {distance_pct:.1f}%)\n"
            
            # Add RSI
            if 'RSI' in data.columns:
                rsi_val = latest['RSI']
                if not pd.isna(rsi_val):
                    rsi_status = 'OVERSOLD' if rsi_val < 30 else 'OVERBOUGHT' if rsi_val > 70 else 'NEUTRAL'
                    summary += f"• RSI: {rsi_val:.1f} ({rsi_status})\n"
            
            # Add MACD
            if 'MACD' in data.columns and 'MACD_Signal' in data.columns:
                macd_val = latest['MACD']
                signal_val = latest['MACD_Signal']
                if not pd.isna(macd_val) and not pd.isna(signal_val):
                    macd_dir = '🟢 BULLISH' if macd_val > signal_val else '🔴 BEARISH'
                    summary += f"• MACD: {macd_dir}\n"
            
            # Add Bollinger Bands position
            if 'BB_Upper' in data.columns and 'BB_Lower' in data.columns:
                try:
                    bb_position = (latest['Close'] - latest['BB_Lower']) / (latest['BB_Upper'] - latest['BB_Lower'])
                    if not pd.isna(bb_position):
                        summary += f"• BB Position: {bb_position*100:.1f}%\n"
                except:
                    pass
            
            # Add reversal patterns if detected
            if reversal_patterns['bullish_reversal'] or reversal_patterns['bearish_reversal']:
                summary += "\n**⚠️ REVERSAL PATTERN DETECTED ⚠️**\n"
                
                if reversal_patterns['bullish_reversal']:
                    summary += "• Type: 🟢 BULLISH REVERSAL\n"
                    summary += f"• Confidence: {reversal_patterns['confidence']}%\n"
                    if reversal_patterns['signals']:
                        summary += "• Signals:\n"
                        for signal in reversal_patterns['signals'][:5]:
                            summary += f"  ◦ {signal}\n"
                
                if reversal_patterns['bearish_reversal']:
                    summary += "• Type: 🔴 BEARISH REVERSAL\n"
                    summary += f"• Confidence: {reversal_patterns['confidence']}%\n"
            
            summary += f"""
**Technical Signals:** {len(signals)} total
• Bullish: {bull_signals} • Bearish: {bear_signals}

**Fundamental Score:** {fundamental['score']}/100
"""
            
            # Overall sentiment
            if reversal_patterns['bullish_reversal']:
                sentiment = '🟢 STRONG BULLISH REVERSAL'
            elif reversal_patterns['bearish_reversal']:
                sentiment = '🔴 STRONG BEARISH REVERSAL'
            elif bull_signals > bear_signals:
                sentiment = '🟢 BULLISH'
            elif bear_signals > bull_signals:
                sentiment = '🔴 BEARISH'
            else:
                sentiment = '⚪ NEUTRAL'
            
            summary += f"**Overall Sentiment:** {sentiment}"
            
            return summary
            
        except Exception as e:
            print(f"Error preparing summary: {e}")
            return f"**ANALYSIS: {ticker}**\n\nError preparing analysis summary."
    
    def _prepare_technical_overview(self, data: pd.DataFrame, reversal_patterns: Dict) -> str:
        """Prepare detailed technical overview"""
        try:
            latest = data.iloc[-1]
            
            overview = "📈 **TECHNICAL OVERVIEW**\n\n"
            
            # Trend Analysis
            overview += "**Trend Analysis:**\n"
            
            if 'SMA_20' in data.columns and 'SMA_50' in data.columns:
                sma20 = latest['SMA_20']
                sma50 = latest['SMA_50']
                close = latest['Close']
                
                if not pd.isna(sma20) and not pd.isna(sma50):
                    # Check MA alignment
                    if sma20 > sma50:
                        overview += "• MA Alignment: 🟢 BULLISH (20 > 50)\n"
                    elif sma20 < sma50:
                        overview += "• MA Alignment: 🔴 BEARISH (20 < 50)\n"
                    else:
                        overview += "• MA Alignment: ⚪ NEUTRAL\n"
                    
                    # Price vs MAs
                    above_count = sum([close > sma20, close > sma50])
                    overview += f"• Price above {above_count}/2 MAs\n"
            
            # Momentum Indicators
            overview += "\n**Momentum Indicators:**\n"
            
            if 'RSI' in data.columns:
                rsi = latest['RSI']
                if not pd.isna(rsi):
                    if rsi < 30:
                        rsi_status = '🟢 OVERSOLD (Bullish reversal likely)'
                    elif rsi < 40:
                        rsi_status = '🟡 NEAR OVERSOLD'
                    elif rsi > 70:
                        rsi_status = '🔴 OVERBOUGHT (Bearish reversal likely)'
                    elif rsi > 60:
                        rsi_status = '🟠 NEAR OVERBOUGHT'
                    else:
                        rsi_status = '⚪ NEUTRAL'
                    
                    overview += f"• RSI: {rsi:.1f} - {rsi_status}\n"
            
            if 'MACD' in data.columns and 'MACD_Signal' in data.columns:
                macd = latest['MACD']
                signal = latest['MACD_Signal']
                if not pd.isna(macd) and not pd.isna(signal):
                    diff = macd - signal
                    overview += f"• MACD vs Signal: {diff:.3f}\n"
                    if diff > 0:
                        overview += "  → 🟢 BULLISH MOMENTUM\n"
                    else:
                        overview += "  → 🔴 BEARISH MOMENTUM\n"
                    
                    # Check histogram trend
                    if 'MACD_Hist' in data.columns and len(data) > 3:
                        hist_values = data['MACD_Hist'].tail(3).values
                        if all(not pd.isna(h) and h < 0 for h in hist_values) and hist_values[-1] > hist_values[0]:
                            overview += "  → 📈 Histogram shrinking (potential reversal)\n"
            
            # Volume Analysis
            overview += "\n**Volume Analysis:**\n"
            
            if 'Volume_Ratio' in data.columns:
                volume_ratio = latest['Volume_Ratio']
                if not pd.isna(volume_ratio):
                    overview += f"• Volume Ratio: {volume_ratio:.1f}x 20-day average\n"
                    if volume_ratio > 1.5:
                        overview += "  → 📈 HIGH VOLUME ACTIVITY\n"
            
            if 'AD_Line' in data.columns and len(data) > 1:
                ad_current = latest['AD_Line']
                ad_prev = data['AD_Line'].iloc[-2]
                if not pd.isna(ad_current) and not pd.isna(ad_prev):
                    ad_trend = "🟢 ACCUMULATION" if ad_current > ad_prev else "🔴 DISTRIBUTION"
                    overview += f"• A/D Line Trend: {ad_trend}\n"
            
            # Bollinger Bands Analysis
            if 'BB_Upper' in data.columns and 'BB_Lower' in data.columns:
                bb_upper = latest['BB_Upper']
                bb_lower = latest['BB_Lower']
                close = latest['Close']
                
                if not pd.isna(bb_upper) and not pd.isna(bb_lower):
                    try:
                        bb_position = (close - bb_lower) / (bb_upper - bb_lower)
                        bb_width = (bb_upper - bb_lower) / close * 100
                        
                        overview += f"\n**Bollinger Bands:**\n"
                        overview += f"• Position: {bb_position*100:.1f}%\n"
                        overview += f"• Width: {bb_width:.1f}% of price\n"
                        
                        if bb_position < 0.1:
                            overview += "  → 📉 PRICE NEAR LOWER BAND (Potential bounce)\n"
                        elif bb_position > 0.9:
                            overview += "  → 📈 PRICE NEAR UPPER BAND (Potential pullback)\n"
                    except:
                        pass
            
            # Reversal Pattern Details
            if reversal_patterns['bullish_reversal'] or reversal_patterns['bearish_reversal']:
                overview += "\n**⚠️ REVERSAL PATTERN ANALYSIS ⚠️**\n"
                
                if reversal_patterns['bullish_reversal']:
                    overview += "• Pattern: BULLISH REVERSAL\n"
                    overview += f"• Confidence: {reversal_patterns['confidence']}%\n"
                
                if reversal_patterns['bearish_reversal']:
                    overview += "• Pattern: BEARISH REVERSAL\n"
                    overview += f"• Confidence: {reversal_patterns['confidence']}%\n"
            
            return overview
            
        except Exception as e:
            print(f"Error preparing technical overview: {e}")
            return "📈 **TECHNICAL OVERVIEW**\n\nError preparing technical overview."
    
    def _format_number(self, num: float) -> str:
        """Format large numbers with K, M, B suffixes"""
        try:
            num = float(num)
            if abs(num) >= 1_000_000_000:
                return f"{num/1_000_000_000:.1f}B"
            elif abs(num) >= 1_000_000:
                return f"{num/1_000_000:.1f}M"
            elif abs(num) >= 1_000:
                return f"{num/1_000:.1f}K"
            else:
                return f"{num:.0f}"
        except:
            return str(num)