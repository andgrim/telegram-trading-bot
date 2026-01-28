import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List
import warnings
import asyncio
import os
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# Using pure Python TA library
import ta

from config import CONFIG

class TradingAnalyzer:
    """Comprehensive analyzer for technical analysis with extended timeframes"""
    
    def __init__(self):
        self.config = CONFIG
    
    async def analyze_ticker(self, ticker_symbol: str, period: str = '1y') -> Dict:
        """Perform comprehensive analysis with complete indicators for all timeframes"""
        try:
            print(f"🔍 Starting analysis for {ticker_symbol} ({period})")
            
            # Map period to yfinance format with extended data for complete indicators
            yf_period_map = {
                '3m': '6mo',    # Get 6 months data for 3m analysis
                '6m': '1y',     # Get 1 year data for 6m analysis
                '1y': '2y',     # Get 2 years data for 1y analysis
                '2y': '3y',     # Get 3 years data for 2y analysis
                '3y': '5y',     # Get 5 years data for 3y analysis
                '5y': '6y'      # Get 6 years data for 5y analysis
            }
            
            fetch_period = yf_period_map.get(period, '2y')
            
            # Fetch extended data with simpler method to avoid scipy issues
            data = await self._fetch_data_simple(ticker_symbol, fetch_period)
            
            if data is None or data.empty:
                return {
                    'success': False, 
                    'error': f'No data found for {ticker_symbol}'
                }
            
            print(f"✅ Data fetched: {len(data)} rows")
            
            # Flatten MultiIndex columns
            data = self._flatten_dataframe(data)
            
            # Trim data to requested period while keeping enough for indicators
            data = self._trim_to_period(data, period)
            
            # Get ticker info
            info = await self._get_ticker_info(ticker_symbol, data)
            
            # Calculate complete technical indicators
            data = self._calculate_complete_indicators(data, period)
            
            # Get fundamental analysis
            fundamental = self._analyze_fundamentals(info)
            
            # Generate signals
            signals = self._generate_signals(data)
            
            # Detect reversal patterns
            reversal_patterns = self._detect_reversal_patterns(data)
            
            # Prepare unified analysis summary
            summary = self._prepare_summary(ticker_symbol, data, info, fundamental, signals, reversal_patterns)
            
            print(f"✅ Analysis complete for {ticker_symbol}")
            
            return {
                'success': True,
                'data': data,
                'info': info,
                'fundamental': fundamental,
                'signals': signals,
                'reversal_patterns': reversal_patterns,
                'summary': summary,
                'requested_period': period
            }
            
        except Exception as e:
            print(f"❌ Error in analyze_ticker: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}
    
    async def _fetch_data_simple(self, ticker_symbol: str, yf_period: str) -> pd.DataFrame:
        """Simple data fetch without complex parameters to avoid scipy dependency"""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                print(f"Simple fetch attempt {attempt + 1}/{max_retries} for {ticker_symbol}")
                
                # Add delay between retries
                if attempt > 0:
                    await asyncio.sleep(attempt * 2)
                
                # Use Ticker object which is simpler
                ticker = yf.Ticker(ticker_symbol)
                
                # Get history with minimal parameters and timeout
                data = ticker.history(period=yf_period, interval="1d", timeout=10)
                
                if data is not None and not data.empty:
                    print(f"✅ Simple fetch successful: {len(data)} rows")
                    print(f"DEBUG: Data columns: {data.columns.tolist()}")
                    print(f"DEBUG: Data sample:\n{data.head()}")
                    return data
                else:
                    print(f"⚠️ No data returned for {ticker_symbol}")
                    
            except Exception as e:
                print(f"Simple fetch attempt {attempt + 1} failed: {str(e)}")
                
                if attempt == max_retries - 1:
                    print(f"❌ All simple fetch attempts failed for {ticker_symbol}")
                    # Try one last time with different method
                    try:
                        # Use download without repair parameter
                        data = yf.download(
                            tickers=ticker_symbol,
                            period=yf_period,
                            interval="1d",
                            progress=False,
                            threads=False,
                            timeout=10
                        )
                        if data is not None and not data.empty:
                            print(f"✅ Fallback download successful: {len(data)} rows")
                            return data
                    except Exception as e2:
                        print(f"Fallback download also failed: {e2}")
        
        return None
    
    def _trim_to_period(self, data: pd.DataFrame, period: str) -> pd.DataFrame:
        """Trim data to requested period while keeping enough for indicator calculations"""
        # Define how many trading days to keep for each period
        # (approximately 252 trading days per year)
        period_trading_days = {
            '3m': 63,    # 3 months
            '6m': 126,   # 6 months
            '1y': 252,   # 1 year
            '2y': 504,   # 2 years
            '3y': 756,   # 3 years
            '5y': 1260   # 5 years
        }
        
        days_to_keep = period_trading_days.get(period, 252)
        
        # Add buffer for indicator calculations (especially for long periods)
        buffer_days = {
            '3m': 60,    # Additional 2 months
            '6m': 90,    # Additional 3 months
            '1y': 180,   # Additional 6 months
            '2y': 250,   # Additional 10 months
            '3y': 300,   # Additional 1 year
            '5y': 500    # Additional 2 years
        }
        
        days_to_keep += buffer_days.get(period, 180)
        
        # Ensure we don't exceed available data
        if len(data) > days_to_keep:
            return data.iloc[-days_to_keep:]
        else:
            return data
    
    async def _get_ticker_info(self, ticker_symbol: str, data: pd.DataFrame) -> Dict:
        """Get ticker information with fallback"""
        try:
            ticker = yf.Ticker(ticker_symbol)
            info = ticker.info
            
            # Validate info
            if not info or 'symbol' not in info:
                raise ValueError("Invalid ticker info")
            
            # Add current price if missing
            if 'regularMarketPrice' not in info and len(data) > 0:
                info['regularMarketPrice'] = data['Close'].iloc[-1]
            
            # Add market cap if missing
            if 'marketCap' not in info:
                if 'regularMarketPrice' in info and 'sharesOutstanding' in info:
                    info['marketCap'] = info['regularMarketPrice'] * info['sharesOutstanding']
            
            return info
            
        except Exception as e:
            print(f"Warning: Could not get complete ticker info: {e}")
            # Return basic info
            return {
                'symbol': ticker_symbol,
                'longName': ticker_symbol,
                'shortName': ticker_symbol,
                'regularMarketPrice': data['Close'].iloc[-1] if len(data) > 0 else 0,
                'currency': 'USD',
                'marketCap': None
            }
    
    def _flatten_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flatten MultiIndex columns to simple column names"""
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    
    def _calculate_complete_indicators(self, data: pd.DataFrame, period: str) -> pd.DataFrame:
        """Calculate complete technical indicators with extended lookback periods for long timeframes"""
        df = data.copy()
        
        # Ensure required columns
        required_columns = ['Close', 'High', 'Low', 'Open', 'Volume']
        for col in required_columns:
            if col not in df.columns:
                print(f"Warning: Missing required column: {col}")
                return df
        
        # For longer timeframes, adjust some indicator parameters for better visualization
        if period in ['2y', '3y', '5y']:
            # Use longer moving averages for long-term trends
            ma_periods = [20, 50, 100, 200]
        else:
            ma_periods = [9, 20, 50]
        
        # Calculate moving averages with forward fill
        for ma in ma_periods:
            if len(df) >= ma:
                # SMA (Simple Moving Average)
                try:
                    sma_indicator = ta.trend.SMAIndicator(df['Close'], window=ma)
                    df[f'SMA_{ma}'] = sma_indicator.sma_indicator().ffill()
                except Exception as e:
                    print(f"Error calculating SMA_{ma}: {e}")
                    df[f'SMA_{ma}'] = np.nan
        
        # Calculate RSI (Relative Strength Index)
        if len(df) >= self.config.RSI_PERIOD:
            try:
                rsi_indicator = ta.momentum.RSIIndicator(df['Close'], window=self.config.RSI_PERIOD)
                df['RSI'] = rsi_indicator.rsi().ffill()
            except Exception as e:
                print(f"Error calculating RSI: {e}")
                df['RSI'] = np.nan
        
        # Calculate MACD with forward fill
        if len(df) >= self.config.MACD_SLOW:
            try:
                macd_indicator = ta.trend.MACD(
                    df['Close'],
                    window_slow=self.config.MACD_SLOW,
                    window_fast=self.config.MACD_FAST,
                    window_sign=self.config.MACD_SIGNAL
                )
                df['MACD'] = macd_indicator.macd().ffill()
                df['MACD_Signal'] = macd_indicator.macd_signal().ffill()
                df['MACD_Hist'] = macd_indicator.macd_diff().ffill()
            except Exception as e:
                print(f"Error calculating MACD: {e}")
                df['MACD'] = np.nan
                df['MACD_Signal'] = np.nan
                df['MACD_Hist'] = np.nan
        
        # Calculate Bollinger Bands with forward fill
        if len(df) >= self.config.BB_PERIOD:
            try:
                bb_indicator = ta.volatility.BollingerBands(
                    df['Close'],
                    window=self.config.BB_PERIOD,
                    window_dev=self.config.BB_STD
                )
                df['BB_Upper'] = bb_indicator.bollinger_hband().ffill()
                df['BB_Middle'] = bb_indicator.bollinger_mavg().ffill()
                df['BB_Lower'] = bb_indicator.bollinger_lband().ffill()
            except Exception as e:
                print(f"Error calculating Bollinger Bands: {e}")
                df['BB_Upper'] = np.nan
                df['BB_Middle'] = np.nan
                df['BB_Lower'] = np.nan
        
        # Calculate A/D Line (Accumulation/Distribution)
        try:
            hl_range = df['High'] - df['Low']
            hl_range = hl_range.replace(0, 0.000001)
            mfm = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / hl_range
            mfv = mfm * df['Volume']
            df['AD_Line'] = mfv.cumsum().ffill()
        except Exception as e:
            print(f"Error calculating A/D Line: {e}")
            df['AD_Line'] = 0
        
        # Calculate volume indicators
        try:
            df['Volume_MA'] = df['Volume'].rolling(window=20, min_periods=1).mean().ffill()
            df['Volume_Ratio'] = (df['Volume'] / df['Volume_MA']).ffill()
        except Exception as e:
            print(f"Error calculating volume indicators: {e}")
            df['Volume_MA'] = np.nan
            df['Volume_Ratio'] = np.nan
        
        # Remove early rows with NaN values for cleaner charts
        # Find first non-NaN for all critical indicators
        critical_cols = ['SMA_20', 'SMA_50', 'RSI', 'MACD']
        if period in ['2y', '3y', '5y']:
            critical_cols.extend(['SMA_100', 'SMA_200'])
        
        first_valid_idx = 0
        for col in critical_cols:
            if col in df.columns:
                first_non_nan = df[col].first_valid_index()
                if first_non_nan:
                    idx_num = df.index.get_loc(first_non_nan)
                    first_valid_idx = max(first_valid_idx, idx_num)
        
        # Keep data from first_valid_idx onwards (with a buffer)
        start_idx = max(0, first_valid_idx - 10)
        df = df.iloc[start_idx:].copy()
        
        # Fill any remaining NaN values with forward fill
        df = df.ffill()
        
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
            # Get recent data (last 20 periods for better detection)
            recent = data.tail(20).copy()
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
                'peg_ratio': info.get('pegRatio', np.nan),
                'market_cap': info.get('marketCap', np.nan)
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
                'dividend_rate': info.get('dividendRate', np.nan),
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
                
                # Price position relative to 200 MA (for long timeframes)
                if 'SMA_200' in data.columns:
                    sma200 = latest['SMA_200']
                    if not pd.isna(sma200):
                        if close > sma200:
                            signals.append({'type': 'ABOVE_200MA', 'strength': 'STRONG', 'direction': 'BULLISH'})
                        elif close < sma200:
                            signals.append({'type': 'BELOW_200MA', 'strength': 'STRONG', 'direction': 'BEARISH'})
            
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
        """Prepare unified analysis summary with all indicators and signals"""
        try:
            latest = data.iloc[-1]
            current_price = latest['Close']
            prev_close = data['Close'].iloc[-2] if len(data) > 1 else current_price
            price_change = ((current_price - prev_close) / prev_close * 100) if prev_close != 0 else 0
            
            # Volume analysis
            volume = latest['Volume']
            avg_volume = data['Volume'].tail(20).mean()
            volume_ratio = volume / avg_volume if avg_volume > 0 else 1
            
            # Yearly performance if available
            yearly_change = None
            if len(data) > 252:
                yearly_price = data['Close'].iloc[-252] if len(data) > 252 else data['Close'].iloc[0]
                yearly_change = ((current_price - yearly_price) / yearly_price * 100) if yearly_price != 0 else 0
            
            # Signal analysis
            bull_signals = [s for s in signals if s['direction'] == 'BULLISH']
            bear_signals = [s for s in signals if s['direction'] == 'BEARISH']
            neutral_signals = [s for s in signals if s['direction'] == 'NEUTRAL']
            
            # Check for specific important signals
            death_cross = any(s['type'] == 'DEATH_CROSS' for s in signals)
            golden_cross = any(s['type'] == 'GOLDEN_CROSS' for s in signals)
            rsi_oversold = any(s['type'] == 'RSI_OVERSOLD' for s in signals)
            rsi_overbought = any(s['type'] == 'RSI_OVERBOUGHT' for s in signals)
            
            # Moving average analysis
            ma_positions = []
            if 'SMA_50' in data.columns and not pd.isna(latest['SMA_50']):
                if current_price > latest['SMA_50']:
                    ma_positions.append(f"ABOVE 50MA (${latest['SMA_50']:.2f})")
                else:
                    ma_positions.append(f"BELOW 50MA (${latest['SMA_50']:.2f})")
            
            if 'SMA_200' in data.columns and not pd.isna(latest['SMA_200']):
                if current_price > latest['SMA_200']:
                    ma_positions.append(f"ABOVE 200MA (${latest['SMA_200']:.2f})")
                else:
                    ma_positions.append(f"BELOW 200MA (${latest['SMA_200']:.2f})")
            
            # RSI analysis
            rsi_analysis = ""
            if 'RSI' in data.columns and not pd.isna(latest['RSI']):
                rsi = latest['RSI']
                if rsi < 30:
                    rsi_analysis = f"RSI {rsi:.1f} - OVERSOLD 📉 (Potential bounce)"
                elif rsi > 70:
                    rsi_analysis = f"RSI {rsi:.1f} - OVERBOUGHT 📈 (Potential pullback)"
                elif rsi < 50:
                    rsi_analysis = f"RSI {rsi:.1f} - Weakness"
                else:
                    rsi_analysis = f"RSI {rsi:.1f} - Strength"
            
            # MACD analysis
            macd_analysis = ""
            if 'MACD' in data.columns and 'MACD_Signal' in data.columns:
                macd = latest['MACD']
                signal = latest['MACD_Signal']
                if not pd.isna(macd) and not pd.isna(signal):
                    if macd > signal:
                        macd_analysis = f"MACD {macd:.3f} > Signal {signal:.3f} - BULLISH 🟢"
                    else:
                        macd_analysis = f"MACD {macd:.3f} < Signal {signal:.3f} - BEARISH 🔴"
            
            # Bollinger Bands analysis
            bb_analysis = ""
            if 'BB_Upper' in data.columns and 'BB_Lower' in data.columns:
                try:
                    bb_position = (latest['Close'] - latest['BB_Lower']) / (latest['BB_Upper'] - latest['BB_Lower'])
                    if not pd.isna(bb_position):
                        if bb_position < 0.2:
                            bb_analysis = f"BB Position: {bb_position*100:.1f}% - Near Lower Band (Oversold)"
                        elif bb_position > 0.8:
                            bb_analysis = f"BB Position: {bb_position*100:.1f}% - Near Upper Band (Overbought)"
                        else:
                            bb_analysis = f"BB Position: {bb_position*100:.1f}% - Neutral"
                except:
                    pass
            
            # Volume analysis text
            volume_analysis = ""
            if volume_ratio > 2.0:
                volume_analysis = "HIGH VOLUME 🚨 (Strong interest)"
            elif volume_ratio > 1.5:
                volume_analysis = "Volume above average"
            elif volume_ratio < 0.5:
                volume_analysis = "Low volume"
            else:
                volume_analysis = "Normal volume"
            
            # Trend analysis
            trend_analysis = ""
            if 'SMA_20' in data.columns and 'SMA_50' in data.columns:
                sma20 = latest['SMA_20']
                sma50 = latest['SMA_50']
                if not pd.isna(sma20) and not pd.isna(sma50):
                    if sma20 > sma50:
                        trend_analysis = "Trend: BULLISH 🟢 (20MA > 50MA)"
                    else:
                        trend_analysis = "Trend: BEARISH 🔴 (20MA < 50MA)"
            
            # Prepare summary
            summary = f"""
📊 **COMPREHENSIVE ANALYSIS: {ticker}**

**PRICE & VOLUME**
• Current Price: ${current_price:.2f}
• Daily Change: {price_change:+.2f}%
"""
            
            if yearly_change is not None:
                summary += f"• Yearly Change: {yearly_change:+.2f}%\n"
            
            summary += f"""• Volume: {self._format_number(volume)} ({volume_ratio:.1f}x avg)
• Volume Analysis: {volume_analysis}

**TREND & MOVING AVERAGES**
• {trend_analysis}
"""
            
            for position in ma_positions:
                summary += f"• Position: {position}\n"
            
            if death_cross:
                summary += "• ⚠️ DEATH CROSS DETECTED (Strong BEARISH signal)\n"
            if golden_cross:
                summary += "• ✅ GOLDEN CROSS DETECTED (Strong BULLISH signal)\n"
            
            summary += f"""
**TECHNICAL INDICATORS**
• {rsi_analysis}
• {macd_analysis}
"""
            
            if bb_analysis:
                summary += f"• {bb_analysis}\n"
            
            # A/D Line analysis
            if 'AD_Line' in data.columns and len(data) > 1:
                ad_current = latest['AD_Line']
                ad_prev = data['AD_Line'].iloc[-2]
                if not pd.isna(ad_current) and not pd.isna(ad_prev):
                    if ad_current > ad_prev:
                        summary += "• A/D Line: ACCUMULATION 🟢\n"
                    else:
                        summary += "• A/D Line: DISTRIBUTION 🔴\n"
            
            # Reversal patterns
            if reversal_patterns['bullish_reversal']:
                summary += f"\n⚠️ **BULLISH REVERSAL PATTERN** ⚠️\n"
                summary += f"• Confidence: {reversal_patterns['confidence']}%\n"
                if reversal_patterns['signals']:
                    summary += "• Signals:\n"
                    for signal in reversal_patterns['signals'][:3]:
                        summary += f"  - {signal}\n"
            
            if reversal_patterns['bearish_reversal']:
                summary += f"\n⚠️ **BEARISH REVERSAL PATTERN** ⚠️\n"
                summary += f"• Confidence: {reversal_patterns['confidence']}%\n"
                if reversal_patterns['signals']:
                    summary += "• Signals:\n"
                    for signal in reversal_patterns['signals'][:3]:
                        summary += f"  - {signal}\n"
            
            # Signal summary
            summary += f"""
**TECHNICAL SIGNALS ({len(signals)} total)**
• Bullish: {len(bull_signals)} 🟢
• Bearish: {len(bear_signals)} 🔴
• Neutral: {len(neutral_signals)} ⚪

**FUNDAMENTAL SCORE**: {fundamental['score']}/100
"""
            
            # Overall sentiment
            if reversal_patterns['bullish_reversal']:
                sentiment = '🟢 STRONG BULLISH (Reversal in progress)'
            elif reversal_patterns['bearish_reversal']:
                sentiment = '🔴 STRONG BEARISH (Reversal in progress)'
            elif len(bull_signals) > len(bear_signals):
                sentiment = '🟢 BULLISH'
            elif len(bear_signals) > len(bull_signals):
                sentiment = '🔴 BEARISH'
            else:
                sentiment = '⚪ NEUTRAL'
            
            summary += f"**OVERALL SENTIMENT**: {sentiment}"
            
            # Add key levels
            summary += f"\n\n**KEY LEVELS**"
            for ma in [9, 20, 50, 100, 200]:
                ma_col = f'SMA_{ma}'
                if ma_col in data.columns and not pd.isna(latest[ma_col]):
                    distance_pct = abs(current_price - latest[ma_col]) / latest[ma_col] * 100
                    direction = "above" if current_price > latest[ma_col] else "below"
                    summary += f"\n• {ma}MA: ${latest[ma_col]:.2f} ({direction}, {distance_pct:.1f}%)"
            
            return summary
            
        except Exception as e:
            print(f"Error preparing summary: {e}")
            return f"**ANALYSIS: {ticker}**\n\nError preparing analysis summary."
    
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