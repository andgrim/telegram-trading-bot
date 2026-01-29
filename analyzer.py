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
    """Comprehensive analyzer for technical analysis with extended timeframes and international markets"""
    
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
            
            # Handle ticker formatting for yfinance
            formatted_ticker = self._format_ticker_for_yfinance(ticker_symbol)
            print(f"Formatted ticker: {ticker_symbol} -> {formatted_ticker}")
            
            # Fetch data
            data = await self._fetch_data_with_retry(formatted_ticker, fetch_period)
            
            if data is None or data.empty:
                # Try alternative formatting
                print(f"First attempt failed, trying alternative format for {ticker_symbol}")
                
                # For European stocks, try different formats
                european_tickers = self._get_european_ticker_variants(ticker_symbol)
                for alt_ticker in european_tickers:
                    print(f"Trying alternative: {alt_ticker}")
                    data = await self._fetch_data_with_retry(alt_ticker, fetch_period)
                    if data is not None and not data.empty:
                        formatted_ticker = alt_ticker
                        print(f"✅ Found data with alternative ticker: {alt_ticker}")
                        break
                
                if data is None or data.empty:
                    return {
                        'success': False, 
                        'error': f'No data found for {ticker_symbol}. Try using Yahoo Finance format like: ISP.MI, AI.PA, etc.'
                    }
            
            print(f"✅ Data fetched: {len(data)} rows")
            
            # Flatten MultiIndex columns
            data = self._flatten_dataframe(data)
            
            # Trim data to requested period while keeping enough for indicators
            data = self._trim_to_period(data, period)
            
            # Get ticker info
            info = await self._get_ticker_info(formatted_ticker, data, ticker_symbol)
            
            # Calculate COMPLETE technical indicators (including A/D)
            data = self._calculate_complete_indicators(data, period)
            
            # Get fundamental analysis
            fundamental = self._analyze_fundamentals(info)
            
            # Generate signals
            signals = self._generate_signals(data)
            
            # Detect reversal patterns
            reversal_patterns = self._detect_reversal_patterns(data)
            
            # Prepare unified analysis summary
            summary = self._prepare_summary(ticker_symbol, data, info, fundamental, signals, reversal_patterns)
            
            # Prepare compact summary for photo caption
            compact_summary = self._prepare_compact_summary(ticker_symbol, data, info, fundamental, signals, reversal_patterns)
            
            print(f"✅ Analysis complete for {ticker_symbol}")
            
            return {
                'success': True,
                'data': data,
                'info': info,
                'fundamental': fundamental,
                'signals': signals,
                'reversal_patterns': reversal_patterns,
                'summary': summary,
                'compact_summary': compact_summary,
                'requested_period': period
            }
            
        except Exception as e:
            print(f"❌ Error in analyze_ticker: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}
    
    def _format_ticker_for_yfinance(self, ticker_symbol: str) -> str:
        """Format ticker symbol for yfinance library"""
        ticker = ticker_symbol.strip().upper()
        
        # Common European exchanges and their Yahoo Finance formats
        european_mapping = {
            # Italy (.MI = Milan)
            'ISP': 'ISP.MI',
            'ENEL': 'ENEL.MI',
            'ENI': 'ENI.MI',
            'UCG': 'UCG.MI',  # Unicredit
            
            # Germany (.DE = Frankfurt)
            'ADS': 'ADS.DE',  # Adidas
            'ALV': 'ALV.DE',  # Allianz
            'BAS': 'BAS.DE',  # BASF
            'BAYN': 'BAYN.DE',  # Bayer
            'BMW': 'BMW.DE',
            'DAI': 'DAI.DE',  # Daimler/Mercedes
            'DB1': 'DB1.DE',  # Deutsche Börse
            'DPW': 'DPW.DE',  # Deutsche Post
            'DTE': 'DTE.DE',  # Deutsche Telekom
            
            # Netherlands (.AS = Amsterdam)
            'ADYEN': 'ADYEN.AS',
            'AD': 'AD.AS',  # Ahold Delhaize
            'ASML': 'ASML.AS',
            
            # France (.PA = Paris)
            'AI': 'AI.PA',  # Air Liquide
            'AIR': 'AIR.PA',  # Airbus
            'BNP': 'BNP.PA',  # BNP Paribas
            'CS': 'CS.PA',  # AXA
            'EL': 'EL.PA',  # EssilorLuxottica
            'ENGI': 'ENGI.PA',
            'BN': 'BN.PA',  # Danone
            
            # Spain (.MC = Madrid)
            'AMS': 'AMS.MC',  # Amadeus
            
            # Belgium (.BR = Brussels)
            'ABI': 'ABI.BR',  # Anheuser-Busch InBev
            
            # Ireland (.IR = Dublin)
            'CRG': 'CRG.IR',  # CRH
        }
        
        # Check if we have a mapping for this ticker
        if ticker in european_mapping:
            return european_mapping[ticker]
        
        # If it already has a dot, keep it as is
        if '.' in ticker:
            return ticker
        
        # Map common indices and commodities
        special_mapping = {
            # US Indices
            'SPX': '^GSPC',        # S&P 500
            'DJI': '^DJI',         # Dow Jones
            'IXIC': '^IXIC',       # NASDAQ
            'RUT': '^RUT',         # Russell 2000
            'VIX': '^VIX',         # VIX
            
            # European Indices
            'DAX': '^GDAXI',       # German DAX
            'CAC': '^FCHI',        # French CAC 40
            'FTSE': '^FTSE',       # UK FTSE 100
            'IBEX': '^IBEX',       # Spanish IBEX 35
            'FTMIB': 'FTSEMIB.MI', # Italian FTSE MIB
            
            # Asian Indices
            'N225': '^N225',       # Nikkei 225
            'HSI': '^HSI',         # Hang Seng
            
            # Commodities
            'GOLD': 'GC=F',        # Gold Futures
            'SILVER': 'SI=F',      # Silver Futures
            'OIL': 'CL=F',         # Crude Oil WTI
            'BRENT': 'BZ=F',       # Brent Crude
            'NATGAS': 'NG=F',      # Natural Gas
            'COPPER': 'HG=F',      # Copper
            
            # Currency pairs
            'EURUSD': 'EURUSD=X',
            'GBPUSD': 'GBPUSD=X',
            'USDJPY': 'JPY=X',
            
            # Crypto
            'BTC': 'BTC-USD',
            'ETH': 'ETH-USD',
            'XRP': 'XRP-USD',
        }
        
        if ticker in special_mapping:
            return special_mapping[ticker]
        
        return ticker
    
    def _get_european_ticker_variants(self, ticker_symbol: str) -> List[str]:
        """Generate alternative ticker formats for European stocks"""
        ticker = ticker_symbol.strip().upper()
        variants = []
        
        # If already has exchange suffix, keep it
        if '.' in ticker:
            variants.append(ticker)
        
        # Try common European exchanges
        exchanges = ['.MI', '.DE', '.AS', '.PA', '.BR', '.IR', '.MC', '.SW', '.L', '.T', '.HK']
        
        for exchange in exchanges:
            variants.append(f"{ticker}{exchange}")
        
        # Also try without dot
        variants.append(ticker.replace('.', ''))
        
        # Try Yahoo Finance alternative formats
        if '.' in ticker:
            base, exchange = ticker.split('.')
            variants.append(f"{base}-{exchange}")  # ISP-MI
            variants.append(f"{base}.{exchange.lower()}")  # ISP.mi
        
        return variants
    
    async def _fetch_data_with_retry(self, ticker_symbol: str, yf_period: str, max_retries: int = 3) -> pd.DataFrame:
        """Fetch data with retry logic"""
        for attempt in range(max_retries):
            try:
                print(f"📥 Fetch attempt {attempt + 1}/{max_retries} for {ticker_symbol}")
                
                if attempt > 0:
                    await asyncio.sleep(attempt * 2)
                
                # Try multiple methods
                data = None
                
                # Method 1: Direct download with auto_adjust
                try:
                    data = yf.download(
                        tickers=ticker_symbol,
                        period=yf_period,
                        interval="1d",
                        progress=False,
                        threads=False,
                        timeout=15,
                        auto_adjust=True
                    )
                    if data is not None and not data.empty:
                        print(f"✅ Method 1 successful: {len(data)} rows")
                        return data
                except Exception as e1:
                    print(f"Method 1 failed: {e1}")
                
                # Method 2: Download without auto_adjust
                if data is None or data.empty:
                    try:
                        data = yf.download(
                            tickers=ticker_symbol,
                            period=yf_period,
                            interval="1d",
                            progress=False,
                            threads=False,
                            timeout=15,
                            auto_adjust=False
                        )
                        if data is not None and not data.empty:
                            print(f"✅ Method 2 successful: {len(data)} rows")
                            return data
                    except Exception as e2:
                        print(f"Method 2 failed: {e2}")
                
                # Method 3: Use Ticker object
                if data is None or data.empty:
                    try:
                        ticker = yf.Ticker(ticker_symbol)
                        data = ticker.history(period=yf_period, interval="1d", timeout=15)
                        if data is not None and not data.empty:
                            print(f"✅ Method 3 successful: {len(data)} rows")
                            return data
                    except Exception as e3:
                        print(f"Method 3 failed: {e3}")
                
                # Method 4: Try with repair parameter
                if data is None or data.empty:
                    try:
                        data = yf.download(
                            tickers=ticker_symbol,
                            period=yf_period,
                            interval="1d",
                            progress=False,
                            threads=False,
                            timeout=20,
                            repair=True
                        )
                        if data is not None and not data.empty:
                            print(f"✅ Method 4 successful: {len(data)} rows")
                            return data
                    except Exception as e4:
                        print(f"Method 4 failed: {e4}")
                
            except Exception as e:
                print(f"Fetch attempt {attempt + 1} failed: {str(e)}")
        
        print(f"❌ All fetch attempts failed for {ticker_symbol}")
        return None
    
    def _trim_to_period(self, data: pd.DataFrame, period: str) -> pd.DataFrame:
        """Trim data to requested period while keeping enough for indicator calculations"""
        # Define how many trading days to keep for each period
        period_trading_days = {
            '3m': 63,    # 3 months
            '6m': 126,   # 6 months
            '1y': 252,   # 1 year
            '2y': 504,   # 2 years
            '3y': 756,   # 3 years
            '5y': 1260   # 5 years
        }
        
        days_to_keep = period_trading_days.get(period, 252)
        
        # Add buffer for indicator calculations
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
    
    async def _get_ticker_info(self, formatted_ticker: str, data: pd.DataFrame, original_ticker: str) -> Dict:
        """Get ticker information with fallback"""
        try:
            ticker = yf.Ticker(formatted_ticker)
            info = ticker.info
            
            # If info is empty, create basic info
            if not info or 'symbol' not in info:
                info = {
                    'symbol': original_ticker,
                    'longName': original_ticker,
                    'shortName': original_ticker,
                    'regularMarketPrice': data['Close'].iloc[-1] if len(data) > 0 else 0,
                    'currency': self._detect_currency(formatted_ticker),
                    'marketCap': None
                }
            
            # Ensure required fields exist
            if 'regularMarketPrice' not in info and len(data) > 0:
                info['regularMarketPrice'] = data['Close'].iloc[-1]
            
            if 'currency' not in info:
                info['currency'] = self._detect_currency(formatted_ticker)
            
            return info
            
        except Exception as e:
            print(f"Warning: Could not get ticker info: {e}")
            # Return basic info
            return {
                'symbol': original_ticker,
                'longName': original_ticker,
                'shortName': original_ticker,
                'regularMarketPrice': data['Close'].iloc[-1] if len(data) > 0 else 0,
                'currency': self._detect_currency(formatted_ticker),
                'marketCap': None
            }
    
    def _detect_currency(self, ticker: str) -> str:
        """Detect currency based on ticker suffix"""
        ticker_upper = ticker.upper()
        
        currency_map = {
            '.MI': 'EUR',  # Milan
            '.DE': 'EUR',  # Germany
            '.AS': 'EUR',  # Amsterdam
            '.PA': 'EUR',  # Paris
            '.BR': 'EUR',  # Brussels
            '.IR': 'EUR',  # Ireland
            '.MC': 'EUR',  # Madrid
            '.SW': 'CHF',  # Switzerland
            '.L': 'GBP',   # London
            '.T': 'JPY',   # Tokyo
            '.HK': 'HKD',  # Hong Kong
        }
        
        for suffix, currency in currency_map.items():
            if ticker_upper.endswith(suffix):
                return currency
        
        return 'USD'
    
    def _flatten_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flatten MultiIndex columns"""
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    
    def _calculate_complete_indicators(self, data: pd.DataFrame, period: str) -> pd.DataFrame:
        """Calculate COMPLETE technical indicators including A/D, Stochastic, OBV, etc."""
        df = data.copy()
        
        # Ensure we have required columns
        required_cols = ['Close', 'High', 'Low', 'Open', 'Volume']
        for col in required_cols:
            if col not in df.columns:
                print(f"Warning: Missing required column: {col}")
                return df
        
        print(f"Calculating indicators for {len(df)} rows")
        
        # Calculate moving averages
        ma_periods = [9, 20, 50]
        if period in ['2y', '3y', '5y']:
            ma_periods.extend([100, 200])
        
        for ma in ma_periods:
            if len(df) >= ma:
                try:
                    # Simple Moving Average
                    df[f'SMA_{ma}'] = ta.trend.SMAIndicator(df['Close'], window=ma).sma_indicator().ffill()
                    
                    # Exponential Moving Average
                    df[f'EMA_{ma}'] = ta.trend.EMAIndicator(df['Close'], window=ma).ema_indicator().ffill()
                except Exception as e:
                    print(f"Error calculating MA {ma}: {e}")
                    df[f'SMA_{ma}'] = np.nan
                    df[f'EMA_{ma}'] = np.nan
        
        # Calculate RSI
        if len(df) >= 14:
            try:
                df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi().ffill()
            except Exception as e:
                print(f"Error calculating RSI: {e}")
                df['RSI'] = np.nan
        
        # Calculate MACD
        if len(df) >= 26:
            try:
                macd = ta.trend.MACD(df['Close'])
                df['MACD'] = macd.macd().ffill()
                df['MACD_Signal'] = macd.macd_signal().ffill()
                df['MACD_Hist'] = macd.macd_diff().ffill()
            except Exception as e:
                print(f"Error calculating MACD: {e}")
                df['MACD'] = np.nan
                df['MACD_Signal'] = np.nan
                df['MACD_Hist'] = np.nan
        
        # Calculate Bollinger Bands
        if len(df) >= 20:
            try:
                bb = ta.volatility.BollingerBands(df['Close'])
                df['BB_Upper'] = bb.bollinger_hband().ffill()
                df['BB_Middle'] = bb.bollinger_mavg().ffill()
                df['BB_Lower'] = bb.bollinger_lband().ffill()
                df['BB_Width'] = ((df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']) * 100
            except Exception as e:
                print(f"Error calculating Bollinger Bands: {e}")
                df['BB_Upper'] = np.nan
                df['BB_Middle'] = np.nan
                df['BB_Lower'] = np.nan
                df['BB_Width'] = np.nan
        
        # Calculate A/D Line (Accumulation/Distribution) - IMPORTANT!
        try:
            # Money Flow Multiplier
            mfm = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
            mfm = mfm.replace([np.inf, -np.inf], 0).fillna(0)
            
            # Money Flow Volume
            mfv = mfm * df['Volume']
            
            # Accumulation/Distribution Line
            df['AD_Line'] = mfv.cumsum().ffill()
            
            # AD Line Moving Average (20 period)
            df['AD_MA_20'] = df['AD_Line'].rolling(window=20, min_periods=1).mean().ffill()
            
            print(f"A/D Line calculated successfully, range: {df['AD_Line'].min():.0f} to {df['AD_Line'].max():.0f}")
        except Exception as e:
            print(f"Error calculating A/D Line: {e}")
            df['AD_Line'] = 0
            df['AD_MA_20'] = np.nan
        
        # Calculate Stochastic Oscillator
        if len(df) >= 14:
            try:
                stoch = ta.momentum.StochasticOscillator(
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    window=14,
                    smooth_window=3
                )
                df['Stoch_%K'] = stoch.stoch().ffill()
                df['Stoch_%D'] = stoch.stoch_signal().ffill()
            except Exception as e:
                print(f"Error calculating Stochastic: {e}")
                df['Stoch_%K'] = np.nan
                df['Stoch_%D'] = np.nan
        
        # Calculate On-Balance Volume (OBV)
        try:
            df['OBV'] = ta.volume.OnBalanceVolumeIndicator(
                close=df['Close'],
                volume=df['Volume']
            ).on_balance_volume().ffill()
        except Exception as e:
            print(f"Error calculating OBV: {e}")
            df['OBV'] = 0
        
        # Calculate Average True Range (ATR) for volatility
        if len(df) >= 14:
            try:
                df['ATR'] = ta.volatility.AverageTrueRange(
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    window=14
                ).average_true_range().ffill()
                
                # ATR as percentage of price
                df['ATR_Pct'] = (df['ATR'] / df['Close']) * 100
            except Exception as e:
                print(f"Error calculating ATR: {e}")
                df['ATR'] = np.nan
                df['ATR_Pct'] = np.nan
        
        # Calculate Volume indicators
        try:
            df['Volume_MA_20'] = df['Volume'].rolling(window=20, min_periods=1).mean().ffill()
            df['Volume_Ratio'] = (df['Volume'] / df['Volume_MA_20']).ffill()
            
            # Volume Weighted Average Price
            df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
        except Exception as e:
            print(f"Error calculating volume indicators: {e}")
            df['Volume_MA_20'] = np.nan
            df['Volume_Ratio'] = np.nan
            df['VWAP'] = np.nan
        
        # Calculate Price Rate of Change (ROC)
        if len(df) >= 12:
            try:
                df['ROC_12'] = ta.momentum.ROCIndicator(df['Close'], window=12).roc().ffill()
            except Exception as e:
                print(f"Error calculating ROC: {e}")
                df['ROC_12'] = np.nan
        
        # Calculate Williams %R
        if len(df) >= 14:
            try:
                df['Williams_%R'] = ta.momentum.WilliamsRIndicator(
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    lbp=14
                ).williams_r().ffill()
            except Exception as e:
                print(f"Error calculating Williams %R: {e}")
                df['Williams_%R'] = np.nan
        
        # Calculate Commodity Channel Index (CCI)
        if len(df) >= 20:
            try:
                df['CCI'] = ta.trend.CCIIndicator(
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    window=20
                ).cci().ffill()
            except Exception as e:
                print(f"Error calculating CCI: {e}")
                df['CCI'] = np.nan
        
        # Fill any remaining NaN values
        df = df.ffill()
        
        print(f"✅ Indicators calculated successfully")
        print(f"Available indicators: {[col for col in df.columns if col not in required_cols][:10]}...")
        
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
            
            # 6. Stochastic oversold
            if 'Stoch_%K' in recent.columns and 'Stoch_%D' in recent.columns:
                try:
                    stoch_k = latest['Stoch_%K']
                    stoch_d = latest['Stoch_%D']
                    if not pd.isna(stoch_k) and not pd.isna(stoch_d):
                        if stoch_k < 20 and stoch_d < 20:
                            bullish_signals.append(f"Stochastic oversold (K={stoch_k:.1f}, D={stoch_d:.1f})")
                except:
                    pass
            
            # 7. OBV bullish divergence
            if 'OBV' in recent.columns and len(recent) >= 5:
                try:
                    price_min_idx = recent['Close'].tail(5).idxmin()
                    obv_at_price_min = recent.loc[price_min_idx, 'OBV']
                    obv_current = latest['OBV']
                    
                    if obv_current > obv_at_price_min:
                        bullish_signals.append("OBV showing bullish divergence")
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
            
            if 'Stoch_%K' in recent.columns and 'Stoch_%D' in recent.columns:
                try:
                    stoch_k = latest['Stoch_%K']
                    stoch_d = latest['Stoch_%D']
                    if not pd.isna(stoch_k) and not pd.isna(stoch_d):
                        if stoch_k > 80 and stoch_d > 80:
                            bearish_signals.append(f"Stochastic overbought (K={stoch_k:.1f}, D={stoch_d:.1f})")
                except:
                    pass
            
            if 'Williams_%R' in recent.columns:
                try:
                    williams_r = latest['Williams_%R']
                    if not pd.isna(williams_r) and williams_r > -20:
                        bearish_signals.append(f"Williams %R overbought ({williams_r:.1f})")
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
        """Comprehensive fundamental data analysis"""
        fundamental = {
            'valuation': {},
            'profitability': {},
            'growth': {},
            'financial_health': {},
            'dividends': {},
            'efficiency': {},
            'cash_flow': {},
            'analyst_data': {},
            'score': 50,
            'grade': 'C'
        }
        
        try:
            # VALUATION METRICS
            fundamental['valuation'] = {
                'pe_ratio': info.get('trailingPE', np.nan),
                'forward_pe': info.get('forwardPE', np.nan),
                'price_to_book': info.get('priceToBook', np.nan),
                'price_to_sales': info.get('priceToSalesTrailing12Months', np.nan),
                'peg_ratio': info.get('pegRatio', np.nan),
                'market_cap': info.get('marketCap', np.nan),
                'enterprise_value': info.get('enterpriseValue', np.nan),
                'ev_to_ebitda': info.get('enterpriseToEbitda', np.nan),
                'ev_to_revenue': info.get('enterpriseToRevenue', np.nan)
            }
            
            # PROFITABILITY
            fundamental['profitability'] = {
                'roe': info.get('returnOnEquity', np.nan),
                'roa': info.get('returnOnAssets', np.nan),
                'roic': info.get('returnOnInvestedCapital', np.nan),
                'profit_margin': info.get('profitMargins', np.nan),
                'operating_margin': info.get('operatingMargins', np.nan),
                'gross_margin': info.get('grossMargins', np.nan),
                'ebitda_margin': info.get('ebitdaMargins', np.nan)
            }
            
            # GROWTH
            fundamental['growth'] = {
                'revenue_growth': info.get('revenueGrowth', np.nan),
                'earnings_growth': info.get('earningsGrowth', np.nan),
                'eps_growth': info.get('earningsQuarterlyGrowth', np.nan),
                'five_year_rev_growth': info.get('revenuePerShareGrowth', np.nan),
                'five_year_eps_growth': info.get('earningsPerShareGrowth', np.nan),
                'quarterly_rev_growth': info.get('quarterlyRevenueGrowth', np.nan),
                'quarterly_earnings_growth': info.get('quarterlyEarningsGrowth', np.nan)
            }
            
            # FINANCIAL HEALTH
            fundamental['financial_health'] = {
                'debt_to_equity': info.get('debtToEquity', np.nan),
                'current_ratio': info.get('currentRatio', np.nan),
                'quick_ratio': info.get('quickRatio', np.nan),
                'interest_coverage': info.get('interestCoverage', np.nan),
                'total_debt': info.get('totalDebt', np.nan),
                'total_cash': info.get('totalCash', np.nan),
                'free_cash_flow': info.get('freeCashflow', np.nan)
            }
            
            # DIVIDENDS
            fundamental['dividends'] = {
                'dividend_yield': info.get('dividendYield', np.nan),
                'dividend_rate': info.get('dividendRate', np.nan),
                'payout_ratio': info.get('payoutRatio', np.nan),
                'dividend_growth': info.get('dividendGrowth', np.nan),
                'five_year_dividend_growth': info.get('dividendGrowthRate5Y', np.nan),
                'trailing_annual_dividend': info.get('trailingAnnualDividendRate', np.nan),
                'forward_annual_dividend': info.get('forwardAnnualDividendRate', np.nan)
            }
            
            # EFFICIENCY
            fundamental['efficiency'] = {
                'asset_turnover': info.get('assetTurnover', np.nan),
                'inventory_turnover': info.get('inventoryTurnover', np.nan),
                'receivables_turnover': info.get('receivablesTurnover', np.nan),
                'days_sales_outstanding': info.get('daysSalesOutstanding', np.nan),
                'days_inventory': info.get('daysInventory', np.nan)
            }
            
            # CASH FLOW
            fundamental['cash_flow'] = {
                'operating_cash_flow': info.get('operatingCashflow', np.nan),
                'free_cash_flow_yield': info.get('freeCashflowYield', np.nan),
                'cash_flow_per_share': info.get('cashPerShare', np.nan),
                'operating_cf_margin': info.get('operatingCashflowMargin', np.nan)
            }
            
            # ANALYST DATA
            fundamental['analyst_data'] = {
                'target_mean_price': info.get('targetMeanPrice', np.nan),
                'target_high_price': info.get('targetHighPrice', np.nan),
                'target_low_price': info.get('targetLowPrice', np.nan),
                'recommendation_mean': info.get('recommendationMean', np.nan),
                'recommendation_key': info.get('recommendationKey', 'hold'),
                'number_of_analyst_opinions': info.get('numberOfAnalystOpinions', 0)
            }
            
            # Calculate fundamental score
            fundamental['score'], fundamental['grade'] = self._calculate_fundamental_score(fundamental)
            
        except Exception as e:
            print(f"Error in fundamental analysis: {e}")
        
        return fundamental
    
    def _calculate_fundamental_score(self, fundamental: Dict) -> tuple:
        """Calculate comprehensive fundamental score (0-100) and grade"""
        score = 50
        metrics_checked = 0
        
        try:
            # 1. VALUATION (20 points max)
            pe = fundamental['valuation'].get('pe_ratio')
            if pe and not np.isnan(pe):
                metrics_checked += 1
                if 0 < pe < 15:
                    score += 15  # Very cheap
                elif 0 < pe < 20:
                    score += 10  # Cheap
                elif 0 < pe < 25:
                    score += 5   # Fair
                elif pe > 40:
                    score -= 10  # Expensive
            
            peg = fundamental['valuation'].get('peg_ratio')
            if peg and not np.isnan(peg):
                metrics_checked += 1
                if 0 < peg < 1:
                    score += 5   # Undervalued
                elif 0 < peg < 1.5:
                    score += 3   # Fairly valued
                elif peg > 2:
                    score -= 5   # Overvalued
            
            # 2. PROFITABILITY (25 points max)
            roe = fundamental['profitability'].get('roe')
            if roe and not np.isnan(roe):
                metrics_checked += 1
                if roe > 0.20:
                    score += 15  # Excellent
                elif roe > 0.15:
                    score += 10  # Good
                elif roe > 0.10:
                    score += 5   # Average
                elif roe < 0:
                    score -= 10  # Negative
            
            profit_margin = fundamental['profitability'].get('profit_margin')
            if profit_margin and not np.isnan(profit_margin):
                metrics_checked += 1
                if profit_margin > 0.20:
                    score += 10  # Excellent margins
                elif profit_margin > 0.15:
                    score += 7   # Good margins
                elif profit_margin > 0.10:
                    score += 4   # Average margins
                elif profit_margin < 0:
                    score -= 5   # Negative margins
            
            # 3. FINANCIAL HEALTH (20 points max)
            debt_equity = fundamental['financial_health'].get('debt_to_equity')
            if debt_equity and not np.isnan(debt_equity):
                metrics_checked += 1
                if debt_equity < 0.5:
                    score += 10  # Low debt
                elif debt_equity < 1.0:
                    score += 7   # Moderate debt
                elif debt_equity > 2.0:
                    score -= 5   # High debt
            
            current_ratio = fundamental['financial_health'].get('current_ratio')
            if current_ratio and not np.isnan(current_ratio):
                metrics_checked += 1
                if current_ratio > 2.0:
                    score += 10  # Excellent liquidity
                elif current_ratio > 1.5:
                    score += 7   # Good liquidity
                elif current_ratio < 1.0:
                    score -= 5   # Poor liquidity
            
            # 4. GROWTH (15 points max)
            revenue_growth = fundamental['growth'].get('revenue_growth')
            if revenue_growth and not np.isnan(revenue_growth):
                metrics_checked += 1
                if revenue_growth > 0.15:
                    score += 10  # High growth
                elif revenue_growth > 0.10:
                    score += 7   # Good growth
                elif revenue_growth > 0.05:
                    score += 4   # Moderate growth
                elif revenue_growth < 0:
                    score -= 5   # Declining
            
            eps_growth = fundamental['growth'].get('eps_growth')
            if eps_growth and not np.isnan(eps_growth):
                metrics_checked += 1
                if eps_growth > 0.20:
                    score += 5   # Strong EPS growth
                elif eps_growth > 0.10:
                    score += 3   # Good EPS growth
                elif eps_growth < 0:
                    score -= 3   # Negative EPS growth
            
            # 5. DIVIDENDS (10 points max)
            dividend_yield = fundamental['dividends'].get('dividend_yield')
            if dividend_yield and not np.isnan(dividend_yield):
                metrics_checked += 1
                if dividend_yield > 0.04:
                    score += 8   # High yield
                elif dividend_yield > 0.02:
                    score += 5   # Good yield
                elif dividend_yield > 0.01:
                    score += 3   # Low yield
            
            payout_ratio = fundamental['dividends'].get('payout_ratio')
            if payout_ratio and not np.isnan(payout_ratio):
                metrics_checked += 1
                if 0 < payout_ratio < 0.5:
                    score += 2   # Sustainable
                elif payout_ratio > 0.8:
                    score -= 3   # Potentially unsustainable
            
            # 6. ANALYST SENTIMENT (10 points max)
            recommendation_mean = fundamental['analyst_data'].get('recommendation_mean')
            if recommendation_mean and not np.isnan(recommendation_mean):
                metrics_checked += 1
                if recommendation_mean < 2.0:  # 1=Strong Buy, 2=Buy, 3=Hold, etc.
                    score += 8   # Strong buy sentiment
                elif recommendation_mean < 2.5:
                    score += 5   # Buy sentiment
                elif recommendation_mean > 3.5:
                    score -= 5   # Sell sentiment
            
            # Normalize score based on number of metrics checked
            if metrics_checked > 0:
                # Adjust for metrics availability
                score = (score - 50) * (min(metrics_checked, 10) / 10) + 50
            
            # Ensure score is within bounds
            score = max(0, min(100, score))
            
            # Assign letter grade
            if score >= 90:
                grade = 'A+'
            elif score >= 85:
                grade = 'A'
            elif score >= 80:
                grade = 'A-'
            elif score >= 75:
                grade = 'B+'
            elif score >= 70:
                grade = 'B'
            elif score >= 65:
                grade = 'B-'
            elif score >= 60:
                grade = 'C+'
            elif score >= 55:
                grade = 'C'
            elif score >= 50:
                grade = 'C-'
            elif score >= 40:
                grade = 'D'
            else:
                grade = 'F'
            
        except Exception as e:
            print(f"Error calculating fundamental score: {e}")
            score = 50
            grade = 'C'
        
        return score, grade
    
    def _generate_signals(self, data: pd.DataFrame) -> List[Dict]:
        """Generate comprehensive trading signals"""
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
                if not pd.isna(volume_ratio):
                    if volume_ratio > 2.0:
                        signals.append({'type': 'HIGH_VOLUME', 'strength': 'STRONG', 'direction': 'NEUTRAL'})
                    elif volume_ratio > 1.5:
                        signals.append({'type': 'ABOVE_AVG_VOLUME', 'strength': 'MODERATE', 'direction': 'NEUTRAL'})
            
            # Stochastic signals
            if 'Stoch_%K' in data.columns and 'Stoch_%D' in data.columns:
                stoch_k = latest['Stoch_%K']
                stoch_d = latest['Stoch_%D']
                
                if not pd.isna(stoch_k) and not pd.isna(stoch_d):
                    if stoch_k < 20 and stoch_d < 20:
                        signals.append({'type': 'STOCH_OVERSOLD', 'strength': 'MODERATE', 'direction': 'BULLISH'})
                    elif stoch_k > 80 and stoch_d > 80:
                        signals.append({'type': 'STOCH_OVERBOUGHT', 'strength': 'MODERATE', 'direction': 'BEARISH'})
            
            # A/D Line signals
            if 'AD_Line' in data.columns and len(data) > 1:
                ad_current = latest['AD_Line']
                ad_prev = data['AD_Line'].iloc[-2]
                if not pd.isna(ad_current) and not pd.isna(ad_prev):
                    if ad_current > ad_prev:
                        signals.append({'type': 'AD_BULLISH', 'strength': 'MODERATE', 'direction': 'BULLISH'})
                    else:
                        signals.append({'type': 'AD_BEARISH', 'strength': 'MODERATE', 'direction': 'BEARISH'})
            
            # Williams %R signals
            if 'Williams_%R' in data.columns:
                williams_r = latest['Williams_%R']
                if not pd.isna(williams_r):
                    if williams_r < -80:
                        signals.append({'type': 'WILLIAMS_OVERSOLD', 'strength': 'MODERATE', 'direction': 'BULLISH'})
                    elif williams_r > -20:
                        signals.append({'type': 'WILLIAMS_OVERBOUGHT', 'strength': 'MODERATE', 'direction': 'BEARISH'})
            
            # CCI signals
            if 'CCI' in data.columns:
                cci = latest['CCI']
                if not pd.isna(cci):
                    if cci < -100:
                        signals.append({'type': 'CCI_OVERSOLD', 'strength': 'MODERATE', 'direction': 'BULLISH'})
                    elif cci > 100:
                        signals.append({'type': 'CCI_OVERBOUGHT', 'strength': 'MODERATE', 'direction': 'BEARISH'})
        
        except Exception as e:
            print(f"Error generating signals: {e}")
        
        return signals
    
    def _prepare_summary(self, ticker: str, data: pd.DataFrame, info: Dict, 
                        fundamental: Dict, signals: List, reversal_patterns: Dict) -> str:
        """Prepare comprehensive analysis summary with ALL indicators"""
        try:
            if data.empty:
                return f"**No data available for {ticker}**"
            
            latest = data.iloc[-1]
            current_price = latest['Close']
            prev_close = data['Close'].iloc[-2] if len(data) > 1 else current_price
            price_change = ((current_price - prev_close) / prev_close * 100) if prev_close != 0 else 0
            
            # Get currency
            currency = info.get('currency', 'USD')
            
            # Volume
            volume = latest.get('Volume', 0)
            volume_ratio = latest.get('Volume_Ratio', 1)
            
            # Calculate additional metrics
            if len(data) >= 20:
                price_20d_high = data['Close'].tail(20).max()
                price_20d_low = data['Close'].tail(20).min()
                price_from_20d_high = ((current_price - price_20d_high) / price_20d_high * 100) if price_20d_high != 0 else 0
                price_from_20d_low = ((current_price - price_20d_low) / price_20d_low * 100) if price_20d_low != 0 else 0
            else:
                price_from_20d_high = 0
                price_from_20d_low = 0
            
            # Signals
            bull_signals = [s for s in signals if s['direction'] == 'BULLISH']
            bear_signals = [s for s in signals if s['direction'] == 'BEARISH']
            neutral_signals = [s for s in signals if s['direction'] == 'NEUTRAL']
            
            # Key signals
            death_cross = any(s['type'] == 'DEATH_CROSS' for s in signals)
            golden_cross = any(s['type'] == 'GOLDEN_CROSS' for s in signals)
            rsi_oversold = any(s['type'] == 'RSI_OVERSOLD' for s in signals)
            rsi_overbought = any(s['type'] == 'RSI_OVERBOUGHT' for s in signals)
            
            # Build comprehensive summary
            summary = f"""
📊 **COMPREHENSIVE ANALYSIS: {ticker.upper()}**

**PRICE & VOLUME**
• Price: {current_price:.2f} {currency}
• Daily Change: {price_change:+.2f}%
• Volume: {self._format_number(volume)} ({volume_ratio:.1f}x avg)
• From 20D High: {price_from_20d_high:+.1f}%
• From 20D Low: {price_from_20d_low:+.1f}%

**TREND ANALYSIS**
"""
            
            # Moving averages with distances
            for ma in [9, 20, 50, 100, 200]:
                for ma_type in ['SMA', 'EMA']:
                    ma_col = f'{ma_type}_{ma}'
                    if ma_col in data.columns and not pd.isna(latest[ma_col]):
                        ma_value = latest[ma_col]
                        position = "ABOVE" if current_price > ma_value else "BELOW"
                        distance = abs(current_price - ma_value) / ma_value * 100
                        summary += f"• {ma_type}{ma}: {ma_value:.2f} ({position}, {distance:.1f}%)\n"
            
            if death_cross:
                summary += "• ⚠️ **DEATH CROSS DETECTED** (Strong BEARISH signal)\n"
            if golden_cross:
                summary += "• ✅ **GOLDEN CROSS DETECTED** (Strong BULLISH signal)\n"
            
            # Momentum Indicators
            summary += "\n**MOMENTUM INDICATORS**\n"
            
            if 'RSI' in data.columns and not pd.isna(latest['RSI']):
                rsi = latest['RSI']
                status = "OVERSOLD" if rsi < 30 else "OVERBOUGHT" if rsi > 70 else "NEUTRAL"
                summary += f"• RSI: {rsi:.1f} ({status})\n"
            
            if 'MACD' in data.columns and 'MACD_Signal' in data.columns:
                macd = latest['MACD']
                signal = latest['MACD_Signal']
                if not pd.isna(macd) and not pd.isna(signal):
                    trend = "BULLISH 🟢" if macd > signal else "BEARISH 🔴"
                    diff = macd - signal
                    summary += f"• MACD: {trend} (Diff: {diff:.4f})\n"
            
            if 'Stoch_%K' in data.columns and 'Stoch_%D' in data.columns:
                stoch_k = latest['Stoch_%K']
                stoch_d = latest['Stoch_%D']
                if not pd.isna(stoch_k) and not pd.isna(stoch_d):
                    stoch_status = "OVERSOLD" if stoch_k < 20 and stoch_d < 20 else "OVERBOUGHT" if stoch_k > 80 and stoch_d > 80 else "NEUTRAL"
                    summary += f"• Stochastic: K={stoch_k:.1f}, D={stoch_d:.1f} ({stoch_status})\n"
            
            if 'Williams_%R' in data.columns and not pd.isna(latest['Williams_%R']):
                williams_r = latest['Williams_%R']
                williams_status = "OVERSOLD" if williams_r < -80 else "OVERBOUGHT" if williams_r > -20 else "NEUTRAL"
                summary += f"• Williams %R: {williams_r:.1f} ({williams_status})\n"
            
            if 'CCI' in data.columns and not pd.isna(latest['CCI']):
                cci = latest['CCI']
                cci_status = "OVERSOLD" if cci < -100 else "OVERBOUGHT" if cci > 100 else "NEUTRAL"
                summary += f"• CCI: {cci:.1f} ({cci_status})\n"
            
            if 'ROC_12' in data.columns and not pd.isna(latest['ROC_12']):
                roc = latest['ROC_12']
                summary += f"• ROC (12): {roc:+.2f}%\n"
            
            # Volume & Money Flow Indicators
            summary += "\n**VOLUME & MONEY FLOW**\n"
            
            if 'AD_Line' in data.columns and len(data) > 1:
                ad_current = latest['AD_Line']
                ad_prev = data['AD_Line'].iloc[-2]
                if not pd.isna(ad_current) and not pd.isna(ad_prev):
                    ad_trend = "ACCUMULATION 🟢" if ad_current > ad_prev else "DISTRIBUTION 🔴"
                    ad_change = ((ad_current - ad_prev) / abs(ad_prev) * 100) if ad_prev != 0 else 0
                    summary += f"• A/D Line: {ad_trend} ({ad_change:+.1f}%)\n"
            
            if 'OBV' in data.columns and len(data) > 1:
                obv_current = latest['OBV']
                obv_prev = data['OBV'].iloc[-2]
                if not pd.isna(obv_current) and not pd.isna(obv_prev):
                    obv_trend = "BULLISH 🟢" if obv_current > obv_prev else "BEARISH 🔴"
                    summary += f"• OBV: {obv_trend}\n"
            
            if 'VWAP' in data.columns and not pd.isna(latest['VWAP']):
                vwap = latest['VWAP']
                vwap_position = "ABOVE VWAP 🟢" if current_price > vwap else "BELOW VWAP 🔴"
                vwap_distance = abs(current_price - vwap) / vwap * 100
                summary += f"• VWAP: {vwap:.2f} ({vwap_position}, {vwap_distance:.1f}%)\n"
            
            # Volatility Indicators
            summary += "\n**VOLATILITY**\n"
            
            if all(col in data.columns for col in ['BB_Upper', 'BB_Lower']):
                try:
                    bb_position = (latest['Close'] - latest['BB_Lower']) / (latest['BB_Upper'] - latest['BB_Lower'])
                    if not pd.isna(bb_position):
                        if bb_position < 0.2:
                            bb_status = "NEAR LOWER BAND (Oversold)"
                        elif bb_position > 0.8:
                            bb_status = "NEAR UPPER BAND (Overbought)"
                        else:
                            bb_status = "MID RANGE"
                        summary += f"• Bollinger Bands: {bb_status} ({bb_position*100:.1f}%)\n"
                except:
                    pass
            
            if 'BB_Width' in data.columns and not pd.isna(latest['BB_Width']):
                bb_width = latest['BB_Width']
                bb_width_status = "HIGH VOLATILITY" if bb_width > 10 else "LOW VOLATILITY" if bb_width < 5 else "NORMAL"
                summary += f"• BB Width: {bb_width:.1f}% ({bb_width_status})\n"
            
            if 'ATR_Pct' in data.columns and not pd.isna(latest['ATR_Pct']):
                atr_pct = latest['ATR_Pct']
                atr_status = "HIGH VOL" if atr_pct > 3 else "LOW VOL" if atr_pct < 1 else "NORMAL"
                summary += f"• ATR: {atr_pct:.1f}% of price ({atr_status})\n"
            
            # Reversal patterns
            if reversal_patterns['bullish_reversal']:
                summary += f"\n⚠️ **BULLISH REVERSAL PATTERN DETECTED** ⚠️\n"
                summary += f"• Confidence: {reversal_patterns['confidence']}%\n"
                for signal in reversal_patterns['signals'][:3]:
                    summary += f"• {signal}\n"
            
            if reversal_patterns['bearish_reversal']:
                summary += f"\n⚠️ **BEARISH REVERSAL PATTERN DETECTED** ⚠️\n"
                summary += f"• Confidence: {reversal_patterns['confidence']}%\n"
                for signal in reversal_patterns['signals'][:3]:
                    summary += f"• {signal}\n"
            
            # Signal summary
            summary += f"\n**TECHNICAL SIGNALS**: {len(signals)} total\n"
            summary += f"• Bullish: {len(bull_signals)} 🟢\n"
            summary += f"• Bearish: {len(bear_signals)} 🔴\n"
            summary += f"• Neutral: {len(neutral_signals)} ⚪\n"
            
            # Fundamental Analysis
            summary += f"\n**FUNDAMENTAL ANALYSIS**\n"
            summary += f"• Score: {fundamental['score']}/100 ({fundamental['grade']})\n"
            
            # Key fundamental metrics
            pe = fundamental['valuation'].get('pe_ratio')
            if pe and not pd.isna(pe):
                pe_status = "UNDERVALUED" if pe < 15 else "FAIR" if pe < 25 else "OVERVALUED"
                summary += f"• P/E Ratio: {pe:.1f} ({pe_status})\n"
            
            roe = fundamental['profitability'].get('roe')
            if roe and not pd.isna(roe):
                roe_status = "EXCELLENT" if roe > 0.20 else "GOOD" if roe > 0.15 else "AVERAGE" if roe > 0.10 else "POOR"
                summary += f"• ROE: {roe*100:.1f}% ({roe_status})\n"
            
            debt_equity = fundamental['financial_health'].get('debt_to_equity')
            if debt_equity and not pd.isna(debt_equity):
                de_status = "LOW DEBT" if debt_equity < 0.5 else "MODERATE" if debt_equity < 1.0 else "HIGH DEBT"
                summary += f"• Debt/Equity: {debt_equity:.2f} ({de_status})\n"
            
            dividend_yield = fundamental['dividends'].get('dividend_yield')
            if dividend_yield and not pd.isna(dividend_yield):
                dy_status = "HIGH YIELD" if dividend_yield > 0.04 else "GOOD YIELD" if dividend_yield > 0.02 else "LOW YIELD"
                summary += f"• Dividend Yield: {dividend_yield*100:.2f}% ({dy_status})\n"
            
            # Analyst data
            target_price = fundamental['analyst_data'].get('target_mean_price')
            if target_price and not pd.isna(target_price):
                upside = ((target_price - current_price) / current_price * 100) if current_price > 0 else 0
                summary += f"• Analyst Target: {target_price:.2f} ({upside:+.1f}% upside)\n"
            
            # Overall sentiment
            if reversal_patterns['bullish_reversal']:
                sentiment = '🟢 STRONG BULLISH (Reversal expected)'
            elif reversal_patterns['bearish_reversal']:
                sentiment = '🔴 STRONG BEARISH (Reversal expected)'
            elif len(bull_signals) > len(bear_signals) + 2:
                sentiment = '🟢 BULLISH'
            elif len(bear_signals) > len(bull_signals) + 2:
                sentiment = '🔴 BEARISH'
            else:
                sentiment = '⚪ NEUTRAL'
            
            summary += f"\n**OVERALL SENTIMENT**: {sentiment}"
            
            return summary
            
        except Exception as e:
            print(f"Error preparing summary: {e}")
            import traceback
            traceback.print_exc()
            return f"**ANALYSIS: {ticker}**\n\nError preparing analysis summary: {str(e)[:100]}"
    
    def _prepare_compact_summary(self, ticker: str, data: pd.DataFrame, info: Dict, 
                               fundamental: Dict, signals: List, reversal_patterns: Dict) -> str:
        """Prepare a compact summary for photo captions (under 1024 characters)"""
        try:
            if data.empty:
                return f"**{ticker}** - No data"
            
            latest = data.iloc[-1]
            current_price = latest['Close']
            prev_close = data['Close'].iloc[-2] if len(data) > 1 else current_price
            price_change = ((current_price - prev_close) / prev_close * 100) if prev_close != 0 else 0
            
            # Get key metrics
            rsi = latest.get('RSI', np.nan)
            macd = latest.get('MACD', np.nan)
            macd_signal = latest.get('MACD_Signal', np.nan)
            
            # Signal counts
            bull_signals = len([s for s in signals if s['direction'] == 'BULLISH'])
            bear_signals = len([s for s in signals if s['direction'] == 'BEARISH'])
            
            # MA trend
            ma_trend = ""
            if 'SMA_20' in data.columns and 'SMA_50' in data.columns:
                sma20 = latest['SMA_20']
                sma50 = latest['SMA_50']
                if not pd.isna(sma20) and not pd.isna(sma50):
                    ma_trend = "BULLISH" if sma20 > sma50 else "BEARISH"
            
            # RSI status
            rsi_status = ""
            if not pd.isna(rsi):
                if rsi < 30:
                    rsi_status = "OVERSOLD"
                elif rsi > 70:
                    rsi_status = "OVERBOUGHT"
                else:
                    rsi_status = "NEUTRAL"
            
            # MACD status
            macd_status = ""
            if not pd.isna(macd) and not pd.isna(macd_signal):
                macd_status = "BULLISH" if macd > macd_signal else "BEARISH"
            
            # Build compact summary
            summary = f"📊 **{ticker.upper()}**\n"
            summary += f"💰 Price: ${current_price:.2f} ({price_change:+.2f}%)\n"
            
            if ma_trend:
                summary += f"📉 Trend: {ma_trend}\n"
            
            if not pd.isna(rsi):
                summary += f"📈 RSI: {rsi:.1f} ({rsi_status})\n"
            
            if macd_status:
                summary += f"📊 MACD: {macd_status}\n"
            
            summary += f"🚦 Signals: {bull_signals} 🟢 | {bear_signals} 🔴\n"
            
            # Add fundamental score
            summary += f"⭐ Fundamental: {fundamental['score']}/100 ({fundamental['grade']})\n"
            
            # Add reversal if present
            if reversal_patterns['bullish_reversal']:
                summary += f"⚠️ Bullish Reversal ({reversal_patterns['confidence']}%)"
            elif reversal_patterns['bearish_reversal']:
                summary += f"⚠️ Bearish Reversal ({reversal_patterns['confidence']}%)"
            
            # Overall sentiment
            if reversal_patterns['bullish_reversal']:
                sentiment = '🟢 STRONG BULLISH'
            elif reversal_patterns['bearish_reversal']:
                sentiment = '🔴 STRONG BEARISH'
            elif bull_signals > bear_signals:
                sentiment = '🟢 BULLISH'
            elif bear_signals > bull_signals:
                sentiment = '🔴 BEARISH'
            else:
                sentiment = '⚪ NEUTRAL'
            
            summary += f"\n**Overall**: {sentiment}"
            
            # Ensure it's under 1024 characters
            if len(summary) > 1024:
                summary = summary[:1020] + "..."
            
            return summary
            
        except Exception as e:
            print(f"Error preparing compact summary: {e}")
            return f"📊 **{ticker}**\nAnalysis complete. See details below."
    
    def _format_number(self, num: float) -> str:
        """Format large numbers"""
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