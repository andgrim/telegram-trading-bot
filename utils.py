import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_stock_data(symbol: str, period_months: int = 3, interval: str = "1d") -> pd.DataFrame:
    """Download data from Yahoo Finance with improved error handling"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_months * 30)
        
        logger.info(f"📥 Downloading {symbol} data: {start_date.date()} to {end_date.date()} ({period_months} months)")
        
        ticker = yf.Ticker(symbol)
        
        # Try different intervals if the default fails
        try:
            df = ticker.history(start=start_date, end=end_date, interval=interval)
        except:
            # Fallback to daily data
            df = ticker.history(start=start_date, end=end_date, interval="1d")
        
        if df.empty:
            logger.warning(f"No data returned for {symbol}")
            # Try with a longer period to get at least some data
            start_date = end_date - timedelta(days=period_months * 31)
            df = ticker.history(start=start_date, end=end_date, interval="1d")
        
        logger.info(f"✅ Downloaded {len(df)} rows for {symbol}")
        return df if not df.empty else pd.DataFrame()
        
    except Exception as e:
        logger.error(f"Error get_stock_data {symbol}: {e}")
        return pd.DataFrame()

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators with improved calculations"""
    if df.empty or len(df) < 20:
        return df
    
    df = df.copy()
    
    # Ensure we have enough data
    min_periods = min(20, len(df))
    
    # Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20, min_periods=min_periods).mean()
    df['SMA_50'] = df['Close'].rolling(window=50, min_periods=min(50, len(df))).mean()
    
    # EMA for MACD
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False, min_periods=1).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False, min_periods=1).mean()
    
    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False, min_periods=1).mean()
    df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Fill NaN values with forward/backward fill
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    
    return df

def search_tickers(query: str, limit: int = 10) -> list:
    """Search for tickers with improved error handling"""
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}&quotesCount={limit}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        data = response.json()
        
        results = []
        if 'quotes' in data:
            for quote in data['quotes'][:limit]:
                if 'symbol' in quote and 'longname' in quote:
                    results.append({
                        'symbol': quote['symbol'],
                        'name': quote.get('longname', quote.get('shortname', quote['symbol'])),
                        'exchange': quote.get('exchange', 'N/A'),
                        'type': quote.get('quoteType', 'N/A')
                    })
        
        logger.info(f"🔍 Search for '{query}' found {len(results)} results")
        return results
        
    except Exception as e:
        logger.error(f"Error searching {query}: {e}")
        return []

def get_ticker_info(symbol: str) -> dict:
    """Get ticker information with improved error handling"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # Get current price
        hist = ticker.history(period="1d")
        current_price = hist['Close'].iloc[-1] if not hist.empty else info.get('currentPrice', 0)
        
        # Calculate daily change if possible
        if not hist.empty and len(hist) > 1:
            prev_close = hist['Close'].iloc[-2]
            daily_change_pct = ((current_price - prev_close) / prev_close) * 100
        else:
            daily_change_pct = 0
        
        return {
            'symbol': symbol,
            'name': info.get('longName', info.get('shortName', symbol)),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'current_price': current_price,
            'daily_change': daily_change_pct,
            'previous_close': info.get('previousClose', current_price),
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('trailingPE', 'N/A'),
            'dividend_yield': info.get('dividendYield', 0),
            'volume': info.get('volume', 0),
            'avg_volume': info.get('averageVolume', 0),
            '52w_high': info.get('fiftyTwoWeekHigh', 0),
            '52w_low': info.get('fiftyTwoWeekLow', 0),
            'beta': info.get('beta', 'N/A'),
            'country': info.get('country', 'N/A'),
            'currency': info.get('currency', 'USD'),
            'website': info.get('website', 'N/A'),
            'description': info.get('longBusinessSummary', 'N/A')[:200] + '...' if info.get('longBusinessSummary') else 'N/A'
        }
        
    except Exception as e:
        logger.error(f"Error getting info for {symbol}: {e}")
        return {}