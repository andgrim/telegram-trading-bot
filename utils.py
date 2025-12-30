import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import requests

def get_stock_data(symbol: str, period_months: int = 3, interval: str = "1d") -> pd.DataFrame:
    """Download data from Yahoo Finance"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_months * 30)
        
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, interval=interval)
        
        return df if not df.empty else pd.DataFrame()
        
    except Exception as e:
        print(f"Error get_stock_data {symbol}: {e}")
        return pd.DataFrame()

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators"""
    if df.empty:
        return df
    
    df = df.copy()
    
    # Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20, min_periods=1).mean()
    df['SMA_50'] = df['Close'].rolling(window=50, min_periods=1).mean()
    
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
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    
    return df

def search_tickers(query: str, limit: int = 5) -> list:
    """Search for tickers"""
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}&quotesCount={limit}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        data = response.json()
        
        results = []
        if 'quotes' in data:
            for quote in data['quotes'][:limit]:
                if 'symbol' in quote:
                    results.append({
                        'symbol': quote['symbol'],
                        'name': quote.get('longname', quote.get('shortname', quote['symbol'])),
                        'exchange': quote.get('exchange', 'N/A'),
                    })
        
        return results
        
    except Exception as e:
        print(f"Error searching {query}: {e}")
        return []

def get_ticker_info(symbol: str) -> dict:
    """Get ticker information"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # Get current price
        hist = ticker.history(period="1d")
        current_price = hist['Close'].iloc[-1] if not hist.empty else 0
        
        return {
            'symbol': symbol,
            'name': info.get('longName', info.get('shortName', symbol)),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'current_price': current_price,
            'previous_close': info.get('previousClose', current_price),
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('trailingPE', 'N/A'),
            'dividend_yield': info.get('dividendYield', 0),
            'volume': info.get('volume', 0),
            '52w_high': info.get('fiftyTwoWeekHigh', 0),
            '52w_low': info.get('fiftyTwoWeekLow', 0),
            'beta': info.get('beta', 'N/A'),
            'country': info.get('country', 'N/A'),
            'currency': info.get('currency', 'USD')
        }
        
    except Exception as e:
        print(f"Error getting info for {symbol}: {e}")
        return {}