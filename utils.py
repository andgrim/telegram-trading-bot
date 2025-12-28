import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import requests

def get_stock_data(symbol: str, period_months: int = 3, interval: str = "1d") -> pd.DataFrame:
    """Scarica dati da Yahoo Finance"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_months * 30)
        
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, interval=interval)
        
        return df if not df.empty else pd.DataFrame()
        
    except Exception as e:
        print(f"Errore get_stock_data {symbol}: {e}")
        return pd.DataFrame()

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calcola indicatori tecnici"""
    if df.empty:
        return df
    
    df = df.copy()
    
    # Medie mobili
    df['SMA_20'] = df['Close'].rolling(window=20, min_periods=1).mean()
    df['SMA_50'] = df['Close'].rolling(window=50, min_periods=1).mean()
    
    # EMA per MACD
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False, min_periods=1).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False, min_periods=1).mean()
    
    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False, min_periods=1).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df

def search_tickers(query: str, limit: int = 5) -> list:
    """Cerca ticker"""
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
        print(f"Errore ricerca {query}: {e}")
        return []

def get_ticker_info(symbol: str) -> dict:
    """Ottiene informazioni ticker"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        return {
            'symbol': symbol,
            'name': info.get('longName', info.get('shortName', symbol)),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'current_price': info.get('currentPrice', info.get('regularMarketPrice', 0)),
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('trailingPE', 'N/A'),
            'dividend_yield': info.get('dividendYield', 0),
            'volume': info.get('volume', 0),
            '52w_high': info.get('fiftyTwoWeekHigh', 0),
            '52w_low': info.get('fiftyTwoWeekLow', 0),
            'beta': info.get('beta', 'N/A'),
        }
        
    except Exception as e:
        print(f"Errore info {symbol}: {e}")
        return {}