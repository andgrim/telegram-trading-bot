import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import requests
import json

def get_stock_data(symbol: str, period_months: int = 3, interval: str = "1d") -> pd.DataFrame:
    """Scarica dati da Yahoo Finance"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=period_months * 30)
    
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, interval=interval)
        
        if df.empty:
            print(f"⚠️ Nessun dato trovato per {symbol}")
            return pd.DataFrame()
        
        return df
    except Exception as e:
        print(f"❌ Errore nel recupero dati per {symbol}: {e}")
        return pd.DataFrame()

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calcola indicatori tecnici"""
    if df.empty:
        return df
    
    # Crea una copia per evitare SettingWithCopyWarning
    df = df.copy()
    
    # Media mobile semplice
    df['SMA_20'] = df['Close'].rolling(window=20, min_periods=1).mean()
    df['SMA_50'] = df['Close'].rolling(window=50, min_periods=1).mean()
    
    # Media mobile esponenziale
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
    df['BB_Middle'] = df['Close'].rolling(window=20, min_periods=1).mean()
    bb_std = df['Close'].rolling(window=20, min_periods=1).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    
    return df

def search_tickers(query: str, limit: int = 10) -> list:
    """Cerca ticker per nome o simbolo"""
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}&quotesCount={limit}&newsCount=0"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        data = response.json()
        
        results = []
        if 'quotes' in data:
            for quote in data['quotes']:
                if 'symbol' in quote:
                    results.append({
                        'symbol': quote['symbol'],
                        'name': quote.get('longname', quote.get('shortname', 'N/A')),
                        'exchange': quote.get('exchange', 'N/A'),
                        'type': quote.get('quoteType', 'N/A')
                    })
        
        return results
    
    except Exception as e:
        print(f"❌ Errore nella ricerca: {e}")
        return []

def get_ticker_info(symbol: str) -> dict:
    """Ottiene informazioni dettagliate su un ticker"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        return {
            'symbol': symbol,
            'name': info.get('longName', info.get('shortName', symbol)),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'country': info.get('country', 'N/A'),
            'market_cap': info.get('marketCap', 0),
            'currency': info.get('currency', 'N/A'),
            'exchange': info.get('exchange', 'N/A'),
            'current_price': info.get('currentPrice', info.get('regularMarketPrice', 0)),
            'previous_close': info.get('previousClose', 0),
            'open': info.get('open', 0),
            'day_high': info.get('dayHigh', 0),
            'day_low': info.get('dayLow', 0),
            'volume': info.get('volume', 0),
            'avg_volume': info.get('averageVolume', 0),
            'dividend_yield': info.get('dividendYield', 0),
            'pe_ratio': info.get('trailingPE', 0),
            'beta': info.get('beta', 0),
            '52w_high': info.get('fiftyTwoWeekHigh', 0),
            '52w_low': info.get('fiftyTwoWeekLow', 0)
        }
    except Exception as e:
        print(f"❌ Errore nel recupero info per {symbol}: {e}")
        return {}