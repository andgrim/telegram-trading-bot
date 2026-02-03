"""
Universal Trading Bot Configuration - Complete Technical Analysis Version
"""
import os
from typing import Dict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class TradingConfig:
    """Universal configuration for all markets - Complete version"""
    
    # Telegram Bot Token
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
    
    # Check if token is set
    if not TELEGRAM_TOKEN:
        print("⚠️ WARNING: TELEGRAM_TOKEN not found in .env file")
        print("The bot will not work without a valid Telegram token.")
        print("Create a .env file with: TELEGRAM_TOKEN=your_token_here")
    
    # Yahoo Finance settings (relaxed for local)
    YAHOO_MAX_RETRIES = 2
    YAHOO_TIMEOUT = 20
    YAHOO_DELAY_SECONDS = 0.3  # Reduced for local
    
    # Cache settings
    CACHE_TTL = 600  # 10 minutes
    MAX_CACHE_SIZE = 100
    
    # Chart colors - Complete set
    CHART_COLORS = {
        'background': '#0a0a0a',
        'text': '#ffffff',
        'price_line': '#00FFFF',  # Cyan
        'ma_20': '#00FF00',      # Green for 20 MA
        'ma_50': '#FF4081',      # Pink for 50 MA
        'ma_200': '#FF9800',     # Orange for 200 MA
        'volume_up': '#00E676',  # Green for up volume
        'volume_down': '#FF5252', # Red for down volume
        'rsi_line': '#FFEB3B',   # Yellow for RSI
        'macd_line': '#00E676',  # Green for MACD line
        'macd_signal': '#FF5252', # Red for MACD signal
        'ad_line': '#7C4DFF',    # Purple for A/D Line
        'grid': '#1a1a1a',
        'bb_upper': '#FF9800',   # Orange for BB upper
        'bb_lower': '#2196F3',   # Blue for BB lower
        'bb_middle': '#9C27B0',  # Purple for BB middle
        'bb_fill': '#4CAF50',    # Green fill for BB area
    }
    
    # Technical analysis settings
    RSI_PERIOD = 14
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    BB_PERIOD = 20
    BB_STD = 2
    STOCH_PERIOD = 14
    WILLIAMS_PERIOD = 14
    CCI_PERIOD = 20
    
    # Exchange suffixes for info only
    EXCHANGE_SUFFIXES = {
        '.MI': 'Milan', '.PA': 'Paris', '.DE': 'Germany',
        '.AS': 'Amsterdam', '.L': 'London', '.T': 'Tokyo',
        '.HK': 'Hong Kong', '.SS': 'Shanghai', '.SZ': 'Shenzhen',
        '.KS': 'Korea', '.AX': 'Australia', '.V': 'Canada',
        '.TO': 'Canada', '.SW': 'Switzerland',
        '.BR': 'Brazil', '.MX': 'Mexico', '.IL': 'Israel',
    }
    
    @classmethod
    def get_exchange_info(cls, ticker: str) -> Dict:
        """Get exchange information for ticker"""
        ticker_upper = ticker.upper()
        
        for suffix, exchange in cls.EXCHANGE_SUFFIXES.items():
            if ticker_upper.endswith(suffix):
                return {
                    'suffix': suffix,
                    'exchange': exchange,
                    'base_ticker': ticker_upper.replace(suffix, '')
                }
        
        # Special formats
        if ticker_upper.startswith('^'):
            return {'suffix': None, 'exchange': 'Index', 'base_ticker': ticker_upper}
        elif '=F' in ticker_upper:
            return {'suffix': None, 'exchange': 'Futures', 'base_ticker': ticker_upper}
        elif '-USD' in ticker_upper:
            return {'suffix': None, 'exchange': 'Crypto', 'base_ticker': ticker_upper}
        elif '.BO' in ticker_upper or '.NS' in ticker_upper:
            return {'suffix': None, 'exchange': 'India', 'base_ticker': ticker_upper}
        
        return {'suffix': None, 'exchange': 'US/Unknown', 'base_ticker': ticker_upper}

CONFIG = TradingConfig()