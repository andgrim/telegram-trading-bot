"""
Universal Trading Bot Configuration - Local Version
"""
import os
from typing import Dict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class TradingConfig:
    """Universal configuration for all markets - Local version"""
    
    # Telegram Bot Token
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
    
    # Check if token is set
    if not TELEGRAM_TOKEN:
        print("⚠️ WARNING: TELEGRAM_TOKEN not found in .env file")
        print("The bot will not work without a valid Telegram token.")
        print("Create a .env file with: TELEGRAM_TOKEN=your_token_here")
    
    # Yahoo Finance settings (relaxed for local)
    YAHOO_MAX_RETRIES = 2
    YAHOO_TIMEOUT = 15
    YAHOO_DELAY_SECONDS = 0.5  # Reduced for local
    
    # Cache settings
    CACHE_TTL = 300
    MAX_CACHE_SIZE = 50
    
    # Chart colors
    CHART_COLORS = {
        'background': '#0a0a0a',
        'text': '#ffffff',
        'price_line': '#00FFFF',
        'ma_20': '#00FF00',  # Green for 20 MA
        'ma_50': '#FF4081',
        'ma_200': '#FF9800',
        'volume_up': '#00E676',
        'volume_down': '#FF5252',
        'rsi_line': '#FFEB3B',
        'macd_line': '#00E676',
        'macd_signal': '#FF5252',
        'grid': '#1a1a1a',
        'ad_line': '#7C4DFF',
    }
    
    # Exchange suffixes for info only
    EXCHANGE_SUFFIXES = {
        '.MI': 'Milan', '.PA': 'Paris', '.DE': 'Germany',
        '.AS': 'Amsterdam', '.L': 'London', '.T': 'Tokyo',
        '.HK': 'Hong Kong', '.SS': 'Shanghai', '.SZ': 'Shenzhen',
        '.KS': 'Korea', '.AX': 'Australia', '.V': 'Canada',
        '.TO': 'Canada', '.SW': 'Switzerland',
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
        
        return {'suffix': None, 'exchange': 'US/Unknown', 'base_ticker': ticker_upper}

CONFIG = TradingConfig()