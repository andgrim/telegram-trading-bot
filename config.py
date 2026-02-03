"""
Universal Trading Bot Configuration
Webhook version for Render deployment
"""
import os
from typing import Dict

class TradingConfig:
    """Universal configuration for all markets"""
    
    # Telegram Bot Token (set in Render environment)
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
    
    # Webhook settings
    WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "default_secret_change_me")
    
    # Yahoo Finance settings
    YAHOO_RATE_LIMIT = True
    YAHOO_MAX_RETRIES = 3
    YAHOO_DELAY_SECONDS = 1.5
    YAHOO_TIMEOUT = 30
    
    # Chart colors
    CHART_COLORS = {
        'background': '#0a0a0a',
        'text': '#ffffff',
        'price_line': '#00FFFF',
        'ma_20': '#FF4081',
        'ma_50': '#7C4DFF',
        'ma_200': '#FF9800',
        'volume_up': '#00E676',
        'volume_down': '#FF5252',
        'rsi_line': '#FFEB3B',
        'macd_line': '#00E676',
        'macd_signal': '#FF5252',
        'grid': '#1a1a1a',
    }
    
    CHART_STYLE = {
        'price_line_width': 1.5,
        'ma_line_width': 1.2,
        'indicator_line_width': 1.0,
        'title_font_size': 14,
        'label_font_size': 10,
        'dpi': 100,
    }
    
    # Technical analysis settings
    RSI_PERIOD = 14
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    
    # Cache settings
    CACHE_TTL = 600
    MAX_CACHE_SIZE = 50
    
    # Exchange suffixes
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