"""
Universal Trading Bot Configuration
Supports all Yahoo Finance markets worldwide
"""
import os
from typing import Dict

# Environment detection
IS_RENDER = os.getenv('RENDER') == 'true'
IS_LOCAL = not IS_RENDER

class TradingConfig:
    """Universal configuration for all markets"""
    
    # Telegram Bot Token
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
    
    # Yahoo Finance settings - Optimized for reliability
    YAHOO_RATE_LIMIT = True
    YAHOO_MAX_RETRIES = 3
    YAHOO_DELAY_SECONDS = 1.5  # Balanced delay
    YAHOO_TIMEOUT = 30
    
    # Environment flags
    IS_RENDER = IS_RENDER
    IS_LOCAL = IS_LOCAL
    
    # Chart colors - Enhanced for better visibility
    CHART_COLORS = {
        'background': '#0a0a0a',
        'text': '#ffffff',
        'price_line': '#00FFFF',  # Bright Cyan
        'ma_20': '#FF4081',      # Bright Pink
        'ma_50': '#7C4DFF',      # Purple
        'ma_200': '#FF9800',     # Orange
        'volume_up': '#00E676',  # Bright Green
        'volume_down': '#FF5252', # Bright Red
        'rsi_line': '#FFEB3B',   # Yellow
        'macd_line': '#00E676',  # Green
        'macd_signal': '#FF5252', # Red
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
    STOCH_PERIOD = 14
    BB_PERIOD = 20
    BB_STD = 2
    
    # Cache settings
    CACHE_TTL = 600  # 10 minutes
    MAX_CACHE_SIZE = 100
    
    # Exchange suffixes for automatic detection (expanded)
    EXCHANGE_SUFFIXES = {
        # Europe
        '.MI': 'Milan',
        '.PA': 'Paris',
        '.DE': 'Germany',
        '.AS': 'Amsterdam',
        '.L': 'London',
        '.SW': 'Switzerland',
        '.BR': 'Brussels',
        '.IR': 'Ireland',
        '.MC': 'Madrid',
        '.CO': 'Copenhagen',
        '.OL': 'Oslo',
        '.ST': 'Stockholm',
        '.HE': 'Helsinki',
        '.VI': 'Vienna',
        '.AT': 'Athens',
        '.LS': 'Lisbon',
        
        # Asia
        '.T': 'Tokyo',
        '.HK': 'Hong Kong',
        '.SS': 'Shanghai',
        '.SZ': 'Shenzhen',
        '.KS': 'Korea',
        '.TW': 'Taiwan',
        '.SI': 'Singapore',
        '.JK': 'Jakarta',
        '.BK': 'Bangkok',
        '.NS': 'India',
        
        # Americas
        '.AX': 'Australia',
        '.V': 'Vancouver',
        '.TO': 'Toronto',
        '.CN': 'Canada',
        '.MX': 'Mexico',
        '.SA': 'São Paulo',
        '.BA': 'Buenos Aires',
    }
    
    @classmethod
    def get_exchange_info(cls, ticker: str) -> Dict:
        """Get exchange information for ticker"""
        ticker_upper = ticker.upper()
        
        # Check for special formats first
        if ticker_upper.startswith('^'):
            return {'suffix': None, 'exchange': 'Index', 'base_ticker': ticker_upper}
        elif '=F' in ticker_upper:
            return {'suffix': None, 'exchange': 'Futures', 'base_ticker': ticker_upper}
        elif '-USD' in ticker_upper or '-EUR' in ticker_upper:
            return {'suffix': None, 'exchange': 'Cryptocurrency', 'base_ticker': ticker_upper}
        elif '=X' in ticker_upper:
            return {'suffix': None, 'exchange': 'Forex', 'base_ticker': ticker_upper}
        elif '.SS' in ticker_upper or '.SZ' in ticker_upper:
            return {'suffix': None, 'exchange': 'China', 'base_ticker': ticker_upper}
        elif '.KS' in ticker_upper:
            return {'suffix': None, 'exchange': 'Korea', 'base_ticker': ticker_upper}
        elif '.NS' in ticker_upper:
            return {'suffix': None, 'exchange': 'India', 'base_ticker': ticker_upper}
        
        # Check exchange suffixes
        for suffix, exchange in cls.EXCHANGE_SUFFIXES.items():
            if ticker_upper.endswith(suffix):
                return {
                    'suffix': suffix,
                    'exchange': exchange,
                    'base_ticker': ticker_upper.replace(suffix, '')
                }
        
        # Default to US/Unknown
        return {'suffix': None, 'exchange': 'US/Unknown', 'base_ticker': ticker_upper}

CONFIG = TradingConfig()

if __name__ == "__main__":
    print(f"✅ Universal Config loaded")
    print(f"📊 Supports ALL Yahoo Finance tickers")
    print(f"⏱️ Yahoo delay: {CONFIG.YAHOO_DELAY_SECONDS}s")
    print(f"🌍 {len(CONFIG.EXCHANGE_SUFFIXES)} exchange suffixes supported")
    print(f"📈 Available periods: 3m, 6m, 1y, 2y, 3y, 5y, max")