import os
from typing import Dict, List
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
# This is for local development only
if os.path.exists('.env'):
    load_dotenv()

@dataclass
class TradingConfig:
    """Configuration for trading analysis system"""
    
    # Telegram Bot Token - from environment variable
    TELEGRAM_TOKEN: str = os.getenv("TELEGRAM_TOKEN", "")
    
    # Environment detection
    IS_RENDER: bool = os.getenv('RENDER') == 'true'
    IS_LOCAL: bool = not IS_RENDER
    
    # Chart settings
    CHART_THEME: str = "dark"
    CHART_COLORS: Dict = None
    CHART_STYLE: Dict = None
    
    # Technical analysis settings
    RSI_PERIOD: int = 14
    MACD_FAST: int = 12
    MACD_SLOW: int = 26
    MACD_SIGNAL: int = 9
    BB_PERIOD: int = 20
    BB_STD: int = 2
    
    # Reversal detection settings
    REVERSAL_SETTINGS: Dict = None
    
    def __post_init__(self):
        # Initialize settings
        if self.CHART_COLORS is None:
            self.CHART_COLORS = self._get_clean_colors()
        
        if self.CHART_STYLE is None:
            self.CHART_STYLE = self._get_chart_style()
        
        if self.REVERSAL_SETTINGS is None:
            self.REVERSAL_SETTINGS = {
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'volume_spike_threshold': 1.5,
            }
    
    def _get_clean_colors(self) -> Dict:
        """Return clean color scheme"""
        return {
            'background': '#0a0a0a',
            'grid': '#1a1a1a',
            'text': '#ffffff',
            'price_line': '#40E0D0',
            'ma_20': '#FF6B9D',
            'ma_50': '#9D4EDD',
            'volume_up': '#00FF9D',
            'volume_down': '#FF0080',
        }
    
    def _get_chart_style(self) -> Dict:
        """Return chart style"""
        return {
            'price_line_width': 1.0,
            'ma_line_width': 0.8,
            'dpi': 100 if self.IS_RENDER else 120,  # Lower DPI for Render to save resources
        }

CONFIG = TradingConfig()