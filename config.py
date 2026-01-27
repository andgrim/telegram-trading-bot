import os
from typing import Dict
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Carica le variabili d'ambiente solo in locale
if os.path.exists('.env'):
    load_dotenv()

@dataclass
class TradingConfig:
    """Configuration for trading analysis system"""
    
    # Telegram Bot Token
    TELEGRAM_TOKEN: str = os.getenv("TELEGRAM_TOKEN", "")
    
    # Environment detection
    IS_RENDER: bool = os.getenv('RENDER') == 'true'
    IS_LOCAL: bool = not IS_RENDER
    
    # Chart settings - IMPORTANTE: usare field() per inizializzazione differita
    CHART_COLORS: Dict = field(default_factory=lambda: None)
    CHART_STYLE: Dict = field(default_factory=lambda: None)
    
    # Technical analysis settings
    RSI_PERIOD: int = 14
    MACD_FAST: int = 12
    MACD_SLOW: int = 26
    MACD_SIGNAL: int = 9
    BB_PERIOD: int = 20
    BB_STD: int = 2
    
    # Reversal detection settings
    REVERSAL_SETTINGS: Dict = field(default_factory=lambda: None)
    
    def __post_init__(self):
        """Initialize attributes after the object is created"""
        # Initialize CHART_COLORS if None
        if self.CHART_COLORS is None:
            self.CHART_COLORS = self._get_clean_colors()
        
        # Initialize CHART_STYLE if None
        if self.CHART_STYLE is None:
            self.CHART_STYLE = self._get_chart_style()
        
        # Initialize REVERSAL_SETTINGS if None
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
            'dpi': 100 if self.IS_RENDER else 120,
        }

# Create a global instance
CONFIG = TradingConfig()