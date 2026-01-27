import os
from typing import Dict, List
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class TradingConfig:
    """Configuration for trading analysis system with extended timeframe support"""
    
    # Telegram Bot Token
    TELEGRAM_TOKEN: str = os.getenv("TELEGRAM_TOKEN", "")
    
    # Chart settings
    CHART_THEME: str = "dark"
    CHART_COLORS: Dict = None
    CHART_STYLE: Dict = None
    
    # Technical analysis settings
    MOVING_AVERAGES: List[int] = None
    RSI_PERIOD: int = 14
    MACD_FAST: int = 12
    MACD_SLOW: int = 26
    MACD_SIGNAL: int = 9
    BB_PERIOD: int = 20
    BB_STD: int = 2
    
    # Reversal detection settings
    REVERSAL_SETTINGS: Dict = None
    
    # Extended timeframes
    TIME_PERIODS: List[str] = None
    
    # yfinance settings
    YFINANCE_MAX_RETRIES: int = 5
    YFINANCE_RETRY_DELAY: int = 2
    
    def __post_init__(self):
        if self.CHART_COLORS is None:
            self.CHART_COLORS = self._get_clean_colors()
        
        if self.CHART_STYLE is None:
            self.CHART_STYLE = self._get_chart_style()
        
        if self.MOVING_AVERAGES is None:
            self.MOVING_AVERAGES = [9, 20, 50, 100, 200]  # Added 100 and 200 for long-term analysis
        
        if self.REVERSAL_SETTINGS is None:
            self.REVERSAL_SETTINGS = {
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'volume_spike_threshold': 1.5,
                'ad_divergence_lookback': 5,
                'macd_histogram_shrinking_periods': 3
            }
        
        if self.TIME_PERIODS is None:
            self.TIME_PERIODS = ['3m', '6m', '1y', '2y', '3y', '5y']
    
    def _get_clean_colors(self) -> Dict:
        """Return clean color scheme for simplified charts"""
        return {
            'background': '#0a0a0a',
            'grid': '#1a1a1a',
            'text': '#ffffff',
            'price_line': '#40E0D0',
            'ma_9': '#00FF9D',
            'ma_20': '#FF6B9D',
            'ma_50': '#9D4EDD',
            'ma_100': '#FFD700',  # Gold for 100 MA
            'ma_200': '#FF8C00',  # Dark orange for 200 MA
            'volume_up': '#00FF9D',
            'volume_down': '#FF0080',
            'ad_line': '#00E0FF',
            'rsi_line': '#FFFF00',
            'macd_line': '#00FF9D',
            'macd_signal': '#FF0080',
            'bb_upper': '#FF00AA',
            'bb_lower': '#00FFAA',
            'reversal_signal': '#FFAA00',
        }
    
    def _get_chart_style(self) -> Dict:
        """Return chart style with thin lines"""
        return {
            'price_line_width': 1.0,
            'ma_line_width': 0.8,
            'indicator_line_width': 0.8,
            'grid_alpha': 0.1,
            'font_size': 9,
            'title_size': 12,
            'figure_size': (14, 10),
            'dpi': 120,
        }

CONFIG = TradingConfig()