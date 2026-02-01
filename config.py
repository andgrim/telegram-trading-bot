import os
from typing import Dict
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables from .env file if it exists (for local development)
if os.path.exists('.env'):
    load_dotenv()

@dataclass
class TradingConfig:
    """Configuration for the trading analysis system."""

    # Telegram Bot Token - Retrieved from environment variable
    TELEGRAM_TOKEN: str = os.getenv("TELEGRAM_TOKEN", "")

    # Environment detection
    IS_RENDER: bool = os.getenv('RENDER') == 'true'
    IS_LOCAL: bool = not IS_RENDER

    # Rate limiting settings for Yahoo Finance
    YAHOO_RATE_LIMIT: bool = True
    YAHOO_MAX_RETRIES: int = 2
    YAHOO_DELAY_SECONDS: float = 2.0 if IS_RENDER else 1.0
    YAHOO_TIMEOUT: int = 15

    # Chart settings
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

    # Caching settings - MUST be defined with field() 
    CACHE_TTL: int = field(default=300)  # 5 minutes

    def __post_init__(self):
        """Initialize attributes after the object is created."""
        if self.CHART_COLORS is None:
            self.CHART_COLORS = self._get_complete_color_scheme()
        if self.CHART_STYLE is None:
            self.CHART_STYLE = self._get_chart_style()
        if self.REVERSAL_SETTINGS is None:
            self.REVERSAL_SETTINGS = {
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'volume_spike_threshold': 1.5,
            }

    def _get_complete_color_scheme(self) -> Dict:
        """Returns a complete color scheme dictionary for all chart elements."""
        return {
            # Background and Base
            'background': '#0a0a0a',
            'grid': '#1a1a1a',
            'text': '#ffffff',

            # Price and Lines
            'price_line': '#40E0D0',  # Turquoise

            # Moving Averages
            'ma_9': '#00FF9D',    # Bright Green
            'ma_20': '#FF6B9D',   # Pink
            'ma_50': '#9D4EDD',   # Purple
            'ma_100': '#FFD700',  # Gold
            'ma_200': '#FF8C00',  # Dark Orange

            # Bollinger Bands
            'bb_upper': '#FF00AA',
            'bb_lower': '#00FFAA',
            'bb_middle': '#AAAAAA',

            # Volume
            'volume_up': '#00FF9D',
            'volume_down': '#FF0080',

            # Technical Indicators
            'rsi_line': '#FFFF00',      # Yellow
            'macd_line': '#00FF9D',     # Green
            'macd_signal': '#FF0080',   # Red
            'macd_hist_positive': '#00FF9D',
            'macd_hist_negative': '#FF0080',
            'ad_line': '#00E0FF',       # Light Blue

            # Signals and Highlights
            'reversal_signal': '#FFAA00',
            'oversold_area': '#00FF9D22',
            'overbought_area': '#FF008022',
        }

    def _get_chart_style(self) -> Dict:
        """Returns styling parameters for charts."""
        return {
            'price_line_width': 1.2,
            'ma_line_width': 0.9,
            'indicator_line_width': 0.8,
            'grid_alpha': 0.15,
            'title_font_size': 14,
            'label_font_size': 10,
            'dpi': 100 if self.IS_RENDER else 120,
            'figure_size': (14, 10),
        }

# Create a single global configuration instance
CONFIG = TradingConfig()

# Optional: Quick validation log (visible in local terminal)
if __name__ == "__main__":
    print(f"Configuration loaded. IS_RENDER: {CONFIG.IS_RENDER}")
    print(f"CHART_COLORS contains 'ma_9': {'ma_9' in CONFIG.CHART_COLORS}")
    print(f"Cache TTL: {CONFIG.CACHE_TTL} seconds")
    print(f"Yahoo delay: {CONFIG.YAHOO_DELAY_SECONDS} seconds")
    print(f"Total config attributes: {len(vars(CONFIG))}")