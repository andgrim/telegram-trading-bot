import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    
    # Trading settings
    DEFAULT_PERIODS = [3, 6]  # in months
    DEFAULT_INTERVAL = "1d"
    
    # Technical indicators
    EMA_SHORT = 12
    EMA_LONG = 26
    SIGNAL_PERIOD = 9
    RSI_PERIOD = 14