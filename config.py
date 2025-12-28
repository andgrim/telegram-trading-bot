import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Telegram Bot Token
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
    
    # App Info
    APP_NAME = "Trading Analysis Bot"
    VERSION = "4.0.0"
    
    # Trading settings
    DEFAULT_PERIODS = [1, 3, 6]
    DEFAULT_INTERVAL = "1d"