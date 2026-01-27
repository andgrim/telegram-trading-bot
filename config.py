import os
from typing import Dict, List
from dataclasses import dataclass
from dotenv import load_dotenv

# Only load .env file if not on Render
if not os.getenv('RENDER'):
    load_dotenv()

@dataclass
class TradingConfig:
    # Configuration remains mostly the same
    TELEGRAM_TOKEN: str = os.getenv("TELEGRAM_TOKEN", "")
    
    # ... rest of your config ...
    # Add a property to check if running on Render
    @property
    def is_render(self):
        return os.getenv('RENDER') == 'true'

CONFIG = TradingConfig()