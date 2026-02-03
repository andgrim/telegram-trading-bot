"""
Main entry point for the Universal Trading Bot (Local Version)
"""
import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Check if token is loaded
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN:
    print("❌ ERROR: TELEGRAM_TOKEN not found in .env file")
    print("Please create a .env file with: TELEGRAM_TOKEN=your_token_here")
    exit(1)

print(f"✅ Telegram Token loaded: {TELEGRAM_TOKEN[:10]}...")

from bot import UniversalTradingBot

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def main():
    """Main function to start the bot"""
    try:
        logger.info("Starting Universal Trading Bot (Local Version)...")
        
        bot = UniversalTradingBot()
        bot.run()
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()