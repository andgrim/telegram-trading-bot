"""
Main entry point for the Universal Trading Bot (Local Version)
"""
import logging
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