"""
Main entry point for Trading Analysis Bot with Extended Timeframes
"""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bot import TradingBot

def main():
    """Main function to run the trading bot"""
    print("🚀 Starting Trading Analysis Bot with Extended Timeframes...")
    
    # Check for required environment variables
    if not os.getenv("TELEGRAM_TOKEN"):
        print("❌ ERROR: TELEGRAM_TOKEN environment variable is required")
        print("Please create a .env file with your Telegram Bot Token")
        print("Format: TELEGRAM_TOKEN=your_token_here")
        sys.exit(1)
    
    # Initialize and run the bot
    bot = TradingBot()
    bot.run()

if __name__ == "__main__":
    main()