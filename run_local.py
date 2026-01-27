#!/usr/bin/env python3
"""
Local runner for the Trading Bot.
Use this script to run the bot on your local machine.
"""
import os
import sys

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """Main function to run the trading bot locally"""
    print("🚀 Starting Trading Bot in LOCAL mode...")
    
    # Check for required environment variables
    token = os.getenv("TELEGRAM_TOKEN")
    if not token:
        print("❌ ERROR: TELEGRAM_TOKEN environment variable is not set.")
        print("Please set it in your .env file or export it in your shell:")
        print("   export TELEGRAM_TOKEN='your_bot_token_here'")
        print("Or create a .env file with: TELEGRAM_TOKEN=your_bot_token_here")
        sys.exit(1)
    
    # Import and run the bot
    try:
        from bot import TradingBot
        
        bot = TradingBot()
        print("✅ Bot initialized successfully in LOCAL mode")
        print("📱 Bot should respond to Telegram messages now...")
        bot.run()
        
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user (Ctrl+C)")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()