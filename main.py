#!/usr/bin/env python3
"""
Main entry point for Trading Bot - Render Compatible
"""
import sys
import os
import traceback

# Add current directory to path for Render
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def is_render():
    """Check if running on Render"""
    return os.getenv('RENDER') == 'true'

def main():
    """Main function to run the trading bot"""
    print("🚀 Starting Trading Analysis Bot...")
    
    # Check for required environment variables
    token = os.getenv("TELEGRAM_TOKEN")
    if not token:
        print("❌ ERROR: TELEGRAM_TOKEN environment variable is required")
        print("Please set TELEGRAM_TOKEN in your Render environment variables")
        sys.exit(1)
    
    # Import here to avoid issues during Render build
    try:
        from bot import TradingBot
        
        bot = TradingBot()
        print("✅ Bot initialized successfully")
        bot.run()
        
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()