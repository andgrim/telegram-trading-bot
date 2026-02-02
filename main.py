#!/usr/bin/env python3
"""
Universal Trading Analysis Bot - Main Entry Point
"""
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Start bot
if __name__ == "__main__":
    print("🚀 Universal Trading Analysis Bot")
    print("=" * 50)
    print("🌍 Supports ALL Yahoo Finance markets worldwide")
    print("📊 Complete technical analysis with 20+ indicators")
    print("⏱️ Real-time data from Yahoo Finance")
    print("=" * 50)
    
    try:
        from bot import UniversalTradingBot
        bot = UniversalTradingBot()
        bot.run()
    except Exception as e:
        print(f"❌ Failed to start bot: {e}")
        import traceback
        traceback.print_exc()