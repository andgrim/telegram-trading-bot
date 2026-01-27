#!/usr/bin/env python3
"""
Main entry point for Render deployment - Hybrid Web Service
"""
import os
import sys
from dotenv import load_dotenv

# Check if we're running locally (for development)
IS_RENDER = os.getenv('RENDER') == 'true'

if not IS_RENDER:
    print("⚠️  Detected LOCAL environment, but using Render's main.py.")
    print("   For local development, use: python run_local.py")
    print("   Continuing with hybrid mode for testing...")
    # Load .env file for local development
    load_dotenv()

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import Flask only if needed for Render
if IS_RENDER:
    import threading
    from flask import Flask
    
    # Initialize Flask app
    app = Flask(__name__)
    
    # Health Check endpoint - REQUIRED by Render
    @app.route('/')
    def health_check():
        return '✅ Trading Bot is running on Render', 200
    
    @app.route('/health')
    def health():
        return {'status': 'healthy', 'environment': 'render'}, 200
    
    def run_flask_server():
        """Run Flask server on Render-provided port"""
        port = int(os.environ.get("PORT", 10000))
        app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)

def run_telegram_bot():
    """Run your Telegram bot logic"""
    print("🤖 Starting Telegram Bot...")
    try:
        from bot import TradingBot
        bot = TradingBot()
        bot.run()
    except Exception as e:
        print(f"❌ Error starting bot: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if IS_RENDER:
        print("🌐 Starting hybrid service for Render (Web Server + Telegram Bot)...")
        # Start Flask in a separate thread
        flask_thread = threading.Thread(target=run_flask_server, daemon=True)
        flask_thread.start()
        print(f"📡 Flask server started on background thread.")
    else:
        print("💻 Running in development mode (local execution)")
    
    # Start the Telegram bot in the main thread
    run_telegram_bot()