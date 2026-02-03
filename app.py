"""
Main Application for Universal Trading Bot
Handles Flask web server and Telegram webhooks for Render deployment
"""
import os
import logging
from flask import Flask, request, jsonify
import threading

# Import bot functionality
from bot import UniversalTradingBot

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Initialize bot IMMEDIATELY
bot = UniversalTradingBot()
bot_initialized = bot.initialize()

if not bot_initialized:
    logger.error("FAILED TO INITIALIZE BOT! Check TELEGRAM_TOKEN")
else:
    logger.info("✅ Bot initialized successfully")

@app.route('/')
def index():
    """Home page"""
    return jsonify({
        'status': 'online',
        'service': 'Universal Trading Analysis Bot',
        'version': '2.0.0',
        'architecture': 'webhook',
        'instructions': 'This is a Telegram bot backend. Access via Telegram @your_bot_username',
        'bot_initialized': bot_initialized
    })

@app.route('/health')
def health():
    """Health check endpoint for Render"""
    return jsonify({
        'status': 'healthy' if bot_initialized else 'warning',
        'service': 'trading-bot',
        'bot_initialized': bot_initialized
    }), 200

@app.route('/webhook/<secret>', methods=['POST'])
def webhook(secret):
    """Telegram webhook endpoint"""
    # Verify webhook secret
    expected_secret = os.getenv('WEBHOOK_SECRET', 'default_secret_change_me')
    if secret != expected_secret:
        logger.warning(f"Invalid webhook secret: {secret}")
        return jsonify({'error': 'Invalid secret'}), 403
    
    if not bot_initialized:
        logger.error("Bot not initialized, rejecting webhook")
        return jsonify({'error': 'Bot not ready'}), 503
    
    try:
        # Process update
        update_data = request.get_json()
        logger.info(f"Received webhook update: {update_data.get('update_id')}")
        
        # Process update in background thread
        threading.Thread(
            target=bot.process_webhook_update,
            args=(update_data,),
            daemon=True
        ).start()
        
        return jsonify({'status': 'ok'}), 200
    except Exception as e:
        logger.error(f"Webhook processing error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/debug')
def debug():
    """Debug endpoint"""
    return jsonify({
        'bot_initialized': bot_initialized,
        'webhook_secret_set': bool(os.getenv('WEBHOOK_SECRET')),
        'telegram_token_set': bool(os.getenv('TELEGRAM_TOKEN')),
        'environment': os.environ.get('RENDER', 'Not on Render')
    })

if __name__ == '__main__':
    port = int(os.getenv('PORT', 10000))
    
    logger.info(f"Starting Flask server on port {port}")
    logger.info(f"Bot initialized: {bot_initialized}")
    
    app.run(host='0.0.0.0', port=port, debug=False)