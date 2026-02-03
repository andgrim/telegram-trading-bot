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

# Initialize bot
bot = UniversalTradingBot()

@app.route('/')
def index():
    """Home page"""
    return jsonify({
        'status': 'online',
        'service': 'Universal Trading Analysis Bot',
        'version': '2.0.0',
        'architecture': 'webhook',
        'instructions': 'This is a Telegram bot backend. Access via Telegram @your_bot_username'
    })

@app.route('/health')
def health():
    """Health check endpoint for Render"""
    return jsonify({'status': 'healthy', 'service': 'trading-bot'}), 200

@app.route('/webhook/<secret>', methods=['POST'])
def webhook(secret):
    """Telegram webhook endpoint"""
    # Verify webhook secret
    expected_secret = os.getenv('WEBHOOK_SECRET', 'default_secret_change_me')
    if secret != expected_secret:
        logger.warning(f"Invalid webhook secret: {secret}")
        return jsonify({'error': 'Invalid secret'}), 403
    
    try:
        # Process update
        update_data = request.get_json()
        logger.info(f"Received webhook update: {update_data.get('update_id')}")
        
        # Process update in background thread
        threading.Thread(
            target=bot.process_webhook_update,
            args=(update_data,)
        ).start()
        
        return jsonify({'status': 'ok'}), 200
    except Exception as e:
        logger.error(f"Webhook processing error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/set_webhook', methods=['POST'])
def set_webhook():
    """Manually set webhook endpoint (for initial setup)"""
    try:
        bot_url = os.getenv('RENDER_EXTERNAL_URL')
        secret = os.getenv('WEBHOOK_SECRET')
        
        if not bot_url or not secret:
            return jsonify({'error': 'Missing environment variables'}), 400
        
        success = bot.set_webhook(bot_url, secret)
        
        if success:
            return jsonify({
                'status': 'success',
                'webhook_url': f"{bot_url}/webhook/{secret}",
                'message': 'Webhook set successfully'
            }), 200
        else:
            return jsonify({'error': 'Failed to set webhook'}), 500
    except Exception as e:
        logger.error(f"Set webhook error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 10000))
    
    # Start the bot initialization in background
    bot.initialize()
    
    # Start Flask server
    logger.info(f"Starting Flask server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)