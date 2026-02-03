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
        update_id = update_data.get('update_id', 'unknown')
        logger.info(f"Received webhook update: {update_id}")
        
        # Log the update type for debugging
        if 'message' in update_data:
            chat_id = update_data['message'].get('chat', {}).get('id')
            text = update_data['message'].get('text', '')
            logger.info(f"Message from chat {chat_id}: {text}")
        elif 'callback_query' in update_data:
            chat_id = update_data['callback_query'].get('message', {}).get('chat', {}).get('id')
            data = update_data['callback_query'].get('data', '')
            logger.info(f"Callback from chat {chat_id}: {data}")
        
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