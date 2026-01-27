#!/usr/bin/env python3
"""
Main entry point - Hybrid Web Service for Render
"""
import os
import threading
from flask import Flask
import sys

# Aggiungi la cartella corrente al path di Python
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Inizializza l'app Flask
app = Flask(__name__)

# Health Check endpoint - RICHIESTO da Render
@app.route('/')
def health_check():
    return f'✅ Trading Bot "telegram-trading-bot" is running', 200

# Endpoint opzionale per forzare un riavvio del servizio
@app.route('/restart', methods=['POST'])
def soft_restart():
    # Logica per riavviare il bot in modo sicuro
    return 'Restart command received', 202

def run_flask_server():
    """Avvia il server Flask sulla porta fornita da Render"""
    port = int(os.environ.get("PORT", 10000))  # Render imposta la variabile PORT
    # NOTA: debug deve essere False in produzione
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)

def run_telegram_bot():
    """Avvia il tuo bot Telegram"""
    print("🚀 Avvio del Trading Bot...")
    try:
        # Importa e avvia la tua logica bot esistente
        from bot import TradingBot  # Assicurati che questo import funzioni
        bot = TradingBot()
        bot.run()  # Presuppone che .run() avvii il polling
    except Exception as e:
        print(f"❌ Errore nell'avvio del bot: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🔧 Avvio servizio ibrido (Web Server + Telegram Bot)...")
    
    # Avvia il server Flask in un thread separato
    flask_thread = threading.Thread(target=run_flask_server, daemon=True)
    flask_thread.start()
    print(f"🌐 Server Flask avviato in background.")
    
    # Avvia il bot Telegram nel thread principale
    run_telegram_bot()