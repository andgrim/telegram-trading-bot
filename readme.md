cat > README.md << 'EOF'
# 🤖 Telegram Trading Bot

Bot Telegram con interfaccia web Streamlit per analisi trading tecnico.

## 🌟 Funzionalità

### 🤖 Bot Telegram
- **Ricerca ticker** per nome o simbolo
- **Analisi tecnica** completa
- **Grafici** con indicatori (RSI, MACD, Medie Mobili)
- **Informazioni dettagliate** su aziende
- **Comandi intuitivi** con menu inline

### 🌐 Interfaccia Web (Streamlit)
- **Dashboard interattiva** con tema scuro
- **Grafici Plotly** interattivi
- **Analisi multi-periodo** (1, 3, 6, 12 mesi)
- **Segnali trading** automatici
- **Export dati** in CSV

## 📋 Requisiti

- Python 3.8+
- Token Telegram Bot (da @BotFather)
- Account Yahoo Finance (gratuito)

## 🚀 Installazione Rapida

```bash
# 1. Clona il repository
git clone https://github.com/tuo-username/telegram-trading-bot.git
cd telegram-trading-bot

# 2. Crea ambiente virtuale
python -m venv venv
source venv/bin/activate  # Su Windows: venv\Scripts\activate

# 3. Installa dipendenze
pip install -r requirements.txt

# 4. Configura il bot
cp .env.example .env
# Modifica .env con il tuo token Telegram