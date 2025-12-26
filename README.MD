cat > README.md << 'EOF'
# 🤖 Telegram Trading Bot

<div align="center">

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![Telegram](https://img.shields.io/badge/Telegram-Bot-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Version](https://img.shields.io/badge/version-2.0.0-orange.svg)

**Bot Telegram per analisi finanziaria con grafici dark mode professionali**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Screenshots](#-screenshots) • [Tech Stack](#-tech-stack)

</div>

## ✨ Features

### 🔍 Ricerca Intelligente
- Supporto 10,000+ titoli globali
- Correzione automatica errori (es: "Apple" → "AAPL")
- Multi-mercato: Azioni, ETF, Crypto, Indici, Valute
- Database locale per ticker comuni

### 📊 Analisi Tecnica Completa
- **Trend:** SMA (20, 50, 200), EMA, MACD, ADX
- **Momentum:** RSI, Stochastic, ROC, Williams %R, CCI
- **Volatilità:** Bollinger Bands, ATR
- **Volume:** OBV, Volume Ratio, Volume MA
- Supporto/Resistenza dinamici

### 📈 Grafici Professionali
- Dark mode (#0d1117 background)
- 6 pannelli indicatori
- Colori personalizzati GitHub-style
- Aree ipercomprato/ipervenduto
- Generazione in tempo reale

### 🤖 Interfaccia Telegram
- Comandi semplici (`/cerca`, `/prezzo`, `/analisi`)
- Pulsanti interattivi
- Formattazione Markdown
- Gestione errori robusta

## 🚀 Installation

### Prerequisiti
- Python 3.10+
- Token Bot Telegram (da [@BotFather](https://t.me/BotFather))
- Account Yahoo Finance (gratuito)

### Setup Rapido
```bash
# 1. Clona il repository
git clone https://github.com/andgrim/telegram-trading-bot.git
cd telegram-trading-bot

# 2. Crea ambiente virtuale
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. Installa dipendenze
pip install -r requirements.txt

# 4. Configura
cp .env.example .env
# Modifica .env con il tuo token Telegram

# 5. Avvia il bot
python bot.py