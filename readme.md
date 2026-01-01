🤖 Telegram Trading Bot

A professional Telegram bot for real-time technical analysis with beautiful black-themed charts.
🌟 Features
📈 Complete Technical Analysis

    Professional charts with black background

    Multiple indicators: RSI, MACD, Moving Averages

    Automatic trading signals

    Stock comparison feature

    CSV data export

🤖 Telegram Bot

    Search tickers by name or symbol

    Multi-period analysis (1, 3, 6, 12 months)

    Interactive inline menus

    User-specific configuration

🚀 Quick Start
Local Installation
bash

# 1. Clone the repository
git clone https://github.com/andgrim/telegram-trading-bot.git
cd telegram-trading-bot

# 2. Create virtual environment
python -m venv venv
# Windows: venv\Scripts\activate
# Mac/Linux: source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure the bot
cp .env.example .env
# Edit .env with your Telegram token

Docker Installation
bash

# Build and run with Docker Compose
docker-compose up -d

🔧 Configuration
1. Get a Telegram Bot Token:

    Visit @BotFather on Telegram

    Use /newbot to create a new bot

    Copy the generated token

2. Configure .env file:
env

TELEGRAM_BOT_TOKEN=your_bot_token_here

📋 Bot Commands
text

/start - Start the bot
/help - Show help guide
/search <query> - Search for a ticker
/analyze <symbol> - Complete technical analysis
/quick <symbol> - Quick analysis
/compare <sym1> <sym2> - Compare two stocks
/settings - User settings
/periods <1,3,6> - Configure analysis periods