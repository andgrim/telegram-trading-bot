#!/usr/bin/env python3
"""
Bot Telegram per analisi finanziaria
Versione semplificata e funzionante
"""
import logging
import os
from datetime import datetime

# Import delle tue librerie
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, InputFile
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler,
    ContextTypes, MessageHandler, filters
)

import yfinance as yf
import pandas as pd

from ticker_searcher import TickerSearcher
from trading_analyzer import TradingAnalyzer

# ============ CARICA VARIABILI AMBIENTE ============
load_dotenv()

# Configura logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ============ OTTIENI TOKEN ============
# Usa TELEGRAM_BOT_TOKEN (non TELEGRAM_TOKEN)
TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')

if not TOKEN:
    print("❌ ERRORE CRITICO: TELEGRAM_BOT_TOKEN non configurato!")
    print("\n📍 DOVE METTERE IL TOKEN:")
    print("1. LOCALMENTE: Crea file .env con:")
    print('   TELEGRAM_BOT_TOKEN="il_tuo_token_qui"')
    print("2. SU RENDER: Environment Variables →")
    print("   Key: TELEGRAM_BOT_TOKEN")
    print("   Value: [il tuo token]")
    exit(1)

print(f"✅ Token trovato: {TOKEN[:15]}...")

class TradingBot:
    def __init__(self):
        self.searcher = TickerSearcher()
        self.analyzer = TradingAnalyzer()
    
    # ==================== COMANDI BASE ====================
    
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comando /start"""
        welcome_text = """
🤖 *BOT TRADING FINANZIARIO*

*COMANDI DISPONIBILI:*

🔍 *RICERCA*
`/cerca Apple` - Cerca titoli
`/cerca TSLA` - Cerca per ticker

💰 *PREZZI*
`/prezzo AAPL` - Prezzo in tempo reale

📊 *ANALISI*
`/analisi MSFT` - Analisi tecnica completa

📈 *GRAFICI*
`/grafico BTC-USD` - Grafico avanzato
`/grafico_semplice AAPL` - Grafico veloce

ℹ️ *UTILITY*
`/help` - Guida comandi
`/info` - Informazioni bot

*Esempi:*
• `/cerca Tesla`
• `/prezzo AAPL`
• `/analisi ENEL.MI`
• `/grafico BTC-USD`
"""
        await update.message.reply_text(welcome_text, parse_mode='Markdown')
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comando /help"""
        await update.message.reply_text(
            "📚 *COMANDI DISPONIBILI:*\n\n"
            "🔍 `/cerca [nome]` - Ricerca titoli\n"
            "💰 `/prezzo [ticker]` - Prezzo rapido\n"
            "📊 `/analisi [ticker]` - Analisi completa\n"
            "📈 `/grafico [ticker]` - Grafico avanzato\n"
            "📉 `/grafico_semplice [ticker]` - Grafico veloce\n"
            "ℹ️ `/info` - Info bot\n\n"
            "*Esempi:*\n"
            "• `/cerca Apple`\n"
            "• `/prezzo AAPL`\n"
            "• `/analisi BTC-USD`",
            parse_mode='Markdown'
        )
    
    # ==================== RICERCA ====================
    
    async def search_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comando /cerca"""
        query = ' '.join(context.args) if context.args else ''
        
        if not query:
            await update.message.reply_text(
                "🔍 *Uso:* `/cerca [nome o ticker]`\n\n"
                "*Esempi:*\n"
                "• `/cerca Apple`\n"
                "• `/cerca TSLA`\n"
                "• `/cerca BITCOIN`",
                parse_mode='Markdown'
            )
            return
        
        await update.message.reply_text(f"🔍 Cerco *{query}*...", parse_mode='Markdown')
        
        results = self.searcher.smart_search(query)
        
        if not results:
            await update.message.reply_text(
                f"❌ Nessun risultato per *{query}*\n\n"
                "💡 *Prova con:*\n"
                "• Nome inglese (Apple, Tesla)\n"
                "• Ticker diretto (AAPL, TSLA)\n"
                "• Es: AAPL, BTC-USD, ENEL.MI",
                parse_mode='Markdown'
            )
            return
        
        response = self.searcher.format_search_results(results)
        
        keyboard = []
        for result in results:
            ticker = result['symbol']
            name = result['name'][:20] + "..." if len(result['name']) > 20 else result['name']
            keyboard.append([
                InlineKeyboardButton(
                    f"{ticker} - {name}",
                    callback_data=f"analyze_{ticker}"
                )
            ])
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            response,
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
    
    # ==================== PREZZO ====================
    
    async def price_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comando /prezzo"""
        if not context.args:
            await update.message.reply_text(
                "💰 *Uso:* `/prezzo [ticker]`\n\n"
                "*Esempi:*\n"
                "• `/prezzo AAPL`\n"
                "• `/prezzo BTC-USD`\n"
                "• `/prezzo ENEL.MI`",
                parse_mode='Markdown'
            )
            return
        
        ticker = context.args[0].upper()
        await self._send_price(update, ticker)
    
    async def _send_price(self, update: Update, ticker: str, is_callback: bool = False):
        """Invia prezzo"""
        try:
            if is_callback and hasattr(update, 'callback_query'):
                query = update.callback_query
            else:
                query = None
            
            # Ottieni dati
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1d")
            info = stock.info
            
            if hist.empty:
                error_msg = f"❌ Ticker *{ticker}* non valido"
                if query:
                    await query.message.reply_text(error_msg, parse_mode='Markdown')
                else:
                    await update.message.reply_text(error_msg, parse_mode='Markdown')
                return
            
            latest = hist.iloc[-1]
            prev_close = info.get('previousClose', latest['Close'])
            change = ((latest['Close'] - prev_close) / prev_close * 100) if prev_close else 0
            
            price_text = f"""
💰 *{ticker} - {info.get('shortName', ticker)}*

📊 *Prezzo:* ${latest['Close']:.2f}
📈 *Variazione:* {change:+.2f}%
📅 *Ultimo:* {latest.name.strftime('%H:%M')}

🔼 Massimo: ${latest['High']:.2f}
🔽 Minimo: ${latest['Low']:.2f}
📈 Chiusura prec.: ${prev_close:.2f}
"""
            
            keyboard = [
                [
                    InlineKeyboardButton("📊 Analisi", callback_data=f"analyze_{ticker}"),
                    InlineKeyboardButton("📈 Grafico", callback_data=f"chart_{ticker}")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            if query:
                await query.message.reply_text(
                    price_text,
                    reply_markup=reply_markup,
                    parse_mode='Markdown'
                )
            else:
                await update.message.reply_text(
                    price_text,
                    reply_markup=reply_markup,
                    parse_mode='Markdown'
                )
                
        except Exception as e:
            logger.error(f"Errore prezzo {ticker}: {e}")
            error_msg = f"❌ Errore prezzo *{ticker}*"
            
            if is_callback and hasattr(update, 'callback_query'):
                await update.callback_query.message.reply_text(error_msg, parse_mode='Markdown')
            else:
                await update.message.reply_text(error_msg, parse_mode='Markdown')
    
    # ==================== ANALISI ====================
    
    async def analyze_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comando /analisi"""
        if not context.args:
            await update.message.reply_text(
                "📊 *Uso:* `/analisi [ticker]`\n\n"
                "*Esempi:*\n"
                "• `/analisi AAPL`\n"
                "• `/analisi TSLA`\n"
                "• `/analisi BTC-USD`",
                parse_mode='Markdown'
            )
            return
        
        ticker = context.args[0].upper()
        await self._send_analysis(update, ticker)
    
    async def _send_analysis(self, update: Update, ticker: str, is_callback: bool = False):
        """Invia analisi"""
        try:
            if is_callback and hasattr(update, 'callback_query'):
                query = update.callback_query
            else:
                query = None
            
            # Ottieni dati
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1mo")
            info = self.searcher.get_ticker_info(ticker)
            
            if hist.empty or not info:
                error_msg = f"❌ Dati insufficienti per *{ticker}*"
                if query:
                    await query.message.reply_text(error_msg, parse_mode='Markdown')
                else:
                    await update.message.reply_text(error_msg, parse_mode='Markdown')
                return
            
            # Calcola indicatori
            df_with_indicators = self.analyzer.calculate_indicators(hist)
            
            # Genera report
            report = self.analyzer.generate_analysis_report(ticker, df_with_indicators, info)
            
            keyboard = [
                [
                    InlineKeyboardButton("💰 Prezzo", callback_data=f"price_{ticker}"),
                    InlineKeyboardButton("📈 Grafico", callback_data=f"chart_{ticker}")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            if query:
                await query.message.reply_text(
                    report,
                    reply_markup=reply_markup,
                    parse_mode='Markdown'
                )
            else:
                await update.message.reply_text(
                    report,
                    reply_markup=reply_markup,
                    parse_mode='Markdown'
                )
                
        except Exception as e:
            logger.error(f"Errore analisi {ticker}: {e}")
            error_msg = f"❌ Errore analisi *{ticker}*"
            
            if is_callback and hasattr(update, 'callback_query'):
                await update.callback_query.message.reply_text(error_msg, parse_mode='Markdown')
            else:
                await update.message.reply_text(error_msg, parse_mode='Markdown')
    
    # ==================== GRAFICI ====================
    
    async def chart_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comando /grafico - Grafico avanzato"""
        if not context.args:
            await update.message.reply_text(
                "📈 *Uso:* `/grafico [ticker]`\n\n"
                "*Esempi:*\n"
                "• `/grafico AAPL`\n"
                "• `/grafico BTC-USD`",
                parse_mode='Markdown'
            )
            return
        
        ticker = context.args[0].upper()
        await self._send_chart(update, ticker)
    
    async def simple_chart_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comando /grafico_semplice - Grafico veloce"""
        if not context.args:
            await update.message.reply_text(
                "📉 *Uso:* `/grafico_semplice [ticker]`\n\n"
                "*Esempi:*\n"
                "• `/grafico_semplice AAPL`\n"
                "• `/grafico_semplice TSLA`",
                parse_mode='Markdown'
            )
            return
        
        ticker = context.args[0].upper()
        await self._send_simple_chart(update, ticker)
    
    async def _send_chart(self, update: Update, ticker: str, is_callback: bool = False):
        """Invia grafico avanzato"""
        try:
            if is_callback and hasattr(update, 'callback_query'):
                query = update.callback_query
                await query.answer()
                chat_id = query.message.chat_id
            else:
                query = None
                chat_id = update.message.chat_id
            
            # Messaggio caricamento
            loading_msg = await update.effective_chat.send_message(
                f"🔄 Generazione grafico per *{ticker}*...",
                parse_mode='Markdown'
            )
            
            # Ottieni dati
            stock = yf.Ticker(ticker)
            hist = stock.history(period="3mo", interval="1d")
            
            if hist.empty:
                await loading_msg.delete()
                error_msg = f"❌ Dati insufficienti per *{ticker}*"
                if query:
                    await query.message.reply_text(error_msg, parse_mode='Markdown')
                else:
                    await update.message.reply_text(error_msg, parse_mode='Markdown')
                return
            
            # Calcola indicatori e genera grafico
            df_with_indicators = self.analyzer.calculate_indicators(hist)
            chart_image = self.analyzer.generate_chart_image(df_with_indicators, ticker, "3mo")
            
            if not chart_image:
                await loading_msg.delete()
                await self._send_simple_chart(update, ticker, is_callback)
                return
            
            # Info per caption
            latest = df_with_indicators.iloc[-1]
            info = stock.info
            name = info.get('shortName', ticker)
            
            caption = f"""
📈 *GRAFICO - {ticker}*
🏢 {name}
💰 Prezzo: ${latest['Close']:.2f}
📊 RSI: {latest.get('RSI', 0):.1f}
📅 Periodo: 3 mesi

🔄 Generato: {datetime.now().strftime('%H:%M')}
"""
            
            await loading_msg.delete()
            
            # Invia immagine
            await update.effective_chat.send_photo(
                photo=InputFile(chart_image, filename=f'chart_{ticker}.png'),
                caption=caption,
                parse_mode='Markdown'
            )
                
        except Exception as e:
            logger.error(f"Errore grafico {ticker}: {e}")
            error_msg = f"❌ Errore grafico *{ticker}*"
            
            try:
                if 'loading_msg' in locals():
                    await loading_msg.delete()
            except:
                pass
            
            if is_callback and hasattr(update, 'callback_query'):
                await update.callback_query.message.reply_text(error_msg, parse_mode='Markdown')
            else:
                await update.message.reply_text(error_msg, parse_mode='Markdown')
    
    async def _send_simple_chart(self, update: Update, ticker: str, is_callback: bool = False):
        """Invia grafico semplice"""
        try:
            if is_callback and hasattr(update, 'callback_query'):
                query = update.callback_query
                await query.answer()
            else:
                query = None
            
            # Ottieni dati
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1mo", interval="1d")
            
            if hist.empty:
                error_msg = f"❌ Dati insufficienti per *{ticker}*"
                if query:
                    await query.message.reply_text(error_msg, parse_mode='Markdown')
                else:
                    await update.message.reply_text(error_msg, parse_mode='Markdown')
                return
            
            # Genera grafico semplice
            chart_image = self.analyzer.generate_simple_chart(hist, ticker)
            
            if not chart_image:
                error_msg = f"❌ Impossibile generare grafico *{ticker}*"
                if query:
                    await query.message.reply_text(error_msg, parse_mode='Markdown')
                else:
                    await update.message.reply_text(error_msg, parse_mode='Markdown')
                return
            
            # Caption
            latest = hist.iloc[-1]
            info = stock.info
            name = info.get('shortName', ticker)
            
            caption = f"""
📉 *GRAFICO SEMPLICE - {ticker}*
🏢 {name}
💰 Prezzo: ${latest['Close']:.2f}
📅 Periodo: 1 mese

⚡ Generazione veloce
"""
            
            # Invia immagine
            if query:
                await query.message.reply_photo(
                    photo=InputFile(chart_image, filename=f'chart_simple_{ticker}.png'),
                    caption=caption,
                    parse_mode='Markdown'
                )
            else:
                await update.message.reply_photo(
                    photo=InputFile(chart_image, filename=f'chart_simple_{ticker}.png'),
                    caption=caption,
                    parse_mode='Markdown'
                )
                
        except Exception as e:
            logger.error(f"Errore grafico semplice {ticker}: {e}")
            error_msg = f"❌ Errore grafico semplice *{ticker}*"
            
            if is_callback and hasattr(update, 'callback_query'):
                await update.callback_query.message.reply_text(error_msg, parse_mode='Markdown')
            else:
                await update.message.reply_text(error_msg, parse_mode='Markdown')
    
    # ==================== INFO ====================
    
    async def info_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comando /info"""
        info_text = """
🤖 *TRADING BOT*

*Versione:* 1.0
*Dati:* Yahoo Finance
*Grafici:* Dark mode

📊 *Funzionalità:*
• Ricerca titoli
• Prezzi in tempo reale
• Analisi tecnica
• Grafici avanzati

⚠️ *Disclaimer:*
Informazioni solo a scopo educativo.
Non è un consiglio di investimento.

👨‍💻 *Sviluppatore:* @mafark
"""
        await update.message.reply_text(info_text, parse_mode='Markdown')
    
    # ==================== CALLBACK ====================
    
    async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Gestisce pulsanti"""
        query = update.callback_query
        await query.answer()
        
        data = query.data
        
        if data.startswith("analyze_"):
            ticker = data.replace("analyze_", "")
            await self._send_analysis(update, ticker, is_callback=True)
        
        elif data.startswith("price_"):
            ticker = data.replace("price_", "")
            await self._send_price(update, ticker, is_callback=True)
        
        elif data.startswith("chart_"):
            ticker = data.replace("chart_", "")
            await self._send_chart(update, ticker, is_callback=True)

def main():
    """Avvia il bot"""
    # TOKEN è già verificato all'inizio
    bot = TradingBot()
    
    # Crea applicazione
    application = Application.builder().token(TOKEN).build()
    
    # Handler comandi
    application.add_handler(CommandHandler("start", bot.start))
    application.add_handler(CommandHandler("help", bot.help_command))
    application.add_handler(CommandHandler("cerca", bot.search_command))
    application.add_handler(CommandHandler("search", bot.search_command))
    application.add_handler(CommandHandler("prezzo", bot.price_command))
    application.add_handler(CommandHandler("price", bot.price_command))
    application.add_handler(CommandHandler("analisi", bot.analyze_command))
    application.add_handler(CommandHandler("analysis", bot.analyze_command))
    application.add_handler(CommandHandler("grafico", bot.chart_command))
    application.add_handler(CommandHandler("chart", bot.chart_command))
    application.add_handler(CommandHandler("grafico_semplice", bot.simple_chart_command))
    application.add_handler(CommandHandler("simple_chart", bot.simple_chart_command))
    application.add_handler(CommandHandler("info", bot.info_command))
    
    # Callback
    application.add_handler(CallbackQueryHandler(bot.button_callback))
    
    # Avvia
    logger.info("🤖 Bot avviato! Premi Ctrl+C per fermare...")
    print("\n" + "="*50)
    print("✅ BOT AVVIATO SU RENDER!")
    print("="*50)
    print(f"Token: {TOKEN[:15]}...")
    print("\n📋 Comandi disponibili:")
    print("• /start - Benvenuto")
    print("• /cerca [nome] - Ricerca titoli")
    print("• /prezzo [ticker] - Prezzo")
    print("• /analisi [ticker] - Analisi")
    print("• /grafico [ticker] - Grafico avanzato")
    print("• /grafico_semplice [ticker] - Grafico veloce")
    print("• /info - Info bot")
    print("="*50 + "\n")
    
    # Avvia polling
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()