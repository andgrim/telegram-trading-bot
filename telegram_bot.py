import asyncio
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Usa backend non interattivo
from trading_analyzer import TradingAnalyzer
from utils import search_tickers, get_ticker_info
import pandas as pd
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
from io import BytesIO
import logging
import os
from dotenv import load_dotenv
import numpy as np

# Carica le variabili d'ambiente dal file .env
load_dotenv()

# Configurazione logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class TelegramTradingBot:
    def __init__(self, token: str):
        self.token = token
        self.user_data = {}
        
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comando /start"""
        user = update.effective_user
        welcome_message = f"""
👋 Benvenuto {user.first_name} nel Trading Bot!

Comandi disponibili:
/search - Cerca un ticker per simbolo o nome
/analyze <symbol> - Analizza un ticker (es: /analyze AAPL)
/quick <symbol> - Analisi rapida
/settings - Configura i parametri di analisi
/help - Mostra questa guida

Esempi:
• /search Apple
• /analyze TSLA
• /analyze BTC-USD
        """
        await update.message.reply_text(welcome_message)
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comando /help"""
        help_text = """
📚 **Guida ai comandi:**

🔍 **RICERCA E ANALISI**
/search - Cerca ticker per nome o simbolo
/analyze <symbol> - Analisi completa di un ticker
/quick <symbol> - Analisi rapida

⚙️ **CONFIGURAZIONE**
/settings - Configura parametri di analisi
/periods <lista> - Imposta periodi (es: 1,3,6,12)

📊 **UTILITY**
/compare <sym1> <sym2> - Confronta due ticker
/top - Ticker più popolari

📈 **SEGNALI**
/signals - Segnali trading attivi

💡 **Esempi:**
• /search Tesla
• /analyze AAPL
• /periods 1,3,6,12
• /compare AAPL MSFT
        """
        await update.message.reply_text(help_text, parse_mode='Markdown')
    
    async def search_ticker(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Cerca ticker per nome o simbolo"""
        query = ' '.join(context.args)
        if not query:
            await update.message.reply_text("❌ Specifica cosa cercare:\n/search Apple\n/search AAPL\n/search TSLA")
            return
        
        await update.message.reply_text(f"🔍 Ricerca di '{query}' in corso...")
        
        try:
            results = search_tickers(query, limit=10)
            
            if not results:
                await update.message.reply_text("❌ Nessun risultato trovato")
                return
            
            # Crea tastiera inline con risultati
            keyboard = []
            for result in results[:8]:  # Massimo 8 risultati
                btn_text = f"{result['symbol']} - {result['name'][:30]}..."
                callback_data = f"select_{result['symbol']}"
                keyboard.append([InlineKeyboardButton(btn_text, callback_data=callback_data)])
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.message.reply_text(
                f"✅ Trovati {len(results)} risultati:",
                reply_markup=reply_markup
            )
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            await update.message.reply_text("❌ Errore nella ricerca")
    
    async def analyze_ticker(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Analizza un ticker"""
        if not context.args:
            await update.message.reply_text("❌ Specifica il simbolo:\n/analyze AAPL\n/analyze TSLA\n/analyze BTC-USD")
            return
        
        symbol = context.args[0].upper()
        user_id = update.effective_user.id
        
        # Salva simbolo nei dati utente
        if user_id not in self.user_data:
            self.user_data[user_id] = {'periods': [3, 6], 'interval': '1d'}
        
        await update.message.reply_text(f"📊 Analisi di {symbol} in corso...")
        
        try:
            # Ottieni info ticker
            ticker_info = get_ticker_info(symbol)
            
            if not ticker_info:
                await update.message.reply_text(f"❌ Ticker {symbol} non trovato")
                return
            
            # Prepara i dati per la formattazione
            current_price = ticker_info.get('current_price', 0)
            prev_close = ticker_info.get('previous_close', current_price)
            volume = ticker_info.get('volume', 0)
            market_cap = ticker_info.get('market_cap', 0)
            
            # Formatta prezzo
            if isinstance(current_price, (int, float)):
                price_str = f"${current_price:.2f}"
            else:
                price_str = 'N/A'
            
            # Calcola variazione
            if isinstance(current_price, (int, float)) and isinstance(prev_close, (int, float)) and prev_close != 0:
                daily_change = ((current_price - prev_close) / prev_close) * 100
                daily_change_str = f"{daily_change:+.2f}%"
            else:
                daily_change_str = "N/A"
            
            # Formatta volume
            if isinstance(volume, (int, float)):
                volume_str = f"{volume:,.0f}"
            else:
                volume_str = "N/A"
            
            # Formatta market cap
            if isinstance(market_cap, (int, float)):
                market_cap_str = f"${market_cap:,.0f}"
            else:
                market_cap_str = "N/A"
            
            # Crea messaggio
            message = f"""
📈 **{ticker_info.get('name', symbol)} ({symbol})**

💰 **Prezzo:** {price_str}
📊 **Variazione:** {daily_change_str}
📈 **Volume:** {volume_str}
🏢 **Market Cap:** {market_cap_str}

🏭 **Settore:** {ticker_info.get('sector', 'N/A')}
📊 **P/E Ratio:** {ticker_info.get('pe_ratio', 'N/A')}
🎯 **52W High:** ${ticker_info.get('52w_high', 'N/A')}
📉 **52W Low:** ${ticker_info.get('52w_low', 'N/A')}
            """
            
            await update.message.reply_text(message, parse_mode='Markdown')
            
            # Esegui analisi tecnica
            periods = self.user_data.get(user_id, {}).get('periods', [3, 6])
            
            analyzer = TradingAnalyzer(symbol)
            summary_data = []
            
            for period in sorted(periods):
                df = analyzer.analyze_period(period)
                
                if df is not None and not df.empty:
                    price_change = ((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1) * 100
                    current_rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns else 50
                    macd_signal = "↑" if df['MACD'].iloc[-1] > 0 else "↓" if 'MACD' in df.columns else "N/A"
                    volume_trend = analyzer._analyze_volume_trend(df) if hasattr(analyzer, '_analyze_volume_trend') else "N/A"
                    
                    summary_data.append({
                        'period': period,
                        'price': df['Close'].iloc[-1],
                        'change': price_change,
                        'rsi': current_rsi,
                        'macd': macd_signal,
                        'volume': volume_trend
                    })
            
            if summary_data:
                # Crea tabella riepilogativa
                summary_text = "\n📊 **ANALISI TECNICA:**\n\n"
                summary_text += "Periodo | Prezzo | Variaz | RSI | MACD | Volume\n"
                summary_text += "--------|--------|--------|-----|------|--------\n"
                
                for data in summary_data:
                    change_emoji = "📈" if data['change'] > 0 else "📉"
                    summary_text += (f"{data['period']}M | ${data['price']:.2f} | "
                                   f"{change_emoji}{data['change']:+.1f}% | "
                                   f"{data['rsi']:.1f} | {data['macd']} | "
                                   f"{data['volume']}\n")
                
                await update.message.reply_text(summary_text, parse_mode='Markdown')
                
                # Crea tastiera per opzioni aggiuntive
                keyboard = [
                    [
                        InlineKeyboardButton("📈 Grafico", callback_data=f"chart_{symbol}_3"),
                        InlineKeyboardButton("📊 Dati", callback_data=f"data_{symbol}")
                    ],
                    [
                        InlineKeyboardButton("🎯 Segnali", callback_data=f"signals_{symbol}"),
                        InlineKeyboardButton("⚙️ Impostazioni", callback_data="settings")
                    ]
                ]
                
                reply_markup = InlineKeyboardMarkup(keyboard)
                await update.message.reply_text(
                    "Scegli un'opzione:",
                    reply_markup=reply_markup
                )
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            await update.message.reply_text(f"❌ Errore nell'analisi di {symbol}")
    
    async def show_chart(self, query, context: ContextTypes.DEFAULT_TYPE, symbol: str, period: int):
        """Mostra grafico del ticker con Matplotlib"""
        try:
            await query.answer("Creazione grafico...")
            
            analyzer = TradingAnalyzer(symbol)
            df = analyzer.analyze_period(period)
            
            if df is None or df.empty:
                await query.edit_message_text("❌ Dati non disponibili per il grafico")
                return
            
            # Crea figura con Matplotlib
            plt.figure(figsize=(10, 6))
            
            # Grafico del prezzo
            plt.plot(df.index, df['Close'], label='Prezzo', color='green', linewidth=2)
            
            # Medie mobili
            if 'SMA_20' in df.columns:
                plt.plot(df.index, df['SMA_20'], label='SMA 20', color='orange', linestyle='--', linewidth=1)
            
            if 'SMA_50' in df.columns:
                plt.plot(df.index, df['SMA_50'], label='SMA 50', color='red', linestyle='--', linewidth=1)
            
            # Configura il grafico
            plt.title(f'{symbol} - {period} mesi', fontsize=14, fontweight='bold')
            plt.xlabel('Data')
            plt.ylabel('Prezzo ($)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            
            # Salva in buffer
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            plt.close()
            buf.seek(0)
            
            # Invia foto
            await context.bot.send_photo(
                chat_id=query.message.chat_id,
                photo=buf,
                caption=f"📈 Grafico {symbol} - {period} mesi"
            )
            
        except Exception as e:
            logger.error(f"Chart error: {e}")
            await query.edit_message_text("❌ Errore nella creazione del grafico")
    
    async def show_signals(self, query, context: ContextTypes.DEFAULT_TYPE, symbol: str):
        """Mostra segnali trading per il ticker"""
        try:
            await query.answer("Analisi segnali...")
            
            analyzer = TradingAnalyzer(symbol)
            df = analyzer.analyze_period(3)  # Usa 3 mesi per segnali
            
            if df is None or df.empty:
                await query.edit_message_text("❌ Dati non disponibili per segnali")
                return
            
            latest = df.iloc[-1]
            
            # Analizza segnali
            signals = []
            score = 0
            
            # RSI
            rsi = latest.get('RSI', 50)
            if rsi < 30:
                signals.append(("🟢 RSI", "Sovravenduto", 1))
                score += 1
            elif rsi > 70:
                signals.append(("🔴 RSI", "Sovracomprato", -1))
                score -= 1
            else:
                signals.append(("⚪ RSI", "Neutrale", 0))
            
            # MACD
            macd = latest.get('MACD', 0)
            signal_line = latest.get('Signal_Line', 0)
            if macd > signal_line:
                signals.append(("🟢 MACD", "Rialzista", 1))
                score += 1
            else:
                signals.append(("🔴 MACD", "Ribassista", -1))
                score -= 1
            
            # Volume
            if hasattr(analyzer, '_analyze_volume_trend'):
                volume_trend = analyzer._analyze_volume_trend(df)
                if volume_trend == "HIGH":
                    signals.append(("🟢 Volume", "Alto", 1))
                    score += 1
                elif volume_trend == "LOW":
                    signals.append(("🔴 Volume", "Basso", -1))
                    score -= 1
                else:
                    signals.append(("⚪ Volume", "Normale", 0))
            
            # Costruisci messaggio
            signals_text = "🎯 **SEGNALI TRADING:**\n\n"
            for icon, text, _ in signals:
                signals_text += f"{icon} {text}\n"
            
            signals_text += f"\n📊 **PUNTEGGIO:** {score}\n\n"
            
            if score > 1:
                signals_text += "🎯 **FORTE SEGNALE ACQUISTO**\nConsidera posizione long"
            elif score > 0:
                signals_text += "📈 **LEGGERO SEGNALE ACQUISTO**\nMonitora per entry"
            elif score < -1:
                signals_text += "⚠️ **FORTE SEGNALE VENDITA**\nConsidera posizione short"
            elif score < 0:
                signals_text += "📉 **LEGGERO SEGNALE VENDITA**\nCautela in acquisto"
            else:
                signals_text += "⚖️ **SEGNALE NEUTRO**\nAttendi segnali più chiari"
            
            signals_text += "\n\n⚠️ *Nota: Segnali automatici, non consigli finanziari*"
            
            await query.edit_message_text(signals_text, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Signals error: {e}")
            await query.edit_message_text("❌ Errore nell'analisi segnali")
    
    async def show_data(self, query, context: ContextTypes.DEFAULT_TYPE, symbol: str):
        """Mostra dati raw del ticker"""
        try:
            await query.answer("Recupero dati...")
            
            analyzer = TradingAnalyzer(symbol)
            df = analyzer.analyze_period(1)  # 1 mese di dati
            
            if df is None or df.empty:
                await query.edit_message_text("❌ Dati non disponibili")
                return
            
            # Prepara ultime 10 righe
            recent_data = df.tail(10)
            
            data_text = f"📋 **DATI RECENTI {symbol}:**\n\n"
            data_text += "Data | Aperto | Alto | Basso | Chiuso | Volume | RSI\n"
            data_text += "-----|--------|------|-------|--------|--------|----\n"
            
            for idx, row in recent_data.iterrows():
                date_str = idx.strftime('%d/%m')
                open_price = row.get('Open', 0)
                high = row.get('High', 0)
                low = row.get('Low', 0)
                close = row.get('Close', 0)
                volume = row.get('Volume', 0)
                rsi = row.get('RSI', 0)
                
                data_text += (f"{date_str} | ${open_price:.2f} | ${high:.2f} | "
                            f"${low:.2f} | ${close:.2f} | "
                            f"{volume:,.0f} | {rsi:.1f}\n")
            
            # Tastiera per scaricare dati
            keyboard = [[
                InlineKeyboardButton("📥 Scarica CSV", callback_data=f"download_{symbol}")
            ]]
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(
                data_text,
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )
            
        except Exception as e:
            logger.error(f"Data error: {e}")
            await query.edit_message_text("❌ Errore nel recupero dati")
    
    async def download_data(self, query, context: ContextTypes.DEFAULT_TYPE, symbol: str):
        """Prepara download dati CSV"""
        try:
            await query.answer("Preparazione download...")
            
            analyzer = TradingAnalyzer(symbol)
            df = analyzer.analyze_period(12)  # Tutti i dati disponibili
            
            if df is None or df.empty:
                await query.answer("❌ Dati non disponibili", show_alert=True)
                return
            
            # Converti in CSV
            csv_data = df.to_csv()
            
            # Crea file in memoria
            csv_file = BytesIO(csv_data.encode())
            csv_file.name = f"{symbol}_data.csv"
            
            # Invia file
            await context.bot.send_document(
                chat_id=query.message.chat_id,
                document=csv_file,
                filename=f"{symbol}_historical_data.csv",
                caption=f"📥 Dati storici {symbol}"
            )
            
        except Exception as e:
            logger.error(f"Download error: {e}")
            await query.answer("❌ Errore nel download", show_alert=True)
    
    async def settings_menu(self, query, context: ContextTypes.DEFAULT_TYPE):
        """Mostra menu impostazioni"""
        user_id = query.from_user.id
        
        if user_id not in self.user_data:
            self.user_data[user_id] = {'periods': [3, 6], 'interval': '1d'}
        
        settings = self.user_data[user_id]
        
        settings_text = f"""
⚙️ **IMPOSTAZIONES UTENTE**

📅 Periodi analisi: {', '.join(map(str, settings.get('periods', [3, 6])))} mesi
📊 Intervallo: {settings.get('interval', '1d')}

Usa /periods <lista> per cambiare periodi
Es: /periods 1,3,6,12
        """
        
        keyboard = [
            [
                InlineKeyboardButton("📅 Periodi", callback_data="set_periods"),
                InlineKeyboardButton("📊 Intervallo", callback_data="set_interval")
            ],
            [
                InlineKeyboardButton("🔙 Indietro", callback_data="back_settings")
            ]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            settings_text,
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
    
    async def set_periods_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Imposta periodi di analisi"""
        if not context.args:
            await update.message.reply_text("❌ Specifica periodi:\n/periods 1,3,6\n/periods 3,12")
            return
        
        try:
            periods_str = context.args[0]
            periods = [int(p.strip()) for p in periods_str.split(',')]
            
            user_id = update.effective_user.id
            if user_id not in self.user_data:
                self.user_data[user_id] = {}
            
            self.user_data[user_id]['periods'] = periods
            
            await update.message.reply_text(
                f"✅ Periodi impostati: {', '.join(map(str, periods))} mesi"
            )
            
        except Exception as e:
            await update.message.reply_text("❌ Formato errato. Es: /periods 1,3,6")
    
    async def quick_analyze(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Analisi rapida di un ticker"""
        if not context.args:
            await update.message.reply_text("❌ Specifica simbolo: /quick AAPL")
            return
        
        symbol = context.args[0].upper()
        
        try:
            ticker_info = get_ticker_info(symbol)
            
            if not ticker_info:
                await update.message.reply_text(f"❌ {symbol} non trovato")
                return
            
            # Analisi rapida
            analyzer = TradingAnalyzer(symbol)
            df = analyzer.analyze_period(1)  # 1 mese
            
            if df is None or df.empty:
                await update.message.reply_text(f"❌ Dati {symbol} non disponibili")
                return
            
            latest = df.iloc[-1]
            
            # Prepara dati
            current_price = ticker_info.get('current_price', latest.get('Close', 0))
            rsi = latest.get('RSI', 0)
            macd = latest.get('MACD', 0)
            
            # Formatta prezzo
            if isinstance(current_price, (int, float)):
                price_str = f"${current_price:.2f}"
            else:
                price_str = "N/A"
            
            # Determina trend SMA
            sma_20 = latest.get('SMA_20', 0)
            sma_50 = latest.get('SMA_50', 0)
            sma_trend = "↑" if sma_20 > sma_50 else "↓"
            
            # Determina segnale RSI
            if rsi < 30:
                signal = "🟢 Acquisto"
            elif rsi > 70:
                signal = "🔴 Vendita"
            else:
                signal = "⚪ Neutro"
            
            message = f"""
⚡ **ANALISI RAPIDA {symbol}**

💰 Prezzo: {price_str}
📊 RSI: {rsi:.1f}
📈 MACD: {"↑" if macd > 0 else "↓"}
📅 SMA Trend: {sma_trend}

🎯 Segnale: {signal}
            """
            
            await update.message.reply_text(message, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Quick analyze error: {e}")
            await update.message.reply_text(f"❌ Errore analisi rapida {symbol}")
    
    async def callback_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Gestisce callback query da tastiere inline"""
        query = update.callback_query
        await query.answer()
        
        data = query.data
        
        if data.startswith("select_"):
            symbol = data.replace("select_", "")
            await self.analyze_ticker_from_callback(query, context, symbol)
        
        elif data.startswith("chart_"):
            parts = data.split("_")
            symbol = parts[1]
            period = int(parts[2]) if len(parts) > 2 else 3
            await self.show_chart(query, context, symbol, period)
        
        elif data.startswith("signals_"):
            symbol = data.replace("signals_", "")
            await self.show_signals(query, context, symbol)
        
        elif data.startswith("data_"):
            symbol = data.replace("data_", "")
            await self.show_data(query, context, symbol)
        
        elif data.startswith("download_"):
            symbol = data.replace("download_", "")
            await self.download_data(query, context, symbol)
        
        elif data.startswith("analyze_full_"):
            symbol = data.replace("analyze_full_", "")
            await self.analyze_ticker_from_callback(query, context, symbol)
        
        elif data == "settings":
            await self.settings_menu(query, context)
        
        elif data == "set_periods":
            await query.edit_message_text(
                "📅 Imposta periodi:\n\nUsa /periods <lista>\nEs: /periods 1,3,6,12"
            )
        
        elif data == "set_interval":
            await query.edit_message_text(
                "📊 Imposta intervallo:\n\nUsa /interval <intervallo>\nEs: /interval 1d (giornaliero)\n1wk (settimanale)\n1mo (mensile)"
            )
        
        elif data == "back_settings":
            await query.edit_message_text("🔙 Ritorno al menu principale")
    
    async def analyze_ticker_from_callback(self, query, context: ContextTypes.DEFAULT_TYPE, symbol: str):
        """Analizza ticker da callback"""
        await query.edit_message_text(f"📊 Analisi {symbol} in corso...")
        
        try:
            ticker_info = get_ticker_info(symbol)
            
            if not ticker_info:
                await query.edit_message_text(f"❌ {symbol} non trovato")
                return
            
            # Prepara dati
            current_price = ticker_info.get('current_price', 0)
            market_cap = ticker_info.get('market_cap', 0)
            
            # Formatta prezzo
            if isinstance(current_price, (int, float)):
                price_str = f"${current_price:.2f}"
            else:
                price_str = "N/A"
            
            # Formatta market cap
            if isinstance(market_cap, (int, float)):
                market_cap_str = f"${market_cap:,.0f}"
            else:
                market_cap_str = "N/A"
            
            message = f"""
📈 **{ticker_info.get('name', symbol)} ({symbol})**

💰 Prezzo: {price_str}
📊 Settore: {ticker_info.get('sector', 'N/A')}
📈 Market Cap: {market_cap_str}

Usa /analyze {symbol} per analisi completa
            """
            
            keyboard = [[
                InlineKeyboardButton("📊 Analisi Completa", callback_data=f"analyze_full_{symbol}"),
                InlineKeyboardButton("📈 Grafico", callback_data=f"chart_{symbol}_3")
            ]]
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(message, reply_markup=reply_markup, parse_mode='Markdown')
        
        except Exception as e:
            logger.error(f"Callback analysis error: {e}")
            await query.edit_message_text(f"❌ Errore analisi {symbol}")
    
    async def handle_text_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Gestisce messaggi di testo (analisi diretta)"""
        text = update.message.text.strip().upper()
        
        # Se sembra un simbolo ticker (1-5 lettere, eventualmente con - o .)
        if 1 <= len(text) <= 10 and all(c.isalpha() or c in '-. ' for c in text):
            await self.analyze_ticker(update, context)
        else:
            await update.message.reply_text(
                f"🔍 Per analizzare '{text}', usa:\n/analyze {text}\n\n"
                f"Oppure cerca con:\n/search {text}"
            )
    
    def run(self):
        """Avvia il bot"""
        application = Application.builder().token(self.token).build()
        
        # Comandi
        application.add_handler(CommandHandler("start", self.start))
        application.add_handler(CommandHandler("help", self.help_command))
        application.add_handler(CommandHandler("search", self.search_ticker))
        application.add_handler(CommandHandler("analyze", self.analyze_ticker))
        application.add_handler(CommandHandler("quick", self.quick_analyze))
        application.add_handler(CommandHandler("settings", self.settings_menu))
        application.add_handler(CommandHandler("periods", self.set_periods_command))
        
        # Callback query
        application.add_handler(CallbackQueryHandler(self.callback_handler))
        
        # Messaggi di testo (per analisi diretta)
        application.add_handler(MessageHandler(
            filters.TEXT & ~filters.COMMAND, 
            self.handle_text_message
        ))
        
        logger.info("Bot avviato...")
        
        # Avvia il polling
        application.run_polling()


def main():
    # Leggi il token dal file .env
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    
    # Verifica se il token è impostato
    if not TELEGRAM_BOT_TOKEN:
        print("❌ ERRORE: Token del bot Telegram non trovato!")
        print("Assicurati di avere un file .env con TELEGRAM_BOT_TOKEN")
        print("\n1. Crea un file .env nella stessa cartella")
        print("2. Aggiungi: TELEGRAM_BOT_TOKEN=\"il-tuo-token-qui\"")
        print("3. Riavvia il bot")
        return
    
    # Crea e avvia il bot
    bot = TelegramTradingBot(TELEGRAM_BOT_TOKEN)
    bot.run()


if __name__ == "__main__":
    main()