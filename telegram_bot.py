#!/usr/bin/env python3
"""
Telegram Trading Bot COMPLETO con matplotlib per grafici
"""
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
from trading_analyzer import TradingAnalyzer
from utils import search_tickers, get_ticker_info
import config
import pandas as pd
import logging
import traceback
import io

# Import matplotlib con backend non interattivo
try:
    import matplotlib
    matplotlib.use('Agg')  # IMPORTANTE per server
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️ Matplotlib non disponibile. I grafici saranno testuali.")

# Configura logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class TelegramTradingBot:
    def __init__(self):
        self.token = config.Config.TELEGRAM_TOKEN
    
    # ========== FUNZIONI PRINCIPALI (COMANDI) ==========
    
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comando /start"""
        user = update.effective_user
        logger.info(f"Utente {user.id} ({user.first_name}) ha usato /start")
        
        welcome_text = """
        🔍 *Trading Bot Analysis*
        
        *Comandi disponibili:*
        /search [nome] - Cerca ticker per nome
        /analyze SYMBOL - Analizza un simbolo
        /info SYMBOL - Informazioni dettagliate
        /help - Guida completa
        
        *Esempi:*
        /search apple
        /analyze AAPL
        /info TSLA
        
        Usa /help per tutti i comandi!
        """
        
        await update.message.reply_text(welcome_text, parse_mode='Markdown')
        print(f"📨 Risposto a {user.first_name} con messaggio di benvenuto")
    
    async def search_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Gestisce comando /search"""
        if not context.args:
            await update.message.reply_text("❌ Specifica cosa cercare. Es: /search apple")
            return
        
        query = " ".join(context.args)
        processing_msg = await update.message.reply_text(f"🔍 Cercando '{query}'...")
        
        try:
            results = search_tickers(query, limit=8)
            
            if not results:
                await update.message.reply_text(f"❌ Nessun risultato per '{query}'")
                return
            
            # Crea messaggio con risultati
            message = f"🔍 *Risultati per '{query}':*\n\n"
            
            for i, result in enumerate(results, 1):
                message += f"{i}. *{result['symbol']}* - {result['name']}\n"
                message += f"   📊 Exchange: {result['exchange']}\n\n"
            
            message += "\nUsa /analyze [SIMBOLO] per analisi dettagliata"
            
            await update.message.reply_text(message, parse_mode='Markdown')
            
            # Pulsanti per selezionare rapidamente
            keyboard = []
            for result in results[:4]:  # Massimo 4 pulsanti per riga
                keyboard.append([
                    InlineKeyboardButton(
                        f"{result['symbol']}",
                        callback_data=f"quick_analyze_{result['symbol']}"
                    )
                ])
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            await update.message.reply_text(
                "Analizza rapidamente:",
                reply_markup=reply_markup
            )
            
        except Exception as e:
            await update.message.reply_text(f"❌ Errore nella ricerca: {str(e)}")
        finally:
            await processing_msg.delete()
    
    async def info_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Gestisce comando /info per informazioni dettagliate"""
        if not context.args:
            await update.message.reply_text("❌ Specifica un simbolo. Es: /info AAPL")
            return
        
        symbol = context.args[0].upper()
        processing_msg = await update.message.reply_text(f"📋 Recupero info per {symbol}...")
        
        try:
            info = get_ticker_info(symbol)
            
            if not info:
                await update.message.reply_text(f"❌ Nessuna informazione per {symbol}")
                return
            
            # Formatta messaggio
            message = f"""
📊 *INFORMAZIONI {symbol}*

*Nome:* {info.get('name', 'N/A')}
*Settore:* {info.get('sector', 'N/A')}
*Industria:* {info.get('industry', 'N/A')}

💵 *Prezzi:*
• Attuale: ${info.get('current_price', 'N/A'):.2f}
• Apertura: ${info.get('open', 'N/A'):.2f}
• Chiusura Prec.: ${info.get('previous_close', 'N/A'):.2f}
• Massimo: ${info.get('day_high', 'N/A'):.2f}
• Minimo: ${info.get('day_low', 'N/A'):.2f}

📈 *Metriche:*
• Market Cap: ${info.get('market_cap', 0):,.0f}
• Volume: {info.get('volume', 0):,}
• P/E Ratio: {info.get('pe_ratio', 'N/A')}
• Dividend Yield: {info.get('dividend_yield', 'N/A')}
• Beta: {info.get('beta', 'N/A')}

🎯 *Range 52 Settimane:*
• High: ${info.get('52w_high', 'N/A'):.2f}
• Low: ${info.get('52w_low', 'N/A'):.2f}

💰 *Valuta:* {info.get('currency', 'N/A')}
🌍 *Paese:* {info.get('country', 'N/A')}
            """
            
            await update.message.reply_text(message, parse_mode='Markdown')
            
            # Pulsanti azioni
            keyboard = [
                [
                    InlineKeyboardButton("📊 Analisi", callback_data=f"analyze_{symbol}"),
                    InlineKeyboardButton("📈 Grafico 3M", callback_data=f"chart_{symbol}_3"),
                    InlineKeyboardButton("📉 Grafico 6M", callback_data=f"chart_{symbol}_6")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.message.reply_text(
                "Azioni disponibili:",
                reply_markup=reply_markup
            )
            
        except Exception as e:
            await update.message.reply_text(f"❌ Errore: {str(e)}")
        finally:
            await processing_msg.delete()
    
    async def analyze_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Gestisce comando /analyze"""
        # Gestisci sia message che callback_query
        if update.message:
            message_source = update.message
        elif update.callback_query and update.callback_query.message:
            message_source = update.callback_query.message
        else:
            # Non possiamo rispondere
            return
        
        if not context.args:
            await message_source.reply_text("❌ Specifica un simbolo. Es: /analyze AAPL")
            return
        
        symbol = context.args[0].upper()
        
        # Controlla se c'è un secondo argomento per il periodo
        period = 3  # Default
        if len(context.args) > 1 and context.args[1].isdigit():
            period = min(int(context.args[1]), 12)  # Massimo 12 mesi
        
        processing_msg = await message_source.reply_text(f"📊 Analizzando {symbol}...")
        
        try:
            analyzer = TradingAnalyzer(symbol)
            
            # Analisi
            df = analyzer.analyze_period(period)
            
            if df is None or df.empty:
                await message_source.reply_text(f"❌ Impossibile analizzare {symbol}")
                return
            
            # Calcola metriche
            latest = df.iloc[-1]
            price_change = ((latest['Close'] / df['Close'].iloc[0]) - 1) * 100
            
            # Crea messaggio di riepilogo
            message = f"""
📈 *ANALISI {symbol} - {period} MESI*

💵 *Prezzi:*
• Attuale: ${latest['Close']:.2f}
• Apertura: ${latest['Open']:.2f}
• Massimo: ${latest['High']:.2f}
• Minimo: ${latest['Low']:.2f}
• Variazione Periodo: {price_change:+.2f}%

📊 *Indicatori Tecnici:*
• RSI: {latest['RSI']:.1f} {'(Sovracomprato)' if latest['RSI'] > 70 else '(Sovravenduto)' if latest['RSI'] < 30 else '(Neutrale)'}
• MACD: {'📈 Rialzista' if latest['MACD'] > latest['Signal_Line'] else '📉 Ribassista'}
• SMA 20: ${latest['SMA_20']:.2f}
• SMA 50: ${latest['SMA_50']:.2f}

📉 *Segnale Tendenza:*
• Medie: {'↑' if latest['SMA_20'] > latest['SMA_50'] else '↓'}
• Volume: {analyzer._analyze_volume_trend(df)}
"""
            
            await message_source.reply_text(message, parse_mode='Markdown')
            
            # Pulsanti per grafici
            keyboard = [
                [
                    InlineKeyboardButton("📊 Info Dettagliate", callback_data=f"info_{symbol}"),
                    InlineKeyboardButton("🔍 Cerca Altro", callback_data="search_more")
                ],
                [
                    InlineKeyboardButton("📈 Grafico 3M", callback_data=f"chart_{symbol}_3"),
                    InlineKeyboardButton("📉 Grafico 6M", callback_data=f"chart_{symbol}_6"),
                    InlineKeyboardButton("📅 Grafico 1M", callback_data=f"chart_{symbol}_1")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await message_source.reply_text(
                "Scegli un'azione:",
                reply_markup=reply_markup
            )
            
        except Exception as e:
            await message_source.reply_text(f"❌ Errore: {str(e)}")
        finally:
            await processing_msg.delete()
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comando /help"""
        help_text = """
        🤖 *Telegram Trading Bot - Guida Completa*
        
        *Comandi Principali:*
        /start - Avvia il bot
        /search [nome] - Cerca ticker per nome
        /analyze [SIMBOLO] - Analisi tecnica
        /info [SIMBOLO] - Informazioni dettagliate
        /help - Mostra questa guida
        
        *Esempi:*
        /search apple
        /analyze AAPL
        /info TSLA 6 (per 6 mesi)
        /analyze BTC-USD
        
        *Indicatori Forniti:*
        • Prezzi (Open, High, Low, Close)
        • Medie Mobili (SMA 20/50, EMA 12/26)
        • MACD e Signal Line
        • RSI (Relative Strength Index)
        • Bollinger Bands
        • Analisi Volume
        
        *Interpretazione:*
        • RSI < 30: Sovravenduto
        • RSI > 70: Sovracomprato
        • MACD > Signal: Rialzista
        • SMA 20 > SMA 50: Trend positivo
        
        *Ticker Supportati:*
        • Azioni: AAPL, TSLA, MSFT, GOOGL
        • ETF: SPY, QQQ, VOO
        • Cripto: BTC-USD, ETH-USD
        • Indici: ^GSPC (S&P 500)
        
        ⚠️ *Disclaimer:* Non è consiglio finanziario.
        """
        
        await update.message.reply_text(help_text, parse_mode='Markdown')
    
    # ========== FUNZIONI PER GRAFICI MATPLOTLIB ==========
    
    def create_matplotlib_chart(self, df: pd.DataFrame, symbol: str, period: str) -> io.BytesIO:
        """Crea grafico con matplotlib e ritorna BytesIO"""
        if not MATPLOTLIB_AVAILABLE:
            return None
            
        if df.empty or len(df) < 5:
            return None
        
        try:
            # Crea figura con tema scuro
            plt.style.use('dark_background')
            fig, axes = plt.subplots(3, 1, figsize=(10, 12), height_ratios=[3, 1, 1])
            
            # Titolo
            fig.suptitle(f'{symbol} - Analisi {period}', 
                        fontsize=14, fontweight='bold', color='white')
            
            # 1. Grafico prezzi
            ax1 = axes[0]
            ax1.plot(df.index, df['Close'], label='Prezzo', color='cyan', linewidth=2, alpha=0.9)
            ax1.plot(df.index, df['SMA_20'], label='SMA 20', color='yellow', alpha=0.7, linestyle='--')
            ax1.plot(df.index, df['SMA_50'], label='SMA 50', color='magenta', alpha=0.7, linestyle=':')
            
            # Colora area
            ax1.fill_between(df.index, df['Close'], df['SMA_20'], 
                            where=(df['Close'] > df['SMA_20']), 
                            alpha=0.2, color='lime', label='Sopra SMA20')
            
            ax1.set_ylabel('Prezzo ($)', color='white')
            ax1.legend(loc='upper left')
            ax1.grid(True, alpha=0.2, color='gray')
            ax1.set_facecolor('#0E1117')
            
            # 2. Grafico RSI
            ax2 = axes[1]
            ax2.plot(df.index, df['RSI'], label='RSI', color='violet', linewidth=1.5)
            ax2.axhline(y=70, color='red', linestyle='--', alpha=0.5, label='Sovracomprato (70)')
            ax2.axhline(y=30, color='green', linestyle='--', alpha=0.5, label='Sovravenduto (30)')
            ax2.fill_between(df.index, df['RSI'], 70, where=(df['RSI'] >= 70), 
                            alpha=0.3, color='red')
            ax2.fill_between(df.index, df['RSI'], 30, where=(df['RSI'] <= 30), 
                            alpha=0.3, color='green')
            
            ax2.set_ylim(0, 100)
            ax2.set_ylabel('RSI', color='white')
            ax2.legend(loc='upper left')
            ax2.grid(True, alpha=0.2, color='gray')
            ax2.set_facecolor('#0E1117')
            
            # 3. Grafico MACD
            ax3 = axes[2]
            ax3.plot(df.index, df['MACD'], label='MACD', color='deepskyblue', linewidth=1.5)
            ax3.plot(df.index, df['Signal_Line'], label='Signal', color='orange', linewidth=1.5, alpha=0.8)
            
            # Istogramma MACD colorato
            colors = ['lime' if h >= 0 else 'red' for h in df['MACD_Histogram']]
            ax3.bar(df.index, df['MACD_Histogram'], color=colors, alpha=0.6, width=0.8)
            
            ax3.axhline(y=0, color='white', linestyle='-', alpha=0.3)
            ax3.set_ylabel('MACD', color='white')
            ax3.legend(loc='upper left')
            ax3.grid(True, alpha=0.2, color='gray')
            ax3.set_facecolor('#0E1117')
            
            # Formatta date
            for ax in axes:
                ax.tick_params(colors='white')
                ax.xaxis.set_tick_params(rotation=45)
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            # Salva in buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=120, bbox_inches='tight',
                       facecolor='#0E1117', edgecolor='none')
            plt.close(fig)  # IMPORTANTE: libera memoria
            buf.seek(0)
            
            return buf
            
        except Exception as e:
            logger.error(f"Errore creazione grafico matplotlib: {e}")
            return None
    
    def create_simple_chart(self, df: pd.DataFrame, symbol: str, period: str) -> io.BytesIO:
        """Crea grafico semplice (solo prezzo e RSI)"""
        if not MATPLOTLIB_AVAILABLE:
            return None
            
        if df.empty or len(df) < 5:
            return None
        
        try:
            plt.style.use('default')
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
            
            # Grafico prezzi
            ax1.plot(df.index, df['Close'], 'b-', linewidth=2, label='Prezzo')
            ax1.set_ylabel('Prezzo ($)', fontsize=10)
            ax1.set_title(f'{symbol} - {period}', fontsize=12)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Grafico RSI
            ax2.plot(df.index, df['RSI'], 'r-', linewidth=1.5, label='RSI')
            ax2.axhline(y=70, color='gray', linestyle='--', alpha=0.5)
            ax2.axhline(y=30, color='gray', linestyle='--', alpha=0.5)
            ax2.set_ylabel('RSI', fontsize=10)
            ax2.set_ylim(0, 100)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            
            return buf
            
        except Exception as e:
            logger.error(f"Errore creazione grafico semplice: {e}")
            return None
    
    # ========== GESTIONE CALLBACK ==========
    
    async def quick_analyze_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE, symbol: str):
        """Analisi rapida da callback"""
        query = update.callback_query
        await query.answer()
        
        await query.edit_message_text(text=f"⚡ Analisi rapida {symbol}...")
        
        try:
            analyzer = TradingAnalyzer(symbol)
            df = analyzer.analyze_period(3)  # Default 3 mesi
            
            if df is not None:
                latest = df.iloc[-1]
                price_change = ((latest['Close'] / df['Close'].iloc[0]) - 1) * 100
                
                message = f"""
⚡ *ANALISI RAPIDA {symbol}*

💵 Prezzo: ${latest['Close']:.2f}
📈 Variazione 3M: {price_change:+.2f}%
📊 RSI: {latest['RSI']:.1f}
📉 MACD: {'↑' if latest['MACD'] > latest['Signal_Line'] else '↓'}
"""
                
                await query.edit_message_text(text=message, parse_mode='Markdown')
                
                # Pulsanti per analisi completa
                keyboard = [
                    [
                        InlineKeyboardButton("📊 Analisi Completa", callback_data=f"analyze_{symbol}"),
                        InlineKeyboardButton("📋 Info", callback_data=f"info_{symbol}")
                    ]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
                
                await context.bot.send_message(
                    chat_id=query.message.chat_id,
                    text="Vuoi maggiori dettagli?",
                    reply_markup=reply_markup
                )
                
        except Exception as e:
            await query.edit_message_text(f"❌ Errore: {str(e)}")
    
    async def callback_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Gestisce tutte le callback"""
        query = update.callback_query
        await query.answer()
        
        data = query.data
        
        try:
            if data.startswith("quick_analyze_"):
                symbol = data.replace("quick_analyze_", "")
                await self.quick_analyze_callback(update, context, symbol)
            
            elif data.startswith("analyze_"):
                symbol = data.replace("analyze_", "")
                await self.analyze_from_callback(update, context, symbol)
            
            elif data.startswith("info_"):
                symbol = data.replace("info_", "")
                await self.info_from_callback(update, context, symbol)
            
            elif data.startswith("chart_"):
                parts = data.split("_")
                symbol = parts[1]
                period = int(parts[2])
                await self.chart_callback(update, context, symbol, period)
            
            elif data == "search_more":
                await query.edit_message_text("🔍 Cosa vuoi cercare? Usa /search [nome]")
        
        except Exception as e:
            logger.error(f"Errore in callback_handler: {e}")
            await query.message.reply_text(f"❌ Errore: {str(e)}")
    
    async def analyze_from_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE, symbol: str):
        """Analisi da callback"""
        query = update.callback_query
        await query.answer()
        
        # Usa query.message invece di update.message
        processing_msg = await query.message.reply_text(f"📊 Analizzando {symbol}...")
        
        try:
            # Simula comando /analyze
            context.args = [symbol]
            await self.analyze_command(update, context)
        finally:
            await processing_msg.delete()
    
    async def info_from_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE, symbol: str):
        """Info da callback"""
        query = update.callback_query
        await query.answer()
        
        # Usa query.message invece di update.message
        processing_msg = await query.message.reply_text(f"📋 Recupero info per {symbol}...")
        
        try:
            # Simula comando /info
            context.args = [symbol]
            await self.info_command(update, context)
        finally:
            await processing_msg.delete()
    
    async def chart_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE, 
                           symbol: str, period: int):
        """Gestisce callback per grafici"""
        query = update.callback_query
        await query.answer()
        
        await query.edit_message_text(text=f"📈 Generando grafico {symbol} ({period} mesi)...")
        
        try:
            analyzer = TradingAnalyzer(symbol)
            df = analyzer.analyze_period(period)
            
            if df is not None and not df.empty:
                # Prova con matplotlib
                chart_buffer = self.create_matplotlib_chart(df, symbol, f"{period} mesi")
                
                if chart_buffer:
                    # Invia immagine
                    await context.bot.send_photo(
                        chat_id=query.message.chat_id,
                        photo=chart_buffer,
                        caption=f"📊 {symbol} - Analisi {period} mesi\n"
                               f"💰 ${df['Close'].iloc[-1]:.2f} | "
                               f"RSI: {df['RSI'].iloc[-1]:.1f}"
                    )
                    chart_buffer.close()
                    
                else:
                    # Fallback: grafico semplice
                    simple_buffer = self.create_simple_chart(df, symbol, f"{period} mesi")
                    if simple_buffer:
                        await context.bot.send_photo(
                            chat_id=query.message.chat_id,
                            photo=simple_buffer,
                            caption=f"📈 {symbol} - {period} mesi (grafico semplice)"
                        )
                        simple_buffer.close()
                    else:
                        # Fallback: dati testuali
                        await self.send_chart_as_table(update, context, df, symbol, period)
                
                # Pulsanti
                keyboard = [
                    [
                        InlineKeyboardButton("📊 Analizza", callback_data=f"analyze_{symbol}"),
                        InlineKeyboardButton("📋 Info", callback_data=f"info_{symbol}")
                    ],
                    [
                        InlineKeyboardButton("📈 Grafico 3M", callback_data=f"chart_{symbol}_3"),
                        InlineKeyboardButton("📉 Grafico 6M", callback_data=f"chart_{symbol}_6")
                    ]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
                
                await context.bot.send_message(
                    chat_id=query.message.chat_id,
                    text="Altre azioni:",
                    reply_markup=reply_markup
                )
                
            else:
                await query.edit_message_text(f"❌ Dati insufficienti per {symbol}")
                
        except Exception as e:
            logger.error(f"Errore chart_callback: {e}")
            await query.edit_message_text(f"❌ Errore: {str(e)}")
    
    async def send_chart_as_table(self, update: Update, context: ContextTypes.DEFAULT_TYPE,
                                df: pd.DataFrame, symbol: str, period: int):
        """Invia dati come tabella (fallback)"""
        query = update.callback_query
        
        # Prendi ultimi 7 giorni
        recent = df.tail(7)
        
        table = f"📊 *{symbol} - Ultimi 7 giorni ({period} mesi)*\n\n"
        table += "```\n"
        table += "Data      | Prezzo   | RSI  | Trend\n"
        table += "----------|----------|------|--------\n"
        
        for idx, row in recent.iterrows():
            date = idx.strftime('%d/%m')
            price = f"${row['Close']:7.2f}"
            rsi = f"{row['RSI']:5.1f}"
            
            if row['RSI'] < 30:
                trend = "SOVRAVENDUTO"
            elif row['RSI'] > 70:
                trend = "SOVRACOMPRATO"
            elif row['MACD'] > row['Signal_Line']:
                trend = "RIALZISTA"
            else:
                trend = "RIBASSISTA"
            
            table += f"{date} | {price} | {rsi} | {trend}\n"
        
        table += "```\n"
        
        latest = df.iloc[-1]
        stats = f"""
📈 *Statistiche:*
• Prezzo: ${latest['Close']:.2f}
• Variazione: {((latest['Close'] / df['Close'].iloc[0]) - 1) * 100:+.2f}%
• RSI: {latest['RSI']:.1f}
• MACD: {'🟢 Positivo' if latest['MACD'] > 0 else '🔴 Negativo'}
"""
        
        await query.edit_message_text(text=table + stats, parse_mode='Markdown')
    
    # ========== ERROR HANDLER ==========
    
    async def error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Gestisce errori"""
        logger.error(f"Errore: {context.error}")
        
        try:
            # Prova a informare l'utente dell'errore
            if update and update.effective_message:
                await update.effective_message.reply_text(
                    "❌ Si è verificato un errore. Riprova più tardi."
                )
        except:
            pass
    
    # ========== AVVIO BOT ==========
    
    def run(self):
        """Avvia il bot"""
        if not self.token:
            print("❌ ERRORE: Token Telegram non trovato!")
            print("Controlla il file .env")
            return
        
        print(f"✅ Token valido: Bot @tradinganalysisrobot")
        print(f"🔗 Collegati a: https://t.me/tradinganalysisrobot")
        
        if not MATPLOTLIB_AVAILABLE:
            print("⚠️ Matplotlib non installato. Per grafici migliori:")
            print("   pip install matplotlib")
        
        try:
            application = Application.builder().token(self.token).build()
            
            # Aggiungi handler
            application.add_handler(CommandHandler("start", self.start))
            application.add_handler(CommandHandler("search", self.search_command))
            application.add_handler(CommandHandler("analyze", self.analyze_command))
            application.add_handler(CommandHandler("info", self.info_command))
            application.add_handler(CommandHandler("help", self.help_command))
            application.add_handler(CallbackQueryHandler(self.callback_handler))
            
            # Aggiungi error handler
            application.add_error_handler(self.error_handler)
            
            print("\n" + "=" * 60)
            print("🤖 TRADING BOT AVVIATO CON SUCCESSO!")
            print("=" * 60)
            print("\n📱 ISTRUZIONI:")
            print("1. Vai su Telegram (web o app)")
            print("2. Cerca: @tradinganalysisrobot")
            print("3. Clicca 'START' o invia /start")
            print("4. Prova i comandi:")
            print("   • /analyze AAPL")
            print("   • Clicca su 'Grafico 3M' per vedere il grafico")
            print("\n🔄 Bot in esecuzione...")
            print("=" * 60)
            
            application.run_polling(
                drop_pending_updates=True,
                allowed_updates=Update.ALL_TYPES
            )
            
        except Exception as e:
            print(f"\n❌ ERRORE: {e}")
            traceback.print_exc()

def main():
    bot = TelegramTradingBot()
    bot.run()

if __name__ == "__main__":
    main()