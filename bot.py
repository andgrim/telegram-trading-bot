"""
Universal Trading Bot for Telegram
Supports all Yahoo Finance markets worldwide
"""
import logging
import os
import asyncio
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler, 
    ContextTypes, MessageHandler, filters
)
from telegram.constants import ParseMode, ChatAction

from analyzer import TradingAnalyzer
from chart_generator import ChartGenerator
from config import CONFIG

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class UniversalTradingBot:
    """Universal Telegram bot for ALL markets"""
    
    def __init__(self):
        self.config = CONFIG
        self.analyzer = TradingAnalyzer()
        self.chart_generator = ChartGenerator()
        
        # All supported periods
        self.periods = ['3m', '6m', '1y', '2y', '3y', '5y', 'max']
        
        # Example tickers from different markets
        self.example_tickers = {
            'US Stocks': ['AAPL', 'MSFT', 'TSLA', 'AMZN', 'GOOGL', 'NVDA', 'META'],
            'European Stocks': ['ENEL.MI', 'AIR.PA', 'SAP.DE', 'ASML.AS', 'HSBA.L', 'BMW.DE', 'TTE.PA'],
            'ETFs & Funds': ['SPY', 'QQQ', 'VOO', 'GLD', 'SLV', 'BND', 'IWM'],
            'Commodities': ['GC=F', 'CL=F', 'SI=F', 'NG=F', 'HG=F', 'ZC=F'],
            'Cryptocurrencies': ['BTC-USD', 'ETH-USD', 'XRP-USD', 'SOL-USD', 'ADA-USD'],
            'Indices': ['^GSPC', '^DJI', '^IXIC', '^FTSE', '^GDAXI', '^FCHI', '^N225'],
            'Forex': ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X', 'USDCAD=X'],
        }
    
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Start command with universal support"""
        welcome = """
🤖 Universal Trading Analysis Bot

🌍 Supports ALL markets worldwide:
• US Stocks (AAPL, MSFT, TSLA)
• European Stocks (ENEL.MI, AIR.PA, SAP.DE)
• ETFs & Funds (SPY, QQQ, VOO)
• Commodities (GOLD, OIL, SILVER)
• Cryptocurrencies (BTC, ETH, XRP)
• Indices (SPX, DJI, FTSE, DAX)
• Forex (EUR/USD, GBP/USD, USD/JPY)

📊 Complete Technical Analysis:
• 25+ indicators (RSI, MACD, Stochastic, BB, ATR, A/D Line)
• Divergence detection
• Reversal pattern recognition
• Performance statistics
• Multi-timeframe charts

📈 Available Periods:
3m, 6m, 1y, 2y, 3y, 5y, max

Commands:
/start - This menu
/analyze TICKER PERIOD - Quick analysis
/examples - Show ticker examples
/periods - Show all available periods
/help - Detailed help
        """
        
        keyboard = [
            [InlineKeyboardButton("📈 Analyze Now", callback_data="ask_ticker")],
            [
                InlineKeyboardButton("🇺🇸 US", callback_data="examples_US Stocks"),
                InlineKeyboardButton("🇪🇺 Europe", callback_data="examples_European Stocks"),
                InlineKeyboardButton("📊 ETFs", callback_data="examples_ETFs & Funds")
            ],
            [
                InlineKeyboardButton("💰 Crypto", callback_data="examples_Cryptocurrencies"),
                InlineKeyboardButton("📉 Indices", callback_data="examples_Indices"),
                InlineKeyboardButton("⏰ Periods", callback_data="show_periods")
            ],
            [InlineKeyboardButton("❓ Help", callback_data="help")]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            welcome,
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML
        )
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Help command with universal instructions"""
        help_text = """
📖 Universal Trading Bot Help

SUPPORTED MARKETS:
• US Stocks: AAPL, MSFT, TSLA, GOOGL, AMZN, NVDA, META
• European Stocks: Use exchange suffix:
  - Italy: .MI (ENEL.MI, ISP.MI)
  - France: .PA (AIR.PA, TTE.PA)
  - Germany: .DE (SAP.DE, BMW.DE)
  - UK: .L (HSBA.L, BP.L)
• ETFs: SPY, QQQ, VOO, GLD, SLV, IWM
• Commodities: GOLD (GC=F), OIL (CL=F), SILVER (SI=F)
• Cryptocurrencies: BTC-USD, ETH-USD, XRP-USD, SOL-USD
• Indices: SPX (^GSPC), DJI (^DJI), FTSE (^FTSE), DAX (^GDAXI)
• Forex: EUR/USD (EURUSD=X), GBP/USD (GBPUSD=X)

📈 AVAILABLE PERIODS:
3m, 6m, 1y, 2y, 3y, 5y, max

📊 TECHNICAL INDICATORS INCLUDED:
• Price: Close, High, Low, Open
• Moving Averages: SMA 20/50/200, EMA 9/20/50
• Momentum: RSI, MACD, Stochastic, Williams %R, CCI, ROC
• Volume: OBV, A/D Line, Volume Ratio, MFI
• Volatility: Bollinger Bands, ATR, BB Width
• Patterns: Divergences, Reversal signals

📝 EXAMPLES:
/analyze AAPL 1y
/analyze ENEL.MI 6m
/analyze GOLD 3m
/analyze BTC-USD 1y
/analyze ^GSPC 2y
/analyze EURUSD=X 3m

🔍 TICKER FORMATS:
• Use Yahoo Finance format
• European stocks need suffix (.MI, .PA, .DE, etc.)
• Indices use ^ prefix (^GSPC, ^DJI)
• Commodities use =F suffix (GC=F, CL=F)
• Crypto use -USD suffix (BTC-USD)
• Forex use =X suffix (EURUSD=X)
        """
        
        keyboard = [
            [InlineKeyboardButton("📈 Start Analysis", callback_data="ask_ticker")],
            [InlineKeyboardButton("⏰ Show Periods", callback_data="show_periods")],
            [InlineKeyboardButton("📋 Examples", callback_data="more_examples")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            help_text,
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML
        )
    
    async def examples_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show ticker examples"""
        examples_text = "📋 Ticker Examples by Market:\n\n"
        
        for market, tickers in self.example_tickers.items():
            examples_text += f"<b>{market}:</b>\n"
            examples_text += " • " + " | ".join(tickers[:5]) + "\n\n"
        
        examples_text += "📈 <b>Available Periods:</b> 3m, 6m, 1y, 2y, 3y, 5y, max\n\n"
        examples_text += "<b>Usage:</b> /analyze TICKER PERIOD\n"
        examples_text += "<b>Example:</b> <code>/analyze ENEL.MI 1y</code>"
        
        keyboard = [
            [InlineKeyboardButton("📈 Analyze Now", callback_data="ask_ticker")],
            [InlineKeyboardButton("⏰ All Periods", callback_data="show_periods")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            examples_text,
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML
        )
    
    async def periods_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show all available periods"""
        periods_text = "⏰ <b>Available Time Periods:</b>\n\n"
        
        period_descriptions = {
            '3m': "• <b>3 Months</b> - Short-term analysis (63 trading days)",
            '6m': "• <b>6 Months</b> - Medium-term analysis (126 trading days)",
            '1y': "• <b>1 Year</b> - Standard analysis (252 trading days)",
            '2y': "• <b>2 Years</b> - Long-term analysis (504 trading days)",
            '3y': "• <b>3 Years</b> - Extended analysis (756 trading days)",
            '5y': "• <b>5 Years</b> - Historical analysis (1260 trading days)",
            'max': "• <b>Max</b> - All available data (varies by ticker)"
        }
        
        for period in self.periods:
            periods_text += period_descriptions.get(period, f"• <b>{period}</b>\n")
        
        periods_text += "\n📊 <b>Recommended:</b>\n"
        periods_text += "• For day trading: 3m or 6m\n"
        periods_text += "• For swing trading: 1y or 2y\n"
        periods_text += "• For investment: 3y or 5y\n"
        periods_text += "• For historical: max\n"
        
        keyboard = [[InlineKeyboardButton("📈 Start Analysis", callback_data="ask_ticker")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            periods_text,
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML
        )
    
    async def analyze_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /analyze command"""
        if not context.args:
            await update.message.reply_text(
                "📊 <b>Usage:</b> <code>/analyze TICKER PERIOD</code>\n\n"
                "<b>Examples:</b>\n"
                "• <code>/analyze AAPL 1y</code> (Apple)\n"
                "• <code>/analyze ENEL.MI 6m</code> (Italian stock)\n"
                "• <code>/analyze GOLD 3m</code> (Gold)\n"
                "• <code>/analyze BTC-USD 1y</code> (Bitcoin)\n"
                "• <code>/analyze ^GSPC 2y</code> (S&P 500)\n\n"
                "<b>Available Periods:</b> 3m, 6m, 1y, 2y, 3y, 5y, max\n\n"
                "Need help? Use <code>/examples</code> or <code>/help</code>",
                parse_mode=ParseMode.HTML
            )
            return
        
        ticker = context.args[0].upper()
        period = context.args[1] if len(context.args) > 1 else '1y'
        
        # Validate period
        if period not in self.periods:
            await update.message.reply_text(
                f"❌ Invalid period: {period}\n\n"
                f"📅 <b>Available periods:</b> {', '.join(self.periods)}\n\n"
                f"Try: <code>/analyze {ticker} 1y</code>",
                parse_mode=ParseMode.HTML
            )
            return
        
        await self.perform_analysis(update, context, ticker, period)
    
    async def show_examples(self, update: Update, context: ContextTypes.DEFAULT_TYPE, market: str):
        """Show examples for specific market"""
        query = update.callback_query
        await query.answer()
        
        if market not in self.example_tickers:
            await query.message.reply_text("Invalid market")
            return
        
        tickers = self.example_tickers[market]
        
        message = f"📋 <b>{market} Ticker Examples:</b>\n\n"
        message += "Click any ticker to analyze:\n"
        
        # Create buttons for tickers
        keyboard = []
        row = []
        for i, ticker in enumerate(tickers):
            row.append(InlineKeyboardButton(ticker, callback_data=f"quick_{ticker}"))
            if (i + 1) % 3 == 0 or i == len(tickers) - 1:
                keyboard.append(row)
                row = []
        
        keyboard.append([
            InlineKeyboardButton("⬅️ Back", callback_data="back_to_start"),
            InlineKeyboardButton("⏰ Periods", callback_data="show_periods")
        ])
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.message.edit_text(
            text=message,
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML
        )
    
    async def show_periods_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show periods in callback"""
        query = update.callback_query
        await query.answer()
        
        await self.periods_command(update, context)
    
    async def ask_for_ticker(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Ask user for ticker"""
        query = update.callback_query
        await query.answer()
        
        message = """
📝 <b>Enter Ticker Symbol</b>

🌍 <b>Universal format:</b> Use Yahoo Finance format

<b>Examples:</b>
• US: AAPL, MSFT, TSLA
• Europe: ENEL.MI (Italy), AIR.PA (France), SAP.DE (Germany)
• ETFs: SPY, QQQ, VOO
• Commodities: GOLD (GC=F), OIL (CL=F)
• Crypto: BTC-USD, ETH-USD
• Indices: SPX (^GSPC), DJI (^DJI)
• Forex: EUR/USD (EURUSD=X)

📅 <b>Enter ticker:</b>
        """
        
        keyboard = [
            [
                InlineKeyboardButton("AAPL", callback_data="quick_AAPL"),
                InlineKeyboardButton("MSFT", callback_data="quick_MSFT"),
                InlineKeyboardButton("TSLA", callback_data="quick_TSLA")
            ],
            [
                InlineKeyboardButton("ENEL.MI", callback_data="quick_ENEL.MI"),
                InlineKeyboardButton("AIR.PA", callback_data="quick_AIR.PA"),
                InlineKeyboardButton("SAP.DE", callback_data="quick_SAP.DE")
            ],
            [
                InlineKeyboardButton("SPY", callback_data="quick_SPY"),
                InlineKeyboardButton("GOLD", callback_data="quick_GOLD"),
                InlineKeyboardButton("BTC-USD", callback_data="quick_BTC-USD")
            ],
            [
                InlineKeyboardButton("📋 More Examples", callback_data="more_examples"),
                InlineKeyboardButton("⏰ Periods", callback_data="show_periods")
            ]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.message.edit_text(
            message,
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML
        )
    
    async def ask_for_period(self, update: Update, context: ContextTypes.DEFAULT_TYPE, ticker: str):
        """Ask for timeframe with all periods"""
        if update.callback_query:
            message_obj = update.callback_query.message
            await update.callback_query.answer()
        else:
            message_obj = update.message
        
        message = f"📊 <b>Select timeframe for {ticker}</b>"
        
        keyboard = [
            [
                InlineKeyboardButton("3 Months", callback_data=f"analyze_{ticker}_3m"),
                InlineKeyboardButton("6 Months", callback_data=f"analyze_{ticker}_6m")
            ],
            [
                InlineKeyboardButton("1 Year", callback_data=f"analyze_{ticker}_1y"),
                InlineKeyboardButton("2 Years", callback_data=f"analyze_{ticker}_2y")
            ],
            [
                InlineKeyboardButton("3 Years", callback_data=f"analyze_{ticker}_3y"),
                InlineKeyboardButton("5 Years", callback_data=f"analyze_{ticker}_5y")
            ],
            [
                InlineKeyboardButton("Max", callback_data=f"analyze_{ticker}_max"),
                InlineKeyboardButton("⬅️ Back", callback_data="ask_ticker")
            ]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await message_obj.reply_text(
            message,
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML
        )
    
    async def perform_analysis(self, update: Update, context: ContextTypes.DEFAULT_TYPE, 
                             ticker: str, period: str):
        """Perform analysis"""
        chat_id = update.effective_chat.id
        
        # Send typing indicator
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        
        # Status message
        status = await context.bot.send_message(
            chat_id=chat_id,
            text=f"🔄 Analyzing {ticker} ({period})...\n⏳ This may take a moment...",
            parse_mode=ParseMode.HTML
        )
        
        try:
            # Perform analysis
            analysis = await self.analyzer.analyze_ticker(ticker, period)
            
            if not analysis['success']:
                error_msg = f"❌ <b>Could not analyze {ticker}</b>\n\n"
                error_msg += f"<b>Error:</b> {analysis.get('error', 'Unknown error')}\n\n"
                error_msg += "<b>Possible solutions:</b>\n"
                error_msg += "1. Check if ticker exists on Yahoo Finance\n"
                error_msg += "2. Try a different period (1y, 6m)\n"
                error_msg += "3. Wait 60 seconds and try again\n"
                error_msg += "4. Check ticker format\n\n"
                error_msg += "<b>Examples:</b>\n"
                error_msg += "• AAPL, MSFT, TSLA (US)\n"
                error_msg += "• ENEL.MI, AIR.PA (Europe)\n"
                error_msg += "• SPY, GOLD, BTC-USD\n"
                error_msg += "• ^GSPC, EURUSD=X\n\n"
                error_msg += "<b>Available Periods:</b> 3m, 6m, 1y, 2y, 3y, 5y, max"
                
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=error_msg,
                    parse_mode=ParseMode.HTML
                )
                return
            
            # Generate chart
            chart_path = None
            try:
                logger.info(f"Generating chart for {ticker} ({period})...")
                chart_path = self.chart_generator.generate_price_chart(
                    analysis['data'], ticker, period
                )
                logger.info(f"Chart generated: {chart_path}")
            except Exception as e:
                logger.error(f"Chart generation failed: {e}")
                chart_path = None
            
            # Action buttons
            keyboard = [
                [
                    InlineKeyboardButton("🔄 New Analysis", callback_data="ask_ticker"),
                    InlineKeyboardButton(f"📈 {ticker}", callback_data=f"analyze_{ticker}_{period}")
                ],
                [
                    InlineKeyboardButton("3m", callback_data=f"analyze_{ticker}_3m"),
                    InlineKeyboardButton("6m", callback_data=f"analyze_{ticker}_6m"),
                    InlineKeyboardButton("1y", callback_data=f"analyze_{ticker}_1y"),
                    InlineKeyboardButton("2y", callback_data=f"analyze_{ticker}_2y")
                ],
                [
                    InlineKeyboardButton("3y", callback_data=f"analyze_{ticker}_3y"),
                    InlineKeyboardButton("5y", callback_data=f"analyze_{ticker}_5y"),
                    InlineKeyboardButton("max", callback_data=f"analyze_{ticker}_max")
                ]
            ]
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            # Send chart if available
            if chart_path and os.path.exists(chart_path):
                try:
                    with open(chart_path, 'rb') as f:
                        caption = analysis['compact_summary']
                        if len(caption) > 1024:
                            caption = caption[:1020] + "..."
                        
                        await context.bot.send_photo(
                            chat_id=chat_id,
                            photo=f,
                            caption=caption,
                            reply_markup=reply_markup,
                            parse_mode=ParseMode.HTML
                        )
                    
                    # Send full analysis
                    await context.bot.send_message(
                        chat_id=chat_id,
                        text=analysis['summary'],
                        parse_mode=ParseMode.HTML
                    )
                    
                    logger.info(f"✅ Chart and analysis sent for {ticker}")
                    
                except Exception as e:
                    logger.error(f"Photo sending error: {e}")
                    # Send analysis without chart
                    await context.bot.send_message(
                        chat_id=chat_id,
                        text=f"📊 {ticker} Analysis ({period})\n\n{analysis['summary']}",
                        reply_markup=reply_markup,
                        parse_mode=ParseMode.HTML
                    )
                finally:
                    # Cleanup
                    try:
                        os.remove(chart_path)
                    except:
                        pass
            else:
                # Send analysis without chart
                logger.info(f"Sending analysis without chart for {ticker}")
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=f"📊 {ticker} Analysis ({period})\n\n{analysis['summary']}",
                    reply_markup=reply_markup,
                    parse_mode=ParseMode.HTML
                )
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"❌ Analysis failed for {ticker}\n\n"
                     f"<b>Error:</b> {str(e)[:100]}\n\n"
                     f"Please try again or use a different ticker/period.",
                parse_mode=ParseMode.HTML
            )
        finally:
            # Delete status
            try:
                await status.delete()
            except:
                pass
    
    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle callback queries"""
        query = update.callback_query
        await query.answer()
        
        data = query.data
        
        if data == "ask_ticker":
            await self.ask_for_ticker(update, context)
        
        elif data == "help":
            await self.help_command(update, context)
        
        elif data == "back_to_start":
            await self.start(update, context)
        
        elif data == "more_examples":
            await self.examples_command(update, context)
        
        elif data == "show_periods":
            await self.periods_command(update, context)
        
        elif data.startswith("examples_"):
            market = data.replace("examples_", "")
            await self.show_examples(update, context, market)
        
        elif data.startswith("quick_"):
            ticker = data.split("_")[1]
            await self.ask_for_period(update, context, ticker)
        
        elif data.startswith("analyze_"):
            parts = data.split("_")
            if len(parts) >= 3:
                ticker = parts[1]
                period = parts[2]
                await self.perform_analysis(update, context, ticker, period)
    
    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle text input"""
        text = update.message.text.strip().upper().replace('$', '')
        
        # Basic validation
        if 1 <= len(text) <= 20 and any(c.isalnum() for c in text):
            await self.ask_for_period(update, context, text)
        else:
            await update.message.reply_text(
                f"❌ <b>Invalid ticker:</b> {text}\n\n"
                "<b>Valid ticker formats:</b>\n"
                "• AAPL, MSFT, TSLA (US stocks)\n"
                "• ENEL.MI, AIR.PA (European stocks)\n"
                "• SPY, GOLD, BTC-USD\n"
                "• ^GSPC, ^DJI (indices)\n"
                "• EURUSD=X (forex)\n\n"
                "<b>Try:</b> <code>/analyze AAPL 1y</code>\n"
                "<b>Or use:</b> <code>/examples</code> for more examples",
                parse_mode=ParseMode.HTML
            )
    
    def run(self):
        """Run the bot"""
        if not self.config.TELEGRAM_TOKEN:
            print("❌ ERROR: TELEGRAM_TOKEN not found!")
            print("Set TELEGRAM_TOKEN environment variable")
            return
        
        app = Application.builder().token(self.config.TELEGRAM_TOKEN).build()
        
        # Add handlers
        app.add_handler(CommandHandler("start", self.start))
        app.add_handler(CommandHandler("help", self.help_command))
        app.add_handler(CommandHandler("analyze", self.analyze_command))
        app.add_handler(CommandHandler("examples", self.examples_command))
        app.add_handler(CommandHandler("periods", self.periods_command))
        app.add_handler(CallbackQueryHandler(self.handle_callback))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text))
        
        # Error handler
        app.add_error_handler(self.error_handler)
        
        # Start
        print("🤖 Universal Trading Bot starting...")
        print("🌍 Supports ALL Yahoo Finance markets")
        print(f"⏰ Available periods: {', '.join(self.periods)}")
        print(f"⏱️ Yahoo delay: {self.config.YAHOO_DELAY_SECONDS}s")
        print("📊 Includes: Price, Volume, RSI, MACD, Stochastic, A/D Line, Bollinger Bands")
        
        app.run_polling(allowed_updates=Update.ALL_TYPES)
    
    async def error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle errors"""
        logger.error(f"Error: {context.error}")
        
        try:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="❌ An error occurred. Please try again or use /start",
                parse_mode=ParseMode.HTML
            )
        except:
            pass