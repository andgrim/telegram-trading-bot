"""
Universal Trading Bot for Telegram - Enhanced Version
Complete analysis with advanced correlations and divergences
"""
import logging
import os
import asyncio
import html
from typing import Dict, Any
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler,
    MessageHandler, filters, ContextTypes
)
from telegram.constants import ParseMode, ChatAction

from analyzer import TradingAnalyzer
from chart_generator import ChartGenerator
from config import CONFIG

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class UniversalTradingBot:
    """Enhanced Universal Telegram bot with advanced correlations"""
    
    def __init__(self):
        self.config = CONFIG
        self.analyzer = TradingAnalyzer()
        self.chart_generator = ChartGenerator()
        
        # Telegram application
        self.application = None
        self.initialized = False
        
        # Period options
        self.periods = {
            '3m': '3 Months',
            '6m': '6 Months', 
            '1y': '1 Year',
            '2y': '2 Years',
            '5y': '5 Years'
        }
        
        # Example tickers
        self.example_tickers = {
            'US Stocks': ['AAPL', 'MSFT', 'TSLA', 'GOOGL', 'NVDA'],
            'European Stocks': ['ENEL.MI', 'AIR.PA', 'SAP.DE', 'HSBA.L'],
            'ETFs': ['SPY', 'QQQ', 'VOO', 'GLD'],
            'Crypto': ['BTC-USD', 'ETH-USD', 'XRP-USD'],
            'Indices': ['^GSPC', '^DJI', '^IXIC'],
        }
    
    def initialize(self) -> bool:
        """Initialize the bot application for polling"""
        if self.initialized:
            return True
        
        try:
            token = self.config.TELEGRAM_TOKEN
            if not token:
                logger.error("TELEGRAM_TOKEN not found in environment")
                return False
            
            logger.info("Initializing enhanced bot application for polling...")
            
            # Create application
            self.application = Application.builder().token(token).build()
            
            # Add handlers
            self._setup_handlers()
            
            self.initialized = True
            logger.info("✅ Enhanced bot application initialized successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize bot: {e}")
            return False
    
    def _setup_handlers(self):
        """Setup all command and message handlers"""
        if not self.application:
            return
        
        # Command handlers
        self.application.add_handler(CommandHandler("start", self._handle_start))
        self.application.add_handler(CommandHandler("help", self._handle_help))
        self.application.add_handler(CommandHandler("analyze", self._handle_analyze))
        self.application.add_handler(CommandHandler("examples", self._handle_examples))
        self.application.add_handler(CommandHandler("test", self._test_command))
        
        # Callback query handler
        self.application.add_handler(CallbackQueryHandler(self._handle_callback))
        
        # Text message handler
        self.application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_text)
        )
        
        logger.info("All handlers setup completed")
    
    def _clean_telegram_text(self, text: str) -> str:
        """Clean text for Telegram to avoid HTML parsing errors"""
        if not text:
            return text
        
        # Replace problematic characters with safe equivalents
        replacements = {
            '<': '&lt;',
            '>': '&gt;',
            '&': '&amp;',
            '"': '&quot;',
            "'": '&apos;'
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Remove any remaining HTML tags (simple approach)
        import re
        text = re.sub(r'<[^>]+>', '', text)
        
        # Ensure text is not too long for Telegram
        if len(text) > 4096:
            text = text[:4000] + "\n... [truncated]"
        
        return text
    
    def _get_period_keyboard(self, ticker: str) -> InlineKeyboardMarkup:
        """Create keyboard with period buttons for a ticker"""
        keyboard = []
        
        # Create two rows of buttons
        row1 = []
        row2 = []
        
        for i, (period, label) in enumerate(self.periods.items()):
            if i < 3:
                row1.append(InlineKeyboardButton(label, callback_data=f"analyze_{ticker}_{period}"))
            else:
                row2.append(InlineKeyboardButton(label, callback_data=f"analyze_{ticker}_{period}"))
        
        if row1:
            keyboard.append(row1)
        if row2:
            keyboard.append(row2)
        
        return InlineKeyboardMarkup(keyboard)
    
    def _get_analysis_keyboard(self, ticker: str, period: str) -> InlineKeyboardMarkup:
        """Create keyboard after analysis"""
        keyboard = [
            [
                InlineKeyboardButton("🔄 Change Period", callback_data=f"change_period_{ticker}"),
                InlineKeyboardButton("📈 New Ticker", callback_data="new_ticker")
            ],
            [
                InlineKeyboardButton("3 Months", callback_data=f"analyze_{ticker}_3m"),
                InlineKeyboardButton("6 Months", callback_data=f"analyze_{ticker}_6m"),
                InlineKeyboardButton("1 Year", callback_data=f"analyze_{ticker}_1y"),
            ],
            [
                InlineKeyboardButton("2 Years", callback_data=f"analyze_{ticker}_2y"),
                InlineKeyboardButton("5 Years", callback_data=f"analyze_{ticker}_5y"),
            ]
        ]
        return InlineKeyboardMarkup(keyboard)
    
    # Command Handlers
    async def _handle_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        welcome = """
🤖 **Enhanced Universal Trading Analysis Bot**

🔗 **NEW: Advanced Correlations & Divergences**
• A/D Line vs Price divergences
• Volume indicator correlations (A/D, OBV, MFI)
• Multi-timeframe RSI analysis
• Bollinger Bands + RSI squeeze detection
• MACD + Volume confirmation

🌍 **Supports ALL markets worldwide:**
• US Stocks (AAPL, MSFT, TSLA)
• European Stocks (ENEL.MI, AIR.PA, SAP.DE)
• ETFs & Funds (SPY, QQQ, VOO)
• Cryptocurrencies (BTC, ETH, XRP)
• Indices (SPX, DJI, NASDAQ)

📊 **Complete Technical Analysis:**
• RSI, MACD, Moving Averages (20 MA in green)
• Volume indicators with A/D Line
• Divergence and correlation detection
• Performance statistics
• Advanced pattern recognition

**Commands:**
/start - This menu
/analyze TICKER PERIOD - Quick analysis
/examples - Show ticker examples
/help - Detailed help

**Simply send a ticker symbol to start enhanced analysis!**
        """
        
        keyboard = [
            [InlineKeyboardButton("📈 Start Analysis", callback_data="new_ticker")],
            [InlineKeyboardButton("🇺🇸 US Stocks", callback_data="examples_US Stocks")],
            [InlineKeyboardButton("🔗 View Features", callback_data="help")]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            welcome,
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML
        )
    
    async def _test_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /test command"""
        await update.message.reply_text(
            "✅ Enhanced bot is working!\n\n"
            "Try these tickers for advanced correlations:\n"
            "/analyze AAPL 1y\n"
            "/analyze ENEL.MI 6m\n"
            "/analyze BTC-USD 3m\n"
            "/analyze SPY 2y",
            parse_mode=ParseMode.HTML
        )
    
    async def _handle_analyze(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /analyze command"""
        if not context.args:
            await update.message.reply_text(
                "Usage: /analyze TICKER PERIOD\n\n"
                "Examples:\n"
                "• /analyze AAPL 1y (Apple)\n"
                "• /analyze ENEL.MI 6m (Italian stock)\n"
                "• /analyze BTC-USD 3m (Bitcoin)\n"
                "• /analyze ^GSPC 2y (S&P 500)\n\n"
                "Available periods: 3m, 6m, 1y, 2y, 5y\n\n"
                "Use /examples for more tickers",
                parse_mode=ParseMode.HTML
            )
            return
        
        ticker = context.args[0].upper()
        
        # Check if period is provided
        if len(context.args) > 1:
            period = context.args[1].lower()
            # Validate period
            if period not in self.periods:
                await update.message.reply_text(
                    f"Invalid period: {period}\n\n"
                    "Available periods: 3m, 6m, 1y, 2y, 5y\n\n"
                    "Example: /analyze AAPL 1y",
                    parse_mode=ParseMode.HTML
                )
                return
        else:
            # Default to 1 year
            period = '1y'
        
        await self._perform_enhanced_analysis(update, context, ticker, period)
    
    async def _perform_enhanced_analysis(self, update: Update, context: ContextTypes.DEFAULT_TYPE, 
                                       ticker: str, period: str):
        """Perform enhanced analysis with advanced correlations"""
        chat_id = update.effective_chat.id
        
        # Send typing indicator
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        
        # Status message with enhanced features mention
        status = await context.bot.send_message(
            chat_id=chat_id,
            text=f"🔗 Enhanced Analysis for {ticker} ({self.periods.get(period, period)})...",
            parse_mode=ParseMode.HTML
        )
        
        try:
            # Perform enhanced analysis
            analysis = await self.analyzer.analyze_ticker(ticker, period)
            
            if not analysis['success']:
                error_msg = f"❌ Could not analyze {ticker}\n\n"
                error_msg += f"Error: {analysis.get('error', 'Unknown error')}\n\n"
                error_msg += "Try:\n"
                error_msg += "• Different ticker\n"
                error_msg += "• Shorter period (3m, 6m)\n"
                error_msg += "• Wait a moment and try again\n\n"
                error_msg += "Examples:\n"
                error_msg += "• AAPL, MSFT, TSLA\n"
                error_msg += "• ENEL.MI, AIR.PA\n"
                error_msg += "• SPY, BTC-USD\n"
                
                keyboard = [[InlineKeyboardButton("Try Again", callback_data="new_ticker")]]
                reply_markup = InlineKeyboardMarkup(keyboard)
                
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=error_msg,
                    reply_markup=reply_markup,
                    parse_mode=ParseMode.HTML
                )
                return
            
            # Generate chart
            chart_path = None
            try:
                chart_path = self.chart_generator.generate_price_chart(
                    analysis['data'], ticker, period
                )
            except Exception as e:
                logger.info(f"Chart generation skipped: {e}")
            
            # Count advanced correlations for notification
            advanced_correlations_count = 0
            if analysis.get('advanced_divergences'):
                for key, value in analysis['advanced_divergences'].items():
                    if isinstance(value, list):
                        advanced_correlations_count += len(value)
            
            if analysis.get('volume_correlations'):
                for key, value in analysis['volume_correlations'].items():
                    if isinstance(value, list) and key != 'correlation_coefficient':
                        advanced_correlations_count += len(value)
            
            # Enhance compact summary with correlation count
            compact_summary = analysis.get('compact_summary', '')
            if advanced_correlations_count > 0:
                correlation_note = f"🔗 {advanced_correlations_count} advanced correlations detected\n"
                compact_summary = correlation_note + compact_summary
            
            # Create keyboard with period options
            reply_markup = self._get_analysis_keyboard(ticker, period)
            
            # Clean the text for Telegram
            compact_summary_clean = self._clean_telegram_text(compact_summary)
            full_summary = analysis.get('summary', '')
            full_summary_clean = self._clean_telegram_text(full_summary)
            
            # Truncate if necessary
            if len(compact_summary_clean) > 1024:
                compact_summary_clean = compact_summary_clean[:1000] + "..."
            
            if len(full_summary_clean) > 4096:
                # Split long message into multiple parts
                full_summary_parts = []
                current_part = ""
                lines = full_summary_clean.split('\n')
                
                for line in lines:
                    if len(current_part) + len(line) + 1 < 4000:
                        current_part += line + '\n'
                    else:
                        full_summary_parts.append(current_part)
                        current_part = line + '\n'
                
                if current_part:
                    full_summary_parts.append(current_part)
                
                # Send chart if available
                if chart_path:
                    try:
                        with open(chart_path, 'rb') as f:
                            await context.bot.send_photo(
                                chat_id=chat_id,
                                photo=f,
                                caption=compact_summary_clean,
                                reply_markup=reply_markup
                            )
                        
                        # Send summary in parts
                        for i, part in enumerate(full_summary_parts):
                            if i == 0:
                                await context.bot.send_message(
                                    chat_id=chat_id,
                                    text=part,
                                    parse_mode=None
                                )
                            else:
                                await context.bot.send_message(
                                    chat_id=chat_id,
                                    text=part,
                                    parse_mode=None
                                )
                        
                    except Exception as e:
                        logger.error(f"Photo error: {e}")
                        # Send analysis without chart
                        for i, part in enumerate(full_summary_parts):
                            if i == 0:
                                await context.bot.send_message(
                                    chat_id=chat_id,
                                    text=part,
                                    reply_markup=reply_markup,
                                    parse_mode=None
                                )
                            else:
                                await context.bot.send_message(
                                    chat_id=chat_id,
                                    text=part,
                                    parse_mode=None
                                )
                    finally:
                        # Cleanup
                        try:
                            os.remove(chart_path)
                        except:
                            pass
                else:
                    # Send analysis without chart
                    for i, part in enumerate(full_summary_parts):
                        if i == 0:
                            await context.bot.send_message(
                                chat_id=chat_id,
                                text=part,
                                reply_markup=reply_markup,
                                parse_mode=None
                            )
                        else:
                            await context.bot.send_message(
                                chat_id=chat_id,
                                text=part,
                                parse_mode=None
                            )
            else:
                # Message is short enough for single message
                # Send chart if available
                if chart_path:
                    try:
                        with open(chart_path, 'rb') as f:
                            await context.bot.send_photo(
                                chat_id=chat_id,
                                photo=f,
                                caption=compact_summary_clean,
                                reply_markup=reply_markup
                            )
                        
                        # Send full analysis
                        await context.bot.send_message(
                            chat_id=chat_id,
                            text=full_summary_clean,
                            parse_mode=None
                        )
                        
                    except Exception as e:
                        logger.error(f"Photo error: {e}")
                        # Send analysis without chart
                        await context.bot.send_message(
                            chat_id=chat_id,
                            text=full_summary_clean,
                            reply_markup=reply_markup,
                            parse_mode=None
                        )
                    finally:
                        # Cleanup
                        try:
                            os.remove(chart_path)
                        except:
                            pass
                else:
                    # Send analysis without chart
                    await context.bot.send_message(
                        chat_id=chat_id,
                        text=full_summary_clean,
                        reply_markup=reply_markup,
                        parse_mode=None
                    )
            
        except Exception as e:
            logger.error(f"Enhanced analysis failed: {e}")
            error_msg = f"❌ Enhanced analysis failed for {ticker}\n\n"
            error_msg += f"Error: {str(e)[:100]}"
            await context.bot.send_message(
                chat_id=chat_id,
                text=error_msg,
                parse_mode=None
            )
        finally:
            # Delete status
            try:
                await status.delete()
            except:
                pass
    
    async def _handle_examples(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /examples command"""
        examples_text = """
📋 **Example Tickers for Enhanced Analysis:**

🔗 **US Stocks:**
• AAPL (Apple), MSFT (Microsoft), TSLA (Tesla)
• GOOGL (Google), NVDA (Nvidia), AMZN (Amazon)

🇪🇺 **European Stocks:**
• ENEL.MI (Enel Italy), AIR.PA (Airbus France)
• SAP.DE (SAP Germany), HSBA.L (HSBC UK)

📈 **ETFs & Funds:**
• SPY (S&P 500 ETF), QQQ (NASDAQ ETF)
• VOO (Vanguard S&P 500), GLD (Gold ETF)

₿ **Cryptocurrencies:**
• BTC-USD (Bitcoin), ETH-USD (Ethereum)
• XRP-USD (Ripple)

📊 **Indices:**
• ^GSPC (S&P 500), ^DJI (Dow Jones)
• ^IXIC (NASDAQ Composite)

**Simply click a ticker below or type one:**
"""
        
        # Create keyboard with example tickers
        keyboard = []
        
        # US Stocks
        keyboard.append([InlineKeyboardButton("AAPL", callback_data=f"ticker_AAPL"),
                        InlineKeyboardButton("MSFT", callback_data=f"ticker_MSFT"),
                        InlineKeyboardButton("TSLA", callback_data=f"ticker_TSLA")])
        
        # European Stocks
        keyboard.append([InlineKeyboardButton("ENEL.MI", callback_data=f"ticker_ENEL.MI"),
                        InlineKeyboardButton("AIR.PA", callback_data=f"ticker_AIR.PA"),
                        InlineKeyboardButton("SAP.DE", callback_data=f"ticker_SAP.DE")])
        
        # ETFs & Crypto
        keyboard.append([InlineKeyboardButton("SPY", callback_data=f"ticker_SPY"),
                        InlineKeyboardButton("BTC-USD", callback_data=f"ticker_BTC-USD"),
                        InlineKeyboardButton("^GSPC", callback_data=f"ticker_^GSPC")])
        
        keyboard.append([InlineKeyboardButton("📈 Enter Custom Ticker", callback_data="new_ticker")])
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            examples_text,
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML
        )
    
    async def _handle_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        help_text = """
📖 **Enhanced Trading Bot Help**

🔗 **NEW ADVANCED FEATURES:**
• A/D Line vs Price Divergences
• Volume Indicator Correlations (A/D, OBV, MFI)
• Multi-timeframe RSI Analysis
• Bollinger Bands + RSI Squeeze Detection
• MACD + Volume Confirmation

🌍 **SUPPORTED MARKETS:**
• US Stocks: AAPL, MSFT, TSLA, GOOGL, AMZN
• European Stocks (with suffix):
  - Italy: .MI (ENEL.MI)
  - France: .PA (AIR.PA)
  - Germany: .DE (SAP.DE)
• ETFs: SPY, QQQ, VOO, GLD
• Cryptocurrencies: BTC-USD, ETH-USD
• Indices: ^GSPC (S&P 500), ^DJI (Dow Jones)

📊 **ENHANCED INDICATORS:**
• Price with 20, 50, 200 MAs (20 MA in green)
• Volume indicators: A/D Line, OBV, MFI
• RSI (14) with multi-timeframe analysis
• MACD (12, 26, 9) with volume correlation
• Bollinger Bands with squeeze detection

⏰ **PERIOD OPTIONS:**
• 3 Months (3m)
• 6 Months (6m)
• 1 Year (1y) - Default
• 2 Years (2y)
• 5 Years (5y)

🎯 **ADVANCED CORRELATIONS:**
1. A/D Line vs Price: Smart money detection
2. Volume Confirmation: A/D, OBV, MFI alignment
3. BB + RSI Squeeze: Breakout probability
4. MACD + Volume: Signal strength confirmation

📈 **HOW TO USE:**
1. Send a ticker symbol (AAPL, ENEL.MI, BTC-USD)
2. Select analysis period
3. View enhanced results with correlations
4. Use buttons to change period or analyze new ticker

**EXAMPLES:**
/analyze AAPL 1y
/analyze ENEL.MI 6m
/analyze BTC-USD 3m
/analyze ^GSPC 2y

**Or simply type: AAPL, TSLA, BTC-USD**
        """
        
        keyboard = [[InlineKeyboardButton("📈 Start Enhanced Analysis", callback_data="new_ticker")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            help_text,
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML
        )
    
    async def _handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle callback queries"""
        query = update.callback_query
        await query.answer()
        
        data = query.data
        
        if data == "new_ticker":
            await query.edit_message_text(
                """
📊 Enter a ticker symbol for enhanced analysis:

**Examples with advanced correlations:**
• AAPL (Apple) - Volume correlations
• ENEL.MI (Enel Italy) - European analysis
• BTC-USD (Bitcoin) - Crypto correlations
• ^GSPC (S&P 500) - Index analysis
• SPY (S&P 500 ETF) - ETF analysis

**Or click examples below:""",
                parse_mode=ParseMode.HTML
            )
        
        elif data == "help":
            await self._handle_help(update, context)
        
        elif data.startswith("ticker_"):
            ticker = data.replace("ticker_", "")
            # Show period selection for this ticker
            await query.edit_message_text(
                f"🔗 Select period for {ticker} (Enhanced Analysis):",
                reply_markup=self._get_period_keyboard(ticker)
            )
        
        elif data.startswith("change_period_"):
            ticker = data.replace("change_period_", "")
            # Show period selection for this ticker
            await query.edit_message_text(
                f"🔄 Change period for {ticker}:",
                reply_markup=self._get_period_keyboard(ticker)
            )
        
        elif data.startswith("analyze_"):
            parts = data.split("_")
            if len(parts) >= 3:
                ticker = parts[1]
                period = parts[2]
                await self._perform_enhanced_analysis(update, context, ticker, period)
    
    async def _handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle text messages (ticker input)"""
        text = update.message.text.strip().upper()
        
        # Simple ticker validation
        if len(text) > 1 and len(text) < 20:
            # Show period selection for this ticker
            await update.message.reply_text(
                f"🔗 Select period for {text} (Enhanced Analysis):",
                reply_markup=self._get_period_keyboard(text)
            )
        else:
            await update.message.reply_text(
                """
Please enter a valid ticker symbol.

**Examples for advanced correlations:**
• AAPL, MSFT, TSLA (US Stocks)
• ENEL.MI, AIR.PA (European)
• BTC-USD, ETH-USD (Crypto)
• SPY, QQQ (ETFs)
• ^GSPC (S&P 500)

**Or use: /analyze TICKER PERIOD**""",
                parse_mode=ParseMode.HTML
            )
    
    def run(self):
        """Run the bot with polling"""
        if not self.initialized:
            if not self.initialize():
                logger.error("Failed to initialize enhanced bot. Exiting.")
                return
        
        logger.info("Starting enhanced bot polling...")
        print("\n" + "="*50)
        print("🤖 ENHANCED UNIVERSAL TRADING BOT STARTED")
        print("🔗 Features: Advanced Correlations & Divergences")
        print("🌍 Supports: All markets worldwide")
        print("📊 Includes: Volume correlations, multi-timeframe analysis")
        print("="*50 + "\n")
        self.application.run_polling(allowed_updates=Update.ALL_TYPES)