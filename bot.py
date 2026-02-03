"""
Universal Trading Bot for Telegram - Local Polling Version
Simple version without webhook for local use
Added period buttons for analysis
FIXED: Telegram HTML parsing error
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
    """Universal Telegram bot for ALL markets - Local polling version"""
    
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
            
            logger.info("Initializing bot application for polling...")
            
            # Create application
            self.application = Application.builder().token(token).build()
            
            # Add handlers
            self._setup_handlers()
            
            self.initialized = True
            logger.info("✅ Bot application initialized successfully")
            
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
🤖 **Universal Trading Analysis Bot - Local Version**

🌍 **Supports ALL markets worldwide:**
• US Stocks (AAPL, MSFT, TSLA)
• European Stocks (ENEL.MI, AIR.PA, SAP.DE)
• ETFs & Funds (SPY, QQQ, VOO)
• Cryptocurrencies (BTC, ETH, XRP)
• Indices (SPX, DJI, NASDAQ)

📊 **Complete Technical Analysis:**
• RSI, MACD, Moving Averages (20 MA in green)
• Volume and A/D Line
• Divergence detection
• Performance statistics

**Commands:**
/start - This menu
/analyze TICKER PERIOD - Quick analysis
/examples - Show ticker examples
/help - Detailed help

**Simply send a ticker symbol to start analysis!**
        """
        
        keyboard = [
            [InlineKeyboardButton("📈 Start Analysis", callback_data="new_ticker")],
            [InlineKeyboardButton("🇺🇸 US Stocks", callback_data="examples_US Stocks")],
            [InlineKeyboardButton("❓ Help", callback_data="help")]
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
            "✅ Bot is working!\n\n"
            "Try these tickers:\n"
            "/analyze AAPL 1y\n"
            "/analyze ENEL.MI 6m\n"
            "/analyze BTC-USD 3m",
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
        
        await self._perform_analysis(update, context, ticker, period)
    
    async def _perform_analysis(self, update: Update, context: ContextTypes.DEFAULT_TYPE, 
                              ticker: str, period: str):
        """Perform analysis - FIXED version with proper text cleaning"""
        chat_id = update.effective_chat.id
        
        # Send typing indicator
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        
        # Status message
        status = await context.bot.send_message(
            chat_id=chat_id,
            text=f"🔄 Analyzing {ticker} ({self.periods.get(period, period)})...",
            parse_mode=ParseMode.HTML
        )
        
        try:
            # Perform analysis
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
            
            # Create keyboard with period options
            reply_markup = self._get_analysis_keyboard(ticker, period)
            
            # Clean the text for Telegram
            compact_summary = analysis.get('compact_summary', '')
            full_summary = analysis.get('summary', '')
            
            # Clean text to avoid HTML parsing errors
            compact_summary_clean = self._clean_telegram_text(compact_summary)
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
            logger.error(f"Analysis failed: {e}")
            error_msg = f"❌ Analysis failed for {ticker}\n\n"
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
        examples_text = "📋 **Example Tickers by Category:**\n\n"
        
        for category, tickers in self.example_tickers.items():
            examples_text += f"**{category}:**\n"
            examples_text += f"• {', '.join(tickers[:5])}\n\n"
        
        examples_text += "**Simply click a ticker below or type one:**"
        
        # Create keyboard with example tickers
        keyboard = []
        for category, tickers in self.example_tickers.items():
            for ticker in tickers[:3]:  # Show first 3 from each category
                keyboard.append([InlineKeyboardButton(ticker, callback_data=f"ticker_{ticker}")])
        
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
📖 **Universal Trading Bot Help - Local Version**

**SUPPORTED MARKETS:**
• US Stocks: AAPL, MSFT, TSLA, GOOGL, AMZN
• European Stocks (with suffix):
  - Italy: .MI (ENEL.MI)
  - France: .PA (AIR.PA)
  - Germany: .DE (SAP.DE)
• ETFs: SPY, QQQ, VOO, GLD
• Cryptocurrencies: BTC-USD, ETH-USD
• Indices: ^GSPC (S&P 500), ^DJI (Dow Jones)

**PERIOD OPTIONS:**
• 3 Months (3m)
• 6 Months (6m)
• 1 Year (1y) - Default
• 2 Years (2y)
• 5 Years (5y)

**INDICATORS INCLUDED:**
• Price with 20, 50, 200 MAs (20 MA in green)
• Volume with A/D Line
• RSI (14)
• MACD (12, 26, 9)

**HOW TO USE:**
1. Send a ticker symbol (AAPL, ENEL.MI, BTC-USD)
2. Select analysis period
3. View results and chart
4. Use buttons to change period or analyze new ticker

**EXAMPLES:**
/analyze AAPL 1y
/analyze ENEL.MI 6m
/analyze BTC-USD 3m
/analyze ^GSPC 2y

**Or simply type: AAPL, TSLA, BTC-USD**
        """
        
        keyboard = [[InlineKeyboardButton("📈 Start Analysis", callback_data="new_ticker")]]
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
                "📊 Enter a ticker symbol:\n\n"
                "Examples:\n"
                "• AAPL (Apple)\n"
                "• ENEL.MI (Enel Italy)\n"
                "• BTC-USD (Bitcoin)\n"
                "• ^GSPC (S&P 500)\n\n"
                "Or click examples below:",
                parse_mode=ParseMode.HTML
            )
        
        elif data == "help":
            await self._handle_help(update, context)
        
        elif data.startswith("ticker_"):
            ticker = data.replace("ticker_", "")
            # Show period selection for this ticker
            await query.edit_message_text(
                f"📊 Select period for {ticker}:",
                reply_markup=self._get_period_keyboard(ticker)
            )
        
        elif data.startswith("change_period_"):
            ticker = data.replace("change_period_", "")
            # Show period selection for this ticker
            await query.edit_message_text(
                f"📊 Select period for {ticker}:",
                reply_markup=self._get_period_keyboard(ticker)
            )
        
        elif data.startswith("analyze_"):
            parts = data.split("_")
            if len(parts) >= 3:
                ticker = parts[1]
                period = parts[2]
                await self._perform_analysis(update, context, ticker, period)
    
    async def _handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle text messages (ticker input)"""
        text = update.message.text.strip().upper()
        
        # Simple ticker validation
        if len(text) > 1 and len(text) < 20:
            # Show period selection for this ticker
            await update.message.reply_text(
                f"📊 Select period for {text}:",
                reply_markup=self._get_period_keyboard(text)
            )
        else:
            await update.message.reply_text(
                "Please enter a valid ticker symbol.\n\n"
                "Examples: AAPL, ENEL.MI, BTC-USD\n\n"
                "Or use: /analyze TICKER PERIOD",
                parse_mode=ParseMode.HTML
            )
    
    def run(self):
        """Run the bot with polling"""
        if not self.initialized:
            if not self.initialize():
                logger.error("Failed to initialize bot. Exiting.")
                return
        
        logger.info("Starting bot polling...")
        self.application.run_polling(allowed_updates=Update.ALL_TYPES)