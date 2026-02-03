"""
Universal Trading Bot for Telegram - Local Polling Version
Simple version without webhook for local use
"""
import logging
import os
import asyncio
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
• RSI, MACD, Moving Averages
• Volume and A/D Line
• Divergence detection
• Performance statistics

**Commands:**
/start - This menu
/analyze TICKER PERIOD - Quick analysis
/examples - Show ticker examples
/help - Detailed help
        """
        
        keyboard = [
            [InlineKeyboardButton("📈 Analyze Now", callback_data="ask_ticker")],
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
                "Use /examples for more tickers",
                parse_mode=ParseMode.HTML
            )
            return
        
        ticker = context.args[0].upper()
        period = context.args[1] if len(context.args) > 1 else '1y'
        
        await self._perform_analysis(update, context, ticker, period)
    
    async def _perform_analysis(self, update: Update, context: ContextTypes.DEFAULT_TYPE, 
                              ticker: str, period: str):
        """Perform analysis"""
        chat_id = update.effective_chat.id
        
        # Send typing indicator
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        
        # Status message
        status = await context.bot.send_message(
            chat_id=chat_id,
            text=f"🔄 Analyzing {ticker} ({period})...",
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
                
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=error_msg,
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
            
            # Action buttons
            keyboard = [
                [
                    InlineKeyboardButton("🔄 New Analysis", callback_data="ask_ticker"),
                    InlineKeyboardButton(f"📈 {ticker}", callback_data=f"analyze_{ticker}_{period}")
                ]
            ]
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            # Send chart if available
            if chart_path:
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
                    
                except Exception as e:
                    logger.error(f"Photo error: {e}")
                    # Send analysis without chart
                    await context.bot.send_message(
                        chat_id=chat_id,
                        text=analysis['summary'],
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
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=analysis['summary'],
                    reply_markup=reply_markup,
                    parse_mode=ParseMode.HTML
                )
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"❌ Analysis failed for {ticker}\n\n"
                     f"Error: {str(e)[:100]}",
                parse_mode=ParseMode.HTML
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
        
        examples_text += "**Usage:**\n/analyze AAPL 1y\n/analyze ENEL.MI 6m\n/analyze BTC-USD 3m"
        
        keyboard = [[InlineKeyboardButton("📈 Analyze Now", callback_data="ask_ticker")]]
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

**PERIODS (default: 1y):**
3m, 6m, 1y, 2y, 3y, 5y

**INDICATORS INCLUDED:**
• Price with 20, 50, 200 MAs (20 MA in green)
• Volume with A/D Line
• RSI (14)
• MACD (12, 26, 9)

**EXAMPLES:**
/analyze AAPL 1y
/analyze ENEL.MI 6m
/analyze BTC-USD 3m
/analyze ^GSPC 2y
        """
        
        keyboard = [[InlineKeyboardButton("📈 Start Analysis", callback_data="ask_ticker")]]
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
        
        if data == "ask_ticker":
            await query.edit_message_text(
                "📊 Enter a ticker symbol:\n\n"
                "Examples:\n"
                "• AAPL (Apple)\n"
                "• ENEL.MI (Enel Italy)\n"
                "• BTC-USD (Bitcoin)\n"
                "• ^GSPC (S&P 500)\n\n"
                "Or use: /analyze TICKER PERIOD",
                parse_mode=ParseMode.HTML
            )
        
        elif data == "help":
            await self._handle_help(update, context)
        
        elif data.startswith("examples_"):
            category = data.replace("examples_", "")
            if category in self.example_tickers:
                tickers = self.example_tickers[category]
                examples_text = f"📋 **{category} Examples:**\n\n"
                for ticker in tickers[:8]:
                    examples_text += f"• /analyze {ticker} 1y\n"
                
                await query.edit_message_text(
                    examples_text,
                    parse_mode=ParseMode.HTML
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
            await self._perform_analysis(update, context, text, '1y')
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