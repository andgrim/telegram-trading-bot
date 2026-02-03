"""
Universal Trading Bot for Telegram - Webhook Version
Optimized for Render deployment - FIXED VERSION
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
logger = logging.getLogger(__name__)

class UniversalTradingBot:
    """Universal Telegram bot for ALL markets - Webhook version FIXED"""
    
    def __init__(self):
        self.config = CONFIG
        self.analyzer = TradingAnalyzer()
        self.chart_generator = ChartGenerator()
        
        # Telegram application - will be created in initialize()
        self.application = None
        self.initialized = False
        
        # Example tickers from different markets
        self.example_tickers = {
            'US Stocks': ['AAPL', 'MSFT', 'TSLA', 'AMZN', 'GOOGL', 'NVDA', 'META'],
            'European Stocks': ['ENEL.MI', 'AIR.PA', 'SAP.DE', 'ASML.AS', 'HSBA.L'],
            'ETFs & Funds': ['SPY', 'QQQ', 'VOO', 'GLD', 'SLV', 'BND'],
            'Commodities': ['GC=F', 'CL=F', 'SI=F', 'NG=F', 'HG=F'],
            'Cryptocurrencies': ['BTC-USD', 'ETH-USD', 'XRP-USD', 'SOL-USD'],
            'Indices': ['^GSPC', '^DJI', '^IXIC', '^FTSE', '^GDAXI'],
        }
    
    def initialize(self) -> bool:
        """Initialize the bot application - SYNC version for Flask"""
        if self.initialized:
            return True
        
        try:
            token = self.config.TELEGRAM_TOKEN
            if not token:
                logger.error("TELEGRAM_TOKEN not found in environment")
                return False
            
            logger.info("Initializing bot application...")
            
            # Create application on current event loop
            self.application = Application.builder().token(token).build()
            
            # Add handlers
            self._setup_handlers()
            
            # Initialize the application
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.application.initialize())
            
            # Start the application (but don't run polling)
            loop.run_until_complete(self.application.start())
            
            self.initialized = True
            logger.info("✅ Bot application initialized successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize bot: {e}")
            import traceback
            traceback.print_exc()
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
    
    async def process_update(self, update_data: Dict[str, Any]):
        """Process incoming webhook update - ASYNC version"""
        if not self.application or not self.initialized:
            logger.error("Bot not initialized")
            return
        
        try:
            # Create update object
            update = Update.de_json(update_data, self.application.bot)
            
            # Process update through the application
            await self.application.process_update(update)
            
        except Exception as e:
            logger.error(f"Error processing webhook update: {e}")
    
    def process_webhook_update(self, update_data: Dict[str, Any]):
        """Process webhook update in a thread (for Flask)"""
        try:
            # Run async function in new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.process_update(update_data))
            loop.close()
        except Exception as e:
            logger.error(f"Error in webhook processing thread: {e}")
    
    # Command Handlers - all methods below are async
    async def _handle_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        # Check if we have a chat id
        if update.effective_chat is None:
            logger.error("No chat id in /start update")
            return
        
        welcome = """
🤖 **Universal Trading Analysis Bot**

🌍 **Supports ALL markets worldwide:**
• US Stocks (AAPL, MSFT, TSLA)
• European Stocks (ENEL.MI, AIR.PA, SAP.DE)
• ETFs & Funds (SPY, QQQ, VOO)
• Commodities (GOLD, OIL, SILVER)
• Cryptocurrencies (BTC, ETH, XRP)
• Indices (SPX, DJI, FTSE, DAX)

📊 **Complete Technical Analysis:**
• 25+ indicators (RSI, MACD, Stochastic, BB, ATR)
• Divergence detection
• Reversal pattern recognition
• Performance statistics

**Commands:**
/start - This menu
/analyze TICKER PERIOD - Quick analysis
/examples - Show ticker examples
/help - Detailed help
/test - Test command (no Yahoo Finance)
        """
        
        keyboard = [
            [InlineKeyboardButton("📈 Analyze Now", callback_data="ask_ticker")],
            [
                InlineKeyboardButton("🇺🇸 US", callback_data="examples_US Stocks"),
                InlineKeyboardButton("🇪🇺 Europe", callback_data="examples_European Stocks")
            ],
            [InlineKeyboardButton("❓ Help", callback_data="help")]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            welcome,
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML
        )
    
    async def _test_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /test command - simple test without yfinance"""
        # Check if we have a chat id
        if update.effective_chat is None:
            logger.error("No chat id in /test update")
            return
        
        await update.message.reply_text(
            "✅ Bot is working!\n\n"
            "Test tickers:\n"
            "• AAPL (might be blocked)\n"
            "• Try European: ENEL.MI\n"
            "• Wait 10 seconds between requests",
            parse_mode=ParseMode.HTML
        )
    
    async def _handle_analyze(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /analyze command"""
        # CRITICAL: Check if we have a valid chat id
        if update.effective_chat is None:
            logger.error("No chat id in /analyze update")
            return
        
        if not context.args:
            await update.message.reply_text(
                "Usage: /analyze TICKER PERIOD\n\n"
                "Examples:\n"
                "• /analyze AAPL 1y (Apple)\n"
                "• /analyze ENEL.MI 6m (Italian stock)\n"
                "• /analyze GOLD 3m (Gold)\n"
                "• /analyze BTC-USD 1y (Bitcoin)\n"
                "• /analyze ^GSPC 2y (S&P 500)\n\n"
                "Need help? Use /examples or /help",
                parse_mode=ParseMode.HTML
            )
            return
        
        ticker = context.args[0].upper()
        period = context.args[1] if len(context.args) > 1 else '1y'
        
        await self._perform_analysis(update, context, ticker, period)
    
    async def _perform_analysis(self, update: Update, context: ContextTypes.DEFAULT_TYPE, 
                              ticker: str, period: str):
        """Perform analysis"""
        # CRITICAL: Check if we have a valid chat id
        if update.effective_chat is None:
            logger.error(f"No chat id for analysis of {ticker}")
            return
        
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
                error_msg += "Possible solutions:\n"
                error_msg += "1. Check if ticker exists on Yahoo Finance\n"
                error_msg += "2. Try a different period (1y, 6m)\n"
                error_msg += "3. Wait 60 seconds and try again\n\n"
                error_msg += "Examples:\n"
                error_msg += "• AAPL, MSFT, TSLA (US)\n"
                error_msg += "• ENEL.MI, AIR.PA (Europe)\n"
                error_msg += "• SPY, GOLD, BTC-USD\n"
                
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
                        import os
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
                     f"Please try again or use a different ticker.",
                parse_mode=ParseMode.HTML
            )
        finally:
            # Delete status
            try:
                await status.delete()
            except:
                pass
    
    # ... (rest of the methods remain the same, but add chat id checks to each) ...

async def _handle_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help command"""
    # Check if we have a chat id
    if update.effective_chat is None:
        logger.error("No chat id in /help update")
        return
    
    help_text = """
📖 **Universal Trading Bot Help**

**SUPPORTED MARKETS:**
• US Stocks: AAPL, MSFT, TSLA, GOOGL, AMZN
• European Stocks: Use exchange suffix:
  - Italy: .MI (ENEL.MI, ISP.MI)
  - France: .PA (AIR.PA, TTE.PA)
  - Germany: .DE (SAP.DE, BMW.DE)
• ETFs: SPY, QQQ, VOO, GLD, SLV
• Commodities: GOLD (GC=F), OIL (CL=F), SILVER (SI=F)
• Cryptocurrencies: BTC-USD, ETH-USD, XRP-USD
• Indices: SPX (^GSPC), DJI (^DJI), FTSE (^FTSE)

**PERIODS:**
3m, 6m, 1y, 2y, 3y, 5y

**EXAMPLES:**
/analyze AAPL 1y
/analyze ENEL.MI 6m
/analyze GOLD 3m
/analyze BTC-USD 1y
/analyze ^GSPC 2y
    """
    
    keyboard = [[InlineKeyboardButton("📈 Start Analysis", callback_data="ask_ticker")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        help_text,
        reply_markup=reply_markup,
        parse_mode=ParseMode.HTML
    )