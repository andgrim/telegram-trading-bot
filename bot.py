"""
Universal Trading Bot for Telegram - Enhanced Version
Complete analysis with advanced correlations, divergences, and ticker search
"""
import logging
import os
import asyncio
import html
import re
from typing import Dict, Any
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler,
    MessageHandler, filters, ContextTypes
)
from telegram.constants import ParseMode, ChatAction

from analyzer import TradingAnalyzer
from chart_generator import ChartGenerator
from ticker_search import TickerSearch
from config import CONFIG

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class UniversalTradingBot:
    """Enhanced Universal Telegram bot with advanced features"""
    
    def __init__(self):
        self.config = CONFIG
        self.analyzer = TradingAnalyzer()
        self.chart_generator = ChartGenerator()
        self.ticker_search = TickerSearch()
        
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
        
        # User sessions for search
        self.user_searches = {}
        
        # Common searches cache
        self.common_searches = {
            'Popular ETFs': ['SPY', 'QQQ', 'VOO', 'GLD', 'IWM', 'EEM', 'TLT', 'HYG'],
            'Tech Giants': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA'],
            'Dividend Stocks': ['JNJ', 'PG', 'KO', 'PEP', 'T', 'VZ', 'XOM'],
            'Crypto': ['BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD', 'ADA-USD'],
            'Indices': ['^GSPC', '^DJI', '^IXIC', '^RUT', '^VIX'],
            'European Stocks': ['ENEL.MI', 'AIR.PA', 'SAP.DE', 'ASML.AS', 'HSBA.L'],
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
            
            logger.info("Initializing enhanced bot application...")
            
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
        self.application.add_handler(CommandHandler("search", self._handle_search))
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
        text = re.sub(r'<[^>]+>', '', text)
        
        # Ensure text is not too long for Telegram
        if len(text) > 4096:
            text = text[:4000] + "\n... [truncated]"
        
        return text
    
    def _get_main_keyboard(self) -> InlineKeyboardMarkup:
        """Create main menu keyboard"""
        keyboard = [
            [InlineKeyboardButton("🔍 Search Ticker", callback_data="new_search")],
            [InlineKeyboardButton("📊 Quick Analysis", callback_data="quick_analysis")],
            [InlineKeyboardButton("📋 Examples", callback_data="show_examples")],
            [InlineKeyboardButton("❓ Help", callback_data="help")]
        ]
        return InlineKeyboardMarkup(keyboard)
    
    def _get_period_keyboard(self, ticker: str) -> InlineKeyboardMarkup:
        """Create keyboard with period buttons for a ticker"""
        keyboard = []
        
        # Create rows of buttons
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
        
        # Add back button
        keyboard.append([InlineKeyboardButton("⬅️ Back to Search", callback_data="new_search")])
        
        return InlineKeyboardMarkup(keyboard)
    
    def _get_analysis_keyboard(self, ticker: str, period: str) -> InlineKeyboardMarkup:
        """Create keyboard after analysis"""
        keyboard = [
            [
                InlineKeyboardButton("🔄 Change Period", callback_data=f"change_period_{ticker}"),
                InlineKeyboardButton("🔍 New Search", callback_data="new_search")
            ],
            [
                InlineKeyboardButton("3M", callback_data=f"analyze_{ticker}_3m"),
                InlineKeyboardButton("6M", callback_data=f"analyze_{ticker}_6m"),
                InlineKeyboardButton("1Y", callback_data=f"analyze_{ticker}_1y"),
            ],
            [
                InlineKeyboardButton("2Y", callback_data=f"analyze_{ticker}_2y"),
                InlineKeyboardButton("5Y", callback_data=f"analyze_{ticker}_5y"),
                InlineKeyboardButton("📊 New", callback_data="quick_analysis"),
            ]
        ]
        return InlineKeyboardMarkup(keyboard)
    
    def _get_search_results_keyboard(self, search_results: list) -> InlineKeyboardMarkup:
        """Create keyboard for search results"""
        keyboard = []
        
        # Add results (max 6)
        for result in search_results[:6]:
            symbol = result['symbol']
            name_short = result['name'][:20] + '...' if len(result['name']) > 20 else result['name']
            button_text = f"{symbol} - {name_short}"
            keyboard.append([InlineKeyboardButton(button_text, callback_data=f"ticker_{symbol}")])
        
        # Add navigation buttons
        keyboard.append([
            InlineKeyboardButton("🔍 New Search", callback_data="new_search"),
            InlineKeyboardButton("📋 Examples", callback_data="show_examples")
        ])
        
        return InlineKeyboardMarkup(keyboard)
    
    def _get_quick_analysis_keyboard(self) -> InlineKeyboardMarkup:
        """Create keyboard for quick analysis options"""
        keyboard = []
        
        # Add popular categories
        for category, tickers in self.common_searches.items():
            row = []
            for ticker in tickers[:2]:  # Show first 2 from each category
                row.append(InlineKeyboardButton(ticker, callback_data=f"ticker_{ticker}"))
            if row:
                keyboard.append(row)
        
        # Add search button
        keyboard.append([InlineKeyboardButton("🔍 Search More...", callback_data="new_search")])
        
        return InlineKeyboardMarkup(keyboard)
    
    # Command Handlers
    async def _handle_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        welcome = """
🤖 **Enhanced Universal Trading Analysis Bot**

🔗 **ADVANCED FEATURES:**
• Complete technical analysis with 30+ indicators
• Advanced correlations & divergences
• A/D Line, OBV, MFI volume analysis
• Multi-timeframe analysis
• Bollinger Bands squeeze detection

🔍 **TICKER SEARCH:**
Search by company name, ETF name, or description
Examples: "Apple", "NASDAQ ETF", "S&P 500", "Gold ETF"

📊 **SUPPORTS ALL MARKETS:**
• US & International Stocks
• ETFs & Mutual Funds
• Cryptocurrencies
• Indices & Commodities

**Quick Actions Below ↓**
        """
        
        await update.message.reply_text(
            welcome,
            reply_markup=self._get_main_keyboard(),
            parse_mode=ParseMode.HTML
        )
    
    async def _test_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /test command"""
        await update.message.reply_text(
            "✅ Enhanced bot is working!\n\n"
            "Try these commands:\n"
            "/search Apple\n"
            "/search NASDAQ ETF\n"
            "/analyze AAPL 1y\n"
            "/examples\n\n"
            "Or click buttons below:",
            reply_markup=self._get_main_keyboard(),
            parse_mode=ParseMode.HTML
        )
    
    async def _handle_search(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /search command"""
        if not context.args:
            await update.message.reply_text(
                "🔍 **Ticker Search**\n\n"
                "Search by company name, ETF, index, or description.\n\n"
                "**Examples:**\n"
                "/search Apple\n"
                "/search NASDAQ ETF\n"
                "/search S&P 500\n"
                "/search Gold ETF\n"
                "/search Bitcoin\n\n"
                "Or type your search query:",
                parse_mode=ParseMode.HTML
            )
            return
        
        query = ' '.join(context.args)
        await self._perform_search(update, context, query)
    
    async def _perform_search(self, update: Update, context: ContextTypes.DEFAULT_TYPE, query: str):
        """Perform ticker search and display results"""
        chat_id = update.effective_chat.id
        
        # Send typing indicator
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        
        # Status message
        status_msg = await context.bot.send_message(
            chat_id=chat_id,
            text=f"🔍 Searching for '{query}'...",
            parse_mode=ParseMode.HTML
        )
        
        try:
            # Special handling for common queries
            query_lower = query.lower()
            
            # Check for ETF queries
            if any(term in query_lower for term in ['etf', 'fund', 'nasdaq', 'sp500', 's&p', 'dow', 'russell']):
                etf_symbol = self.ticker_search.search_etf_by_name(query)
                if etf_symbol:
                    await status_msg.delete()
                    await self._show_period_selection(update, context, etf_symbol, f"Found ETF: {etf_symbol}")
                    return
            
            # Check for index queries
            if any(term in query_lower for term in ['index', 'indices', 'spx', 'dji', 'comp']):
                index_symbol = self.ticker_search.search_index_by_name(query)
                if index_symbol:
                    await status_msg.delete()
                    await self._show_period_selection(update, context, index_symbol, f"Found Index: {index_symbol}")
                    return
            
            # General search
            results = self.ticker_search.search_ticker(query, max_results=10)
            
            if not results:
                await status_msg.edit_text(
                    f"❌ No results found for '{query}'\n\n"
                    "Try:\n"
                    "• Different search terms\n"
                    "• Company full name\n"
                    "• Ticker symbol if known\n\n"
                    "Or use /examples for popular tickers.",
                    parse_mode=ParseMode.HTML
                )
                return
            
            # Store results in user session
            self.user_searches[chat_id] = {
                'results': results,
                'query': query
            }
            
            # Format and display results
            formatted_results = self.ticker_search.format_search_results(results)
            
            await status_msg.edit_text(
                formatted_results,
                reply_markup=self._get_search_results_keyboard(results),
                parse_mode=ParseMode.HTML
            )
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            await status_msg.edit_text(
                f"❌ Search failed for '{query}'\n\n"
                "Error: Service temporarily unavailable.\n"
                "Try again in a moment or use direct ticker symbols.",
                parse_mode=ParseMode.HTML
            )
    
    async def _show_period_selection(self, update: Update, context: ContextTypes.DEFAULT_TYPE, 
                               ticker: str, message: str = None):
        """Show period selection for a ticker"""
        if not message:
            message = f"📊 Select analysis period for **{ticker}**:"
        
        # Check if we're in a callback query context
        if update.callback_query is not None:
            # Edit existing message (from callback query)
            await update.callback_query.edit_message_text(
                message,
                reply_markup=self._get_period_keyboard(ticker),
                parse_mode=ParseMode.HTML
            )
        elif hasattr(update, 'message') and update.message:
            # Send new message (from regular text message)
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=message,
                reply_markup=self._get_period_keyboard(ticker),
                parse_mode=ParseMode.HTML
            )
        else:
            # Fallback (shouldn't happen)
            logger.warning(f"Could not determine message context for ticker: {ticker}")
        
    async def _handle_analyze(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /analyze command"""
        if not context.args:
            await update.message.reply_text(
                "📊 **Quick Analysis**\n\n"
                "Usage: /analyze TICKER PERIOD\n\n"
                "**Examples:**\n"
                "/analyze AAPL 1y\n"
                "/analyze ENEL.MI 6m\n"
                "/analyze BTC-USD 3m\n"
                "/analyze ^GSPC 2y\n\n"
                "**Periods:** 3m, 6m, 1y, 2y, 5y\n\n"
                "Use /search to find tickers by name.",
                parse_mode=ParseMode.HTML
            )
            return
        
        ticker = context.args[0].upper()
        
        # Validate period
        period = '1y'  # Default
        if len(context.args) > 1:
            user_period = context.args[1].lower()
            if user_period in self.periods:
                period = user_period
            else:
                await update.message.reply_text(
                    f"⚠️ Invalid period: {user_period}\n\n"
                    "Available: 3m, 6m, 1y, 2y, 5y\n"
                    "Using default: 1 year",
                    parse_mode=ParseMode.HTML
                )
        
        await self._perform_enhanced_analysis(update, context, ticker, period)
    
    async def _perform_enhanced_analysis(self, update: Update, context: ContextTypes.DEFAULT_TYPE, 
                                       ticker: str, period: str):
        """Perform enhanced analysis with advanced correlations"""
        chat_id = update.effective_chat.id
        
        # Send typing indicator
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        
        # Status message
        status_msg = await context.bot.send_message(
            chat_id=chat_id,
            text=f"📊 Analyzing {ticker} ({self.periods.get(period, period)})...",
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
                error_msg += "• Use /search to find valid tickers\n\n"
                error_msg += "Use /examples for working tickers."
                
                await status_msg.edit_text(
                    error_msg,
                    reply_markup=self._get_main_keyboard(),
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
            
            # Prepare summary
            compact_summary = analysis.get('compact_summary', '')
            full_summary = analysis.get('summary', '')
            
            # Clean text
            compact_summary_clean = self._clean_telegram_text(compact_summary)
            full_summary_clean = self._clean_telegram_text(full_summary)
            
            # Create keyboard
            reply_markup = self._get_analysis_keyboard(ticker, period)
            
            # Send results
            if chart_path and os.path.exists(chart_path):
                try:
                    with open(chart_path, 'rb') as f:
                        await context.bot.send_photo(
                            chat_id=chat_id,
                            photo=f,
                            caption=compact_summary_clean[:1024],
                            reply_markup=reply_markup
                        )
                    
                    # Send full analysis if needed
                    if len(full_summary_clean) > 0:
                        await context.bot.send_message(
                            chat_id=chat_id,
                            text=full_summary_clean,
                            parse_mode=None
                        )
                    
                except Exception as e:
                    logger.error(f"Photo error: {e}")
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
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=full_summary_clean,
                    reply_markup=reply_markup,
                    parse_mode=None
                )
            
            await status_msg.delete()
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            await status_msg.edit_text(
                f"❌ Analysis failed for {ticker}\n\n"
                f"Error: {str(e)[:200]}\n\n"
                "Try a different ticker or period.",
                parse_mode=ParseMode.HTML
            )
    
    async def _handle_examples(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /examples command"""
        examples_text = """
📋 **Popular Tickers by Category:**

🔵 **US Stocks:**
• AAPL (Apple), MSFT (Microsoft), GOOGL (Google)
• AMZN (Amazon), META (Facebook), NVDA (Nvidia)
• TSLA (Tesla), JPM (JPMorgan), JNJ (Johnson & Johnson)

📈 **ETFs & Funds:**
• SPY (S&P 500 ETF), QQQ (NASDAQ 100 ETF)
• VOO (Vanguard S&P 500), GLD (Gold ETF)
• IWM (Russell 2000), EEM (Emerging Markets)

🇪🇺 **European Stocks:**
• ENEL.MI (Enel Italy), AIR.PA (Airbus France)
• SAP.DE (SAP Germany), ASML.AS (ASML Netherlands)
• HSBA.L (HSBC UK), BMW.DE (BMW Germany)

₿ **Cryptocurrencies:**
• BTC-USD (Bitcoin), ETH-USD (Ethereum)
• SOL-USD (Solana), XRP-USD (Ripple)

📊 **Indices:**
• ^GSPC (S&P 500), ^DJI (Dow Jones)
• ^IXIC (NASDAQ), ^RUT (Russell 2000)

**Click a ticker below or use /search to find more:**
"""
        
        await update.message.reply_text(
            examples_text,
            reply_markup=self._get_quick_analysis_keyboard(),
            parse_mode=ParseMode.HTML
        )
    
    async def _handle_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        help_text = """
📖 **Enhanced Trading Bot Help**

🔍 **SEARCH FUNCTIONALITY:**
• Use /search COMPANY_NAME to find tickers
• Examples: /search Apple, /search NASDAQ ETF
• Or simply type company name in chat

📊 **ANALYSIS FEATURES:**
• Complete technical analysis with 30+ indicators
• Advanced volume correlations (A/D, OBV, MFI)
• Multi-timeframe analysis
• Bollinger Bands squeeze detection
• Divergence and pattern recognition

⏰ **PERIOD OPTIONS:**
• 3 Months (3m)
• 6 Months (6m)
• 1 Year (1y) - Default
• 2 Years (2y)
• 5 Years (5y)

🌍 **SUPPORTED MARKETS:**
• US & International Stocks (with exchange suffixes)
• ETFs, Mutual Funds, Index Funds
• Cryptocurrencies (BTC-USD, ETH-USD, etc.)
• Indices (^GSPC, ^DJI, ^IXIC)

🎯 **QUICK START:**
1. Use /search or type company name
2. Select ticker from results
3. Choose analysis period
4. View complete analysis

**Need help?** Use /examples for working tickers.
"""
        
        await update.message.reply_text(
            help_text,
            reply_markup=self._get_main_keyboard(),
            parse_mode=ParseMode.HTML
        )
    
    async def _handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle callback queries"""
        query = update.callback_query
        await query.answer()
        
        data = query.data
        
        if data == "new_search":
            await query.edit_message_text(
                "🔍 **Ticker Search**\n\n"
                "Enter company name, ETF, or description:\n\n"
                "**Examples:**\n"
                "• Apple\n"
                "• NASDAQ ETF\n"
                "• S&P 500\n"
                "• Gold ETF\n"
                "• Microsoft\n\n"
                "Or click below for quick options:",
                reply_markup=self._get_quick_analysis_keyboard(),
                parse_mode=ParseMode.HTML
            )
        
        elif data == "quick_analysis":
            await query.edit_message_text(
                "📊 **Quick Analysis**\n\n"
                "Select a popular ticker or use search:\n",
                reply_markup=self._get_quick_analysis_keyboard(),
                parse_mode=ParseMode.HTML
            )
        
        elif data == "show_examples":
            await self._handle_examples(update, context)
        
        elif data == "help":
            await self._handle_help(update, context)
        
        elif data.startswith("ticker_"):
            ticker = data.replace("ticker_", "")
            await self._show_period_selection(update, context, ticker)
        
        elif data.startswith("change_period_"):
            ticker = data.replace("change_period_", "")
            await self._show_period_selection(update, context, ticker, f"Change period for {ticker}:")
        
        elif data.startswith("analyze_"):
            parts = data.split("_")
            if len(parts) >= 3:
                ticker = parts[1]
                period = parts[2]
                await self._perform_enhanced_analysis(update, context, ticker, period)
    
    async def _handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle text messages (ticker input or search query)"""
        text = update.message.text.strip()
        
        # Check if it looks like a ticker symbol
        ticker_pattern = r'^[A-Z0-9\^\.\-\=]{1,15}$'
        if re.match(ticker_pattern, text.upper()):
            # Looks like a ticker symbol
            await self._show_period_selection(update, context, text.upper())
        else:
            # Treat as search query
            await self._perform_search(update, context, text)
    
    def run(self):
        """Run the bot with polling"""
        if not self.initialized:
            if not self.initialize():
                logger.error("Failed to initialize enhanced bot. Exiting.")
                return
        
        logger.info("Starting enhanced bot polling...")
        print("\n" + "="*60)
        print("🤖 ENHANCED UNIVERSAL TRADING BOT")
        print("🔗 Features: Advanced Correlations & Ticker Search")
        print("🌍 Supports: All markets worldwide")
        print("📊 Includes: 30+ indicators, volume analysis")
        print("🔍 New: Search by company name, ETF, index")
        print("="*60 + "\n")
        print("Commands:")
        print("• /start - Show main menu")
        print("• /search - Find ticker by name")
        print("• /analyze - Quick analysis")
        print("• /examples - Popular tickers")
        print("• /help - Detailed help\n")
        
        self.application.run_polling(allowed_updates=Update.ALL_TYPES)