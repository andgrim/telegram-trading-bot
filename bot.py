import logging
import os
import asyncio
import traceback
from typing import Dict, List, Optional
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import (
    Application, 
    CommandHandler, 
    CallbackQueryHandler, 
    ContextTypes,
    MessageHandler,
    filters,
    ConversationHandler
)
from telegram.constants import ParseMode, ChatAction

from analyzer import TradingAnalyzer
from chart_generator import ChartGenerator
from config import CONFIG

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Conversation states
(
    MAIN_MENU,
    SELECT_TICKER,
    SELECT_PERIOD,
    SHOW_ANALYSIS,
    WATCHLIST,
    SETTINGS
) = range(6)

class TradingBot:
    """Telegram bot for technical analysis with extended timeframes"""
    
    def __init__(self):
        self.config = CONFIG
        self.analyzer = TradingAnalyzer()
        self.chart_generator = ChartGenerator()
        
        # User data storage
        self.user_data = {}
        self.watchlists = {}
        
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Start command - show main menu"""
        user_id = update.effective_user.id
        
        # Initialize user data
        self.user_data[user_id] = {
            'current_ticker': None,
            'current_period': None,
            'analysis_history': []
        }
        
        welcome_message = """
🎯 **Trading Analysis Bot - Extended Timeframes**

Welcome to the advanced trading analysis platform with support for:
• 3 Months • 6 Months • 1 Year
• **2 Years** • **3 Years** • **5 Years**

**Quick Actions:**
• Analyze any stock with detailed technical indicators
• View beautiful interactive charts across multiple timeframes
• Get real-time trading signals
• Detect reversal patterns

Tap buttons below to get started!
        """
        
        keyboard = [
            [InlineKeyboardButton("📈 Analyze Stock", callback_data="analyze")],
            [InlineKeyboardButton("⭐ My Watchlist", callback_data="watchlist")],
            [InlineKeyboardButton("⚙️ Settings", callback_data="settings")],
            [InlineKeyboardButton("📊 Market Overview", callback_data="market")],
            [InlineKeyboardButton("❓ Help", callback_data="help")]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        if update.message:
            await update.message.reply_text(
                welcome_message,
                reply_markup=reply_markup,
                parse_mode=ParseMode.MARKDOWN
            )
        else:
            await update.callback_query.edit_message_text(
                welcome_message,
                reply_markup=reply_markup,
                parse_mode=ParseMode.MARKDOWN
            )
        
        return MAIN_MENU
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show help message"""
        help_message = """
📖 **How to use this bot:**

**Main Features:**
• **Stock Analysis** - Technical analysis with 10+ indicators
• **Chart Generation** - Beautiful dark theme charts for all timeframes
• **Signal Detection** - Automatic buy/sell signals
• **Reversal Patterns** - Early trend reversal detection
• **Fundamental Analysis** - Company health scoring

**Available Timeframes:**
• **3 Months (3m)** - Short-term analysis
• **6 Months (6m)** - Medium-term analysis  
• **1 Year (1y)** - Long-term analysis
• **2 Years (2y)** - Extended trend analysis
• **3 Years (3y)** - Multi-year trend analysis
• **5 Years (5y)** - Full historical analysis

**Popular Tickers:**
AAPL, GOOGL, MSFT, TSLA, NVDA, AMZN, META, NFLX, AMD

**Commands:**
/start - Show main menu
/analyze - Quick analysis (e.g., /analyze AAPL 3m)
/watchlist - Manage your watchlist
/help - This help message

**Tips:**
• Use buttons for quick navigation
• You can input any valid ticker symbol
• Charts are generated with complete indicators
• All analysis includes full historical data
        """
        
        keyboard = [
            [InlineKeyboardButton("⬅️ Back to Menu", callback_data="back_to_menu")],
            [InlineKeyboardButton("📈 Start Analysis", callback_data="analyze")]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        if update.callback_query:
            await update.callback_query.edit_message_text(
                help_message,
                reply_markup=reply_markup,
                parse_mode=ParseMode.MARKDOWN
            )
        else:
            await update.message.reply_text(
                help_message,
                reply_markup=reply_markup,
                parse_mode=ParseMode.MARKDOWN
            )
        
        return MAIN_MENU
    
    async def analyze_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /analyze command with parameters"""
        if not context.args:
            await update.message.reply_text(
                "Please use the interactive menu or provide a ticker:\n"
                "Example: `/analyze AAPL 3m`\n\n"
                "Or use buttons below:",
                parse_mode=ParseMode.MARKDOWN
            )
            return await self.show_ticker_selection(update, context)
        
        ticker = context.args[0].upper()
        
        # Set default period if not provided
        if len(context.args) > 1:
            period = context.args[1].lower()
            period_map = {
                '3m': '3m', '3months': '3m', '3': '3m',
                '6m': '6m', '6months': '6m', '6': '6m',
                '1y': '1y', '1year': '1y', '1': '1y',
                '2y': '2y', '2years': '2y', '2': '2y',
                '3y': '3y', '3years': '3y', '3': '3y',
                '5y': '5y', '5years': '5y', '5': '5y'
            }
            period = period_map.get(period, '1y')
        else:
            period = '1y'
        
        # Store user data
        user_id = update.effective_user.id
        self.user_data[user_id]['current_ticker'] = ticker
        self.user_data[user_id]['current_period'] = period
        
        # Perform analysis
        await self.perform_analysis(update, context, ticker, period)
        
        return SHOW_ANALYSIS
    
    async def show_ticker_selection(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show ticker selection menu"""
        message = """
📈 **Select a Stock to Analyze**

Choose from popular stocks or enter your own ticker:

**Popular Stocks:**
• **AAPL** - Apple Inc.
• **GOOGL** - Alphabet (Google)
• **MSFT** - Microsoft
• **TSLA** - Tesla
• **NVDA** - NVIDIA
• **AMZN** - Amazon
• **META** - Meta Platforms
• **NFLX** - Netflix
• **AMD** - Advanced Micro Devices

**Or type any ticker symbol** (e.g., JPM, BAC, IBM, etc.)
        """
        
        # Create keyboard with popular tickers
        keyboard = []
        
        # First row: Popular tech stocks
        keyboard.append([
            InlineKeyboardButton("AAPL", callback_data="ticker_AAPL"),
            InlineKeyboardButton("GOOGL", callback_data="ticker_GOOGL"),
            InlineKeyboardButton("MSFT", callback_data="ticker_MSFT")
        ])
        
        # Second row
        keyboard.append([
            InlineKeyboardButton("TSLA", callback_data="ticker_TSLA"),
            InlineKeyboardButton("NVDA", callback_data="ticker_NVDA"),
            InlineKeyboardButton("AMZN", callback_data="ticker_AMZN")
        ])
        
        # Third row
        keyboard.append([
            InlineKeyboardButton("META", callback_data="ticker_META"),
            InlineKeyboardButton("NFLX", callback_data="ticker_NFLX"),
            InlineKeyboardButton("AMD", callback_data="ticker_AMD")
        ])
        
        # Fourth row: Custom and navigation
        keyboard.append([
            InlineKeyboardButton("🔍 Custom Ticker", callback_data="custom_ticker"),
            InlineKeyboardButton("⭐ Watchlist", callback_data="watchlist_tickers")
        ])
        
        keyboard.append([
            InlineKeyboardButton("⬅️ Back", callback_data="back_to_menu"),
            InlineKeyboardButton("❓ Help", callback_data="help")
        ])
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        if update.callback_query:
            await update.callback_query.edit_message_text(
                message,
                reply_markup=reply_markup,
                parse_mode=ParseMode.MARKDOWN
            )
        else:
            await update.message.reply_text(
                message,
                reply_markup=reply_markup,
                parse_mode=ParseMode.MARKDOWN
            )
        
        return SELECT_TICKER
    
    async def show_period_selection(self, update: Update, context: ContextTypes.DEFAULT_TYPE, ticker: str):
        """Show period selection menu with extended timeframes"""
        message = f"""
📊 **Select Analysis Period for {ticker}**

Choose the time period for analysis:

• **3 Months** - Short-term trends, recent performance
• **6 Months** - Medium-term analysis, clearer trends  
• **1 Year** - Long-term analysis, full picture
• **2 Years** - Extended trend analysis
• **3 Years** - Multi-year trend analysis
• **5 Years** - Full historical analysis

**Recommendation:**
- Day Trading: 3 Months
- Swing Trading: 6 Months  
- Investing: 1 Year or more
- Long-term Analysis: 2+ Years
        """
        
        keyboard = [
            [
                InlineKeyboardButton("3M", callback_data="period_3m"),
                InlineKeyboardButton("6M", callback_data="period_6m"),
                InlineKeyboardButton("1Y", callback_data="period_1y")
            ],
            [
                InlineKeyboardButton("2Y", callback_data="period_2y"),
                InlineKeyboardButton("3Y", callback_data="period_3y"),
                InlineKeyboardButton("5Y", callback_data="period_5y")
            ],
            [
                InlineKeyboardButton("⬅️ Back", callback_data="back_to_ticker"),
                InlineKeyboardButton("🏠 Menu", callback_data="back_to_menu")
            ]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        if update.callback_query:
            await update.callback_query.edit_message_text(
                message,
                reply_markup=reply_markup,
                parse_mode=ParseMode.MARKDOWN
            )
        else:
            await update.message.reply_text(
                message,
                reply_markup=reply_markup,
                parse_mode=ParseMode.MARKDOWN
            )
        
        # Store ticker in user data
        user_id = update.effective_user.id
        self.user_data[user_id]['current_ticker'] = ticker
        
        return SELECT_PERIOD
    
    async def perform_analysis(self, update: Update, context: ContextTypes.DEFAULT_TYPE, 
                             ticker: str, period: str):
        """Perform analysis and show results"""
        user_id = update.effective_user.id
        
        # Show typing indicator
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action=ChatAction.TYPING
        )
        
        # Send initial message
        status_message = await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=f"🔍 **Analyzing {ticker} ({period})...**\n\n"
                 f"• Fetching extended historical data...\n"
                 f"• Calculating complete indicators...\n"
                 f"• Generating high-resolution charts...\n\n"
                 f"*This may take 15-20 seconds for extended timeframes*",
            parse_mode=ParseMode.MARKDOWN
        )
        
        try:
            # Perform analysis
            analysis = await self.analyzer.analyze_ticker(ticker, period)
            
            if not analysis['success']:
                await status_message.edit_text(
                    f"❌ **Analysis Failed for {ticker}**\n\n"
                    f"Error: {analysis.get('error', 'Unknown error')}\n\n"
                    f"Please try another ticker or period.",
                    parse_mode=ParseMode.MARKDOWN
                )
                return await self.show_ticker_selection(update, context)
            
            # Generate chart
            chart_path = self.chart_generator.generate_price_chart(
                analysis['data'], 
                ticker, 
                period
            )
            
            # Create keyboard for analysis actions
            keyboard = [
                [
                    InlineKeyboardButton("📊 New Analysis", callback_data="analyze"),
                    InlineKeyboardButton("⭐ Add to Watchlist", callback_data=f"add_{ticker}")
                ],
                [
                    InlineKeyboardButton("📈 Same Ticker", callback_data=f"analyze_{ticker}"),
                    InlineKeyboardButton("🔄 Different Period", callback_data=f"period_{ticker}")
                ],
                [
                    InlineKeyboardButton("🏠 Main Menu", callback_data="back_to_menu"),
                    InlineKeyboardButton("📋 Full Report", callback_data=f"report_{ticker}")
                ]
            ]
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            # Send chart
            with open(chart_path, 'rb') as chart_file:
                await context.bot.send_photo(
                    chat_id=update.effective_chat.id,
                    photo=chart_file,
                    caption=analysis['summary'],
                    reply_markup=reply_markup,
                    parse_mode=ParseMode.MARKDOWN
                )
            
            # Send technical overview
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=analysis['technical_overview'],
                parse_mode=ParseMode.MARKDOWN
            )
            
            # Send extended timeframe analysis note for long periods
            if period in ['2y', '3y', '5y']:
                extended_note = f"""
📅 **Extended Timeframe Analysis ({period.upper()})**

**Key Insights for Long-term Analysis:**
• **Trend Identification**: Clearer long-term trend visualization
• **Support/Resistance**: Major price levels more evident
• **Cycle Analysis**: Identification of market cycles
• **Volatility Patterns**: Long-term volatility trends
• **Fundamental Correlation**: Price vs. company performance over years

*Extended timeframes provide better perspective for long-term investment decisions.*
                """
                
                await context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text=extended_note,
                    parse_mode=ParseMode.MARKDOWN
                )
            
            # Send signals if available
            if analysis['signals']:
                signals_text = "🎯 **TRADING SIGNALS**\n\n"
                bullish_count = len([s for s in analysis['signals'] if s['direction'] == 'BULLISH'])
                bearish_count = len([s for s in analysis['signals'] if s['direction'] == 'BEARISH'])
                
                signals_text += f"📈 Bullish: {bullish_count} | 📉 Bearish: {bearish_count}\n\n"
                
                for i, signal in enumerate(analysis['signals'][:5], 1):
                    emoji = "🟢" if signal['direction'] == 'BULLISH' else "🔴" if signal['direction'] == 'BEARISH' else "⚪"
                    strength_emoji = "🔥" if signal['strength'] == 'STRONG' else "⚡" if signal['strength'] == 'MODERATE' else "💡"
                    signals_text += f"{i}. {emoji} **{signal['type']}** {strength_emoji}\n"
                
                if len(analysis['signals']) > 5:
                    signals_text += f"\n*+{len(analysis['signals']) - 5} more signals...*"
                
                await context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text=signals_text,
                    parse_mode=ParseMode.MARKDOWN
                )
            
            # Add to history
            self.user_data[user_id]['analysis_history'].append({
                'ticker': ticker,
                'period': period,
                'timestamp': context.bot_data.get('timestamp', 'N/A')
            })
            
            # Clean up
            os.remove(chart_path)
            await status_message.delete()
            
        except Exception as e:
            logger.error(f"Error in analysis: {e}")
            await status_message.edit_text(
                f"❌ **Error analyzing {ticker}**\n\n"
                f"Please try again or select a different ticker.\n"
                f"Error: {str(e)[:100]}",
                parse_mode=ParseMode.MARKDOWN
            )
            
            keyboard = [
                [InlineKeyboardButton("⬅️ Try Again", callback_data="analyze")],
                [InlineKeyboardButton("🏠 Main Menu", callback_data="back_to_menu")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="What would you like to do?",
                reply_markup=reply_markup
            )
        
        return SHOW_ANALYSIS
    
    async def handle_text_input(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle text input from user (for custom tickers)"""
        user_input = update.message.text.strip().upper()
        user_id = update.effective_user.id
        
        # Check if input looks like a ticker
        if 1 <= len(user_input) <= 10 and user_input.isalpha():
            await update.message.reply_text(
                f"✅ Selected: **{user_input}**\n\n"
                f"Now select analysis timeframe:",
                parse_mode=ParseMode.MARKDOWN
            )
            return await self.show_period_selection(update, context, user_input)
        else:
            await update.message.reply_text(
                f"❌ **Invalid ticker format:** {user_input}\n\n"
                f"Tickers should be 1-10 letters (e.g., AAPL, TSLA)\n"
                f"Please try again or use the buttons below:",
                parse_mode=ParseMode.MARKDOWN
            )
            return await self.show_ticker_selection(update, context)
    
    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle callback queries from inline keyboards"""
        query = update.callback_query
        await query.answer()
        
        callback_data = query.data
        user_id = update.effective_user.id
        
        if callback_data == "analyze":
            return await self.show_ticker_selection(update, context)
        
        elif callback_data == "help":
            return await self.help_command(update, context)
        
        elif callback_data == "back_to_menu":
            return await self.start(update, context)
        
        elif callback_data == "back_to_ticker":
            return await self.show_ticker_selection(update, context)
        
        elif callback_data.startswith("ticker_"):
            ticker = callback_data.replace("ticker_", "")
            return await self.show_period_selection(update, context, ticker)
        
        elif callback_data == "custom_ticker":
            await query.edit_message_text(
                "🔍 **Enter Ticker Symbol**\n\n"
                "Please type the stock ticker symbol you want to analyze.\n"
                "Examples: AAPL, TSLA, GOOGL, MSFT, etc.\n\n"
                "*Note: Use uppercase letters without spaces*",
                parse_mode=ParseMode.MARKDOWN
            )
            return SELECT_TICKER
        
        elif callback_data.startswith("period_"):
            period_ticker = callback_data.replace("period_", "")
            
            # Check if this is just period selection for current ticker
            if period_ticker in ['3m', '6m', '1y', '2y', '3y', '5y']:
                period = period_ticker
                ticker = self.user_data[user_id].get('current_ticker')
                if not ticker:
                    await query.edit_message_text(
                        "Please select a ticker first.",
                        parse_mode=ParseMode.MARKDOWN
                    )
                    return await self.show_ticker_selection(update, context)
            else:
                # This is for different period button
                ticker = period_ticker
                await query.edit_message_text(
                    f"Select period for **{ticker}**:",
                    parse_mode=ParseMode.MARKDOWN
                )
                return await self.show_period_selection(update, context, ticker)
            
            # Perform analysis
            self.user_data[user_id]['current_period'] = period
            return await self.perform_analysis(update, context, ticker, period)
        
        elif callback_data.startswith("analyze_"):
            ticker = callback_data.replace("analyze_", "")
            return await self.show_period_selection(update, context, ticker)
        
        elif callback_data.startswith("add_"):
            ticker = callback_data.replace("add_", "")
            # Initialize watchlist if not exists
            if user_id not in self.watchlists:
                self.watchlists[user_id] = []
            
            if ticker not in self.watchlists[user_id]:
                self.watchlists[user_id].append(ticker)
                await query.edit_message_text(
                    f"✅ **{ticker} added to your watchlist!**\n\n"
                    f"*You can view your watchlist from the main menu.*",
                    parse_mode=ParseMode.MARKDOWN
                )
            else:
                await query.edit_message_text(
                    f"ℹ️ **{ticker} is already in your watchlist.**",
                    parse_mode=ParseMode.MARKDOWN
                )
            
            keyboard = [
                [InlineKeyboardButton("📈 New Analysis", callback_data="analyze")],
                [InlineKeyboardButton("⭐ View Watchlist", callback_data="watchlist")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="What would you like to do next?",
                reply_markup=reply_markup
            )
            
            return SHOW_ANALYSIS
        
        elif callback_data == "watchlist":
            return await self.show_watchlist(update, context)
        
        elif callback_data == "watchlist_tickers":
            return await self.show_watchlist(update, context)
        
        elif callback_data == "market":
            await query.edit_message_text(
                "📊 **Market Overview**\n\n"
                "*This feature is coming soon!*\n\n"
                "Future updates will include:\n"
                "• Major indices (S&P 500, NASDAQ)\n"
                "• Sector performance\n"
                "• Market sentiment indicators\n"
                "• Top gainers/losers",
                parse_mode=ParseMode.MARKDOWN
            )
            return await self.start(update, context)
        
        elif callback_data == "settings":
            await query.edit_message_text(
                "⚙️ **Settings**\n\n"
                "*Settings menu coming soon!*\n\n"
                "Future settings will include:\n"
                "• Chart theme preferences\n"
                "• Indicator preferences\n"
                "• Alert settings\n"
                "• Notification preferences",
                parse_mode=ParseMode.MARKDOWN
            )
            return await self.start(update, context)
        
        elif callback_data.startswith("report_"):
            ticker = callback_data.replace("report_", "")
            await query.edit_message_text(
                f"📋 **Full Report for {ticker}**\n\n"
                "*Full report feature coming soon!*\n\n"
                "Future reports will include:\n"
                "• Detailed financial metrics\n"
                "• Technical analysis breakdown\n"
                "• Risk assessment\n"
                "• Investment recommendation",
                parse_mode=ParseMode.MARKDOWN
            )
            return await self.start(update, context)
    
    async def show_watchlist(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show user's watchlist"""
        user_id = update.effective_user.id
        
        if user_id not in self.watchlists or not self.watchlists[user_id]:
            message = "⭐ **Your Watchlist is Empty**\n\nAdd stocks to your watchlist during analysis!"
            keyboard = [[InlineKeyboardButton("📈 Browse Stocks", callback_data="analyze")]]
        else:
            message = "⭐ **Your Watchlist**\n\n"
            keyboard = []
            
            for i, ticker in enumerate(self.watchlists[user_id], 1):
                message += f"{i}. **{ticker}**\n"
                if i % 2 == 1:
                    keyboard.append([
                        InlineKeyboardButton(f"📊 {ticker}", callback_data=f"analyze_{ticker}")
                    ])
                else:
                    keyboard[-1].append(InlineKeyboardButton(f"📊 {ticker}", callback_data=f"analyze_{ticker}"))
            
            message += f"\n*{len(self.watchlists[user_id])} stocks in watchlist*"
        
        keyboard.append([
            InlineKeyboardButton("⬅️ Back", callback_data="analyze"),
            InlineKeyboardButton("🏠 Menu", callback_data="back_to_menu")
        ])
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        if update.callback_query:
            await update.callback_query.edit_message_text(
                message,
                reply_markup=reply_markup,
                parse_mode=ParseMode.MARKDOWN
            )
        else:
            await update.message.reply_text(
                message,
                reply_markup=reply_markup,
                parse_mode=ParseMode.MARKDOWN
            )
        
        return WATCHLIST
    
    async def cancel(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Cancel conversation and return to main menu"""
        await update.message.reply_text(
            "Operation cancelled. Returning to main menu...",
            parse_mode=ParseMode.MARKDOWN
        )
        return await self.start(update, context)
    
    def run(self):
        """Run the bot with enhanced conversation handling"""
        if not self.config.TELEGRAM_TOKEN:
            logger.error("TELEGRAM_TOKEN not found in environment variables")
            return
        
        application = Application.builder().token(self.config.TELEGRAM_TOKEN).build()
        
        # Create conversation handler
        conv_handler = ConversationHandler(
            entry_points=[
                CommandHandler("start", self.start),
                CommandHandler("help", self.help_command),
                CommandHandler("analyze", self.analyze_command),
                CommandHandler("watchlist", self.show_watchlist),
                CallbackQueryHandler(self.handle_callback)
            ],
            states={
                MAIN_MENU: [
                    CallbackQueryHandler(self.handle_callback)
                ],
                SELECT_TICKER: [
                    CallbackQueryHandler(self.handle_callback),
                    MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text_input)
                ],
                SELECT_PERIOD: [
                    CallbackQueryHandler(self.handle_callback)
                ],
                SHOW_ANALYSIS: [
                    CallbackQueryHandler(self.handle_callback)
                ],
                WATCHLIST: [
                    CallbackQueryHandler(self.handle_callback)
                ],
                SETTINGS: [
                    CallbackQueryHandler(self.handle_callback)
                ]
            },
            fallbacks=[
                CommandHandler("start", self.start),
                CommandHandler("cancel", self.cancel)
            ],
            allow_reentry=True
        )
        
        application.add_handler(conv_handler)
        
        # Start the bot
        logger.info("Enhanced Trading Bot with Extended Timeframes starting...")
        application.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)