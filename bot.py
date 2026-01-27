import logging
import os
import asyncio
import traceback
from typing import Dict, List
from datetime import datetime
import telegram
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, 
    CommandHandler, 
    CallbackQueryHandler, 
    ContextTypes,
    MessageHandler,
    filters
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

class TradingBot:
    """Simple Telegram bot for technical analysis"""
    
    def __init__(self):
        self.config = CONFIG
        self.analyzer = TradingAnalyzer()
        self.chart_generator = ChartGenerator()
    
    async def error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle errors in the bot"""
        logger.error(msg="Exception while handling an update:", exc_info=context.error)
        
        try:
            if isinstance(update, Update) and update.effective_chat:
                error_message = "❌ An error occurred. Please try again or use /start"
                await context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text=error_message,
                    parse_mode=ParseMode.MARKDOWN
                )
        except Exception as e:
            logger.error(f"Error sending error message: {e}")
    
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Start command - show simple menu"""
        welcome_message = """
📊 **Stock Analysis Bot**

Welcome! I can analyze any stock with technical indicators.

**How to use:**
1. Send me a stock ticker (e.g., AAPL, TSLA)
2. I'll ask for timeframe
3. You'll get analysis + chart

**Available timeframes:**
• 3m (3 months)
• 6m (6 months)  
• 1y (1 year)
• 2y (2 years)
• 3y (3 years)
• 5y (5 years)

**Quick command:** `/analyze AAPL 1y`
        """
        
        keyboard = [
            [InlineKeyboardButton("📈 Analyze Stock", callback_data="ask_ticker")],
            [InlineKeyboardButton("❓ Help", callback_data="help")]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            welcome_message,
            reply_markup=reply_markup,
            parse_mode=ParseMode.MARKDOWN
        )
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show help message"""
        help_message = """
📖 **How to use:**

Just send me a stock ticker like:
• AAPL
• TSLA  
• GOOGL
• MSFT
• NVDA

Or use: `/analyze TICKER PERIOD`

**Example commands:**
• `/analyze AAPL 3m`
• `/analyze TSLA 1y`
• `/analyze NVDA 5y`

**Supported timeframes:**
3m, 6m, 1y, 2y, 3y, 5y
        """
        
        keyboard = [
            [InlineKeyboardButton("📈 Analyze Now", callback_data="ask_ticker")],
            [InlineKeyboardButton("⬅️ Back", callback_data="back_to_start")]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        if update.callback_query:
            await update.callback_query.message.reply_text(
                text=help_message,
                reply_markup=reply_markup,
                parse_mode=ParseMode.MARKDOWN
            )
        else:
            await update.message.reply_text(
                help_message,
                reply_markup=reply_markup,
                parse_mode=ParseMode.MARKDOWN
            )
    
    async def analyze_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /analyze command"""
        if not context.args:
            await update.message.reply_text(
                "Please specify a ticker and optional period.\n"
                "Example: `/analyze AAPL 1y`\n"
                "Or just send a ticker like: AAPL",
                parse_mode=ParseMode.MARKDOWN
            )
            return
        
        ticker = context.args[0].upper()
        
        # Default period
        period = '1y'
        if len(context.args) > 1:
            period_input = context.args[1].lower()
            period_map = {
                '3m': '3m', '3months': '3m',
                '6m': '6m', '6months': '6m',
                '1y': '1y', '1year': '1y',
                '2y': '2y', '2years': '2y',
                '3y': '3y', '3years': '3y',
                '5y': '5y', '5years': '5y'
            }
            period = period_map.get(period_input, '1y')
        
        await self.perform_analysis(update, context, ticker, period)
    
    async def ask_for_ticker(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Ask user to input ticker"""
        message = "📝 **Enter Stock Ticker**\n\nPlease type the stock symbol (e.g., AAPL, TSLA, GOOGL)"
        
        if update.callback_query:
            await update.callback_query.message.reply_text(message)
        else:
            await update.message.reply_text(message)
    
    async def ask_for_period(self, update: Update, context: ContextTypes.DEFAULT_TYPE, ticker: str):
        """Ask user to select timeframe for a ticker"""
        message = f"📊 **Select timeframe for {ticker}**\n\nChoose analysis period:"
        
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
            ]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            message,
            reply_markup=reply_markup,
            parse_mode=ParseMode.MARKDOWN
        )
    
    async def perform_analysis(self, update: Update, context: ContextTypes.DEFAULT_TYPE, 
                             ticker: str, period: str):
        """Perform analysis and show results"""
        user_id = update.effective_user.id
        
        # Send typing indicator
        try:
            await context.bot.send_chat_action(
                chat_id=update.effective_chat.id,
                action=ChatAction.TYPING
            )
        except:
            pass
        
        # Send status message
        status_msg = await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=f"🔄 **Analyzing {ticker} ({period})...**\nPlease wait...",
            parse_mode=ParseMode.MARKDOWN
        )
        
        try:
            # Perform analysis
            analysis = await self.analyzer.analyze_ticker(ticker, period)
            
            if not analysis['success']:
                await context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text=f"❌ **Failed to analyze {ticker}**\n\n"
                         f"Error: {analysis.get('error', 'Unknown error')}",
                    parse_mode=ParseMode.MARKDOWN
                )
                return
            
            # Generate chart
            chart_path = None
            try:
                chart_path = self.chart_generator.generate_price_chart(
                    analysis['data'], ticker, period
                )
            except Exception as e:
                logger.error(f"Chart error: {e}")
            
            # Create action buttons
            keyboard = [
                [
                    InlineKeyboardButton("🔄 New Analysis", callback_data="ask_ticker"),
                    InlineKeyboardButton(f"📈 {ticker} Again", callback_data=f"analyze_{ticker}_1y")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            # Send chart if available
            if chart_path and os.path.exists(chart_path):
                try:
                    with open(chart_path, 'rb') as chart_file:
                        await context.bot.send_photo(
                            chat_id=update.effective_chat.id,
                            photo=chart_file,
                            caption=analysis['summary'],
                            reply_markup=reply_markup,
                            parse_mode=ParseMode.MARKDOWN
                        )
                except Exception as e:
                    logger.error(f"Photo send error: {e}")
                    # Fallback to text
                    await context.bot.send_message(
                        chat_id=update.effective_chat.id,
                        text=analysis['summary'],
                        reply_markup=reply_markup,
                        parse_mode=ParseMode.MARKDOWN
                    )
                finally:
                    # Clean up
                    try:
                        os.remove(chart_path)
                    except:
                        pass
            else:
                # No chart, just send analysis
                await context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text=analysis['summary'],
                    reply_markup=reply_markup,
                    parse_mode=ParseMode.MARKDOWN
                )
            
            # Send technical overview
            try:
                await context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text=analysis['technical_overview'],
                    parse_mode=ParseMode.MARKDOWN
                )
            except:
                pass
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=f"❌ **Analysis failed**\n\n{str(e)[:200]}",
                parse_mode=ParseMode.MARKDOWN
            )
        
        finally:
            # Delete status message
            try:
                await status_msg.delete()
            except:
                pass
    
    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle callback queries"""
        query = update.callback_query
        await query.answer()
        
        callback_data = query.data
        
        if callback_data == "ask_ticker":
            await query.message.reply_text(
                "📝 **Enter Stock Ticker**\n\n"
                "Type the stock symbol (e.g., AAPL, TSLA, GOOGL, MSFT, NVDA)"
            )
        
        elif callback_data == "help":
            await self.help_command(update, context)
        
        elif callback_data == "back_to_start":
            await self.start(update, context)
        
        elif callback_data.startswith("analyze_"):
            # Format: analyze_TICKER_PERIOD
            parts = callback_data.split("_")
            if len(parts) >= 3:
                ticker = parts[1]
                period = parts[2]
                await self.perform_analysis(update, context, ticker, period)
    
    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle text input - treat as ticker"""
        text = update.message.text.strip().upper()
        
        # Check if it looks like a ticker
        if 1 <= len(text) <= 6 and text.isalpha():
            await self.ask_for_period(update, context, text)
        else:
            await update.message.reply_text(
                f"❌ **Invalid ticker:** {text}\n\n"
                f"Please enter a valid stock symbol (1-6 letters).\n"
                f"Examples: AAPL, TSLA, GOOGL",
                parse_mode=ParseMode.MARKDOWN
            )
    
    def run(self):
        """Run the bot"""
        if not self.config.TELEGRAM_TOKEN:
            logger.error("TELEGRAM_TOKEN not found")
            return
        
        app = Application.builder().token(self.config.TELEGRAM_TOKEN).build()
        
        # Add error handler
        app.add_error_handler(self.error_handler)
        
        # Add handlers
        app.add_handler(CommandHandler("start", self.start))
        app.add_handler(CommandHandler("help", self.help_command))
        app.add_handler(CommandHandler("analyze", self.analyze_command))
        app.add_handler(CallbackQueryHandler(self.handle_callback))
        
        # Handle text as ticker input
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text))
        
        # Start bot
        logger.info("Bot starting...")
        app.run_polling(allowed_updates=Update.ALL_TYPES)