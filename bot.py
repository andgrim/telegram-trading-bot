import asyncio
import logging
import sys
import os
from typing import Dict, List
import pandas as pd
import yfinance_fix
yfinance_fix.apply_yfinance_fix()

from telegram import (
    Update, 
    InlineKeyboardButton, 
    InlineKeyboardMarkup
)
from telegram.ext import (
    Application, 
    CommandHandler, 
    MessageHandler, 
    CallbackQueryHandler,
    ContextTypes,
    filters
)

from config import CONFIG
from analyzer import TradingAnalyzer
from chart_generator import ChartGenerator
from data_manager import DataManager
import yfinance as yf

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class TradingTelegramBot:
    """Telegram Bot for Technical Analysis with Reversal Detection"""
    
    def __init__(self, token: str):
        self.token = token
        self.analyzer = TradingAnalyzer()
        self.chart_gen = ChartGenerator()
        self.data_manager = DataManager()
        
        # Available commands
        self.commands = {
            'start': '🚀 Start the bot',
            'help': '📚 Show help',
            'analyze': '📊 Analyze ticker',
        }
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        user = update.effective_user
        
        welcome_msg = f"""
🤖 **Welcome to Trading Analyzer Pro, {user.first_name}!**

**Available Commands:**
"""
        for cmd, desc in self.commands.items():
            welcome_msg += f"• /{cmd} - {desc}\n"
        
        welcome_msg += "\n**Features:**"
        welcome_msg += "\n• Multi-period analysis (3M, 6M, 1Y)"
        welcome_msg += "\n• Reversal pattern detection"
        welcome_msg += "\n• Bollinger Bands, RSI, MACD, A/D Line"
        welcome_msg += "\n• Fundamental scoring"
        welcome_msg += "\n\n**Usage:**"
        welcome_msg += "\nUse `/analyze <TICKER>` to analyze any stock"
        welcome_msg += "\nExample: `/analyze AAPL` or `/analyze TSLA 6m`"
        
        keyboard = [
            [InlineKeyboardButton("📊 Analyze AAPL", callback_data='analyze_AAPL')],
            [InlineKeyboardButton("📊 Analyze TSLA", callback_data='analyze_TSLA')],
            [InlineKeyboardButton("📊 Analyze MSFT", callback_data='analyze_MSFT')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            welcome_msg,
            parse_mode='Markdown',
            reply_markup=reply_markup
        )
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        help_msg = """
**📚 Trading Analyzer Pro - Help Guide**

**Basic Commands:**
• `/start` - Start bot
• `/help` - Show this help message
• `/analyze <ticker>` - Analyze ticker (default: 1Y)

**Analysis Periods:**
• 3 Months (3m)
• 6 Months (6m) 
• 1 Year (1y)

**Technical Indicators:**
• Moving Averages (9, 20, 50)
• RSI (Relative Strength Index)
• MACD (Moving Average Convergence Divergence)
• Bollinger Bands
• Accumulation/Distribution Line

**Examples:**
• `/analyze AAPL` - Analyze Apple (1 year)
• `/analyze TSLA 6m` - Analyze Tesla (6 months)
• `/analyze MSFT 3m` - Analyze Microsoft (3 months)

**Note:** Some tickers may not be available due to Yahoo Finance limitations.
"""
        
        await update.message.reply_text(help_msg, parse_mode='Markdown')
    
    async def analyze_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /analyze command with better error handling"""
        if not context.args:
            await update.message.reply_text(
                "Please provide a ticker symbol.\nExample: `/analyze AAPL` or `/analyze AAPL 6m`",
                parse_mode='Markdown'
            )
            return
        
        ticker_symbol = context.args[0].upper()
        period = context.args[1] if len(context.args) > 1 else '1y'
        
        # Validate period
        if period not in CONFIG.TIME_PERIODS:
            period = '1y'
        
        await update.message.reply_text(
            f"📊 Analyzing *{ticker_symbol}* ({period.upper()})...",
            parse_mode='Markdown'
        )
        
        try:
            # Validate ticker with simple check
            try:
                ticker_test = yf.Ticker(ticker_symbol)
                # Quick check without full info
                _ = ticker_test.history(period="1d")
                logger.info(f"Ticker validation passed for {ticker_symbol}")
            except Exception as e:
                logger.error(f"Ticker validation error: {e}")
                await update.message.reply_text(
                    f"❌ Ticker *{ticker_symbol}* not found or no data available.\n\n"
                    f"Try popular tickers like: AAPL, TSLA, MSFT, GOOGL, AMZN",
                    parse_mode='Markdown'
                )
                return
            
            # Perform analysis
            analysis = await self.analyzer.analyze_ticker(ticker_symbol, period)
            
            if not analysis['success']:
                error_msg = analysis['error']
                if 'No data found' in error_msg or 'No price data' in error_msg:
                    await update.message.reply_text(
                        f"❌ No data available for *{ticker_symbol}* ({period.upper()}).\n\n"
                        f"Possible reasons:\n"
                        f"• Ticker may be delisted\n"
                        f"• No trading data for this period\n"
                        f"• Yahoo Finance API limitation\n\n"
                        f"Try a different ticker or period.",
                        parse_mode='Markdown'
                    )
                else:
                    await update.message.reply_text(f"❌ Analysis failed: {error_msg}")
                return
            
            # Send summary
            await update.message.reply_text(analysis['summary'], parse_mode='Markdown')
            
            # Send technical overview
            await update.message.reply_text(analysis['technical_overview'], parse_mode='Markdown')
            
            # Generate and send chart (only for 6m and 1y periods)
            if period in ['6m', '1y']:
                try:
                    await self._send_chart(update, ticker_symbol, period, analysis['data'])
                except Exception as e:
                    logger.error(f"Chart generation error: {e}")
                    await update.message.reply_text("⚠️ Could not generate chart (chart data unavailable).")
            
            # Show period selector
            await self._show_period_selector(update, ticker_symbol)
            
        except Exception as e:
            logger.error(f"Error analyzing: {e}")
            await update.message.reply_text(
                f"❌ Error analyzing *{ticker_symbol}*.\n\n"
                f"This may be due to:\n"
                f"• Temporary Yahoo Finance API issue\n"
                f"• Network connectivity problem\n"
                f"• Unsupported ticker symbol\n\n"
                f"Please try again in a few moments.",
                parse_mode='Markdown'
            )
    
    async def callback_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle callback queries"""
        query = update.callback_query
        await query.answer()
        
        data = query.data
        
        try:
            if data.startswith('analyze_'):
                ticker = data.split('_')[1]
                await self._handle_analyze_callback(query, ticker)
            
            elif data.startswith('chart_'):
                parts = data.split('_')
                if len(parts) >= 3:
                    ticker = parts[1]
                    period = parts[2]
                    await self._handle_chart_callback(query, ticker, period)
            
            elif data.startswith('period_'):
                ticker = data.split('_')[1]
                await self._show_period_selector_callback(query, ticker)
        
        except Exception as e:
            logger.error(f"Callback error: {e}")
            await query.edit_message_text(f"❌ Error: {str(e)}")
    
    async def _handle_analyze_callback(self, query, ticker: str):
        """Handle analyze callback"""
        await query.edit_message_text(f"📊 Analyzing *{ticker}*...", parse_mode='Markdown')
        
        try:
            analysis = await self.analyzer.analyze_ticker(ticker, '1y')
            
            if not analysis['success']:
                await query.edit_message_text(f"❌ Analysis failed: {analysis['error']}")
                return
            
            # Send summary and overview
            await query.edit_message_text(analysis['summary'], parse_mode='Markdown')
            await query.message.reply_text(analysis['technical_overview'], parse_mode='Markdown')
            
            # Send 1Y chart
            try:
                chart_path = self.chart_gen.generate_price_chart(analysis['data'], ticker, '1y')
                await self._send_photo_callback(query, chart_path, f"📈 {ticker} - 1 Year Chart")
                if os.path.exists(chart_path):
                    os.remove(chart_path)
            except Exception as e:
                logger.error(f"Chart error in callback: {e}")
            
            # Show period selector
            await self._show_period_selector_callback(query, ticker)
            
        except Exception as e:
            logger.error(f"Analyze callback error: {e}")
            await query.edit_message_text(f"❌ Error: {str(e)}")
    
    async def _handle_chart_callback(self, query, ticker: str, period: str):
        """Handle chart callback"""
        await query.edit_message_text(f"📈 Generating {period.upper()} chart for *{ticker}*...", parse_mode='Markdown')
        
        try:
            # Use the analyzer's method to get data
            analysis = await self.analyzer.analyze_ticker(ticker, period)
            
            if not analysis['success']:
                await query.edit_message_text(f"❌ No data available for {ticker} ({period})")
                return
            
            data = analysis['data']
            
            # Generate chart
            chart_path = self.chart_gen.generate_price_chart(data, ticker, period)
            
            await self._send_photo_callback(query, chart_path, f"📈 {ticker} - {period.upper()} Chart")
            if os.path.exists(chart_path):
                os.remove(chart_path)
            
        except Exception as e:
            logger.error(f"Chart callback error: {e}")
            await query.edit_message_text(f"❌ Error generating chart: {str(e)}")
    
    async def _show_period_selector(self, update, ticker: str):
        """Show period selection buttons"""
        periods = [('6 Months', '6m'), ('1 Year', '1y')]
        
        keyboard = []
        row = []
        for label, period in periods:
            row.append(InlineKeyboardButton(
                label,
                callback_data=f'chart_{ticker}_{period}'
            ))
        keyboard.append(row)
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            f"Select chart period for *{ticker}*:",
            parse_mode='Markdown',
            reply_markup=reply_markup
        )
    
    async def _show_period_selector_callback(self, query, ticker: str):
        """Show period selector in callback"""
        periods = [('6 Months', '6m'), ('1 Year', '1y')]
        
        keyboard = []
        row = []
        for label, period in periods:
            row.append(InlineKeyboardButton(
                label,
                callback_data=f'chart_{ticker}_{period}'
            ))
        keyboard.append(row)
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.message.reply_text(
            f"Select chart period for *{ticker}*:",
            parse_mode='Markdown',
            reply_markup=reply_markup
        )
    
    async def _send_chart(self, update, ticker: str, period: str, data: pd.DataFrame):
        """Send chart from update"""
        try:
            chart_path = self.chart_gen.generate_price_chart(data, ticker, period)
            await self._send_photo(update, chart_path, f"📈 {ticker} - {period.upper()} Chart")
            if os.path.exists(chart_path):
                os.remove(chart_path)
        except Exception as e:
            logger.error(f"Error sending chart: {e}")
            await update.message.reply_text("⚠️ Could not generate chart.")
    
    async def _send_photo(self, update, photo_path: str, caption: str):
        """Send photo from update"""
        try:
            with open(photo_path, 'rb') as photo:
                await update.message.reply_photo(photo=photo, caption=caption)
        except Exception as e:
            logger.error(f"Error sending photo: {e}")
    
    async def _send_photo_callback(self, query, photo_path: str, caption: str):
        """Send photo from callback"""
        try:
            with open(photo_path, 'rb') as photo:
                await query.message.reply_photo(photo=photo, caption=caption)
        except Exception as e:
            logger.error(f"Error sending photo in callback: {e}")
    
    async def _handle_text_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle text messages - treat as ticker symbols"""
        text = update.message.text.strip().upper()
        
        # Check if it looks like a ticker (1-10 characters, mostly alphanumeric)
        if 1 <= len(text) <= 10 and text.replace('-', '').replace('.', '').isalnum():
            keyboard = [
                [
                    InlineKeyboardButton("📊 Analyze 1Y", callback_data=f'analyze_{text}'),
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.message.reply_text(
                f"Ticker detected: *{text}*\nSelect action:",
                parse_mode='Markdown',
                reply_markup=reply_markup
            )
        else:
            # If it doesn't look like a ticker, show help
            await update.message.reply_text(
                f"Please use a valid ticker symbol.\n\nExamples:\n"
                f"• `AAPL` (Apple)\n"
                f"• `TSLA` (Tesla)\n"
                f"• `MSFT` (Microsoft)\n\n"
                f"Use `/analyze <TICKER>` to analyze.",
                parse_mode='Markdown'
            )
    
    async def error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle errors"""
        logger.error(f"Error: {context.error}")
        
        if update and update.effective_message:
            await update.effective_message.reply_text(
                "❌ An error occurred. Please try again."
            )
    
    async def run(self):
        """Start the bot"""
        # Get token from environment
        token = os.getenv('TELEGRAM_TOKEN')
        if not token:
            logger.error("❌ TELEGRAM_TOKEN environment variable is not set!")
            logger.error("Please set it in Render dashboard -> Environment")
            return
        
        logger.info(f"🤖 Starting Trading Analyzer Pro with token: {token[:10]}...")
        
        application = Application.builder().token(token).build()
        
        # Add error handler
        application.add_error_handler(self.error_handler)
        
        # Add command handlers
        application.add_handler(CommandHandler("start", self.start_command))
        application.add_handler(CommandHandler("help", self.help_command))
        application.add_handler(CommandHandler("analyze", self.analyze_command))
        
        # Add callback handler
        application.add_handler(CallbackQueryHandler(self.callback_handler))
        
        # Add message handler
        application.add_handler(MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            self._handle_text_message
        ))
        
        # Start bot
        logger.info("🤖 Trading Analyzer Pro starting on Render...")
        
        try:
            # For Render, use polling with proper timeouts
            await application.initialize()
            await application.start()
            
            # Start polling
            await application.updater.start_polling(
                poll_interval=1.0,
                timeout=20,
                read_timeout=20,
                write_timeout=20,
                connect_timeout=20,
                pool_timeout=20,
                bootstrap_retries=3
            )
            
            logger.info("✅ Bot is running and polling for updates...")
            
            # Keep running until stopped
            await asyncio.Event().wait()
            
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received.")
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            raise
        finally:
            if application.running:
                await application.stop()

async def main():
    """Main function"""
    bot = TradingTelegramBot("dummy_token")  # Token will be read from env
    await bot.run()

if __name__ == "__main__":
    # Check if running on Render
    if os.getenv('RENDER'):
        print("🚀 Running on Render.com")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Bot stopped.")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)