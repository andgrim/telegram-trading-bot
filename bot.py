import logging
import os
import asyncio
import traceback
import re
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

def setup_logging():
    """Configure logging for production"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
        ]
    )
    
    # Suppress yfinance debug logs
    logging.getLogger('yfinance').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('telegram').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

class TradingBot:
    """Advanced Telegram bot for technical analysis of global markets"""
    
    def __init__(self):
        self.config = CONFIG
        self.analyzer = TradingAnalyzer()
        self.chart_generator = ChartGenerator()
        
        # Recommended stocks that work well with yfinance
        self.recommended_tickers = {
            'US_STOCKS': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX'],
            'ETFS': ['SPY', 'QQQ', 'VOO', 'VTI', 'IWM', 'GLD', 'SLV'],
            'INDICES': ['SPX', 'DJI', 'IXIC', 'DAX', 'FTSE', 'N225'],
            'COMMODITIES': ['GOLD', 'OIL', 'SILVER'],
            'CRYPTO': ['BTC', 'ETH'],
        }
    
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Start command - show comprehensive menu"""
        welcome_message = """
🌍 **Trading Analysis Bot**

Welcome! I provide technical analysis for financial markets.

**⚠️ IMPORTANT NOTE:**
Due to API rate limits on free tier, please use:
• Major US stocks (AAPL, MSFT, TSLA)
• Popular ETFs (SPY, QQQ, VOO)
• Major indices (SPX, GOLD, OIL)
• Avoid obscure or international stocks

**Recommended tickers:**
• **US Stocks:** AAPL, MSFT, TSLA, GOOGL
• **ETFs:** SPY, QQQ, VOO, GLD
• **Indices:** SPX, DJI, GOLD, OIL
• **Crypto:** BTC, ETH

**How to use:**
1. Send me a ticker symbol
2. Select timeframe
3. Get analysis + chart

**Commands:**
• `/start` - This menu
• `/help` - Detailed help
• `/analyze TICKER PERIOD` - Quick analysis
• `/recommended` - Show working tickers
        """
        
        keyboard = [
            [InlineKeyboardButton("📈 Analyze Now", callback_data="ask_ticker")],
            [
                InlineKeyboardButton("📊 US Stocks", callback_data="rec_us"),
                InlineKeyboardButton("💰 ETFs", callback_data="rec_etfs")
            ],
            [
                InlineKeyboardButton("📈 Indices", callback_data="rec_indices"),
                InlineKeyboardButton("🛢️ Commodities", callback_data="rec_commodities")
            ],
            [InlineKeyboardButton("❓ Help", callback_data="help")]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        if update.callback_query:
            # For callback queries, edit the existing message
            await update.callback_query.message.edit_text(
                welcome_message,
                reply_markup=reply_markup,
                parse_mode=ParseMode.MARKDOWN
            )
        else:
            # For regular commands, send new message
            await update.message.reply_text(
                welcome_message,
                reply_markup=reply_markup,
                parse_mode=ParseMode.MARKDOWN
            )
    
    async def show_recommended(self, update: Update, context: ContextTypes.DEFAULT_TYPE, category: str):
        """Show recommended tickers that work well"""
        query = update.callback_query
        await query.answer()
        
        categories = {
            'us': {
                'title': '📊 **Recommended US Stocks**',
                'tickers': self.recommended_tickers['US_STOCKS'],
                'description': 'These stocks have reliable data and work well with the bot.'
            },
            'etfs': {
                'title': '💰 **Recommended ETFs**',
                'tickers': self.recommended_tickers['ETFS'],
                'description': 'ETFs are great for analysis and have consistent data.'
            },
            'indices': {
                'title': '📈 **Recommended Indices**',
                'tickers': self.recommended_tickers['INDICES'],
                'description': 'Major market indices work reliably.'
            },
            'commodities': {
                'title': '🛢️ **Recommended Commodities**',
                'tickers': self.recommended_tickers['COMMODITIES'],
                'description': 'Commodity futures and ETFs.'
            }
        }
        
        if category not in categories:
            await query.message.reply_text("Invalid category")
            return
        
        data = categories[category]
        message = f"{data['title']}\n\n"
        message += f"{data['description']}\n\n"
        message += "**Tickers:**\n"
        
        # Group tickers for better display
        tickers = data['tickers']
        for i in range(0, len(tickers), 4):
            row = tickers[i:i+4]
            message += " • " + " | ".join(row) + "\n"
        
        message += "\n**Click any ticker below to analyze:**"
        
        # Create buttons for tickers
        keyboard = []
        row = []
        for i, ticker in enumerate(tickers):
            row.append(InlineKeyboardButton(ticker, callback_data=f"quick_{ticker}"))
            if (i + 1) % 4 == 0 or i == len(tickers) - 1:
                keyboard.append(row)
                row = []
        
        keyboard.append([InlineKeyboardButton("⬅️ Back", callback_data="back_to_start")])
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.message.edit_text(
            text=message,
            reply_markup=reply_markup,
            parse_mode=ParseMode.MARKDOWN
        )
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show comprehensive help message"""
        help_message = """
📖 **How to Use - Rate Limit Aware**

**⚠️ IMPORTANT:**
• Free tier has API rate limits
• Use major, liquid instruments
• Avoid obscure international stocks
• Wait 2-3 seconds between requests

**RECOMMENDED TICKERS:**
• **US Stocks:** AAPL, MSFT, GOOGL, AMZN, TSLA
• **ETFs:** SPY (S&P 500), QQQ (NASDAQ), VOO
• **Indices:** SPX, DJI, GOLD, OIL
• **Crypto:** BTC, ETH

**COMMANDS:**
• `/start` - Main menu
• `/help` - This message
• `/analyze TICKER PERIOD` - Quick analysis
• `/recommended` - Show working tickers

**EXAMPLES:**
• `/analyze AAPL 1y`
• `/analyze SPY 6m`
• `/analyze GOLD 3m`
• `/analyze BTC 1y`

**PERIODS:**
3m, 6m, 1y, 2y, 3y, 5y

**TIPS:**
1. Start with AAPL or SPY to test
2. Use 1y period for best results
3. Charts may fail on Render free tier
        """
        
        keyboard = [
            [
                InlineKeyboardButton("📊 US Stocks", callback_data="rec_us"),
                InlineKeyboardButton("💰 ETFs", callback_data="rec_etfs")
            ],
            [
                InlineKeyboardButton("📈 Indices", callback_data="rec_indices"),
                InlineKeyboardButton("🛢️ Commodities", callback_data="rec_commodities")
            ],
            [InlineKeyboardButton("📈 Analyze Now", callback_data="ask_ticker")]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        if update.callback_query:
            await update.callback_query.message.edit_text(
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
                "Please specify a ticker and optional period.\n\n"
                "**Examples:**\n"
                "• `/analyze AAPL 1y` (Apple)\n"
                "• `/analyze SPY 6m` (S&P 500 ETF)\n"
                "• `/analyze GOLD 3m` (Gold)\n"
                "• `/analyze BTC 1y` (Bitcoin)\n\n"
                "**Recommended:** AAPL, MSFT, TSLA, SPY, GOLD, BTC",
                parse_mode=ParseMode.MARKDOWN
            )
            return
        
        ticker = context.args[0].upper()
        
        # Default period
        period = '1y'
        if len(context.args) > 1:
            period_input = context.args[1].lower()
            period_map = {
                '3m': '3m', '3months': '3m', '3month': '3m',
                '6m': '6m', '6months': '6m', '6month': '6m',
                '1y': '1y', '1year': '1y',
                '2y': '2y', '2years': '2y',
                '3y': '3y', '3years': '3y',
                '5y': '5y', '5years': '5y'
            }
            period = period_map.get(period_input, '1y')
        
        await self.perform_analysis(update, context, ticker, period)
    
    async def ask_for_ticker(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Ask user to input ticker"""
        message = """
📝 **Enter Ticker Symbol**

Due to rate limits, please use:

**Recommended tickers:**
• **US Stocks:** AAPL, MSFT, TSLA, GOOGL, AMZN
• **ETFs:** SPY, QQQ, VOO, GLD
• **Indices:** SPX, GOLD, OIL
• **Crypto:** BTC, ETH

**Avoid:**
• Obscure international stocks
• Stocks with dots/suffixes (.MI, .PA, etc.)
• Penny stocks or low volume stocks

**Enter ticker:**
        """
        
        keyboard = [
            [
                InlineKeyboardButton("AAPL", callback_data="quick_AAPL"),
                InlineKeyboardButton("MSFT", callback_data="quick_MSFT"),
                InlineKeyboardButton("TSLA", callback_data="quick_TSLA")
            ],
            [
                InlineKeyboardButton("SPY", callback_data="quick_SPY"),
                InlineKeyboardButton("GOLD", callback_data="quick_GOLD"),
                InlineKeyboardButton("BTC", callback_data="quick_BTC")
            ],
            [InlineKeyboardButton("📊 More Options", callback_data="more_tickers")]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        if update.callback_query:
            await update.callback_query.message.edit_text(
                text=message,
                reply_markup=reply_markup,
                parse_mode=ParseMode.MARKDOWN
            )
        else:
            await update.message.reply_text(
                message,
                reply_markup=reply_markup,
                parse_mode=ParseMode.MARKDOWN
            )
    
    async def ask_for_period(self, update: Update, context: ContextTypes.DEFAULT_TYPE, ticker: str):
        """Ask user to select timeframe for a ticker"""
        # Determine which message to reply to
        if update.callback_query:
            # For callback queries, reply to the query's message
            message_obj = update.callback_query.message
        else:
            # For regular messages, use update.message
            message_obj = update.message
        
        message_text = f"📊 **Select timeframe for {ticker}**\n\n"
        message_text += "Choose analysis period:"
        
        keyboard = [
            [
                InlineKeyboardButton("3 Months", callback_data=f"analyze_{ticker}_3m"),
                InlineKeyboardButton("6 Months", callback_data=f"analyze_{ticker}_6m")
            ],
            [
                InlineKeyboardButton("1 Year", callback_data=f"analyze_{ticker}_1y"),
                InlineKeyboardButton("2 Years", callback_data=f"analyze_{ticker}_2y")
            ],
            [InlineKeyboardButton("⬅️ Back", callback_data="ask_ticker")]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await message_obj.reply_text(
            message_text,
            reply_markup=reply_markup,
            parse_mode=ParseMode.MARKDOWN
        )
    
    async def perform_analysis(self, update: Update, context: ContextTypes.DEFAULT_TYPE, 
                             ticker: str, period: str):
        """Perform analysis with improved error handling"""
        # Determine chat ID
        if update.callback_query:
            chat_id = update.callback_query.message.chat_id
        else:
            chat_id = update.effective_chat.id
        
        # Send typing indicator
        try:
            await context.bot.send_chat_action(
                chat_id=chat_id,
                action=ChatAction.TYPING
            )
        except:
            pass
        
        # Check if ticker is in recommended list
        is_recommended = False
        for category in self.recommended_tickers.values():
            if ticker in category:
                is_recommended = True
                break
        
        status_text = f"🔄 **Analyzing {ticker}**\nPeriod: {period}"
        if not is_recommended:
            status_text += "\n⚠️ This ticker may not work due to rate limits"
        
        status_msg = await context.bot.send_message(
            chat_id=chat_id,
            text=status_text + "\nPlease wait...",
            parse_mode=ParseMode.MARKDOWN
        )
        
        try:
            # Perform analysis
            analysis = await self.analyzer.analyze_ticker(ticker, period)
            
            if not analysis['success']:
                error_msg = f"❌ **Failed to analyze {ticker}**\n\n"
                error_msg += f"Error: {analysis.get('error', 'Unknown error')}\n\n"
                error_msg += "**Likely causes:**\n"
                error_msg += "• Yahoo Finance rate limit (try again in 60 seconds)\n"
                error_msg += "• Ticker not found or delisted\n"
                error_msg += "• International ticker format issue\n\n"
                error_msg += "**Try these instead:**\n"
                error_msg += "• AAPL, MSFT, TSLA (US stocks)\n"
                error_msg += "• SPY, GOLD (ETFs/commodities)\n"
                error_msg += "• BTC, ETH (crypto)\n\n"
                error_msg += "**Command:** `/analyze AAPL 1y`"
                
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=error_msg,
                    parse_mode=ParseMode.MARKDOWN
                )
                return
            
            # Generate chart (may fail on Render free tier)
            chart_path = None
            try:
                chart_path = self.chart_generator.generate_price_chart(
                    analysis['data'], ticker, period
                )
            except Exception as e:
                logger.info(f"Chart generation skipped: {e}")
            
            # Create action buttons
            keyboard = [
                [
                    InlineKeyboardButton("🔄 New Analysis", callback_data="ask_ticker"),
                    InlineKeyboardButton(f"📈 {ticker} Again", callback_data=f"analyze_{ticker}_{period}")
                ]
            ]
            
            # Add timeframe buttons for recommended tickers only
            if is_recommended:
                keyboard.append([
                    InlineKeyboardButton("3m", callback_data=f"analyze_{ticker}_3m"),
                    InlineKeyboardButton("6m", callback_data=f"analyze_{ticker}_6m"),
                    InlineKeyboardButton("1y", callback_data=f"analyze_{ticker}_1y")
                ])
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            # Send chart if available
            if chart_path and os.path.exists(chart_path):
                try:
                    with open(chart_path, 'rb') as chart_file:
                        caption = analysis['compact_summary']
                        if len(caption) > 1024:
                            caption = caption[:1020] + "..."
                        
                        await context.bot.send_photo(
                            chat_id=chat_id,
                            photo=chart_file,
                            caption=caption,
                            reply_markup=reply_markup,
                            parse_mode=ParseMode.MARKDOWN
                        )
                        
                        # Send full analysis
                        await self._send_comprehensive_analysis(
                            context, chat_id, analysis
                        )
                        
                except Exception as e:
                    logger.error(f"Photo send error: {e}")
                    await self._send_comprehensive_analysis(
                        context, chat_id, analysis
                    )
                finally:
                    try:
                        os.remove(chart_path)
                    except:
                        pass
            else:
                # Send analysis without chart
                analysis_text = analysis['summary']
                
                # Add note about chart
                if not is_recommended:
                    analysis_text += "\n\n⚠️ *Chart generation skipped due to rate limits*"
                
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=analysis_text,
                    reply_markup=reply_markup,
                    parse_mode=ParseMode.MARKDOWN
                )
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            error_text = f"❌ **Analysis failed for {ticker}**\n\n"
            error_text += "Yahoo Finance rate limit reached.\n\n"
            error_text += "**Please:**\n"
            error_text += "1. Wait 60 seconds\n"
            error_text += "2. Try a recommended ticker\n"
            error_text += "3. Use command: `/analyze AAPL 1y`\n\n"
            error_text += "**Recommended:** AAPL, MSFT, SPY, GOLD, BTC"
            
            await context.bot.send_message(
                chat_id=chat_id,
                text=error_text,
                parse_mode=ParseMode.MARKDOWN
            )
        
        finally:
            # Delete status message
            try:
                await status_msg.delete()
            except:
                pass
    
    async def _send_comprehensive_analysis(self, context: ContextTypes.DEFAULT_TYPE, 
                                         chat_id: int, analysis: Dict):
        """Send the comprehensive analysis in one or more messages"""
        try:
            analysis_text = analysis['summary']
            
            # Telegram has a 4096 character limit per message
            # Split if needed
            if len(analysis_text) <= 4096:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=analysis_text,
                    parse_mode=ParseMode.MARKDOWN
                )
            else:
                # Split the analysis into multiple parts
                parts = self._split_long_message(analysis_text)
                for i, part in enumerate(parts):
                    # Add part indicator
                    if len(parts) > 1:
                        part_text = f"**📊 Analysis Part {i+1}/{len(parts)}**\n\n{part}"
                    else:
                        part_text = part
                    
                    await context.bot.send_message(
                        chat_id=chat_id,
                        text=part_text,
                        parse_mode=ParseMode.MARKDOWN
                    )
                    await asyncio.sleep(0.5)  # Small delay between messages
                    
        except Exception as e:
            logger.error(f"Error sending comprehensive analysis: {e}")
            # Send a simplified version
            simplified = "📊 **Analysis Complete**\n\n"
            simplified += "Comprehensive analysis generated. Full details may be truncated.\n"
            simplified += f"• Signals: {len(analysis['signals'])} total\n"
            simplified += f"• Fundamental Score: {analysis['fundamental']['score']}/100\n"
            
            await context.bot.send_message(
                chat_id=chat_id,
                text=simplified,
                parse_mode=ParseMode.MARKDOWN
            )
    
    def _split_long_message(self, text: str, max_length: int = 4000) -> List[str]:
        """Split a long message into multiple parts"""
        parts = []
        
        # Try to split at meaningful boundaries
        lines = text.split('\n')
        current_part = []
        current_length = 0
        
        for line in lines:
            line_length = len(line) + 1  # +1 for newline
            
            if current_length + line_length > max_length:
                # Start new part
                if current_part:
                    parts.append('\n'.join(current_part))
                    current_part = []
                    current_length = 0
            
            current_part.append(line)
            current_length += line_length
        
        # Add the last part
        if current_part:
            parts.append('\n'.join(current_part))
        
        return parts
    
    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle callback queries for global markets"""
        query = update.callback_query
        await query.answer()
        
        callback_data = query.data
        
        if callback_data == "ask_ticker":
            await self.ask_for_ticker(update, context)
        
        elif callback_data == "help":
            await self.help_command(update, context)
        
        elif callback_data == "back_to_start":
            await self.start(update, context)
        
        elif callback_data.startswith("rec_"):
            # Format: rec_category (rec_us, rec_etfs, etc.)
            category = callback_data.split("_")[1]
            await self.show_recommended(update, context, category)
        
        elif callback_data == "more_tickers":
            await self.help_command(update, context)
        
        elif callback_data.startswith("quick_"):
            # Format: quick_TICKER (quick_AAPL, quick_SPY, etc.)
            ticker = callback_data.split("_")[1]
            await self.ask_for_period(update, context, ticker)
        
        elif callback_data.startswith("analyze_"):
            # Format: analyze_TICKER_PERIOD
            parts = callback_data.split("_")
            if len(parts) >= 3:
                ticker = parts[1]
                period = parts[2]
                await self.perform_analysis(update, context, ticker, period)
    
    def _is_valid_ticker_symbol(self, text: str) -> bool:
        """Check if text is a valid ticker symbol"""
        # Simplified validation
        pattern = r'^[A-Z0-9.\-=\$]{1,10}$'
        return bool(re.match(pattern, text))
    
    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle text input"""
        text = update.message.text.strip().upper()
        
        # Remove $ prefix
        if text.startswith('$'):
            text = text[1:]
        
        if self._is_valid_ticker_symbol(text):
            await self.ask_for_period(update, context, text)
        else:
            error_message = f"""
❌ **Invalid ticker: {text}**

**Please use:**
• AAPL, MSFT, TSLA (US stocks)
• SPY, QQQ, GLD (ETFs)
• GOLD, OIL, BTC (commodities/crypto)
• SPX, DJI (indices)

**Examples:**
/analyze AAPL 1y
/analyze SPY 6m
/analyze GOLD 3m
            """
            
            keyboard = [
                [
                    InlineKeyboardButton("AAPL", callback_data="quick_AAPL"),
                    InlineKeyboardButton("MSFT", callback_data="quick_MSFT"),
                    InlineKeyboardButton("SPY", callback_data="quick_SPY")
                ],
                [InlineKeyboardButton("📊 More Options", callback_data="more_tickers")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.message.reply_text(
                error_message,
                reply_markup=reply_markup,
                parse_mode=ParseMode.MARKDOWN
            )
    
    def run(self):
        """Run the bot"""
        if not self.config.TELEGRAM_TOKEN:
            logger.error("TELEGRAM_TOKEN not found")
            print("❌ ERROR: TELEGRAM_TOKEN not found!")
            print("Set TELEGRAM_TOKEN environment variable")
            return
        
        app = Application.builder().token(self.config.TELEGRAM_TOKEN).build()
        
        # Add error handler
        app.add_error_handler(self.error_handler)
        
        # Add handlers
        app.add_handler(CommandHandler("start", self.start))
        app.add_handler(CommandHandler("help", self.help_command))
        app.add_handler(CommandHandler("analyze", self.analyze_command))
        app.add_handler(CommandHandler("recommended", self.help_command))
        app.add_handler(CallbackQueryHandler(self.handle_callback))
        
        # Handle text as ticker input
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text))
        
        # Start bot
        logger.info("📈 Trading Bot starting with rate limiting...")
        print("🤖 Bot starting...")
        print("⚠️  IMPORTANT: Using rate limiting for Yahoo Finance")
        print("✅ Recommended tickers: AAPL, MSFT, SPY, GOLD, BTC")
        
        app.run_polling(allowed_updates=Update.ALL_TYPES)
    
    async def error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle errors in the bot"""
        logger.error(msg="Exception while handling an update:", exc_info=context.error)
        
        try:
            # Try to send error message to user
            if isinstance(update, Update):
                if update.callback_query:
                    chat_id = update.callback_query.message.chat_id
                elif update.message:
                    chat_id = update.effective_chat.id
                else:
                    return
                
                error_message = "❌ An error occurred. Please try again or use /start"
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=error_message,
                    parse_mode=ParseMode.MARKDOWN
                )
        except Exception as e:
            logger.error(f"Error sending error message: {e}")