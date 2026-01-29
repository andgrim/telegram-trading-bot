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

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class TradingBot:
    """Advanced Telegram bot for technical analysis of global markets"""
    
    def __init__(self):
        self.config = CONFIG
        self.analyzer = TradingAnalyzer()
        self.chart_generator = ChartGenerator()
        
        # Market examples for help messages
        self.market_examples = {
            'US': ['AAPL', 'TSLA', 'GOOGL', 'MSFT', 'NVDA'],
            'EUROPE': ['ISP.MI', 'AI.PA', 'ADS.DE', 'ASML.AS'],
            'INDICES': ['SPX', 'DAX', 'CAC', 'GOLD', 'OIL'],
            'CRYPTO': ['BTC', 'ETH', 'XRP']
        }
    
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
        """Start command - show comprehensive menu"""
        welcome_message = """
🌍 **Global Markets Analysis Bot**

Welcome! I can analyze any financial instrument with technical indicators.

**Supported Markets:**
• US Stocks (AAPL, TSLA, MSFT)
• European Stocks (ISP.MI, AI.PA, ADS.DE)
• Indices (SPX, DAX, CAC, FTSE)
• Commodities (GOLD, OIL, SILVER)
• Cryptocurrencies (BTC, ETH, XRP)
• Currency pairs (EURUSD, GBPUSD)

**How to use:**
1. Send me a ticker symbol
2. I'll ask for timeframe
3. You'll get analysis + chart

**Available timeframes:**
• 3m (3 months)
• 6m (6 months)  
• 1y (1 year)
• 2y (2 years)
• 3y (3 years)
• 5y (5 years)

**Quick command:** `/analyze ISP.MI 1y`
        """
        
        keyboard = [
            [InlineKeyboardButton("📈 Analyze Now", callback_data="ask_ticker")],
            [
                InlineKeyboardButton("🇺🇸 US Stocks", callback_data="market_us"),
                InlineKeyboardButton("🇪🇺 European", callback_data="market_eu")
            ],
            [
                InlineKeyboardButton("📊 Indices", callback_data="market_indices"),
                InlineKeyboardButton("🛢️ Commodities", callback_data="market_commodities")
            ],
            [InlineKeyboardButton("❓ Help", callback_data="help")]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            welcome_message,
            reply_markup=reply_markup,
            parse_mode=ParseMode.MARKDOWN
        )
    
    async def market_examples_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE, market_type: str):
        """Show examples for a specific market"""
        query = update.callback_query
        await query.answer()
        
        market_titles = {
            'us': '🇺🇸 **US Stocks Examples**',
            'eu': '🇪🇺 **European Stocks Examples**',
            'indices': '📊 **Indices & Commodities Examples**',
            'commodities': '🛢️ **Commodities & Futures Examples**'
        }
        
        market_data = {
            'us': {
                'title': '🇺🇸 **US STOCKS**',
                'examples': ['AAPL', 'TSLA', 'GOOGL', 'MSFT', 'NVDA', 'AMZN', 'META', 'NFLX'],
                'description': 'Just type the ticker symbol (no exchange suffix needed)'
            },
            'eu': {
                'title': '🇪🇺 **EUROPEAN STOCKS**',
                'examples': [
                    'ISP.MI (Italy)', 'AI.PA (France)', 'ADS.DE (Germany)', 
                    'ASML.AS (Netherlands)', 'ABI.BR (Belgium)', 'AMS.MC (Spain)'
                ],
                'description': 'Use the exchange suffix: .MI, .PA, .DE, .AS, .BR, .MC'
            },
            'indices': {
                'title': '📊 **INDICES**',
                'examples': [
                    'SPX (S&P 500)', 'DJI (Dow Jones)', 'DAX (Germany)', 
                    'CAC (France)', 'FTSE (UK)', 'N225 (Japan)'
                ],
                'description': 'Common index symbols (no . suffix needed)'
            },
            'commodities': {
                'title': '🛢️ **COMMODITIES & FUTURES**',
                'examples': [
                    'GOLD (Gold Futures)', 'OIL (Crude Oil)', 'SILVER (Silver)',
                    'NATGAS (Natural Gas)', 'COPPER', 'EURUSD (Euro/USD)'
                ],
                'description': 'Common commodity and currency symbols'
            }
        }
        
        if market_type not in market_data:
            await query.message.reply_text("Invalid market type")
            return
        
        data = market_data[market_type]
        message = f"{data['title']}\n\n"
        message += f"**Examples:**\n"
        
        for example in data['examples']:
            message += f"• {example}\n"
        
        message += f"\n{data['description']}\n\n"
        message += "**How to use:**\nJust type or click any example below:"
        
        # Create quick action buttons for examples
        keyboard = []
        if market_type == 'us':
            for ticker in ['AAPL', 'TSLA', 'GOOGL', 'MSFT']:
                keyboard.append([InlineKeyboardButton(ticker, callback_data=f"quick_{ticker}")])
        elif market_type == 'eu':
            for ticker in ['ISP.MI', 'AI.PA', 'ADS.DE', 'ASML.AS']:
                keyboard.append([InlineKeyboardButton(ticker, callback_data=f"quick_{ticker}")])
        elif market_type == 'indices':
            for ticker in ['SPX', 'DAX', 'CAC', 'FTSE']:
                keyboard.append([InlineKeyboardButton(ticker, callback_data=f"quick_{ticker}")])
        elif market_type == 'commodities':
            for ticker in ['GOLD', 'OIL', 'SILVER', 'EURUSD']:
                keyboard.append([InlineKeyboardButton(ticker, callback_data=f"quick_{ticker}")])
        
        keyboard.append([InlineKeyboardButton("⬅️ Back", callback_data="back_to_start")])
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.message.edit_text(
            text=message,
            reply_markup=reply_markup,
            parse_mode=ParseMode.MARKDOWN
        )
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show comprehensive help message for global markets"""
        help_message = """
📖 **How to Use - Global Markets**

**BASIC USAGE:**
Just send me a ticker symbol like:
• AAPL (Apple - US)
• ISP.MI (Intesa Sanpaolo - Italy)
• AI.PA (Air Liquide - France)
• SPX (S&P 500 Index)
• GOLD (Gold Futures)
• BTC (Bitcoin)

**COMMANDS:**
• `/start` - Show main menu
• `/help` - This help message
• `/analyze TICKER PERIOD` - Quick analysis
  Example: `/analyze ISP.MI 1y`

**SUPPORTED MARKETS:**
**🇺🇸 US Stocks:** AAPL, TSLA, GOOGL, MSFT, NVDA, AMZN
**🇪🇺 European Stocks:**
  • Italy: .MI (ISP.MI, ENEL.MI, ENI.MI)
  • France: .PA (AI.PA, AIR.PA, BNP.PA)
  • Germany: .DE (ADS.DE, ALV.DE, BMW.DE)
  • Netherlands: .AS (ADYEN.AS, ASML.AS)
  • Spain: .MC (AMS.MC)
  • Belgium: .BR (ABI.BR)
  • Ireland: .IR (CRG.IR)

**📊 Indices:**
  • SPX (S&P 500), DJI (Dow Jones)
  • DAX (Germany), CAC (France), FTSE (UK)
  • N225 (Japan), HSI (Hong Kong)

**🛢️ Commodities:**
  • GOLD, SILVER, OIL, NATGAS, COPPER

**💱 Currencies:**
  • EURUSD, GBPUSD, USDJPY

**💰 Cryptocurrencies:**
  • BTC, ETH, XRP

**Timeframes:**
3m, 6m, 1y, 2y, 3y, 5y

**Tip:** European stocks require the exchange suffix (.MI, .PA, .DE, etc.)
        """
        
        keyboard = [
            [
                InlineKeyboardButton("🇺🇸 US", callback_data="market_us"),
                InlineKeyboardButton("🇪🇺 Europe", callback_data="market_eu"),
                InlineKeyboardButton("📊 Indices", callback_data="market_indices")
            ],
            [
                InlineKeyboardButton("🛢️ Commodities", callback_data="market_commodities"),
                InlineKeyboardButton("📈 Analyze Now", callback_data="ask_ticker")
            ],
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
        """Handle /analyze command with global market support"""
        if not context.args:
            await update.message.reply_text(
                "Please specify a ticker and optional period.\n\n"
                "**Examples:**\n"
                "• `/analyze AAPL 1y` (US stock)\n"
                "• `/analyze ISP.MI 6m` (European stock)\n"
                "• `/analyze SPX 2y` (Index)\n"
                "• `/analyze GOLD 3m` (Commodity)\n\n"
                "Or just send a ticker like: AAPL or ISP.MI",
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
        """Ask user to input ticker with global market examples"""
        message = """
📝 **Enter Ticker Symbol**

Please type the symbol for the financial instrument you want to analyze:

**Examples:**
• **US Stocks:** AAPL, TSLA, GOOGL
• **European Stocks:** ISP.MI (Italy), AI.PA (France), ADS.DE (Germany)
• **Indices:** SPX, DAX, CAC, FTSE
• **Commodities:** GOLD, OIL, SILVER
• **Cryptocurrencies:** BTC, ETH, XRP
• **Currencies:** EURUSD, GBPUSD

**Note:** European stocks require exchange suffix (.MI, .PA, .DE, .AS, etc.)
        """
        
        keyboard = [
            [
                InlineKeyboardButton("🇺🇸 US", callback_data="market_us"),
                InlineKeyboardButton("🇪🇺 Europe", callback_data="market_eu")
            ],
            [
                InlineKeyboardButton("📊 Indices", callback_data="market_indices"),
                InlineKeyboardButton("🛢️ Commodities", callback_data="market_commodities")
            ],
            [InlineKeyboardButton("⬅️ Back", callback_data="back_to_start")]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        if update.callback_query:
            await update.callback_query.message.reply_text(
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
    
    async def ask_for_period(self, update: Update, context: ContextTypes.DEFAULT_TYPE, ticker: str):
        """Ask user to select timeframe for a ticker"""
        # Show market type in the message
        market_type = self._get_market_type_display(ticker)
        
        message = f"📊 **Select timeframe for {ticker}**\n\n"
        message += f"**Market:** {market_type}\n"
        message += "Choose analysis period:"
        
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
            [InlineKeyboardButton("⬅️ Back", callback_data="ask_ticker")]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            message,
            reply_markup=reply_markup,
            parse_mode=ParseMode.MARKDOWN
        )
    
    def _get_market_type_display(self, ticker: str) -> str:
        """Get display string for market type"""
        ticker_upper = ticker.upper()
        
        if '.' in ticker_upper:
            suffix = ticker_upper.split('.')[-1]
            if suffix == 'MI':
                return '🇮🇹 Italian Stock'
            elif suffix == 'PA':
                return '🇫🇷 French Stock'
            elif suffix == 'DE':
                return '🇩🇪 German Stock'
            elif suffix == 'AS':
                return '🇳🇱 Dutch Stock'
            elif suffix == 'BR':
                return '🇧🇪 Belgian Stock'
            elif suffix == 'MC':
                return '🇪🇸 Spanish Stock'
            elif suffix == 'IR':
                return '🇮🇪 Irish Stock'
            elif suffix == 'SW':
                return '🇨🇭 Swiss Stock'
            elif suffix == 'L':
                return '🇬🇧 UK Stock'
            else:
                return 'European Stock'
        
        # Check for indices and commodities
        index_map = {
            'SPX': '🇺🇸 S&P 500 Index',
            'DJI': '🇺🇸 Dow Jones Index',
            'IXIC': '🇺🇸 NASDAQ Index',
            'DAX': '🇩🇪 DAX Index',
            'CAC': '🇫🇷 CAC 40 Index',
            'FTSE': '🇬🇧 FTSE 100 Index',
            'N225': '🇯🇵 Nikkei 225 Index',
            'HSI': '🇭🇰 Hang Seng Index'
        }
        
        commodity_map = {
            'GOLD': '🟡 Gold Futures',
            'SILVER': '⚪ Silver Futures',
            'OIL': '🛢️ Crude Oil Futures',
            'BRENT': '🛢️ Brent Crude',
            'NATGAS': '🔥 Natural Gas',
            'COPPER': '🔴 Copper Futures'
        }
        
        crypto_map = {
            'BTC': '₿ Bitcoin',
            'ETH': 'Ξ Ethereum',
            'XRP': 'XRP Ripple'
        }
        
        if ticker_upper in index_map:
            return index_map[ticker_upper]
        elif ticker_upper in commodity_map:
            return commodity_map[ticker_upper]
        elif ticker_upper in crypto_map:
            return crypto_map[ticker_upper]
        elif any(ticker_upper.endswith(f'={c}') for c in ['X', 'F']):
            return 'Currency/Future'
        
        return '🇺🇸 US Stock'
    
    async def perform_analysis(self, update: Update, context: ContextTypes.DEFAULT_TYPE, 
                             ticker: str, period: str):
        """Perform analysis and show results for global markets"""
        user_id = update.effective_user.id
        
        # Send typing indicator
        try:
            await context.bot.send_chat_action(
                chat_id=update.effective_chat.id,
                action=ChatAction.TYPING
            )
        except:
            pass
        
        # Send status message with market type
        market_type = self._get_market_type_display(ticker)
        status_msg = await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=f"🔄 **Analyzing {ticker} ({market_type})**\nPeriod: {period}\nPlease wait...",
            parse_mode=ParseMode.MARKDOWN
        )
        
        try:
            # Perform analysis
            analysis = await self.analyzer.analyze_ticker(ticker, period)
            
            if not analysis['success']:
                error_msg = f"❌ **Failed to analyze {ticker}**\n\n"
                error_msg += f"Error: {analysis.get('error', 'Unknown error')}\n\n"
                error_msg += "**Tips:**\n"
                error_msg += "• Check the ticker symbol is correct\n"
                error_msg += "• European stocks need exchange suffix (.MI, .PA, .DE, etc.)\n"
                error_msg += "• Try alternative ticker formats\n"
                
                await context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text=error_msg,
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
                    InlineKeyboardButton(f"📈 {ticker} Again", callback_data=f"analyze_{ticker}_{period}")
                ],
                [
                    InlineKeyboardButton("3m", callback_data=f"analyze_{ticker}_3m"),
                    InlineKeyboardButton("6m", callback_data=f"analyze_{ticker}_6m"),
                    InlineKeyboardButton("1y", callback_data=f"analyze_{ticker}_1y"),
                    InlineKeyboardButton("2y", callback_data=f"analyze_{ticker}_2y")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            # Send chart if available
            if chart_path and os.path.exists(chart_path):
                try:
                    with open(chart_path, 'rb') as chart_file:
                        # Use compact summary for photo caption (under 1024 chars)
                        caption = analysis['compact_summary']
                        
                        # Ensure caption doesn't exceed Telegram limits
                        if len(caption) > 1024:
                            caption = caption[:1020] + "..."
                        
                        await context.bot.send_photo(
                            chat_id=update.effective_chat.id,
                            photo=chart_file,
                            caption=caption,
                            reply_markup=reply_markup,
                            parse_mode=ParseMode.MARKDOWN
                        )
                        
                        # Send the FULL analysis as a separate message(s)
                        await self._send_comprehensive_analysis(
                            context, update.effective_chat.id, analysis
                        )
                        
                except Exception as e:
                    logger.error(f"Photo send error: {e}")
                    # Fallback to text only
                    await self._send_comprehensive_analysis(
                        context, update.effective_chat.id, analysis
                    )
                finally:
                    # Clean up
                    try:
                        os.remove(chart_path)
                    except:
                        pass
            else:
                # No chart, just send analysis
                await self._send_comprehensive_analysis(
                    context, update.effective_chat.id, analysis
                )
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            error_text = f"❌ **Analysis failed for {ticker}**\n\n"
            error_text += f"Error: {str(e)[:200]}\n\n"
            error_text += "Please try again with a different ticker or timeframe."
            
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
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
        
        elif callback_data.startswith("market_"):
            # Format: market_type (market_us, market_eu, etc.)
            market_type = callback_data.split("_")[1]
            await self.market_examples_handler(update, context, market_type)
        
        elif callback_data.startswith("quick_"):
            # Format: quick_TICKER (quick_AAPL, quick_ISP.MI, etc.)
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
        """Check if text is a valid ticker symbol for global markets"""
        # Allow: letters, numbers, dots, hyphens, equals sign (for futures/currencies)
        # Examples: AAPL, ISP.MI, BTC-USD, EURUSD=X, GC=F
        pattern = r'^[A-Z0-9.\-=\$]+$'
        return bool(re.match(pattern, text)) and 1 <= len(text) <= 15
    
    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle text input - treat as ticker for global markets"""
        text = update.message.text.strip().upper()
        
        # Clean common variations
        if text.startswith('$'):
            text = text[1:]  # Remove $ prefix if present
        
        # Check if it looks like a valid ticker
        if self._is_valid_ticker_symbol(text):
            await self.ask_for_period(update, context, text)
        else:
            # Provide helpful error message
            error_message = f"""
❌ **Invalid ticker format:** {text}

**Valid ticker examples:**
• **US Stocks:** AAPL, TSLA, GOOGL
• **European Stocks:** ISP.MI, AI.PA, ADS.DE
• **Indices:** SPX, DAX, CAC
• **Commodities:** GOLD, OIL, SILVER
• **Cryptocurrencies:** BTC, ETH, XRP
• **Currencies:** EURUSD, GBPUSD

**Common issues:**
• European stocks need exchange suffix (.MI, .PA, .DE, .AS, etc.)
• No spaces in ticker symbols
• Maximum 15 characters

**Try these examples:**
/analyze AAPL 1y
/analyze ISP.MI 6m
/analyze SPX 2y
/analyze GOLD 3m
            """
            
            keyboard = [
                [
                    InlineKeyboardButton("🇺🇸 Examples", callback_data="market_us"),
                    InlineKeyboardButton("🇪🇺 Examples", callback_data="market_eu")
                ],
                [InlineKeyboardButton("📊 Examples", callback_data="market_indices")]
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
            logger.error("TELEGRAM_TOKEN not found in environment variables")
            print("❌ ERROR: TELEGRAM_TOKEN not found!")
            print("Please set the TELEGRAM_TOKEN environment variable")
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
        logger.info("🌍 Global Markets Bot starting...")
        print("🤖 Bot starting with global market support...")
        print("✅ Supported markets:")
        print("   • US Stocks (AAPL, TSLA, etc.)")
        print("   • European Stocks (ISP.MI, AI.PA, ADS.DE, etc.)")
        print("   • Indices (SPX, DAX, CAC, etc.)")
        print("   • Commodities (GOLD, OIL, SILVER, etc.)")
        print("   • Cryptocurrencies (BTC, ETH, XRP)")
        
        app.run_polling(allowed_updates=Update.ALL_TYPES)