import asyncio
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from trading_analyzer import TradingAnalyzer
from utils import search_tickers, get_ticker_info
import pandas as pd
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
from io import BytesIO
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class TelegramTradingBot:
    def __init__(self, token: str):
        self.token = token
        self.user_data = {}
        
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Start command"""
        user = update.effective_user
        welcome_message = f"""
👋 Welcome {user.first_name} to the Trading Bot!

📈 **Available commands:**
/search - Search for a ticker by symbol or name
/analyze <symbol> - Complete ticker analysis
/quick <symbol> - Quick analysis
/compare <sym1> <sym2> - Compare two tickers
/settings - Configure analysis parameters
/help - Show this guide

📊 **Examples:**
• /search Apple
• /analyze TSLA
• /analyze BTC-USD
• /compare AAPL MSFT
        """
        await update.message.reply_text(welcome_message)
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Help command"""
        help_text = """
📚 **Command Guide:**

🔍 **SEARCH & ANALYSIS**
/search - Search ticker by name or symbol
/analyze <symbol> - Complete ticker analysis
/quick <symbol> - Quick analysis
/compare <sym1> <sym2> - Compare two tickers

⚙️ **CONFIGURATION**
/settings - Configure analysis parameters
/periods <list> - Set periods (e.g.: 1,3,6,12)

📊 **UTILITY**
/top - Most popular tickers

📈 **SIGNALS**
/signals - Active trading signals

💡 **Examples:**
• /search Tesla
• /analyze AAPL
• /periods 1,3,6,12
• /compare AAPL MSFT
        """
        await update.message.reply_text(help_text, parse_mode='Markdown')
    
    async def search_ticker(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Search ticker by name or symbol"""
        query = ' '.join(context.args)
        if not query:
            await update.message.reply_text("❌ Specify what to search:\n/search Apple\n/search AAPL\n/search TSLA")
            return
        
        await update.message.reply_text(f"🔍 Searching for '{query}'...")
        
        try:
            results = search_tickers(query, limit=10)
            
            if not results:
                await update.message.reply_text("❌ No results found")
                return
            
            # Create inline keyboard with results
            keyboard = []
            for result in results[:8]:  # Max 8 results
                btn_text = f"{result['symbol']} - {result['name'][:30]}..."
                callback_data = f"select_{result['symbol']}"
                keyboard.append([InlineKeyboardButton(btn_text, callback_data=callback_data)])
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.message.reply_text(
                f"✅ Found {len(results)} results:",
                reply_markup=reply_markup
            )
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            await update.message.reply_text("❌ Search error")
    
    async def analyze_ticker(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Analyze a ticker"""
        if not context.args:
            await update.message.reply_text("❌ Specify the symbol:\n/analyze AAPL\n/analyze TSLA\n/analyze BTC-USD")
            return
        
        symbol = context.args[0].upper()
        user_id = update.effective_user.id
        
        # Save symbol in user data
        if user_id not in self.user_data:
            self.user_data[user_id] = {'periods': [1, 3, 6], 'interval': '1d'}
        
        await update.message.reply_text(f"📊 Analyzing {symbol}...")
        
        try:
            # Get ticker info
            ticker_info = get_ticker_info(symbol)
            
            if not ticker_info:
                await update.message.reply_text(f"❌ Ticker {symbol} not found")
                return
            
            # Prepare data for formatting
            current_price = ticker_info.get('current_price', 0)
            prev_close = ticker_info.get('previous_close', current_price)
            volume = ticker_info.get('volume', 0)
            market_cap = ticker_info.get('market_cap', 0)
            
            # Format price
            if isinstance(current_price, (int, float)):
                price_str = f"${current_price:.2f}"
            else:
                price_str = 'N/A'
            
            # Calculate daily change
            if isinstance(current_price, (int, float)) and isinstance(prev_close, (int, float)) and prev_close != 0:
                daily_change = ((current_price - prev_close) / prev_close) * 100
                daily_change_str = f"{daily_change:+.2f}%"
            else:
                daily_change_str = "N/A"
            
            # Format volume
            if isinstance(volume, (int, float)):
                volume_str = f"{volume:,.0f}"
            else:
                volume_str = "N/A"
            
            # Format market cap
            if isinstance(market_cap, (int, float)):
                market_cap_str = f"${market_cap:,.0f}"
            else:
                market_cap_str = "N/A"
            
            # Create message
            message = f"""
📈 **{ticker_info.get('name', symbol)} ({symbol})**

💰 **Price:** {price_str}
📊 **Change:** {daily_change_str}
📈 **Volume:** {volume_str}
🏢 **Market Cap:** {market_cap_str}

🏭 **Sector:** {ticker_info.get('sector', 'N/A')}
📊 **P/E Ratio:** {ticker_info.get('pe_ratio', 'N/A')}
🎯 **52W High:** ${ticker_info.get('52w_high', 'N/A')}
📉 **52W Low:** ${ticker_info.get('52w_low', 'N/A')}
            """
            
            await update.message.reply_text(message, parse_mode='Markdown')
            
            # Execute technical analysis
            periods = self.user_data.get(user_id, {}).get('periods', [1, 3, 6])
            
            analyzer = TradingAnalyzer(symbol)
            summary_data = []
            
            for period in sorted(periods):
                df = analyzer.analyze_period(period)
                
                if df is not None and not df.empty:
                    # Calculate performance for THIS period
                    start_idx = 0
                    end_idx = -1
                    
                    price_change = ((df['Close'].iloc[end_idx] / df['Close'].iloc[start_idx]) - 1) * 100
                    current_rsi = df['RSI'].iloc[end_idx] if 'RSI' in df.columns else 50
                    macd_signal = "↑" if df['MACD'].iloc[end_idx] > df['Signal_Line'].iloc[end_idx] else "↓" if 'MACD' in df.columns else "N/A"
                    volume_trend = analyzer._analyze_volume_trend(df)
                    
                    summary_data.append({
                        'period': period,
                        'price': df['Close'].iloc[end_idx],
                        'change': price_change,
                        'rsi': current_rsi,
                        'macd': macd_signal,
                        'volume': volume_trend
                    })
            
            if summary_data:
                # Create summary table
                summary_text = "\n📊 **TECHNICAL ANALYSIS:**\n\n"
                summary_text += "Period | Price | Change | RSI | MACD | Volume\n"
                summary_text += "--------|--------|--------|-----|------|--------\n"
                
                for data in summary_data:
                    change_emoji = "📈" if data['change'] > 0 else "📉"
                    summary_text += (f"{data['period']}M | ${data['price']:.2f} | "
                                   f"{change_emoji}{data['change']:+.1f}% | "
                                   f"{data['rsi']:.1f} | {data['macd']} | "
                                   f"{data['volume']}\n")
                
                await update.message.reply_text(summary_text, parse_mode='Markdown')
                
                # Create keyboard for additional options
                keyboard = [
                    [
                        InlineKeyboardButton("📈 Chart 1M", callback_data=f"chart_{symbol}_1"),
                        InlineKeyboardButton("📊 Chart 3M", callback_data=f"chart_{symbol}_3")
                    ],
                    [
                        InlineKeyboardButton("📈 Chart 6M", callback_data=f"chart_{symbol}_6"),
                        InlineKeyboardButton("📊 Chart 12M", callback_data=f"chart_{symbol}_12")
                    ],
                    [
                        InlineKeyboardButton("🎯 Signals", callback_data=f"signals_{symbol}"),
                        InlineKeyboardButton("📋 Data", callback_data=f"data_{symbol}")
                    ],
                    [
                        InlineKeyboardButton("⚙️ Settings", callback_data="settings")
                    ]
                ]
                
                reply_markup = InlineKeyboardMarkup(keyboard)
                await update.message.reply_text(
                    "Choose an option:",
                    reply_markup=reply_markup
                )
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            await update.message.reply_text(f"❌ Error analyzing {symbol}")
    
    async def show_chart(self, query, context: ContextTypes.DEFAULT_TYPE, symbol: str, period: int):
        """Show ticker chart"""
        try:
            # Answer the callback immediately to avoid timeout
            await query.answer()
            
            await query.edit_message_text(f"📊 Creating chart for {symbol} ({period} months)...")
            
            analyzer = TradingAnalyzer(symbol)
            df = analyzer.analyze_period(period)
            
            if df is None or df.empty:
                await query.edit_message_text("❌ No data available for chart")
                return
            
            # Create matplotlib figure
            fig = analyzer.create_technical_chart(df, str(period))
            
            # Save to bytes
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=120, facecolor='black')
            buf.seek(0)
            
            # Send photo
            await context.bot.send_photo(
                chat_id=query.message.chat_id,
                photo=buf,
                caption=f"📈 {symbol} Chart - {period} months\nPrice, Volume, Moving Averages, MACD, RSI"
            )
            
            # Close the figure to free memory
            plt.close(fig)
            
        except Exception as e:
            logger.error(f"Chart error: {e}")
            await query.edit_message_text("❌ Error creating chart")
    
    async def compare_tickers(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Compare two tickers"""
        if len(context.args) < 2:
            await update.message.reply_text(
                "❌ Specify two symbols to compare:\n/compare AAPL MSFT\n/compare TSLA BTC-USD"
            )
            return
        
        symbol1 = context.args[0].upper()
        symbol2 = context.args[1].upper()
        
        await update.message.reply_text(f"📊 Comparing {symbol1} vs {symbol2}...")
        
        try:
            # Get data for both symbols
            analyzer1 = TradingAnalyzer(symbol1)
            analyzer2 = TradingAnalyzer(symbol2)
            
            # Get ticker info
            ticker_info1 = get_ticker_info(symbol1)
            ticker_info2 = get_ticker_info(symbol2)
            
            if not ticker_info1:
                await update.message.reply_text(f"❌ Ticker {symbol1} not found")
                return
            
            if not ticker_info2:
                await update.message.reply_text(f"❌ Ticker {symbol2} not found")
                return
            
            # Analyze 6 months for comparison
            df1 = analyzer1.analyze_period(6)
            df2 = analyzer2.analyze_period(6)
            
            if df1 is None or df1.empty:
                await update.message.reply_text(f"❌ No data for {symbol1}")
                return
            
            if df2 is None or df2.empty:
                await update.message.reply_text(f"❌ No data for {symbol2}")
                return
            
            # Send comparison info
            price1 = ticker_info1.get('current_price', 0)
            price2 = ticker_info2.get('current_price', 0)
            change1 = ((df1['Close'].iloc[-1] / df1['Close'].iloc[0]) - 1) * 100
            change2 = ((df2['Close'].iloc[-1] / df2['Close'].iloc[0]) - 1) * 100
            
            message = f"""
📊 **COMPARISON: {symbol1} vs {symbol2}**

**{symbol1}:**
• Price: ${price1:.2f}
• 6M Change: {change1:+.2f}%
• Sector: {ticker_info1.get('sector', 'N/A')}
• Market Cap: ${ticker_info1.get('market_cap', 0):,.0f}

**{symbol2}:**
• Price: ${price2:.2f}
• 6M Change: {change2:+.2f}%
• Sector: {ticker_info2.get('sector', 'N/A')}
• Market Cap: ${ticker_info2.get('market_cap', 0):,.0f}

**Difference:** {change1 - change2:+.2f}%
            """
            
            await update.message.reply_text(message, parse_mode='Markdown')
            
            # Create and send comparison chart
            fig = analyzer1.create_comparison_chart(df1, symbol1, df2, symbol2, "6")
            
            # Save to bytes
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=120, facecolor='black')
            buf.seek(0)
            
            # Send photo
            await context.bot.send_photo(
                chat_id=update.message.chat_id,
                photo=buf,
                caption=f"📊 Comparison Chart: {symbol1} vs {symbol2} (6 months)"
            )
            
            # Close the figure to free memory
            plt.close(fig)
            
        except Exception as e:
            logger.error(f"Comparison error: {e}")
            await update.message.reply_text(f"❌ Error comparing {symbol1} and {symbol2}")
    
    async def show_signals(self, query, context: ContextTypes.DEFAULT_TYPE, symbol: str):
        """Show trading signals for the ticker"""
        try:
            # Answer the callback immediately
            await query.answer()
            
            await query.edit_message_text(f"🎯 Analyzing signals for {symbol}...")
            
            analyzer = TradingAnalyzer(symbol)
            df = analyzer.analyze_period(3)  # Use 3 months for signals
            
            if df is None or df.empty:
                await query.edit_message_text("❌ No data available for signals")
                return
            
            latest = df.iloc[-1]
            
            # Analyze signals
            signals = []
            score = 0
            
            # RSI
            rsi = latest.get('RSI', 50)
            if rsi < 30:
                signals.append(("🟢 RSI", "Oversold", 1))
                score += 1
            elif rsi > 70:
                signals.append(("🔴 RSI", "Overbought", -1))
                score -= 1
            else:
                signals.append(("⚪ RSI", "Neutral", 0))
            
            # MACD
            macd = latest.get('MACD', 0)
            signal_line = latest.get('Signal_Line', 0)
            if macd > signal_line:
                signals.append(("🟢 MACD", "Bullish", 1))
                score += 1
            else:
                signals.append(("🔴 MACD", "Bearish", -1))
                score -= 1
            
            # Moving Averages
            sma_20 = latest.get('SMA_20', 0)
            sma_50 = latest.get('SMA_50', 0)
            if sma_20 > sma_50:
                signals.append(("🟢 SMA", "Uptrend", 1))
                score += 1
            else:
                signals.append(("🔴 SMA", "Downtrend", -1))
                score -= 1
            
            # Volume
            volume_trend = analyzer._analyze_volume_trend(df)
            if volume_trend == "HIGH":
                signals.append(("🟢 Volume", "High", 1))
                score += 1
            elif volume_trend == "LOW":
                signals.append(("🔴 Volume", "Low", -1))
                score -= 1
            else:
                signals.append(("⚪ Volume", "Normal", 0))
            
            # Build message
            signals_text = "🎯 **TRADING SIGNALS:**\n\n"
            for icon, text, _ in signals:
                signals_text += f"{icon} {text}\n"
            
            signals_text += f"\n📊 **SCORE:** {score}/4\n\n"
            
            if score >= 3:
                signals_text += "🎯 **STRONG BUY SIGNAL**\nConsider long position"
            elif score >= 1:
                signals_text += "📈 **MODERATE BUY SIGNAL**\nWatch for entry"
            elif score <= -3:
                signals_text += "⚠️ **STRONG SELL SIGNAL**\nConsider short position"
            elif score <= -1:
                signals_text += "📉 **MODERATE SELL SIGNAL**\nBe cautious buying"
            else:
                signals_text += "⚖️ **NEUTRAL SIGNAL**\nWait for clearer signals"
            
            signals_text += "\n\n⚠️ *Note: Automatic signals, not financial advice*"
            
            await query.edit_message_text(signals_text, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Signals error: {e}")
            await query.edit_message_text("❌ Error analyzing signals")
    
    async def show_data(self, query, context: ContextTypes.DEFAULT_TYPE, symbol: str):
        """Show raw ticker data"""
        try:
            # Answer the callback immediately
            await query.answer()
            
            await query.edit_message_text(f"📋 Retrieving data for {symbol}...")
            
            analyzer = TradingAnalyzer(symbol)
            df = analyzer.analyze_period(1)  # 1 month of data
            
            if df is None or df.empty:
                await query.edit_message_text("❌ No data available")
                return
            
            # Prepare last 10 rows
            recent_data = df.tail(10)
            
            data_text = f"📋 **RECENT DATA {symbol}:**\n\n"
            data_text += "Date | Open | High | Low | Close | Volume | RSI\n"
            data_text += "-----|------|------|-----|-------|--------|----\n"
            
            for idx, row in recent_data.iterrows():
                date_str = idx.strftime('%d/%m')
                open_price = row.get('Open', 0)
                high = row.get('High', 0)
                low = row.get('Low', 0)
                close = row.get('Close', 0)
                volume = row.get('Volume', 0)
                rsi = row.get('RSI', 0)
                
                data_text += (f"{date_str} | ${open_price:.2f} | ${high:.2f} | "
                            f"${low:.2f} | ${close:.2f} | "
                            f"{volume:,.0f} | {rsi:.1f}\n")
            
            # Keyboard for download
            keyboard = [[
                InlineKeyboardButton("📥 Download CSV", callback_data=f"download_{symbol}")
            ]]
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(
                data_text,
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )
            
        except Exception as e:
            logger.error(f"Data error: {e}")
            await query.edit_message_text("❌ Error retrieving data")
    
    async def download_data(self, query, context: ContextTypes.DEFAULT_TYPE, symbol: str):
        """Prepare CSV data download"""
        try:
            await query.answer("Preparing download...")
            
            analyzer = TradingAnalyzer(symbol)
            df = analyzer.analyze_period(12)  # All available data
            
            if df is None or df.empty:
                await query.answer("❌ No data available", show_alert=True)
                return
            
            # Convert to CSV
            csv_data = df.to_csv()
            
            # Create file in memory
            csv_file = BytesIO(csv_data.encode())
            csv_file.name = f"{symbol}_data.csv"
            
            # Send file
            await context.bot.send_document(
                chat_id=query.message.chat_id,
                document=csv_file,
                filename=f"{symbol}_historical_data.csv",
                caption=f"📥 Historical data for {symbol}"
            )
            
        except Exception as e:
            logger.error(f"Download error: {e}")
            await query.answer("❌ Download error", show_alert=True)
    
    async def settings_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show settings menu"""
        if isinstance(update, Update) and update.message:
            # Command from message
            message = update.message
            user_id = message.from_user.id
        else:
            # Callback from query
            query = update.callback_query
            await query.answer()
            message = query.message
            user_id = query.from_user.id
        
        if user_id not in self.user_data:
            self.user_data[user_id] = {'periods': [1, 3, 6], 'interval': '1d'}
        
        settings = self.user_data[user_id]
        
        settings_text = f"""
⚙️ **USER SETTINGS**

📅 Analysis periods: {', '.join(map(str, settings.get('periods', [1, 3, 6])))} months
📊 Interval: {settings.get('interval', '1d')}

Use /periods <list> to change periods
Example: /periods 1,3,6,12
        """
        
        keyboard = [
            [
                InlineKeyboardButton("📅 Set Periods", callback_data="set_periods"),
                InlineKeyboardButton("📊 Set Interval", callback_data="set_interval")
            ],
            [
                InlineKeyboardButton("🔄 Reset to Default", callback_data="reset_settings")
            ],
            [
                InlineKeyboardButton("🔙 Back", callback_data="back_settings")
            ]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await message.reply_text(
            settings_text,
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
    
    async def set_periods_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Set analysis periods"""
        if not context.args:
            await update.message.reply_text("❌ Specify periods:\n/periods 1,3,6\n/periods 3,12\n/periods 1,3,6,12")
            return
        
        try:
            periods_str = context.args[0]
            periods = [int(p.strip()) for p in periods_str.split(',')]
            
            # Validate periods
            valid_periods = []
            for p in periods:
                if p > 0 and p <= 60:  # Max 5 years
                    valid_periods.append(p)
                else:
                    await update.message.reply_text(f"⚠️ Period {p} months ignored (must be 1-60)")
            
            if not valid_periods:
                await update.message.reply_text("❌ No valid periods specified")
                return
            
            user_id = update.effective_user.id
            if user_id not in self.user_data:
                self.user_data[user_id] = {}
            
            self.user_data[user_id]['periods'] = sorted(valid_periods)
            
            await update.message.reply_text(
                f"✅ Periods set: {', '.join(map(str, sorted(valid_periods)))} months"
            )
            
        except Exception as e:
            await update.message.reply_text("❌ Wrong format. Example: /periods 1,3,6")
    
    async def quick_analyze(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Quick ticker analysis"""
        if not context.args:
            await update.message.reply_text("❌ Specify symbol: /quick AAPL")
            return
        
        symbol = context.args[0].upper()
        
        try:
            ticker_info = get_ticker_info(symbol)
            
            if not ticker_info:
                await update.message.reply_text(f"❌ {symbol} not found")
                return
            
            # Quick analysis
            analyzer = TradingAnalyzer(symbol)
            df = analyzer.analyze_period(1)  # 1 month
            
            if df is None or df.empty:
                await update.message.reply_text(f"❌ No data for {symbol}")
                return
            
            latest = df.iloc[-1]
            
            # Prepare data
            current_price = ticker_info.get('current_price', latest.get('Close', 0))
            rsi = latest.get('RSI', 0)
            macd = latest.get('MACD', 0)
            
            # Format price
            if isinstance(current_price, (int, float)):
                price_str = f"${current_price:.2f}"
            else:
                price_str = "N/A"
            
            # Determine SMA trend
            sma_20 = latest.get('SMA_20', 0)
            sma_50 = latest.get('SMA_50', 0)
            sma_trend = "↑" if sma_20 > sma_50 else "↓"
            
            # Determine RSI signal
            if rsi < 30:
                signal = "🟢 Buy"
            elif rsi > 70:
                signal = "🔴 Sell"
            else:
                signal = "⚪ Neutral"
            
            message = f"""
⚡ **QUICK ANALYSIS {symbol}**

💰 Price: {price_str}
📊 RSI: {rsi:.1f}
📈 MACD: {"↑" if macd > 0 else "↓"}
📅 SMA Trend: {sma_trend}

🎯 Signal: {signal}
            """
            
            await update.message.reply_text(message, parse_mode='Markdown')
            
            # Send quick chart
            fig = analyzer.create_quick_chart(df)
            
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=120, facecolor='black')
            buf.seek(0)
            
            await context.bot.send_photo(
                chat_id=update.message.chat_id,
                photo=buf,
                caption=f"📈 Quick Chart - {symbol}"
            )
            
            # Close figure
            plt.close(fig)
            
        except Exception as e:
            logger.error(f"Quick analyze error: {e}")
            await update.message.reply_text(f"❌ Quick analysis error for {symbol}")
    
    async def callback_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle callback queries from inline keyboards"""
        query = update.callback_query
        
        try:
            # Answer immediately to prevent timeout
            await query.answer()
            
            data = query.data
            
            if data.startswith("select_"):
                symbol = data.replace("select_", "")
                await self.analyze_ticker_from_callback(query, context, symbol)
            
            elif data.startswith("chart_"):
                parts = data.split("_")
                if len(parts) >= 3:
                    symbol = parts[1]
                    try:
                        period = int(parts[2])
                        await self.show_chart(query, context, symbol, period)
                    except ValueError:
                        await query.answer("❌ Invalid period", show_alert=True)
                else:
                    await query.answer("❌ Invalid chart request", show_alert=True)
            
            elif data.startswith("signals_"):
                symbol = data.replace("signals_", "")
                await self.show_signals(query, context, symbol)
            
            elif data.startswith("data_"):
                symbol = data.replace("data_", "")
                await self.show_data(query, context, symbol)
            
            elif data.startswith("download_"):
                symbol = data.replace("download_", "")
                await self.download_data(query, context, symbol)
            
            elif data == "settings":
                await self.settings_menu(update, context)
            
            elif data == "set_periods":
                await query.edit_message_text(
                    "📅 **Set Analysis Periods:**\n\n"
                    "Use command: /periods <months>\n"
                    "Example: `/periods 1,3,6`\n"
                    "Example: `/periods 1,3,6,12`\n\n"
                    "Maximum period: 60 months (5 years)"
                )
            
            elif data == "set_interval":
                await query.edit_message_text(
                    "📊 **Set Data Interval:**\n\n"
                    "Use command: /interval <value>\n"
                    "Available intervals:\n"
                    "• 1d (daily)\n"
                    "• 1wk (weekly)\n"
                    "• 1mo (monthly)\n\n"
                    "Example: `/interval 1d`"
                )
            
            elif data == "reset_settings":
                user_id = query.from_user.id
                self.user_data[user_id] = {'periods': [1, 3, 6], 'interval': '1d'}
                await query.edit_message_text(
                    "✅ Settings reset to default:\n"
                    "• Periods: 1, 3, 6 months\n"
                    "• Interval: 1d (daily)"
                )
            
            elif data == "back_settings":
                await query.edit_message_text("🔙 Back to main menu")
            
            else:
                await query.answer("⚠️ Unknown command", show_alert=True)
        
        except Exception as e:
            logger.error(f"Callback handler error: {e}")
            try:
                await query.answer("⚠️ Error processing request", show_alert=True)
            except:
                pass
    
    async def analyze_ticker_from_callback(self, query, context: ContextTypes.DEFAULT_TYPE, symbol: str):
        """Analyze ticker from callback"""
        user_id = query.from_user.id
        
        try:
            ticker_info = get_ticker_info(symbol)
            
            if not ticker_info:
                await query.edit_message_text(f"❌ {symbol} not found")
                return
            
            # Prepare data
            current_price = ticker_info.get('current_price', 0)
            market_cap = ticker_info.get('market_cap', 0)
            
            # Format price
            if isinstance(current_price, (int, float)):
                price_str = f"${current_price:.2f}"
            else:
                price_str = "N/A"
            
            # Format market cap
            if isinstance(market_cap, (int, float)):
                market_cap_str = f"${market_cap:,.0f}"
            else:
                market_cap_str = "N/A"
            
            message = f"""
📈 **{ticker_info.get('name', symbol)} ({symbol})**

💰 Price: {price_str}
📊 Sector: {ticker_info.get('sector', 'N/A')}
📈 Market Cap: {market_cap_str}

Click buttons below for detailed analysis:
            """
            
            keyboard = [
                [
                    InlineKeyboardButton("📊 Full Analysis", callback_data=f"full_analyze_{symbol}"),
                    InlineKeyboardButton("📈 Chart 3M", callback_data=f"chart_{symbol}_3")
                ],
                [
                    InlineKeyboardButton("🎯 Signals", callback_data=f"signals_{symbol}"),
                    InlineKeyboardButton("📋 Data", callback_data=f"data_{symbol}")
                ]
            ]
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(message, reply_markup=reply_markup, parse_mode='Markdown')
        
        except Exception as e:
            logger.error(f"Callback analysis error: {e}")
            await query.edit_message_text(f"❌ Error analyzing {symbol}")
    
    async def handle_text_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle text messages (direct analysis)"""
        text = update.message.text.strip().upper()
        
        # If it looks like a ticker symbol
        if 1 <= len(text) <= 10 and all(c.isalpha() or c in '-. ' for c in text):
            # Simulate /analyze command
            context.args = [text]
            await self.analyze_ticker(update, context)
        else:
            await update.message.reply_text(
                f"🔍 To analyze '{text}', use:\n/analyze {text}\n\n"
                f"Or search with:\n/search {text}"
            )
    
    def run(self):
        """Start the bot"""
        application = Application.builder().token(self.token).build()
        
        # Commands
        application.add_handler(CommandHandler("start", self.start))
        application.add_handler(CommandHandler("help", self.help_command))
        application.add_handler(CommandHandler("search", self.search_ticker))
        application.add_handler(CommandHandler("analyze", self.analyze_ticker))
        application.add_handler(CommandHandler("quick", self.quick_analyze))
        application.add_handler(CommandHandler("compare", self.compare_tickers))
        application.add_handler(CommandHandler("settings", self.settings_menu))
        application.add_handler(CommandHandler("periods", self.set_periods_command))
        
        # Add interval command handler
        async def set_interval_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
            """Set data interval"""
            if not context.args:
                await update.message.reply_text(
                    "❌ Specify interval:\n/interval 1d\n/interval 1wk\n/interval 1mo"
                )
                return
            
            interval = context.args[0].lower()
            valid_intervals = ['1d', '1wk', '1mo']
            
            if interval not in valid_intervals:
                await update.message.reply_text(
                    f"❌ Invalid interval. Use: {', '.join(valid_intervals)}"
                )
                return
            
            user_id = update.effective_user.id
            if user_id not in self.user_data:
                self.user_data[user_id] = {}
            
            self.user_data[user_id]['interval'] = interval
            await update.message.reply_text(f"✅ Interval set to: {interval}")
        
        application.add_handler(CommandHandler("interval", set_interval_command))
        
        # Callback query
        application.add_handler(CallbackQueryHandler(self.callback_handler))
        
        # Text messages (for direct analysis)
        application.add_handler(MessageHandler(
            filters.TEXT & ~filters.COMMAND, 
            self.handle_text_message
        ))
        
        logger.info("Bot starting...")
        
        # Start polling
        application.run_polling()


def main():

    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    
    if not TELEGRAM_BOT_TOKEN:
        print("❌ ERROR: Telegram bot token not found!")
        print("Make sure you have a .env file with TELEGRAM_BOT_TOKEN")
        print("\n1. Create a .env file in the same folder")
        print("2. Add: TELEGRAM_BOT_TOKEN=\"your-token-here\"")
        print("3. Restart the bot")
        return
    
    print("🚀 Starting Telegram Trading Bot with health server...")
    
    import threading
    from health_server import start_health_server
    
    health_thread = start_health_server()
    
    bot = TelegramTradingBot(TELEGRAM_BOT_TOKEN)
    bot.run()


if __name__ == "__main__":
    main()