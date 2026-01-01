import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from trading_analyzer import TradingAnalyzer
from utils import search_tickers, get_ticker_info
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

# Page configuration
st.set_page_config(
    page_title="Trading Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme matching bot style
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0a0a0a;
        color: #ffffff;
    }
    
    /* Cards like bot messages */
    .analysis-card {
        background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        border-left: 4px solid #00a8ff;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    .ticker-card {
        background: #1a1a1a;
        border-radius: 8px;
        padding: 15px;
        margin: 8px 0;
        border: 1px solid #333;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .ticker-card:hover {
        background: #2a2a2a;
        border-color: #00a8ff;
        transform: translateY(-2px);
    }
    
    /* Buttons styled like bot */
    .stButton > button {
        background: linear-gradient(135deg, #0088ff 0%, #0055cc 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 500;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #0055cc 0%, #003399 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 85, 204, 0.4);
    }
    
    /* Input fields */
    .stTextInput > div > div > input {
        background-color: #1a1a1a;
        color: white;
        border: 1px solid #444;
        border-radius: 8px;
        padding: 10px;
    }
    
    /* Select boxes */
    .stSelectbox > div > div {
        background-color: #1a1a1a;
        color: white;
        border-radius: 8px;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #1a1a1a;
        border-radius: 8px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #2d2d2d;
        border-radius: 6px;
        padding: 10px 20px;
        color: #aaa;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #0088ff !important;
        color: white !important;
    }
    
    /* Metrics */
    [data-testid="metric-container"] {
        background-color: #1a1a1a;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #333;
    }
    
    /* Dataframe */
    .stDataFrame {
        background-color: #1a1a1a;
        border-radius: 8px;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #1a1a1a;
        border-radius: 8px;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f0f 0%, #1a1a1a 100%);
        border-right: 1px solid #333;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff;
    }
    
    /* Code blocks */
    code {
        background-color: #1a1a1a;
        color: #00ff9d;
        padding: 2px 6px;
        border-radius: 4px;
    }
    
    /* Divider */
    hr {
        border-color: #333;
        margin: 20px 0;
    }
    
    /* Loading spinner */
    .stSpinner > div {
        border-color: #0088ff;
    }
    
    /* Success/Error messages */
    .stAlert {
        background-color: #1a1a1a;
        border: 1px solid #333;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

def plot_matplotlib_chart(df, symbol, period):
    """Generate matplotlib chart in bot style"""
    # Create figure with black background
    fig = plt.figure(figsize=(14, 8), facecolor='black')
    gs = plt.GridSpec(3, 1, hspace=0.3, height_ratios=[3, 1, 1])
    
    # Colors matching bot
    WHITE = '#FFFFFF'
    ORANGE = '#FF6B00'
    BLUE = '#00A8FF'
    GREEN = '#00FF9D'
    PURPLE = '#9D4EDD'
    RED = '#FF0033'
    
    # 1. Price Chart
    ax1 = plt.subplot(gs[0])
    ax1.set_facecolor('black')
    
    # Plot price
    ax1.plot(df.index, df['Close'], color=WHITE, linewidth=2.5, label='Price')
    
    # Plot moving averages
    if 'SMA_20' in df.columns:
        ax1.plot(df.index, df['SMA_20'], color=ORANGE, linewidth=1.8, label='SMA 20')
    
    if 'SMA_50' in df.columns:
        ax1.plot(df.index, df['SMA_50'], color=BLUE, linewidth=1.8, label='SMA 50')
    
    # Fill above SMA20
    if 'SMA_20' in df.columns:
        above_sma20 = df['Close'] > df['SMA_20']
        ax1.fill_between(df.index, df['SMA_20'], df['Close'], 
                         where=above_sma20, color=GREEN, alpha=0.2, label='Above SMA20')
    
    ax1.set_ylabel('Price ($)', color=WHITE, fontsize=11)
    ax1.tick_params(axis='y', colors=WHITE)
    ax1.grid(True, alpha=0.15, color='gray', linestyle='--', linewidth=0.5)
    ax1.legend(loc='upper left', facecolor='black', labelcolor='white', fontsize=9)
    
    # 2. MACD
    ax2 = plt.subplot(gs[1], sharex=ax1)
    ax2.set_facecolor('black')
    
    ax2.plot(df.index, df['MACD'], color=GREEN, linewidth=1.5, label='MACD')
    ax2.plot(df.index, df['Signal_Line'], color=RED, linewidth=1.5, label='Signal', linestyle='--')
    
    # MACD histogram
    macd_colors = [GREEN if val >= 0 else RED for val in df['MACD_Histogram']]
    ax2.bar(df.index, df['MACD_Histogram'], color=macd_colors, alpha=0.6, width=0.8)
    ax2.axhline(y=0, color=WHITE, linewidth=0.8, alpha=0.6)
    
    ax2.set_ylabel('MACD', color=WHITE, fontsize=11)
    ax2.tick_params(colors=WHITE)
    ax2.grid(True, alpha=0.15, color='gray', linestyle='--', linewidth=0.5)
    ax2.legend(loc='upper left', facecolor='black', labelcolor='white', fontsize=8)
    
    # 3. RSI
    ax3 = plt.subplot(gs[2], sharex=ax1)
    ax3.set_facecolor('black')
    
    ax3.plot(df.index, df['RSI'], color=PURPLE, linewidth=2.0)
    ax3.axhline(y=70, color=RED, linestyle='--', linewidth=1.3, alpha=0.8)
    ax3.axhline(y=30, color=GREEN, linestyle='--', linewidth=1.3, alpha=0.8)
    
    # Fill zones
    ax3.fill_between(df.index, 30, 70, color='gray', alpha=0.1)
    ax3.fill_between(df.index, 70, 100, color=RED, alpha=0.1)
    ax3.fill_between(df.index, 0, 30, color=GREEN, alpha=0.1)
    
    ax3.set_ylabel('RSI', color=WHITE, fontsize=11)
    ax3.set_ylim(0, 100)
    ax3.tick_params(colors=WHITE)
    ax3.grid(True, alpha=0.15, color='gray', linestyle='--', linewidth=0.5)
    
    # Format x-axis
    ax3.xaxis.set_tick_params(rotation=45)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right', color=WHITE)
    
    # Hide x ticks for upper plots
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    
    # Title
    price_change = ((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1) * 100
    current_price = df['Close'].iloc[-1]
    title_text = f'{symbol} - {period} months (Price: ${current_price:.2f} | Change: {price_change:+.2f}%)'
    ax1.set_title(title_text, color=WHITE, fontsize=13, fontweight='bold', pad=15)
    
    plt.tight_layout()
    return fig

def display_ticker_analysis(symbol, period):
    """Display complete ticker analysis like bot"""
    analyzer = TradingAnalyzer(symbol)
    df = analyzer.analyze_period(period)
    
    if df is None or df.empty:
        st.error(f"❌ No data available for {symbol}")
        return
    
    # Get ticker info
    ticker_info = get_ticker_info(symbol)
    
    # Header with ticker info
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Price", f"${ticker_info.get('current_price', 0):.2f}")
    
    with col2:
        prev_close = ticker_info.get('previous_close')
        current = ticker_info.get('current_price')
        if isinstance(prev_close, (int, float)) and isinstance(current, (int, float)):
            change = ((current - prev_close) / prev_close) * 100
            st.metric("Daily Change", f"{change:+.2f}%", delta=f"{change:+.2f}%")
    
    with col3:
        st.metric("Volume", f"{ticker_info.get('volume', 0):,}")
    
    with col4:
        st.metric("Market Cap", f"${ticker_info.get('market_cap', 0):,.0f}")
    
    # Tabs for different analysis
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Chart", 
        "📊 Technical", 
        "🎯 Signals", 
        "📋 Data", 
        "ℹ️ Info"
    ])
    
    with tab1:
        # Matplotlib chart in bot style
        st.markdown("### 📈 Technical Chart")
        fig = plot_matplotlib_chart(df, symbol, period)
        
        # Convert to base64 for display
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=120, facecolor='black', bbox_inches='tight')
        buf.seek(0)
        
        # Display image
        st.image(buf, caption=f"{symbol} Technical Analysis - {period} months", use_column_width=True)
        plt.close(fig)
        
        # Download button
        st.download_button(
            label="📥 Download Chart",
            data=buf,
            file_name=f"{symbol}_chart_{period}m.png",
            mime="image/png",
            use_container_width=True
        )
    
    with tab2:
        st.markdown("### 📊 Technical Analysis")
        
        # Technical indicators
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("RSI", f"{df['RSI'].iloc[-1]:.1f}")
        
        with col2:
            macd_val = df['MACD'].iloc[-1]
            signal_val = df['Signal_Line'].iloc[-1]
            st.metric("MACD", f"{macd_val:.4f}", 
                     delta="Bullish" if macd_val > signal_val else "Bearish")
        
        with col3:
            if 'SMA_20' in df.columns and 'SMA_50' in df.columns:
                sma_trend = "↑" if df['SMA_20'].iloc[-1] > df['SMA_50'].iloc[-1] else "↓"
                st.metric("SMA Trend", sma_trend)
        
        with col4:
            volume_trend = analyzer._analyze_volume_trend(df)
            st.metric("Volume", volume_trend)
        
        # Detailed technical table
        st.markdown("#### Technical Indicators History")
        tech_df = df[['Close', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'Signal_Line']].tail(20)
        st.dataframe(tech_df.style.format({
            'Close': '${:.2f}',
            'SMA_20': '${:.2f}',
            'SMA_50': '${:.2f}',
            'RSI': '{:.1f}',
            'MACD': '{:.4f}',
            'Signal_Line': '{:.4f}'
        }), use_container_width=True)
    
    with tab3:
        st.markdown("### 🎯 Trading Signals")
        
        latest = df.iloc[-1]
        signals = []
        score = 0
        
        # RSI signal
        if latest['RSI'] < 30:
            signals.append(("🟢 RSI", "Oversold", 1))
            score += 1
        elif latest['RSI'] > 70:
            signals.append(("🔴 RSI", "Overbought", -1))
            score -= 1
        else:
            signals.append(("⚪ RSI", "Neutral", 0))
        
        # MACD signal
        if latest['MACD'] > latest['Signal_Line']:
            signals.append(("🟢 MACD", "Bullish", 1))
            score += 1
        else:
            signals.append(("🔴 MACD", "Bearish", -1))
            score -= 1
        
        # Moving Averages signal
        if 'SMA_20' in latest and 'SMA_50' in latest:
            if latest['SMA_20'] > latest['SMA_50']:
                signals.append(("🟢 SMA", "Uptrend", 1))
                score += 1
            else:
                signals.append(("🔴 SMA", "Downtrend", -1))
                score -= 1
        
        # Volume signal
        volume_trend = analyzer._analyze_volume_trend(df)
        if volume_trend == "HIGH":
            signals.append(("🟢 Volume", "High", 1))
            score += 1
        elif volume_trend == "LOW":
            signals.append(("🔴 Volume", "Low", -1))
            score -= 1
        else:
            signals.append(("⚪ Volume", "Normal", 0))
        
        # Display signals
        cols = st.columns(len(signals))
        for idx, (icon, text, _) in enumerate(signals):
            with cols[idx]:
                st.markdown(f"""
                <div style="text-align: center; padding: 15px; border-radius: 10px; 
                            background: #1a1a1a; border: 1px solid #333;">
                    <div style="font-size: 28px; margin-bottom: 5px;">{icon}</div>
                    <div style="font-weight: 500;">{text}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Overall signal
        st.markdown("---")
        st.markdown(f"### 📊 Signal Score: **{score}/4**")
        
        if score >= 3:
            st.success("""
            🎯 **STRONG BUY SIGNAL**
            
            Multiple indicators suggest bullish momentum. Consider long position with proper risk management.
            """)
        elif score >= 1:
            st.info("""
            📈 **MODERATE BUY SIGNAL**
            
            Some bullish indicators present. Watch for entry opportunities with stop loss.
            """)
        elif score <= -3:
            st.error("""
            ⚠️ **STRONG SELL SIGNAL**
            
            Multiple indicators suggest bearish momentum. Consider short position or exiting longs.
            """)
        elif score <= -1:
            st.warning("""
            📉 **MODERATE SELL SIGNAL**
            
            Some bearish indicators present. Be cautious buying and consider taking profits.
            """)
        else:
            st.info("""
            ⚖️ **NEUTRAL SIGNAL**
            
            Mixed or unclear signals. Wait for clearer trend confirmation before entering positions.
            """)
    
    with tab4:
        st.markdown("### 📋 Historical Data")
        
        # Data filters
        col1, col2 = st.columns(2)
        with col1:
            rows = st.slider("Show last N days", 10, 100, 20)
        
        with col2:
            show_cols = st.multiselect(
                "Columns to display",
                options=df.columns.tolist(),
                default=['Open', 'High', 'Low', 'Close', 'Volume', 'RSI']
            )
        
        # Display data
        st.dataframe(df[show_cols].tail(rows), use_container_width=True)
        
        # Download buttons
        csv = df.to_csv()
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📥 Download Full Data (CSV)",
                data=csv,
                file_name=f"{symbol}_full_data.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            st.download_button(
                label="📥 Download Latest Data (CSV)",
                data=df.tail(rows).to_csv(),
                file_name=f"{symbol}_latest_{rows}d.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with tab5:
        st.markdown("### ℹ️ Company Information")
        
        if ticker_info:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                **Name:** {ticker_info.get('name', 'N/A')}
                
                **Sector:** {ticker_info.get('sector', 'N/A')}
                
                **Industry:** {ticker_info.get('industry', 'N/A')}
                
                **Country:** {ticker_info.get('country', 'N/A')}
                
                **Currency:** {ticker_info.get('currency', 'USD')}
                """)
            
            with col2:
                st.markdown(f"""
                **P/E Ratio:** {ticker_info.get('pe_ratio', 'N/A')}
                
                **Dividend Yield:** {ticker_info.get('dividend_yield', 'N/A')}
                
                **Beta:** {ticker_info.get('beta', 'N/A')}
                
                **52W High:** ${ticker_info.get('52w_high', 'N/A')}
                
                **52W Low:** ${ticker_info.get('52w_low', 'N/A')}
                """)
        else:
            st.warning("Company information not available")

def main():
    # Title and header
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("https://img.icons8.com/color/96/000000/stock-share.png", width=80)
    with col2:
        st.title("📊 Trading Analysis Dashboard")
        st.markdown("Professional trading analysis platform - Like your Telegram bot, but on web")
    
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🔍 Search & Analysis")
        
        # Search mode
        search_mode = st.radio(
            "Search Mode:",
            ["🔤 Search by Name", "⚡ Direct Symbol"],
            label_visibility="collapsed"
        )
        
        selected_symbol = None
        
        if search_mode == "🔤 Search by Name":
            query = st.text_input("Search company or symbol:", 
                                  placeholder="e.g., Apple, AAPL, Tesla...")
            
            if query and len(query) >= 2:
                with st.spinner("Searching..."):
                    results = search_tickers(query, limit=10)
                
                if results:
                    st.success(f"Found {len(results)} results")
                    
                    for result in results[:8]:  # Limit to 8
                        if st.button(
                            f"**{result['symbol']}** - {result['name'][:30]}...",
                            key=f"search_{result['symbol']}",
                            use_container_width=True
                        ):
                            selected_symbol = result['symbol']
                            st.session_state.selected_symbol = selected_symbol
                            st.rerun()
                elif query:
                    st.warning("No results found")
        
        else:  # Direct Symbol
            default_symbol = st.session_state.get('selected_symbol', 'AAPL')
            symbol = st.text_input("Enter symbol:", 
                                   value=default_symbol,
                                   placeholder="e.g., AAPL, TSLA, BTC-USD").upper()
            
            if symbol:
                selected_symbol = symbol
                st.session_state.selected_symbol = symbol
        
        st.markdown("---")
        st.markdown("## ⚙️ Analysis Settings")
        
        # Period selection
        periods = st.multiselect(
            "Analysis Periods (months):",
            options=[1, 3, 6, 12],
            default=[3, 6],
            format_func=lambda x: f"{x}M"
        )
        
        # Interval
        interval = st.selectbox(
            "Data Interval:",
            options=["1d", "1wk", "1mo"],
            index=0
        )
        
        st.markdown("---")
        st.markdown("## 🎯 Quick Actions")
        
        # Quick analysis buttons
        if st.button("📊 Quick Analysis AAPL", use_container_width=True):
            selected_symbol = "AAPL"
            st.session_state.selected_symbol = "AAPL"
            st.rerun()
        
        if st.button("⚡ Quick Analysis TSLA", use_container_width=True):
            selected_symbol = "TSLA"
            st.session_state.selected_symbol = "TSLA"
            st.rerun()
        
        if st.button("📈 Quick Analysis MSFT", use_container_width=True):
            selected_symbol = "MSFT"
            st.session_state.selected_symbol = "MSFT"
            st.rerun()
        
        st.markdown("---")
        st.markdown("## 📱 Connect")
        
        st.markdown("""
        **Bot Commands:**
        - `/search` - Find tickers
        - `/analyze` - Technical analysis
        - `/compare` - Compare stocks
        - `/quick` - Quick analysis
        """)
    
    # Main content area
    if selected_symbol or st.session_state.get('selected_symbol'):
        current_symbol = selected_symbol if selected_symbol else st.session_state.selected_symbol
        
        # Header with symbol info
        st.markdown(f"""
        <div class="analysis-card">
            <h2 style="margin: 0; color: #00a8ff;">{current_symbol}</h2>
            <p style="color: #aaa; margin: 5px 0 0 0;">Technical Analysis Dashboard</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Period selector
        if 'periods' in locals() and periods:
            col1, col2, col3, col4 = st.columns(4)
            for idx, period in enumerate(sorted(periods)):
                with [col1, col2, col3, col4][idx % 4]:
                    if st.button(f"{period} Months", key=f"period_{period}", use_container_width=True):
                        st.session_state.current_period = period
                        st.rerun()
            
            current_period = st.session_state.get('current_period', sorted(periods)[0])
            
            # Display analysis for selected period
            display_ticker_analysis(current_symbol, current_period)
        else:
            st.warning("Please select analysis periods in the sidebar")
    
    else:
        # Welcome screen
        st.markdown("""
        <div style="text-align: center; padding: 60px 20px;">
            <h1 style="font-size: 48px; margin-bottom: 20px;">🤖</h1>
            <h2>Welcome to Trading Analysis Dashboard</h2>
            <p style="color: #aaa; font-size: 18px; max-width: 600px; margin: 20px auto;">
                Professional trading analysis platform with technical indicators, 
                charting tools, and real-time market data.
            </p>
            
            <div style="margin-top: 40px; display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px;">
                <div style="background: #1a1a1a; padding: 20px; border-radius: 10px; border-left: 4px solid #00ff9d;">
                    <h3 style="color: #00ff9d;">📈 Technical Analysis</h3>
                    <p>RSI, MACD, Moving Averages, Volume analysis</p>
                </div>
                
                <div style="background: #1a1a1a; padding: 20px; border-radius: 10px; border-left: 4px solid #00a8ff;">
                    <h3 style="color: #00a8ff;">📊 Professional Charts</h3>
                    <p>Interactive charts with multiple timeframes</p>
                </div>
                
                <div style="background: #1a1a1a; padding: 20px; border-radius: 10px; border-left: 4px solid #ff6b00;">
                    <h3 style="color: #ff6b00;">🎯 Trading Signals</h3>
                    <p>Automated buy/sell signals with scoring</p>
                </div>
                
                <div style="background: #1a1a1a; padding: 20px; border-radius: 10px; border-left: 4px solid #9d4edd;">
                    <h3 style="color: #9d4edd;">📋 Data Export</h3>
                    <p>Download CSV data for further analysis</p>
                </div>
            </div>
            
            <div style="margin-top: 40px;">
                <h3>How to start:</h3>
                <ol style="text-align: left; max-width: 500px; margin: 20px auto; color: #aaa;">
                    <li>Search for a ticker in the sidebar</li>
                    <li>Select analysis periods</li>
                    <li>Explore different analysis tabs</li>
                    <li>Download data or charts</li>
                </ol>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Popular tickers grid
        st.markdown("### 🎯 Popular Tickers")
        
        popular_tickers = [
            ("AAPL", "Apple Inc.", "#333"),
            ("MSFT", "Microsoft", "#0078d7"),
            ("GOOGL", "Alphabet", "#4285f4"),
            ("TSLA", "Tesla", "#e31937"),
            ("AMZN", "Amazon", "#ff9900"),
            ("META", "Meta Platforms", "#1877f2"),
            ("NVDA", "NVIDIA", "#76b900"),
            ("JPM", "JPMorgan Chase", "#1e5cb3"),
        ]
        
        cols = st.columns(4)
        for idx, (symbol, name, color) in enumerate(popular_tickers):
            with cols[idx % 4]:
                if st.button(
                    f"**{symbol}**\n{name}",
                    key=f"pop_{symbol}",
                    use_container_width=True,
                    type="secondary"
                ):
                    st.session_state.selected_symbol = symbol
                    st.rerun()

if __name__ == "__main__":
    # Initialize session state
    if 'selected_symbol' not in st.session_state:
        st.session_state.selected_symbol = None
    if 'current_period' not in st.session_state:
        st.session_state.current_period = 3
    
    main()