import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from utils import get_stock_data, calculate_technical_indicators
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingAnalyzer:
    def __init__(self, symbol: str):
        self.symbol = symbol.upper()
        logger.info(f"Initialized analyzer for {self.symbol}")
    
    def analyze_period(self, period_months: int = 3):
        """Analyze for a specific period"""
        try:
            logger.info(f"Analyzing {self.symbol} for {period_months} months")
            
            # Get data for the SPECIFIC period
            df = get_stock_data(self.symbol, period_months)
            
            if df is None or df.empty:
                logger.error(f"No data returned for {self.symbol}, period {period_months} months")
                return None
            
            logger.info(f"Data shape: {df.shape}, columns: {df.columns.tolist()}")
            
            # Calculate indicators
            df = calculate_technical_indicators(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error analyzing {self.symbol}: {e}")
            return None
    
    def _analyze_volume_trend(self, df: pd.DataFrame) -> str:
        """Analyze volume trend"""
        if len(df) < 10:
            return "N/A"
        
        try:
            avg_volume = df['Volume'].mean()
            recent_volume = df['Volume'].iloc[-5:].mean()
            
            if recent_volume > avg_volume * 1.2:
                return "HIGH"
            elif recent_volume < avg_volume * 0.8:
                return "LOW"
            return "NORMAL"
        except:
            return "N/A"
    
    def _calculate_price_change(self, df: pd.DataFrame) -> float:
        """Calculate price change percentage"""
        if len(df) < 2:
            return 0
        
        try:
            start_price = df['Close'].iloc[0]
            end_price = df['Close'].iloc[-1]
            return ((end_price / start_price) - 1) * 100
        except:
            return 0
    
    def _get_macd_signal(self, df: pd.DataFrame) -> str:
        """Determine MACD signal"""
        if 'MACD' not in df.columns or 'Signal_Line' not in df.columns:
            return "N/A"
        
        try:
            if df['MACD'].iloc[-1] > df['Signal_Line'].iloc[-1]:
                return "↑"  # BULLISH
            elif df['MACD'].iloc[-1] < df['Signal_Line'].iloc[-1]:
                return "↓"  # BEARISH
            return "→"  # NEUTRAL
        except:
            return "N/A"
    
    def create_technical_chart(self, df: pd.DataFrame, period: str):
        """Create professional technical chart in black style with thin lines"""
        logger.info(f"Creating chart for {self.symbol}, period {period} months")
        
        try:
            # COLORS - Thin lines
            AQUA_MARINE = '#00FF9D'  # Aqua marine for price line
            LIGHT_ORANGE = '#FF8C42'  # SMA 20
            LIGHT_BLUE = '#4CC9F0'    # SMA 50
            PURPLE = '#9D4EDD'       # RSI
            RED = '#FF0033'          # Negative MACD
            WHITE = '#FFFFFF'        # Text
            
            # Figure with black background
            fig = plt.figure(figsize=(14, 10), facecolor='black')
            logger.info("Figure created")
            
            # GridSpec with 4 rows
            gs = GridSpec(4, 1, figure=fig, height_ratios=[3, 1, 1, 1], hspace=0.12)
            
            # ========== 1. PRICE CHART (THIN LINE) ==========
            ax_price = fig.add_subplot(gs[0])
            ax_price.set_facecolor('black')
            
            # Plot price with AQUA MARINE thin line
            ax_price.plot(df.index, df['Close'], color=AQUA_MARINE, linewidth=1.5, 
                         label='Price', alpha=0.95, zorder=5)
            
            # Moving averages with thin lines
            if 'SMA_20' in df.columns and not pd.isna(df['SMA_20']).all():
                ax_price.plot(df.index, df['SMA_20'], color=LIGHT_ORANGE, linewidth=1.0,
                             label='SMA 20', alpha=0.7, zorder=4, linestyle='--')
            
            if 'SMA_50' in df.columns and not pd.isna(df['SMA_50']).all():
                ax_price.plot(df.index, df['SMA_50'], color=LIGHT_BLUE, linewidth=1.0,
                             label='SMA 50', alpha=0.7, zorder=3, linestyle='-.')
            
            # Highlight area above SMA20 (transparent)
            if 'SMA_20' in df.columns and not pd.isna(df['SMA_20']).all():
                above_sma20 = df['Close'] > df['SMA_20']
                ax_price.fill_between(df.index, df['SMA_20'], df['Close'],
                                     where=above_sma20, color=AQUA_MARINE, alpha=0.1,
                                     label='Above SMA20', zorder=2)
            
            # Title
            price_change = self._calculate_price_change(df)
            current_price = df['Close'].iloc[-1]
            start_price = df['Close'].iloc[0]
            title_text = (f'{self.symbol} - {period} months analysis\n'
                         f'Current: ${current_price:.2f} | '
                         f'Change: {price_change:+.2f}% | '
                         f'From: ${start_price:.2f}')
            ax_price.set_title(title_text, fontsize=13, color=WHITE, 
                              fontweight='bold', pad=12, loc='left')
            
            # Labels and ticks
            ax_price.set_ylabel('Price ($)', color=WHITE, fontsize=10)
            ax_price.tick_params(axis='y', colors=WHITE, labelsize=8)
            ax_price.tick_params(axis='x', colors=WHITE, labelsize=8)
            
            # Thin grid
            ax_price.grid(True, alpha=0.1, color='gray', linestyle=':', linewidth=0.3)
            
            # Legend
            ax_price.legend(loc='upper left', facecolor='#111111', edgecolor=WHITE,
                           labelcolor=WHITE, fontsize=8, framealpha=0.9)
            
            # ========== 2. MACD CHART (THIN LINE) ==========
            ax_macd = fig.add_subplot(gs[1], sharex=ax_price)
            ax_macd.set_facecolor('black')
            
            # Check if MACD columns exist
            if 'MACD' in df.columns and 'Signal_Line' in df.columns:
                # MACD thin lines
                ax_macd.plot(df.index, df['MACD'], color=AQUA_MARINE, linewidth=1.2,
                            label='MACD', alpha=0.8, zorder=3)
                ax_macd.plot(df.index, df['Signal_Line'], color=RED, linewidth=1.2,
                            label='Signal', alpha=0.8, linestyle='--', zorder=2)
                
                # MACD thin histogram
                if 'MACD_Histogram' in df.columns:
                    macd_colors = [AQUA_MARINE if val >= 0 else RED for val in df['MACD_Histogram']]
                    ax_macd.bar(df.index, df['MACD_Histogram'], color=macd_colors,
                               alpha=0.5, width=0.6, edgecolor='none', linewidth=0.5, zorder=1)
            
            # Zero thin line
            ax_macd.axhline(y=0, color=WHITE, linestyle='-', linewidth=0.5, alpha=0.4)
            
            # Labels
            ax_macd.set_ylabel('MACD Signal', color=WHITE, fontsize=10)
            ax_macd.tick_params(colors=WHITE, labelsize=8)
            ax_macd.grid(True, alpha=0.1, color='gray', linestyle=':', linewidth=0.3)
            ax_macd.legend(loc='upper left', facecolor='#111111', edgecolor=AQUA_MARINE,
                          labelcolor=WHITE, fontsize=7, framealpha=0.9)
            
            # ========== 3. RSI CHART (THIN LINE) ==========
            ax_rsi = fig.add_subplot(gs[2], sharex=ax_price)
            ax_rsi.set_facecolor('black')
            
            # RSI thin line
            if 'RSI' in df.columns:
                ax_rsi.plot(df.index, df['RSI'], color=PURPLE, linewidth=1.2, alpha=0.8)
            
            # RSI thin levels
            ax_rsi.axhline(y=70, color=RED, linestyle='--', linewidth=0.8, alpha=0.6)
            ax_rsi.axhline(y=30, color=AQUA_MARINE, linestyle='--', linewidth=0.8, alpha=0.6)
            
            # Zone colorate (transparent)
            ax_rsi.fill_between(df.index, 30, 70, color='gray', alpha=0.05)
            ax_rsi.fill_between(df.index, 70, 100, color=RED, alpha=0.05)
            ax_rsi.fill_between(df.index, 0, 30, color=AQUA_MARINE, alpha=0.05)
            
            # Labels
            ax_rsi.text(0.02, 1.02, 'Overbought (70)', transform=ax_rsi.transAxes,
                       color=RED, fontsize=7, verticalalignment='bottom')
            ax_rsi.text(0.02, -0.02, 'Oversold (30)', transform=ax_rsi.transAxes,
                       color=AQUA_MARINE, fontsize=7, verticalalignment='top')
            
            # RSI current value
            if 'RSI' in df.columns:
                last_rsi = df['RSI'].iloc[-1]
                ax_rsi.text(0.98, 0.95, f'RSI: {last_rsi:.1f}', transform=ax_rsi.transAxes,
                           color=WHITE, fontsize=8, fontweight='bold', ha='right',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.8))
            
            ax_rsi.set_ylabel('RSI', color=WHITE, fontsize=10)
            ax_rsi.set_ylim(0, 100)
            ax_rsi.tick_params(colors=WHITE, labelsize=8)
            ax_rsi.grid(True, alpha=0.1, color='gray', linestyle=':', linewidth=0.3)
            
            # ========== 4. VOLUME CHART (THIN BARS) ==========
            ax_volume = fig.add_subplot(gs[3], sharex=ax_price)
            ax_volume.set_facecolor('black')
            
            # Volume bars colorate con linee sottili
            volume_colors = []
            if 'Volume' in df.columns:
                for i in range(len(df)):
                    if i == 0:
                        volume_colors.append(AQUA_MARINE)
                    else:
                        volume_colors.append(AQUA_MARINE if df['Close'].iloc[i] >= df['Close'].iloc[i-1] else RED)
                
                ax_volume.bar(df.index, df['Volume'], color=volume_colors,
                             alpha=0.6, width=0.6, edgecolor='none', linewidth=0.3)
            
            # Labels
            ax_volume.set_ylabel('Volume', color=WHITE, fontsize=10)
            ax_volume.tick_params(colors=WHITE, labelsize=8)
            ax_volume.grid(True, alpha=0.1, color='gray', linestyle=':', linewidth=0.3, axis='y')
            
            # Format dates
            date_format = mdates.DateFormatter('%Y-%m')
            ax_volume.xaxis.set_major_formatter(date_format)
            
            # Show only some dates for readability
            if len(df) > 120:
                ax_volume.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            else:
                ax_volume.xaxis.set_major_locator(mdates.MonthLocator())
            
            plt.setp(ax_volume.xaxis.get_majorticklabels(), rotation=45, ha='right',
                    color=WHITE, fontsize=8)
            
            # Hide x ticks in upper charts
            plt.setp(ax_price.get_xticklabels(), visible=False)
            plt.setp(ax_macd.get_xticklabels(), visible=False)
            plt.setp(ax_rsi.get_xticklabels(), visible=False)
            
            # Final alignment
            plt.subplots_adjust(left=0.08, right=0.95, top=0.94, bottom=0.1, hspace=0.12)
            
            logger.info("Chart creation successful")
            return fig
            
        except Exception as e:
            logger.error(f"Error creating chart: {e}")
            # Create a simple fallback chart
            return self._create_fallback_chart(df, period)
    
    def _create_fallback_chart(self, df: pd.DataFrame, period: str):
        """Create simple fallback chart if main chart fails"""
        try:
            logger.info("Creating fallback chart")
            fig, ax = plt.subplots(figsize=(10, 6), facecolor='black')
            ax.set_facecolor('black')
            
            # Plot just the price
            ax.plot(df.index, df['Close'], color='#00FF9D', linewidth=2)
            ax.set_title(f'{self.symbol} - {period} months (Simple Chart)', color='white')
            ax.set_ylabel('Price ($)', color='white')
            ax.tick_params(colors='white')
            ax.grid(True, alpha=0.2, color='gray')
            
            plt.tight_layout()
            logger.info("Fallback chart created successfully")
            return fig
        except Exception as e:
            logger.error(f"Even fallback chart failed: {e}")
            # Create minimal error chart
            fig = plt.figure(figsize=(8, 4), facecolor='black')
            ax = fig.add_subplot(111)
            ax.set_facecolor('black')
            ax.text(0.5, 0.5, f'Chart Error\n{self.symbol}\nTry another period', 
                   color='red', ha='center', va='center', fontsize=14)
            ax.set_xticks([])
            ax.set_yticks([])
            return fig
    
    def create_comparison_chart(self, df1: pd.DataFrame, symbol1: str, 
                               df2: pd.DataFrame, symbol2: str, period: str):
        """Create comparison chart between two symbols"""
        logger.info(f"Creating comparison chart: {symbol1} vs {symbol2}")
        
        AQUA_MARINE = '#00FF9D'
        LIGHT_BLUE = '#4CC9F0'
        LIGHT_ORANGE = '#FF8C42'
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
        fig.patch.set_facecolor('black')
        
        # Normalize prices to percentage change
        norm_price1 = (df1['Close'] / df1['Close'].iloc[0]) * 100
        norm_price2 = (df2['Close'] / df2['Close'].iloc[0]) * 100
        
        # Plot normalized prices with thin lines
        ax1.plot(df1.index, norm_price1, color=AQUA_MARINE, linewidth=1.5, 
                label=f'{symbol1}', alpha=0.9)
        ax1.plot(df2.index, norm_price2, color=LIGHT_BLUE, linewidth=1.5, 
                label=f'{symbol2}', alpha=0.9)
        
        ax1.set_title(f'{symbol1} vs {symbol2} - Performance Comparison ({period} months)',
                     color='white', fontsize=14, fontweight='bold', pad=15)
        ax1.set_ylabel('Performance (%)', color='white', fontsize=11)
        ax1.legend(loc='upper left', facecolor='#111111', edgecolor='white',
                  labelcolor='white', fontsize=10)
        ax1.grid(True, alpha=0.15, color='gray', linestyle=':', linewidth=0.3)
        ax1.tick_params(colors='white', labelsize=9)
        ax1.set_facecolor('black')
        
        # Calculate and plot price ratio
        common_idx = df1.index.intersection(df2.index)
        if len(common_idx) > 0:
            ratio_series = (df1.loc[common_idx, 'Close'] / 
                          df2.loc[common_idx, 'Close'])
        else:
            min_len = min(len(df1), len(df2))
            ratio_series = (df1['Close'].iloc[:min_len].values / 
                          df2['Close'].iloc[:min_len].values)
        
        # Ratio thin line
        ax2.plot(df1.index[:len(ratio_series)], ratio_series, 
                color=LIGHT_ORANGE, linewidth=1.5)
        ax2.axhline(y=1.0, color='white', linestyle='--', linewidth=0.8, alpha=0.5)
        
        # Fill between ratio and 1 (transparent)
        ax2.fill_between(df1.index[:len(ratio_series)], ratio_series, 1, 
                        where=ratio_series >= 1,
                        facecolor=AQUA_MARINE, alpha=0.1)
        ax2.fill_between(df1.index[:len(ratio_series)], ratio_series, 1,
                        where=ratio_series < 1,
                        facecolor='red', alpha=0.1)
        
        ax2.set_xlabel('Date', color='white', fontsize=11)
        ax2.set_ylabel(f'{symbol1}/{symbol2} Ratio', color='white', fontsize=11)
        ax2.grid(True, alpha=0.15, color='gray', linestyle=':', linewidth=0.3)
        ax2.tick_params(colors='white', labelsize=9)
        ax2.set_facecolor('black')
        
        # Format x-axis dates
        date_format = mdates.DateFormatter('%b %d')
        ax2.xaxis.set_major_formatter(date_format)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right', 
                color='white', fontsize=9)
        
        # Add performance summary
        perf1 = ((df1['Close'].iloc[-1] / df1['Close'].iloc[0]) - 1) * 100
        perf2 = ((df2['Close'].iloc[-1] / df2['Close'].iloc[0]) - 1) * 100
        diff = perf1 - perf2
        
        summary_text = (f'{symbol1}: {perf1:+.2f}% | '
                       f'{symbol2}: {perf2:+.2f}% | '
                       f'Difference: {diff:+.2f}%')
        
        fig.text(0.02, 0.02, summary_text, color='white', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='black', 
                         edgecolor=AQUA_MARINE))
        
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.1, hspace=0.1)
        
        logger.info("Comparison chart created successfully")
        return fig
    
    def create_quick_chart(self, df: pd.DataFrame):
        """Create simplified chart for quick analysis"""
        logger.info("Creating quick chart")
        
        AQUA_MARINE = '#00FF9D'  # Aqua marine
        LIGHT_ORANGE = '#FF8C42'
        LIGHT_BLUE = '#4CC9F0'
        WHITE = '#FFFFFF'
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), 
                                       gridspec_kw={'height_ratios': [3, 1]},
                                       sharex=True)
        
        # Set black background
        fig.patch.set_facecolor('black')
        ax1.set_facecolor('black')
        ax2.set_facecolor('black')
        
        # Price chart - AQUA MARINE THIN LINE
        ax1.plot(df.index, df['Close'], color=AQUA_MARINE, linewidth=1.5, label='Price')
        if 'SMA_20' in df.columns:
            ax1.plot(df.index, df['SMA_20'], color=LIGHT_ORANGE, linewidth=1.0, 
                    linestyle='--', label='SMA 20')
        if 'SMA_50' in df.columns:
            ax1.plot(df.index, df['SMA_50'], color=LIGHT_BLUE, linewidth=1.0, 
                    linestyle='--', label='SMA 50')
        
        # Highlight area above SMA20 (transparent)
        if 'SMA_20' in df.columns:
            above_sma20 = df['Close'] > df['SMA_20']
            ax1.fill_between(df.index, df['SMA_20'], df['Close'],
                            where=above_sma20, color=AQUA_MARINE, alpha=0.1,
                            label='Above SMA20')
        
        ax1.set_title(f'{self.symbol} - Quick Analysis', color='white', fontsize=13)
        ax1.legend(loc='upper left', facecolor='black', labelcolor='white', fontsize=9)
        ax1.grid(True, alpha=0.15, color='gray', linewidth=0.3)
        ax1.tick_params(colors='white', labelsize=8)
        
        # Volume thin bars
        colors_volume = []
        if 'Volume' in df.columns:
            for i in range(len(df)):
                if i == 0:
                    colors_volume.append(AQUA_MARINE)
                elif df['Close'].iloc[i] >= df['Close'].iloc[i-1]:
                    colors_volume.append(AQUA_MARINE)
                else:
                    colors_volume.append('#FF0000')
            
            ax2.bar(df.index, df['Volume'], color=colors_volume, alpha=0.6, width=0.6)
        
        ax2.set_xlabel('Date', color='white', fontsize=10)
        ax2.tick_params(colors='white', labelsize=8)
        ax2.grid(True, alpha=0.15, color='gray', linewidth=0.3)
        
        # Format dates
        date_format = mdates.DateFormatter('%b %d')
        ax2.xaxis.set_major_formatter(date_format)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right', 
                color='white', fontsize=8)
        
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.1, hspace=0.1)
        
        logger.info("Quick chart created successfully")
        return fig