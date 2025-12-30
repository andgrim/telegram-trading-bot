import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from utils import get_stock_data, calculate_technical_indicators

class TradingAnalyzer:
    def __init__(self, symbol: str):
        self.symbol = symbol.upper()
    
    def analyze_period(self, period_months: int = 3):
        """Analyze for a specific period"""
        try:
            # Get data
            df = get_stock_data(self.symbol, period_months)
            
            if df.empty:
                return None
            
            # Calculate indicators
            df = calculate_technical_indicators(df)
            
            return df
            
        except Exception as e:
            print(f"Error analyzing {self.symbol}: {e}")
            return None
    
    def _analyze_volume_trend(self, df: pd.DataFrame) -> str:
        """Analyze volume trend"""
        if len(df) < 10:
            return "N/A"
        
        avg_volume = df['Volume'].mean()
        recent_volume = df['Volume'].iloc[-5:].mean()
        
        if recent_volume > avg_volume * 1.2:
            return "HIGH"
        elif recent_volume < avg_volume * 0.8:
            return "LOW"
        return "NORMAL"
    
    def _calculate_price_change(self, df: pd.DataFrame) -> float:
        """Calculate price change percentage"""
        if len(df) < 2:
            return 0
        return ((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1) * 100
    
    def _get_macd_signal(self, df: pd.DataFrame) -> str:
        """Determine MACD signal"""
        if df['MACD'].iloc[-1] > df['Signal_Line'].iloc[-1]:
            return "BULLISH"
        elif df['MACD'].iloc[-1] < df['Signal_Line'].iloc[-1]:
            return "BEARISH"
        return "NEUTRAL"
    
    def create_technical_chart(self, df: pd.DataFrame, period: str):
        """Create professional technical chart in black style (6 months style)"""
        # Colori stile foto RMNI
        WHITE = '#FFFFFF'
        ORANGE = '#FF6B00'  # SMA 20
        BLUE = '#00A8FF'    # SMA 50
        GREEN = '#00FF9D'   # Sopra SMA20, MACD positivo
        PURPLE = '#9D4EDD'  # RSI
        RED = '#FF0033'     # MACD negativo, Sovracomprato
        
        # Figura con sfondo nero
        fig = plt.figure(figsize=(14, 10), facecolor='black')
        
        # GridSpec con 4 righe: Prezzo, MACD, RSI, Volume
        gs = GridSpec(4, 1, figure=fig, height_ratios=[3, 1, 1, 1], hspace=0.12)
        
        # ========== 1. GRAFICO PREZZI ==========
        ax_price = fig.add_subplot(gs[0])
        ax_price.set_facecolor('black')
        
        # Plot prezzo (linea principale)
        ax_price.plot(df.index, df['Close'], color=WHITE, linewidth=2.5, 
                     label='Prezzo', alpha=0.95, zorder=5)
        
        # Medie mobili
        if 'SMA_20' in df.columns:
            ax_price.plot(df.index, df['SMA_20'], color=ORANGE, linewidth=1.8,
                         label='SMA 20', alpha=0.85, zorder=4)
        
        if 'SMA_50' in df.columns:
            ax_price.plot(df.index, df['SMA_50'], color=BLUE, linewidth=1.8,
                         label='SMA 50', alpha=0.85, zorder=3)
        
        # Evidenzia area sopra SMA20 (come nella foto)
        if 'SMA_20' in df.columns:
            above_sma20 = df['Close'] > df['SMA_20']
            ax_price.fill_between(df.index, df['SMA_20'], df['Close'],
                                 where=above_sma20, color=GREEN, alpha=0.2,
                                 label='Sopra SMA20', zorder=2)
        
        # Titolo
        price_change = self._calculate_price_change(df)
        current_price = df['Close'].iloc[-1]
        title_text = (f'{self.symbol} - Analisi {period} mesi\n'
                     f'Prezzo: ${current_price:.2f} | Variazione: {price_change:+.2f}%')
        ax_price.set_title(title_text, fontsize=14, color=WHITE, 
                          fontweight='bold', pad=12, loc='left')
        
        # Labels e ticks
        ax_price.set_ylabel('Prezzo ($)', color=WHITE, fontsize=11)
        ax_price.tick_params(axis='y', colors=WHITE, labelsize=9)
        ax_price.tick_params(axis='x', colors=WHITE, labelsize=9)
        
        # Grid
        ax_price.grid(True, alpha=0.15, color='gray', linestyle='--', linewidth=0.5)
        
        # Legend
        ax_price.legend(loc='upper left', facecolor='#111111', edgecolor=WHITE,
                       labelcolor=WHITE, fontsize=9, framealpha=0.9)
        
        # ========== 2. GRAFICO MACD ==========
        ax_macd = fig.add_subplot(gs[1], sharex=ax_price)
        ax_macd.set_facecolor('black')
        
        # MACD lines
        ax_macd.plot(df.index, df['MACD'], color=GREEN, linewidth=1.8,
                    label='MACD', alpha=0.9, zorder=3)
        ax_macd.plot(df.index, df['Signal_Line'], color=RED, linewidth=1.8,
                    label='Signal', alpha=0.9, linestyle='--', zorder=2)
        
        # MACD histogram
        macd_colors = [GREEN if val >= 0 else RED for val in df['MACD_Histogram']]
        ax_macd.bar(df.index, df['MACD_Histogram'], color=macd_colors,
                   alpha=0.6, width=0.8, edgecolor='none', zorder=1)
        
        # Zero line
        ax_macd.axhline(y=0, color=WHITE, linestyle='-', linewidth=0.8, alpha=0.6)
        
        # Labels
        ax_macd.set_ylabel('MACD Signal', color=WHITE, fontsize=11)
        ax_macd.tick_params(colors=WHITE, labelsize=9)
        ax_macd.grid(True, alpha=0.15, color='gray', linestyle='--', linewidth=0.5)
        ax_macd.legend(loc='upper left', facecolor='#111111', edgecolor=GREEN,
                      labelcolor=WHITE, fontsize=8, framealpha=0.9)
        
        # ========== 3. GRAFICO RSI ==========
        ax_rsi = fig.add_subplot(gs[2], sharex=ax_price)
        ax_rsi.set_facecolor('black')
        
        # RSI line
        ax_rsi.plot(df.index, df['RSI'], color=PURPLE, linewidth=2.0, alpha=0.9)
        
        # RSI levels
        ax_rsi.axhline(y=70, color=RED, linestyle='--', linewidth=1.3, alpha=0.8)
        ax_rsi.axhline(y=30, color=GREEN, linestyle='--', linewidth=1.3, alpha=0.8)
        
        # Zone colorate
        ax_rsi.fill_between(df.index, 30, 70, color='gray', alpha=0.1)
        ax_rsi.fill_between(df.index, 70, 100, color=RED, alpha=0.1)
        ax_rsi.fill_between(df.index, 0, 30, color=GREEN, alpha=0.1)
        
        # Labels
        ax_rsi.text(0.02, 1.02, 'Sovracomprato (70)', transform=ax_rsi.transAxes,
                   color=RED, fontsize=8, verticalalignment='bottom')
        ax_rsi.text(0.02, -0.02, 'Sovravenduto (30)', transform=ax_rsi.transAxes,
                   color=GREEN, fontsize=8, verticalalignment='top')
        
        # RSI current value
        last_rsi = df['RSI'].iloc[-1]
        ax_rsi.text(0.98, 0.95, f'RSI: {last_rsi:.1f}', transform=ax_rsi.transAxes,
                   color=WHITE, fontsize=9, fontweight='bold', ha='right',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.8))
        
        ax_rsi.set_ylabel('RSI', color=WHITE, fontsize=11)
        ax_rsi.set_ylim(0, 100)
        ax_rsi.tick_params(colors=WHITE, labelsize=9)
        ax_rsi.grid(True, alpha=0.15, color='gray', linestyle='--', linewidth=0.5)
        
        # ========== 4. GRAFICO VOLUME ==========
        ax_volume = fig.add_subplot(gs[3], sharex=ax_price)
        ax_volume.set_facecolor('black')
        
        # Volume bars colorate (verde se prezzo sale, rosso se scende)
        volume_colors = []
        for i in range(len(df)):
            if i == 0:
                volume_colors.append(GREEN)
            else:
                volume_colors.append(GREEN if df['Close'].iloc[i] >= df['Close'].iloc[i-1] else RED)
        
        ax_volume.bar(df.index, df['Volume'], color=volume_colors,
                     alpha=0.7, width=0.8, edgecolor='none')
        
        # Labels
        ax_volume.set_ylabel('Volume', color=WHITE, fontsize=11)
        ax_volume.tick_params(colors=WHITE, labelsize=9)
        ax_volume.grid(True, alpha=0.15, color='gray', linestyle='--', linewidth=0.5, axis='y')
        
        # Formatta date (stile foto: 2025-01, 2025-08, etc.)
        date_format = mdates.DateFormatter('%Y-%m')
        ax_volume.xaxis.set_major_formatter(date_format)
        
        # Mostra solo alcune date per leggibilità
        if len(df) > 120:  # 6 mesi con dati giornalieri
            ax_volume.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        else:
            ax_volume.xaxis.set_major_locator(mdates.MonthLocator())
        
        plt.setp(ax_volume.xaxis.get_majorticklabels(), rotation=45, ha='right',
                color=WHITE, fontsize=9)
        
        # Nascondi ticks x nei grafici superiori
        plt.setp(ax_price.get_xticklabels(), visible=False)
        plt.setp(ax_macd.get_xticklabels(), visible=False)
        plt.setp(ax_rsi.get_xticklabels(), visible=False)
        
        # Allineamento finale
        plt.subplots_adjust(left=0.08, right=0.95, top=0.94, bottom=0.1, hspace=0.12)
        
        return fig
    
    def create_comparison_chart(self, df1: pd.DataFrame, symbol1: str, 
                               df2: pd.DataFrame, symbol2: str, period: str):
        """Create comparison chart between two symbols"""
        # NEON COLORS
        NEON_GREEN = '#00FF00'
        NEON_BLUE = '#00FFFF'
        NEON_ORANGE = '#FF7700'
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
        fig.patch.set_facecolor('black')
        
        # Normalize prices to percentage change
        norm_price1 = (df1['Close'] / df1['Close'].iloc[0]) * 100
        norm_price2 = (df2['Close'] / df2['Close'].iloc[0]) * 100
        
        # Plot normalized prices
        ax1.plot(df1.index, norm_price1, color=NEON_GREEN, linewidth=2.5, 
                label=f'{symbol1}', alpha=0.9)
        ax1.plot(df2.index, norm_price2, color=NEON_BLUE, linewidth=2.5, 
                label=f'{symbol2}', alpha=0.9)
        
        ax1.set_title(f'{symbol1} vs {symbol2} - Performance Comparison ({period} months)',
                     color='white', fontsize=16, fontweight='bold', pad=15)
        ax1.set_ylabel('Performance (%)', color='white', fontsize=13)
        ax1.legend(loc='upper left', facecolor='#111111', edgecolor='white',
                  labelcolor='white', fontsize=11)
        ax1.grid(True, alpha=0.2, color='gray', linestyle='--')
        ax1.tick_params(colors='white', labelsize=10)
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
        
        ax2.plot(df1.index[:len(ratio_series)], ratio_series, 
                color=NEON_ORANGE, linewidth=2.5)
        ax2.axhline(y=1.0, color='white', linestyle='--', linewidth=1.2, alpha=0.6)
        
        # Fill between ratio and 1
        ax2.fill_between(df1.index[:len(ratio_series)], ratio_series, 1, 
                        where=ratio_series >= 1,
                        facecolor=NEON_GREEN, alpha=0.2)
        ax2.fill_between(df1.index[:len(ratio_series)], ratio_series, 1,
                        where=ratio_series < 1,
                        facecolor='red', alpha=0.2)
        
        ax2.set_xlabel('Date', color='white', fontsize=13)
        ax2.set_ylabel(f'{symbol1}/{symbol2} Ratio', color='white', fontsize=13)
        ax2.grid(True, alpha=0.2, color='gray', linestyle='--')
        ax2.tick_params(colors='white', labelsize=10)
        ax2.set_facecolor('black')
        
        # Format x-axis dates
        date_format = mdates.DateFormatter('%b %d')
        ax2.xaxis.set_major_formatter(date_format)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right', 
                color='white', fontsize=10)
        
        # Add performance summary
        perf1 = ((df1['Close'].iloc[-1] / df1['Close'].iloc[0]) - 1) * 100
        perf2 = ((df2['Close'].iloc[-1] / df2['Close'].iloc[0]) - 1) * 100
        diff = perf1 - perf2
        
        summary_text = (f'{symbol1}: {perf1:+.2f}% | '
                       f'{symbol2}: {perf2:+.2f}% | '
                       f'Difference: {diff:+.2f}%')
        
        fig.text(0.02, 0.02, summary_text, color='white', fontsize=11,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='black', 
                         edgecolor=NEON_GREEN))
        
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.1, hspace=0.1)
        
        return fig
    
    def create_quick_chart(self, df: pd.DataFrame):
        """Create simplified chart for quick analysis"""
        # Colori stile RMNI
        WHITE = '#FFFFFF'
        ORANGE = '#FF6B00'
        BLUE = '#00A8FF'
        GREEN = '#00FF9D'
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), 
                                       gridspec_kw={'height_ratios': [3, 1]},
                                       sharex=True)
        
        # Set black background
        fig.patch.set_facecolor('black')
        ax1.set_facecolor('black')
        ax2.set_facecolor('black')
        
        # Price chart
        ax1.plot(df.index, df['Close'], color=WHITE, linewidth=2.5, label='Prezzo')
        if 'SMA_20' in df.columns:
            ax1.plot(df.index, df['SMA_20'], color=ORANGE, linewidth=2.0, 
                    linestyle='--', label='SMA 20')
        if 'SMA_50' in df.columns:
            ax1.plot(df.index, df['SMA_50'], color=BLUE, linewidth=2.0, 
                    linestyle='--', label='SMA 50')
        
        # Evidenzia area sopra SMA20
        if 'SMA_20' in df.columns:
            above_sma20 = df['Close'] > df['SMA_20']
            ax1.fill_between(df.index, df['SMA_20'], df['Close'],
                            where=above_sma20, color=GREEN, alpha=0.2,
                            label='Sopra SMA20')
        
        ax1.set_title(f'{self.symbol} - Analisi Rapida', color='white', fontsize=14)
        ax1.legend(loc='upper left', facecolor='black', labelcolor='white', fontsize=10)
        ax1.grid(True, alpha=0.3, color='gray')
        ax1.tick_params(colors='white', labelsize=9)
        
        # Volume
        colors_volume = []
        for i in range(len(df)):
            if i == 0:
                colors_volume.append(GREEN)
            elif df['Close'].iloc[i] >= df['Close'].iloc[i-1]:
                colors_volume.append(GREEN)
            else:
                colors_volume.append('#FF0000')
        
        ax2.bar(df.index, df['Volume'], color=colors_volume, alpha=0.7, width=0.8)
        ax2.set_xlabel('Date', color='white', fontsize=11)
        ax2.tick_params(colors='white', labelsize=9)
        ax2.grid(True, alpha=0.3, color='gray')
        
        # Format dates
        date_format = mdates.DateFormatter('%b %d')
        ax2.xaxis.set_major_formatter(date_format)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right', 
                color='white', fontsize=9)
        
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.1, hspace=0.1)
        
        return fig