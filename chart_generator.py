import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from typing import Dict, List
import tempfile
import os
from datetime import datetime, timedelta

from config import CONFIG

class ChartGenerator:
    """Generate clean trading charts with extended timeframe support"""
    
    def __init__(self):
        self.config = CONFIG
        self.colors = self.config.CHART_COLORS
        self.style = self.config.CHART_STYLE
        
        # Set dark theme with thin lines
        plt.style.use('dark_background')
        
        # Configure thin lines globally
        plt.rcParams.update({
            'figure.facecolor': self.colors['background'],
            'axes.facecolor': self.colors['background'],
            'axes.edgecolor': self.colors['text'],
            'axes.labelcolor': self.colors['text'],
            'text.color': self.colors['text'],
            'xtick.color': self.colors['text'],
            'ytick.color': self.colors['text'],
            'grid.color': self.colors['grid'],
            'grid.alpha': 0.1,
            'lines.linewidth': 1.0,
            'axes.linewidth': 0.5,
            'xtick.major.width': 0.5,
            'ytick.major.width': 0.5,
        })
    
    def _apply_style_to_axes(self, ax):
        """Apply consistent thin style to axes"""
        ax.tick_params(colors=self.colors['text'], width=0.5, length=3)
        ax.title.set_color(self.colors['text'])
        ax.xaxis.label.set_color(self.colors['text'])
        ax.yaxis.label.set_color(self.colors['text'])
        ax.grid(True, alpha=0.1, linewidth=0.3)
    
    def generate_price_chart(self, data: pd.DataFrame, ticker: str, 
                           period: str = '1y') -> str:
        """Generate clean price chart for all timeframes including extended periods"""
        # Ensure we have enough data points
        if len(data) < 10:
            raise ValueError(f"Not enough data points: {len(data)}")
        
        print(f"DEBUG: Generating chart for {ticker}, period {period}")
        print(f"DEBUG: Data shape: {data.shape}")
        print(f"DEBUG: Data columns: {data.columns.tolist()}")
        print(f"DEBUG: Data index type: {type(data.index)}")
        print(f"DEBUG: First date: {data.index[0]}, Last date: {data.index[-1]}")
        
        # Verify required columns exist
        required_cols = ['Close', 'High', 'Low', 'Open', 'Volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            print(f"WARNING: Missing columns: {missing_cols}")
            # Try to continue with what we have
        
        # Use appropriate number of periods based on timeframe
        # Trading days approximation: 252 days per year
        period_display_map = {
            '3m': 63,    # 3 months
            '6m': 126,   # 6 months
            '1y': 252,   # 1 year
            '2y': 504,   # 2 years
            '3y': 756,   # 3 years
            '5y': 1260   # 5 years
        }
        
        days_to_display = period_display_map.get(period, 252)
        
        # Ensure we don't exceed available data
        if len(data) > days_to_display:
            display_data = data.iloc[-days_to_display:].copy()
        else:
            display_data = data.copy()
        
        fig = plt.figure(figsize=(14, 10), 
                        facecolor=self.colors['background'],
                        dpi=self.style['dpi'])
        
        # Create grid for main chart and indicators
        gs = fig.add_gridspec(4, 1, hspace=0.15, height_ratios=[3, 1, 1, 1])
        
        # 1. PRICE PANEL
        ax_price = fig.add_subplot(gs[0])
        
        # Plot price line (thin)
        ax_price.plot(display_data.index, display_data['Close'], 
                     color=self.colors['price_line'],
                     linewidth=self.style['price_line_width'],
                     alpha=0.9,
                     label=f'{ticker} Price',
                     zorder=5)
        
        # Plot moving averages - different sets for different timeframes
        if period in ['2y', '3y', '5y']:
            # For long timeframes, show 50, 100, 200 MAs
            ma_configs = [
                (50, 'SMA_50', self.colors['ma_50'], '50 MA', '-'),
                (100, 'SMA_100', self.colors['ma_20'], '100 MA', '--'),
                (200, 'SMA_200', self.colors['ma_9'], '200 MA', '-.')
            ]
        else:
            # For short/medium timeframes
            ma_configs = [
                (9, 'SMA_9', self.colors['ma_9'], '9 MA', '--'),
                (20, 'SMA_20', self.colors['ma_20'], '20 MA', '-'),
                (50, 'SMA_50', self.colors['ma_50'], '50 MA', '-'),
            ]
        
        for ma_val, ma_col, color, label, linestyle in ma_configs:
            if ma_col in display_data.columns:
                ma_data = display_data[ma_col]
                valid_mask = ~ma_data.isna()
                if valid_mask.any():
                    ax_price.plot(display_data.index[valid_mask], ma_data[valid_mask],
                                color=color,
                                linewidth=self.style['ma_line_width'],
                                alpha=0.7,
                                label=label,
                                linestyle=linestyle,
                                zorder=4)
        
        # Plot Bollinger Bands if available (only for 1y and shorter for clarity)
        if 'BB_Upper' in display_data.columns and 'BB_Lower' in display_data.columns and period not in ['2y', '3y', '5y']:
            # Fill between Bollinger Bands
            ax_price.fill_between(display_data.index, 
                                display_data['BB_Lower'], 
                                display_data['BB_Upper'], 
                                alpha=0.1, color=self.colors['bb_lower'],
                                label='Bollinger Bands')
            
            # Plot BB lines
            ax_price.plot(display_data.index, display_data['BB_Upper'],
                         color=self.colors['bb_upper'],
                         linewidth=0.6, alpha=0.5, linestyle='--')
            ax_price.plot(display_data.index, display_data['BB_Lower'],
                         color=self.colors['bb_lower'],
                         linewidth=0.6, alpha=0.5, linestyle='--')
        
        # Set title based on timeframe
        timeframe_label = {
            '3m': '3 Months',
            '6m': '6 Months',
            '1y': '1 Year',
            '2y': '2 Years',
            '3y': '3 Years',
            '5y': '5 Years'
        }.get(period, period.upper())
        
        ax_price.set_title(f'{ticker} - Technical Analysis ({timeframe_label})', 
                          fontsize=self.style['title_size'], 
                          fontweight='bold',
                          color=self.colors['text'])
        ax_price.set_ylabel('Price (USD)', color=self.colors['text'], fontsize=10)
        
        # Add legend only if we have multiple lines
        handles, labels = ax_price.get_legend_handles_labels()
        if len(handles) > 1:
            ax_price.legend(loc='upper left', fontsize=8, framealpha=0.3)
        
        self._apply_style_to_axes(ax_price)
        
        # 2. VOLUME PANEL
        ax_volume = fig.add_subplot(gs[1], sharex=ax_price)
        
        # Plot volume bars with thin styling
        colors_volume = [
            self.colors['volume_up'] if close >= open_ else self.colors['volume_down']
            for close, open_ in zip(display_data['Close'], display_data['Open'])
        ]
        
        # For long timeframes, use weekly volume bars for better visualization
        if period in ['2y', '3y', '5y']:
            # Resample to weekly volume
            volume_resampled = display_data['Volume'].resample('W').sum()
            prices_resampled = display_data['Close'].resample('W').ohlc()
            
            colors_volume_resampled = [
                self.colors['volume_up'] if close >= open_ else self.colors['volume_down']
                for close, open_ in zip(prices_resampled['close'], prices_resampled['open'])
            ]
            
            ax_volume.bar(volume_resampled.index, volume_resampled, 
                         color=colors_volume_resampled, 
                         alpha=0.5,
                         width=5,
                         edgecolor='none',
                         linewidth=0.1)
        else:
            ax_volume.bar(display_data.index, display_data['Volume'], 
                         color=colors_volume, 
                         alpha=0.5,
                         width=0.6,
                         edgecolor='none',
                         linewidth=0.1)
        
        # Plot volume moving average
        if 'Volume_MA' in display_data.columns:
            if period in ['2y', '3y', '5y']:
                # Use weekly resampled volume MA
                volume_ma_resampled = display_data['Volume_MA'].resample('W').mean()
                ax_volume.plot(volume_ma_resampled.index, volume_ma_resampled,
                              color=self.colors['text'],
                              linewidth=0.8, alpha=0.7, label='Volume MA')
            else:
                ax_volume.plot(display_data.index, display_data['Volume_MA'],
                              color=self.colors['text'],
                              linewidth=0.8, alpha=0.7, label='20-day MA')
        
        ax_volume.set_ylabel('Volume', color=self.colors['text'], fontsize=10)
        self._apply_style_to_axes(ax_volume)
        
        # 3. RSI PANEL
        ax_rsi = fig.add_subplot(gs[2], sharex=ax_price)
        
        if 'RSI' in display_data.columns:
            rsi_values = display_data['RSI']
            rsi_valid = ~rsi_values.isna()
            
            if rsi_valid.any():
                # For long timeframes, resample RSI for cleaner visualization
                if period in ['2y', '3y', '5y']:
                    rsi_resampled = display_data['RSI'].resample('W').mean()
                    ax_rsi.plot(rsi_resampled.index, rsi_resampled, 
                               color=self.colors['rsi_line'],
                               linewidth=self.style['indicator_line_width'],
                               alpha=0.8)
                else:
                    ax_rsi.plot(display_data.index[rsi_valid], rsi_values[rsi_valid], 
                               color=self.colors['rsi_line'],
                               linewidth=self.style['indicator_line_width'],
                               alpha=0.8)
                
                # Add RSI levels
                ax_rsi.axhline(y=70, color=self.colors['volume_down'], 
                              linestyle='--', alpha=0.4, linewidth=0.5)
                ax_rsi.axhline(y=30, color=self.colors['volume_up'], 
                              linestyle='--', alpha=0.4, linewidth=0.5)
                ax_rsi.axhline(y=50, color=self.colors['text'], 
                              linestyle='--', alpha=0.2, linewidth=0.3)
                
                # Fill oversold/overbought areas
                if period in ['2y', '3y', '5y']:
                    ax_rsi.fill_between(rsi_resampled.index, 70, 100, 
                                      alpha=0.05, color=self.colors['volume_down'])
                    ax_rsi.fill_between(rsi_resampled.index, 0, 30, 
                                      alpha=0.05, color=self.colors['volume_up'])
                else:
                    ax_rsi.fill_between(display_data.index[rsi_valid], 70, 100, 
                                      alpha=0.05, color=self.colors['volume_down'])
                    ax_rsi.fill_between(display_data.index[rsi_valid], 0, 30, 
                                      alpha=0.05, color=self.colors['volume_up'])
                
                ax_rsi.set_ylim(0, 100)
                ax_rsi.set_ylabel('RSI', color=self.colors['text'], fontsize=10)
                self._apply_style_to_axes(ax_rsi)
        
        # 4. MACD PANEL
        ax_macd = fig.add_subplot(gs[3], sharex=ax_price)
        
        if 'MACD' in display_data.columns and 'MACD_Signal' in display_data.columns:
            macd_valid = ~display_data['MACD'].isna() & ~display_data['MACD_Signal'].isna()
            
            if macd_valid.any():
                # For long timeframes, resample MACD
                if period in ['2y', '3y', '5y']:
                    macd_resampled = display_data['MACD'].resample('W').mean()
                    signal_resampled = display_data['MACD_Signal'].resample('W').mean()
                    
                    ax_macd.plot(macd_resampled.index, macd_resampled, 
                                color=self.colors['macd_line'],
                                linewidth=self.style['indicator_line_width'],
                                alpha=0.8,
                                label='MACD')
                    
                    ax_macd.plot(signal_resampled.index, signal_resampled, 
                                color=self.colors['macd_signal'],
                                linewidth=self.style['indicator_line_width'],
                                alpha=0.8,
                                label='Signal')
                    
                    # MACD histogram (resampled)
                    if 'MACD_Hist' in display_data.columns:
                        hist_resampled = display_data['MACD_Hist'].resample('W').mean()
                        hist_valid = ~hist_resampled.isna()
                        if hist_valid.any():
                            colors_hist = [
                                self.colors['macd_line'] if val >= 0 else self.colors['macd_signal']
                                for val in hist_resampled[hist_valid]
                            ]
                            
                            ax_macd.bar(hist_resampled.index[hist_valid], 
                                       hist_resampled[hist_valid], 
                                       color=colors_hist,
                                       alpha=0.4,
                                       width=5,
                                       edgecolor='none',
                                       linewidth=0.1)
                else:
                    # Standard plotting for shorter timeframes
                    ax_macd.plot(display_data.index[macd_valid], display_data['MACD'][macd_valid], 
                                color=self.colors['macd_line'],
                                linewidth=self.style['indicator_line_width'],
                                alpha=0.8,
                                label='MACD')
                    
                    ax_macd.plot(display_data.index[macd_valid], display_data['MACD_Signal'][macd_valid], 
                                color=self.colors['macd_signal'],
                                linewidth=self.style['indicator_line_width'],
                                alpha=0.8,
                                label='Signal')
                    
                    # MACD histogram
                    if 'MACD_Hist' in display_data.columns:
                        hist_valid = ~display_data['MACD_Hist'].isna()
                        if hist_valid.any():
                            colors_hist = [
                                self.colors['macd_line'] if val >= 0 else self.colors['macd_signal']
                                for val in display_data['MACD_Hist'][hist_valid]
                            ]
                            
                            ax_macd.bar(display_data.index[hist_valid], 
                                       display_data['MACD_Hist'][hist_valid], 
                                       color=colors_hist,
                                       alpha=0.4,
                                       width=0.5,
                                       edgecolor='none',
                                       linewidth=0.1)
                
                ax_macd.axhline(y=0, color=self.colors['text'], 
                               linestyle='-', linewidth=0.3, alpha=0.3)
                
                ax_macd.set_ylabel('MACD', color=self.colors['text'], fontsize=10)
                ax_macd.legend(loc='upper left', fontsize=7, framealpha=0.3)
                self._apply_style_to_axes(ax_macd)
        
        # Format x-axis based on timeframe
        bottom_ax = ax_macd if 'MACD' in display_data.columns else ax_rsi if 'RSI' in display_data.columns else ax_volume
        
        # Adjust date formatting based on period
        if period == '3m':
            bottom_ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
            bottom_ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        elif period == '6m':
            bottom_ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
            bottom_ax.xaxis.set_major_locator(mdates.MonthLocator())
        elif period == '1y':
            bottom_ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            bottom_ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        elif period == '2y':
            bottom_ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            bottom_ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        elif period == '3y':
            bottom_ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            bottom_ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
        elif period == '5y':
            bottom_ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            bottom_ax.xaxis.set_major_locator(mdates.YearLocator())
        else:
            bottom_ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        
        plt.setp(bottom_ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
        
        plt.tight_layout()
        
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        plt.savefig(temp_file.name, 
                   facecolor=self.colors['background'], 
                   edgecolor='none', 
                   bbox_inches='tight',
                   dpi=self.style['dpi'])
        plt.close()
        
        return temp_file.name