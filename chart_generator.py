"""
Chart Generator for Trading Analysis - Simplified Local Version
Only includes Price, Volume+A/D, RSI, and MACD
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import tempfile
import os

from config import CONFIG

class ChartGenerator:
    """Generate simplified charts with key indicators"""
    
    def __init__(self):
        self.config = CONFIG
        
        # Configure dark theme
        plt.style.use('dark_background')
        
        plt.rcParams.update({
            'figure.facecolor': self.config.CHART_COLORS['background'],
            'axes.facecolor': self.config.CHART_COLORS['background'],
            'axes.edgecolor': self.config.CHART_COLORS['text'],
            'axes.labelcolor': self.config.CHART_COLORS['text'],
            'text.color': self.config.CHART_COLORS['text'],
            'xtick.color': self.config.CHART_COLORS['text'],
            'ytick.color': self.config.CHART_COLORS['text'],
            'grid.color': self.config.CHART_COLORS['grid'],
            'grid.alpha': 0.1,
            'lines.linewidth': 1.0,
        })
    
    def generate_price_chart(self, data: pd.DataFrame, ticker: str, period: str) -> str:
        """Generate simplified price chart with 4 indicators"""
        if len(data) < 10:
            raise ValueError(f"Not enough data for chart: {len(data)} rows")
        
        try:
            # Clean the data
            display_data = data.copy()
            
            # Ensure index is datetime
            if not isinstance(display_data.index, pd.DatetimeIndex):
                display_data.index = pd.to_datetime(display_data.index)
            
            # Determine days to show
            days_to_show = self._get_days_for_period(period, len(display_data))
            
            # Ensure we have enough data
            if len(display_data) > days_to_show:
                display_data = display_data.tail(days_to_show)
            
            # Create figure with 4 subplots
            fig = plt.figure(figsize=(14, 12), 
                            facecolor=self.config.CHART_COLORS['background'],
                            dpi=100)
            
            # Create subplots grid - 4 subplots
            gs = fig.add_gridspec(4, 1, hspace=0.15, height_ratios=[3, 1, 1, 1])
            
            # 1. PRICE CHART with Moving Averages
            ax1 = fig.add_subplot(gs[0])
            
            # Price line
            ax1.plot(display_data.index, display_data['Close'], 
                    color=self.config.CHART_COLORS['price_line'],
                    linewidth=1.5,
                    label=f'{ticker} Price',
                    zorder=5)
            
            # Moving averages with different colors (20 MA in green)
            ma_configs = [
                (20, self.config.CHART_COLORS['ma_20'], '20 MA'),  # Green
                (50, self.config.CHART_COLORS['ma_50'], '50 MA'),
                (200, self.config.CHART_COLORS['ma_200'], '200 MA')
            ]
            
            for period_ma, color, label in ma_configs:
                col = f'SMA_{period_ma}'
                if col in display_data.columns:
                    valid_data = display_data[col].dropna()
                    if len(valid_data) > 0:
                        ax1.plot(valid_data.index, valid_data,
                                color=color,
                                linewidth=1.2,
                                alpha=0.8,
                                label=label)
            
            ax1.set_title(f'{ticker} - Technical Analysis ({period})', 
                         fontsize=14,
                         fontweight='bold',
                         pad=12)
            ax1.set_ylabel('Price', fontsize=10)
            ax1.legend(loc='upper left', fontsize=9)
            ax1.grid(True, alpha=0.1)
            ax1.tick_params(axis='both', which='major', labelsize=9)
            
            # 2. VOLUME CHART with A/D Line overlay
            ax2 = fig.add_subplot(gs[1], sharex=ax1)
            
            # Volume bars
            if 'Volume' in display_data.columns:
                try:
                    volume_data = display_data['Volume'].fillna(0)
                    close_data = display_data['Close']
                    
                    # Volume coloring based on price movement
                    colors = []
                    for i in range(len(volume_data)):
                        if i == 0:
                            colors.append(self.config.CHART_COLORS['volume_up'])
                        else:
                            try:
                                if close_data.iloc[i] >= close_data.iloc[i-1]:
                                    colors.append(self.config.CHART_COLORS['volume_up'])
                                else:
                                    colors.append(self.config.CHART_COLORS['volume_down'])
                            except:
                                colors.append(self.config.CHART_COLORS['volume_up'])
                    
                    # Plot volume bars
                    ax2.bar(volume_data.index, volume_data, 
                           color=colors, alpha=0.6, width=0.8, label='Volume')
                    
                    # Plot A/D Line on secondary axis
                    if 'AD_Line' in display_data.columns:
                        ax2b = ax2.twinx()
                        ad_data = display_data['AD_Line'].dropna()
                        if len(ad_data) > 0:
                            ax2b.plot(ad_data.index, ad_data,
                                     color=self.config.CHART_COLORS['ad_line'],
                                     linewidth=1.0,
                                     label='A/D Line',
                                     alpha=0.8)
                            ax2b.set_ylabel('A/D Line', fontsize=10, color=self.config.CHART_COLORS['ad_line'])
                            ax2b.tick_params(axis='y', labelcolor=self.config.CHART_COLORS['ad_line'])
                            
                            # Add legend for A/D line
                            lines1, labels1 = ax2.get_legend_handles_labels()
                            lines2, labels2 = ax2b.get_legend_handles_labels()
                            ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)
                    
                except Exception as e:
                    print(f"Volume chart error: {e}")
            
            ax2.set_ylabel('Volume', fontsize=10)
            ax2.grid(True, alpha=0.1)
            ax2.tick_params(axis='both', which='major', labelsize=8)
            
            # 3. RSI CHART
            ax3 = fig.add_subplot(gs[2], sharex=ax1)
            
            if 'RSI' in display_data.columns:
                try:
                    rsi_data = display_data['RSI'].dropna()
                    if len(rsi_data) > 0:
                        ax3.plot(rsi_data.index, rsi_data,
                                color=self.config.CHART_COLORS['rsi_line'],
                                linewidth=1.0,
                                label='RSI')
                        
                        # RSI levels
                        ax3.axhline(y=70, color='red', 
                                  linestyle='--', alpha=0.6, linewidth=0.8, label='Overbought (70)')
                        ax3.axhline(y=30, color='green', 
                                  linestyle='--', alpha=0.6, linewidth=0.8, label='Oversold (30)')
                        ax3.axhline(y=50, color='white', 
                                  linestyle=':', alpha=0.3, linewidth=0.5)
                        
                        # Fill overbought/oversold areas
                        ax3.fill_between(rsi_data.index, 70, 100, 
                                        color='red', alpha=0.1)
                        ax3.fill_between(rsi_data.index, 0, 30, 
                                        color='green', alpha=0.1)
                        
                        ax3.set_ylim(0, 100)
                        ax3.set_ylabel('RSI', fontsize=10)
                        ax3.legend(loc='upper left', fontsize=8)
                        ax3.grid(True, alpha=0.1)
                except Exception as e:
                    print(f"RSI chart error: {e}")
            
            ax3.tick_params(axis='both', which='major', labelsize=8)
            
            # 4. MACD CHART
            ax4 = fig.add_subplot(gs[3], sharex=ax1)
            
            if 'MACD' in display_data.columns and 'MACD_Signal' in display_data.columns:
                try:
                    macd_data = display_data['MACD'].dropna()
                    signal_data = display_data['MACD_Signal'].dropna()
                    
                    if len(macd_data) > 0 and len(signal_data) > 0:
                        # Plot MACD and Signal lines
                        ax4.plot(macd_data.index, macd_data,
                                color=self.config.CHART_COLORS['macd_line'],
                                linewidth=1.0,
                                label='MACD')
                        
                        ax4.plot(signal_data.index, signal_data,
                                color=self.config.CHART_COLORS['macd_signal'],
                                linewidth=1.0,
                                label='Signal')
                        
                        # Plot MACD histogram
                        if 'MACD_Hist' in display_data.columns:
                            hist_data = display_data['MACD_Hist'].dropna()
                            colors = ['green' if x >= 0 else 'red' for x in hist_data]
                            ax4.bar(hist_data.index, hist_data,
                                   color=colors, alpha=0.5, width=0.8, label='Histogram')
                        
                        ax4.axhline(y=0, color='white', linestyle='-', alpha=0.3, linewidth=0.5)
                        ax4.set_ylabel('MACD', fontsize=10)
                        ax4.legend(loc='upper left', fontsize=8)
                        ax4.grid(True, alpha=0.1)
                except Exception as e:
                    print(f"MACD chart error: {e}")
            
            ax4.tick_params(axis='both', which='major', labelsize=8)
            
            # Format dates
            self._format_date_axis(ax4, period, display_data)
            
            plt.tight_layout(pad=1.5)
            
            # Save to temp file
            temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            plt.savefig(temp_file.name, 
                       facecolor=self.config.CHART_COLORS['background'],
                       bbox_inches='tight',
                       dpi=100,
                       format='png')
            plt.close()
            
            print(f"✅ Chart generated: {temp_file.name}")
            return temp_file.name
            
        except Exception as e:
            print(f"❌ Chart generation error: {e}")
            raise
    
    def _get_days_for_period(self, period: str, max_days: int) -> int:
        """Get appropriate number of days to show for each period"""
        period_map = {
            '3m': min(63, max_days),
            '6m': min(126, max_days),
            '1y': min(252, max_days),
            '2y': min(504, max_days),
            '3y': min(756, max_days),
            '5y': min(1260, max_days),
            'max': max_days
        }
        return period_map.get(period, min(252, max_days))
    
    def _format_date_axis(self, ax, period: str, data: pd.DataFrame):
        """Format date axis based on timeframe"""
        try:
            if len(data) < 10:
                return
            
            date_range = (data.index[-1] - data.index[0]).days
            
            if date_range < 90:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
                ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
            elif date_range < 180:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
                ax.xaxis.set_major_locator(mdates.MonthLocator())
            elif date_range < 365:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            else:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                ax.xaxis.set_major_locator(mdates.YearLocator())
            
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
            
            # Adjust x-axis limits
            if len(data) > 0:
                ax.set_xlim([data.index[0], data.index[-1]])
        except Exception as e:
            print(f"Date formatting error: {e}")
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)