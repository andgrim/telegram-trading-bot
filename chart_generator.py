"""
Chart Generator for Trading Analysis
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
    """Generate comprehensive charts with multiple indicators"""
    
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
        """Generate comprehensive price chart with indicators"""
        if len(data) < 10:
            raise ValueError(f"Not enough data for chart: {len(data)} rows")
        
        try:
            # Clean the data - ensure all columns are 1D
            display_data = data.copy()
            
            # Debug: Print data info
            print(f"📊 Chart data shape: {display_data.shape}")
            print(f"📊 Chart columns: {list(display_data.columns)[:15]}")
            
            # Ensure index is datetime
            if not isinstance(display_data.index, pd.DatetimeIndex):
                display_data.index = pd.to_datetime(display_data.index)
            
            # Flatten any 2D columns
            for col in display_data.columns:
                if col in display_data.columns:
                    values = display_data[col].values
                    if hasattr(values, 'ndim') and values.ndim > 1:
                        display_data[col] = pd.Series(values.flatten(), index=display_data.index)
            
            # Determine days to show based on period
            days_to_show = self._get_days_for_period(period, len(display_data))
            
            # Ensure we have enough data
            if len(display_data) > days_to_show:
                display_data = display_data.tail(days_to_show)
            
            # Create figure with appropriate size
            fig = plt.figure(figsize=(16, 14), 
                            facecolor=self.config.CHART_COLORS['background'],
                            dpi=100)
            
            # Create subplots grid - 6 subplots
            gs = fig.add_gridspec(6, 1, hspace=0.15, height_ratios=[3, 1, 1, 1, 1, 1])
            
            # 1. PRICE CHART with Moving Averages
            ax1 = fig.add_subplot(gs[0])
            
            # Ensure we have price data
            if 'Close' not in display_data.columns or len(display_data['Close'].dropna()) < 5:
                plt.close()
                raise ValueError("Insufficient price data for chart")
            
            # Price line
            ax1.plot(display_data.index, display_data['Close'], 
                    color=self.config.CHART_COLORS['price_line'],
                    linewidth=1.5,
                    label=f'{ticker} Price',
                    zorder=5)
            
            # Moving averages with different colors
            ma_configs = [
                (20, self.config.CHART_COLORS['ma_20'], '20 MA'),
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
            
            # Bollinger Bands if available
            if all(col in display_data.columns for col in ['BB_Upper', 'BB_Lower']):
                ax1.fill_between(display_data.index, 
                                display_data['BB_Upper'], 
                                display_data['BB_Lower'],
                                color=self.config.CHART_COLORS['grid'],
                                alpha=0.2,
                                label='Bollinger Bands')
            
            ax1.set_title(f'{ticker} - Technical Analysis ({period})', 
                         fontsize=14,
                         fontweight='bold',
                         pad=12)
            ax1.set_ylabel('Price', fontsize=10)
            ax1.legend(loc='upper left', fontsize=9)
            ax1.grid(True, alpha=0.1)
            ax1.tick_params(axis='both', which='major', labelsize=9)
            
            # 2. VOLUME CHART
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
                    
                    # Volume moving average
                    if 'Volume_MA_20' in display_data.columns:
                        vol_ma = display_data['Volume_MA_20'].dropna()
                        if len(vol_ma) > 0:
                            ax2.plot(vol_ma.index, vol_ma,
                                    color='yellow',
                                    linewidth=0.8,
                                    alpha=0.7,
                                    label='20-day Volume MA')
                except Exception as e:
                    print(f"Volume chart error: {e}")
            
            ax2.set_ylabel('Volume', fontsize=10)
            ax2.legend(loc='upper left', fontsize=8)
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
                        
                        # RSI levels with different styles
                        ax3.axhline(y=70, color=self.config.CHART_COLORS['volume_down'], 
                                  linestyle='--', alpha=0.6, linewidth=0.8, label='Overbought (70)')
                        ax3.axhline(y=30, color=self.config.CHART_COLORS['volume_up'], 
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
            
            # 5. A/D LINE CHART
            ax5 = fig.add_subplot(gs[4], sharex=ax1)
            
            if 'AD_Line' in display_data.columns:
                try:
                    ad_data = display_data['AD_Line'].dropna()
                    if len(ad_data) > 0:
                        ax5.plot(ad_data.index, ad_data,
                                color='cyan',
                                linewidth=1.0,
                                label='A/D Line')
                        
                        # Calculate and plot A/D Line EMA
                        if len(ad_data) > 20:
                            ad_ema = ad_data.ewm(span=20, adjust=False).mean()
                            ax5.plot(ad_ema.index, ad_ema,
                                    color='magenta',
                                    linewidth=0.8,
                                    alpha=0.7,
                                    label='A/D EMA (20)')
                        
                        ax5.set_ylabel('A/D Line', fontsize=10)
                        ax5.legend(loc='upper left', fontsize=8)
                        ax5.grid(True, alpha=0.1)
                except Exception as e:
                    print(f"A/D Line chart error: {e}")
            
            ax5.tick_params(axis='both', which='major', labelsize=8)
            
            # 6. STOCHASTIC CHART
            ax6 = fig.add_subplot(gs[5], sharex=ax1)
            
            if 'Stoch_%K' in display_data.columns and 'Stoch_%D' in display_data.columns:
                try:
                    stoch_k = display_data['Stoch_%K'].dropna()
                    stoch_d = display_data['Stoch_%D'].dropna()
                    
                    if len(stoch_k) > 0 and len(stoch_d) > 0:
                        ax6.plot(stoch_k.index, stoch_k,
                                color='yellow',
                                linewidth=1.0,
                                label='%K')
                        
                        ax6.plot(stoch_d.index, stoch_d,
                                color='orange',
                                linewidth=1.0,
                                label='%D')
                        
                        # Stochastic levels
                        ax6.axhline(y=80, color='red', 
                                  linestyle='--', alpha=0.5, linewidth=0.6, label='Overbought (80)')
                        ax6.axhline(y=20, color='green', 
                                  linestyle='--', alpha=0.5, linewidth=0.6, label='Oversold (20)')
                        
                        # Fill overbought/oversold areas
                        ax6.fill_between(stoch_k.index, 80, 100, 
                                        color='red', alpha=0.1)
                        ax6.fill_between(stoch_k.index, 0, 20, 
                                        color='green', alpha=0.1)
                        
                        ax6.set_ylim(0, 100)
                        ax6.set_ylabel('Stochastic', fontsize=10)
                        ax6.legend(loc='upper left', fontsize=8)
                        ax6.grid(True, alpha=0.1)
                except Exception as e:
                    print(f"Stochastic chart error: {e}")
            
            ax6.tick_params(axis='both', which='major', labelsize=8)
            
            # Format dates based on period
            self._format_date_axis(ax6, period, display_data)
            
            plt.tight_layout(pad=1.5)
            
            # Save to temp file
            temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            plt.savefig(temp_file.name, 
                       facecolor=self.config.CHART_COLORS['background'],
                       bbox_inches='tight',
                       dpi=100,
                       format='png',
                       transparent=False)
            plt.close()
            
            print(f"✅ Chart generated: {temp_file.name}")
            return temp_file.name
            
        except Exception as e:
            print(f"❌ Chart generation error: {e}")
            import traceback
            traceback.print_exc()
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
            
            # Determine date format based on period length
            date_range = (data.index[-1] - data.index[0]).days
            
            if date_range < 90:  # Less than 3 months
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
                ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
                ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
            elif date_range < 180:  # Less than 6 months
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
                ax.xaxis.set_major_locator(mdates.MonthLocator())
                ax.xaxis.set_minor_locator(mdates.WeekdayLocator(interval=1))
            elif date_range < 365:  # Less than 1 year
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            elif date_range < 730:  # Less than 2 years
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            else:  # More than 2 years
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                ax.xaxis.set_major_locator(mdates.YearLocator())
            
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
            
            # Adjust x-axis limits to show all data
            if len(data) > 0:
                ax.set_xlim([data.index[0], data.index[-1]])
        except Exception as e:
            print(f"Date formatting error: {e}")
            # Fallback to simple formatting
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)