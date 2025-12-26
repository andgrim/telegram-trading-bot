"""
Modulo per l'analisi tecnica dei titoli
Completo con tutti i metodi necessari
"""
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import io
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class TradingAnalyzer:
    """Analizzatore tecnico per trading - Versione completa"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Configurazione stile grafico dark
        plt.style.use('dark_background')
    
    # ==================== METODI BASE ====================
    
    def get_recommended_period(self, quote_type: str) -> str:
        """Raccomanda periodo dati in base al tipo di asset"""
        period_map = {
            'CRYPTOCURRENCY': '3mo',
            'EQUITY': '3mo',
            'ETF': '6mo',
            'INDEX': '1y',
            'MUTUALFUND': '1y',
            'CURRENCY': '1mo',
            'FUTURE': '3mo',
            'OPTION': '1mo',
        }
        return period_map.get(quote_type, '3mo')
    
    def calculate_indicators(self, df: pd.DataFrame, period: str = "3mo") -> pd.DataFrame:
        """Calcola tutti gli indicatori tecnici"""
        try:
            if df.empty or len(df) < 20:
                return df
            
            df_indicators = df.copy()
            
            # 1. MEDIE MOBILI SEMPLICI (SMA)
            if len(df) >= 20:
                df_indicators['SMA_20'] = df_indicators['Close'].rolling(window=20, min_periods=1).mean()
            
            if len(df) >= 50:
                df_indicators['SMA_50'] = df_indicators['Close'].rolling(window=50, min_periods=1).mean()
            
            if len(df) >= 200:
                df_indicators['SMA_200'] = df_indicators['Close'].rolling(window=200, min_periods=1).mean()
            
            # 2. RSI (Relative Strength Index)
            if len(df) >= 15:
                delta = df_indicators['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
                rs = gain / loss
                df_indicators['RSI'] = 100 - (100 / (1 + rs))
                df_indicators['RSI'] = df_indicators['RSI'].clip(0, 100)
            
            # 3. MEDIE MOBILI ESPONENZIALI (EMA)
            if len(df) >= 12:
                df_indicators['EMA_12'] = df_indicators['Close'].ewm(span=12, adjust=False).mean()
            
            if len(df) >= 26:
                df_indicators['EMA_26'] = df_indicators['Close'].ewm(span=26, adjust=False).mean()
                
                # MACD
                df_indicators['MACD'] = df_indicators['EMA_12'] - df_indicators['EMA_26']
                df_indicators['MACD_signal'] = df_indicators['MACD'].ewm(span=9, adjust=False).mean()
                df_indicators['MACD_hist'] = df_indicators['MACD'] - df_indicators['MACD_signal']
            
            # 4. BOLLINGER BANDS
            if len(df) >= 20:
                df_indicators['BB_middle'] = df_indicators['Close'].rolling(window=20, min_periods=1).mean()
                bb_std = df_indicators['Close'].rolling(window=20, min_periods=1).std()
                df_indicators['BB_upper'] = df_indicators['BB_middle'] + (bb_std * 2)
                df_indicators['BB_lower'] = df_indicators['BB_middle'] - (bb_std * 2)
                df_indicators['BB_width'] = (df_indicators['BB_upper'] - df_indicators['BB_lower']) / df_indicators['BB_middle']
            
            # 5. VOLUME ANALYSIS
            if 'Volume' in df_indicators.columns:
                df_indicators['Volume_MA_20'] = df_indicators['Volume'].rolling(window=20, min_periods=1).mean()
                df_indicators['Volume_Ratio'] = df_indicators['Volume'] / df_indicators['Volume_MA_20']
            
            # 6. MOMENTUM INDICATORS
            # Rate of Change (ROC)
            if len(df) >= 14:
                df_indicators['ROC_14'] = ((df_indicators['Close'] - df_indicators['Close'].shift(14)) / 
                                          df_indicators['Close'].shift(14)) * 100
            
            # Stochastic Oscillator
            if len(df) >= 14:
                low_14 = df_indicators['Low'].rolling(window=14, min_periods=1).min()
                high_14 = df_indicators['High'].rolling(window=14, min_periods=1).max()
                df_indicators['Stoch_%K'] = ((df_indicators['Close'] - low_14) / (high_14 - low_14)) * 100
                df_indicators['Stoch_%D'] = df_indicators['Stoch_%K'].rolling(window=3, min_periods=1).mean()
            
            # Williams %R
            if len(df) >= 14:
                highest_high = df_indicators['High'].rolling(window=14, min_periods=1).max()
                lowest_low = df_indicators['Low'].rolling(window=14, min_periods=1).min()
                df_indicators['Williams_%R'] = ((highest_high - df_indicators['Close']) / 
                                               (highest_high - lowest_low)) * -100
            
            # CCI (Commodity Channel Index)
            if len(df) >= 20:
                typical_price = (df_indicators['High'] + df_indicators['Low'] + df_indicators['Close']) / 3
                sma_tp = typical_price.rolling(window=20, min_periods=1).mean()
                mean_deviation = typical_price.rolling(window=20, min_periods=1).std()
                df_indicators['CCI'] = (typical_price - sma_tp) / (0.015 * mean_deviation)
            
            # 7. SUPPORTO/RESISTENZA
            if len(df) >= 20:
                df_indicators['Resistance'] = df_indicators['High'].rolling(window=20, min_periods=1).max()
                df_indicators['Support'] = df_indicators['Low'].rolling(window=20, min_periods=1).min()
            
            self.logger.info(f"Indicatori calcolati per {len(df)} giorni")
            return df_indicators
            
        except Exception as e:
            self.logger.error(f"Errore calcolo indicatori: {e}")
            return df
    
    def generate_signals(self, df: pd.DataFrame) -> str:
        """Genera segnali di trading"""
        if df.empty or len(df) < 20:
            return "📊 Dati insufficienti per segnali"
        
        latest = df.iloc[-1]
        signals = []
        
        # Segnale RSI
        if pd.notna(latest.get('RSI')):
            rsi = latest['RSI']
            if rsi < 30:
                signals.append(f"📈 RSI {rsi:.1f}: FORTE IPERVENDUTO")
            elif rsi < 40:
                signals.append(f"📈 RSI {rsi:.1f}: Leggermente ipervenduto")
            elif rsi > 70:
                signals.append(f"📉 RSI {rsi:.1f}: FORTE IPERCOMPRATO")
            elif rsi > 60:
                signals.append(f"📉 RSI {rsi:.1f}: Leggermente ipercomprato")
            else:
                signals.append(f"⚖️ RSI {rsi:.1f}: Neutrale")
        
        # Segnale MACD
        if pd.notna(latest.get('MACD')) and pd.notna(latest.get('MACD_signal')):
            if latest['MACD'] > latest['MACD_signal']:
                signals.append("📈 MACD: Bullish crossover")
            else:
                signals.append("📉 MACD: Bearish crossover")
        
        # Segnale Trend SMA
        if pd.notna(latest.get('SMA_20')) and pd.notna(latest.get('SMA_50')):
            if latest['SMA_20'] > latest['SMA_50']:
                signals.append("📈 Trend: Bullish (SMA20 > SMA50)")
            else:
                signals.append("📉 Trend: Bearish (SMA20 < SMA50)")
        
        # Segnale Stochastic
        if pd.notna(latest.get('Stoch_%K')):
            stoch = latest['Stoch_%K']
            if stoch < 20:
                signals.append(f"📉 Stochastic {stoch:.1f}: Ipervenduto")
            elif stoch > 80:
                signals.append(f"📈 Stochastic {stoch:.1f}: Ipercomprato")
        
        if not signals:
            return "📊 Nessun segnale significativo"
        
        return "\n".join(signals)
    
    def generate_analysis_report(self, symbol: str, df: pd.DataFrame, info: dict) -> str:
        """Genera report completo di analisi"""
        if df.empty:
            return "❌ Nessun dato disponibile"
        
        latest = df.iloc[-1]
        
        # Funzione formattazione
        def fmt(value, format_str=".2f", prefix="$", default="N/A"):
            if pd.isna(value) or value is None:
                return default
            try:
                return f"{prefix}{value:{format_str}}" if prefix else f"{value:{format_str}}"
            except:
                return str(value)
        
        # Calcola statistiche
        if len(df) > 1:
            daily_change = ((latest['Close'] - df.iloc[-2]['Close']) / df.iloc[-2]['Close'] * 100)
            weekly_change = ((latest['Close'] - df.iloc[-6]['Close']) / df.iloc[-6]['Close'] * 100) if len(df) > 6 else 0
        else:
            daily_change = weekly_change = 0
        
        # Segnali
        signals = self.generate_signals(df)
        
        # Report
        report = f"""
📈 *ANALISI TECNICA - {symbol}*
📅 Periodo: {len(df)} giorni di dati

💰 *PREZZO E PERFORMANCE*
• Prezzo attuale: {fmt(latest['Close'])}
• Variazione 1 giorno: {fmt(daily_change, '.2f', '', '%')}
• Variazione 1 settimana: {fmt(weekly_change, '.2f', '', '%')}
• Volume: {fmt(latest.get('Volume', 0), '.0f', '')}

📊 *INDICATORI TECNICI*
• RSI (14): {fmt(latest.get('RSI'), '.1f', '')}
• MACD: {fmt(latest.get('MACD'), '.3f', '')}
• Signal: {fmt(latest.get('MACD_signal'), '.3f', '')}
• SMA 20: {fmt(latest.get('SMA_20'))}
• SMA 50: {fmt(latest.get('SMA_50'))}
• SMA 200: {fmt(latest.get('SMA_200'), default='N/D')}

📉 *MOMENTUM*
• Stochastic %K: {fmt(latest.get('Stoch_%K'), '.1f', '')}
• ROC (14): {fmt(latest.get('ROC_14', 0), '.1f', '', '%')}
• CCI: {fmt(latest.get('CCI', 0), '.1f', '')}

🎯 *SEGNALI DI TRADING*
{signals}

🏢 *INFORMAZIONI*
• Nome: {info.get('name', 'N/A')}
• Settore: {info.get('sector', 'N/A')}
• Valuta: {info.get('currency', 'USD')}
• Capitalizzazione: {fmt(info.get('marketCap', 0), '.0f')}
"""
        return report
    
    # ==================== GRAFICI ====================
    
    def generate_chart_image(self, df: pd.DataFrame, ticker: str, period: str = "3mo") -> Optional[io.BytesIO]:
        """Genera grafico avanzato dark mode"""
        try:
            if df.empty or len(df) < 10:
                return None
            
            # Configurazione stile
            plt.rcParams.update({
                'axes.facecolor': '#0d1117',
                'figure.facecolor': '#0d1117',
                'axes.edgecolor': '#30363d',
                'axes.labelcolor': '#c9d1d9',
                'text.color': '#c9d1d9',
                'xtick.color': '#8b949e',
                'ytick.color': '#8b949e',
                'grid.color': '#30363d',
                'grid.alpha': 0.3,
            })
            
            # Figura principale
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), 
                                                facecolor='#0d1117',
                                                height_ratios=[3, 1, 1])
            
            # 1. PREZZI
            ax1.plot(df.index, df['Close'], 
                    color='#58a6ff', 
                    linewidth=2.5,
                    label='Prezzo')
            
            if 'SMA_20' in df.columns:
                ax1.plot(df.index, df['SMA_20'], 
                        color='#f78166',
                        linewidth=1.5,
                        alpha=0.8,
                        label='SMA 20')
            
            if 'SMA_50' in df.columns:
                ax1.plot(df.index, df['SMA_50'], 
                        color='#56d364',
                        linewidth=1.5,
                        alpha=0.8,
                        label='SMA 50')
            
            ax1.set_title(f'{ticker} - Analisi Tecnica', 
                         fontsize=16, 
                         fontweight='bold',
                         pad=20)
            ax1.set_ylabel('Prezzo ($)', fontsize=12)
            ax1.grid(True, alpha=0.2)
            ax1.legend(loc='upper left', fontsize=9, framealpha=0.3)
            
            # 2. RSI
            if 'RSI' in df.columns:
                ax2.plot(df.index, df['RSI'], 
                        color='#d29922',
                        linewidth=2,
                        label='RSI')
                ax2.axhline(y=70, color='#da3633', linestyle='--', alpha=0.5)
                ax2.axhline(y=30, color='#238636', linestyle='--', alpha=0.5)
                ax2.fill_between(df.index, 70, 100, color='#da3633', alpha=0.1)
                ax2.fill_between(df.index, 0, 30, color='#238636', alpha=0.1)
                ax2.set_ylabel('RSI', fontsize=11)
                ax2.set_ylim(0, 100)
                ax2.grid(True, alpha=0.2)
                ax2.legend(loc='upper left', framealpha=0.3)
            
            # 3. MACD
            if all(col in df.columns for col in ['MACD', 'MACD_signal']):
                ax3.plot(df.index, df['MACD'], 
                        color='#58a6ff',
                        linewidth=1.5,
                        label='MACD')
                ax3.plot(df.index, df['MACD_signal'], 
                        color='#f78166',
                        linewidth=1.5,
                        label='Signal',
                        linestyle='--')
                ax3.axhline(y=0, color='#8b949e', linestyle='-', alpha=0.5)
                ax3.set_ylabel('MACD', fontsize=11)
                ax3.set_xlabel('Data', fontsize=11)
                ax3.grid(True, alpha=0.2)
                ax3.legend(loc='upper left', framealpha=0.3)
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Info box
            latest = df.iloc[-1]
            info_text = f"""
Prezzo: ${latest['Close']:.2f}
RSI: {latest.get('RSI', 0):.1f}
MACD: {latest.get('MACD', 0):.3f}
Periodo: {period}
"""
            ax1.text(0.98, 0.98, info_text,
                    transform=ax1.transAxes,
                    fontsize=9,
                    verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='#161b22', alpha=0.9))
            
            # Salva
            buf = io.BytesIO()
            plt.savefig(buf, 
                       format='png', 
                       dpi=120, 
                       bbox_inches='tight',
                       facecolor='#0d1117')
            buf.seek(0)
            plt.close(fig)
            
            return buf
            
        except Exception as e:
            self.logger.error(f"Errore generazione grafico: {e}")
            plt.close('all')
            return None
    
    def generate_simple_chart(self, df: pd.DataFrame, ticker: str) -> Optional[io.BytesIO]:
        """Grafico semplice veloce"""
        try:
            if df.empty:
                return None
            
            plt.rcParams.update({
                'axes.facecolor': '#0d1117',
                'figure.facecolor': '#0d1117',
            })
            
            fig, ax = plt.subplots(figsize=(10, 6), facecolor='#0d1117')
            
            ax.plot(df.index, df['Close'], 
                   color='#58a6ff', 
                   linewidth=2.5,
                   label=ticker)
            
            latest_price = df.iloc[-1]['Close']
            change = ((latest_price - df.iloc[0]['Close']) / df.iloc[0]['Close'] * 100)
            
            ax.set_title(f'{ticker} - ${latest_price:.2f} ({change:+.1f}%)', 
                        fontsize=14, 
                        fontweight='bold',
                        pad=20,
                        color='white')
            ax.set_ylabel('Prezzo ($)', fontsize=11, color='white')
            ax.set_xlabel('Data', fontsize=11, color='white')
            ax.tick_params(colors='white')
            ax.grid(True, alpha=0.2, color='gray')
            ax.legend(loc='upper left', framealpha=0.3)
            plt.xticks(rotation=45, color='white')
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, 
                       format='png', 
                       dpi=100, 
                       bbox_inches='tight',
                       facecolor='#0d1117')
            buf.seek(0)
            plt.close(fig)
            
            return buf
            
        except Exception as e:
            self.logger.error(f"Errore grafico semplice: {e}")
            plt.close('all')
            return None