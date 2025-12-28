import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from utils import get_stock_data, calculate_technical_indicators

class TradingAnalyzer:
    def __init__(self, symbol: str):
        self.symbol = symbol.upper()
    
    def analyze_period(self, period_months: int = 3):
        """Analizza per un periodo specifico"""
        try:
            # Ottieni dati
            df = get_stock_data(self.symbol, period_months)
            
            if df.empty:
                return None
            
            # Calcola indicatori
            df = calculate_technical_indicators(df)
            
            return df
            
        except Exception as e:
            print(f"Errore analisi {self.symbol}: {e}")
            return None
    
    def _analyze_volume_trend(self, df):
        """Analizza trend volume (semplificato)"""
        if len(df) < 10:
            return "N/A"
        
        avg_volume = df['Volume'].mean()
        recent_volume = df['Volume'].iloc[-5:].mean()
        
        if recent_volume > avg_volume * 1.2:
            return "ALTO"
        elif recent_volume < avg_volume * 0.8:
            return "BASSO"
        return "NORMALE"
    
    def _calculate_price_change(self, df: pd.DataFrame) -> float:
        """Calcola variazione prezzo"""
        if len(df) < 2:
            return 0
        return ((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1) * 100
    
    def _get_macd_signal(self, df: pd.DataFrame) -> str:
        """Determina segnale MACD"""
        if df['MACD'].iloc[-1] > df['Signal_Line'].iloc[-1]:
            return "BULLISH"
        elif df['MACD'].iloc[-1] < df['Signal_Line'].iloc[-1]:
            return "BEARISH"
        return "NEUTRAL"
    
    def create_chart(self, df: pd.DataFrame, period: str) -> go.Figure:
        """Crea grafico con indicatori"""
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.5, 0.15, 0.15, 0.15],
            subplot_titles=(f'{self.symbol} - Price & Moving Averages', 
                          'MACD', 'RSI', 'Volume')
        )
        
        # Prezzo e medie mobili
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Price'
            ), row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=df.index, y=df['SMA_20'], 
                      line=dict(color='orange', width=1),
                      name='SMA 20'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=df.index, y=df['SMA_50'], 
                      line=dict(color='blue', width=1),
                      name='SMA 50'),
            row=1, col=1
        )
        
        # MACD
        fig.add_trace(
            go.Scatter(x=df.index, y=df['MACD'],
                      line=dict(color='blue', width=1),
                      name='MACD'),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=df.index, y=df['Signal_Line'],
                      line=dict(color='red', width=1),
                      name='Signal Line'),
            row=2, col=1
        )
        
        # Istogramma MACD
        colors = ['green' if val >= 0 else 'red' for val in df['MACD_Histogram']]
        fig.add_trace(
            go.Bar(x=df.index, y=df['MACD_Histogram'],
                  marker_color=colors,
                  name='MACD Histogram'),
            row=2, col=1
        )
        
        # RSI
        fig.add_trace(
            go.Scatter(x=df.index, y=df['RSI'],
                      line=dict(color='purple', width=1),
                      name='RSI'),
            row=3, col=1
        )
        
        # Linee RSI
        fig.add_hline(y=70, line_dash="dash", line_color="red", 
                     opacity=0.5, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", 
                     opacity=0.5, row=3, col=1)
        
        # Volume
        colors_volume = ['green' if df['Close'].iloc[i] >= df['Open'].iloc[i] 
                        else 'red' for i in range(len(df))]
        fig.add_trace(
            go.Bar(x=df.index, y=df['Volume'],
                  marker_color=colors_volume,
                  name='Volume'),
            row=4, col=1
        )
        
        fig.update_layout(
            title=f'Technical Analysis - {period} months',
            xaxis_title='Date',
            yaxis_title='Price',
            template='plotly_dark',
            height=1000,
            showlegend=True
        )
        
        fig.update_xaxes(rangeslider_visible=False)
        
        return fig