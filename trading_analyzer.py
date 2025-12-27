import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from utils import get_stock_data, calculate_technical_indicators, search_tickers, get_ticker_info  # <-- Aggiungi tutti

class TradingAnalyzer:
    def __init__(self, symbol: str):
        self.symbol = symbol.upper()
    
        
    def analyze_period(self, period_months: int = 3):
        """Analizza un periodo specifico"""
        df = get_stock_data(self.symbol, period_months)
        if df.empty:
            return None
        
        df = calculate_technical_indicators(df)
        return df
    
    def generate_summary(self, df_3m: pd.DataFrame, df_6m: pd.DataFrame) -> dict:
        """Genera riepilogo analisi"""
        if df_3m is None or df_6m is None:
            return {}
        
        latest_price = df_3m['Close'].iloc[-1]
        
        summary = {
            'symbol': self.symbol,
            'current_price': latest_price,
            'price_change_3m': self._calculate_price_change(df_3m),
            'price_change_6m': self._calculate_price_change(df_6m),
            'rsi_3m': df_3m['RSI'].iloc[-1],
            'rsi_6m': df_6m['RSI'].iloc[-1],
            'macd_signal_3m': self._get_macd_signal(df_3m),
            'macd_signal_6m': self._get_macd_signal(df_6m),
            'volume_trend': self._analyze_volume_trend(df_3m),
        }
        
        return summary
    
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
    
    def _analyze_volume_trend(self, df: pd.DataFrame) -> str:
        """Analizza trend volume"""
        avg_volume = df['Volume'].mean()
        recent_volume = df['Volume'].iloc[-5:].mean()
        
        if recent_volume > avg_volume * 1.2:
            return "HIGH"
        elif recent_volume < avg_volume * 0.8:
            return "LOW"
        return "NORMAL"
    
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