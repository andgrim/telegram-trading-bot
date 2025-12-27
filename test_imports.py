#!/usr/bin/env python3
# test_imports.py - Verifica che tutti gli import funzionino

print("🔍 Test importazioni...")

try:
    from utils import get_stock_data, calculate_technical_indicators
    print("✅ utils.py importato correttamente")
except ImportError as e:
    print(f"❌ Errore import utils: {e}")

try:
    from trading_analyzer import TradingAnalyzer
    print("✅ trading_analyzer.py importato correttamente")
except ImportError as e:
    print(f"❌ Errore import trading_analyzer: {e}")

try:
    import pandas as pd
    import numpy as np
    import yfinance as yf
    import plotly.graph_objects as go
    import streamlit as st
    from telegram.ext import Application
    import dotenv
    import requests
    
    print("✅ Tutte le dipendenze esterne importate")
    
    # Test funzioni
    print("\n🔧 Test funzioni utils...")
    data = get_stock_data("AAPL", 1)
    if not data.empty:
        print("✅ get_stock_data funziona")
        data_with_indicators = calculate_technical_indicators(data)
        print("✅ calculate_technical_indicators funziona")
    else:
        print("⚠️  Nessun dato ricevuto, ma API funziona")
    
except Exception as e:
    print(f"❌ Errore generale: {e}")