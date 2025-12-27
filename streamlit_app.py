import streamlit as st
import plotly.graph_objects as go
from trading_analyzer import TradingAnalyzer
from utils import search_tickers, get_ticker_info
import pandas as pd

# Configurazione pagina
st.set_page_config(
    page_title="Telegram Trading Bot - Web Analysis",
    page_icon="📈",
    layout="wide"
)

# CSS per tema scuro con miglioramenti
st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
        color: white;
    }
    
    .metric-card {
        background-color: #262730;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #FF4B4B;
    }
    
    .ticker-card {
        background-color: #1E1E1E;
        padding: 15px;
        border-radius: 8px;
        margin: 8px 0;
        border: 1px solid #444;
        cursor: pointer;
        transition: all 0.3s;
    }
    
    .ticker-card:hover {
        background-color: #2D2D2D;
        border-color: #FF4B4B;
    }
    
    .positive {
        color: #00FF00;
    }
    
    .negative {
        color: #FF0000;
    }
    
    .search-result {
        background-color: #262730;
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #262730;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
    }
    
    /* Migliora la visualizzazione delle tabelle */
    .stDataFrame {
        background-color: #262730;
        border-radius: 8px;
    }
    
    /* Migliora input */
    .stTextInput > div > div > input {
        background-color: #262730;
        color: white;
        border: 1px solid #444;
    }
    
    .stSelectbox > div > div {
        background-color: #262730;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

def display_ticker_info(ticker_info: dict):
    """Mostra informazioni dettagliate del ticker"""
    if not ticker_info:
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Prezzo Attuale", 
                 f"${ticker_info.get('current_price', 'N/A'):.2f}" 
                 if isinstance(ticker_info.get('current_price'), (int, float)) 
                 else ticker_info.get('current_price', 'N/A'))
    
    with col2:
        prev_close = ticker_info.get('previous_close')
        current = ticker_info.get('current_price')
        if isinstance(prev_close, (int, float)) and isinstance(current, (int, float)):
            change = ((current - prev_close) / prev_close) * 100
            st.metric("Variazione Giornaliera", 
                     f"{change:+.2f}%",
                     delta=f"{change:+.2f}%")
        else:
            st.metric("Variazione Giornaliera", "N/A")
    
    with col3:
        st.metric("Volume", 
                 f"{ticker_info.get('volume', 0):,}" 
                 if ticker_info.get('volume') 
                 else "N/A")
    
    with col4:
        st.metric("Market Cap", 
                 f"${ticker_info.get('market_cap', 0):,.0f}" 
                 if isinstance(ticker_info.get('market_cap'), (int, float)) 
                 else "N/A")
    
    # Altre informazioni
    with st.expander("📋 Dettagli Azienda"):
        cols = st.columns(3)
        with cols[0]:
            st.write(f"**Settore:** {ticker_info.get('sector', 'N/A')}")
            st.write(f"**Industria:** {ticker_info.get('industry', 'N/A')}")
            st.write(f"**Paese:** {ticker_info.get('country', 'N/A')}")
        
        with cols[1]:
            st.write(f"**P/E Ratio:** {ticker_info.get('pe_ratio', 'N/A')}")
            st.write(f"**Dividend Yield:** {ticker_info.get('dividend_yield', 'N/A')}")
            st.write(f"**Beta:** {ticker_info.get('beta', 'N/A')}")
        
        with cols[2]:
            st.write(f"**52W High:** ${ticker_info.get('52w_high', 'N/A')}")
            st.write(f"**52W Low:** ${ticker_info.get('52w_low', 'N/A')}")
            st.write(f"**Valuta:** {ticker_info.get('currency', 'N/A')}")

def main():
    # Header
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.title("🔍 Telegram Trading Bot - Analisi Mercato")
        st.markdown("---")
    
    # Sidebar con ricerca avanzata
    with st.sidebar:
        st.header("🔎 Ricerca Ticker")
        
        # Modalità di ricerca
        search_mode = st.radio(
            "Modalità ricerca:",
            ["🔤 Ricerca per Nome", "⚡ Inserimento Diretto"],
            horizontal=True,
            label_visibility="collapsed"
        )
        
        symbol = ""
        
        if search_mode == "🔤 Ricerca per Nome":
            # Ricerca per nome/simbolo
            search_query = st.text_input(
                "Cerca azienda o simbolo:",
                placeholder="Es: Apple, AAPL, Tesla...",
                key="search_query"
            )
            
            if search_query:
                with st.spinner("Ricerca in corso..."):
                    results = search_tickers(search_query, limit=10)
                    
                if results:
                    st.success(f"Trovati {len(results)} risultati")
                    
                    # Mostra risultati
                    for result in results:
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.markdown(f"**`{result['symbol']}`**")
                        with col2:
                            st.write(result['name'])
                        
                        # Pulsante per selezionare
                        if st.button(f"Seleziona {result['symbol']}", 
                                   key=f"select_{result['symbol']}"):
                            st.session_state.selected_symbol = result['symbol']
                            st.rerun()
                    
                    st.markdown("---")
                else:
                    st.warning("Nessun risultato trovato")
        
        else:  # Inserimento diretto
            symbol = st.text_input(
                "Simbolo Ticker:",
                value=st.session_state.get('selected_symbol', 'AAPL'),
                placeholder="Es: AAPL, TSLA, BTC-USD",
                key="direct_symbol"
            ).upper()
        
        # Periodi di analisi
        st.markdown("---")
        st.header("⚙️ Configurazione Analisi")
        
        periods = st.multiselect(
            "Periodi di analisi:",
            options=[1, 3, 6, 12],
            default=[3, 6],
            format_func=lambda x: f"{x} mesi"
        )
        
        # Intervallo temporale
        interval = st.selectbox(
            "Intervallo dati:",
            options=["1d", "1wk", "1mo"],
            index=0,
            help="Intervallo temporale dei dati (giornaliero, settimanale, mensile)"
        )
        
        # Indicatori aggiuntivi
        st.markdown("---")
        st.header("📊 Indicatori")
        
        show_indicators = st.multiselect(
            "Indicatori da mostrare:",
            options=["BB", "Stochastic", "ATR", "Volume Profile"],
            default=["BB"],
            help="Seleziona indicatori aggiuntivi da visualizzare"
        )
        
        # Ticker suggeriti
        st.markdown("---")
        st.header("🎯 Ticker Popolari")
        
        popular_tickers = {
            "AAPL": "Apple Inc.",
            "MSFT": "Microsoft",
            "GOOGL": "Alphabet",
            "TSLA": "Tesla",
            "AMZN": "Amazon",
            "META": "Meta Platforms",
            "NVDA": "NVIDIA",
            "JPM": "JPMorgan Chase",
            "V": "Visa",
            "JNJ": "Johnson & Johnson"
        }
        
        cols = st.columns(2)
        for idx, (tick, name) in enumerate(popular_tickers.items()):
            with cols[idx % 2]:
                if st.button(f"**{tick}**\n{name[:15]}...", 
                           key=f"pop_{tick}",
                           use_container_width=True):
                    st.session_state.selected_symbol = tick
                    st.rerun()
    
    # Contenuto principale
    if symbol or st.session_state.get('selected_symbol'):
        current_symbol = symbol if symbol else st.session_state.selected_symbol
        
        try:
            # Informazioni del ticker
            with st.spinner(f"Recupero informazioni per {current_symbol}..."):
                ticker_info = get_ticker_info(current_symbol)
            
            if ticker_info:
                # Header con informazioni
                st.subheader(f"📊 {ticker_info.get('name', current_symbol)} ({current_symbol})")
                display_ticker_info(ticker_info)
                
                # Analisi tecnica
                st.markdown("---")
                st.subheader("📈 Analisi Tecnica")
                
                analyzer = TradingAnalyzer(current_symbol)
                
                # Analisi per ogni periodo
                all_data = {}
                summary_data = []
                
                for period in sorted(periods):
                    with st.spinner(f"Analisi {period} mesi..."):
                        df = analyzer.analyze_period(period)
                        
                        if df is not None and not df.empty:
                            all_data[period] = df
                            
                            # Calcola metriche
                            price_change = ((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1) * 100
                            current_rsi = df['RSI'].iloc[-1]
                            macd_signal = "↑" if df['MACD'].iloc[-1] > 0 else "↓"
                            
                            summary_data.append({
                                'Periodo': f'{period} mesi',
                                'Prezzo': f"${df['Close'].iloc[-1]:.2f}",
                                'Variazione': f"{price_change:+.2f}%",
                                'RSI': f"{current_rsi:.1f}",
                                'MACD': macd_signal,
                                'Volume Trend': analyzer._analyze_volume_trend(df)
                            })
                
                if summary_data:
                    # Tabella riepilogativa
                    st.dataframe(
                        pd.DataFrame(summary_data),
                        width='stretch',  # CORRETTO: sostituito use_container_width
                        hide_index=True
                    )
                    
                    # Seleziona periodo per grafico dettagliato
                    st.markdown("### 📊 Grafico Interattivo")
                    
                    selected_period = st.selectbox(
                        "Seleziona periodo per grafico dettagliato:",
                        options=sorted(periods),
                        format_func=lambda x: f"{x} mesi",
                        key="detailed_chart"
                    )
                    
                    if selected_period in all_data:
                        df = all_data[selected_period]
                        
                        # Tabs per diversi tipi di visualizzazione
                        tab1, tab2, tab3, tab4 = st.tabs([
                            "📈 Grafico Completo", 
                            "🔄 Confronto Periodi",
                            "📋 Dati Raw", 
                            "🎯 Segnali Trading"
                        ])
                        
                        with tab1:
                            # Configurazione grafico
                            col1, col2 = st.columns([3, 1])
                            with col2:
                                show_sma = st.checkbox("Medie Mobili", value=True)
                                show_bollinger = st.checkbox("Bollinger Bands", 
                                                           value="BB" in show_indicators)
                                show_volume = st.checkbox("Volume", value=True)
                            
                            fig = analyzer.create_chart(df, f"{selected_period}")
                            
                            # Personalizza grafico in base alle selezioni
                            if not show_volume:
                                fig.update_traces(selector=dict(name="Volume"), visible=False)
                            
                            st.plotly_chart(fig, width='stretch')  # CORRETTO
                        
                        with tab2:
                            # Confronto tra periodi
                            if len(all_data) >= 2:
                                st.write("### 📊 Confronto Prezzi tra Periodi")
                                
                                fig_comparison = go.Figure()
                                
                                for period, data in all_data.items():
                                    fig_comparison.add_trace(
                                        go.Scatter(
                                            x=data.index,
                                            y=data['Close'],
                                            name=f'{period} mesi',
                                            mode='lines'
                                        )
                                    )
                                
                                fig_comparison.update_layout(
                                    title=f'Confronto Prezzi {current_symbol}',
                                    template='plotly_dark',
                                    height=500
                                )
                                
                                st.plotly_chart(fig_comparison, width='stretch')  # CORRETTO
                            else:
                                st.info("Seleziona almeno 2 periodi per il confronto")
                        
                        with tab3:
                            # Dati grezzi
                            st.write(f"### 📄 Dati Storici - {selected_period} mesi")
                            
                            # Filtri dati
                            col1, col2 = st.columns(2)
                            with col1:
                                rows_to_show = st.slider("Righe da mostrare", 
                                                        min_value=10, 
                                                        max_value=100, 
                                                        value=20)
                            
                            with col2:
                                show_columns = st.multiselect(
                                    "Colonne da mostrare",
                                    options=df.columns.tolist(),
                                    default=['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD']
                                )
                            
                            st.dataframe(
                                df[show_columns].tail(rows_to_show),
                                width='stretch'  # CORRETTO
                            )
                            
                            # Download dati
                            csv = df.to_csv()
                            st.download_button(
                                label="📥 Scarica tutti i dati (CSV)",
                                data=csv,
                                file_name=f"{current_symbol}_{selected_period}m_full.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                        
                        with tab4:
                            # Analisi segnali trading
                            st.write("### 🎯 Analisi Segnali Trading")
                            
                            latest = df.iloc[-1]
                            
                            # Creazione segnali
                            signals = []
                            score = 0
                            
                            # RSI
                            if latest['RSI'] < 30:
                                signals.append(("🟢 RSI", "Sovravenduto", 1))
                                score += 1
                            elif latest['RSI'] > 70:
                                signals.append(("🔴 RSI", "Sovracomprato", -1))
                                score -= 1
                            else:
                                signals.append(("⚪ RSI", "Neutrale", 0))
                            
                            # MACD
                            if latest['MACD'] > latest['Signal_Line']:
                                signals.append(("🟢 MACD", "Rialzista", 1))
                                score += 1
                            else:
                                signals.append(("🔴 MACD", "Ribassista", -1))
                                score -= 1
                            
                            # Medie Mobili
                            if latest['SMA_20'] > latest['SMA_50']:
                                signals.append(("🟢 Medie Mobili", "Trend ↑", 1))
                                score += 1
                            else:
                                signals.append(("🔴 Medie Mobili", "Trend ↓", -1))
                                score -= 1
                            
                            # Volume
                            volume_trend = analyzer._analyze_volume_trend(df)
                            if volume_trend == "HIGH":
                                signals.append(("🟢 Volume", "Volume alto", 1))
                                score += 1
                            elif volume_trend == "LOW":
                                signals.append(("🔴 Volume", "Volume basso", -1))
                                score -= 1
                            else:
                                signals.append(("⚪ Volume", "Normale", 0))
                            
                            # Mostra segnali
                            cols = st.columns(len(signals))
                            for idx, (icon, text, _) in enumerate(signals):
                                with cols[idx]:
                                    st.markdown(f"""
                                    <div style="text-align: center; padding: 10px; border-radius: 5px; background: #262730;">
                                        <div style="font-size: 24px;">{icon}</div>
                                        <div>{text}</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                            
                            # Score complessivo
                            st.markdown("---")
                            st.subheader("📊 Punteggio Segnali")
                            
                            if score > 1:
                                st.success(f"""
                                🎯 **FORTE SEGNALE ACQUISTO**
                                
                                **Punteggio:** +{score}
                                **Raccomandazione:** Considera posizione long
                                """)
                            elif score > 0:
                                st.info(f"""
                                📈 **LEGGERO SEGNALE ACQUISTO**
                                
                                **Punteggio:** +{score}
                                **Raccomandazione:** Monitora per entry
                                """)
                            elif score < -1:
                                st.error(f"""
                                ⚠️ **FORTE SEGNALE VENDITA**
                                
                                **Punteggio:** {score}
                                **Raccomandazione:** Considera posizione short
                                """)
                            elif score < 0:
                                st.warning(f"""
                                📉 **LEGGERO SEGNALE VENDITA**
                                
                                **Punteggio:** {score}
                                **Raccomandazione:** Cautela in acquisto
                                """)
                            else:
                                st.info(f"""
                                ⚖️ **SEGNALE NEUTRO**
                                
                                **Punteggio:** {score}
                                **Raccomandazione:** Attendi segnali più chiari
                                """)
                            
                            # Disclaimer
                            st.markdown("---")
                            st.caption("""
                            ⚠️ **Disclaimer:** Questi segnali sono generati automaticamente e non costituiscono 
                            consigli finanziari. Fai sempre la tua ricerca e consulta un professionista 
                            prima di investire.
                            """)
                    
                    else:
                        st.warning("Dati non disponibili per il periodo selezionato")
                else:
                    st.error(f"Impossibile analizzare {current_symbol}. Verifica il simbolo e riprova.")
            
            else:
                st.error(f"Ticker {current_symbol} non trovato o dati non disponibili")
        
        except Exception as e:
            st.error(f"❌ Errore nell'analisi di {current_symbol}: {str(e)}")
            st.info("""
            Possibili cause:
            1. Simbolo errato
            2. Mercato chiuso
            3. Problemi di connessione
            4. Ticker non supportato
            """)
    else:
        # Schermata iniziale
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div style="text-align: center; padding: 50px;">
                <h1>🔍 Benvenuto!</h1>
                <p style="font-size: 18px;">Cerca un ticker nella sidebar per iniziare l'analisi</p>
                <p style="color: #888; margin-top: 30px;">
                    Utilizza la ricerca per nome o inserisci direttamente il simbolo
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Quick start guide
            with st.expander("📚 Come usare questa app"):
                st.markdown("""
                1. **🔎 Ricerca Ticker**: Usa la ricerca nella sidebar per trovare aziende
                2. **⚙️ Configura**: Scegli periodi e indicatori
                3. **📊 Analizza**: Esamina grafici e segnali
                4. **📥 Esporta**: Scarica i dati per ulteriori analisi
                
                **Esempi di ticker:**
                - Azioni: AAPL (Apple), TSLA (Tesla), MSFT (Microsoft)
                - ETF: SPY (S&P 500), QQQ (Nasdaq 100)
                - Cripto: BTC-USD (Bitcoin), ETH-USD (Ethereum)
                """)

if __name__ == "__main__":
    main()