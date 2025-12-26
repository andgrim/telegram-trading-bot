"""
Modulo per la ricerca intelligente di ticker
"""
import yfinance as yf
import requests
import json
import re
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

class TickerSearcher:
    """Sistema di ricerca ticker in tempo reale"""
    
    def __init__(self):
        self.yahoo_search_url = "https://query2.finance.yahoo.com/v1/finance/search"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def search_yahoo(self, query: str, max_results: int = 10) -> List[Dict]:
        """Cerca ticker su Yahoo Finance"""
        try:
            params = {
                'q': query,
                'quotesCount': max_results,
                'newsCount': 0,
                'enableFuzzyQuery': True,
                'quotesQueryId': 'tss_match_phrase_query'
            }
            
            response = requests.get(
                self.yahoo_search_url, 
                params=params, 
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                quotes = data.get('quotes', [])
                
                # Filtra solo risultati validi
                valid_quotes = []
                for quote in quotes:
                    if (quote.get('symbol') and 
                        quote.get('quoteType') in ['EQUITY', 'ETF', 'INDEX', 'CRYPTOCURRENCY']):
                        
                        valid_quotes.append({
                            'symbol': quote['symbol'],
                            'name': quote.get('longname') or quote.get('shortname', ''),
                            'exchange': quote.get('exchange', ''),
                            'type': quote.get('quoteType', ''),
                            'score': quote.get('score', 0)
                        })
                
                return valid_quotes[:max_results]
            
        except Exception as e:
            logger.error(f"Errore ricerca Yahoo: {e}")
        
        return []
    
    def search_local(self, query: str) -> List[Dict]:
        """Ricerca locale per errori comuni"""
        query = query.upper().strip()
        
        # Mappatura completa
        local_map = {
            # USA
            'APPLE': ('AAPL', 'Apple Inc.'),
            'TESLA': ('TSLA', 'Tesla Inc.'),
            'MICROSOFT': ('MSFT', 'Microsoft Corporation'),
            'GOOGLE': ('GOOGL', 'Alphabet Inc.'),
            'AMAZON': ('AMZN', 'Amazon.com Inc.'),
            'META': ('META', 'Meta Platforms Inc.'),
            'NVIDIA': ('NVDA', 'NVIDIA Corporation'),
            'NETFLIX': ('NFLX', 'Netflix Inc.'),
            'ADOBE': ('ADBE', 'Adobe Inc.'),
            'INTEL': ('INTC', 'Intel Corporation'),
            'AMD': ('AMD', 'Advanced Micro Devices Inc.'),
            'IBM': ('IBM', 'International Business Machines'),
            'COCA COLA': ('KO', 'The Coca-Cola Company'),
            'PEPSI': ('PEP', 'PepsiCo Inc.'),
            'MCDONALDS': ('MCD', "McDonald's Corporation"),
            'NIKE': ('NKE', 'NIKE Inc.'),
            'WALMART': ('WMT', 'Walmart Inc.'),
            'VISA': ('V', 'Visa Inc.'),
            'MASTERCARD': ('MA', 'Mastercard Incorporated'),
            'JPMORGAN': ('JPM', 'JPMorgan Chase & Co.'),
            'BANK OF AMERICA': ('BAC', 'Bank of America Corporation'),
            'WELLS FARGO': ('WFC', 'Wells Fargo & Company'),
            'BERKSHIRE HATHAWAY': ('BRK-B', 'Berkshire Hathaway Inc.'),
            'JOHNSON & JOHNSON': ('JNJ', 'Johnson & Johnson'),
            'PFIZER': ('PFE', 'Pfizer Inc.'),
            'MODERNA': ('MRNA', 'Moderna Inc.'),
            
            # Italia
            'ENEL': ('ENEL.MI', 'Enel SpA'),
            'ENI': ('ENI.MI', 'Eni SpA'),
            'INTESA SANPAOLO': ('ISP.MI', 'Intesa Sanpaolo'),
            'INTESA': ('ISP.MI', 'Intesa Sanpaolo'),
            'UNICREDIT': ('UCG.MI', 'UniCredit SpA'),
            'STELLANTIS': ('STLA', 'Stellantis NV'),
            'FIAT': ('STLA', 'Stellantis NV'),
            'FERRARI': ('RACE', 'Ferrari N.V.'),
            'GENERALI': ('G.MI', 'Assicurazioni Generali SpA'),
            'TELECOM ITALIA': ('TIT.MI', 'Telecom Italia S.p.A.'),
            'TELECOM': ('TIT.MI', 'Telecom Italia S.p.A.'),
            'MONCLER': ('MONC.MI', 'Moncler S.p.A.'),
            'PRADA': ('1913.HK', 'Prada S.p.A.'),
            'SAIPEM': ('SPM.MI', 'Saipem S.p.A.'),
            'LEONARDO': ('LDO.MI', 'Leonardo S.p.A.'),
            
            # Germania
            'VOLKSWAGEN': ('VOW3.DE', 'Volkswagen AG'),
            'BMW': ('BMW.DE', 'Bayerische Motoren Werke AG'),
            'MERCEDES': ('MBG.DE', 'Mercedes-Benz Group AG'),
            'SAP': ('SAP.DE', 'SAP SE'),
            'SIEMENS': ('SIE.DE', 'Siemens AG'),
            'ALLIANZ': ('ALV.DE', 'Allianz SE'),
            'DEUTSCHE BANK': ('DBK.DE', 'Deutsche Bank AG'),
            'ADIDAS': ('ADS.DE', 'adidas AG'),
            'BASF': ('BAS.DE', 'BASF SE'),
            'BAYER': ('BAYN.DE', 'Bayer AG'),
            
            # Francia
            'LVMH': ('MC.PA', 'LVMH Moët Hennessy Louis Vuitton SE'),
            'L OREAL': ('OR.PA', "L'Oréal S.A."),
            'SANOFI': ('SAN.PA', 'Sanofi'),
            'TOTAL': ('TTE.PA', 'TotalEnergies SE'),
            'BNP PARIBAS': ('BNP.PA', 'BNP Paribas SA'),
            'AIRBUS': ('AIR.PA', 'Airbus SE'),
            'SCHNEIDER ELECTRIC': ('SU.PA', 'Schneider Electric SE'),
            
            # UK
            'HSBC': ('HSBA.L', 'HSBC Holdings plc'),
            'BP': ('BP.L', 'BP p.l.c.'),
            'SHELL': ('SHEL.L', 'Shell plc'),
            'ASTRAZENECA': ('AZN.L', 'AstraZeneca PLC'),
            'GLAXOSMITHKLINE': ('GSK.L', 'GSK plc'),
            'UNILEVER': ('ULVR.L', 'Unilever PLC'),
            'DIAGEO': ('DGE.L', 'Diageo plc'),
            'BARCLAYS': ('BARC.L', 'Barclays PLC'),
            'LLOYDS': ('LLOY.L', 'Lloyds Banking Group plc'),
            
            # Cripto
            'BITCOIN': ('BTC-USD', 'Bitcoin USD'),
            'ETHEREUM': ('ETH-USD', 'Ethereum USD'),
            'BNB': ('BNB-USD', 'Binance Coin USD'),
            'SOLANA': ('SOL-USD', 'Solana USD'),
            'CARDANO': ('ADA-USD', 'Cardano USD'),
            'DOGECOIN': ('DOGE-USD', 'Dogecoin USD'),
            'RIPPLE': ('XRP-USD', 'Ripple USD'),
            'POLKADOT': ('DOT-USD', 'Polkadot USD'),
            
            # ETF e Indici
            'S&P500': ('SPY', 'SPDR S&P 500 ETF Trust'),
            'SP500': ('SPY', 'SPDR S&P 500 ETF Trust'),
            'SP 500': ('SPY', 'SPDR S&P 500 ETF Trust'),
            'NASDAQ': ('QQQ', 'Invesco QQQ Trust'),
            'DOW JONES': ('DIA', 'SPDR Dow Jones Industrial Average ETF Trust'),
            'DOW': ('DIA', 'SPDR Dow Jones Industrial Average ETF Trust'),
            'RUSSELL 2000': ('IWM', 'iShares Russell 2000 ETF'),
            'VIX': ('VXX', 'iPath Series B S&P 500 VIX Short-Term Futures ETN'),
            'GOLD': ('GLD', 'SPDR Gold Shares'),
            'SILVER': ('SLV', 'iShares Silver Trust'),
            'OIL': ('USO', 'United States Oil Fund LP'),
            'GAS NATURALE': ('UNG', 'United States Natural Gas Fund LP'),
            
            # Indici (simboli)
            'S&P 500 INDEX': ('^GSPC', 'S&P 500'),
            'NASDAQ COMPOSITE': ('^IXIC', 'NASDAQ Composite'),
            'DOW JONES INDUSTRIAL': ('^DJI', 'Dow Jones Industrial Average'),
            'FTSE 100': ('^FTSE', 'FTSE 100 Index'),
            'DAX': ('^GDAXI', 'DAX Performance Index'),
            'CAC 40': ('^FCHI', 'CAC 40 Index'),
            'NIKKEI 225': ('^N225', 'Nikkei 225'),
            'HANG SENG': ('^HSI', 'Hang Seng Index'),
            
            # Valute
            'EURO USD': ('EURUSD=X', 'EUR/USD'),
            'USD JPY': ('JPY=X', 'USD/JPY'),
            'GBP USD': ('GBPUSD=X', 'GBP/USD'),
            'USD CHF': ('CHF=X', 'USD/CHF'),
            'AUD USD': ('AUDUSD=X', 'AUD/USD'),
            'USD CAD': ('CAD=X', 'USD/CAD'),
        }
        
        results = []
        for key, (symbol, name) in local_map.items():
            if query in key or key in query or query == symbol:
                results.append({
                    'symbol': symbol,
                    'name': name,
                    'exchange': 'AUTO',
                    'type': 'EQUITY',
                    'score': 100
                })
        
        return results
    
    def validate_ticker(self, ticker: str) -> Tuple[bool, Dict]:
        """Verifica se un ticker esiste e ottieni info"""
        try:
            ticker = ticker.strip().upper()
            stock = yf.Ticker(ticker)
            info = stock.info
            
            if info and 'symbol' in info:
                return True, {
                    'symbol': info['symbol'],
                    'name': info.get('longName') or info.get('shortName', ticker),
                    'exchange': info.get('exchange', ''),
                    'currency': info.get('currency', 'USD'),
                    'type': info.get('quoteType', 'EQUITY')
                }
        
        except Exception as e:
            logger.warning(f"Ticker {ticker} non valido: {e}")
        
        return False, {}
    
    def search_tickers(self, query: str) -> List[Dict]:
        """Ricerca completa ticker"""
        if not query or len(query) < 2:
            return []
        
        # Prima cerca su Yahoo Finance
        yahoo_results = self.search_yahoo(query)
        
        # Poi cerca localmente
        local_results = self.search_local(query)
        
        # Combina risultati, evitando duplicati
        all_results = []
        seen_symbols = set()
        
        # Aggiungi prima risultati Yahoo
        for result in yahoo_results:
            if result['symbol'] not in seen_symbols:
                all_results.append(result)
                seen_symbols.add(result['symbol'])
        
        # Aggiungi risultati locali se non già presenti
        for result in local_results:
            if result['symbol'] not in seen_symbols:
                all_results.append(result)
                seen_symbols.add(result['symbol'])
        
        # Ordina per rilevanza (score)
        all_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        return all_results[:10]  # Massimo 10 risultati
    
    def format_search_results(self, results: List[Dict]) -> str:
        """Formatta risultati per Telegram"""
        if not results:
            return "❌ Nessun risultato trovato. Prova con:\n• Nome azienda (es: Apple)\n• Ticker (es: AAPL)\n• ETF (es: SPY)"
        
        formatted = "🔍 *RISULTATI RICERCA:*\n\n"
        
        for i, result in enumerate(results[:8], 1):  # Mostra max 8 risultati
            symbol = result['symbol']
            name = result['name'][:40] + "..." if len(result['name']) > 40 else result['name']
            exchange = result.get('exchange', '')
            
            formatted += f"{i}. *{symbol}* - {name}\n"
            if exchange and exchange != 'AUTO':
                formatted += f"   📍 {exchange}\n"
        
        formatted += "\n📌 *Come usare:*\n"
        formatted += "Clicca su un ticker per analisi completa\n"
        
        return formatted
    
    def get_ticker_info(self, ticker: str) -> Optional[Dict]:
        """Ottieni informazioni dettagliate su un ticker"""
        try:
            ticker = ticker.strip().upper()
            stock = yf.Ticker(ticker)
            info = stock.info
            
            if not info or 'symbol' not in info:
                return None
            
            # Estrai info importanti
            return {
                'symbol': info.get('symbol', ticker),
                'name': info.get('longName') or info.get('shortName', ticker),
                'exchange': info.get('exchange', 'Unknown'),
                'currency': info.get('currency', 'USD'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'country': info.get('country', 'N/A'),
                'marketCap': info.get('marketCap'),
                'currentPrice': info.get('currentPrice'),
                'previousClose': info.get('previousClose'),
                'dayHigh': info.get('dayHigh'),
                'dayLow': info.get('dayLow'),
                'volume': info.get('volume'),
                'fiftyDayAverage': info.get('fiftyDayAverage'),
                'twoHundredDayAverage': info.get('twoHundredDayAverage'),
                'website': info.get('website'),
                'longBusinessSummary': info.get('longBusinessSummary', '')[:300] + "..."
            }
        
        except Exception as e:
            logger.error(f"Errore ottenimento info ticker {ticker}: {e}")
            return None
    
    def smart_search(self, query: str) -> List[Dict]:
        """Ricerca intelligente con correzione errori"""
        query = query.strip()
        
        # Correzione errori comuni
        corrections = {
            'appel': 'apple',
            'apl': 'apple',
            'googl': 'google',
            'goog': 'google',
            'micorsoft': 'microsoft',
            'msft': 'microsoft',
            'tsla': 'tesla',
            'nvdia': 'nvidia',
            'nvda': 'nvidia',
            'amzn': 'amazon',
            'meta': 'facebook',  # Per vecchi nomi
            'fb': 'facebook',
            'enel': 'enel',
            'eni': 'eni',
            'intesa': 'intesa sanpaolo',
            'unicredit': 'unicredit',
            'ferrari': 'ferrari',
            'stellantis': 'stellantis',
            'bitcoin': 'bitcoin',
            'btc': 'bitcoin',
            'ethereum': 'ethereum',
            'eth': 'ethereum',
            'sp500': 's&p500',
            'spy': 's&p500',
            'qqq': 'nasdaq',
            'gold': 'gold',
            'gld': 'gold',
        }
        
        # Applica correzioni
        corrected_query = corrections.get(query.lower(), query)
        
        # Esegui ricerca
        return self.search_tickers(corrected_query)