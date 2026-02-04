"""
Ticker Search Module for Universal Trading Bot
Uses Yahoo Finance search API to find ticker symbols
"""
import requests
import json
from typing import Dict, List, Optional
import time

class TickerSearch:
    """Search for ticker symbols using Yahoo Finance API"""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://finance.yahoo.com/'
        }
        self.base_url = "https://query2.finance.yahoo.com/v1/finance/search"
    
    def search_ticker(self, query: str, max_results: int = 10) -> List[Dict]:
        """
        Search for ticker symbols based on query
        
        Args:
            query: Search term (company name, ticker, ETF name, etc.)
            max_results: Maximum number of results to return
            
        Returns:
            List of dictionaries with ticker information
        """
        try:
            # Prepare query parameters
            params = {
                'q': query,
                'lang': 'en-US',
                'region': 'US',
                'quotesCount': max_results,
                'newsCount': 0,
                'enableFuzzyQuery': True,
                'quotesQueryId': 'tss_match_phrase_query',
                'multiQuoteQueryId': 'multi_quote_single_token_query'
            }
            
            # Make request to Yahoo Finance search API
            response = requests.get(
                self.base_url, 
                params=params, 
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code != 200:
                print(f"Search API error: {response.status_code}")
                return []
            
            data = response.json()
            results = []
            
            # Process quotes
            if 'quotes' in data:
                for quote in data['quotes'][:max_results]:
                    if 'symbol' in quote:
                        symbol = quote['symbol']
                        
                        # Skip invalid symbols
                        if not symbol or symbol == 'null':
                            continue
                        
                        # Get name (prefer longname, fallback to shortname)
                        name = quote.get('longname') or quote.get('shortname') or 'Unknown'
                        
                        # Determine type
                        quote_type = quote.get('quoteType', '').upper()
                        type_display = self._format_type(quote_type)
                        
                        # Get exchange
                        exchange = quote.get('exchDisp', 'N/A')
                        
                        # Add to results
                        results.append({
                            'symbol': symbol,
                            'name': name[:50] + '...' if len(name) > 50 else name,
                            'exchange': exchange,
                            'type': type_display,
                            'score': quote.get('score', 0)
                        })
            
            # Sort by relevance score
            results.sort(key=lambda x: x.get('score', 0), reverse=True)
            
            print(f"✅ Found {len(results)} results for query: '{query}'")
            return results[:max_results]
            
        except requests.exceptions.Timeout:
            print(f"Search timeout for query: '{query}'")
            return []
        except Exception as e:
            print(f"Search error for '{query}': {e}")
            return []
    
    def search_etf_by_name(self, etf_name: str) -> Optional[str]:
        """
        Specialized search for ETFs by name
        
        Args:
            etf_name: ETF name or description
            
        Returns:
            Ticker symbol if found, None otherwise
        """
        try:
            # Common ETF mappings for quick results
            etf_mappings = {
                'nasdaq': 'QQQ',
                'nasdaq 100': 'QQQ',
                'qqq': 'QQQ',
                'sp500': 'SPY',
                's&p 500': 'SPY',
                'spy': 'SPY',
                'dow jones': 'DIA',
                'dow': 'DIA',
                'dia': 'DIA',
                'russell 2000': 'IWM',
                'russell': 'IWM',
                'iwm': 'IWM',
                'gold': 'GLD',
                'gld': 'GLD',
                'silver': 'SLV',
                'slv': 'SLV',
                'oil': 'USO',
                'uso': 'USO',
                'vanguard s&p 500': 'VOO',
                'voo': 'VOO',
                'technology': 'XLK',
                'xlk': 'XLK',
                'financial': 'XLF',
                'xlf': 'XLF',
                'healthcare': 'XLV',
                'xlv': 'XLV',
                'energy': 'XLE',
                'xle': 'XLE',
                'utilities': 'XLU',
                'xlu': 'XLU',
                'consumer staples': 'XLP',
                'xlp': 'XLP',
                'consumer discretionary': 'XLY',
                'xly': 'XLY',
                'materials': 'XLB',
                'xlb': 'XLB',
                'industrials': 'XLI',
                'xli': 'XLI',
                'real estate': 'VNQ',
                'vnq': 'VNQ',
                'bonds': 'BND',
                'bnd': 'BND',
                'treasury': 'TLT',
                'tlt': 'TLT',
                'emerging markets': 'EEM',
                'eem': 'EEM',
                'europe': 'VGK',
                'vgk': 'VGK',
                'japan': 'EWJ',
                'ewj': 'EWJ',
                'china': 'FXI',
                'fxi': 'FXI',
                'bitcoin': 'BITO',
                'bito': 'BITO',
                'crypto': 'BITO',
            }
            
            # Check direct mappings first
            query_lower = etf_name.lower().strip()
            for key, symbol in etf_mappings.items():
                if key in query_lower:
                    return symbol
            
            # Fall back to general search
            results = self.search_ticker(f"{etf_name} ETF", max_results=5)
            
            # Prioritize ETF results
            for result in results:
                if result['type'] == 'ETF':
                    return result['symbol']
            
            # Return first result if any
            if results:
                return results[0]['symbol']
            
            return None
            
        except Exception as e:
            print(f"ETF search error: {e}")
            return None
    
    def search_index_by_name(self, index_name: str) -> Optional[str]:
        """
        Specialized search for indices by name
        
        Args:
            index_name: Index name
            
        Returns:
            Ticker symbol if found, None otherwise
        """
        try:
            # Common index mappings
            index_mappings = {
                's&p 500': '^GSPC',
                'sp500': '^GSPC',
                'spx': '^GSPC',
                'dow jones': '^DJI',
                'dow': '^DJI',
                'dji': '^DJI',
                'nasdaq': '^IXIC',
                'nasdaq composite': '^IXIC',
                'comp': '^IXIC',
                'russell 2000': '^RUT',
                'rut': '^RUT',
                'ftse 100': '^FTSE',
                'ftse': '^FTSE',
                'dax': '^GDAXI',
                'nikkei': '^N225',
                'n225': '^N225',
                'vix': '^VIX',
                'fear index': '^VIX',
                'volatility': '^VIX',
            }
            
            # Check direct mappings
            query_lower = index_name.lower().strip()
            for key, symbol in index_mappings.items():
                if key in query_lower:
                    return symbol
            
            # Fall back to general search
            results = self.search_ticker(f"{index_name} index", max_results=5)
            
            # Prioritize index results (usually start with ^)
            for result in results:
                if result['symbol'].startswith('^'):
                    return result['symbol']
            
            return None
            
        except Exception as e:
            print(f"Index search error: {e}")
            return None
    
    def _format_type(self, quote_type: str) -> str:
        """Format quote type for display"""
        type_map = {
            'EQUITY': 'Stock',
            'ETF': 'ETF',
            'MUTUALFUND': 'Fund',
            'FUTURE': 'Future',
            'CURRENCY': 'Currency',
            'CRYPTOCURRENCY': 'Crypto',
            'INDEX': 'Index',
            'OPTION': 'Option'
        }
        return type_map.get(quote_type, quote_type)
    
    def format_search_results(self, results: List[Dict]) -> str:
        """
        Format search results for display
        
        Args:
            results: List of search results
            
        Returns:
            Formatted string
        """
        if not results:
            return "No results found."
        
        formatted = "🔍 **Search Results:**\n\n"
        
        for i, result in enumerate(results[:8], 1):  # Show max 8 results
            symbol = result['symbol']
            name = result['name']
            exchange = result['exchange']
            type_display = result['type']
            
            formatted += f"{i}. **{symbol}** - {name}\n"
            formatted += f"   📊 Type: {type_display}"
            if exchange != 'N/A':
                formatted += f" | 📍 Exchange: {exchange}"
            formatted += "\n\n"
        
        if len(results) > 8:
            formatted += f"*Showing 8 of {len(results)} results*\n\n"
        
        formatted += "Click a ticker below to analyze it:"
        return formatted


# Test the module if run directly
if __name__ == "__main__":
    searcher = TickerSearch()
    results = searcher.search_ticker("Apple", 5)
    print(f"Search results: {len(results)} found")
    for result in results:
        print(f"{result['symbol']} - {result['name']} ({result['type']})")