"""
yfinance_fix.py - Fixes for yfinance to work on cloud platforms like Render.
"""
import yfinance as yf
import time
import random
import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def create_resilient_session():
    """Create a requests session that mimics a real web browser and handles retries."""
    session = requests.Session()
    
    # Retry strategy for temporary failures (common on cloud platforms)
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],  # 429 is "Too Many Requests"[citation:1]
        allowed_methods=["GET", "POST"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    
    # Headers to mimic a real Chrome browser on Windows
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
        'Cache-Control': 'max-age=0',
    })
    
    return session

def apply_yfinance_fix():
    """Apply fixes to yfinance for better compatibility with Render."""
    print("🔧 Applying yfinance fixes for Render...")
    
    # Clear any old cache to prevent stale data issues
    cache_dir = "/tmp/yfinance_cache"
    try:
        import shutil
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir, ignore_errors=True)
        os.makedirs(cache_dir, exist_ok=True)
        yf.set_tz_cache_location(cache_dir)
    except:
        pass
    
    # Get the yfinance module to patch its internal functions
    import yfinance as yf_module
    
    # Create a global, resilient session for all yfinance requests
    resilient_session = create_resilient_session()
    yf_module.session = resilient_session
    
    # Patch the Ticker.__init__ to use our session by default
    original_Ticker_init = yf_module.Ticker.__init__
    
    def patched_Ticker_init(self, ticker, session=None, *args, **kwargs):
        if session is None:
            session = resilient_session
        original_Ticker_init(self, ticker, session=session, *args, **kwargs)
    
    yf_module.Ticker.__init__ = patched_Ticker_init
    
    # Patch the download function
    original_download = yf_module.download
    
    def patched_download(tickers, session=None, **kwargs):
        # Use our resilient session if none provided
        if session is None:
            session = resilient_session
        
        # Force certain parameters for better compatibility
        kwargs['progress'] = False
        kwargs['threads'] = False
        if 'ignore_tz' not in kwargs:
            kwargs['ignore_tz'] = True
        
        max_retries = 3
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                # Add increasing delay between retries
                if attempt > 0:
                    delay = 2 + attempt * 2 + random.uniform(0, 1)
                    print(f"📡 Retry {attempt} for {tickers} after {delay:.1f}s")
                    time.sleep(delay)
                
                result = original_download(tickers, session=session, **kwargs)
                
                if result is not None and not result.empty:
                    print(f"✅ Successfully downloaded {tickers}")
                    return result
                else:
                    print(f"⚠️ Empty data for {tickers} on attempt {attempt+1}")
                    
            except Exception as e:
                last_exception = e
                print(f"❌ Download attempt {attempt+1} failed: {str(e)[:100]}")
                continue
        
        # If all retries failed, raise the last exception
        if last_exception:
            raise last_exception
        
        return None
    
    yf_module.download = patched_download
    yf.download = patched_download
    
    print("✅ yfinance fixes applied successfully")
    return True

# Apply the fix when this module is imported
apply_yfinance_fix()