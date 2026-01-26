import yfinance as yf
import time
import random
import os
import sys
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def apply_yfinance_fix():
    """Apply fixes for yfinance to work on Render"""
    print("🔧 Applying yfinance fixes for Render...", file=sys.stderr)
    
    # Clear cache directory
    cache_dir = "/tmp/yfinance_cache"
    try:
        if os.path.exists(cache_dir):
            import shutil
            shutil.rmtree(cache_dir, ignore_errors=True)
        os.makedirs(cache_dir, exist_ok=True)
    except:
        pass
    
    # Set cache location
    yf.set_tz_cache_location(cache_dir)
    
    # Create a custom session with retry logic
    session = requests.Session()
    
    # Setup retry strategy
    retry_strategy = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"]
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    
    # Custom headers to mimic a real browser
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    })
    
    # Import yfinance module for patching
    import yfinance as yf_module
    
    # Store the session in yfinance module
    yf_module.session = session
    
    # Patch the download function
    original_download = yf_module.download
    
    def patched_download(tickers, session=None, **kwargs):
        max_retries = 7  # Increased retries
        
        # Use our custom session if none provided
        if session is None:
            session = yf_module.session
        
        # Remove problematic parameters
        kwargs_copy = kwargs.copy()
        if 'threads' in kwargs_copy:
            del kwargs_copy['threads']
        
        # Always disable progress
        kwargs_copy['progress'] = False
        
        for attempt in range(max_retries):
            try:
                # Add delay between retries
                if attempt > 0:
                    delay = 3 + attempt * 3 + random.uniform(0, 3)
                    print(f"📡 Retry {attempt} for {tickers} after {delay:.1f}s", file=sys.stderr)
                    time.sleep(delay)
                
                # Try to download
                result = original_download(tickers, session=session, **kwargs_copy)
                
                if result is not None and not result.empty:
                    print(f"✅ Successfully downloaded {tickers} on attempt {attempt + 1}", file=sys.stderr)
                    return result
                else:
                    print(f"⚠️ Empty data for {tickers} on attempt {attempt + 1}", file=sys.stderr)
                    
            except Exception as e:
                print(f"❌ Download attempt {attempt + 1} failed: {str(e)}", file=sys.stderr)
                if attempt == max_retries - 1:
                    print(f"❌ All {max_retries} attempts failed for {tickers}", file=sys.stderr)
                    # Try one more time with a different approach
                    try:
                        print("🔄 Trying alternative download method...", file=sys.stderr)
                        ticker_obj = yf_module.Ticker(tickers)
                        result = ticker_obj.history(period="1y")
                        if not result.empty:
                            return result
                    except:
                        pass
                    raise e
        
        return None
    
    # Apply patches
    yf_module.download = patched_download
    yf.download = patched_download
    
    # Also patch Ticker class to use our session
    original_Ticker_init = yf_module.Ticker.__init__
    
    def patched_Ticker_init(self, ticker, session=None, *args, **kwargs):
        if session is None:
            session = yf_module.session
        original_Ticker_init(self, ticker, session=session, *args, **kwargs)
    
    yf_module.Ticker.__init__ = patched_Ticker_init
    
    print("✅ yfinance fixes applied successfully", file=sys.stderr)
    return True

# Apply fix automatically when imported
apply_yfinance_fix()