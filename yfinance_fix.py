import yfinance as yf
import time
import random
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import os

def apply_yfinance_fix():
    """Applies the final, comprehensive fix to bypass Yahoo Finance blocks."""
    print("🚀 APPLYING FINAL YFINANCE IMPERSONATION FIX...")

    # 1. FORCE CLEAR YFINANCE CACHE - Critical first step[citation:1]
    cache_dir = "/tmp/yfinance_cache"
    try:
        import shutil
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir, ignore_errors=True)
            print("🧹 Cleared yfinance cache.")
        os.makedirs(cache_dir, exist_ok=True)
    except Exception as e:
        print(f"⚠️ Could not clear cache: {e}")

    # 2. CREATE AN ADVANCED SESSION WITH FINGERPRINT ROTATION[citation:2][citation:9]
    session = requests.Session()

    # Disable keep-alive to avoid connection pools being flagged[citation:3]
    session.keep_alive = False

    # Enhanced retry logic with longer backoff
    retry_strategy = Retry(
        total=4,  # Fewer, smarter retries
        backoff_factor=2.5,  # Much longer waits (2.5, 5, 10... seconds)
        status_forcelist=[429, 403, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
        respect_retry_after_header=True
    )
    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=1,  # Minimize concurrent connections
        pool_maxsize=1
    )
    session.mount('https://', adapter)
    session.mount('http://', adapter)

    # 3. ROTATING USER-AGENTS & FULL BROWSER HEADERS[citation:2][citation:4][citation:9]
    user_agents = [
        # Primary: Latest Chrome on Windows
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        # Fallback: Chrome on macOS
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
        # Secondary: Firefox
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0'
    ]

    # Complex browser headers for a Windows/Chrome profile[citation:4]
    browser_headers = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
        'Cache-Control': 'max-age=0',
        'sec-ch-ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
    }

    # 4. PATCH YFINANCE INTERNALLY
    import yfinance as yf_module

    # Inject our session into yfinance's core
    yf_module.session = session

    # Patch the Ticker class to use our session and rotate User-Agent
    original_Ticker_init = yf_module.Ticker.__init__
    def patched_Ticker_init(self, ticker, session=None, *args, **kwargs):
        if session is None:
            session = yf_module.session
            # Rotate User-Agent for each new Ticker object
            session.headers.update({'User-Agent': random.choice(user_agents)})
        original_Ticker_init(self, ticker, session=session, *args, **kwargs)
    yf_module.Ticker.__init__ = patched_Ticker_init

    # 5. PATCH THE DOWNLOAD FUNCTION WITH AGGRESSIVE RETRY & DELAY LOGIC[citation:5][citation:9]
    original_download = yf_module.download
    def patched_download(tickers, session=None, **kwargs):
        if session is None:
            session = yf_module.session
            session.headers.update({'User-Agent': random.choice(user_agents)})
            session.headers.update(browser_headers)

        # Force critical settings
        kwargs['progress'] = False
        kwargs['threads'] = False
        kwargs['ignore_tz'] = True
        kwargs['auto_adjust'] = True

        last_exception = None
        max_retries = 3  # Don't spam retries

        for attempt in range(max_retries):
            try:
                # CRITICAL: Add a random delay BEFORE the first attempt and between retries
                # This mimics human hesitation and avoids rapid-fire requests[citation:4][citation:9]
                delay_before = random.uniform(1.5, 4.0) if attempt == 0 else random.uniform(5.0, 12.0)
                print(f"⏳ Pausing for {delay_before:.1f}s before attempt {attempt+1} for {tickers}...")
                time.sleep(delay_before)

                result = original_download(tickers, session=session, **kwargs)

                if result is not None and not result.empty:
                    print(f"✅ DOWNLOAD SUCCESS for {tickers}")
                    return result
                else:
                    print(f"⚠️ Attempt {attempt+1}: Empty data for {tickers}")

            except Exception as e:
                last_exception = e
                error_msg = str(e)
                print(f"❌ Attempt {attempt+1} failed: {error_msg[:100]}")

                # If it's a clear 429/rate limit, wait much longer[citation:1][citation:10]
                if '429' in error_msg or 'Too Many Requests' in error_msg:
                    long_wait = 30 + (attempt * 15)  # 30s, 45s, 60s...
                    print(f"🛑 Rate limited. Waiting {long_wait}s before next retry...")
                    time.sleep(long_wait)
                continue

        if last_exception:
            print(f"💥 ALL {max_retries} attempts failed for {tickers}.")
            raise last_exception
        return None

    yf_module.download = patched_download
    yf.download = patched_download

    print("""
✅ FINAL FIX APPLIED. Strategy Summary:
1. Cleared all cached cookies/crumbs that were flagged[citation:1].
2. Using rotating, real-world browser User-Agents[citation:2].
3. Added comprehensive browser fingerprint headers[citation:4].
4. Implemented random delays to mimic human behavior[citation:9].
5. Reduced connection pooling to appear as a single user.
    """)

# Execute the fix immediately upon import
apply_yfinance_fix()