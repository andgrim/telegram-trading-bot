"""
yfinance_fix.py - Fix for yfinance issues on Render
"""
import yfinance as yf
import time
import random
import os
import sys

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
    
    # Import yfinance module for patching
    import yfinance as yf_module
    
    # Patch the download function with retry logic
    original_download = yf_module.download
    
    def patched_download(tickers, **kwargs):
        max_retries = 5
        
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
                    delay = 2 + attempt * 2 + random.uniform(0, 2)
                    print(f"📡 Retry {attempt} for {tickers} after {delay:.1f}s", file=sys.stderr)
                    time.sleep(delay)
                
                # Try to download
                result = original_download(tickers, **kwargs_copy)
                
                if result is not None and not result.empty:
                    print(f"✅ Successfully downloaded {tickers} on attempt {attempt + 1}", file=sys.stderr)
                    return result
                else:
                    print(f"⚠️ Empty data for {tickers} on attempt {attempt + 1}", file=sys.stderr)
                    
            except Exception as e:
                print(f"❌ Download attempt {attempt + 1} failed: {str(e)}", file=sys.stderr)
                if attempt == max_retries - 1:
                    print(f"❌ All {max_retries} attempts failed for {tickers}", file=sys.stderr)
                    raise e
        
        return None
    
    # Apply patches
    yf_module.download = patched_download
    yf.download = patched_download
    
    print("✅ yfinance fixes applied successfully", file=sys.stderr)
    return True

# Apply fix automatically when imported
apply_yfinance_fix()