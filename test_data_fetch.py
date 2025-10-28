#!/usr/bin/env python3
"""
Simple Data Fetch Test
Tests data fetching with rate limiting handling
"""

import sys
from pathlib import Path
import time

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def test_simple_fetch():
    """Test simple data fetch with rate limiting"""
    try:
        import yfinance as yf
        import pandas as pd
        
        print("🧪 Testing data fetch with AAPL...")
        
        # Test with a single symbol first
        ticker = yf.Ticker("AAPL")
        
        # Add delay to avoid rate limiting
        print("Waiting 2 seconds to avoid rate limiting...")
        time.sleep(2)
        
        data = ticker.history(period="5d")
        
        if not data.empty:
            print(f"✅ Data fetch successful! Got {len(data)} rows")
            print(f"Latest close price: ${data['Close'].iloc[-1]:.2f}")
            print(f"Data columns: {list(data.columns)}")
            return True
        else:
            print("❌ No data received")
            return False
            
    except Exception as e:
        print(f"❌ Data fetch failed: {e}")
        return False

def test_multiple_symbols():
    """Test fetching multiple symbols with delays"""
    try:
        import yfinance as yf
        
        symbols = ["AAPL", "MSFT", "TSLA"]
        results = {}
        
        print(f"\n🧪 Testing multiple symbols: {symbols}")
        
        for i, symbol in enumerate(symbols):
            print(f"Fetching {symbol}...")
            
            # Add delay between requests
            if i > 0:
                print("Waiting 3 seconds to avoid rate limiting...")
                time.sleep(3)
            
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="5d")
                
                if not data.empty:
                    results[symbol] = {
                        'success': True,
                        'rows': len(data),
                        'latest_price': data['Close'].iloc[-1]
                    }
                    print(f"✅ {symbol}: {len(data)} rows, ${data['Close'].iloc[-1]:.2f}")
                else:
                    results[symbol] = {'success': False, 'error': 'No data'}
                    print(f"❌ {symbol}: No data")
                    
            except Exception as e:
                results[symbol] = {'success': False, 'error': str(e)}
                print(f"❌ {symbol}: {e}")
        
        successful = sum(1 for r in results.values() if r['success'])
        print(f"\n📊 Results: {successful}/{len(symbols)} symbols successful")
        
        return successful > 0
        
    except Exception as e:
        print(f"❌ Multiple symbols test failed: {e}")
        return False

def main():
    print("📊 BLACKHOLE INFIVERSE - DATA FETCH TEST")
    print("="*50)
    
    # Test 1: Single symbol
    print("\n[TEST 1] Single Symbol Fetch")
    single_success = test_simple_fetch()
    
    # Test 2: Multiple symbols
    print("\n[TEST 2] Multiple Symbols Fetch")
    multiple_success = test_multiple_symbols()
    
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    if single_success and multiple_success:
        print("🎉 All tests passed! Data fetching works correctly.")
        print("\nYou can now run:")
        print("python setup_step_by_step.py")
    elif single_success:
        print("⚠️  Single symbol fetch works, but multiple symbols had issues.")
        print("This might be due to rate limiting. The system should still work.")
        print("\nYou can try running:")
        print("python setup_step_by_step.py")
    else:
        print("❌ Data fetching tests failed.")
        print("Please check:")
        print("1. Internet connection")
        print("2. yfinance package: pip install yfinance")
        print("3. Try again later (Yahoo Finance might be down)")

if __name__ == "__main__":
    main()
