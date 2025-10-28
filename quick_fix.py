#!/usr/bin/env python3
"""
Quick Fix Script
Fixes common setup issues
"""

import subprocess
import sys
import time
from pathlib import Path

def install_missing_packages():
    """Install missing packages"""
    print("🔧 Installing missing packages...")
    
    packages_to_install = [
        "ta",  # Technical analysis library
    ]
    
    for package in packages_to_install:
        print(f"Installing {package}...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)
            print(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
            return False
    
    return True

def test_imports():
    """Test if all imports work now"""
    print("\n🧪 Testing imports...")
    
    try:
        import pandas
        print("✅ pandas")
    except ImportError as e:
        print(f"❌ pandas: {e}")
        return False
    
    try:
        import numpy
        print("✅ numpy")
    except ImportError as e:
        print(f"❌ numpy: {e}")
        return False
    
    try:
        import ta
        print("✅ ta (technical analysis)")
    except ImportError as e:
        print(f"❌ ta: {e}")
        return False
    
    try:
        import lightgbm
        print("✅ lightgbm")
    except ImportError as e:
        print(f"❌ lightgbm: {e}")
        return False
    
    try:
        import yfinance
        print("✅ yfinance")
    except ImportError as e:
        print(f"❌ yfinance: {e}")
        return False
    
    return True

def test_network_with_retry():
    """Test network with retry logic"""
    print("\n🌐 Testing network connectivity...")
    
    import requests
    
    urls_to_test = [
        "https://finance.yahoo.com",
        "https://www.google.com",
        "https://httpbin.org/get"
    ]
    
    for url in urls_to_test:
        try:
            print(f"Testing {url}...")
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                print(f"✅ {url} - OK")
                return True
            elif response.status_code == 429:
                print(f"⚠️  {url} - Rate limited (429), but network works")
                return True
            else:
                print(f"⚠️  {url} - Status {response.status_code}")
        except Exception as e:
            print(f"❌ {url} - Error: {e}")
    
    return False

def create_simple_data_fetch():
    """Create a simple data fetch test"""
    print("\n📊 Creating simple data fetch test...")
    
    test_script = '''
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

try:
    import yfinance as yf
    import pandas as pd
    
    print("Testing data fetch with AAPL...")
    
    # Test with a single symbol
    ticker = yf.Ticker("AAPL")
    data = ticker.history(period="5d")
    
    if not data.empty:
        print(f"✅ Data fetch successful! Got {len(data)} rows")
        print(f"Latest close price: ${data['Close'].iloc[-1]:.2f}")
        return True
    else:
        print("❌ No data received")
        return False
        
except Exception as e:
    print(f"❌ Data fetch failed: {e}")
    return False
'''
    
    with open("test_data_fetch.py", "w") as f:
        f.write(test_script)
    
    print("✅ Test script created: test_data_fetch.py")

def main():
    print("🚀 BLACKHOLE INFIVERSE - QUICK FIX")
    print("="*50)
    
    # Step 1: Install missing packages
    if not install_missing_packages():
        print("❌ Package installation failed")
        return
    
    # Step 2: Test imports
    if not test_imports():
        print("❌ Import test failed")
        return
    
    # Step 3: Test network
    if not test_network_with_retry():
        print("❌ Network test failed")
        print("Please check your internet connection")
        return
    
    # Step 4: Create test script
    create_simple_data_fetch()
    
    print("\n🎉 QUICK FIX COMPLETE!")
    print("="*50)
    print("\nNext steps:")
    print("1. Test data fetch:")
    print("   python test_data_fetch.py")
    print("\n2. Run diagnostic again:")
    print("   python diagnostic_check.py")
    print("\n3. If everything looks good, run setup:")
    print("   python setup_step_by_step.py")

if __name__ == "__main__":
    main()
