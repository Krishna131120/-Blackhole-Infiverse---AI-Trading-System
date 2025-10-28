#!/usr/bin/env python3
"""
Quick Diagnostic Script
Checks what might be causing setup issues
"""

import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check Python version"""
    print(f"Python version: {sys.version}")
    if sys.version_info < (3, 8):
        print("⚠️  WARNING: Python 3.8+ recommended")
        return False
    else:
        print("✅ Python version OK")
        return True

def check_packages():
    """Check if required packages are installed"""
    print("\nChecking required packages...")
    packages = [
        "pandas", "numpy", "lightgbm", "requests", "fastapi", 
        "uvicorn", "scikit-learn", "yfinance", "ta"
    ]
    
    missing = []
    for package in packages:
        try:
            if package == "ta":
                # Special handling for ta-lib
                import ta
                print(f"✅ ta (technical analysis)")
            else:
                __import__(package)
                print(f"✅ {package}")
        except ImportError:
            if package == "ta":
                print(f"❌ ta-lib - MISSING (install with: pip install ta)")
                missing.append("ta-lib")
            else:
                print(f"❌ {package} - MISSING")
                missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        if "ta-lib" in missing:
            print("For ta-lib, try: pip install ta")
        print("Or run: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All packages installed")
        return True

def check_files():
    """Check if required files exist"""
    print("\nChecking required files...")
    files = [
        "universe.txt",
        "requirements.txt", 
        "core/data_ingest.py",
        "core/enhanced_features.py",
        "fetch_more_data.py",
        "update_enhanced_features.py"
    ]
    
    missing = []
    for file_path in files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
            missing.append(file_path)
    
    if missing:
        print(f"\n⚠️  Missing files: {', '.join(missing)}")
        return False
    else:
        print("\n✅ All files present")
        return True

def test_data_fetch():
    """Test if data fetching works"""
    print("\nTesting data fetch (dry run)...")
    try:
        # Try to import the data ingestion module
        sys.path.append(str(Path(__file__).parent))
        from core.data_ingest import DataIngestion
        print("✅ Data ingestion module imports OK")
        
        # Check if we can create the ingestion object
        ingestion = DataIngestion()
        print("✅ Data ingestion object created OK")
        
        return True
    except Exception as e:
        print(f"❌ Data ingestion test failed: {e}")
        return False

def test_network():
    """Test network connectivity"""
    print("\nTesting network connectivity...")
    try:
        import requests
        response = requests.get("https://finance.yahoo.com", timeout=10)
        if response.status_code == 200:
            print("✅ Network connectivity OK")
            return True
        else:
            print(f"⚠️  Network response: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Network test failed: {e}")
        return False

def main():
    print("🔍 BLACKHOLE INFIVERSE - DIAGNOSTIC CHECK")
    print("="*60)
    
    checks = [
        ("Python Version", check_python_version),
        ("Required Packages", check_packages),
        ("Required Files", check_files),
        ("Data Ingestion", test_data_fetch),
        ("Network Connectivity", test_network)
    ]
    
    results = []
    for name, check_func in checks:
        print(f"\n{'='*20} {name} {'='*20}")
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} check failed: {e}")
            results.append((name, False))
    
    print(f"\n{'='*60}")
    print("DIAGNOSTIC SUMMARY")
    print(f"{'='*60}")
    
    all_ok = True
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name}: {status}")
        if not result:
            all_ok = False
    
    if all_ok:
        print("\n🎉 All checks passed! The system should work.")
        print("\nTry running:")
        print("python setup_step_by_step.py")
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("1. Install packages: pip install -r requirements.txt")
        print("2. Check internet connection")
        print("3. Make sure you're in the project directory")

if __name__ == "__main__":
    main()
