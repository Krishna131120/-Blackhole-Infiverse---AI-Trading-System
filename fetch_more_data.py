#!/usr/bin/env python3
"""
Fetch More Data - Direct data fetching for all symbols in universe.txt
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from core.data_ingest import DataIngestion
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_universe(universe_file: str = "./universe.txt") -> list:
    """Load symbols from universe file with proper filtering"""
    try:
        with open(universe_file, 'r') as f:
            symbols = []
            for line in f:
                line = line.strip()
                # Skip empty lines and comment lines
                if line and not line.startswith('#'):
                    symbols.append(line)
        return symbols
    except FileNotFoundError:
        logger.error(f"Universe file not found: {universe_file}")
        return []
    except Exception as e:
        logger.error(f"Error loading universe: {e}")
        return []

def main():
    print("="*60)
    print("FETCH MORE DATA - ALL SYMBOLS FROM UNIVERSE.TXT")
    print("="*60)
    
    # Load symbols from universe file
    print("\n[1] Loading symbol universe...")
    symbols = load_universe()
    if not symbols:
        print("[ERROR] No symbols found in universe.txt")
        return
    
    print(f"[OK] Loaded {len(symbols)} symbols from universe.txt")
    
    # Filter out invalid symbols
    print("\n[2] Filtering valid symbols...")
    valid_symbols = []
    invalid_symbols = []
    
    for symbol in symbols:
        # Skip comment lines and empty lines
        if symbol.startswith('#') or not symbol.strip():
            continue
        
        # Basic validation
        if len(symbol) < 2:
            invalid_symbols.append(symbol)
            continue
            
        valid_symbols.append(symbol)
    
    print(f"[OK] Found {len(valid_symbols)} valid symbols")
    if invalid_symbols:
        print(f"[WARNING] Skipped {len(invalid_symbols)} invalid symbols")
    
    # Initialize data ingestion
    print("\n[3] Initializing data ingestion...")
    data_ingestion = DataIngestion(cache_dir="./data/cache")
    
    # Fetch data for all symbols
    print("\n[4] Fetching market data for all symbols...")
    print("This may take a few minutes for many symbols...")
    
    successful = 0
    failed = 0
    
    for i, symbol in enumerate(valid_symbols, 1):
        print(f"[{i}/{len(valid_symbols)}] Fetching {symbol}...")
        
        try:
            # Fetch data for individual symbol
            data = data_ingestion.fetch_auto(symbol, period="1y", interval="1d")
            
            if data is not None and not data.empty:
                print(f"[OK] {symbol}: {len(data)} rows")
                successful += 1
            else:
                print(f"[SKIP] {symbol}: No data available")
                failed += 1
                
        except Exception as e:
            print(f"[ERROR] {symbol}: {str(e)}")
            failed += 1
        
        # Rate limiting
        import time
        time.sleep(0.5)
    
    print(f"\n[5] Data ingestion complete!")
    print(f"[SUCCESS] {successful} symbols fetched successfully")
    print(f"[FAILED] {failed} symbols failed")
    print(f"[INFO] Data cached in: ./data/cache/")
    
    print("\n" + "="*60)
    print("NEXT STEP: Run 'python update_enhanced_features.py'")
    print("="*60)

if __name__ == "__main__":
    main()
