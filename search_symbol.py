#!/usr/bin/env python3
"""
Quick Symbol Search Tool
Search for any symbol in your dataset and get prediction
"""

import requests
import json
import sys

def search_symbol(symbol, base_url="http://localhost:8000", jwt_token=None):
    """
    Search for a symbol and get prediction
    """
    
    # If no token provided, try to get one
    if not jwt_token:
        print("Getting JWT token...")
        try:
            auth_response = requests.post(
                f"{base_url}/auth/token",
                json={"username": "admin", "password": "admin123"}
            )
            if auth_response.status_code == 200:
                jwt_token = auth_response.json()["access_token"]
                print("✅ JWT token obtained")
            else:
                print("❌ Failed to get JWT token")
                return
        except Exception as e:
            print(f"❌ Error getting JWT token: {e}")
            return
    
    # Search for the symbol
    print(f"\n🔍 Searching for symbol: {symbol}")
    print("="*50)
    
    try:
        response = requests.post(
            f"{base_url}/prediction_agent/tools/predict",
            headers={
                "Authorization": f"Bearer {jwt_token}",
                "Content-Type": "application/json"
            },
            json={
                "symbol": symbol,
                "horizon": "daily"
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            if data["success"]:
                result = data["data"][0]
                print(f"✅ Symbol found: {result['symbol']}")
                print(f"📊 Predicted Price: ${result['predicted_price']:.2f}")
                print(f"🎯 Action: {result['action'].upper()}")
                print(f"📈 Confidence: {result['confidence']:.2%}")
                print(f"📝 Score: {result['score']:.4f}")
                print(f"💡 Reason: {result['reason']}")
                print(f"⏰ Timestamp: {result['timestamp']}")
                
                # Debug info
                if "_debug" in result:
                    debug = result["_debug"]
                    print(f"\n🔧 Debug Info:")
                    print(f"   Current Price: ${debug.get('current_price', 'N/A')}")
                    print(f"   RL Score: {debug.get('rl_score_normalized', 'N/A')}")
                    print(f"   Combined Score: {debug.get('combined_score', 'N/A')}")
            else:
                print(f"❌ Error: {data.get('error', 'Unknown error')}")
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python search_symbol.py <SYMBOL>")
        print("\nExamples:")
        print("  python search_symbol.py AAPL")
        print("  python search_symbol.py TSLA")
        print("  python search_symbol.py BTC-USD")
        print("  python search_symbol.py GC=F")
        print("  python search_symbol.py HDFCBANK")
        print("\nAvailable symbols: AAPL, TSLA, MSFT, GOOGL, BTC-USD, ETH-USD, GC=F, HDFCBANK, etc.")
        return
    
    symbol = sys.argv[1].upper()
    search_symbol(symbol)

if __name__ == "__main__":
    main()
