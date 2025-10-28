#!/usr/bin/env python3
"""
Test Enhanced Prediction Analysis System
Demonstrates comprehensive analysis with detailed reasoning
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import requests
import json
import time
from datetime import datetime

def test_enhanced_analysis():
    """Test the enhanced prediction analysis system"""
    
    base_url = "http://localhost:8000"
    
    print("=" * 80)
    print("TESTING ENHANCED PREDICTION ANALYSIS SYSTEM")
    print("=" * 80)
    
    # Step 1: Get JWT Token
    print("\n🔐 Getting JWT Token...")
    try:
        auth_response = requests.post(f"{base_url}/auth/token")
        if auth_response.status_code == 200:
            token = auth_response.json()["access_token"]
            headers = {"Authorization": f"Bearer {token}"}
            print("✅ Authentication successful")
        else:
            print(f"❌ Authentication failed: {auth_response.status_code}")
            return
    except Exception as e:
        print(f"❌ Authentication error: {e}")
        return
    
    # Step 2: Test Enhanced Predictions
    print("\n📊 Testing Enhanced Predictions...")
    
    test_symbols = ["AAPL", "MSFT", "TSLA", "BTC-USD", "NVDA"]
    
    for symbol in test_symbols:
        print(f"\n🔍 Analyzing {symbol}...")
        
        try:
            # Test prediction with enhanced analysis
            predict_payload = {
                "symbols": [symbol],
                "horizon": "daily",
                "risk_profile": {
                    "stop_loss_pct": 2.0,
                    "capital_risk_pct": 1.5,
                    "drawdown_limit_pct": 10.0
                }
            }
            
            response = requests.post(
                f"{base_url}/prediction_agent/tools/predict",
                json=predict_payload,
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success") and data.get("data"):
                    prediction = data["data"][0]
                    
                    print(f"\n📈 {symbol} PREDICTION ANALYSIS:")
                    print(f"Action: {prediction['action'].upper()}")
                    print(f"Score: {prediction['score']:.4f}")
                    print(f"Confidence: {prediction['confidence']:.4f}")
                    print(f"Predicted Price: ${prediction['predicted_price']:.2f}")
                    print(f"Current Price: ${prediction['_debug']['current_price']:.2f}")
                    
                    print(f"\n📋 DETAILED REASONING:")
                    print("-" * 60)
                    print(prediction['reason'])
                    print("-" * 60)
                    
                    # Show debug information
                    debug = prediction.get('_debug', {})
                    print(f"\n🔧 DEBUG INFO:")
                    print(f"RL Score Raw: {debug.get('rl_score_raw', 'N/A')}")
                    print(f"RL Score Normalized: {debug.get('rl_score_normalized', 'N/A')}")
                    print(f"Baseline Score: {debug.get('baseline_score', 'N/A')}")
                    print(f"Combined Score: {debug.get('combined_score', 'N/A')}")
                    
                else:
                    print(f"❌ No prediction data for {symbol}")
            else:
                print(f"❌ Prediction failed for {symbol}: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error analyzing {symbol}: {e}")
        
        time.sleep(1)  # Rate limiting
    
    # Step 3: Test Enhanced Analysis Endpoint
    print(f"\n📊 Testing Enhanced Analysis Endpoint...")
    
    try:
        analyze_payload = {
            "symbols": ["AAPL", "MSFT", "TSLA"]
        }
        
        response = requests.post(
            f"{base_url}/prediction_agent/tools/analyze",
            json=analyze_payload,
            headers=headers
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get("success") and data.get("data"):
                analyses = data["data"]
                
                print(f"\n📈 COMPREHENSIVE ANALYSIS RESULTS:")
                print("=" * 80)
                
                for analysis in analyses:
                    symbol = analysis['symbol']
                    print(f"\n🔍 {symbol} ANALYSIS:")
                    print(f"Price: ${analysis['price']:.2f}")
                    print(f"Suggested Action: {analysis['suggested_action'].upper()}")
                    print(f"Score: {analysis['score']:.4f}")
                    print(f"Confidence: {analysis['confidence']:.4f}")
                    
                    print(f"\n📊 TECHNICAL SIGNALS:")
                    signals = analysis['signals']
                    for signal, value in signals.items():
                        if isinstance(value, (int, float)):
                            print(f"  {signal}: {value:.4f}")
                        else:
                            print(f"  {signal}: {value}")
                    
                    print(f"\n📋 DETAILED REASONING:")
                    print("-" * 60)
                    print(analysis['reason'])
                    print("-" * 60)
                    
            else:
                print("❌ No analysis data received")
        else:
            print(f"❌ Analysis failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error in analysis: {e}")
    
    # Step 4: Test Scan All with Enhanced Analysis
    print(f"\n🔍 Testing Scan All with Enhanced Analysis...")
    
    try:
        scan_payload = {
            "limit": 5,
            "min_confidence": 0.3
        }
        
        response = requests.post(
            f"{base_url}/prediction_agent/tools/scan_all",
            json=scan_payload,
            headers=headers
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get("success") and data.get("data"):
                scans = data["data"]
                
                print(f"\n📈 TOP SCAN RESULTS:")
                print("=" * 80)
                
                for i, scan in enumerate(scans, 1):
                    symbol = scan['symbol']
                    print(f"\n{i}. {symbol}")
                    print(f"   Action: {scan['action'].upper()}")
                    print(f"   Score: {scan['score']:.4f}")
                    print(f"   Confidence: {scan['confidence']:.4f}")
                    print(f"   Timestamp: {scan['timestamp']}")
                    
            else:
                print("❌ No scan data received")
        else:
            print(f"❌ Scan failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error in scan: {e}")
    
    print(f"\n✅ Enhanced Analysis Testing Complete!")
    print("=" * 80)

if __name__ == "__main__":
    test_enhanced_analysis()
