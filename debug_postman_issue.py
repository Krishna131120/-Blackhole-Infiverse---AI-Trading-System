#!/usr/bin/env python3
"""
Debug script to help troubleshoot Postman API issues
"""

import requests
import json
import sys
from datetime import datetime

def test_api_connection():
    """Test basic API connectivity"""
    print("="*60)
    print("BLACKHOLE INFIVERSE API DEBUG")
    print("="*60)
    print(f"Testing at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    base_url = "http://localhost:8000"
    
    # Test 1: Health Check
    print("\n1. Testing Health Check...")
    try:
        response = requests.get(f"{base_url}/tools/health", timeout=10)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✓ API is running - Status: {data.get('status')}")
            print(f"   ✓ Models loaded: {data.get('models_loaded')}")
            print(f"   ✓ Feature store: {data.get('feature_store_info', {}).get('symbols', 0)} symbols")
        else:
            print(f"   ✗ Health check failed: {response.text}")
            return False
    except Exception as e:
        print(f"   ✗ Cannot connect to API: {e}")
        print("   Make sure the API server is running with: python api/server.py")
        return False
    
    # Test 2: Authentication
    print("\n2. Testing Authentication...")
    try:
        auth_response = requests.post(f"{base_url}/auth/token", 
                                    json={"username": "admin", "password": "admin123"}, 
                                    timeout=10)
        if auth_response.status_code == 200:
            token_data = auth_response.json()
            jwt_token = token_data.get("access_token")
            print(f"   ✓ JWT token obtained: {jwt_token[:20]}...")
        else:
            print(f"   ✗ Authentication failed: {auth_response.text}")
            return False
    except Exception as e:
        print(f"   ✗ Authentication error: {e}")
        return False
    
    # Test 3: Prediction Endpoint
    print("\n3. Testing Prediction Endpoint...")
    headers = {
        "Authorization": f"Bearer {jwt_token}",
        "Content-Type": "application/json"
    }
    
    # Test with minimal request
    test_payload = {
        "symbols": ["AAPL"],
        "horizon": "daily"
    }
    
    try:
        pred_response = requests.post(f"{base_url}/prediction_agent/tools/predict",
                                    headers=headers, 
                                    json=test_payload, 
                                    timeout=30)
        
        print(f"   Status: {pred_response.status_code}")
        print(f"   Response headers: {dict(pred_response.headers)}")
        
        if pred_response.status_code == 200:
            result = pred_response.json()
            print(f"   ✓ Prediction successful!")
            print(f"   ✓ Response keys: {list(result.keys())}")
            if result.get("success"):
                print(f"   ✓ Data returned: {len(result.get('data', []))} predictions")
            else:
                print(f"   ⚠ Response indicates failure: {result}")
        elif pred_response.status_code == 503:
            print(f"   ✗ Service not initialized: {pred_response.text}")
            print("   This means models are not loaded. Check the server logs.")
        elif pred_response.status_code == 401:
            print(f"   ✗ Authentication failed: {pred_response.text}")
            print("   JWT token might be invalid or expired.")
        elif pred_response.status_code == 422:
            print(f"   ✗ Validation error: {pred_response.text}")
            print("   Request format is incorrect.")
        elif pred_response.status_code == 429:
            print(f"   ✗ Rate limit exceeded: {pred_response.text}")
        else:
            print(f"   ✗ Unexpected error: {pred_response.text}")
            
    except Exception as e:
        print(f"   ✗ Prediction request failed: {e}")
    
    # Test 4: Rate Limit Status
    print("\n4. Testing Rate Limit Status...")
    try:
        rate_response = requests.get(f"{base_url}/tools/rate-limit-status",
                                   headers=headers, timeout=10)
        if rate_response.status_code == 200:
            rate_data = rate_response.json()
            print(f"   ✓ Rate limit status: {rate_data}")
        else:
            print(f"   ⚠ Rate limit check failed: {rate_response.text}")
    except Exception as e:
        print(f"   ⚠ Rate limit check error: {e}")
    
    print("\n" + "="*60)
    print("DEBUG COMPLETE")
    print("="*60)
    
    return True

def show_postman_tips():
    """Show Postman configuration tips"""
    print("\n" + "="*60)
    print("POSTMAN CONFIGURATION TIPS")
    print("="*60)
    
    print("\n1. Environment Variables:")
    print("   - Set 'base_url' to: http://localhost:8000")
    print("   - Set 'jwt_token' to: (will be auto-populated)")
    
    print("\n2. Request Headers:")
    print("   - Authorization: Bearer {{jwt_token}}")
    print("   - Content-Type: application/json")
    
    print("\n3. Request Body (for /prediction_agent/tools/predict):")
    print("""   {
     "symbols": ["AAPL"],
     "horizon": "daily"
   }""")
    
    print("\n4. Valid horizon values:")
    print("   - intraday")
    print("   - daily")
    print("   - weekly") 
    print("   - monthly")
    
    print("\n5. Common Issues:")
    print("   - Make sure API server is running")
    print("   - Check JWT token is not expired")
    print("   - Verify request body is valid JSON")
    print("   - Check rate limits")
    print("   - Ensure models are loaded (check /tools/health)")

if __name__ == "__main__":
    success = test_api_connection()
    show_postman_tips()
    
    if not success:
        print("\n❌ API connection failed. Please check:")
        print("   1. Is the API server running? (python api/server.py)")
        print("   2. Is the server accessible at http://localhost:8000?")
        print("   3. Are there any error messages in the server logs?")
        sys.exit(1)
    else:
        print("\n✅ API is working correctly!")
        print("   If Postman still fails, check the configuration tips above.")
