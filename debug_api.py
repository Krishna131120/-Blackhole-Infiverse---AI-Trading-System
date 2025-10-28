#!/usr/bin/env python3
"""
Simple Debug Script for API Endpoints
"""

import requests
import json

def test_endpoint(url, headers=None, data=None, method="GET"):
    """Test a single endpoint"""
    try:
        if method == "GET":
            response = requests.get(url, headers=headers, timeout=10)
        elif method == "POST":
            response = requests.post(url, headers=headers, json=data, timeout=10)
        
        print(f"URL: {url}")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text[:500]}...")
        print("-" * 50)
        
        return response.status_code == 200, response.text
    except Exception as e:
        print(f"Error: {e}")
        return False, str(e)

def main():
    base_url = "http://localhost:8000"
    
    # Get JWT token
    print("1. Testing Authentication...")
    success, response = test_endpoint(
        f"{base_url}/auth/token",
        data={"username": "admin", "password": "admin123"},
        method="POST"
    )
    
    if not success:
        print("Authentication failed!")
        return
    
    try:
        token_data = json.loads(response)
        token = token_data["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        print(f"Token obtained: {token[:20]}...")
    except:
        print("Could not parse token response!")
        return
    
    print("\n2. Testing Health Endpoint...")
    test_endpoint(f"{base_url}/tools/health")
    
    print("\n3. Testing Dashboard Endpoint...")
    test_endpoint(f"{base_url}/data/dashboard", headers=headers)
    
    print("\n4. Testing Live Feed Endpoint...")
    test_endpoint(f"{base_url}/feed/live", headers=headers)
    
    print("\n5. Testing Resilience Health Endpoint...")
    test_endpoint(f"{base_url}/resilience/health", headers=headers)
    
    print("\n6. Testing Predict Endpoint...")
    test_endpoint(
        f"{base_url}/prediction_agent/tools/predict",
        headers=headers,
        data={"symbols": ["AAPL"], "horizon": "daily"},
        method="POST"
    )

if __name__ == "__main__":
    main()
