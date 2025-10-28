#!/usr/bin/env python3
"""
Test script to verify the fixed feedback validation system
"""

import requests
import json
import time
from datetime import datetime

def test_feedback_validation():
    """Test the feedback validation system"""
    base_url = "http://localhost:8000"
    
    print("="*60)
    print("TESTING FIXED FEEDBACK VALIDATION SYSTEM")
    print("="*60)
    
    # Step 1: Get JWT token
    print("\n1. Getting JWT token...")
    auth_response = requests.post(f"{base_url}/auth/token", 
                                json={"username": "admin", "password": "admin123"})
    if auth_response.status_code != 200:
        print(f"❌ Authentication failed: {auth_response.text}")
        return False
    
    jwt_token = auth_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {jwt_token}", "Content-Type": "application/json"}
    print("✅ JWT token obtained")
    
    # Step 2: Make a prediction
    print("\n2. Making prediction for MSFT...")
    pred_response = requests.post(f"{base_url}/prediction_agent/tools/predict",
                                headers=headers,
                                json={"symbols": ["MSFT"], "horizon": "daily"})
    
    if pred_response.status_code != 200:
        print(f"❌ Prediction failed: {pred_response.text}")
        return False
    
    pred_data = pred_response.json()
    actual_action = pred_data["data"][0]["action"]
    print(f"✅ Prediction made: MSFT = {actual_action}")
    
    # Step 3: Test correct feedback
    print(f"\n3. Testing CORRECT feedback (claiming {actual_action})...")
    correct_feedback = {
        "symbol": "MSFT",
        "predicted_action": actual_action,
        "user_feedback": "correct",
        "horizon": "daily"
    }
    
    feedback_response = requests.post(f"{base_url}/prediction_agent/tools/feedback",
                                   headers=headers,
                                   json=correct_feedback)
    
    if feedback_response.status_code == 200:
        result = feedback_response.json()
        print(f"✅ Correct feedback accepted: {result}")
    else:
        print(f"❌ Correct feedback rejected: {feedback_response.text}")
    
    # Step 4: Test incorrect feedback (wrong action)
    wrong_action = "long" if actual_action == "short" else "short"
    print(f"\n4. Testing INCORRECT feedback (claiming {wrong_action} instead of {actual_action})...")
    incorrect_feedback = {
        "symbol": "MSFT",
        "predicted_action": wrong_action,
        "user_feedback": "correct",
        "horizon": "daily"
    }
    
    feedback_response2 = requests.post(f"{base_url}/prediction_agent/tools/feedback",
                                     headers=headers,
                                     json=incorrect_feedback)
    
    if feedback_response2.status_code == 200:
        result2 = feedback_response2.json()
        print(f"❌ INCORRECT feedback was accepted (this is wrong!): {result2}")
    else:
        print(f"✅ INCORRECT feedback correctly rejected: {feedback_response2.text}")
    
    # Step 5: Test feedback for non-existent prediction
    print(f"\n5. Testing feedback for symbol with no recent prediction...")
    no_pred_feedback = {
        "symbol": "INVALID",
        "predicted_action": "long",
        "user_feedback": "correct",
        "horizon": "daily"
    }
    
    feedback_response3 = requests.post(f"{base_url}/prediction_agent/tools/feedback",
                                     headers=headers,
                                     json=no_pred_feedback)
    
    if feedback_response3.status_code == 200:
        result3 = feedback_response3.json()
        print(f"❌ Feedback for non-existent prediction was accepted: {result3}")
    else:
        print(f"✅ Feedback for non-existent prediction correctly rejected: {feedback_response3.text}")
    
    print("\n" + "="*60)
    print("FEEDBACK VALIDATION TEST COMPLETE")
    print("="*60)
    
    return True

if __name__ == "__main__":
    test_feedback_validation()
