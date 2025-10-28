#!/usr/bin/env python3
"""
Comprehensive API Test Script
Tests all endpoints and verifies the system is working correctly
"""

import requests
import json
import time
from datetime import datetime

class APITester:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.jwt_token = None
        self.test_results = []
        
    def log_test(self, test_name, success, details=""):
        """Log test results"""
        result = {
            "test": test_name,
            "success": success,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
        self.test_results.append(result)
        status = "PASS" if success else "FAIL"
        print(f"{status} {test_name}: {details}")
        
    def test_health_endpoint(self):
        """Test the health endpoint"""
        try:
            response = requests.get(f"{self.base_url}/tools/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                self.log_test("Health Check", True, f"Status: {data.get('status')}")
                return True
            else:
                self.log_test("Health Check", False, f"Status code: {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Health Check", False, f"Error: {str(e)}")
            return False
    
    def test_auth_endpoint(self):
        """Test authentication"""
        try:
            response = requests.post(f"{self.base_url}/auth/token", 
                                   json={"username": "admin", "password": "admin123"}, 
                                   timeout=10)
            if response.status_code == 200:
                data = response.json()
                self.jwt_token = data.get("access_token")
                self.log_test("Authentication", True, "JWT token obtained")
                return True
            else:
                self.log_test("Authentication", False, f"Status code: {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Authentication", False, f"Error: {str(e)}")
            return False
    
    def test_predict_endpoint(self):
        """Test prediction endpoint"""
        if not self.jwt_token:
            self.log_test("Predict Endpoint", False, "No JWT token available")
            return False
            
        try:
            headers = {"Authorization": f"Bearer {self.jwt_token}"}
            data = {"symbols": ["AAPL"], "horizon": "daily"}
            
            response = requests.post(f"{self.base_url}/prediction_agent/tools/predict",
                                   headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success") and result.get("data"):
                    prediction = result["data"][0]
                    self.log_test("Predict Endpoint", True, 
                                f"Predicted {prediction.get('symbol')}: {prediction.get('action')} "
                                f"(confidence: {prediction.get('confidence', 0):.3f})")
                    return True
                else:
                    self.log_test("Predict Endpoint", False, "No prediction data returned")
                    return False
            else:
                self.log_test("Predict Endpoint", False, f"Status code: {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Predict Endpoint", False, f"Error: {str(e)}")
            return False
    
    def test_scan_all_endpoint(self):
        """Test scan all endpoint"""
        if not self.jwt_token:
            self.log_test("Scan All Endpoint", False, "No JWT token available")
            return False
            
        try:
            headers = {"Authorization": f"Bearer {self.jwt_token}"}
            data = {"horizon": "daily", "top_k": 5}
            
            response = requests.post(f"{self.base_url}/prediction_agent/tools/scan_all",
                                   headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success") and result.get("data"):
                    count = len(result["data"])
                    self.log_test("Scan All Endpoint", True, f"Scanned {count} symbols")
                    return True
                else:
                    self.log_test("Scan All Endpoint", False, "No scan data returned")
                    return False
            else:
                self.log_test("Scan All Endpoint", False, f"Status code: {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Scan All Endpoint", False, f"Error: {str(e)}")
            return False
    
    def test_dashboard_endpoint(self):
        """Test dashboard endpoint"""
        if not self.jwt_token:
            self.log_test("Dashboard Endpoint", False, "No JWT token available")
            return False
            
        try:
            headers = {"Authorization": f"Bearer {self.jwt_token}"}
            response = requests.get(f"{self.base_url}/data/dashboard", headers=headers, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                stats = result.get("stats", {})
                self.log_test("Dashboard Endpoint", True, 
                            f"Total requests: {stats.get('total_requests', 0)}")
                return True
            else:
                self.log_test("Dashboard Endpoint", False, f"Status code: {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Dashboard Endpoint", False, f"Error: {str(e)}")
            return False
    
    def test_live_feed_endpoint(self):
        """Test live feed endpoint"""
        if not self.jwt_token:
            self.log_test("Live Feed Endpoint", False, "No JWT token available")
            return False
            
        try:
            headers = {"Authorization": f"Bearer {self.jwt_token}"}
            response = requests.get(f"{self.base_url}/feed/live?limit=10", headers=headers, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                symbols = result.get("symbols", [])
                self.log_test("Live Feed Endpoint", True, f"Feed contains {len(symbols)} symbols")
                return True
            else:
                self.log_test("Live Feed Endpoint", False, f"Status code: {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Live Feed Endpoint", False, f"Error: {str(e)}")
            return False
    
    def test_resilience_endpoint(self):
        """Test resilience health endpoint"""
        if not self.jwt_token:
            self.log_test("Resilience Endpoint", False, "No JWT token available")
            return False
            
        try:
            headers = {"Authorization": f"Bearer {self.jwt_token}"}
            response = requests.get(f"{self.base_url}/resilience/health", headers=headers, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                status = result.get("status", "unknown")
                self.log_test("Resilience Endpoint", True, f"Status: {status}")
                return True
            else:
                self.log_test("Resilience Endpoint", False, f"Status code: {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Resilience Endpoint", False, f"Error: {str(e)}")
            return False
    
    def run_all_tests(self):
        """Run all API tests"""
        print("="*80)
        print("BLACKHOLE INFIVERSE API TESTING")
        print("="*80)
        print(f"Testing API at: {self.base_url}")
        print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        # Run tests in order
        tests = [
            self.test_health_endpoint,
            self.test_auth_endpoint,
            self.test_predict_endpoint,
            self.test_scan_all_endpoint,
            self.test_dashboard_endpoint,
            self.test_live_feed_endpoint,
            self.test_resilience_endpoint
        ]
        
        passed = 0
        total = len(tests)
        
        for test in tests:
            try:
                if test():
                    passed += 1
            except Exception as e:
                print(f"FAIL {test.__name__}: Unexpected error: {str(e)}")
            time.sleep(1)  # Brief pause between tests
        
        # Summary
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        print(f"Total Tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {total - passed}")
        print(f"Success Rate: {(passed/total)*100:.1f}%")
        
        if passed == total:
            print("\nALL TESTS PASSED! System is ready for integration.")
        else:
            print(f"\n{total - passed} tests failed. Check the errors above.")
        
        print("="*80)
        
        return passed == total

def main():
    """Main function to run API tests"""
    tester = APITester()
    success = tester.run_all_tests()
    
    # Save test results
    with open("test_results.json", "w") as f:
        json.dump(tester.test_results, f, indent=2)
    
    print(f"\nTest results saved to: test_results.json")
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
