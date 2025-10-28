#!/usr/bin/env python3
"""
Demo System - End-to-End Flow Test and Load Testing
Tests the complete prediction agent system with sample load
"""

import asyncio
import aiohttp
import json
import time
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from pathlib import Path
import random
import statistics

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DemoSystem:
    """
    Demo System for testing the complete prediction agent
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None
        self.jwt_token: Optional[str] = None
        self.test_results: Dict[str, Any] = {}
        
        # Test symbols
        self.test_symbols = [
            "AAPL", "TSLA", "MSFT", "GOOGL", "META",
            "HDFCBANK", "ICICIBANK", "RELIANCE", "TCS", "INFY",
            "BTC-USD", "ETH-USD", "SOL-USD",
            "GC=F", "CL=F", "SI=F"
        ]
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def authenticate(self) -> bool:
        """Authenticate and get JWT token"""
        try:
            auth_data = {
                "username": "admin",
                "password": "admin123"
            }
            
            async with self.session.post(f"{self.base_url}/auth/token", json=auth_data) as response:
                if response.status == 200:
                    data = await response.json()
                    self.jwt_token = data["access_token"]
                    logger.info("✓ Authentication successful")
                    return True
                else:
                    logger.error(f"Authentication failed: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Check system health"""
        try:
            async with self.session.get(f"{self.base_url}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info("✓ Health check passed")
                    return data
                else:
                    logger.error(f"Health check failed: {response.status}")
                    return {}
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return {}
    
    async def test_predict_endpoint(self, symbol: str) -> Dict[str, Any]:
        """Test predict endpoint for a single symbol"""
        try:
            headers = {"Authorization": f"Bearer {self.jwt_token}"}
            data = {
                "symbols": [symbol],
                "horizon": "daily"
            }
            
            start_time = time.time()
            async with self.session.post(
                f"{self.base_url}/prediction_agent/tools/predict",
                headers=headers,
                json=data
            ) as response:
                end_time = time.time()
                response_time = end_time - start_time
                
                if response.status == 200:
                    result = await response.json()
                    return {
                        "success": True,
                        "response_time": response_time,
                        "data": result,
                        "status_code": response.status
                    }
                else:
                    error_text = await response.text()
                    return {
                        "success": False,
                        "response_time": response_time,
                        "error": error_text,
                        "status_code": response.status
                    }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response_time": 0
            }
    
    async def test_scan_all_endpoint(self) -> Dict[str, Any]:
        """Test scan_all endpoint"""
        try:
            headers = {"Authorization": f"Bearer {self.jwt_token}"}
            data = {
                "horizon": "daily",
                "top_k": 10
            }
            
            start_time = time.time()
            async with self.session.post(
                f"{self.base_url}/prediction_agent/tools/scan_all",
                headers=headers,
                json=data
            ) as response:
                end_time = time.time()
                response_time = end_time - start_time
                
                if response.status == 200:
                    result = await response.json()
                    return {
                        "success": True,
                        "response_time": response_time,
                        "data": result,
                        "status_code": response.status
                    }
                else:
                    error_text = await response.text()
                    return {
                        "success": False,
                        "response_time": response_time,
                        "error": error_text,
                        "status_code": response.status
                    }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response_time": 0
            }
    
    async def test_analyze_endpoint(self, symbols: List[str]) -> Dict[str, Any]:
        """Test analyze endpoint"""
        try:
            headers = {"Authorization": f"Bearer {self.jwt_token}"}
            data = {
                "symbols": symbols,
                "horizon": "daily",
                "detailed": True
            }
            
            start_time = time.time()
            async with self.session.post(
                f"{self.base_url}/prediction_agent/tools/analyze",
                headers=headers,
                json=data
            ) as response:
                end_time = time.time()
                response_time = end_time - start_time
                
                if response.status == 200:
                    result = await response.json()
                    return {
                        "success": True,
                        "response_time": response_time,
                        "data": result,
                        "status_code": response.status
                    }
                else:
                    error_text = await response.text()
                    return {
                        "success": False,
                        "response_time": response_time,
                        "error": error_text,
                        "status_code": response.status
                    }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response_time": 0
            }
    
    async def test_confirm_endpoint(self, request_id: str) -> Dict[str, Any]:
        """Test confirm endpoint"""
        try:
            headers = {"Authorization": f"Bearer {self.jwt_token}"}
            data = {
                "request_id": request_id,
                "confirmation_type": "execute",
                "user_notes": "Demo test confirmation"
            }
            
            start_time = time.time()
            async with self.session.post(
                f"{self.base_url}/prediction_agent/tools/confirm",
                headers=headers,
                json=data
            ) as response:
                end_time = time.time()
                response_time = end_time - start_time
                
                if response.status == 200:
                    result = await response.json()
                    return {
                        "success": True,
                        "response_time": response_time,
                        "data": result,
                        "status_code": response.status
                    }
                else:
                    error_text = await response.text()
                    return {
                        "success": False,
                        "response_time": response_time,
                        "error": error_text,
                        "status_code": response.status
                    }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response_time": 0
            }
    
    async def test_dashboard_endpoint(self) -> Dict[str, Any]:
        """Test dashboard endpoint"""
        try:
            headers = {"Authorization": f"Bearer {self.jwt_token}"}
            
            start_time = time.time()
            async with self.session.get(
                f"{self.base_url}/data/dashboard",
                headers=headers
            ) as response:
                end_time = time.time()
                response_time = end_time - start_time
                
                if response.status == 200:
                    result = await response.json()
                    return {
                        "success": True,
                        "response_time": response_time,
                        "data": result,
                        "status_code": response.status
                    }
                else:
                    error_text = await response.text()
                    return {
                        "success": False,
                        "response_time": response_time,
                        "error": error_text,
                        "status_code": response.status
                    }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response_time": 0
            }
    
    async def test_live_feed_endpoint(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Test live feed endpoint"""
        try:
            headers = {"Authorization": f"Bearer {self.jwt_token}"}
            url = f"{self.base_url}/feed/live"
            if symbol:
                url += f"?symbol={symbol}&limit=50"
            else:
                url += "?limit=50"
            
            start_time = time.time()
            async with self.session.get(url, headers=headers) as response:
                end_time = time.time()
                response_time = end_time - start_time
                
                if response.status == 200:
                    result = await response.json()
                    return {
                        "success": True,
                        "response_time": response_time,
                        "data": result,
                        "status_code": response.status
                    }
                else:
                    error_text = await response.text()
                    return {
                        "success": False,
                        "response_time": response_time,
                        "error": error_text,
                        "status_code": response.status
                    }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response_time": 0
            }
    
    async def test_resilience_system(self) -> Dict[str, Any]:
        """Test resilience system endpoints"""
        try:
            headers = {"Authorization": f"Bearer {self.jwt_token}"}
            
            # Test health endpoint
            async with self.session.get(
                f"{self.base_url}/resilience/health",
                headers=headers
            ) as response:
                if response.status == 200:
                    health_data = await response.json()
                    logger.info(f"✓ Resilience health: {health_data.get('status', 'unknown')}")
                else:
                    logger.warning(f"Resilience health check failed: {response.status}")
            
            # Test alerts endpoint
            async with self.session.get(
                f"{self.base_url}/resilience/alerts",
                headers=headers
            ) as response:
                if response.status == 200:
                    alerts_data = await response.json()
                    alert_count = alerts_data.get('count', 0)
                    logger.info(f"✓ Resilience alerts: {alert_count} active")
                else:
                    logger.warning(f"Resilience alerts check failed: {response.status}")
            
            return {"success": True}
        except Exception as e:
            logger.error(f"Resilience system test error: {e}")
            return {"success": False, "error": str(e)}
    
    async def run_load_test(self, num_requests: int = 50) -> Dict[str, Any]:
        """Run load test with multiple concurrent requests"""
        logger.info(f"Starting load test with {num_requests} requests...")
        
        # Create tasks for concurrent execution
        tasks = []
        for i in range(num_requests):
            symbol = random.choice(self.test_symbols)
            task = self.test_predict_endpoint(symbol)
            tasks.append(task)
        
        # Execute all tasks concurrently
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        total_time = end_time - start_time
        successful_requests = sum(1 for r in results if isinstance(r, dict) and r.get('success', False))
        failed_requests = num_requests - successful_requests
        
        # Calculate response times
        response_times = [r.get('response_time', 0) for r in results if isinstance(r, dict) and r.get('response_time', 0) > 0]
        
        return {
            "total_requests": num_requests,
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "success_rate": successful_requests / num_requests * 100,
            "total_time": total_time,
            "requests_per_second": num_requests / total_time,
            "avg_response_time": statistics.mean(response_times) if response_times else 0,
            "min_response_time": min(response_times) if response_times else 0,
            "max_response_time": max(response_times) if response_times else 0,
            "results": results
        }
    
    async def run_end_to_end_test(self) -> Dict[str, Any]:
        """Run complete end-to-end test"""
        logger.info("="*60)
        logger.info("STARTING END-TO-END DEMO TEST")
        logger.info("="*60)
        
        test_results = {
            "start_time": datetime.now(timezone.utc).isoformat(),
            "tests": {},
            "overall_success": True
        }
        
        # 1. Authentication Test
        logger.info("\n[1/8] Testing Authentication...")
        auth_success = await self.authenticate()
        test_results["tests"]["authentication"] = {
            "success": auth_success,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        if not auth_success:
            logger.error("Authentication failed. Stopping tests.")
            test_results["overall_success"] = False
            return test_results
        
        # 2. Health Check Test
        logger.info("\n[2/8] Testing Health Check...")
        health_data = await self.health_check()
        test_results["tests"]["health_check"] = {
            "success": bool(health_data),
            "data": health_data,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # 3. Predict Endpoint Test
        logger.info("\n[3/8] Testing Predict Endpoint...")
        predict_results = []
        for symbol in self.test_symbols[:5]:  # Test first 5 symbols
            result = await self.test_predict_endpoint(symbol)
            predict_results.append(result)
            if result.get('success'):
                logger.info(f"✓ Predict test passed for {symbol}")
            else:
                logger.error(f"✗ Predict test failed for {symbol}: {result.get('error', 'Unknown error')}")
        
        test_results["tests"]["predict_endpoint"] = {
            "success": all(r.get('success', False) for r in predict_results),
            "results": predict_results,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # 4. Scan All Endpoint Test
        logger.info("\n[4/8] Testing Scan All Endpoint...")
        scan_result = await self.test_scan_all_endpoint()
        test_results["tests"]["scan_all_endpoint"] = {
            "success": scan_result.get('success', False),
            "result": scan_result,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        if scan_result.get('success'):
            logger.info("✓ Scan all test passed")
        else:
            logger.error(f"✗ Scan all test failed: {scan_result.get('error', 'Unknown error')}")
        
        # 5. Analyze Endpoint Test
        logger.info("\n[5/8] Testing Analyze Endpoint...")
        analyze_result = await self.test_analyze_endpoint(self.test_symbols[:3])
        test_results["tests"]["analyze_endpoint"] = {
            "success": analyze_result.get('success', False),
            "result": analyze_result,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        if analyze_result.get('success'):
            logger.info("✓ Analyze test passed")
        else:
            logger.error(f"✗ Analyze test failed: {analyze_result.get('error', 'Unknown error')}")
        
        # 6. Confirm Endpoint Test
        logger.info("\n[6/8] Testing Confirm Endpoint...")
        confirm_result = await self.test_confirm_endpoint("demo_request_123")
        test_results["tests"]["confirm_endpoint"] = {
            "success": confirm_result.get('success', False),
            "result": confirm_result,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        if confirm_result.get('success'):
            logger.info("✓ Confirm test passed")
        else:
            logger.error(f"✗ Confirm test failed: {confirm_result.get('error', 'Unknown error')}")
        
        # 7. Frontend Endpoints Test
        logger.info("\n[7/8] Testing Frontend Endpoints...")
        dashboard_result = await self.test_dashboard_endpoint()
        live_feed_result = await self.test_live_feed_endpoint()
        
        test_results["tests"]["frontend_endpoints"] = {
            "dashboard_success": dashboard_result.get('success', False),
            "live_feed_success": live_feed_result.get('success', False),
            "dashboard_result": dashboard_result,
            "live_feed_result": live_feed_result,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        if dashboard_result.get('success') and live_feed_result.get('success'):
            logger.info("✓ Frontend endpoints test passed")
        else:
            logger.error("✗ Frontend endpoints test failed")
        
        # 8. Resilience System Test
        logger.info("\n[8/8] Testing Resilience System...")
        resilience_result = await self.test_resilience_system()
        test_results["tests"]["resilience_system"] = {
            "success": resilience_result.get('success', False),
            "result": resilience_result,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        if resilience_result.get('success'):
            logger.info("✓ Resilience system test passed")
        else:
            logger.error(f"✗ Resilience system test failed: {resilience_result.get('error', 'Unknown error')}")
        
        # 9. Load Test
        logger.info("\n[9/9] Running Load Test...")
        load_test_result = await self.run_load_test(num_requests=20)
        test_results["tests"]["load_test"] = {
            "success": load_test_result.get('success_rate', 0) > 80,  # 80% success rate threshold
            "result": load_test_result,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        if load_test_result.get('success_rate', 0) > 80:
            logger.info(f"✓ Load test passed: {load_test_result.get('success_rate', 0):.1f}% success rate")
        else:
            logger.error(f"✗ Load test failed: {load_test_result.get('success_rate', 0):.1f}% success rate")
        
        # Calculate overall success
        test_successes = [
            test_results["tests"]["authentication"]["success"],
            test_results["tests"]["health_check"]["success"],
            test_results["tests"]["predict_endpoint"]["success"],
            test_results["tests"]["scan_all_endpoint"]["success"],
            test_results["tests"]["analyze_endpoint"]["success"],
            test_results["tests"]["confirm_endpoint"]["success"],
            test_results["tests"]["frontend_endpoints"]["dashboard_success"],
            test_results["tests"]["frontend_endpoints"]["live_feed_success"],
            test_results["tests"]["resilience_system"]["success"],
            test_results["tests"]["load_test"]["success"]
        ]
        
        test_results["overall_success"] = all(test_successes)
        test_results["success_rate"] = sum(test_successes) / len(test_successes) * 100
        test_results["end_time"] = datetime.now(timezone.utc).isoformat()
        
        # Final report
        logger.info("\n" + "="*60)
        logger.info("DEMO TEST COMPLETED")
        logger.info("="*60)
        logger.info(f"Overall Success: {'✓ PASSED' if test_results['overall_success'] else '✗ FAILED'}")
        logger.info(f"Success Rate: {test_results['success_rate']:.1f}%")
        logger.info(f"Tests Passed: {sum(test_successes)}/{len(test_successes)}")
        
        if load_test_result.get('success_rate'):
            logger.info(f"Load Test Success Rate: {load_test_result['success_rate']:.1f}%")
            logger.info(f"Average Response Time: {load_test_result.get('avg_response_time', 0):.3f}s")
            logger.info(f"Requests per Second: {load_test_result.get('requests_per_second', 0):.2f}")
        
        logger.info("="*60)
        
        return test_results
    
    def save_test_results(self, results: Dict[str, Any], filename: str = None):
        """Save test results to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"demo_test_results_{timestamp}.json"
        
        results_file = Path("logs") / filename
        results_file.parent.mkdir(exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Test results saved to: {results_file}")


async def main():
    """Main demo function"""
    print("\n" + "="*60)
    print("BLACKHOLE INFIVERSE - DEMO SYSTEM")
    print("="*60)
    print("Testing complete prediction agent system...")
    print("="*60)
    
    async with DemoSystem() as demo:
        # Run end-to-end test
        results = await demo.run_end_to_end_test()
        
        # Save results
        demo.save_test_results(results)
        
        # Return results for further processing
        return results


if __name__ == "__main__":
    asyncio.run(main())
