#!/usr/bin/env python3
"""
Integration Test Script for Krishna→Karan Handshake Testing
Tests the complete prediction agent pipeline with MCP endpoints
"""

import asyncio
import httpx
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import sys
import os

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from core.feedback_loop import get_feedback_loop, TradeOutcome

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntegrationTester:
    """Comprehensive integration tester for Krishna Prediction Agent"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)
        self.token = None
        self.test_results = []
        self.feedback_loop = get_feedback_loop()
    
    async def authenticate(self) -> bool:
        """Authenticate with the API"""
        try:
            response = await self.client.post(
                f"{self.base_url}/auth/token",
                json={"username": "demo", "password": "demo"}
            )
            response.raise_for_status()
            data = response.json()
            self.token = data["access_token"]
            logger.info("✅ Authentication successful")
            return True
        except Exception as e:
            logger.error(f"❌ Authentication failed: {e}")
            return False
    
    def _headers(self) -> Dict[str, str]:
        """Get request headers with auth token"""
        return {"Authorization": f"Bearer {self.token}"}
    
    async def test_health_check(self) -> Dict[str, Any]:
        """Test health check endpoint"""
        logger.info("🔍 Testing health check...")
        try:
            response = await self.client.get(f"{self.base_url}/tools/health")
            response.raise_for_status()
            data = response.json()
            
            result = {
                "test": "health_check",
                "status": "passed",
                "response_time": response.elapsed.total_seconds(),
                "data": data
            }
            logger.info("✅ Health check passed")
            return result
            
        except Exception as e:
            result = {
                "test": "health_check",
                "status": "failed",
                "error": str(e)
            }
            logger.error(f"❌ Health check failed: {e}")
            return result
    
    async def test_predict_endpoint(self, symbols: List[str] = None) -> Dict[str, Any]:
        """Test predict endpoint"""
        logger.info("🔍 Testing predict endpoint...")
        try:
            symbols = symbols or ["AAPL", "MSFT", "BTC-USD"]
            payload = {
                "symbols": symbols,
                "horizon": "daily",
                "risk_profile": {
                    "stop_loss_pct": 2.0,
                    "capital_risk_pct": 1.5,
                    "drawdown_limit_pct": 10.0
                }
            }
            
            start_time = time.time()
            response = await self.client.post(
                f"{self.base_url}/prediction_agent/tools/predict",
                json=payload,
                headers=self._headers()
            )
            response_time = time.time() - start_time
            response.raise_for_status()
            data = response.json()
            
            result = {
                "test": "predict",
                "status": "passed",
                "response_time": response_time,
                "predictions_count": len(data.get('data', [])),
                "data": data
            }
            logger.info(f"✅ Predict endpoint passed - {len(data.get('data', []))} predictions")
            return result
            
        except Exception as e:
            result = {
                "test": "predict",
                "status": "failed",
                "error": str(e)
            }
            logger.error(f"❌ Predict endpoint failed: {e}")
            return result
    
    async def test_scan_all_endpoint(self) -> Dict[str, Any]:
        """Test scan_all endpoint"""
        logger.info("🔍 Testing scan_all endpoint...")
        try:
            payload = {
                "top_k": 10,
                "horizon": "daily",
                "min_score": 0.0
            }
            
            start_time = time.time()
            response = await self.client.post(
                f"{self.base_url}/prediction_agent/tools/scan_all",
                json=payload,
                headers=self._headers()
            )
            response_time = time.time() - start_time
            response.raise_for_status()
            data = response.json()
            
            result = {
                "test": "scan_all",
                "status": "passed",
                "response_time": response_time,
                "scanned_count": len(data.get('data', [])),
                "data": data
            }
            logger.info(f"✅ Scan_all endpoint passed - {len(data.get('data', []))} results")
            return result
            
        except Exception as e:
            result = {
                "test": "scan_all",
                "status": "failed",
                "error": str(e)
            }
            logger.error(f"❌ Scan_all endpoint failed: {e}")
            return result
    
    async def test_analyze_endpoint(self, symbols: List[str] = None) -> Dict[str, Any]:
        """Test analyze endpoint"""
        logger.info("🔍 Testing analyze endpoint...")
        try:
            symbols = symbols or ["AAPL", "MSFT"]
            payload = {
                "symbols": symbols,
                "horizon": "daily",
                "detailed": True
            }
            
            start_time = time.time()
            response = await self.client.post(
                f"{self.base_url}/prediction_agent/tools/analyze",
                json=payload,
                headers=self._headers()
            )
            response_time = time.time() - start_time
            response.raise_for_status()
            data = response.json()
            
            result = {
                "test": "analyze",
                "status": "passed",
                "response_time": response_time,
                "analysis_count": len(data.get('data', [])),
                "data": data
            }
            logger.info(f"✅ Analyze endpoint passed - {len(data.get('data', []))} analyses")
            return result
            
        except Exception as e:
            result = {
                "test": "analyze",
                "status": "failed",
                "error": str(e)
            }
            logger.error(f"❌ Analyze endpoint failed: {e}")
            return result
    
    async def test_feedback_endpoint(self) -> Dict[str, Any]:
        """Test feedback endpoint"""
        logger.info("🔍 Testing feedback endpoint...")
        try:
            payload = {
                "symbol": "AAPL",
                "predicted_action": "long",
                "user_feedback": "correct",
                "horizon": "daily"
            }
            
            start_time = time.time()
            response = await self.client.post(
                f"{self.base_url}/prediction_agent/tools/feedback",
                json=payload,
                headers=self._headers()
            )
            response_time = time.time() - start_time
            response.raise_for_status()
            data = response.json()
            
            result = {
                "test": "feedback",
                "status": "passed",
                "response_time": response_time,
                "data": data
            }
            logger.info("✅ Feedback endpoint passed")
            return result
            
        except Exception as e:
            result = {
                "test": "feedback",
                "status": "failed",
                "error": str(e)
            }
            logger.error(f"❌ Feedback endpoint failed: {e}")
            return result
    
    async def test_train_rl_endpoint(self) -> Dict[str, Any]:
        """Test train_rl endpoint"""
        logger.info("🔍 Testing train_rl endpoint...")
        try:
            payload = {
                "agent_type": "linucb",
                "rounds": 10,
                "top_k": 5,
                "horizon": 1
            }
            
            start_time = time.time()
            response = await self.client.post(
                f"{self.base_url}/prediction_agent/tools/train_rl",
                json=payload,
                headers=self._headers()
            )
            response_time = time.time() - start_time
            response.raise_for_status()
            data = response.json()
            
            result = {
                "test": "train_rl",
                "status": "passed",
                "response_time": response_time,
                "data": data
            }
            logger.info("✅ Train_rl endpoint passed")
            return result
            
        except Exception as e:
            result = {
                "test": "train_rl",
                "status": "failed",
                "error": str(e)
            }
            logger.error(f"❌ Train_rl endpoint failed: {e}")
            return result
    
    async def test_fetch_data_endpoint(self) -> Dict[str, Any]:
        """Test fetch_data endpoint"""
        logger.info("🔍 Testing fetch_data endpoint...")
        try:
            payload = {
                "symbols": ["AAPL", "MSFT", "BTC-USD"],
                "period": "1mo",
                "interval": "1d"
            }
            
            start_time = time.time()
            response = await self.client.post(
                f"{self.base_url}/prediction_agent/tools/fetch_data",
                json=payload,
                headers=self._headers()
            )
            response_time = time.time() - start_time
            response.raise_for_status()
            data = response.json()
            
            result = {
                "test": "fetch_data",
                "status": "passed",
                "response_time": response_time,
                "data": data
            }
            logger.info("✅ Fetch_data endpoint passed")
            return result
            
        except Exception as e:
            result = {
                "test": "fetch_data",
                "status": "failed",
                "error": str(e)
            }
            logger.error(f"❌ Fetch_data endpoint failed: {e}")
            return result
    
    async def test_feedback_loop_integration(self) -> Dict[str, Any]:
        """Test feedback loop integration"""
        logger.info("🔍 Testing feedback loop integration...")
        try:
            # Test adding feedback
            feedback_result = self.feedback_loop.add_feedback(
                symbol="TEST",
                predicted_action="long",
                user_feedback="correct",
                confidence=0.8,
                features=[0.1, 0.2, 0.3, 0.4, 0.5]
            )
            
            # Test adding trade outcome
            trade_outcome = TradeOutcome(
                symbol="TEST",
                action="long",
                entry_price=100.0,
                exit_price=105.0,
                entry_time=datetime.now(),
                exit_time=datetime.now(),
                actual_return=0.05,
                predicted_return=0.03,
                confidence=0.8,
                karma_score=0.9,
                outcome="positive"
            )
            
            outcome_result = self.feedback_loop.add_trade_outcome(trade_outcome)
            
            # Test karma weighting
            test_predictions = [
                {"symbol": "TEST", "score": 0.8, "confidence": 0.7},
                {"symbol": "OTHER", "score": 0.6, "confidence": 0.5}
            ]
            
            weighted_predictions = self.feedback_loop.apply_karma_weighting(test_predictions)
            
            result = {
                "test": "feedback_loop_integration",
                "status": "passed",
                "feedback_result": feedback_result,
                "outcome_result": outcome_result,
                "weighted_predictions": weighted_predictions,
                "feedback_stats": self.feedback_loop.get_feedback_stats()
            }
            logger.info("✅ Feedback loop integration passed")
            return result
            
        except Exception as e:
            result = {
                "test": "feedback_loop_integration",
                "status": "failed",
                "error": str(e)
            }
            logger.error(f"❌ Feedback loop integration failed: {e}")
            return result
    
    async def test_rate_limiting(self) -> Dict[str, Any]:
        """Test rate limiting functionality"""
        logger.info("🔍 Testing rate limiting...")
        try:
            # Make multiple rapid requests to test rate limiting
            requests_made = 0
            rate_limited = False
            
            for i in range(5):  # Make 5 rapid requests
                try:
                    response = await self.client.post(
                        f"{self.base_url}/prediction_agent/tools/predict",
                        json={"symbols": ["AAPL"], "horizon": "daily"},
                        headers=self._headers()
                    )
                    requests_made += 1
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 429:
                        rate_limited = True
                        break
            
            result = {
                "test": "rate_limiting",
                "status": "passed",
                "requests_made": requests_made,
                "rate_limited": rate_limited
            }
            logger.info(f"✅ Rate limiting test passed - {requests_made} requests made")
            return result
            
        except Exception as e:
            result = {
                "test": "rate_limiting",
                "status": "failed",
                "error": str(e)
            }
            logger.error(f"❌ Rate limiting test failed: {e}")
            return result
    
    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive integration test suite"""
        logger.info("=" * 60)
        logger.info("STARTING COMPREHENSIVE INTEGRATION TEST")
        logger.info("=" * 60)
        
        # Authenticate first
        if not await self.authenticate():
            return {"error": "Authentication failed"}
        
        # Run all tests
        tests = [
            self.test_health_check,
            self.test_predict_endpoint,
            self.test_scan_all_endpoint,
            self.test_analyze_endpoint,
            self.test_feedback_endpoint,
            self.test_train_rl_endpoint,
            self.test_fetch_data_endpoint,
            self.test_feedback_loop_integration,
            self.test_rate_limiting
        ]
        
        results = []
        passed_tests = 0
        total_tests = len(tests)
        
        for test_func in tests:
            try:
                result = await test_func()
                results.append(result)
                if result["status"] == "passed":
                    passed_tests += 1
            except Exception as e:
                error_result = {
                    "test": test_func.__name__,
                    "status": "failed",
                    "error": str(e)
                }
                results.append(error_result)
                logger.error(f"❌ Test {test_func.__name__} failed: {e}")
        
        # Generate summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "success_rate": passed_tests / total_tests,
            "results": results
        }
        
        # Save results
        results_file = Path("logs/integration_test_results.json")
        results_file.parent.mkdir(exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info("=" * 60)
        logger.info("INTEGRATION TEST COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Tests Passed: {passed_tests}/{total_tests}")
        logger.info(f"Success Rate: {summary['success_rate']:.1%}")
        logger.info(f"Results saved to: {results_file}")
        
        return summary
    
    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()


async def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Integration Test for Krishna Prediction Agent")
    parser.add_argument("--base-url", default="http://localhost:8000", 
                       help="API base URL")
    parser.add_argument("--test", choices=["all", "health", "predict", "feedback", "rate-limit"],
                       default="all", help="Specific test to run")
    
    args = parser.parse_args()
    
    tester = IntegrationTester(base_url=args.base_url)
    
    try:
        if args.test == "all":
            results = await tester.run_comprehensive_test()
        elif args.test == "health":
            await tester.authenticate()
            results = await tester.test_health_check()
        elif args.test == "predict":
            await tester.authenticate()
            results = await tester.test_predict_endpoint()
        elif args.test == "feedback":
            await tester.authenticate()
            results = await tester.test_feedback_endpoint()
        elif args.test == "rate-limit":
            await tester.authenticate()
            results = await tester.test_rate_limiting()
        
        print("\n" + "=" * 60)
        print("TEST RESULTS")
        print("=" * 60)
        print(json.dumps(results, indent=2, default=str))
        
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Test failed: {e}")
    finally:
        await tester.close()


if __name__ == "__main__":
    asyncio.run(main())
