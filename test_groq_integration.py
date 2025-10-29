"""
Test script for Groq API integration with stock analysis reason generation
"""

import os
import sys
import json
import logging
from datetime import datetime

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.llm_reason_generator import LLMReasonGenerator
from core.llm_config import get_llm_config
from core.prediction_analyzer import PredictionAnalyzer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set the Groq API key from environment variable
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY not found in environment variables")
    print("Please set GROQ_API_KEY environment variable to test Groq integration")
    sys.exit(1)


def create_sample_analysis_data():
    """Create sample analysis data for testing"""
    
    # Sample technical analysis data
    technical_analysis = {
        'momentum': {
            'signals': [
                'RSI oversold (28.3) - bullish momentum',
                'Stochastic oversold (15.2)',
                'Williams %R oversold (-82.1)'
            ],
            'strength': 0.4,
            'interpretation': 'Strong bullish momentum'
        },
        'trend': {
            'signals': [
                'Price above both SMAs - strong uptrend',
                'Price above SMA20 - short-term uptrend',
                'ADX strong trend (32.5)'
            ],
            'strength': 0.5,
            'interpretation': 'Strong uptrend'
        },
        'volume': {
            'signals': [
                'High volume (2.3x average) - strong conviction',
                'OBV positive - accumulation'
            ],
            'strength': 0.3,
            'interpretation': 'High volume confirmation'
        },
        'volatility': {
            'signals': [
                'Price near lower Bollinger Band (0.15) - potential reversal',
                'ATR at 3.45 - moderate volatility level'
            ],
            'strength': 0.2,
            'interpretation': 'Moderate volatility with reversal potential'
        },
        'patterns': {
            'signals': [
                'Hammer pattern detected - potential reversal',
                'Engulfing pattern detected - strong reversal signal'
            ],
            'interpretation': '2 pattern(s) detected'
        },
        'support_resistance': {
            'signals': [
                'Price above resistance - bullish breakout',
                'Price above pivot - bullish bias'
            ],
            'interpretation': 'Price above resistance - bullish breakout'
        }
    }
    
    # Sample sentiment analysis
    sentiment_analysis = {
        'signals': [
            'Bullish sentiment (0.45)',
            'Low put/call ratio (0.65) - bullish sentiment',
            'Low short interest (8.2%) - limited upside'
        ],
        'strength': 0.25,
        'interpretation': 'Bullish sentiment'
    }
    
    # Sample risk assessment
    risk_assessment = {
        'factors': [
            'Low volatility (2.1%) - reduced risk',
            'Normal volume levels'
        ],
        'level': 0.1,
        'interpretation': 'LOW RISK'
    }
    
    return {
        'technical_analysis': technical_analysis,
        'sentiment_analysis': sentiment_analysis,
        'risk_assessment': risk_assessment,
        'key_factors': [
            'Strong momentum indicators',
            'Strong trend alignment',
            'High volume confirmation',
            'Positive sentiment'
        ],
        'contradictory_signals': [],
        'recommendation_strength': 'STRONG'
    }


def test_groq_configuration():
    """Test Groq configuration"""
    
    print("Testing Groq Configuration")
    print("=" * 30)
    
    try:
        config = get_llm_config(provider="groq")
        
        print(f"   Provider: {config.provider}")
        print(f"   Model: {config.model}")
        print(f"   Max Tokens: {config.max_tokens}")
        print(f"   Temperature: {config.temperature}")
        print(f"   API Available: {config.is_available()}")
        print(f"   Base URL: {config.base_url}")
        print()
        
        return True
        
    except Exception as e:
        print(f"ERROR: Error testing Groq configuration: {e}")
        return False


def test_groq_reason_generation():
    """Test Groq reason generation with sample data"""
    
    print("Testing Groq-Powered Stock Analysis Reason Generation")
    print("=" * 60)
    
    # Create sample data
    symbol = "AAPL"
    action = "long"
    score = 0.7234
    current_price = 254.94
    
    analysis_data = create_sample_analysis_data()
    
    print(f"Testing with {symbol} - {action.upper()} recommendation")
    print(f"   Score: {score:.3f}, Price: ${current_price:.2f}")
    print()
    
    # Test Groq reason generator
    try:
        config = get_llm_config(provider="groq", model="llama-3.1-8b-instant")
        llm_generator = LLMReasonGenerator(config)
        
        print("Generating Groq-powered reason...")
        llm_reason = llm_generator.generate_stock_reason(
            symbol, analysis_data, action, score, current_price
        )
        
        print("SUCCESS: Groq Reason Generated:")
        print("-" * 50)
        print(llm_reason)
        print("-" * 50)
        print()
        
        # Test fallback reason generation
        print("Testing fallback reason generation...")
        fallback_reason = llm_generator._generate_fallback_reason(
            analysis_data, action, score
        )
        
        print("SUCCESS: Fallback Reason Generated:")
        print("-" * 50)
        print(fallback_reason)
        print("-" * 50)
        print()
        
        # Compare lengths
        print("Comparison:")
        print(f"   Groq reason length: {len(llm_reason)} characters")
        print(f"   Fallback reason length: {len(fallback_reason)} characters")
        print()
        
        # Check if Groq reason is different from fallback
        if llm_reason != fallback_reason:
            print("SUCCESS: Groq generated a unique response!")
        else:
            print("WARNING: Groq response same as fallback")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Error testing Groq integration: {e}")
        return False


def test_different_groq_models():
    """Test different Groq models"""
    
    print("Testing Different Groq Models")
    print("=" * 35)
    
    models_to_test = [
        "llama-3.1-8b-instant",
        "llama-3.1-70b-versatile", 
        "mixtral-8x7b-32768"
    ]
    
    symbol = "TSLA"
    action = "short"
    score = 0.234
    current_price = 180.50
    
    analysis_data = create_sample_analysis_data()
    
    results = []
    
    for model in models_to_test:
        print(f"Testing model: {model}")
        try:
            config = get_llm_config(provider="groq", model=model)
            llm_generator = LLMReasonGenerator(config)
            
            reason = llm_generator.generate_stock_reason(
                symbol, analysis_data, action, score, current_price
            )
            
            print(f"   SUCCESS: Generated {len(reason)} characters")
            results.append((model, True, len(reason)))
            
        except Exception as e:
            print(f"   ERROR: {e}")
            results.append((model, False, 0))
        
        print()
    
    # Summary
    print("Model Test Summary:")
    for model, success, length in results:
        status = "SUCCESS" if success else "FAILED"
        print(f"   {model}: {status} ({length} chars)")
    
    return all(success for _, success, _ in results)


def test_prediction_analyzer_integration():
    """Test integration with PredictionAnalyzer using Groq"""
    
    print("Testing PredictionAnalyzer Integration with Groq")
    print("=" * 50)
    
    try:
        # Create sample features
        features = {
            'close': 254.94,
            'rsi_14': 28.3,
            'stoch_k': 15.2,
            'williams_r': -82.1,
            'sma_20': 250.12,
            'sma_50': 245.67,
            'volume_ratio': 2.3,
            'bb_position': 0.15,
            'atr': 3.45,
            'hammer': 1,
            'engulfing': 1
        }
        
        symbol = "AAPL"
        action = "long"
        score = 0.7234
        
        # Test PredictionAnalyzer
        analyzer = PredictionAnalyzer()
        analysis = analyzer.analyze_prediction(symbol, features, action, score)
        
        print("SUCCESS: PredictionAnalyzer analysis generated")
        print(f"   Key factors: {len(analysis['key_factors'])}")
        print(f"   Technical signals: {len(analysis['technical_analysis']['momentum']['signals'])}")
        print()
        
        # Test Groq integration
        config = get_llm_config(provider="groq", model="llama-3.1-8b-instant")
        llm_generator = LLMReasonGenerator(config)
        llm_reason = llm_generator.generate_stock_reason(
            symbol, analysis, action, score, features['close']
        )
        
        print("SUCCESS: Groq integration successful")
        print(f"   Generated reason length: {len(llm_reason)} characters")
        print()
        
        # Show a preview of the reason
        preview = llm_reason[:200] + "..." if len(llm_reason) > 200 else llm_reason
        print("Reason Preview:")
        print("-" * 30)
        print(preview)
        print("-" * 30)
        print()
        
        return True
        
    except Exception as e:
        print(f"ERROR: Error testing PredictionAnalyzer integration: {e}")
        return False


def main():
    """Run all Groq tests"""
    
    print("Groq Integration Test Suite")
    print("=" * 50)
    print()
    
    tests = [
        ("Groq Configuration", test_groq_configuration),
        ("Groq Reason Generation", test_groq_reason_generation),
        ("Different Groq Models", test_different_groq_models),
        ("PredictionAnalyzer Integration", test_prediction_analyzer_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"Running {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
            print(f"SUCCESS: {test_name}: {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            print(f"ERROR: {test_name}: FAILED - {e}")
            results.append((test_name, False))
        print()
    
    # Summary
    print("Test Summary")
    print("=" * 20)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"   {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("SUCCESS: All Groq tests passed! Integration is ready.")
    else:
        print("WARNING: Some tests failed. Check the output above for details.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
