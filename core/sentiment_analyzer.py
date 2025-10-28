"""
Sentiment Analysis Module for Blackhole Infiverse
Handles market sentiment indicators and news analysis
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import requests
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Market sentiment analysis and indicators"""
    
    def __init__(self, sentiment_cache_dir: str = "data/sentiment"):
        self.sentiment_cache_dir = Path(sentiment_cache_dir)
        self.sentiment_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Mock sentiment data (in production, this would connect to real APIs)
        self.sentiment_data = self._load_sentiment_cache()
    
    def _load_sentiment_cache(self) -> Dict[str, Dict]:
        """Load cached sentiment data"""
        cache_file = self.sentiment_cache_dir / "sentiment_cache.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load sentiment cache: {e}")
        return {}
    
    def _save_sentiment_cache(self):
        """Save sentiment data to cache"""
        cache_file = self.sentiment_cache_dir / "sentiment_cache.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(self.sentiment_data, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save sentiment cache: {e}")
    
    def compute_put_call_ratio(self, symbol: str) -> float:
        """Compute Put/Call Ratio (mock implementation)"""
        # In production, this would fetch real options data
        # For now, generate realistic mock data
        np.random.seed(hash(symbol) % 2**32)
        return np.random.uniform(0.5, 2.0)
    
    def compute_order_book_imbalance(self, symbol: str) -> float:
        """Compute Order Book Imbalance (mock implementation)"""
        # In production, this would analyze real order book data
        np.random.seed(hash(symbol) % 2**32)
        return np.random.uniform(-1.0, 1.0)
    
    def compute_bid_ask_spread(self, symbol: str) -> float:
        """Compute Bid-Ask Spread (mock implementation)"""
        # In production, this would use real market data
        np.random.seed(hash(symbol) % 2**32)
        return np.random.uniform(0.001, 0.01)
    
    def compute_short_interest(self, symbol: str) -> float:
        """Compute Short Interest Ratio (mock implementation)"""
        # In production, this would fetch real short interest data
        np.random.seed(hash(symbol) % 2**32)
        return np.random.uniform(0.1, 0.5)
    
    def compute_options_flow_sentiment(self, symbol: str) -> float:
        """Compute Options Flow Sentiment (mock implementation)"""
        # In production, this would analyze real options flow
        np.random.seed(hash(symbol) % 2**32)
        return np.random.uniform(-1.0, 1.0)
    
    def compute_news_sentiment(self, symbol: str) -> Dict[str, float]:
        """Compute News Sentiment Analysis (mock implementation)"""
        # In production, this would use NLP APIs like NewsAPI, Alpha Vantage, etc.
        np.random.seed(hash(symbol) % 2**32)
        
        return {
            'sentiment_score': np.random.uniform(-1.0, 1.0),
            'positive_ratio': np.random.uniform(0.0, 1.0),
            'negative_ratio': np.random.uniform(0.0, 1.0),
            'neutral_ratio': np.random.uniform(0.0, 1.0),
            'news_volume': np.random.uniform(0.0, 1.0),
            'sentiment_trend': np.random.uniform(-0.1, 0.1)
        }
    
    def compute_social_sentiment(self, symbol: str) -> Dict[str, float]:
        """Compute Social Media Sentiment (mock implementation)"""
        # In production, this would analyze Twitter, Reddit, etc.
        np.random.seed(hash(symbol) % 2**32)
        
        return {
            'twitter_sentiment': np.random.uniform(-1.0, 1.0),
            'reddit_sentiment': np.random.uniform(-1.0, 1.0),
            'social_volume': np.random.uniform(0.0, 1.0),
            'mentions_count': np.random.randint(100, 10000),
            'sentiment_momentum': np.random.uniform(-0.2, 0.2)
        }
    
    def compute_fear_greed_index(self) -> float:
        """Compute Fear & Greed Index (mock implementation)"""
        # In production, this would use CNN Fear & Greed Index API
        return np.random.uniform(0.0, 100.0)
    
    def compute_vix_sentiment(self) -> float:
        """Compute VIX-based Sentiment (mock implementation)"""
        # In production, this would use real VIX data
        return np.random.uniform(10.0, 50.0)
    
    def compute_institutional_sentiment(self, symbol: str) -> Dict[str, float]:
        """Compute Institutional Sentiment (mock implementation)"""
        # In production, this would analyze 13F filings, insider trading, etc.
        np.random.seed(hash(symbol) % 2**32)
        
        return {
            'institutional_flow': np.random.uniform(-1.0, 1.0),
            'insider_trading': np.random.uniform(-1.0, 1.0),
            'analyst_ratings': np.random.uniform(1.0, 5.0),
            'price_target_ratio': np.random.uniform(0.8, 1.2),
            'institutional_confidence': np.random.uniform(0.0, 1.0)
        }
    
    def compute_market_sentiment_features(self, symbol: str) -> Dict[str, float]:
        """Compute comprehensive market sentiment features"""
        try:
            # Get or generate sentiment data for symbol
            if symbol not in self.sentiment_data:
                self.sentiment_data[symbol] = {}
            
            symbol_data = self.sentiment_data[symbol]
            
            # Compute all sentiment indicators
            sentiment_features = {
                # Options sentiment
                'put_call_ratio': self.compute_put_call_ratio(symbol),
                'options_flow_sentiment': self.compute_options_flow_sentiment(symbol),
                
                # Market microstructure
                'order_book_imbalance': self.compute_order_book_imbalance(symbol),
                'bid_ask_spread': self.compute_bid_ask_spread(symbol),
                'short_interest': self.compute_short_interest(symbol),
                
                # News sentiment
                **self.compute_news_sentiment(symbol),
                
                # Social sentiment
                **self.compute_social_sentiment(symbol),
                
                # Market-wide sentiment
                'fear_greed_index': self.compute_fear_greed_index(),
                'vix_sentiment': self.compute_vix_sentiment(),
                
                # Institutional sentiment
                **self.compute_institutional_sentiment(symbol),
                
                # Composite sentiment score
                'composite_sentiment': 0.0,  # Will be calculated below
                'sentiment_momentum': 0.0,  # Will be calculated below
                'sentiment_volatility': 0.0  # Will be calculated below
            }
            
            # Calculate composite sentiment score
            sentiment_scores = [
                sentiment_features['sentiment_score'],
                sentiment_features['twitter_sentiment'],
                sentiment_features['reddit_sentiment'],
                sentiment_features['institutional_flow'],
                sentiment_features['options_flow_sentiment']
            ]
            
            sentiment_features['composite_sentiment'] = np.mean(sentiment_scores)
            
            # Calculate sentiment momentum (change over time)
            if 'previous_sentiment' in symbol_data:
                sentiment_features['sentiment_momentum'] = (
                    sentiment_features['composite_sentiment'] - symbol_data['previous_sentiment']
                )
            else:
                sentiment_features['sentiment_momentum'] = 0.0
            
            # Calculate sentiment volatility
            if 'sentiment_history' in symbol_data:
                history = symbol_data['sentiment_history']
                if len(history) > 1:
                    sentiment_features['sentiment_volatility'] = np.std(history)
                else:
                    sentiment_features['sentiment_volatility'] = 0.0
            else:
                sentiment_features['sentiment_volatility'] = 0.0
            
            # Update symbol data
            symbol_data['previous_sentiment'] = sentiment_features['composite_sentiment']
            if 'sentiment_history' not in symbol_data:
                symbol_data['sentiment_history'] = []
            symbol_data['sentiment_history'].append(sentiment_features['composite_sentiment'])
            
            # Keep only last 20 sentiment readings
            if len(symbol_data['sentiment_history']) > 20:
                symbol_data['sentiment_history'] = symbol_data['sentiment_history'][-20:]
            
            # Save updated data
            self.sentiment_data[symbol] = symbol_data
            self._save_sentiment_cache()
            
            logger.info(f"Computed {len(sentiment_features)} sentiment features for {symbol}")
            return sentiment_features
            
        except Exception as e:
            logger.error(f"Error computing sentiment features for {symbol}: {e}")
            # Return default sentiment features
            return {
                'put_call_ratio': 1.0,
                'options_flow_sentiment': 0.0,
                'order_book_imbalance': 0.0,
                'bid_ask_spread': 0.005,
                'short_interest': 0.2,
                'sentiment_score': 0.0,
                'positive_ratio': 0.33,
                'negative_ratio': 0.33,
                'neutral_ratio': 0.34,
                'news_volume': 0.5,
                'sentiment_trend': 0.0,
                'twitter_sentiment': 0.0,
                'reddit_sentiment': 0.0,
                'social_volume': 0.5,
                'mentions_count': 1000,
                'sentiment_momentum': 0.0,
                'fear_greed_index': 50.0,
                'vix_sentiment': 20.0,
                'institutional_flow': 0.0,
                'insider_trading': 0.0,
                'analyst_ratings': 3.0,
                'price_target_ratio': 1.0,
                'institutional_confidence': 0.5,
                'composite_sentiment': 0.0,
                'sentiment_momentum': 0.0,
                'sentiment_volatility': 0.0
            }
    
    def get_sentiment_summary(self, symbol: str) -> Dict[str, any]:
        """Get sentiment summary for a symbol"""
        sentiment_features = self.compute_market_sentiment_features(symbol)
        
        # Categorize sentiment
        composite = sentiment_features['composite_sentiment']
        if composite > 0.3:
            sentiment_category = "Very Bullish"
        elif composite > 0.1:
            sentiment_category = "Bullish"
        elif composite > -0.1:
            sentiment_category = "Neutral"
        elif composite > -0.3:
            sentiment_category = "Bearish"
        else:
            sentiment_category = "Very Bearish"
        
        return {
            'symbol': symbol,
            'composite_sentiment': composite,
            'sentiment_category': sentiment_category,
            'sentiment_momentum': sentiment_features['sentiment_momentum'],
            'sentiment_volatility': sentiment_features['sentiment_volatility'],
            'fear_greed_index': sentiment_features['fear_greed_index'],
            'vix_sentiment': sentiment_features['vix_sentiment'],
            'put_call_ratio': sentiment_features['put_call_ratio'],
            'short_interest': sentiment_features['short_interest'],
            'timestamp': datetime.now().isoformat()
        }
