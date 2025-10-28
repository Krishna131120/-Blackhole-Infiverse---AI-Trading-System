#!/usr/bin/env python3
"""
Enhanced Feature Store Update Script
Updates existing feature store with comprehensive technical indicators and sentiment analysis
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List
import time

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from core.enhanced_features import EnhancedFeaturePipeline
from core.sentiment_analyzer import SentimentAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureStoreUpdater:
    """Update feature store with enhanced indicators"""
    
    def __init__(self):
        self.enhanced_pipeline = EnhancedFeaturePipeline()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.updated_count = 0
        self.error_count = 0
    
    def update_feature_store(self, force_update: bool = False):
        """Update the entire feature store with enhanced features"""
        logger.info("="*60)
        logger.info("ENHANCED FEATURE STORE UPDATE")
        logger.info("="*60)
        
        # Load existing feature store
        existing_store = self.enhanced_pipeline.load_feature_store()
        
        if not existing_store:
            logger.error("No existing feature store found!")
            logger.info("Please run: python core/features.py first")
            return False
        
        logger.info(f"Found {len(existing_store)} symbols in existing feature store")
        
        # Process each symbol
        for symbol, df in existing_store.items():
            try:
                logger.info(f"Processing {symbol}...")
                
                # Check if already has enhanced features
                enhanced_features = [
                    'stoch_k', 'williams_r', 'cci', 'mfi', 'adx', 'vwap', 
                    'put_call_ratio', 'sentiment_score'
                ]
                
                has_enhanced = any(feat in df.columns for feat in enhanced_features)
                
                if has_enhanced and not force_update:
                    logger.info(f"  {symbol} already has enhanced features, skipping...")
                    continue
                
                # Update with enhanced features
                updated_df = self.enhanced_pipeline.compute_all_features(df)
                
                # Add sentiment features
                sentiment_features = self.sentiment_analyzer.compute_market_sentiment_features(symbol)
                for key, value in sentiment_features.items():
                    updated_df[key] = value
                
                # Save updated dataframe
                self.enhanced_pipeline.save_feature_store({symbol: updated_df})
                
                logger.info(f"  ✅ {symbol} updated: {updated_df.shape[1]} features")
                self.updated_count += 1
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"  ❌ Error updating {symbol}: {e}")
                self.error_count += 1
        
        # Summary
        logger.info("="*60)
        logger.info("UPDATE SUMMARY")
        logger.info("="*60)
        logger.info(f"Symbols updated: {self.updated_count}")
        logger.info(f"Errors: {self.error_count}")
        logger.info(f"Total symbols: {len(existing_store)}")
        
        if self.updated_count > 0:
            logger.info("✅ Feature store successfully updated with enhanced indicators!")
            logger.info("🎯 New indicators include:")
            logger.info("   • Momentum Oscillators: Stochastic, Williams %R, CCI, MFI, TRIX, CMO, Aroon, Ultimate")
            logger.info("   • Trend Indicators: ADX, Parabolic SAR, Keltner, Donchian")
            logger.info("   • Volume Indicators: ADL, CMF, EMV, Volume Trend")
            logger.info("   • Support/Resistance: Pivot Points, Fibonacci")
            logger.info("   • Candlestick Patterns: Doji, Hammer, Engulfing")
            logger.info("   • Advanced Analysis: VWAP, Sharpe, Beta, Alpha")
            logger.info("   • Sentiment Analysis: Put/Call, News, Social, Fear/Greed")
        else:
            logger.info("✅ All symbols already have enhanced features!")
            logger.info("🎯 Enhanced indicators already present:")
            logger.info("   • Momentum Oscillators: Stochastic, Williams %R, CCI, MFI, TRIX, CMO, Aroon, Ultimate")
            logger.info("   • Trend Indicators: ADX, Parabolic SAR, Keltner, Donchian")
            logger.info("   • Volume Indicators: ADL, CMF, EMV, Volume Trend")
            logger.info("   • Support/Resistance: Pivot Points, Fibonacci")
            logger.info("   • Candlestick Patterns: Doji, Hammer, Engulfing")
            logger.info("   • Advanced Analysis: VWAP, Sharpe, Beta, Alpha")
            logger.info("   • Sentiment Analysis: Put/Call, News, Social, Fear/Greed")
        
        # Return True if no errors occurred (even if no updates were needed)
        return self.error_count == 0
    
    def verify_enhanced_features(self):
        """Verify that enhanced features are present"""
        logger.info("Verifying enhanced features...")
        
        store = self.enhanced_pipeline.load_feature_store()
        if not store:
            logger.error("No feature store found!")
            return False
        
        # Check first symbol for enhanced features
        first_symbol = list(store.keys())[0]
        df = store[first_symbol]
        
        enhanced_features = [
            'stoch_k', 'williams_r', 'cci', 'mfi', 'adx', 'vwap',
            'put_call_ratio', 'sentiment_score', 'fear_greed_index'
        ]
        
        found_features = [feat for feat in enhanced_features if feat in df.columns]
        
        logger.info(f"Enhanced features found in {first_symbol}: {len(found_features)}/{len(enhanced_features)}")
        logger.info(f"Features: {found_features}")
        
        if len(found_features) >= len(enhanced_features) * 0.8:  # 80% threshold
            logger.info("✅ Enhanced features verification passed!")
            return True
        else:
            logger.warning("⚠️ Some enhanced features may be missing")
            return False


def main():
    """Main function"""
    updater = FeatureStoreUpdater()
    
    # Check command line arguments
    force_update = "--force" in sys.argv
    
    if force_update:
        logger.info("Force update mode enabled")
    
    # Update feature store
    success = updater.update_feature_store(force_update=force_update)
    
    if success:
        # Verify the update
        updater.verify_enhanced_features()
        
        logger.info("\n🚀 Next steps:")
        logger.info("1. Start your API server: python api/server.py")
        logger.info("2. Test predictions: python test_enhanced_analysis.py")
        logger.info("3. Use Postman collection for API testing")
    else:
        logger.error("Feature store update failed! Check errors above.")


if __name__ == "__main__":
    main()
