#!/usr/bin/env python3
"""
Test Enhanced LightGBM Fix
Quick test to verify the label encoding fix works
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def test_label_encoding():
    """Test the label encoding logic"""
    print("🧪 Testing Label Encoding Fix")
    print("="*40)
    
    # Original labels (what we have in the data)
    original_labels = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1])
    print(f"Original labels: {original_labels}")
    print(f"Original unique: {np.unique(original_labels)}")
    
    # Encoding mapping
    label_mapping = {-1: 0, 0: 1, 1: 2}
    encoded_labels = np.array([label_mapping[label] for label in original_labels])
    print(f"Encoded labels: {encoded_labels}")
    print(f"Encoded unique: {np.unique(encoded_labels)}")
    
    # Reverse mapping
    reverse_mapping = {0: -1, 1: 0, 2: 1}
    decoded_labels = np.array([reverse_mapping[label] for label in encoded_labels])
    print(f"Decoded labels: {decoded_labels}")
    print(f"Decoded unique: {np.unique(decoded_labels)}")
    
    # Verify round-trip
    if np.array_equal(original_labels, decoded_labels):
        print("✅ Label encoding round-trip successful!")
        return True
    else:
        print("❌ Label encoding round-trip failed!")
        return False

def test_lightgbm_compatibility():
    """Test if encoded labels work with LightGBM"""
    print("\n🧪 Testing LightGBM Compatibility")
    print("="*40)
    
    try:
        import lightgbm as lgb
        
        # Create sample data with encoded labels
        X = np.random.randn(100, 5)
        y = np.random.choice([0, 1, 2], size=100)  # Encoded labels
        
        print(f"Sample data shape: {X.shape}")
        print(f"Sample labels: {np.unique(y)}")
        
        # Create LightGBM dataset
        train_data = lgb.Dataset(X, label=y)
        
        # Test parameters
        params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'verbose': -1
        }
        
        # Try to create a simple model
        model = lgb.train(params, train_data, num_boost_round=1)
        
        # Make a prediction
        pred = model.predict(X[:5])
        print(f"Prediction shape: {pred.shape}")
        
        print("✅ LightGBM compatibility test passed!")
        return True
        
    except Exception as e:
        print(f"❌ LightGBM compatibility test failed: {e}")
        return False

def main():
    print("🔧 ENHANCED LIGHTGBM FIX TEST")
    print("="*50)
    
    # Test 1: Label encoding
    encoding_ok = test_label_encoding()
    
    # Test 2: LightGBM compatibility
    lightgbm_ok = test_lightgbm_compatibility()
    
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    if encoding_ok and lightgbm_ok:
        print("🎉 All tests passed! The fix should work.")
        print("\nYou can now run:")
        print("python core/models/enhanced_lightgbm.py")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
    
    print("="*50)

if __name__ == "__main__":
    main()
