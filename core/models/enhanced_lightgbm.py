#!/usr/bin/env python3
"""
Enhanced LightGBM Model with Hyperparameter Optimization and Advanced Features
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import optuna
import joblib
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class EnhancedLightGBM:
    """Enhanced LightGBM model with advanced features and optimization"""
    
    def __init__(self, model_dir: str = "./models", task: str = "classification", model_name: str = "enhanced-lightgbm"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.task = task
        self.model_name = model_name
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_trained = False
        self.best_params = None
        self.metrics = {}
        
    def _create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced technical indicators and features"""
        
        # Basic features
        df['hl_range'] = df['high'] - df['low']
        df['price_change'] = df['close'].pct_change()
        df['volume_change'] = df['volume'].pct_change()
        
        # Price-based features
        for window in [5, 10, 20, 50]:
            df[f'price_change_{window}'] = df['close'].pct_change(window)
            df[f'volatility_{window}'] = df['close'].rolling(window).std()
            df[f'sma_{window}'] = df['close'].rolling(window).mean()
            df[f'price_vs_sma_{window}'] = df['close'] / df[f'sma_{window}'] - 1
            
        # Volume-based features
        df['volume_sma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
        # Technical indicators
        df['rsi_14'] = self._calculate_rsi(df['close'], 14)
        df['rsi_21'] = self._calculate_rsi(df['close'], 21)
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = self._calculate_macd(df['close'])
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = self._calculate_bollinger_bands(df['close'])
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Momentum indicators
        df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
        df['momentum_20'] = df['close'] / df['close'].shift(20) - 1
        
        # Volatility features
        df['volatility_ratio'] = df['volatility_10'] / df['volatility_20']
        
        # Drop NaN values
        df = df.dropna()
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist
    
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, num_std: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window).mean()
        std = prices.rolling(window).std()
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        return upper, sma, lower
    
    def _optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series, n_trials: int = 50) -> Dict:
        """Optimize hyperparameters using Optuna"""
        
        def objective(trial):
            params = {
                'objective': 'multiclass',
                'num_class': 3,
                'metric': 'multi_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': trial.suggest_int('num_leaves', 10, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
                'verbose': -1,
                'random_state': 42
            }
            
            # Use time series split for validation
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Apply SMOTE to training data
                smote = SMOTE(random_state=42)
                X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_balanced)
                X_val_scaled = scaler.transform(X_val)
                
                # Train model
                train_data = lgb.Dataset(X_train_scaled, label=y_train_balanced)
                val_data = lgb.Dataset(X_val_scaled, label=y_val, reference=train_data)
                
                model = lgb.train(
                    params,
                    train_data,
                    valid_sets=[val_data],
                    num_boost_round=1000,
                    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
                )
                
                # Predict and calculate score
                y_pred = model.predict(X_val_scaled, num_iteration=model.best_iteration)
                y_pred_class = np.argmax(y_pred, axis=1)
                score = accuracy_score(y_val, y_pred_class)
                scores.append(score)
            
            return np.mean(scores)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        return study.best_params
    
    def train(self, feature_store_path: str = "./data/features", force_retrain: bool = False):
        """Train the enhanced LightGBM model"""
        
        print("="*60)
        print("ENHANCED LIGHTGBM TRAINING")
        print("="*60)
        
        # Check if model already exists
        model_path = self.model_dir / f"{self.model_name}.pkl"
        if model_path.exists() and not force_retrain:
            print(f"Model {self.model_name} already exists. Use force_retrain=True to retrain.")
            return
        
        # Load feature store
        print("[1] Loading feature store...")
        feature_store_path = Path(feature_store_path)
        
        if not feature_store_path.exists():
            raise FileNotFoundError(f"Feature store not found: {feature_store_path}")
        
        feature_files = list(feature_store_path.glob("*.parquet"))
        if not feature_files:
            raise FileNotFoundError(f"No feature files found in {feature_store_path}")
        
        # Load and combine features
        feature_dict = {}
        for file_path in feature_files:
            symbol = file_path.stem
            df = pd.read_parquet(file_path)
            if not df.empty:
                feature_dict[symbol] = df
        
        print(f"[OK] Loaded features for {len(feature_dict)} symbols")
        
        # Combine features from all symbols
        print("[2] Combining features from all symbols...")
        all_data = []
        
        for symbol, df in feature_dict.items():
            if len(df) < 2:
                continue
            
            # Create advanced features
            df_enhanced = self._create_advanced_features(df.copy())
            
            if len(df_enhanced) < 2:
                continue
            
            # Add symbol column
            df_enhanced['symbol'] = symbol
            all_data.append(df_enhanced)
        
        if not all_data:
            raise ValueError("No valid data found after feature engineering")
        
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"[OK] Combined data shape: {combined_df.shape}")
        
        # Prepare features and target
        print("[3] Preparing features and target...")
        
        # Get numeric columns only
        numeric_cols = combined_df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in ['target_direction', 'symbol']]
        
        X = combined_df[feature_cols].copy()
        y = combined_df['target_direction'].copy()
        
        # Remove rows with NaN targets
        valid_idx = ~y.isna()
        X = X[valid_idx]
        y = y[valid_idx]
        
        # Fix infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        
        # Remove any remaining NaN values
        valid_idx = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_idx]
        y = y[valid_idx]
        
        print(f"[OK] Feature matrix shape: {X.shape}")
        print(f"[OK] Target distribution:\n{y.value_counts()}")
        print(f"[OK] Target classes: {sorted(y.unique())}")
        print(f"[OK] Number of classes: {len(y.unique())}")
        
        # Encode target labels to start from 0 (LightGBM requirement)
        # Original: -1 (short), 0 (hold), 1 (long)
        # Encoded:  0 (short), 1 (hold), 2 (long)
        label_mapping = {-1: 0, 0: 1, 1: 2}
        y_encoded = y.map(label_mapping)
        
        print(f"[OK] Encoded target distribution:\n{y_encoded.value_counts()}")
        print(f"[OK] Encoded target classes: {sorted(y_encoded.unique())}")
        
        # Use encoded target
        y = y_encoded
        
        # Split data
        print("[4] Splitting data...")
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"[OK] Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        
        # Get actual number of classes
        n_classes = len(y.unique())
        print(f"[OK] Detected {n_classes} classes: {sorted(y.unique())}")
        
        # Optimize hyperparameters
        print("[5] Optimizing hyperparameters...")
        self.best_params = self._optimize_hyperparameters(X_train, y_train, n_trials=50)
        # Ensure num_class matches actual classes
        self.best_params['num_class'] = n_classes
        print(f"[OK] Best parameters: {self.best_params}")
        
        # Prepare final training data with SMOTE
        print("[6] Applying SMOTE for class balancing...")
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        print(f"[OK] Balanced training data shape: {X_train_balanced.shape}")
        print(f"[OK] Balanced target distribution:\n{pd.Series(y_train_balanced).value_counts()}")
        
        # Scale features
        print("[7] Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train_balanced)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train final model
        print("[8] Training final model...")
        train_data = lgb.Dataset(X_train_scaled, label=y_train_balanced)
        test_data = lgb.Dataset(X_test_scaled, label=y_test, reference=train_data)
        
        # Ensure all required parameters are set
        final_params = self.best_params.copy()
        final_params.update({
            'objective': 'multiclass',
            'metric': 'multi_logloss',
            'verbose': -1,
            'random_state': 42
        })
        
        self.model = lgb.train(
            final_params,
            train_data,
            valid_sets=[test_data],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)]
        )
        
        # Evaluate model
        print("[9] Evaluating model...")
        y_pred_proba = self.model.predict(X_test_scaled, num_iteration=self.model.best_iteration)
        
        # Handle different prediction formats
        if y_pred_proba.ndim == 1:
            # Single dimension - convert to class predictions
            y_pred = np.round(y_pred_proba).astype(int)
            # Create dummy probability array for metrics
            y_pred_proba_2d = np.column_stack([1-y_pred_proba, y_pred_proba])
        else:
            # Multi-dimensional - use argmax
            y_pred = np.argmax(y_pred_proba, axis=1)
            y_pred_proba_2d = y_pred_proba
        
        # Calculate metrics
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted')
        }
        
        # Calculate AUC only if we have proper multiclass probabilities
        try:
            if y_pred_proba_2d.shape[1] >= 3:  # 3 classes: 0, 1, 2
                self.metrics['auc'] = roc_auc_score(y_test, y_pred_proba_2d, multi_class='ovr', average='weighted')
            else:
                # For binary case, use the positive class probability
                self.metrics['auc'] = roc_auc_score(y_test, y_pred_proba_2d[:, 1] if y_pred_proba_2d.shape[1] == 2 else y_pred_proba_2d.flatten())
        except Exception as e:
            logger.warning(f"Could not calculate AUC: {e}")
            self.metrics['auc'] = 0.0
        
        print(f"\nEnhanced Model Performance:")
        for metric, value in self.metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # Save model
        print("[10] Saving model...")
        self.feature_names = feature_cols
        self.is_trained = True
        
        # Save model components
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, self.model_dir / f"{self.model_name}_scaler.pkl")
        
        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'feature_names': self.feature_names,
            'best_params': self.best_params,
            'metrics': self.metrics,
            'training_date': datetime.now().isoformat(),
            'is_trained': self.is_trained
        }
        
        with open(self.model_dir / f"{self.model_name}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"[OK] Enhanced model saved to {model_path}")
        print(f"[OK] Metrics saved to {self.model_dir / f'{self.model_name}_metadata.json'}")
        
        print("\n" + "="*60)
        print("ENHANCED MODEL TRAINING COMPLETE")
        print("="*60)
        
        return self.metrics
    
    def load(self):
        """Load the trained model"""
        
        model_path = self.model_dir / f"{self.model_name}.pkl"
        scaler_path = self.model_dir / f"{self.model_name}_scaler.pkl"
        metadata_path = self.model_dir / f"{self.model_name}_metadata.json"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load model components
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
        # Load metadata
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.feature_names = metadata.get('feature_names', [])
                self.best_params = metadata.get('best_params', {})
                self.metrics = metadata.get('metrics', {})
                self.is_trained = metadata.get('is_trained', False)
        
        print(f"Enhanced model loaded: {self.model_name}")
        return True
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Make probability predictions"""
        
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Align features
        if self.feature_names:
            X = X.reindex(columns=self.feature_names, fill_value=0)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict
        predictions = self.model.predict(X_scaled, num_iteration=self.model.best_iteration)
        
        # Handle different prediction formats
        if predictions.ndim == 1:
            # Single dimension - create probability matrix
            predictions = np.column_stack([1-predictions, predictions])
        
        return predictions
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make class predictions"""
        
        proba = self.predict_proba(X)
        if proba.ndim == 1:
            predictions = np.round(proba).astype(int)
        else:
            predictions = np.argmax(proba, axis=1)
        
        # Decode predictions back to original labels
        # Encoded: 0 (short), 1 (hold), 2 (long)
        # Original: -1 (short), 0 (hold), 1 (long)
        reverse_mapping = {0: -1, 1: 0, 2: 1}
        decoded_predictions = np.array([reverse_mapping[pred] for pred in predictions])
        
        return decoded_predictions

if __name__ == "__main__":
    # Train enhanced model
    enhanced_model = EnhancedLightGBM(
        model_dir="./models",
        task="classification",
        model_name="enhanced-lightgbm-v2"
    )
    
    try:
        metrics = enhanced_model.train(force_retrain=True)
        print(f"\nEnhanced Model Performance:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    except Exception as e:
        print(f"Error training enhanced model: {e}")
        import traceback
        traceback.print_exc()
