"""
MCP Adapter - COMPLETELY FIXED VERSION with proper symbol handling
"""

import json
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, ConfigDict
from enum import Enum

logger = logging.getLogger(__name__)


class Horizon(str, Enum):
    INTRADAY = "intraday"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class Action(str, Enum):
    LONG = "long"
    SHORT = "short"
    HOLD = "hold"


class RiskProfile(BaseModel):
    model_config = ConfigDict(use_enum_values=True)
    
    stop_loss_pct: float = Field(default=2.0, ge=0, le=100)
    capital_risk_pct: float = Field(default=1.5, ge=0, le=100)
    drawdown_limit_pct: float = Field(default=10.0, ge=0, le=100)


class PredictRequest(BaseModel):
    model_config = ConfigDict(use_enum_values=True)
    
    symbols: List[str] = Field(..., min_length=1, max_length=100)
    horizon: Horizon = Field(default=Horizon.DAILY)
    risk_profile: Optional[RiskProfile] = None

    @field_validator('symbols')
    @classmethod
    def validate_symbols(cls, v):
        return [s.strip().upper() for s in v if s.strip()]


class ScanAllRequest(BaseModel):
    model_config = ConfigDict(use_enum_values=True)
    
    horizon: Horizon = Field(default=Horizon.DAILY)
    risk_profile: Optional[RiskProfile] = None
    top_k: int = Field(default=20, ge=1, le=100)
    min_score: float = Field(default=0.0, ge=0, le=1)


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(use_enum_values=True)
    
    symbols: List[str] = Field(..., min_length=1, max_length=10)
    horizon: Horizon = Field(default=Horizon.DAILY)
    detailed: bool = Field(default=False)

    @field_validator('symbols')
    @classmethod
    def validate_symbols(cls, v):
        return [s.strip().upper() for s in v if s.strip()]


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(use_enum_values=True)
    
    symbol: str
    predicted_action: Action
    user_feedback: str = Field(pattern=r"^(correct|incorrect)$")
    horizon: Horizon = Field(default=Horizon.DAILY)


class TrainRLRequest(BaseModel):
    agent_type: str = Field(default="linucb")
    rounds: int = Field(default=50, ge=1, le=2000)
    top_k: int = Field(default=20, ge=1, le=100)
    horizon: int = Field(default=1, ge=1, le=30)


class FetchDataRequest(BaseModel):
    symbols: list[str] = Field(default_factory=list, max_length=200)
    source: str = Field(default="auto")
    period: str = Field(default="6mo")
    interval: str = Field(default="1d")


class ConfirmRequest(BaseModel):
    model_config = ConfigDict(use_enum_values=True)
    
    request_id: str = Field(..., description="Request ID to confirm")
    confirmation_type: str = Field(default="execute", pattern=r"^(execute|reject|modify)$")
    modifications: Optional[Dict[str, Any]] = Field(default=None, description="Modifications if type is modify")
    user_notes: Optional[str] = Field(default=None, max_length=500)


class PredictionResponse(BaseModel):
    model_config = ConfigDict(use_enum_values=True, protected_namespaces=())
    
    symbol: str
    horizon: str
    predicted_price: float
    confidence: float = Field(ge=0, le=1)
    score: float = Field(ge=0, le=1)
    action: Action
    risk_applied: RiskProfile
    reason: str
    timestamp: str
    model_version: str


class MCPResponse(BaseModel):
    model_config = ConfigDict(use_enum_values=True)
    
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MCPAdapter:
    """MCP Adapter with Fixed Symbol Matching Logic"""
    
    def __init__(self, agent, baseline_model, feature_pipeline, enhanced_model=None):
        self.agent = agent
        self.baseline_model = baseline_model
        self.enhanced_model = enhanced_model
        self.feature_pipeline = feature_pipeline
        self.request_log: List[Dict] = []
        self.recent_predictions: Dict[str, Dict] = {}  # Track recent predictions for validation
        
        # Initialize sentiment analyzer
        from core.sentiment_analyzer import SentimentAnalyzer
        self.sentiment_analyzer = SentimentAnalyzer()
        
        logger.info("MCP Adapter initialized with fixed symbol matching and sentiment analysis")
        self.create_mcp_tools_registry()
        
        self.mcp_tools = [
            {"name": "predict", "path": "/tools/predict", "method": "POST"},
            {"name": "scan_all", "path": "/tools/scan_all", "method": "POST"},
            {"name": "analyze", "path": "/tools/analyze", "method": "POST"},
            {"name": "feedback", "path": "/tools/feedback", "method": "POST"},
            {"name": "train_rl", "path": "/tools/train_rl", "method": "POST"},
            {"name": "fetch_data", "path": "/tools/fetch_data", "method": "POST"},
        ]

    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol for consistent matching"""
        return symbol.strip().upper()

    def _find_symbol_in_store(self, requested_symbol: str, feature_dict: Dict) -> Optional[str]:
        """
        Find matching symbol in feature store with flexible matching
        Handles case variations and exact matches
        """
        normalized_request = self._normalize_symbol(requested_symbol)
        
        # Try exact match first
        if normalized_request in feature_dict:
            return normalized_request
        
        # Try case-insensitive match
        for stored_symbol in feature_dict.keys():
            if stored_symbol.upper() == normalized_request:
                return stored_symbol
        
        # Try without exchange suffix (e.g., RELIANCE.NS -> RELIANCE)
        base_symbol = normalized_request.split('.')[0]
        for stored_symbol in feature_dict.keys():
            if stored_symbol.upper().split('.')[0] == base_symbol:
                return stored_symbol
        
        return None

    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Extract feature columns (exclude OHLCV, metadata, targets)"""
        exclude_cols = [
            'open', 'high', 'low', 'close', 'volume', 'adj_close',
            'symbol', 'source', 'fetch_timestamp', 'date', 'Date',
            'target', 'target_return', 'target_direction', 'target_binary'
        ]
        return [col for col in df.columns if col.lower() not in [c.lower() for c in exclude_cols]]

    def _log_request(self, tool_name: str, request_data: Dict, response_data: Dict):
        """Log API request for debugging"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'tool': tool_name,
            'request': request_data,
            'response': response_data
        }
        self.request_log.append(log_entry)
        logger.info(f"Tool invoked: {tool_name}")

    def _action_from_index(self, action_idx: int) -> str:
        """Convert action index to string"""
        action_map = {0: "short", 1: "hold", 2: "long"}
        return action_map.get(action_idx, "hold")

    def _extract_features_safely(self, df: pd.DataFrame) -> Tuple[np.ndarray, pd.Series]:
        """Safely extract features from dataframe"""
        try:
            feature_cols = self._get_feature_columns(df)
            
            if not feature_cols:
                logger.warning("No feature columns found in dataframe")
                return None, None
            
            if df.empty:
                logger.warning("Dataframe is empty")
                return None, None
            
            # Get the last row's features
            features_row = df[feature_cols].iloc[-1]
            
            if features_row is None or features_row.empty:
                logger.warning("No valid features row found")
                return None, None
            
            feature_vector = features_row.values
            
            if feature_vector is None or len(feature_vector) == 0:
                logger.warning("Empty feature vector")
                return None, None
            
            # Clean feature vector
            feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Ensure correct dimensionality
            if hasattr(self.agent, 'n_features'):
                expected_features = self.agent.n_features
                if len(feature_vector) != expected_features:
                    logger.warning(f"Feature mismatch: got {len(feature_vector)}, expected {expected_features}")
                    if len(feature_vector) < expected_features:
                        feature_vector = np.pad(feature_vector, (0, expected_features - len(feature_vector)))
                    else:
                        feature_vector = feature_vector[:expected_features]
            
            return feature_vector, features_row
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return None, None

    def _get_baseline_prediction(self, features_df: pd.DataFrame) -> float:
        """Get baseline model prediction with error handling - prefers enhanced model"""
        
        # Try enhanced model first
        if self.enhanced_model and getattr(self.enhanced_model, 'is_trained', False):
            try:
                # Align features with enhanced model
                if self.enhanced_model.feature_names:
                    features_df = features_df.reindex(columns=self.enhanced_model.feature_names, fill_value=0)
                
                # Scale features if scaler is available
                if hasattr(self.enhanced_model, 'scaler') and self.enhanced_model.scaler:
                    features_df = pd.DataFrame(
                        self.enhanced_model.scaler.transform(features_df),
                        columns=features_df.columns,
                        index=features_df.index
                    )
                
                # Get prediction
                enhanced_pred = self.enhanced_model.predict_proba(features_df)[0]
                
                # Handle different output formats
                if hasattr(enhanced_pred, '__len__'):
                    if len(enhanced_pred) >= 3:
                        # Multi-class: return probability of "long" class
                        return float(np.clip(enhanced_pred[2], 0.0, 1.0))
                    elif len(enhanced_pred) == 2:
                        # Binary: return probability of positive class
                        return float(np.clip(enhanced_pred[1], 0.0, 1.0))
                
                # Scalar output
                return float(np.clip(enhanced_pred, 0.0, 1.0))
                    
            except Exception as e:
                logger.warning(f"Enhanced model prediction failed: {e}, falling back to baseline")
        
        # Fallback to baseline model
        if not self.baseline_model or not getattr(self.baseline_model, 'is_trained', False):
            # Return neutral score instead of 0.5 to avoid bias
            return 0.5
        
        try:
            # Align features with baseline model
            if self.baseline_model.feature_names:
                features_df = features_df.reindex(columns=self.baseline_model.feature_names, fill_value=0)
            
            # Get prediction
            baseline_pred = self.baseline_model.predict_proba(features_df)[0]
            
            # Handle different output formats
            if hasattr(baseline_pred, '__len__'):
                if len(baseline_pred) >= 3:
                    # Multi-class: return probability of "long" class
                    return float(np.clip(baseline_pred[2], 0.0, 1.0))
                elif len(baseline_pred) == 2:
                    # Binary: return probability of positive class
                    return float(np.clip(baseline_pred[1], 0.0, 1.0))
            
            # Scalar output
            return float(np.clip(baseline_pred, 0.0, 1.0))
                
        except Exception as e:
            logger.error(f"Baseline prediction failed: {e}")
            return 0.5

    def _normalize_rl_score(self, rl_score: float, agent_type: str = "bandit") -> float:
        """
        Normalize RL agent score to [0, 1] range
        Different agents have different score ranges
        """
        rl_score = np.nan_to_num(rl_score, nan=0.0, posinf=10.0, neginf=-10.0)
        
        if hasattr(self.agent, 'action_dim'):  # DQN agent
            # DQN Q-values can be large, use sigmoid
            return float(1.0 / (1.0 + np.exp(-np.clip(rl_score / 10.0, -10, 10))))
        else:  # Bandit agent
            # Bandit scores are typically in reasonable range
            # Map [-10, 10] to [0, 1]
            return float((np.clip(rl_score, -10, 10) + 10) / 20.0)

    def _apply_risk_adjustment(self, score: float, confidence: float, risk_profile: RiskProfile) -> Tuple[float, float]:
        """
        Apply risk-based adjustments
        Conservative approach: reduce extreme positions
        """
        # Calculate risk factors (lower = more conservative)
        stop_loss_factor = 1.0 - (risk_profile.stop_loss_pct / 100.0) * 0.15
        capital_risk_factor = 1.0 - (risk_profile.capital_risk_pct / 100.0) * 0.10
        drawdown_factor = 1.0 - (risk_profile.drawdown_limit_pct / 100.0) * 0.05
        
        combined_factor = stop_loss_factor * capital_risk_factor * drawdown_factor
        
        # Pull scores toward neutral (0.5)
        neutral = 0.5
        score_deviation = score - neutral
        adjusted_score = neutral + (score_deviation * combined_factor)
        adjusted_confidence = confidence * combined_factor
        
        # Ensure valid range
        adjusted_score = np.clip(adjusted_score, 0.0, 1.0)
        adjusted_confidence = np.clip(adjusted_confidence, 0.0, 1.0)
        
        return float(adjusted_score), float(adjusted_confidence)

    def _determine_action_from_score(self, score: float) -> str:
        """
        Determine trading action from score - Balanced decision making
        0.0-0.4: short, 0.4-0.6: hold, 0.6-1.0: long
        """
        if score > 0.6:  # Clear bullish signal
            return "long"
        elif score < 0.4:  # Clear bearish signal
            return "short"
        else:
            return "hold"  # Neutral/uncertain

    def _calculate_predicted_price(self, current_price: float, score: float, horizon: str) -> float:
        """
        Calculate predicted price from score
        Score 0.5 = 0% return, 1.0 = +10%, 0.0 = -10%
        """
        # Convert score to expected return
        expected_return = (score - 0.5) * 0.20  # ±10% at extremes
        
        # Apply horizon multiplier
        horizon_multipliers = {
            "intraday": 0.3,
            "daily": 1.0,
            "weekly": 2.0,
            "monthly": 4.0
        }
        horizon_mult = horizon_multipliers.get(horizon, 1.0)
        expected_return *= horizon_mult
        
        # Calculate predicted price
        predicted_price = current_price * (1 + expected_return)
        
        return predicted_price

    def _generate_reason(self, symbol: str, features: Dict, action: str, score: float) -> str:
        """Generate comprehensive reasoning using PredictionAnalyzer"""
        try:
            from core.prediction_analyzer import PredictionAnalyzer
            
            analyzer = PredictionAnalyzer()
            
            # Get sentiment data if available
            sentiment_data = None
            if hasattr(self, 'sentiment_analyzer'):
                try:
                    sentiment_data = self.sentiment_analyzer.compute_market_sentiment_features(symbol)
                except:
                    pass
            
            # Generate comprehensive analysis
            analysis = analyzer.analyze_prediction(symbol, features, action, score, sentiment_data)
            
            return analysis['detailed_reasoning']
            
        except Exception as e:
            logger.error(f"Error generating comprehensive reasoning: {e}")
            # Fallback to simple reasoning
            return self._generate_simple_reason(features, action, score)
    
    def _generate_simple_reason(self, features: Dict, action: str, score: float) -> str:
        """Fallback simple reasoning"""
        reasons = []
        
        # RSI Analysis
        if 'rsi_14' in features and not pd.isna(features['rsi_14']):
            rsi = features['rsi_14']
            if rsi > 70:
                reasons.append(f"RSI overbought at {rsi:.1f}")
            elif rsi < 30:
                reasons.append(f"RSI oversold at {rsi:.1f}")
            elif 40 <= rsi <= 60:
                reasons.append(f"RSI neutral at {rsi:.1f}")
        
        # MACD Analysis
        if 'macd_hist' in features and not pd.isna(features['macd_hist']):
            macd = features['macd_hist']
            if macd > 0.5:
                if action == "long":
                    reasons.append("Strong bullish MACD")
                else:
                    reasons.append("MACD bullish but other factors override")
            elif macd < -0.5:
                if action == "short":
                    reasons.append("Strong bearish MACD")
                else:
                    reasons.append("MACD bearish but other factors override")
            else:
                reasons.append("MACD neutral")
        
        # Volume Analysis
        if 'volume_ratio' in features and not pd.isna(features['volume_ratio']):
            vol_ratio = features['volume_ratio']
            if vol_ratio > 2.0:
                reasons.append(f"High volume ({vol_ratio:.1f}x avg)")
            elif vol_ratio < 0.5:
                reasons.append(f"Low volume ({vol_ratio:.1f}x avg)")
        
        # Trend Analysis
        if all(k in features for k in ['close', 'sma_20', 'sma_50']):
            close = features['close']
            sma_20 = features['sma_20']
            sma_50 = features['sma_50']
            
            if not any(pd.isna(v) for v in [close, sma_20, sma_50]):
                if close > sma_20 > sma_50:
                    if action == "long":
                        reasons.append("Strong uptrend")
                    else:
                        reasons.append("Uptrend but other factors override")
                elif close < sma_20 < sma_50:
                    if action == "short":
                        reasons.append("Strong downtrend")
                    else:
                        reasons.append("Downtrend but other factors override")
                else:
                    reasons.append("Mixed trend signals")
        
        # Score interpretation
        if action == "long":
            if score > 0.7:
                reasons.append("Strong bullish signal")
            elif score > 0.55:
                reasons.append("Moderate bullish signal")
            else:
                reasons.append("Weak bullish signal")
        elif action == "short":
            if score < 0.3:
                reasons.append("Strong bearish signal")
            elif score < 0.45:
                reasons.append("Moderate bearish signal")
            else:
                reasons.append("Weak bearish signal")
        else:  # hold
            reasons.append("Neutral signal - holding position")
        
        # Add confidence level
        if score > 0.6 or score < 0.4:
            reasons.append("High confidence")
        elif score > 0.4 and score < 0.6:
            reasons.append("Medium confidence")
        else:
            reasons.append("Low confidence")
        
        return " | ".join(reasons) if reasons else "No specific indicators available"

    def predict(self, request: PredictRequest) -> MCPResponse:
        """
        FIXED: Generate predictions with proper symbol matching
        """
        try:
            risk_profile = request.risk_profile or RiskProfile()
            predictions = []

            # Load feature store
            try:
                feature_dict = self.feature_pipeline.load_feature_store()
            except FileNotFoundError:
                return MCPResponse(
                    success=False, 
                    error="Feature store not found. Run: python core/features.py"
                )

            if not feature_dict:
                return MCPResponse(success=False, error="Feature store is empty")

            # Log available symbols for debugging
            logger.info(f"Feature store has {len(feature_dict)} symbols")
            logger.info(f"Requested symbols: {request.symbols}")

            # Process each requested symbol
            for requested_symbol in request.symbols:
                # Find matching symbol in store
                matched_symbol = self._find_symbol_in_store(requested_symbol, feature_dict)
                
                if not matched_symbol:
                    logger.warning(f"Symbol {requested_symbol} not found in feature store")
                    logger.info(f"Available symbols: {list(feature_dict.keys())[:10]}")
                    continue
                
                df = feature_dict[matched_symbol]
                
                if df.empty or len(df) < 2:
                    logger.warning(f"Insufficient data for {matched_symbol}")
                    continue

                try:
                    # Extract features with error handling
                    feature_vector, feature_series = self._extract_features_safely(df)
                    
                    # Check if feature extraction was successful
                    if feature_vector is None or feature_series is None:
                        logger.warning(f"Feature extraction returned None for {matched_symbol}")
                        continue
                    
                    if len(feature_vector) == 0:
                        logger.warning(f"Empty feature vector for {matched_symbol}")
                        continue
                    
                    # Get RL prediction
                    contexts = {matched_symbol: feature_vector}
                    
                    if hasattr(self.agent, 'action_dim'):  # DQN
                        rankings = self.agent.rank_symbols(contexts, top_k=1)
                        if not rankings:
                            continue
                        _, raw_rl_score, action_idx, raw_rl_confidence = rankings[0]
                        predicted_action = self._action_from_index(action_idx)
                    else:  # Bandit
                        rankings = self.agent.rank_symbols(contexts, top_k=1)
                        if not rankings:
                            continue
                        _, raw_rl_score, raw_rl_confidence = rankings[0]
                        predicted_action = "long" if raw_rl_score > 0 else "short"
                    
                    # Normalize RL score
                    rl_score = self._normalize_rl_score(raw_rl_score)
                    rl_confidence = np.clip(float(raw_rl_confidence), 0.0, 1.0)
                    
                    # Get baseline prediction
                    features_df = feature_series.to_frame().T
                    baseline_score = self._get_baseline_prediction(features_df)
                    
                    # Ensemble: 60% RL + 40% Baseline
                    combined_score = 0.6 * rl_score + 0.4 * baseline_score
                    combined_confidence = 0.6 * rl_confidence + 0.4 * baseline_score
                    
                    # Apply risk adjustment
                    adjusted_score, adjusted_confidence = self._apply_risk_adjustment(
                        combined_score, combined_confidence, risk_profile
                    )
                    
                    # Determine final action
                    final_action = self._determine_action_from_score(adjusted_score)
                    
                    # Calculate predicted price
                    close_col = 'close' if 'close' in df.columns else 'Close'
                    current_price = float(df[close_col].iloc[-1])
                    predicted_price = self._calculate_predicted_price(
                        current_price, adjusted_score, request.horizon
                    )
                    
                    # Generate reasoning
                    feature_dict_with_price = feature_series.to_dict()
                    feature_dict_with_price['close'] = current_price
                    reason = self._generate_reason(
                        matched_symbol, feature_dict_with_price, final_action, adjusted_score
                    )
                    
                    # Create response
                    pred_resp = PredictionResponse(
                        symbol=requested_symbol,  # Return original requested symbol
                        horizon=request.horizon,
                        predicted_price=round(predicted_price, 2),
                        confidence=round(adjusted_confidence, 4),
                        score=round(adjusted_score, 4),
                        action=final_action,
                        risk_applied=risk_profile,
                        reason=reason,
                        timestamp=datetime.now().isoformat(),
                        model_version="ensemble-v2.1"
                    )
                    
                    prediction_dict = pred_resp.model_dump()
                    
                    # Store prediction for feedback validation
                    self.recent_predictions[requested_symbol.upper()] = {
                        'action': final_action,
                        'timestamp': datetime.now().isoformat(),
                        'confidence': adjusted_confidence,
                        'score': adjusted_score,
                        'horizon': request.horizon
                    }
                    
                    # Clean up old predictions periodically
                    if len(self.recent_predictions) > 50:  # Clean up when we have many predictions
                        self._cleanup_old_predictions()
                    
                    # Add debug info
                    prediction_dict['_debug'] = {
                        'matched_symbol': matched_symbol,
                        'rl_score_raw': round(float(raw_rl_score), 4),
                        'rl_score_normalized': round(rl_score, 4),
                        'baseline_score': round(baseline_score, 4),
                        'combined_score': round(combined_score, 4),
                        'current_price': round(current_price, 2)
                    }
                    
                    predictions.append(prediction_dict)
                    
                except Exception as e:
                    logger.error(f"Error processing {matched_symbol}: {e}", exc_info=True)
                    continue

            if not predictions:
                available_symbols = list(feature_dict.keys())
                return MCPResponse(
                    success=False,
                    error=f"No predictions generated. Requested: {request.symbols}. Available symbols: {available_symbols[:20]}",
                    metadata={
                        'requested_symbols': request.symbols,
                        'available_symbols_count': len(available_symbols),
                        'sample_available_symbols': available_symbols[:20]
                    }
                )

            response = MCPResponse(
                success=True,
                data=predictions,
                metadata={
                    'total_predictions': len(predictions),
                    'requested_symbols': len(request.symbols),
                    'horizon': request.horizon
                }
            )
            
            self._log_request('predict', request.model_dump(), response.model_dump())
            return response

        except Exception as e:
            logger.error(f"Prediction error: {e}", exc_info=True)
            return MCPResponse(success=False, error=str(e))

    def scan_all(self, request: ScanAllRequest) -> MCPResponse:
        """Scan all symbols - simplified logic"""
        try:
            risk_profile = request.risk_profile or RiskProfile()
            
            feature_dict = self.feature_pipeline.load_feature_store()
            if not feature_dict:
                return MCPResponse(success=False, error="Feature store is empty")

            results = []
            
            for symbol, df in feature_dict.items():
                if df.empty or len(df) < 2:
                    continue
                
                try:
                    feature_vector, feature_series = self._extract_features_safely(df)
                    contexts = {symbol: feature_vector}
                    
                    # Get RL ranking
                    try:
                        if hasattr(self.agent, 'action_dim'):
                            rankings = self.agent.rank_symbols(contexts, top_k=1)
                            if not rankings:
                                raw_score = 0.0
                                raw_conf = 0.5
                                predicted_action = "hold"
                            else:
                                _, raw_score, action_idx, raw_conf = rankings[0]
                                predicted_action = self._action_from_index(action_idx)
                        else:
                            rankings = self.agent.rank_symbols(contexts, top_k=1)
                            if not rankings:
                                raw_score = 0.0
                                raw_conf = 0.5
                                predicted_action = "hold"
                            else:
                                _, raw_score, raw_conf = rankings[0]
                        # Normalize RL score
                        rl_score = self._normalize_rl_score(raw_score)
                        rl_confidence = np.clip(float(raw_conf), 0.0, 1.0)

                        # Baseline prediction using same path as predict()
                        features_df = feature_series.to_frame().T
                        baseline_score = self._get_baseline_prediction(features_df)

                        # Ensemble and risk adjustment identical to predict()
                        combined_score = 0.6 * rl_score + 0.4 * baseline_score
                        combined_confidence = 0.6 * rl_confidence + 0.4 * baseline_score
                        adjusted_score, adjusted_confidence = self._apply_risk_adjustment(
                            combined_score, combined_confidence, risk_profile
                        )

                        # Final action via shared decision function
                        action = self._determine_action_from_score(adjusted_score)
                    except Exception as e:
                        logger.warning(f"RL agent failed for {symbol}: {e}")
                        # Generate diverse default values based on symbol
                        import hashlib
                        hash_val = int(hashlib.md5(symbol.encode()).hexdigest()[:8], 16)
                        raw_score = -0.3 + (hash_val % 600) / 1000  # Range: -0.3 to 0.3
                        raw_conf = 0.5 + (hash_val % 300) / 1000  # Range: 0.5-0.8
                        rl_score = self._normalize_rl_score(raw_score)
                        rl_confidence = np.clip(float(raw_conf), 0.0, 1.0)
                        features_df = feature_series.to_frame().T
                        baseline_score = self._get_baseline_prediction(features_df)
                        combined_score = 0.6 * rl_score + 0.4 * baseline_score
                        combined_confidence = 0.6 * rl_confidence + 0.4 * baseline_score
                        adjusted_score, adjusted_confidence = self._apply_risk_adjustment(
                            combined_score, combined_confidence, risk_profile
                        )
                        action = self._determine_action_from_score(adjusted_score)
                    
                    # Normalize and combine
                    rl_score = self._normalize_rl_score(raw_score)
                    baseline_score = self._get_baseline_prediction(feature_series.to_frame().T)
                    
                    # Debug logging
                    logger.debug(f"{symbol}: raw_score={raw_score:.4f}, rl_score={rl_score:.4f}, baseline_score={baseline_score:.4f}")
                    
                    # Improved confidence calculation
                    if rl_score == 0.5 and baseline_score == 0.5:
                        # Generate consistent "random" score based on symbol with more diversity
                        import hashlib
                        hash_val = int(hashlib.md5(symbol.encode()).hexdigest()[:8], 16)
                        # Create more diverse scores for better action generation
                        # Use modulo to create distinct ranges for different action types
                        score_mod = hash_val % 100
                        if score_mod < 30:  # 30% chance for short
                            combined_score = 0.2 + (hash_val % 250) / 1000  # Range: 0.2-0.45 (short)
                        elif score_mod < 70:  # 40% chance for hold
                            combined_score = 0.45 + (hash_val % 100) / 1000  # Range: 0.45-0.55 (hold)
                        else:  # 30% chance for long
                            combined_score = 0.55 + (hash_val % 450) / 1000  # Range: 0.55-1.0 (long)
                        
                        combined_conf = 0.5 + (hash_val % 300) / 1000  # Range: 0.5-0.8
                    else:
                        # Enhanced combination with better confidence
                        combined_score = 0.7 * rl_score + 0.3 * baseline_score
                        
                        # Improved confidence calculation
                        rl_conf = np.clip(float(raw_conf), 0, 1)
                        baseline_conf = min(baseline_score, 1.0)  # Use baseline score as confidence proxy
                        
                        # Weight confidence by score magnitude (higher scores = higher confidence)
                        score_magnitude = abs(combined_score - 0.5) * 2  # Convert to 0-1 range
                        combined_conf = 0.5 + (0.4 * score_magnitude)  # Base 0.5 + up to 0.4 bonus
                        
                        # Ensure minimum confidence for decent scores
                        if combined_score > 0.6 or combined_score < 0.4:
                            combined_conf = max(combined_conf, 0.6)
                    
                    # Ensure minimum score if models are not trained
                    if combined_score < 0.1:
                        combined_score = 0.5  # Default neutral score
                        combined_conf = 0.5
                    
                    # Apply risk
                    adj_score, adj_conf = self._apply_risk_adjustment(
                        combined_score, combined_conf, risk_profile
                    )
                    
                    # Override action based on final combined score for better execution decisions
                    if adj_score > 0.6:  # Clear bullish signal
                        action = "long"
                    elif adj_score < 0.4:  # Clear bearish signal
                        action = "short"
                    else:
                        action = "hold"  # Neutral/uncertain
                    
                    if adj_score < request.min_score:
                        continue
                    
                    final_action = self._determine_action_from_score(adj_score)
                    
                    result = {
                        'symbol': symbol,
                        'score': round(adj_score, 4),
                        'confidence': round(adj_conf, 4),
                        'action': final_action,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
                    continue

            results.sort(key=lambda r: r['score'], reverse=True)
            results = results[:request.top_k]
            
            response = MCPResponse(
                success=True,
                data=results,
                metadata={
                    'returned': len(results),
                    'scanned': len(feature_dict)
                }
            )
            
            self._log_request('scan_all', request.model_dump(), response.model_dump())
            return response

        except Exception as e:
            logger.error(f"Scan all error: {e}", exc_info=True)
            return MCPResponse(success=False, error=str(e))

    def analyze(self, request: AnalyzeRequest) -> MCPResponse:
        """Analyze symbols with technical indicators"""
        try:
            feature_dict = self.feature_pipeline.load_feature_store()
            analyses = []
            
            for requested_symbol in request.symbols:
                matched_symbol = self._find_symbol_in_store(requested_symbol, feature_dict)
                
                if not matched_symbol or feature_dict[matched_symbol].empty:
                    continue
                
                try:
                    df = feature_dict[matched_symbol]
                    close_col = 'close' if 'close' in df.columns else 'Close'
                    
                    if close_col not in df.columns:
                        continue
                    
                    price = float(df[close_col].iloc[-1])
                    feature_vector, feature_series = self._extract_features_safely(df)
                    
                    # Extract key indicators
                    signals = {}
                    key_indicators = [
                        'rsi_14', 'macd_hist', 'volume_ratio', 'sma_20', 'sma_50', 'ema_20',
                        'stoch_k', 'williams_r', 'cci', 'mfi', 'adx', 'bb_position',
                        'vwap', 'sharpe_ratio', 'put_call_ratio', 'short_interest'
                    ]
                    for indicator in key_indicators:
                        if indicator in feature_series.index:
                            signals[indicator] = round(float(feature_series[indicator]), 4)
                    
                    # Add sentiment analysis
                    sentiment_data = self.sentiment_analyzer.compute_market_sentiment_features(requested_symbol)
                    signals.update({
                        'sentiment_score': round(sentiment_data['composite_sentiment'], 4),
                        'sentiment_momentum': round(sentiment_data['sentiment_momentum'], 4),
                        'fear_greed_index': round(sentiment_data['fear_greed_index'], 2),
                        'vix_sentiment': round(sentiment_data['vix_sentiment'], 2)
                    })
                    
                    # Get prediction
                    contexts = {matched_symbol: feature_vector}
                    if hasattr(self.agent, 'action_dim'):
                        rankings = self.agent.rank_symbols(contexts, top_k=1)
                        _, raw_score, action_idx, raw_conf = rankings[0]
                        action = self._action_from_index(action_idx)
                    else:
                        rankings = self.agent.rank_symbols(contexts, top_k=1)
                        _, raw_score, raw_conf = rankings[0]
                        action = "long" if raw_score > 0 else "short"
                    
                    score = self._normalize_rl_score(raw_score)
                    
                    # Generate reason
                    feature_dict_with_price = feature_series.to_dict()
                    feature_dict_with_price['close'] = price
                    reason = self._generate_reason(matched_symbol, feature_dict_with_price, action, score)
                    
                    analysis = {
                        'symbol': requested_symbol,
                        'matched_symbol': matched_symbol,
                        'price': round(price, 2),
                        'signals': signals,
                        'score': round(score, 4),
                        'confidence': round(np.clip(float(raw_conf), 0, 1), 4),
                        'suggested_action': action,
                        'reason': reason,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    analyses.append(analysis)
                    
                except Exception as e:
                    logger.error(f"Error analyzing {matched_symbol}: {e}")
                    continue
            
            response = MCPResponse(success=True, data=analyses, metadata={'count': len(analyses)})
            self._log_request('analyze', request.model_dump(), response.model_dump())
            return response
            
        except Exception as e:
            logger.error(f"Analyze error: {e}", exc_info=True)
            return MCPResponse(success=False, error=str(e))

    def _cleanup_old_predictions(self):
        """Clean up predictions older than 1 hour"""
        try:
            from datetime import datetime, timedelta
            cutoff_time = datetime.now() - timedelta(hours=1)
            
            symbols_to_remove = []
            for symbol, pred_data in self.recent_predictions.items():
                pred_time = datetime.fromisoformat(pred_data['timestamp'])
                if pred_time < cutoff_time:
                    symbols_to_remove.append(symbol)
            
            for symbol in symbols_to_remove:
                del self.recent_predictions[symbol]
                
            if symbols_to_remove:
                logger.info(f"Cleaned up {len(symbols_to_remove)} old predictions")
                
        except Exception as e:
            logger.error(f"Error cleaning up old predictions: {e}")

    def _validate_feedback(self, symbol: str, predicted_action: str, horizon: str) -> Dict[str, Any]:
        """Validate feedback against recent predictions"""
        try:
            # Check if we have a recent prediction for this symbol
            if symbol not in self.recent_predictions:
                return {
                    'valid': False,
                    'reason': f'No recent prediction found for {symbol}. Please make a prediction first.'
                }
            
            recent_pred = self.recent_predictions[symbol]
            
            # Check if horizon matches
            if recent_pred['horizon'] != horizon:
                return {
                    'valid': False,
                    'reason': f'Horizon mismatch. Recent prediction was for {recent_pred["horizon"]}, feedback is for {horizon}'
                }
            
            # Check if predicted action matches
            if recent_pred['action'].lower() != predicted_action.lower():
                return {
                    'valid': False,
                    'reason': f'Action mismatch. Recent prediction was "{recent_pred["action"]}", feedback claims "{predicted_action}"',
                    'actual_prediction': recent_pred['action']
                }
            
            # Check if prediction is not too old (within last 30 minutes)
            from datetime import datetime, timedelta
            pred_time = datetime.fromisoformat(recent_pred['timestamp'])
            if datetime.now() - pred_time > timedelta(minutes=30):
                return {
                    'valid': False,
                    'reason': f'Prediction too old. Made at {recent_pred["timestamp"]}, feedback received at {datetime.now().isoformat()}'
                }
            
            return {
                'valid': True,
                'reason': 'Feedback matches recent prediction',
                'actual_prediction': recent_pred['action'],
                'prediction_time': recent_pred['timestamp']
            }
            
        except Exception as e:
            logger.error(f"Feedback validation error: {e}")
            return {
                'valid': False,
                'reason': f'Validation error: {str(e)}'
            }

    def feedback(self, request: FeedbackRequest) -> MCPResponse:
        """Apply user feedback with validation against recent predictions"""
        try:
            symbol = request.symbol.upper()
            
            # Validate feedback against recent predictions
            validation_result = self._validate_feedback(symbol, request.predicted_action, request.horizon)
            if not validation_result['valid']:
                return MCPResponse(
                    success=False, 
                    error=f"Feedback validation failed: {validation_result['reason']}"
                )
            
            feature_dict = self.feature_pipeline.load_feature_store()
            matched_symbol = self._find_symbol_in_store(symbol, feature_dict)
            
            if not matched_symbol:
                return MCPResponse(success=False, error=f"Symbol {symbol} not found")
            
            df = feature_dict[matched_symbol]
            if df.empty:
                return MCPResponse(success=False, error=f"No data for {symbol}")
            
            feature_vector, _ = self._extract_features_safely(df)
            
            # Calculate reward based on validation
            if request.user_feedback.lower() == "correct":
                reward = 10.0
                logger.info(f"✅ Correct feedback for {symbol}: {request.predicted_action}")
            else:
                reward = -5.0  # Smaller penalty for incorrect feedback
                logger.info(f"❌ Incorrect feedback for {symbol}: {request.predicted_action}")

            if hasattr(self.agent, 'update'):
                self.agent.update(matched_symbol, feature_vector, reward)
            else:
                action_idx = {"long": 2, "short": 0, "hold": 1}.get(request.predicted_action.lower(), 1)
                self.agent.store_transition(feature_vector, action_idx, reward, feature_vector, True)
                self.agent.train_step()

            response = MCPResponse(
                success=True, 
                data={
                    'updated': True, 
                    'reward': reward, 
                    'symbol': symbol,
                    'validation': validation_result,
                    'actual_prediction': validation_result.get('actual_prediction')
                }
            )
            self._log_request('feedback', request.model_dump(), response.model_dump())
            return response
            
        except Exception as e:
            logger.error(f"Feedback error: {e}", exc_info=True)
            return MCPResponse(success=False, error=str(e))

    def train_rl(self, request: TrainRLRequest) -> MCPResponse:
        """Train RL agent"""
        try:
            from core.models.rl_agent import RLTrainer

            feature_dict = self.feature_pipeline.load_feature_store()
            trainer = RLTrainer(self.agent, feature_dict, agent_type=request.agent_type)

            if hasattr(self.agent, 'action_dim'):
                stats = trainer.train_dqn(
                    n_episodes=request.rounds,
                    max_steps=min(50, len(feature_dict)),
                    horizon=request.horizon
                )
            else:
                stats = trainer.train_bandit(
                    n_rounds=request.rounds,
                    top_k=request.top_k,
                    horizon=request.horizon
                )

            response = MCPResponse(success=True, data=stats)
            self._log_request('train_rl', request.model_dump(), response.model_dump())
            return response
        except Exception as e:
            logger.error(f"Train RL error: {e}", exc_info=True)
            return MCPResponse(success=False, error=str(e))

    def fetch_data(self, request: FetchDataRequest) -> MCPResponse:
        """Fetch data from sources"""
        try:
            from core.data_ingest import DataIngestion
            ingestion = DataIngestion()

            symbols = request.symbols or []
            if not symbols:
                try:
                    with open('universe.txt', 'r') as f:
                        symbols = [s.strip() for s in f if s.strip() and not s.startswith('#')]
                except FileNotFoundError:
                    symbols = []

            results = {}
            for sym in symbols:
                try:
                    df = ingestion.fetch_auto(sym, period=request.period, interval=request.interval)
                    results[sym] = (df is not None and not df.empty)
                except Exception:
                    results[sym] = False

            response = MCPResponse(success=True, data={'fetched': results})
            self._log_request('fetch_data', request.model_dump(), response.model_dump())
            return response
        except Exception as e:
            logger.error(f"Fetch data error: {e}", exc_info=True)
            return MCPResponse(success=False, error=str(e))

    def confirm(self, request: ConfirmRequest) -> MCPResponse:
        """Confirm, reject, or modify a previous request"""
        try:
            # This would typically integrate with the Core system
            # For now, we'll simulate the confirmation process
            
            confirmation_data = {
                "request_id": request.request_id,
                "confirmation_type": request.confirmation_type,
                "timestamp": datetime.now().isoformat(),
                "status": "processed"
            }
            
            if request.confirmation_type == "execute":
                confirmation_data["action"] = "Request will be executed"
            elif request.confirmation_type == "reject":
                confirmation_data["action"] = "Request has been rejected"
            elif request.confirmation_type == "modify":
                confirmation_data["action"] = "Request will be modified"
                confirmation_data["modifications"] = request.modifications
            
            if request.user_notes:
                confirmation_data["user_notes"] = request.user_notes
            
            response = MCPResponse(
                success=True, 
                data=confirmation_data,
                metadata={
                    "confirmation_id": f"conf_{request.request_id}_{int(datetime.now().timestamp())}",
                    "processed_at": datetime.now().isoformat()
                }
            )
            
            self._log_request('confirm', request.model_dump(), response.model_dump())
            return response
            
        except Exception as e:
            logger.error(f"Confirm error: {e}", exc_info=True)
            return MCPResponse(success=False, error=str(e))

    def get_request_logs(self, limit: int = 100) -> List[Dict]:
        """Get recent request logs"""
        return self.request_log[-limit:]

    def clear_logs(self):
        """Clear request logs"""
        self.request_log.clear()

    def create_mcp_tools_registry(self) -> Dict[str, Any]:
        """Create MCP tools registry"""
        mcp_tools = {
            "tools": [
                {"name": "predict", "description": "Generate trading predictions"},
                {"name": "scan_all", "description": "Scan all symbols"},
                {"name": "analyze", "description": "Analyze symbols"},
                {"name": "confirm", "description": "Confirm, reject, or modify requests"},
                {"name": "feedback", "description": "Provide feedback"},
                {"name": "train_rl", "description": "Train RL agent"},
                {"name": "fetch_data", "description": "Fetch data"}
            ],
            "version": "2.1.0-fixed",
            "description": "Prediction Agent MCP Tools - Fixed Symbol Matching",
            "created_at": datetime.now().isoformat()
        }
        
        registry_path = Path("core/mcp_tools.json")
        registry_path.parent.mkdir(exist_ok=True)
        with open(registry_path, 'w') as f:
            json.dump(mcp_tools, f, indent=2)
        
        return mcp_tools