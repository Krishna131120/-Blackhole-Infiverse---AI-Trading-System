"""
Simple Feedback Loop System for Integration Testing
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class TradeOutcome:
    """Trade outcome data structure"""
    symbol: str
    action: str  # 'long', 'short', 'hold'
    predicted_price: float
    actual_price: float
    timestamp: str
    user_feedback: str  # 'correct', 'incorrect'
    confidence: float
    karma_score: float = 0.0


class SimpleFeedbackLoop:
    """Simple feedback loop for testing purposes"""
    
    def __init__(self, feedback_file: str = "logs/feedback_loop.json"):
        self.feedback_file = Path(feedback_file)
        self.feedback_file.parent.mkdir(exist_ok=True)
        self.feedback_data = self._load_feedback_data()
    
    def _load_feedback_data(self) -> Dict[str, Any]:
        """Load existing feedback data"""
        if self.feedback_file.exists():
            try:
                with open(self.feedback_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load feedback data: {e}")
        
        return {
            "feedbacks": [],
            "trade_outcomes": [],
            "karma_scores": {},
            "stats": {
                "total_feedbacks": 0,
                "correct_predictions": 0,
                "incorrect_predictions": 0,
                "accuracy": 0.0
            }
        }
    
    def _save_feedback_data(self):
        """Save feedback data to file"""
        try:
            with open(self.feedback_file, 'w') as f:
                json.dump(self.feedback_data, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save feedback data: {e}")
    
    def add_feedback(self, symbol: str, predicted_action: str, user_feedback: str, 
                    confidence: float = 0.5, features: List[float] = None) -> Dict[str, Any]:
        """Add user feedback"""
        feedback = {
            "symbol": symbol,
            "action": predicted_action,  # Use predicted_action as action
            "predicted_action": predicted_action,
            "user_feedback": user_feedback,
            "confidence": confidence,
            "features": features or [],
            "timestamp": datetime.now().isoformat()
        }
        
        self.feedback_data["feedbacks"].append(feedback)
        self.feedback_data["stats"]["total_feedbacks"] += 1
        
        if user_feedback == "correct":
            self.feedback_data["stats"]["correct_predictions"] += 1
        else:
            self.feedback_data["stats"]["incorrect_predictions"] += 1
        
        # Update accuracy
        total = self.feedback_data["stats"]["correct_predictions"] + self.feedback_data["stats"]["incorrect_predictions"]
        if total > 0:
            self.feedback_data["stats"]["accuracy"] = self.feedback_data["stats"]["correct_predictions"] / total
        
        self._save_feedback_data()
        
        return {
            "success": True,
            "feedback_id": f"fb_{len(self.feedback_data['feedbacks'])}",
            "message": "Feedback recorded successfully"
        }
    
    def add_trade_outcome(self, trade_outcome: TradeOutcome) -> Dict[str, Any]:
        """Add trade outcome"""
        outcome_dict = asdict(trade_outcome)
        self.feedback_data["trade_outcomes"].append(outcome_dict)
        
        # Update karma score
        symbol = trade_outcome.symbol
        if symbol not in self.feedback_data["karma_scores"]:
            self.feedback_data["karma_scores"][symbol] = 0.0
        
        # Simple karma calculation
        if trade_outcome.user_feedback == "correct":
            self.feedback_data["karma_scores"][symbol] += 0.1
        else:
            self.feedback_data["karma_scores"][symbol] -= 0.05
        
        # Keep karma between 0 and 1
        self.feedback_data["karma_scores"][symbol] = max(0.0, min(1.0, self.feedback_data["karma_scores"][symbol]))
        
        self._save_feedback_data()
        
        return {
            "success": True,
            "outcome_id": f"outcome_{len(self.feedback_data['trade_outcomes'])}",
            "karma_score": self.feedback_data["karma_scores"][symbol]
        }
    
    def apply_karma_weighting(self, predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply karma weighting to predictions"""
        weighted_predictions = []
        
        for pred in predictions:
            symbol = pred.get("symbol", "")
            karma_score = self.feedback_data["karma_scores"].get(symbol, 0.5)
            
            # Apply karma weighting to confidence
            original_confidence = pred.get("confidence", 0.5)
            weighted_confidence = original_confidence * (0.5 + karma_score * 0.5)
            
            weighted_pred = pred.copy()
            weighted_pred["confidence"] = weighted_confidence
            weighted_pred["karma_score"] = karma_score
            weighted_pred["original_confidence"] = original_confidence
            
            weighted_predictions.append(weighted_pred)
        
        return weighted_predictions
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """Get feedback statistics"""
        return self.feedback_data["stats"].copy()


def get_feedback_loop() -> SimpleFeedbackLoop:
    """Get feedback loop instance"""
    return SimpleFeedbackLoop()