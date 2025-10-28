"""
Resilience and Validation System
Handles retry logic, validation, mismatch detection, and automatic halt signals
"""

import asyncio
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class ValidationLevel(str, Enum):
    STRICT = "strict"
    MODERATE = "moderate"
    LENIENT = "lenient"


class MismatchSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ValidationRule:
    """Validation rule definition"""
    name: str
    field: str
    validator: Callable[[Any], bool]
    error_message: str
    severity: MismatchSeverity = MismatchSeverity.MEDIUM


@dataclass
class RetryConfig:
    """Retry configuration"""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True


@dataclass
class MismatchAlert:
    """Mismatch alert record"""
    alert_id: str
    timestamp: datetime
    severity: MismatchSeverity
    field: str
    expected: Any
    actual: Any
    difference: float
    threshold: float
    context: Dict[str, Any]
    resolved: bool = False


class ResilienceSystem:
    """
    Resilience and Validation System
    Provides retry logic, validation, mismatch detection, and automatic halt signals
    """
    
    def __init__(self, 
                 validation_level: ValidationLevel = ValidationLevel.MODERATE,
                 mismatch_threshold: float = 0.03,  # 3% threshold as specified
                 alert_file: str = "logs/mismatch_alerts.json"):
        self.validation_level = validation_level
        self.mismatch_threshold = mismatch_threshold
        self.alert_file = Path(alert_file)
        self.alert_file.parent.mkdir(parents=True, exist_ok=True)
        
        # State tracking
        self.active_alerts: List[MismatchAlert] = []
        self.halt_signals: Dict[str, bool] = {}
        self.validation_rules: List[ValidationRule] = []
        
        # Statistics
        self.stats = {
            'total_validations': 0,
            'failed_validations': 0,
            'mismatch_alerts': 0,
            'halt_signals_triggered': 0,
            'retry_attempts': 0,
            'successful_retries': 0
        }
        
        # Load existing alerts
        self._load_alerts()
        
        # Initialize default validation rules
        self._setup_default_rules()
        
        logger.info(f"Resilience system initialized: level={validation_level}, threshold={mismatch_threshold}")
    
    def _load_alerts(self):
        """Load existing alerts from file"""
        try:
            if self.alert_file.exists():
                with open(self.alert_file, 'r') as f:
                    alerts_data = json.load(f)
                    for alert_data in alerts_data:
                        alert_data['timestamp'] = datetime.fromisoformat(alert_data['timestamp'])
                        alert_data['severity'] = MismatchSeverity(alert_data['severity'])
                        self.active_alerts.append(MismatchAlert(**alert_data))
                logger.info(f"Loaded {len(self.active_alerts)} existing alerts")
        except Exception as e:
            logger.error(f"Failed to load alerts: {e}")
    
    def _save_alerts(self):
        """Save alerts to file"""
        try:
            alerts_data = []
            for alert in self.active_alerts:
                alert_dict = {
                    'alert_id': alert.alert_id,
                    'timestamp': alert.timestamp.isoformat(),
                    'severity': alert.severity.value,
                    'field': alert.field,
                    'expected': alert.expected,
                    'actual': alert.actual,
                    'difference': alert.difference,
                    'threshold': alert.threshold,
                    'context': alert.context,
                    'resolved': alert.resolved
                }
                alerts_data.append(alert_dict)
            
            with open(self.alert_file, 'w') as f:
                json.dump(alerts_data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save alerts: {e}")
    
    def _setup_default_rules(self):
        """Setup default validation rules"""
        # Prediction validation rules
        self.add_validation_rule(
            ValidationRule(
                name="confidence_range",
                field="confidence",
                validator=lambda x: 0.0 <= x <= 1.0,
                error_message="Confidence must be between 0.0 and 1.0",
                severity=MismatchSeverity.HIGH
            )
        )
        
        self.add_validation_rule(
            ValidationRule(
                name="score_range",
                field="score",
                validator=lambda x: 0.0 <= x <= 1.0,
                error_message="Score must be between 0.0 and 1.0",
                severity=MismatchSeverity.HIGH
            )
        )
        
        self.add_validation_rule(
            ValidationRule(
                name="price_positive",
                field="predicted_price",
                validator=lambda x: x > 0,
                error_message="Predicted price must be positive",
                severity=MismatchSeverity.CRITICAL
            )
        )
        
        self.add_validation_rule(
            ValidationRule(
                name="action_valid",
                field="action",
                validator=lambda x: x in ["long", "short", "hold"],
                error_message="Action must be long, short, or hold",
                severity=MismatchSeverity.HIGH
            )
        )
        
        # Consistency validation rules
        self.add_validation_rule(
            ValidationRule(
                name="action_score_consistency",
                field="action_score_consistency",
                validator=self._validate_action_score_consistency,
                error_message="Action and score must be consistent",
                severity=MismatchSeverity.MEDIUM
            )
        )
        
        self.add_validation_rule(
            ValidationRule(
                name="reason_action_consistency",
                field="reason_action_consistency",
                validator=self._validate_reason_action_consistency,
                error_message="Reason and action must be consistent",
                severity=MismatchSeverity.MEDIUM
            )
        )
    
    def add_validation_rule(self, rule: ValidationRule):
        """Add a validation rule"""
        self.validation_rules.append(rule)
        logger.info(f"Added validation rule: {rule.name}")
    
    def _validate_action_score_consistency(self, data: Dict[str, Any]) -> bool:
        """Validate that action and score are consistent"""
        action = data.get('action')
        score = data.get('score')
        
        if action == "long" and score < 0.6:
            return False
        elif action == "short" and score > 0.4:
            return False
        elif action == "hold" and (score < 0.4 or score > 0.6):
            return False
        
        return True
    
    def _validate_reason_action_consistency(self, data: Dict[str, Any]) -> bool:
        """Validate that reason and action are consistent"""
        action = data.get('action')
        reason = data.get('reason', '').lower()
        
        if action == "long" and any(word in reason for word in ["bearish", "short", "sell", "decline"]):
            return False
        elif action == "short" and any(word in reason for word in ["bullish", "long", "buy", "rise"]):
            return False
        
        return True
    
    async def validate_response(self, response: Dict[str, Any], context: Dict[str, Any] = None) -> Tuple[bool, List[str]]:
        """
        Validate a response against all rules
        
        Returns:
            (is_valid, error_messages)
        """
        self.stats['total_validations'] += 1
        errors = []
        context = context or {}
        
        for rule in self.validation_rules:
            try:
                # Get field value
                field_value = self._get_nested_field(response, rule.field)
                
                # Validate
                if not rule.validator(field_value if rule.field in response else response):
                    error_msg = f"{rule.name}: {rule.error_message}"
                    errors.append(error_msg)
                    
                    # Create mismatch alert if severity is high enough
                    if rule.severity in [MismatchSeverity.HIGH, MismatchSeverity.CRITICAL]:
                        await self._create_mismatch_alert(
                            field=rule.field,
                            expected="valid",
                            actual=field_value,
                            severity=rule.severity,
                            context=context
                        )
                
            except Exception as e:
                logger.error(f"Validation error for rule {rule.name}: {e}")
                errors.append(f"{rule.name}: Validation error - {e}")
        
        is_valid = len(errors) == 0
        if not is_valid:
            self.stats['failed_validations'] += 1
        
        return is_valid, errors
    
    def _get_nested_field(self, data: Dict[str, Any], field: str) -> Any:
        """Get nested field value from data"""
        if '.' in field:
            keys = field.split('.')
            value = data
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return None
            return value
        else:
            return data.get(field)
    
    async def _create_mismatch_alert(self, 
                                   field: str, 
                                   expected: Any, 
                                   actual: Any, 
                                   severity: MismatchSeverity,
                                   context: Dict[str, Any]):
        """Create a mismatch alert"""
        # Calculate difference
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            if expected != 0:
                difference = abs(actual - expected) / abs(expected)
            else:
                difference = abs(actual)
        else:
            difference = 1.0 if expected != actual else 0.0
        
        # Create alert
        alert = MismatchAlert(
            alert_id=f"alert_{int(time.time())}_{len(self.active_alerts)}",
            timestamp=datetime.now(timezone.utc),
            severity=severity,
            field=field,
            expected=expected,
            actual=actual,
            difference=difference,
            threshold=self.mismatch_threshold,
            context=context
        )
        
        self.active_alerts.append(alert)
        self.stats['mismatch_alerts'] += 1
        
        # Check if halt signal should be triggered
        if difference > self.mismatch_threshold:
            await self._trigger_halt_signal(field, difference, context)
        
        # Save alerts
        self._save_alerts()
        
        logger.warning(f"Mismatch alert created: {field} - expected={expected}, actual={actual}, diff={difference:.4f}")
    
    async def _trigger_halt_signal(self, field: str, difference: float, context: Dict[str, Any]):
        """Trigger halt signal for critical mismatches"""
        halt_key = f"{field}_{context.get('symbol', 'unknown')}"
        self.halt_signals[halt_key] = True
        self.stats['halt_signals_triggered'] += 1
        
        logger.critical(f"HALT SIGNAL TRIGGERED: {field} mismatch {difference:.4f} exceeds threshold {self.mismatch_threshold}")
        
        # Log halt signal to bucket
        try:
            from core.bhiv_core import BHIVCore
            # This would be injected in real usage
            pass
        except Exception as e:
            logger.error(f"Failed to log halt signal: {e}")
    
    def is_halted(self, field: str, context: Dict[str, Any] = None) -> bool:
        """Check if system is halted for a field"""
        if context:
            halt_key = f"{field}_{context.get('symbol', 'unknown')}"
            return self.halt_signals.get(halt_key, False)
        else:
            return any(self.halt_signals.values())
    
    def clear_halt_signal(self, field: str, context: Dict[str, Any] = None):
        """Clear halt signal for a field"""
        if context:
            halt_key = f"{field}_{context.get('symbol', 'unknown')}"
            self.halt_signals.pop(halt_key, None)
        else:
            self.halt_signals.clear()
        
        logger.info(f"Halt signal cleared for {field}")
    
    async def retry_with_backoff(self, 
                               func: Callable, 
                               *args, 
                               retry_config: RetryConfig = None,
                               validation_func: Callable = None,
                               **kwargs) -> Any:
        """
        Retry function with exponential backoff and validation
        
        Args:
            func: Function to retry
            *args: Function arguments
            retry_config: Retry configuration
            validation_func: Optional validation function
            **kwargs: Function keyword arguments
        
        Returns:
            Function result
        """
        retry_config = retry_config or RetryConfig()
        last_error = None
        
        for attempt in range(retry_config.max_retries + 1):
            try:
                # Execute function
                result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
                
                # Validate result if validation function provided
                if validation_func:
                    is_valid, errors = await validation_func(result)
                    if not is_valid:
                        raise ValueError(f"Validation failed: {errors}")
                
                # Success
                if attempt > 0:
                    self.stats['successful_retries'] += 1
                    logger.info(f"Retry successful after {attempt} attempts")
                
                return result
                
            except Exception as e:
                last_error = e
                self.stats['retry_attempts'] += 1
                
                if attempt < retry_config.max_retries:
                    # Calculate delay
                    delay = min(
                        retry_config.base_delay * (retry_config.exponential_base ** attempt),
                        retry_config.max_delay
                    )
                    
                    # Add jitter
                    if retry_config.jitter:
                        delay *= (0.5 + np.random.random() * 0.5)
                    
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All retry attempts failed. Last error: {e}")
        
        # All retries failed
        raise last_error
    
    def get_stats(self) -> Dict[str, Any]:
        """Get resilience system statistics"""
        return {
            **self.stats,
            'active_alerts': len([a for a in self.active_alerts if not a.resolved]),
            'halt_signals_active': len(self.halt_signals),
            'validation_rules': len(self.validation_rules)
        }
    
    def get_active_alerts(self) -> List[MismatchAlert]:
        """Get active (unresolved) alerts"""
        return [alert for alert in self.active_alerts if not alert.resolved]
    
    def resolve_alert(self, alert_id: str):
        """Resolve an alert"""
        for alert in self.active_alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                self._save_alerts()
                logger.info(f"Alert {alert_id} resolved")
                break
    
    def clear_resolved_alerts(self):
        """Clear resolved alerts"""
        self.active_alerts = [alert for alert in self.active_alerts if not alert.resolved]
        self._save_alerts()
        logger.info("Cleared resolved alerts")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check of resilience system"""
        stats = self.get_stats()
        active_alerts = self.get_active_alerts()
        
        # Calculate health score
        total_validations = stats['total_validations']
        if total_validations > 0:
            failure_rate = stats['failed_validations'] / total_validations
            health_score = max(0, 1.0 - failure_rate - (len(active_alerts) * 0.1))
        else:
            health_score = 1.0
        
        # Determine health status
        if health_score >= 0.9:
            status = "healthy"
        elif health_score >= 0.7:
            status = "warning"
        else:
            status = "critical"
        
        return {
            "status": status,
            "health_score": round(health_score, 3),
            "stats": stats,
            "active_alerts": len(active_alerts),
            "halt_signals": len(self.halt_signals),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
