"""
Enhanced Rate Limiting System
Implements IP-based and user-based rate limiting with different scopes
"""

import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, deque
from dataclasses import dataclass
import asyncio
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Rate limiting configuration"""
    requests_per_window: int = 100
    window_seconds: int = 3600  # 1 hour
    burst_limit: int = 20  # Burst requests allowed
    burst_window: int = 60  # Burst window in seconds
    user_based: bool = False  # Whether to use user-based limiting
    ip_based: bool = True  # Whether to use IP-based limiting
    strict_mode: bool = False  # Strict enforcement vs warning mode


@dataclass
class RateLimitEntry:
    """Individual rate limit entry"""
    timestamp: float
    user_id: Optional[str] = None
    endpoint: Optional[str] = None
    request_size: int = 1


class RateLimiter:
    """
    Enhanced rate limiter with multiple scopes and strategies
    """
    
    def __init__(self, config: RateLimitConfig = None):
        self.config = config or RateLimitConfig()
        
        # Rate limit stores
        self.ip_requests: Dict[str, deque] = defaultdict(lambda: deque())
        self.user_requests: Dict[str, deque] = defaultdict(lambda: deque())
        self.endpoint_requests: Dict[str, deque] = defaultdict(lambda: deque())
        
        # Burst protection
        self.burst_requests: Dict[str, deque] = defaultdict(lambda: deque())
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'blocked_requests': 0,
            'rate_limited_ips': set(),
            'rate_limited_users': set(),
            'start_time': time.time()
        }
        
        # Load existing data
        self._load_rate_limit_data()
        
        logger.info(f"Rate limiter initialized: "
                   f"IP-based={self.config.ip_based}, "
                   f"User-based={self.config.user_based}")
    
    def _load_rate_limit_data(self):
        """Load existing rate limit data"""
        data_file = Path("logs/rate_limit_data.json")
        if data_file.exists():
            try:
                with open(data_file, 'r') as f:
                    data = json.load(f)
                    self.stats.update(data.get('stats', self.stats))
                    logger.info("Loaded existing rate limit data")
            except Exception as e:
                logger.error(f"Failed to load rate limit data: {e}")
    
    def _save_rate_limit_data(self):
        """Save rate limit data"""
        data_file = Path("logs/rate_limit_data.json")
        data_file.parent.mkdir(exist_ok=True)
        
        try:
            data = {
                'stats': self.stats,
                'config': {
                    'requests_per_window': self.config.requests_per_window,
                    'window_seconds': self.config.window_seconds,
                    'burst_limit': self.config.burst_limit,
                    'burst_window': self.config.burst_window,
                    'user_based': self.config.user_based,
                    'ip_based': self.config.ip_based,
                    'strict_mode': self.config.strict_mode
                },
                'last_updated': datetime.now().isoformat()
            }
            with open(data_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save rate limit data: {e}")
    
    def _clean_old_entries(self, request_deque: deque, window_seconds: int):
        """Clean old entries from request deque"""
        current_time = time.time()
        cutoff_time = current_time - window_seconds
        
        # Remove old entries
        while request_deque and request_deque[0].timestamp < cutoff_time:
            request_deque.popleft()
    
    def _check_burst_limit(self, identifier: str) -> bool:
        """Check burst limit for identifier"""
        current_time = time.time()
        burst_deque = self.burst_requests[identifier]
        
        # Clean old burst entries
        self._clean_old_entries(burst_deque, self.config.burst_window)
        
        # Check burst limit
        if len(burst_deque) >= self.config.burst_limit:
            return False
        
        # Add current request
        burst_deque.append(RateLimitEntry(timestamp=current_time))
        return True
    
    def _check_rate_limit(self, identifier: str, request_deque: deque, 
                         window_seconds: int, limit: int) -> Tuple[bool, Dict[str, Any]]:
        """Check rate limit for identifier"""
        current_time = time.time()
        
        # Clean old entries
        self._clean_old_entries(request_deque, window_seconds)
        
        # Check limit
        if len(request_deque) >= limit:
            return False, {
                'limit_exceeded': True,
                'current_requests': len(request_deque),
                'limit': limit,
                'window_seconds': window_seconds,
                'reset_time': current_time + window_seconds
            }
        
        # Add current request
        request_deque.append(RateLimitEntry(timestamp=current_time))
        return True, {
            'limit_exceeded': False,
            'current_requests': len(request_deque),
            'limit': limit,
            'remaining_requests': limit - len(request_deque)
        }
    
    def check_rate_limit(self, client_ip: str, user_id: Optional[str] = None,
                        endpoint: Optional[str] = None) -> Tuple[bool, Dict[str, Any]]:
        """
        Check rate limit for request
        
        Args:
            client_ip: Client IP address
            user_id: User ID (if user-based limiting)
            endpoint: API endpoint (for endpoint-specific limiting)
        
        Returns:
            (allowed, details)
        """
        self.stats['total_requests'] += 1
        
        # Check burst limit first
        burst_identifier = f"{client_ip}:{user_id or 'anonymous'}"
        if not self._check_burst_limit(burst_identifier):
            self.stats['blocked_requests'] += 1
            self.stats['rate_limited_ips'].add(client_ip)
            if user_id:
                self.stats['rate_limited_users'].add(user_id)
            
            logger.warning(f"Burst limit exceeded for {burst_identifier}")
            return False, {
                'error': 'Burst limit exceeded',
                'burst_limit': self.config.burst_limit,
                'burst_window': self.config.burst_window,
                'identifier': burst_identifier
            }
        
        # Check IP-based rate limit
        if self.config.ip_based:
            ip_allowed, ip_details = self._check_rate_limit(
                client_ip, 
                self.ip_requests[client_ip],
                self.config.window_seconds,
                self.config.requests_per_window
            )
            
            if not ip_allowed:
                self.stats['blocked_requests'] += 1
                self.stats['rate_limited_ips'].add(client_ip)
                
                logger.warning(f"IP rate limit exceeded for {client_ip}")
                return False, {
                    'error': 'IP rate limit exceeded',
                    'scope': 'ip',
                    'identifier': client_ip,
                    **ip_details
                }
        
        # Check user-based rate limit
        if self.config.user_based and user_id:
            user_allowed, user_details = self._check_rate_limit(
                user_id,
                self.user_requests[user_id],
                self.config.window_seconds,
                self.config.requests_per_window
            )
            
            if not user_allowed:
                self.stats['blocked_requests'] += 1
                self.stats['rate_limited_users'].add(user_id)
                
                logger.warning(f"User rate limit exceeded for {user_id}")
                return False, {
                    'error': 'User rate limit exceeded',
                    'scope': 'user',
                    'identifier': user_id,
                    **user_details
                }
        
        # Check endpoint-specific rate limit (if configured)
        if endpoint:
            endpoint_allowed, endpoint_details = self._check_rate_limit(
                endpoint,
                self.endpoint_requests[endpoint],
                self.config.window_seconds,
                self.config.requests_per_window // 2  # Half limit for specific endpoints
            )
            
            if not endpoint_allowed:
                self.stats['blocked_requests'] += 1
                
                logger.warning(f"Endpoint rate limit exceeded for {endpoint}")
                return False, {
                    'error': 'Endpoint rate limit exceeded',
                    'scope': 'endpoint',
                    'identifier': endpoint,
                    **endpoint_details
                }
        
        # All checks passed
        return True, {
            'allowed': True,
            'ip': client_ip,
            'user': user_id,
            'endpoint': endpoint,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_rate_limit_status(self, client_ip: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get current rate limit status for client"""
        current_time = time.time()
        
        # Clean old entries
        self._clean_old_entries(self.ip_requests[client_ip], self.config.window_seconds)
        if user_id:
            self._clean_old_entries(self.user_requests[user_id], self.config.window_seconds)
        
        status = {
            'ip': {
                'requests': len(self.ip_requests[client_ip]),
                'limit': self.config.requests_per_window,
                'remaining': self.config.requests_per_window - len(self.ip_requests[client_ip]),
                'reset_time': current_time + self.config.window_seconds
            }
        }
        
        if user_id:
            status['user'] = {
                'requests': len(self.user_requests[user_id]),
                'limit': self.config.requests_per_window,
                'remaining': self.config.requests_per_window - len(self.user_requests[user_id]),
                'reset_time': current_time + self.config.window_seconds
            }
        
        return status
    
    def get_global_stats(self) -> Dict[str, Any]:
        """Get global rate limiting statistics"""
        current_time = time.time()
        uptime = current_time - self.stats['start_time']
        
        return {
            'total_requests': self.stats['total_requests'],
            'blocked_requests': self.stats['blocked_requests'],
            'block_rate': self.stats['blocked_requests'] / max(1, self.stats['total_requests']),
            'uptime_seconds': uptime,
            'requests_per_second': self.stats['total_requests'] / max(1, uptime),
            'rate_limited_ips': len(self.stats['rate_limited_ips']),
            'rate_limited_users': len(self.stats['rate_limited_users']),
            'config': {
                'requests_per_window': self.config.requests_per_window,
                'window_seconds': self.config.window_seconds,
                'burst_limit': self.config.burst_limit,
                'burst_window': self.config.burst_window,
                'user_based': self.config.user_based,
                'ip_based': self.config.ip_based,
                'strict_mode': self.config.strict_mode
            }
        }
    
    def reset_rate_limits(self, identifier: Optional[str] = None, 
                         scope: str = 'all') -> Dict[str, Any]:
        """Reset rate limits for identifier or all"""
        if identifier and scope == 'ip':
            self.ip_requests[identifier].clear()
            self.burst_requests[identifier].clear()
            self.stats['rate_limited_ips'].discard(identifier)
            logger.info(f"Reset rate limits for IP: {identifier}")
            
        elif identifier and scope == 'user':
            self.user_requests[identifier].clear()
            self.burst_requests[identifier].clear()
            self.stats['rate_limited_users'].discard(identifier)
            logger.info(f"Reset rate limits for user: {identifier}")
            
        elif scope == 'all':
            self.ip_requests.clear()
            self.user_requests.clear()
            self.endpoint_requests.clear()
            self.burst_requests.clear()
            self.stats['rate_limited_ips'].clear()
            self.stats['rate_limited_users'].clear()
            logger.info("Reset all rate limits")
        
        return {"success": True, "message": f"Rate limits reset for {scope}"}
    
    def update_config(self, new_config: RateLimitConfig) -> Dict[str, Any]:
        """Update rate limiting configuration"""
        old_config = self.config
        self.config = new_config
        
        logger.info(f"Updated rate limit config: "
                   f"requests={new_config.requests_per_window}, "
                   f"window={new_config.window_seconds}s, "
                   f"burst={new_config.burst_limit}")
        
        return {
            "success": True,
            "old_config": {
                'requests_per_window': old_config.requests_per_window,
                'window_seconds': old_config.window_seconds,
                'burst_limit': old_config.burst_limit
            },
            "new_config": {
                'requests_per_window': new_config.requests_per_window,
                'window_seconds': new_config.window_seconds,
                'burst_limit': new_config.burst_limit
            }
        }
    
    def cleanup_old_data(self, days_to_keep: int = 7):
        """Clean up old rate limit data"""
        cutoff_time = time.time() - (days_to_keep * 24 * 3600)
        
        # Clean old entries from all deques
        for ip in list(self.ip_requests.keys()):
            self._clean_old_entries(self.ip_requests[ip], self.config.window_seconds)
            if not self.ip_requests[ip]:
                del self.ip_requests[ip]
        
        for user in list(self.user_requests.keys()):
            self._clean_old_entries(self.user_requests[user], self.config.window_seconds)
            if not self.user_requests[user]:
                del self.user_requests[user]
        
        for endpoint in list(self.endpoint_requests.keys()):
            self._clean_old_entries(self.endpoint_requests[endpoint], self.config.window_seconds)
            if not self.endpoint_requests[endpoint]:
                del self.endpoint_requests[endpoint]
        
        # Clean burst requests
        for identifier in list(self.burst_requests.keys()):
            self._clean_old_entries(self.burst_requests[identifier], self.config.burst_window)
            if not self.burst_requests[identifier]:
                del self.burst_requests[identifier]
        
        logger.info(f"Cleaned up rate limit data older than {days_to_keep} days")
        
        return {
            "success": True,
            "message": f"Cleaned up data older than {days_to_keep} days"
        }


# Global rate limiter instance
_rate_limiter = None


def get_rate_limiter() -> RateLimiter:
    """Get global rate limiter instance"""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter


def initialize_rate_limiter(config: RateLimitConfig = None) -> RateLimiter:
    """Initialize rate limiter with configuration"""
    global _rate_limiter
    _rate_limiter = RateLimiter(config)
    return _rate_limiter
