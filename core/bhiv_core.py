"""
BHIV Core - Central orchestration and context management system
Handles request routing, context persistence, and integration with Bucket logging
"""

import json
import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import hashlib
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class RequestType(str, Enum):
    PREDICT = "predict"
    ANALYZE = "analyze"
    SCAN_ALL = "scan_all"
    CONFIRM = "confirm"
    FEEDBACK = "feedback"
    TRAIN_RL = "train_rl"
    FETCH_DATA = "fetch_data"


class RequestStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class RequestContext:
    """Request context for tracking and persistence"""
    request_id: str
    request_type: RequestType
    status: RequestStatus
    timestamp: datetime
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    parameters: Dict[str, Any] = None
    response: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time_ms: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


@dataclass
class BucketEntry:
    """Bucket entry for logging and persistence"""
    entry_id: str
    request_id: str
    timestamp: datetime
    data_type: str  # prediction, trade, feedback, error, etc.
    symbol: Optional[str] = None
    data: Dict[str, Any] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.data is None:
            self.data = {}
        if self.metadata is None:
            self.metadata = {}


class BHIVCore:
    """
    BHIV Core - Central orchestration system
    Manages request routing, context persistence, and Bucket logging
    """
    
    def __init__(self, 
                 core_dir: str = "data/core",
                 bucket_dir: str = "data/bucket",
                 max_context_age_hours: int = 24):
        self.core_dir = Path(core_dir)
        self.bucket_dir = Path(bucket_dir)
        self.max_context_age_hours = max_context_age_hours
        
        # Create directories
        self.core_dir.mkdir(parents=True, exist_ok=True)
        self.bucket_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory context store (for active requests)
        self.active_contexts: Dict[str, RequestContext] = {}
        
        # Request routing
        self.request_handlers: Dict[RequestType, callable] = {}
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_processing_time_ms': 0.0,
            'active_requests': 0,
            'bucket_entries': 0
        }
        
        # Load existing data
        self._load_core_data()
        
        logger.info(f"BHIV Core initialized: core_dir={self.core_dir}, bucket_dir={self.bucket_dir}")
    
    def _load_core_data(self):
        """Load existing core data"""
        try:
            stats_file = self.core_dir / "stats.json"
            if stats_file.exists():
                with open(stats_file, 'r') as f:
                    self.stats.update(json.load(f))
            
            # Load active contexts (if any)
            contexts_file = self.core_dir / "active_contexts.json"
            if contexts_file.exists():
                with open(contexts_file, 'r') as f:
                    contexts_data = json.load(f)
                    for req_id, ctx_data in contexts_data.items():
                        ctx_data['timestamp'] = datetime.fromisoformat(ctx_data['timestamp'])
                        ctx_data['request_type'] = RequestType(ctx_data['request_type'])
                        ctx_data['status'] = RequestStatus(ctx_data['status'])
                        self.active_contexts[req_id] = RequestContext(**ctx_data)
            
            logger.info(f"Loaded core data: {len(self.active_contexts)} active contexts")
        except Exception as e:
            logger.error(f"Failed to load core data: {e}")
    
    def _save_core_data(self):
        """Save core data to disk"""
        try:
            # Save stats
            stats_file = self.core_dir / "stats.json"
            with open(stats_file, 'w') as f:
                json.dump(self.stats, f, indent=2, default=str)
            
            # Save active contexts
            contexts_file = self.core_dir / "active_contexts.json"
            contexts_data = {}
            for req_id, ctx in self.active_contexts.items():
                ctx_dict = asdict(ctx)
                ctx_dict['timestamp'] = ctx.timestamp.isoformat()
                contexts_data[req_id] = ctx_dict
            
            with open(contexts_file, 'w') as f:
                json.dump(contexts_data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Failed to save core data: {e}")
    
    def _generate_request_id(self, request_type: RequestType, parameters: Dict[str, Any]) -> str:
        """Generate unique request ID"""
        # Create hash from parameters for deduplication
        param_hash = hashlib.md5(json.dumps(parameters, sort_keys=True).encode()).hexdigest()[:8]
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"{request_type.value}_{timestamp}_{param_hash}"
    
    def register_handler(self, request_type: RequestType, handler: callable):
        """Register request handler for specific type"""
        self.request_handlers[request_type] = handler
        logger.info(f"Registered handler for {request_type.value}")
    
    async def route_request(self, 
                          request_type: RequestType,
                          parameters: Dict[str, Any],
                          user_id: Optional[str] = None,
                          session_id: Optional[str] = None) -> str:
        """
        Route request through Core system
        
        Returns:
            request_id: Unique identifier for tracking
        """
        request_id = self._generate_request_id(request_type, parameters)
        
        # Create request context
        context = RequestContext(
            request_id=request_id,
            request_type=request_type,
            status=RequestStatus.PENDING,
            timestamp=datetime.now(timezone.utc),
            user_id=user_id,
            session_id=session_id,
            parameters=parameters
        )
        
        # Store context
        self.active_contexts[request_id] = context
        self.stats['total_requests'] += 1
        self.stats['active_requests'] += 1
        
        # Log to bucket
        await self._log_to_bucket(
            request_id=request_id,
            data_type="request_started",
            data={
                "request_type": request_type.value,
                "parameters": parameters,
                "user_id": user_id,
                "session_id": session_id
            }
        )
        
        # Process request asynchronously
        asyncio.create_task(self._process_request(request_id))
        
        logger.info(f"Routed request {request_id} of type {request_type.value}")
        return request_id
    
    async def _process_request(self, request_id: str):
        """Process request asynchronously"""
        context = self.active_contexts.get(request_id)
        if not context:
            logger.error(f"Context not found for request {request_id}")
            return
        
        try:
            context.status = RequestStatus.PROCESSING
            start_time = datetime.now(timezone.utc)
            
            # Get handler
            handler = self.request_handlers.get(context.request_type)
            if not handler:
                raise ValueError(f"No handler registered for {context.request_type.value}")
            
            # Execute handler
            response = await handler(context.parameters)
            
            # Update context
            end_time = datetime.now(timezone.utc)
            context.status = RequestStatus.COMPLETED
            context.response = response
            context.processing_time_ms = (end_time - start_time).total_seconds() * 1000
            
            # Update stats
            self.stats['successful_requests'] += 1
            self.stats['active_requests'] -= 1
            
            # Update average processing time
            if self.stats['successful_requests'] > 0:
                total_time = self.stats['avg_processing_time_ms'] * (self.stats['successful_requests'] - 1)
                self.stats['avg_processing_time_ms'] = (total_time + context.processing_time_ms) / self.stats['successful_requests']
            
            # Log to bucket
            await self._log_to_bucket(
                request_id=request_id,
                data_type="request_completed",
                data=response,
                metadata={
                    "processing_time_ms": context.processing_time_ms,
                    "status": "success"
                }
            )
            
            logger.info(f"Request {request_id} completed in {context.processing_time_ms:.2f}ms")
            
        except Exception as e:
            # Handle error
            context.status = RequestStatus.FAILED
            context.error = str(e)
            context.processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            # Update stats
            self.stats['failed_requests'] += 1
            self.stats['active_requests'] -= 1
            
            # Log to bucket
            await self._log_to_bucket(
                request_id=request_id,
                data_type="request_failed",
                data={"error": str(e)},
                metadata={
                    "processing_time_ms": context.processing_time_ms,
                    "status": "error"
                }
            )
            
            logger.error(f"Request {request_id} failed: {e}")
        
        finally:
            # Save context
            self._save_core_data()
    
    async def _log_to_bucket(self, 
                           request_id: str,
                           data_type: str,
                           data: Dict[str, Any],
                           symbol: Optional[str] = None,
                           metadata: Optional[Dict[str, Any]] = None):
        """Log entry to Bucket system"""
        try:
            entry = BucketEntry(
                entry_id=str(uuid.uuid4()),
                request_id=request_id,
                timestamp=datetime.now(timezone.utc),
                data_type=data_type,
                symbol=symbol,
                data=data,
                metadata=metadata or {}
            )
            
            # Save to bucket file
            bucket_file = self.bucket_dir / f"{data_type}_{datetime.now().strftime('%Y%m%d')}.jsonl"
            with open(bucket_file, 'a') as f:
                f.write(json.dumps(asdict(entry), default=str) + '\n')
            
            self.stats['bucket_entries'] += 1
            
        except Exception as e:
            logger.error(f"Failed to log to bucket: {e}")
    
    def get_request_status(self, request_id: str) -> Optional[RequestContext]:
        """Get request status and context"""
        return self.active_contexts.get(request_id)
    
    def get_active_requests(self) -> List[RequestContext]:
        """Get all active requests"""
        return list(self.active_contexts.values())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get core statistics"""
        return self.stats.copy()
    
    async def cleanup_old_contexts(self):
        """Clean up old contexts"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=self.max_context_age_hours)
        
        to_remove = []
        for req_id, context in self.active_contexts.items():
            if context.timestamp < cutoff_time:
                to_remove.append(req_id)
        
        for req_id in to_remove:
            del self.active_contexts[req_id]
        
        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} old contexts")
            self._save_core_data()
    
    async def get_bucket_entries(self, 
                               data_type: Optional[str] = None,
                               symbol: Optional[str] = None,
                               start_date: Optional[datetime] = None,
                               end_date: Optional[datetime] = None,
                               limit: int = 1000) -> List[BucketEntry]:
        """Get entries from bucket with filters"""
        entries = []
        
        try:
            # Get all bucket files
            bucket_files = list(self.bucket_dir.glob("*.jsonl"))
            
            for bucket_file in sorted(bucket_files, reverse=True):
                if len(entries) >= limit:
                    break
                
                with open(bucket_file, 'r') as f:
                    for line in f:
                        if len(entries) >= limit:
                            break
                        
                        try:
                            entry_data = json.loads(line.strip())
                            
                            # Apply filters
                            if data_type and entry_data.get('data_type') != data_type:
                                continue
                            if symbol and entry_data.get('symbol') != symbol:
                                continue
                            
                            # Convert timestamp
                            entry_data['timestamp'] = datetime.fromisoformat(entry_data['timestamp'])
                            
                            # Apply date filters
                            if start_date and entry_data['timestamp'] < start_date:
                                continue
                            if end_date and entry_data['timestamp'] > end_date:
                                continue
                            
                            entries.append(BucketEntry(**entry_data))
                            
                        except Exception as e:
                            logger.warning(f"Failed to parse bucket entry: {e}")
                            continue
            
        except Exception as e:
            logger.error(f"Failed to get bucket entries: {e}")
        
        return entries
    
    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Get dashboard data for frontend"""
        try:
            # Get recent predictions
            recent_predictions = await self.get_bucket_entries(
                data_type="prediction",
                limit=50
            )
            
            # Get recent trades
            recent_trades = await self.get_bucket_entries(
                data_type="trade",
                limit=20
            )
            
            # Get system stats
            stats = self.get_stats()
            
            # Calculate success rate
            total_requests = stats['total_requests']
            success_rate = (stats['successful_requests'] / total_requests * 100) if total_requests > 0 else 0
            
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "stats": {
                    "total_requests": total_requests,
                    "successful_requests": stats['successful_requests'],
                    "failed_requests": stats['failed_requests'],
                    "success_rate": round(success_rate, 2),
                    "avg_processing_time_ms": round(stats['avg_processing_time_ms'], 2),
                    "active_requests": stats['active_requests'],
                    "bucket_entries": stats['bucket_entries']
                },
                "recent_predictions": [
                    {
                        "symbol": entry.symbol,
                        "timestamp": entry.timestamp.isoformat(),
                        "data": entry.data
                    }
                    for entry in recent_predictions[:10]
                ],
                "recent_trades": [
                    {
                        "symbol": entry.symbol,
                        "timestamp": entry.timestamp.isoformat(),
                        "data": entry.data
                    }
                    for entry in recent_trades[:10]
                ]
            }
            
        except Exception as e:
            logger.error(f"Failed to get dashboard data: {e}")
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": str(e),
                "stats": {},
                "recent_predictions": [],
                "recent_trades": []
            }
    
    async def get_live_feed(self, symbol: Optional[str] = None, limit: int = 100) -> Dict[str, Any]:
        """Get live feed data for frontend"""
        try:
            # Get recent entries
            entries = await self.get_bucket_entries(
                data_type="prediction",
                symbol=symbol,
                limit=limit
            )
            
            # Group by symbol
            symbol_data = {}
            for entry in entries:
                if entry.symbol not in symbol_data:
                    symbol_data[entry.symbol] = {
                        "symbol": entry.symbol,
                        "latest_prediction": None,
                        "prediction_history": [],
                        "confidence_trend": [],
                        "action_trend": []
                    }
                
                symbol_data[entry.symbol]["prediction_history"].append({
                    "timestamp": entry.timestamp.isoformat(),
                    "data": entry.data
                })
                
                # Update latest prediction
                if not symbol_data[entry.symbol]["latest_prediction"] or \
                   entry.timestamp > datetime.fromisoformat(symbol_data[entry.symbol]["latest_prediction"]["timestamp"]):
                    symbol_data[entry.symbol]["latest_prediction"] = {
                        "timestamp": entry.timestamp.isoformat(),
                        "data": entry.data
                    }
            
            # Calculate trends
            for symbol, data in symbol_data.items():
                if len(data["prediction_history"]) > 1:
                    # Sort by timestamp
                    data["prediction_history"].sort(key=lambda x: x["timestamp"])
                    
                    # Extract confidence and action trends
                    for pred in data["prediction_history"][-10:]:  # Last 10 predictions
                        if "confidence" in pred["data"]:
                            data["confidence_trend"].append(pred["data"]["confidence"])
                        if "action" in pred["data"]:
                            data["action_trend"].append(pred["data"]["action"])
            
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "symbols": list(symbol_data.values()),
                "total_symbols": len(symbol_data)
            }
            
        except Exception as e:
            logger.error(f"Failed to get live feed: {e}")
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": str(e),
                "symbols": [],
                "total_symbols": 0
            }
