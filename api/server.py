"""
FastAPI Server
Exposes MCP-style endpoints with JWT authentication and rate limiting.
Works with features from both Yahoo Finance and Alpha Vantage data.
"""

import os
import time
import psutil
import sys
from datetime import datetime, timedelta, timezone
from typing import Optional
from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from jose import JWTError, jwt
from pydantic import BaseModel
import logging
from collections import defaultdict
from pathlib import Path
import json

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "core"))

from core.mcp_adapter import (
    MCPAdapter, 
    PredictRequest, 
    ScanAllRequest, 
    AnalyzeRequest,
    ConfirmRequest,
    MCPResponse,
    FeedbackRequest,
    TrainRLRequest,
    FetchDataRequest
)
from core.bhiv_core import BHIVCore, RequestType
from core.resilience_system import ResilienceSystem, ValidationLevel
from core.enhanced_features import EnhancedFeaturePipeline
# from core.models.baseline_lightgbm import BaselineLightGBM  # Removed - using enhanced model only
from core.models.rl_agent import LinUCBAgent, ThompsonSamplingAgent, DQNAgent

# ------------------- Logging -------------------
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/api_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ------------------- JWT Config -------------------
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "change_this_secret_key_in_production")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

if JWT_SECRET_KEY == "change_this_secret_key_in_production":
    logger.warning("⚠️  Using default JWT secret key! Set JWT_SECRET_KEY in .env for production")

security = HTTPBearer()

# ------------------- Rate Limit -------------------
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "3600"))
rate_limit_store = defaultdict(list)

# ------------------- FastAPI -------------------
app = FastAPI(
    title="Prediction Agent API",
    description="RL-powered prediction agent with MCP-style endpoints. Supports Yahoo Finance and Alpha Vantage data sources.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ------------------- Models -------------------
class TokenRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    uptime_seconds: float
    system: dict
    models_loaded: bool
    feature_store_info: dict


# ------------------- Global State -------------------
class AppState:
    def __init__(self):
        self.mcp_adapter: Optional[MCPAdapter] = None
        self.bhiv_core: Optional[BHIVCore] = None
        self.resilience_system: Optional[ResilienceSystem] = None
        self.start_time = time.time()
        self.request_count = 0
        self.feature_pipeline: Optional[EnhancedFeaturePipeline] = None
        # self.baseline_model: Optional[BaselineLightGBM] = None  # Removed - using enhanced model only
        self.enhanced_model = None
        self.agent = None


app_state = AppState()


# ------------------- Auth -------------------
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=JWT_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """Verify JWT token"""
    try:
        if not credentials or not credentials.credentials:
            logger.error("No credentials provided")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Not authenticated"
            )
            
        payload = jwt.decode(
            credentials.credentials, 
            JWT_SECRET_KEY, 
            algorithms=[JWT_ALGORITHM]
        )
        logger.debug(f"Token verified for user: {payload.get('sub', 'unknown')}")
        return payload
    except JWTError as e:
        logger.error(f"Token verification failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )


# ------------------- Enhanced Rate Limiting -------------------
from core.rate_limiter import get_rate_limiter, RateLimitConfig

# Initialize enhanced rate limiter
rate_limiter = get_rate_limiter()

def check_rate_limit(request: Request, user_id: str = None, endpoint: str = None):
    """Enhanced rate limiting with IP and user-based controls"""
    client_ip = request.client.host
    
    # Check rate limit
    allowed, details = rate_limiter.check_rate_limit(
        client_ip=client_ip,
        user_id=user_id,
        endpoint=endpoint
    )
    
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=details.get('error', 'Rate limit exceeded'),
            headers={
                'X-RateLimit-Limit': str(details.get('limit', RATE_LIMIT_REQUESTS)),
                'X-RateLimit-Remaining': str(details.get('remaining_requests', 0)),
                'X-RateLimit-Reset': str(int(details.get('reset_time', time.time() + RATE_LIMIT_WINDOW)))
            }
        )
    
    return details


# ------------------- Periodic Cleanup -------------------
def cleanup_rate_limit_store():
    """Periodically clean up rate limit store"""
    now = time.time()
    for ip in list(rate_limit_store.keys()):
        rate_limit_store[ip] = [
            t for t in rate_limit_store[ip] 
            if now - t < RATE_LIMIT_WINDOW
        ]
        if not rate_limit_store[ip]:
            del rate_limit_store[ip]


# ------------------- Startup -------------------
@app.on_event("startup")
async def startup_event():
    """Initialize models and services on startup"""
    try:
        logger.info("="*60)
        logger.info("Starting Prediction Agent API...")
        logger.info("="*60)

        # Create directories
        for directory in ["logs", "models", "data/cache", "data/features"]:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"✓ Directory ready: {directory}")

        # Initialize feature pipeline
        logger.info("\n[1/4] Initializing feature pipeline...")
        app_state.feature_pipeline = EnhancedFeaturePipeline(feature_store_dir="./data/features")
        
        # Load feature store to determine n_features
        try:
            feature_dict = app_state.feature_pipeline.load_feature_store()
            
            # Get feature columns from first symbol
            sample_df = next(iter(feature_dict.values()))
            exclude_cols = [
                'open', 'high', 'low', 'close', 'volume', 'adj_close',
                'symbol', 'source', 'fetch_timestamp', 'date', 'Date',
                'target', 'target_return', 'target_direction', 'target_binary'
            ]
            feature_cols = [col for col in sample_df.columns if col not in exclude_cols]
            n_features = len(feature_cols)
            n_symbols = len(feature_dict)
            
            logger.info(f"✓ Feature store loaded: {n_symbols} symbols, {n_features} features")
            logger.info(f"Feature columns: {feature_cols[:10]}...")  # Show first 10 features
            
        except FileNotFoundError:
            logger.warning("⚠️  Feature store not found, using default n_features=50")
            n_features = 50
            n_symbols = 0
        except Exception as e:
            logger.error(f"Error loading feature store: {e}")
            n_features = 50
            n_symbols = 0

        # Load enhanced model
        logger.info("\n[2/4] Loading enhanced model...")
        from core.models.enhanced_lightgbm import EnhancedLightGBM
        app_state.enhanced_model = EnhancedLightGBM(
            model_dir="./models",
            task="classification",
            model_name="enhanced-lightgbm-v2"
        )
        try:
            app_state.enhanced_model.load()
            logger.info(f"✓ Enhanced model loaded: {app_state.enhanced_model.model_name}")
        except FileNotFoundError:
            logger.warning("⚠️  Enhanced model not found. Will use RL agent only.")
        except Exception as e:
            logger.error(f"Error loading enhanced model: {e}")

        # Initialize RL agent
        logger.info("\n[3/5] Initializing RL agent...")
        agent_type = os.getenv("RL_AGENT_TYPE", "linucb").lower()
        logger.info(f"Agent type: {agent_type}")
        
        if agent_type == "linucb":
            app_state.agent = LinUCBAgent(n_features=n_features, alpha=1.0)
            model_file = "linucb_agent.pkl"
        elif agent_type == "thompson":
            app_state.agent = ThompsonSamplingAgent(n_features=n_features)
            model_file = "thompson_agent.pkl"
        elif agent_type == "dqn":
            device = "cuda" if os.getenv("GPU_DEVICE", "cpu") != "cpu" else "cpu"
            app_state.agent = DQNAgent(
                state_dim=n_features,
                action_dim=3,
                device=device
            )
            model_file = "dqn_agent.pt"
        else:
            logger.error(f"Unknown agent type: {agent_type}")
            raise ValueError(f"Invalid RL_AGENT_TYPE: {agent_type}")
        
        # Load agent if exists
        try:
            app_state.agent.load(model_file)
            logger.info(f"✓ RL agent loaded: {agent_type}")
        except FileNotFoundError:
            logger.warning(f"⚠️  RL agent not found. Using untrained {agent_type} agent.")
        except Exception as e:
            logger.error(f"Error loading RL agent: {e}")

        # Initialize BHIV Core
        logger.info("\n[4/7] Initializing BHIV Core...")
        app_state.bhiv_core = BHIVCore(
            core_dir="./data/core",
            bucket_dir="./data/bucket"
        )
        logger.info("✓ BHIV Core initialized")
        
        # Initialize Resilience System
        logger.info("\n[5/7] Initializing Resilience System...")
        app_state.resilience_system = ResilienceSystem(
            validation_level=ValidationLevel.MODERATE,
            mismatch_threshold=0.03  # 3% threshold as specified
        )
        logger.info("✓ Resilience System initialized")
        
        # Initialize MCP Adapter
        logger.info("\n[6/7] Initializing MCP Adapter...")
        app_state.mcp_adapter = MCPAdapter(
            app_state.agent,
            None,  # No baseline model - using enhanced model only
            app_state.feature_pipeline,
            enhanced_model=app_state.enhanced_model
        )
        logger.info("✓ MCP Adapter initialized")
        
        # Register Core handlers
        logger.info("\n[7/7] Registering Core handlers...")
        app_state.bhiv_core.register_handler(RequestType.PREDICT, app_state.mcp_adapter.predict)
        app_state.bhiv_core.register_handler(RequestType.ANALYZE, app_state.mcp_adapter.analyze)
        app_state.bhiv_core.register_handler(RequestType.SCAN_ALL, app_state.mcp_adapter.scan_all)
        app_state.bhiv_core.register_handler(RequestType.CONFIRM, app_state.mcp_adapter.confirm)
        app_state.bhiv_core.register_handler(RequestType.FEEDBACK, app_state.mcp_adapter.feedback)
        app_state.bhiv_core.register_handler(RequestType.TRAIN_RL, app_state.mcp_adapter.train_rl)
        app_state.bhiv_core.register_handler(RequestType.FETCH_DATA, app_state.mcp_adapter.fetch_data)
        logger.info("✓ Core handlers registered")
        
        # Final status
        logger.info("\n[8/8] Final status...")
        logger.info("\n" + "="*60)
        logger.info("✓ Prediction Agent API started successfully!")
        logger.info("="*60)
        logger.info(f"Feature store: {n_symbols} symbols")
        logger.info(f"RL agent: {agent_type}")
        logger.info(f"Enhanced model: {'loaded' if app_state.enhanced_model.is_trained else 'not loaded'}")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Startup failed: {e}", exc_info=True)
        logger.error("\n⚠️  Server started but models not loaded properly")
        logger.error("Run these commands to set up:")
        logger.error("  1. python fetch_more_data.py")
        logger.error("  2. python core/features.py")
        logger.error("  3. python core/models/enhanced_lightgbm.py")


# ------------------- Shutdown -------------------
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Prediction Agent API...")
    cleanup_rate_limit_store()


# ------------------- Endpoints -------------------
@app.post("/auth/token", response_model=TokenResponse, tags=["Authentication"])
async def login(request: TokenRequest):
    """
    Get JWT access token.
    
    For demo purposes, accepts any username/password.
    In production, implement proper authentication.
    """
    if not request.username or not request.password:
        raise HTTPException(
            status_code=400,
            detail="Username and password required"
        )
    
    # TODO: Implement actual authentication
    # For now, accept any credentials
    token = create_access_token({"sub": request.username})
    
    return TokenResponse(
        access_token=token,
        expires_in=JWT_EXPIRE_MINUTES * 60
    )


@app.get("/tools/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Check system health and resource usage.
    
    Returns:
        System status, uptime, resource usage, and model status
    """
    try:
        # System metrics
        memory = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=0.1)
        
        # GPU info (if available)
        gpu_info = {"available": False}
        try:
            import torch
            if torch.cuda.is_available():
                gpu_info = {
                    "available": True,
                    "device_name": torch.cuda.get_device_name(0),
                    "memory_allocated_mb": torch.cuda.memory_allocated(0) / 1024**2,
                    "memory_reserved_mb": torch.cuda.memory_reserved(0) / 1024**2
                }
        except:
            pass
        
        # Feature store info
        feature_store_info = {"symbols": 0, "features": 0, "loaded": False}
        try:
            if app_state.feature_pipeline:
                feature_dict = app_state.feature_pipeline.load_feature_store()
                sample_df = next(iter(feature_dict.values()))
                exclude_cols = [
                    'open', 'high', 'low', 'close', 'volume', 'adj_close',
                    'symbol', 'source', 'fetch_timestamp', 'date', 'Date',
                    'target', 'target_return', 'target_direction', 'target_binary'
                ]
                n_features = len([c for c in sample_df.columns if c not in exclude_cols])
                feature_store_info = {
                    "symbols": len(feature_dict),
                    "features": n_features,
                    "loaded": True
                }
        except:
            pass
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            uptime_seconds=time.time() - app_state.start_time,
            system={
                "cpu_percent": cpu,
                "memory_percent": memory.percent,
                "memory_available_mb": memory.available / 1024**2,
                "gpu": gpu_info,
                "request_count": app_state.request_count
            },
            models_loaded=app_state.mcp_adapter is not None,
            feature_store_info=feature_store_info
        )
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return HealthResponse(
            status="degraded",
            timestamp=datetime.now().isoformat(),
            uptime_seconds=time.time() - app_state.start_time,
            system={"error": str(e)},
            models_loaded=False,
            feature_store_info={"loaded": False}
        )


@app.post("/prediction_agent/tools/predict", response_model=MCPResponse, tags=["Prediction"])
async def predict(
    request: PredictRequest,
    token: dict = Depends(verify_token),
    rate_limit: None = Depends(check_rate_limit)
):
    """
    Predict for specific symbols.
    
    Args:
        symbols: List of ticker symbols
        horizon: Trading horizon (intraday, daily, weekly, monthly)
        risk_profile: Risk parameters (optional)
    
    Returns:
        Predictions with scores, confidence, and suggested actions
    """
    if not app_state.mcp_adapter:
        raise HTTPException(
            status_code=503,
            detail="Service not initialized. Models not loaded."
        )
    
    app_state.request_count += 1
    
    try:
        response = app_state.mcp_adapter.predict(request)
        return response
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/prediction_agent/tools/scan_all", response_model=MCPResponse, tags=["Prediction"])
async def scan_all(
    request: ScanAllRequest,
    token: dict = Depends(verify_token),
    rate_limit: None = Depends(check_rate_limit)
):
    """
    Scan all symbols in feature store and return top-ranked.
    
    Args:
        horizon: Trading horizon
        risk_profile: Risk parameters (optional)
        top_k: Number of top symbols to return
        min_score: Minimum score threshold
    
    Returns:
        Top-ranked symbols with scores and predicted actions
    """
    if not app_state.mcp_adapter:
        raise HTTPException(
            status_code=503,
            detail="Service not initialized. Models not loaded."
        )
    
    app_state.request_count += 1
    
    try:
        response = app_state.mcp_adapter.scan_all(request)
        return response
    except Exception as e:
        logger.error(f"Scan all error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/prediction_agent/tools/analyze", response_model=MCPResponse, tags=["Analysis"])
async def analyze(
    request: AnalyzeRequest,
    token: dict = Depends(verify_token),
    rate_limit: None = Depends(check_rate_limit)
):
    """
    Analyze specific symbols with detailed technical indicators.
    
    Args:
        symbols: List of ticker symbols (max 10)
        horizon: Trading horizon
        detailed: Return detailed analysis
    
    Returns:
        Technical analysis with indicators and RL predictions
    """
    if not app_state.mcp_adapter:
        raise HTTPException(
            status_code=503,
            detail="Service not initialized. Models not loaded."
        )
    
    app_state.request_count += 1
    
    try:
        response = app_state.mcp_adapter.analyze(request)
        return response
    except Exception as e:
        logger.error(f"Analyze error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/prediction_agent/tools/feedback", response_model=MCPResponse, tags=["Training"])
async def feedback(
    request: FeedbackRequest,
    token: dict = Depends(verify_token),
    rate_limit: None = Depends(check_rate_limit)
):
    if not app_state.mcp_adapter:
        raise HTTPException(status_code=503, detail="Service not initialized. Models not loaded.")
    try:
        response = app_state.mcp_adapter.feedback(request)
        return response
    except Exception as e:
        logger.error(f"Feedback error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/prediction_agent/tools/train_rl", response_model=MCPResponse, tags=["Training"])
async def train_rl(
    request: TrainRLRequest,
    token: dict = Depends(verify_token),
    rate_limit: None = Depends(check_rate_limit)
):
    if not app_state.mcp_adapter:
        raise HTTPException(status_code=503, detail="Service not initialized. Models not loaded.")
    try:
        response = app_state.mcp_adapter.train_rl(request)
        return response
    except Exception as e:
        logger.error(f"Train RL error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/prediction_agent/tools/fetch_data", response_model=MCPResponse, tags=["Data"])
async def fetch_data(
    request: FetchDataRequest,
    token: dict = Depends(verify_token),
    rate_limit: None = Depends(check_rate_limit)
):
    if not app_state.mcp_adapter:
        raise HTTPException(status_code=503, detail="Service not initialized. Models not loaded.")
    try:
        response = app_state.mcp_adapter.fetch_data(request)
        return response
    except Exception as e:
        logger.error(f"Fetch data error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/prediction_agent/tools/confirm", response_model=MCPResponse, tags=["Prediction"])
async def confirm(
    request: ConfirmRequest,
    token: dict = Depends(verify_token),
    rate_limit: None = Depends(check_rate_limit)
):
    """
    Confirm, reject, or modify a previous request.
    
    Args:
        request_id: ID of the request to confirm
        confirmation_type: Type of confirmation (execute, reject, modify)
        modifications: Modifications if type is modify
        user_notes: Optional user notes
    
    Returns:
        Confirmation response with status
    """
    if not app_state.mcp_adapter:
        raise HTTPException(
            status_code=503,
            detail="Service not initialized. Models not loaded."
        )
    
    app_state.request_count += 1
    
    try:
        response = app_state.mcp_adapter.confirm(request)
        return response
    except Exception as e:
        logger.error(f"Confirm error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/data/dashboard", tags=["Frontend"])
async def get_dashboard_data(
    token: dict = Depends(verify_token)
):
    """
    Get dashboard data for frontend integration.
    
    Returns:
        Dashboard data with stats, recent predictions, and trades
    """
    if not app_state.bhiv_core:
        raise HTTPException(
            status_code=503,
            detail="Core system not initialized"
        )
    
    try:
        dashboard_data = await app_state.bhiv_core.get_dashboard_data()
        return dashboard_data
    except Exception as e:
        logger.error(f"Dashboard data error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/feed/live", tags=["Frontend"])
async def get_live_feed(
    symbol: Optional[str] = None,
    limit: int = 100,
    token: dict = Depends(verify_token)
):
    """
    Get live feed data for frontend integration.
    
    Args:
        symbol: Optional symbol filter
        limit: Maximum number of entries to return
    
    Returns:
        Live feed data with recent predictions and trends
    """
    if not app_state.bhiv_core:
        raise HTTPException(
            status_code=503,
            detail="Core system not initialized"
        )
    
    try:
        live_feed = await app_state.bhiv_core.get_live_feed(symbol=symbol, limit=limit)
        return live_feed
    except Exception as e:
        logger.error(f"Live feed error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/resilience/health", tags=["System"])
async def get_resilience_health(
    token: dict = Depends(verify_token)
):
    """
    Get resilience system health status.
    
    Returns:
        Resilience system health and statistics
    """
    if not app_state.resilience_system:
        raise HTTPException(
            status_code=503,
            detail="Resilience system not initialized"
        )
    
    try:
        health_data = await app_state.resilience_system.health_check()
        return health_data
    except Exception as e:
        logger.error(f"Resilience health check error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/resilience/alerts", tags=["System"])
async def get_resilience_alerts(
    token: dict = Depends(verify_token)
):
    """
    Get active resilience alerts.
    
    Returns:
        List of active mismatch alerts
    """
    if not app_state.resilience_system:
        raise HTTPException(
            status_code=503,
            detail="Resilience system not initialized"
        )
    
    try:
        alerts = app_state.resilience_system.get_active_alerts()
        return {
            "success": True,
            "data": [
                {
                    "alert_id": alert.alert_id,
                    "timestamp": alert.timestamp.isoformat(),
                    "severity": alert.severity.value,
                    "field": alert.field,
                    "expected": alert.expected,
                    "actual": alert.actual,
                    "difference": alert.difference,
                    "threshold": alert.threshold,
                    "context": alert.context
                }
                for alert in alerts
            ],
            "count": len(alerts)
        }
    except Exception as e:
        logger.error(f"Resilience alerts error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/resilience/alerts/{alert_id}/resolve", tags=["System"])
async def resolve_alert(
    alert_id: str,
    token: dict = Depends(verify_token)
):
    """
    Resolve a specific alert.
    
    Args:
        alert_id: ID of the alert to resolve
    
    Returns:
        Success status
    """
    if not app_state.resilience_system:
        raise HTTPException(
            status_code=503,
            detail="Resilience system not initialized"
        )
    
    try:
        app_state.resilience_system.resolve_alert(alert_id)
        return {"success": True, "message": f"Alert {alert_id} resolved"}
    except Exception as e:
        logger.error(f"Resolve alert error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/logs/requests", tags=["System"])
async def get_request_logs(
    limit: int = 100,
    token: dict = Depends(verify_token)
):
    """
    Get recent API request logs.
    
    Args:
        limit: Maximum number of logs to return
    
    Returns:
        Recent request logs
    """
    if not app_state.mcp_adapter:
        raise HTTPException(
            status_code=503,
            detail="Service not initialized"
        )
    
    logs = app_state.mcp_adapter.get_request_logs(limit=limit)
    
    return {
        "success": True,
        "data": logs,
        "metadata": {"count": len(logs)}
    }


@app.get("/tools/rate-limit-status", tags=["System"])
async def get_rate_limit_status(
    request: Request,
    token: dict = Depends(verify_token)
):
    """
    Get current rate limiting status for the client.
    
    Returns:
        Rate limiting status and statistics
    """
    client_ip = request.client.host
    user_id = token.get('sub', 'anonymous')
    
    # Get rate limit status
    status = rate_limiter.get_rate_limit_status(client_ip, user_id)
    global_stats = rate_limiter.get_global_stats()
    
    return {
        "success": True,
        "data": {
            "client_status": status,
            "global_stats": global_stats,
            "client_ip": client_ip,
            "user_id": user_id
        }
    }


@app.post("/tools/reset-rate-limits", tags=["System"])
async def reset_rate_limits(
    identifier: Optional[str] = None,
    scope: str = "all",
    token: dict = Depends(verify_token)
):
    """
    Reset rate limits for specific identifier or all.
    
    Args:
        identifier: IP address or user ID to reset
        scope: Reset scope (all, ip, user)
    
    Returns:
        Reset operation result
    """
    result = rate_limiter.reset_rate_limits(identifier, scope)
    
    return {
        "success": True,
        "data": result
    }


@app.get("/", tags=["System"])
async def root():
    """API root endpoint"""
    return {
        "name": "Prediction Agent API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/tools/health"
    }


# ------------------- Error Handlers -------------------
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """Handle 404 errors"""
    return JSONResponse(
        status_code=404,
        content={"detail": "Endpoint not found. Visit /docs for API documentation."}
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    """Handle 500 errors"""
    logger.error(f"Internal error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Check logs for details."}
    )


# ------------------- Run Server -------------------
if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    
    print("\n" + "="*60)
    print("STARTING PREDICTION AGENT API")
    print("="*60)
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"Docs: http://localhost:{port}/docs")
    print("="*60 + "\n")
    
    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )