# MCP Integration Kit for Krishna Prediction Agent

## Overview

This document provides comprehensive integration guidelines for the Krishna Prediction Agent's Model Context Protocol (MCP) endpoints. The agent exposes RESTful APIs that follow MCP standards for seamless integration with external systems.

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Krishna      │    │   MCP Adapter   │    │   External     │
│   Prediction   │◄──►│   (FastAPI)     │◄──►│   Systems      │
│   Agent        │    │                 │    │   (Karan, n8n)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Authentication

All MCP endpoints require JWT authentication:

```bash
# Get token
curl -X POST "http://localhost:8000/auth/token" \
  -H "Content-Type: application/json" \
  -d '{"username": "demo", "password": "demo"}'

# Use token in requests
curl -H "Authorization: Bearer YOUR_TOKEN" \
  "http://localhost:8000/prediction_agent/tools/predict"
```

## MCP Endpoints

### 1. Health Check
**Endpoint:** `GET /tools/health`

```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
  "http://localhost:8000/tools/health"
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-18T10:00:00Z",
  "uptime_seconds": 3600,
  "system": {
    "cpu_percent": 15.2,
    "memory_percent": 45.8,
    "gpu": {"available": true}
  },
  "models_loaded": true,
  "feature_store_info": {
    "symbols": 150,
    "features": 50,
    "loaded": true
  }
}
```

### 2. Predict
**Endpoint:** `POST /prediction_agent/tools/predict`

Generate predictions for specific symbols.

```bash
curl -X POST "http://localhost:8000/prediction_agent/tools/predict" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": ["AAPL", "MSFT", "BTC-USD"],
    "horizon": "daily",
    "risk_profile": {
      "stop_loss_pct": 2.0,
      "capital_risk_pct": 1.5,
      "drawdown_limit_pct": 10.0
    }
  }'
```

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "symbol": "AAPL",
      "horizon": "daily",
      "predicted_price": 185.50,
      "confidence": 0.75,
      "score": 0.68,
      "action": "long",
      "risk_applied": {...},
      "reason": "Strong uptrend | RSI neutral at 55.2",
      "timestamp": "2025-01-18T10:00:00Z",
      "model_version": "ensemble-v2.1"
    }
  ],
  "metadata": {
    "total_predictions": 3,
    "requested_symbols": 3,
    "horizon": "daily"
  }
}
```

### 3. Scan All
**Endpoint:** `POST /prediction_agent/tools/scan_all`

Scan all symbols and return top-ranked predictions.

```bash
curl -X POST "http://localhost:8000/prediction_agent/tools/scan_all" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "top_k": 20,
    "horizon": "daily",
    "min_score": 0.0
  }'
```

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "symbol": "NVDA",
      "score": 0.85,
      "confidence": 0.78,
      "action": "long",
      "timestamp": "2025-01-18T10:00:00Z"
    }
  ],
  "metadata": {
    "returned": 20,
    "scanned": 150
  }
}
```

### 4. Analyze
**Endpoint:** `POST /prediction_agent/tools/analyze`

Detailed technical analysis for specific symbols.

```bash
curl -X POST "http://localhost:8000/prediction_agent/tools/analyze" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": ["AAPL", "MSFT"],
    "horizon": "daily",
    "detailed": true
  }'
```

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "symbol": "AAPL",
      "matched_symbol": "AAPL",
      "price": 185.20,
      "signals": {
        "rsi_14": 55.2,
        "macd_hist": 0.15,
        "volume_ratio": 1.2,
        "sma_20": 182.50,
        "sma_50": 178.30
      },
      "score": 0.68,
      "confidence": 0.75,
      "suggested_action": "long",
      "reason": "Strong uptrend | RSI neutral at 55.2",
      "timestamp": "2025-01-18T10:00:00Z"
    }
  ],
  "metadata": {"count": 2}
}
```

### 5. Feedback
**Endpoint:** `POST /prediction_agent/tools/feedback`

Submit user feedback for RL improvement.

**Required Fields:**
- `symbol`: Stock symbol (string)
- `predicted_action`: The action that was predicted ("long", "short", "hold")
- `user_feedback`: Whether the prediction was "correct" or "incorrect"

**Optional Fields:**
- `horizon`: Trading horizon (defaults to "daily")

```bash
curl -X POST "http://localhost:8000/prediction_agent/tools/feedback" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "predicted_action": "long",
    "user_feedback": "correct",
    "horizon": "daily"
  }'
```

**Response:**
```json
{
  "success": true,
  "data": {
    "updated": true,
    "reward": 10.0,
    "symbol": "AAPL"
  }
}
```

### 6. Train RL
**Endpoint:** `POST /prediction_agent/tools/train_rl`

Trigger RL agent retraining.

```bash
curl -X POST "http://localhost:8000/prediction_agent/tools/train_rl" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_type": "linucb",
    "rounds": 50,
    "top_k": 20,
    "horizon": 1
  }'
```

**Response:**
```json
{
  "success": true,
  "data": {
    "cumulative_reward": 45.2,
    "avg_reward": 0.904,
    "total_rounds": 50
  }
}
```

### 7. Fetch Data
**Endpoint:** `POST /prediction_agent/tools/fetch_data`

Fetch market data for symbols.

```bash
curl -X POST "http://localhost:8000/prediction_agent/tools/fetch_data" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": ["AAPL", "MSFT", "BTC-USD"],
    "period": "6mo",
    "interval": "1d"
  }'
```

**Response:**
```json
{
  "success": true,
  "data": {
    "fetched": {
      "AAPL": true,
      "MSFT": true,
      "BTC-USD": true
    }
  }
}
```

## Integration Patterns

### 1. Prediction Pipeline
```python
# 1. Get predictions
predictions = await client.predict(["AAPL", "MSFT"])

# 2. Filter by confidence
high_confidence = [p for p in predictions if p['confidence'] > 0.7]

# 3. Execute trades
for pred in high_confidence:
    if pred['action'] == 'long':
        await execute_buy_order(pred['symbol'])
```

### 2. Feedback Loop
```python
# 1. Get prediction
prediction = await client.predict(["AAPL"])

# 2. Execute trade
trade_result = await execute_trade(prediction[0])

# 3. Submit feedback
feedback = "correct" if trade_result.profitable else "incorrect"
await client.feedback("AAPL", "long", feedback)
```

### 3. Continuous Learning
```python
# 1. Regular retraining
await client.train_rl(agent_type="linucb", rounds=100)

# 2. Monitor performance
health = await client.health_check()
if health['models_loaded']:
    # System is ready for predictions
    pass
```

## Error Handling

### Common Error Responses

**401 Unauthorized:**
```json
{
  "detail": "Invalid or expired token"
}
```

**429 Rate Limited:**
```json
{
  "detail": "Rate limit exceeded. Max 100 requests per 3600s"
}
```

**503 Service Unavailable:**
```json
{
  "detail": "Service not initialized. Models not loaded."
}
```

### Retry Logic
```python
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def predict_with_retry(symbols):
    return await client.predict(symbols)
```

## Rate Limiting

- **Limit:** 100 requests per hour per IP
- **Headers:** Rate limit info in response headers
- **Strategy:** Exponential backoff with jitter

## Monitoring

### Health Check Integration
```python
async def monitor_health():
    while True:
        health = await client.health_check()
        if health['status'] != 'healthy':
            alert_system_administrator()
        await asyncio.sleep(60)
```

### Performance Metrics
- Response time tracking
- Error rate monitoring
- Model accuracy metrics
- Feedback loop effectiveness

## Security Considerations

1. **JWT Token Management**
   - Tokens expire in 30 minutes
   - Implement token refresh logic
   - Store tokens securely

2. **Rate Limiting**
   - IP-based rate limiting
   - Consider user-based limits for production

3. **Input Validation**
   - All inputs are validated by Pydantic
   - Symbol names are normalized
   - Risk profiles have bounds

## Testing

### Integration Test Suite
```bash
# Run comprehensive tests
python integration_test.py

# Test specific components
python integration_test.py --test health
python integration_test.py --test predict
python integration_test.py --test feedback
```

### Load Testing
```bash
# Test rate limiting
python integration_test.py --test rate-limit
```

## Deployment

### Docker Deployment
```dockerfile
FROM python:3.11-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8000
CMD ["python", "api/server.py"]
```

### Environment Variables
```bash
JWT_SECRET_KEY=your_secret_key_here
API_HOST=0.0.0.0
API_PORT=8000
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=3600
```

## Troubleshooting

### Common Issues

1. **Models Not Loaded**
   - Check feature store exists
   - Verify model files are present
   - Run training pipeline

2. **Authentication Failures**
   - Verify JWT secret key
   - Check token expiration
   - Ensure proper headers

3. **Rate Limiting**
   - Implement exponential backoff
   - Consider request batching
   - Monitor rate limit headers

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python api/server.py
```

## Support

For integration support:
- Check logs in `logs/api_server.log`
- Run integration tests
- Review this documentation
- Contact system administrator

## Changelog

### v2.1.0 (Current)
- Fixed symbol matching logic
- Enhanced feedback loop
- Improved error handling
- Added comprehensive testing

### v2.0.0
- Initial MCP implementation
- JWT authentication
- Rate limiting
- Basic endpoints

---

*This integration kit ensures seamless connectivity between Krishna Prediction Agent and external systems while maintaining high performance and reliability.*