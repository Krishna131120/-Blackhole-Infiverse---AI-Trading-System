MCP Learning Kit
================

What is MCP?
------------
Model Context Protocol (MCP) is a lightweight convention for exposing tools as JSON-style RPC endpoints so an orchestrator (or AI agent framework) can discover and call them in a uniform way.

This repo exposes the following tools via FastAPI:

- predict: POST /tools/predict
- scan_all: POST /tools/scan_all
- analyze: POST /tools/analyze
- feedback: POST /tools/feedback
- train_rl: POST /tools/train_rl
- fetch_data: POST /tools/fetch_data

Each tool accepts a JSON body and returns a structured JSON result that the orchestrator can consume.

Example Calls (JSON-RPC style via HTTP)
--------------------------------------

1) Get predictions

POST /tools/predict
{
  "jsonrpc": "2.0",
  "method": "predict",
  "id": 1,
  "params": {
    "symbols": ["AAPL"],
    "horizon": "daily"
  }
}

2) Submit feedback

POST /tools/feedback
{
  "jsonrpc": "2.0",
  "method": "feedback",
  "id": 2,
  "params": {
    "symbol": "BTC-USD",
    "predicted_action": "long",
    "user_feedback": "correct"
  }
}

3) Trigger training

POST /tools/train_rl
{
  "jsonrpc": "2.0",
  "method": "train_rl",
  "id": 3,
  "params": {
    "agent_type": "linucb",
    "rounds": 50,
    "top_k": 20,
    "horizon": 1
  }
}

4) Fetch data

POST /tools/fetch_data
{
  "jsonrpc": "2.0",
  "method": "fetch_data",
  "id": 4,
  "params": {
    "symbols": ["BTC-USD", "GC=F"],
    "period": "6mo",
    "interval": "1d"
  }
}

MCP Registry
------------
The adapter includes a small in-memory registry (`mcp_tools`) listing the available tools and their HTTP paths. An orchestrator can use this list to discover callable tools.

Notes
-----
- All endpoints are protected by JWT; obtain a token at POST /auth/token.
- Inputs are validated with Pydantic models.
- Feedback is logged to `logs/feedback_loop.json` for auditability.
- Data source is exclusively Yahoo Finance (yfinance). All other providers removed.


