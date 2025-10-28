# Blackhole Infiverse - AI Trading System

## What is This?

This is an **AI-powered stock trading system** that uses machine learning to predict whether stocks will go up (long), down (short), or stay the same (hold). It analyzes 50+ technical indicators and market sentiment to make smart trading decisions.

## Quick Start

### 1. Install Everything
```bash
# Install Python packages
pip install -r requirements.txt
```

### 2. Start the System
```bash
# Start the API server
python api/server.py
```

### 3. Test It Works
```bash
# Test the system
python test_enhanced_analysis.py
```

### 4. Use Postman
- Import `Blackhole_Infiverse_API_Complete.postman_collection.json` into Postman
- Test all the API endpoints

## How It Works

### The System Has 4 Main Parts:

1. **Data Collection** - Gets stock prices and market data
2. **Feature Engineering** - Creates 50+ technical indicators from raw data
3. **AI Models** - Machine learning models that make predictions
4. **API Server** - Web interface to use the system

### Technical Indicators Used (50+ Total):

#### **Momentum Indicators** (Tell us if price is moving fast)
- **RSI** - Shows if stock is overbought (too expensive) or oversold (too cheap)
- **Stochastic** - Another overbought/oversold indicator
- **Williams %R** - Momentum oscillator
- **CCI** - Commodity Channel Index
- **MFI** - Money Flow Index
- **TRIX** - Triple Exponential Moving Average
- **CMO** - Chande Momentum Oscillator
- **Aroon** - Trend strength indicator
- **Ultimate Oscillator** - Multi-timeframe momentum

#### **Trend Indicators** (Tell us which direction price is going)
- **SMA** - Simple Moving Average (10, 20, 50, 200 days)
- **EMA** - Exponential Moving Average (12, 26 days)
- **ADX** - Average Directional Index (trend strength)
- **Parabolic SAR** - Trend following indicator
- **Keltner Channels** - Volatility-based trend lines
- **Donchian Channels** - Price breakout indicator

#### **Volume Indicators** (Tell us how much trading is happening)
- **Volume Ratio** - Current volume vs average
- **OBV** - On Balance Volume
- **ADL** - Accumulation/Distribution Line
- **CMF** - Chaikin Money Flow
- **EMV** - Ease of Movement
- **Volume Trend** - Volume direction over time

#### **Volatility Indicators** (Tell us how much price moves)
- **Bollinger Bands** - Price range indicator
- **ATR** - Average True Range
- **Standard Deviation** - Price volatility measure

#### **Pattern Recognition** (Finds chart patterns)
- **Candlestick Patterns** - Doji, Hammer, Engulfing patterns
- **Support/Resistance** - Price levels where stock bounces
- **Pivot Points** - Key price levels
- **Fibonacci Levels** - Mathematical price retracements

#### **Advanced Analysis**
- **VWAP** - Volume Weighted Average Price
- **Sharpe Ratio** - Risk-adjusted returns
- **Beta/Alpha** - Stock vs market performance

#### **Sentiment Analysis** (Market mood)
- **Put/Call Ratio** - Options sentiment
- **Short Interest** - How many people betting against stock
- **Fear & Greed Index** - Overall market mood
- **VIX** - Volatility index (fear gauge)
- **News Sentiment** - AI analysis of news articles

## API Endpoints

### 1. **Get Predictions**
```bash
POST /prediction_agent/tools/predict
```
**What it does:** Gets AI predictions for specific stocks
**Input:**
```json
{
  "symbols": ["AAPL", "MSFT", "TSLA"],
  "horizon": "daily",
  "risk_profile": {
    "stop_loss_pct": 2.0,
    "capital_risk_pct": 1.5,
    "drawdown_limit_pct": 10.0
  }
}
```
**Output:**
```json
{
  "success": true,
  "data": [
    {
      "symbol": "AAPL",
      "action": "long",
      "score": 0.7234,
      "confidence": 0.8567,
      "predicted_price": 254.94,
      "reason": "**LONG RECOMMENDATION** (Score: 0.723)\n\n**TECHNICAL ANALYSIS:**\n• Momentum: Strong bullish momentum\n  - RSI oversold (28.3) - bullish momentum\n• Trend: Strong uptrend\n  - Price above both SMAs - strong uptrend\n• Volume: High volume confirmation\n  - High volume (2.3x average) - strong conviction\n\n**SENTIMENT ANALYSIS:**\n• Market Sentiment: Bullish sentiment\n  - Bullish sentiment (0.45)\n  - Low put/call ratio (0.65) - bullish sentiment\n\n**RISK ASSESSMENT:**\n• Risk Level: LOW RISK"
    }
  ]
}
```

### 2. **Analyze Stocks**
```bash
POST /prediction_agent/tools/analyze
```
**What it does:** Detailed technical analysis of stocks
**Input:**
```json
{
  "symbols": ["AAPL", "MSFT", "TSLA"]
}
```
**Output:**
- Set `detailed: true` to return ALL computed indicators (≈86 technical + 13 sentiment)
```json
{
  "symbols": ["AAPL"],
  "horizon": "daily",
  "detailed": true
}
```

### 3. **Scan All Stocks**
```bash
POST /prediction_agent/tools/scan_all
```
**What it does:** Finds best trading opportunities across all stocks
**Input:**
```json
{
  "limit": 10,
  "min_confidence": 0.3
}
```

### 4. **Give Feedback**
```bash
POST /prediction_agent/tools/feedback
```
**What it does:** Tell the AI if its prediction was right or wrong
**Input:**
```json
{
  "symbol": "AAPL",
  "predicted_action": "long",
  "user_feedback": "correct",
  "horizon": "daily"
}
```

### 5. **Train AI**
```bash
POST /prediction_agent/tools/train_rl
```
**What it does:** Trains the AI with new data

### 6. **Get More Data**
```bash
POST /prediction_agent/tools/fetch_data
```
**What it does:** Downloads latest stock data

## File Structure

```
Blackhole Infiverse/
├── api/
│   └── server.py              # Main API server
├── core/
│   ├── bhiv_core.py          # Core system logic
│   ├── mcp_adapter.py        # API interface
│   ├── enhanced_features.py   # 50+ technical indicators
│   ├── prediction_analyzer.py # Detailed analysis system
│   ├── sentiment_analyzer.py  # Market sentiment analysis
│   ├── feedback_loop.py       # Learning system
│   ├── resilience_system.py  # System health monitoring
│   └── models/
│       ├── enhanced_lightgbm.py # Machine learning model
│       └── rl_agent.py         # Reinforcement learning
├── data/
│   ├── cache/                # Cached stock data
│   ├── features/             # Processed features
│   └── bucket/              # System logs
├── models/                   # Trained AI models
├── logs/                     # System logs
└── requirements.txt          # Python packages needed
```

## How to Use

### **Step 1: Start the System**
```bash
python api/server.py
```
The server will start on `http://localhost:8000`

### **Step 2: Get a Token**
```bash
curl -X POST "http://localhost:8000/auth/token"
```
This gives you a JWT token to use the API

### **Step 3: Make Predictions**
```bash
curl -X POST "http://localhost:8000/prediction_agent/tools/predict" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": ["AAPL", "MSFT"],
    "horizon": "daily"
  }'
```

### **Step 4: Analyze Results**
The system will give you:
- **Action**: long, short, or hold
- **Score**: How confident the AI is (0-1)
- **Reason**: Detailed explanation of why
- **Risk Level**: How risky the trade is

## Understanding the Output

### **Action Types:**
- **LONG** = Buy the stock (expect price to go up)
- **SHORT** = Sell the stock (expect price to go down)
- **HOLD** = Don't trade (price will stay same)

### **Score Meaning:**
- **0.7-1.0** = Very confident prediction
- **0.5-0.7** = Confident prediction
- **0.3-0.5** = Moderate confidence
- **0.0-0.3** = Low confidence

### **Risk Levels:**
- **LOW RISK** = Safe to trade
- **MEDIUM RISK** = Some risk involved
- **HIGH RISK** = Very risky trade

## Technical Details

### **Machine Learning Models:**
1. **Enhanced LightGBM** - Gradient boosting model for predictions
2. **Reinforcement Learning Agent** - Learns from trading results
3. **Ensemble System** - Combines multiple models for better accuracy

### **Data Sources:**
- **Yahoo Finance** - Stock prices and volume
- **Alpha Vantage** - Additional market data
- **News APIs** - Sentiment analysis
- **Options Data** - Put/call ratios

### **Performance:**
- **Prediction Speed**: < 1 second per stock
- **Accuracy**: 60-70% on backtesting
- **Coverage**: 143+ stocks supported
- **Update Frequency**: Real-time data

## Troubleshooting

### **Common Issues:**

1. **"Module not found" error**
   ```bash
   pip install -r requirements.txt
   ```

2. **"Connection refused" error**
   - Make sure server is running: `python api/server.py`
   - Check if port 8000 is free

3. **"No data for symbol" error**
   - Symbol might not be supported
   - Try running: `python fetch_more_data.py`

4. **"Authentication failed" error**
   - Get new token: `curl -X POST "http://localhost:8000/auth/token"`

### **Check System Health:**
```bash
curl "http://localhost:8000/tools/health"
```

## Advanced Usage

### **Custom Risk Settings:**
```json
{
  "risk_profile": {
    "stop_loss_pct": 2.0,        // Stop loss at 2%
    "capital_risk_pct": 1.5,     // Risk 1.5% of capital
    "drawdown_limit_pct": 10.0   // Max 10% drawdown
  }
}
```

### **Training the AI:**
```bash
# Train with new data
python train_rl_agent.py

# Update features
python update_enhanced_features.py
```

### **Monitoring:**
- Check logs in `logs/` folder
- Monitor system health via API
- View performance metrics

## Support

### **Files to Check:**
- `logs/api_server.log` - Server logs
- `logs/integration_test_results.json` - Test results
- `test_results.json` - API test results

### **Useful Commands:**
```bash
# Test everything
python integration_test.py

# Debug API
python debug_api.py

# Test enhanced analysis
python test_enhanced_analysis.py

# Search for symbols
python search_symbol.py
```

## What Makes This Special?

1. **50+ Technical Indicators** - More analysis than most trading systems
2. **AI Learning** - Gets smarter with each trade
3. **Sentiment Analysis** - Understands market mood
4. **Risk Management** - Built-in safety features
5. **Real-time Updates** - Always uses latest data
6. **Detailed Explanations** - Know exactly why each decision was made

## Final Notes

- **This is for educational purposes** - Always do your own research
- **Past performance doesn't guarantee future results**
- **Start with small amounts** when testing
- **Monitor the system** regularly for best results
- **Give feedback** to help the AI learn

The system is designed to be **easy to use** but **powerful under the hood**. It combines traditional technical analysis with modern AI to give you the best possible trading insights.
