# LLM Integration Setup Guide

This guide explains how to set up and use the LLM-powered stock analysis reason generation feature using Groq API.

## Overview

The LLM integration automatically generates personalized, human-readable explanations for stock trading decisions using Groq's fast and cost-effective language models. Instead of showing technical analysis in a structured format, it provides clear, conversational explanations that are easy to understand.

## Features

- **Personalized Explanations**: Each stock gets a unique, tailored explanation
- **Simple Language**: Technical indicators explained in plain English
- **Context-Aware**: Considers the specific stock's profile and market conditions
- **Fallback Support**: Falls back to original analysis if LLM is unavailable
- **Configurable**: Supports different models and settings

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables

Create a `.env` file in the project root:

```bash
# Required: Groq API Key (Primary)
GROQ_API_KEY=your_groq_api_key_here

# Optional: LLM Configuration
LLM_PROVIDER=groq
LLM_MODEL=llama-3.1-8b-instant
LLM_MAX_TOKENS=500
LLM_TEMPERATURE=0.7

# Fallback: OpenAI API Key (Optional)
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Get Groq API Key

1. Go to [Groq Console](https://console.groq.com/)
2. Sign up or log in
3. Go to API Keys section
4. Create a new API key
5. Copy the key to your `.env` file

**Note**: Groq offers faster inference and lower costs compared to OpenAI, making it ideal for real-time stock analysis.

## Usage

### Automatic Integration

The LLM integration is automatically enabled when you use the existing API endpoints:

```bash
# Get predictions with LLM-generated reasons
curl -X POST "http://localhost:8000/prediction_agent/tools/predict" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": ["AAPL", "MSFT"],
    "horizon": "daily"
  }'
```

### Testing the Integration

Run the test script to verify everything works:

```bash
python test_llm_integration.py
```

## Configuration Options

### Model Selection

**Groq Models (Recommended):**
- `llama-3.1-8b-instant` (default, fast and cost-effective)
- `llama-3.1-70b-versatile` (higher quality, still fast)
- `mixtral-8x7b-32768` (balanced quality and speed)

**OpenAI Models (Fallback):**
- `gpt-3.5-turbo` (cost-effective)
- `gpt-4` (higher quality, more expensive)
- `gpt-4-turbo` (balanced quality and cost)

### Generation Settings

- **max_tokens**: Maximum length of generated response (default: 500)
- **temperature**: Creativity level 0.0-1.0 (default: 0.7)
- **timeout**: API call timeout in seconds (default: 30)

### Fallback Behavior

If LLM generation fails, the system automatically falls back to the original detailed technical analysis format.

## Example Output

### Before (Original Format)
```
**LONG RECOMMENDATION** (Score: 0.723)

**TECHNICAL ANALYSIS:**
• Momentum: Strong bullish momentum
  - RSI oversold (28.3) - bullish momentum
  - Stochastic oversold (15.2)
• Trend: Strong uptrend
  - Price above both SMAs - strong uptrend
  - Price above SMA20 - short-term uptrend
```

### After (LLM-Generated Format)
```
Based on the technical analysis, AAPL shows strong bullish momentum with the RSI indicating the stock is oversold at 28.3, which historically suggests a potential upward move. The price is currently above both the 20-day and 50-day moving averages, confirming a solid uptrend pattern. Additionally, trading volume is 2.3x higher than average, indicating strong conviction from investors. The combination of oversold conditions, clear uptrend, and high volume makes this an attractive long opportunity with relatively low risk.
```

## API Integration

The LLM integration is seamlessly integrated into the existing MCP adapter:

```python
# In core/mcp_adapter.py
def _generate_reason(self, symbol: str, features: Dict, action: str, score: float) -> str:
    # ... existing analysis ...
    
    # Try LLM-powered reason generation
    llm_generator = LLMReasonGenerator()
    llm_reason = llm_generator.generate_stock_reason(
        symbol, analysis, action, score, current_price
    )
    
    return llm_reason if llm_reason else analysis['detailed_reasoning']
```

## Troubleshooting

### Common Issues

1. **"No OpenAI API key found"**
   - Set the `OPENAI_API_KEY` environment variable
   - Ensure the key is valid and has sufficient credits

2. **"LLM API call failed"**
   - Check your internet connection
   - Verify the API key is correct
   - Check OpenAI service status

3. **Fallback to original format**
   - This is normal behavior when LLM is unavailable
   - Check logs for specific error messages

### Debug Mode

Enable debug logging to see detailed information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Performance Considerations

- **API Latency**: LLM calls add ~1-2 seconds per prediction
- **Cost**: Each prediction costs ~$0.001-0.01 depending on model
- **Rate Limits**: OpenAI has rate limits (check your plan)
- **Caching**: Consider implementing caching for repeated analyses

## Security

- Never commit API keys to version control
- Use environment variables for sensitive data
- Consider using a secrets management service for production

## Future Enhancements

- Support for other LLM providers (Anthropic, Google, etc.)
- Caching of generated reasons
- Batch processing for multiple stocks
- Custom prompt templates
- A/B testing of different models
