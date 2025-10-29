"""
LLM-Powered Stock Analysis Reason Generator
Generates personalized, human-readable explanations for stock trading decisions
"""

import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import requests
import os
from .llm_config import get_llm_config, LLMConfig

logger = logging.getLogger(__name__)


class LLMReasonGenerator:
    """Generates personalized stock analysis reasons using LLM"""
    
    def __init__(self, config: LLMConfig = None):
        self.config = config or get_llm_config()
        
        if not self.config.is_available():
            logger.warning(f"No {self.config.provider.upper()} API key found. LLM reason generation will be disabled.")
    
    def generate_stock_reason(self, symbol: str, analysis_data: Dict, action: str, 
                            score: float, current_price: float) -> str:
        """Generate personalized LLM-powered reason for stock action"""
        
        if not self.config.is_available():
            return self._generate_fallback_reason(analysis_data, action, score)
        
        try:
            # Prepare context for LLM
            context = self._prepare_llm_context(symbol, analysis_data, action, score, current_price)
            
            # Generate prompt
            prompt = self._create_llm_prompt(context)
            
            # Call LLM API
            response = self._call_llm_api(prompt)
            
            if response:
                return response
            else:
                return self._generate_fallback_reason(analysis_data, action, score)
                
        except Exception as e:
            logger.error(f"Error generating LLM reason: {e}")
            return self._generate_fallback_reason(analysis_data, action, score)
    
    def _prepare_llm_context(self, symbol: str, analysis_data: Dict, action: str, 
                           score: float, current_price: float) -> Dict[str, Any]:
        """Prepare structured context for LLM"""
        
        technical = analysis_data.get('technical_analysis', {})
        sentiment = analysis_data.get('sentiment_analysis', {})
        risk = analysis_data.get('risk_assessment', {})
        
        # Extract key indicators
        momentum_signals = technical.get('momentum', {}).get('signals', [])
        trend_signals = technical.get('trend', {}).get('signals', [])
        volume_signals = technical.get('volume', {}).get('signals', [])
        volatility_signals = technical.get('volatility', {}).get('signals', [])
        pattern_signals = technical.get('patterns', {}).get('signals', [])
        support_resistance_signals = technical.get('support_resistance', {}).get('signals', [])
        
        sentiment_signals = sentiment.get('signals', []) if sentiment else []
        risk_factors = risk.get('factors', [])
        
        # Calculate confidence level
        confidence_level = self._get_confidence_level(score)
        
        # Determine market context
        market_context = self._determine_market_context(analysis_data)
        
        return {
            'symbol': symbol,
            'action': action,
            'score': score,
            'confidence_level': confidence_level,
            'current_price': current_price,
            'market_context': market_context,
            'key_indicators': {
                'momentum': momentum_signals[:3],  # Top 3 momentum signals
                'trend': trend_signals[:3],        # Top 3 trend signals
                'volume': volume_signals[:2],      # Top 2 volume signals
                'volatility': volatility_signals[:2],  # Top 2 volatility signals
                'patterns': pattern_signals[:2],   # Top 2 pattern signals
                'support_resistance': support_resistance_signals[:2],  # Top 2 S/R signals
                'sentiment': sentiment_signals[:3],  # Top 3 sentiment signals
                'risk': risk_factors[:3]           # Top 3 risk factors
            },
            'strengths': self._identify_strengths(analysis_data),
            'concerns': self._identify_concerns(analysis_data),
            'timestamp': datetime.now().isoformat()
        }
    
    def _create_llm_prompt(self, context: Dict[str, Any]) -> str:
        """Create optimized prompt for LLM"""
        
        symbol = context['symbol']
        action = context['action'].upper()
        score = context['score']
        confidence = context['confidence_level']
        price = context['current_price']
        
        # Build indicators summary
        indicators_text = self._format_indicators_for_llm(context['key_indicators'])
        
        # Build strengths and concerns
        strengths_text = "\n".join([f"• {s}" for s in context['strengths']]) if context['strengths'] else "• No major strengths identified"
        concerns_text = "\n".join([f"• {c}" for c in context['concerns']]) if context['concerns'] else "• No major concerns identified"
        
        prompt = f"""You are an expert stock analyst providing clear, personalized explanations for trading decisions. 

STOCK: {symbol} (${price:.2f})
RECOMMENDATION: {action} (Confidence: {confidence}, Score: {score:.3f})

TECHNICAL INDICATORS:
{indicators_text}

STRENGTHS:
{strengths_text}

CONCERNS:
{concerns_text}

MARKET CONTEXT: {context['market_context']}

Please provide a clear, personalized explanation for why {symbol} is recommended for {action.lower()} action. 

Requirements:
1. Write in simple, easy-to-understand language
2. Focus on the 2-3 most important reasons
3. Explain what each indicator means for this specific stock
4. Address any concerns or risks
5. Keep it concise (2-3 paragraphs max)
6. Use a conversational, confident tone
7. Make it specific to {symbol}, not generic

Format your response as a clear explanation without bullet points or technical jargon."""
        
        return prompt
    
    def _format_indicators_for_llm(self, indicators: Dict[str, List[str]]) -> str:
        """Format indicators in a readable way for LLM"""
        
        formatted = []
        
        for category, signals in indicators.items():
            if signals:
                category_name = category.replace('_', ' ').title()
                formatted.append(f"{category_name}:")
                for signal in signals:
                    formatted.append(f"  • {signal}")
                formatted.append("")
        
        return "\n".join(formatted)
    
    def _call_llm_api(self, prompt: str) -> Optional[str]:
        """Call LLM API (Groq or OpenAI) to generate reason"""
        
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert stock analyst who provides clear, personalized explanations for trading decisions. Always be specific to the stock being analyzed and explain technical indicators in simple terms."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            **self.config.to_dict()
        }
        
        try:
            response = requests.post(self.config.base_url, headers=headers, json=data, timeout=self.config.timeout)
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
            
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            return None
    
    def _get_confidence_level(self, score: float) -> str:
        """Get confidence level based on score"""
        if score > 0.7 or score < 0.3:
            return "HIGH"
        elif score > 0.6 or score < 0.4:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _determine_market_context(self, analysis_data: Dict) -> str:
        """Determine overall market context"""
        
        technical = analysis_data.get('technical_analysis', {})
        sentiment = analysis_data.get('sentiment_analysis', {})
        
        # Analyze overall trend strength
        trend_strength = technical.get('trend', {}).get('strength', 0)
        momentum_strength = technical.get('momentum', {}).get('strength', 0)
        volume_strength = technical.get('volume', {}).get('strength', 0)
        
        # Analyze sentiment
        sentiment_strength = sentiment.get('strength', 0) if sentiment else 0
        
        # Determine context
        if trend_strength > 0.3 and momentum_strength > 0.2:
            return "Strong bullish momentum with clear uptrend"
        elif trend_strength < -0.3 and momentum_strength < -0.2:
            return "Strong bearish momentum with clear downtrend"
        elif abs(sentiment_strength) > 0.3:
            return f"Market sentiment driven ({'bullish' if sentiment_strength > 0 else 'bearish'})"
        elif volume_strength > 0.2:
            return "High volume activity indicating strong conviction"
        else:
            return "Mixed signals with moderate market activity"
    
    def _identify_strengths(self, analysis_data: Dict) -> List[str]:
        """Identify key strengths in the analysis"""
        
        strengths = []
        technical = analysis_data.get('technical_analysis', {})
        sentiment = analysis_data.get('sentiment_analysis', {})
        
        # Momentum strengths
        momentum_strength = technical.get('momentum', {}).get('strength', 0)
        if momentum_strength > 0.3:
            strengths.append("Strong bullish momentum indicators")
        elif momentum_strength < -0.3:
            strengths.append("Strong bearish momentum indicators")
        
        # Trend strengths
        trend_strength = technical.get('trend', {}).get('strength', 0)
        if trend_strength > 0.3:
            strengths.append("Clear uptrend with price above key moving averages")
        elif trend_strength < -0.3:
            strengths.append("Clear downtrend with price below key moving averages")
        
        # Volume strengths
        volume_strength = technical.get('volume', {}).get('strength', 0)
        if volume_strength > 0.2:
            strengths.append("High volume confirmation supporting the move")
        
        # Sentiment strengths
        if sentiment:
            sentiment_strength = sentiment.get('strength', 0)
            if sentiment_strength > 0.2:
                strengths.append("Positive market sentiment alignment")
            elif sentiment_strength < -0.2:
                strengths.append("Negative market sentiment alignment")
        
        return strengths
    
    def _identify_concerns(self, analysis_data: Dict) -> List[str]:
        """Identify key concerns in the analysis"""
        
        concerns = []
        technical = analysis_data.get('technical_analysis', {})
        risk = analysis_data.get('risk_assessment', {})
        
        # Volatility concerns
        volatility_signals = technical.get('volatility', {}).get('signals', [])
        for signal in volatility_signals:
            if 'high volatility' in signal.lower() or 'increased risk' in signal.lower():
                concerns.append("High volatility increases risk")
                break
        
        # Volume concerns
        volume_signals = technical.get('volume', {}).get('signals', [])
        for signal in volume_signals:
            if 'low volume' in signal.lower() or 'weak conviction' in signal.lower():
                concerns.append("Low volume suggests weak conviction")
                break
        
        # Risk factors
        risk_factors = risk.get('factors', [])
        for factor in risk_factors:
            if 'high risk' in factor.lower() or 'elevated risk' in factor.lower():
                concerns.append("Elevated risk factors present")
                break
        
        # Contradictory signals
        contradictory = analysis_data.get('contradictory_signals', [])
        if contradictory:
            concerns.append("Some contradictory signals detected")
        
        return concerns
    
    def _generate_fallback_reason(self, analysis_data: Dict, action: str, score: float) -> str:
        """Generate fallback reason when LLM is not available"""
        
        technical = analysis_data.get('technical_analysis', {})
        
        # Get key signals
        momentum_signals = technical.get('momentum', {}).get('signals', [])
        trend_signals = technical.get('trend', {}).get('signals', [])
        volume_signals = technical.get('volume', {}).get('signals', [])
        
        # Build simple reason
        reasons = []
        
        if momentum_signals:
            reasons.append(momentum_signals[0])
        if trend_signals:
            reasons.append(trend_signals[0])
        if volume_signals:
            reasons.append(volume_signals[0])
        
        if reasons:
            return f"**{action.upper()} RECOMMENDATION** (Score: {score:.3f})\n\n" + \
                   "Key factors:\n" + "\n".join([f"• {r}" for r in reasons[:3]])
        else:
            return f"**{action.upper()} RECOMMENDATION** (Score: {score:.3f})\n\n" + \
                   "Based on technical analysis and market indicators."
