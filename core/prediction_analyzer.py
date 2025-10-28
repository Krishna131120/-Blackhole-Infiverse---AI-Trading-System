"""
Enhanced Prediction Analysis System
Provides comprehensive reasoning for all predictions with detailed technical analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class PredictionAnalyzer:
    """Comprehensive prediction analysis with detailed reasoning"""
    
    def __init__(self):
        self.indicator_weights = {
            'momentum': 0.25,
            'trend': 0.25,
            'volume': 0.15,
            'volatility': 0.15,
            'sentiment': 0.10,
            'patterns': 0.10
        }
    
    def analyze_prediction(self, symbol: str, features: Dict, action: str, score: float, 
                          sentiment_data: Dict = None) -> Dict[str, Any]:
        """Generate comprehensive prediction analysis"""
        
        analysis = {
            'symbol': symbol,
            'action': action,
            'score': score,
            'confidence_level': self._get_confidence_level(score),
            'timestamp': datetime.now().isoformat(),
            'technical_analysis': {},
            'sentiment_analysis': {},
            'risk_assessment': {},
            'detailed_reasoning': '',
            'key_factors': [],
            'contradictory_signals': [],
            'recommendation_strength': ''
        }
        
        # Technical Analysis
        analysis['technical_analysis'] = self._analyze_technical_indicators(features, action)
        
        # Sentiment Analysis
        if sentiment_data:
            analysis['sentiment_analysis'] = self._analyze_sentiment(sentiment_data, action)
        
        # Risk Assessment
        analysis['risk_assessment'] = self._assess_risk(features, sentiment_data)
        
        # Generate detailed reasoning
        analysis['detailed_reasoning'] = self._generate_detailed_reasoning(
            analysis['technical_analysis'], 
            analysis['sentiment_analysis'], 
            analysis['risk_assessment'],
            action, score
        )
        
        # Extract key factors
        analysis['key_factors'] = self._extract_key_factors(analysis)
        
        # Find contradictory signals
        analysis['contradictory_signals'] = self._find_contradictory_signals(analysis)
        
        # Determine recommendation strength
        analysis['recommendation_strength'] = self._get_recommendation_strength(analysis)
        
        return analysis
    
    def _analyze_technical_indicators(self, features: Dict, action: str) -> Dict[str, Any]:
        """Comprehensive technical indicator analysis"""
        
        analysis = {
            'momentum': {},
            'trend': {},
            'volume': {},
            'volatility': {},
            'patterns': {},
            'support_resistance': {}
        }
        
        # MOMENTUM ANALYSIS
        momentum_signals = []
        momentum_strength = 0
        
        # RSI Analysis
        if 'rsi_14' in features and not pd.isna(features['rsi_14']):
            rsi = features['rsi_14']
            if rsi > 70:
                momentum_signals.append(f"RSI overbought ({rsi:.1f}) - bearish momentum")
                momentum_strength -= 0.3
            elif rsi < 30:
                momentum_signals.append(f"RSI oversold ({rsi:.1f}) - bullish momentum")
                momentum_strength += 0.3
            elif 40 <= rsi <= 60:
                momentum_signals.append(f"RSI neutral ({rsi:.1f}) - balanced momentum")
            else:
                momentum_signals.append(f"RSI at {rsi:.1f} - moderate momentum")
        
        # Stochastic Analysis
        if 'stoch_k' in features and not pd.isna(features['stoch_k']):
            stoch = features['stoch_k']
            if stoch > 80:
                momentum_signals.append(f"Stochastic overbought ({stoch:.1f})")
                momentum_strength -= 0.2
            elif stoch < 20:
                momentum_signals.append(f"Stochastic oversold ({stoch:.1f})")
                momentum_strength += 0.2
        
        # Williams %R Analysis
        if 'williams_r' in features and not pd.isna(features['williams_r']):
            williams = features['williams_r']
            if williams > -20:
                momentum_signals.append(f"Williams %R overbought ({williams:.1f})")
                momentum_strength -= 0.15
            elif williams < -80:
                momentum_signals.append(f"Williams %R oversold ({williams:.1f})")
                momentum_strength += 0.15
        
        # CCI Analysis
        if 'cci' in features and not pd.isna(features['cci']):
            cci = features['cci']
            if cci > 100:
                momentum_signals.append(f"CCI bullish ({cci:.1f})")
                momentum_strength += 0.1
            elif cci < -100:
                momentum_signals.append(f"CCI bearish ({cci:.1f})")
                momentum_strength -= 0.1
        
        analysis['momentum'] = {
            'signals': momentum_signals,
            'strength': momentum_strength,
            'interpretation': self._interpret_momentum(momentum_strength, action)
        }
        
        # TREND ANALYSIS
        trend_signals = []
        trend_strength = 0
        
        # Moving Average Analysis
        if all(k in features for k in ['close', 'sma_20', 'sma_50']):
            close = features['close']
            sma_20 = features['sma_20']
            sma_50 = features['sma_50']
            
            if not any(pd.isna(v) for v in [close, sma_20, sma_50]):
                if close > sma_20 > sma_50:
                    trend_signals.append("Price above both SMAs - strong uptrend")
                    trend_strength += 0.4
                elif close < sma_20 < sma_50:
                    trend_signals.append("Price below both SMAs - strong downtrend")
                    trend_strength -= 0.4
                elif close > sma_20:
                    trend_signals.append("Price above SMA20 - short-term uptrend")
                    trend_strength += 0.2
                elif close < sma_20:
                    trend_signals.append("Price below SMA20 - short-term downtrend")
                    trend_strength -= 0.2
        
        # EMA Analysis
        if all(k in features for k in ['close', 'ema_20', 'ema_50']):
            close = features['close']
            ema_20 = features['ema_20']
            ema_50 = features['ema_50']
            
            if not any(pd.isna(v) for v in [close, ema_20, ema_50]):
                if close > ema_20 > ema_50:
                    trend_signals.append("Price above both EMAs - exponential uptrend")
                    trend_strength += 0.3
                elif close < ema_20 < ema_50:
                    trend_signals.append("Price below both EMAs - exponential downtrend")
                    trend_strength -= 0.3
        
        # ADX Analysis
        if 'adx' in features and not pd.isna(features['adx']):
            adx = features['adx']
            if adx > 25:
                trend_signals.append(f"ADX strong trend ({adx:.1f})")
                trend_strength *= 1.2  # Amplify trend strength
            elif adx < 20:
                trend_signals.append(f"ADX weak trend ({adx:.1f})")
                trend_strength *= 0.8  # Reduce trend strength
        
        analysis['trend'] = {
            'signals': trend_signals,
            'strength': trend_strength,
            'interpretation': self._interpret_trend(trend_strength, action)
        }
        
        # VOLUME ANALYSIS
        volume_signals = []
        volume_strength = 0
        
        if 'volume_ratio' in features and not pd.isna(features['volume_ratio']):
            vol_ratio = features['volume_ratio']
            if vol_ratio > 2.0:
                volume_signals.append(f"High volume ({vol_ratio:.1f}x average) - strong conviction")
                volume_strength += 0.3
            elif vol_ratio < 0.5:
                volume_signals.append(f"Low volume ({vol_ratio:.1f}x average) - weak conviction")
                volume_strength -= 0.2
            else:
                volume_signals.append(f"Normal volume ({vol_ratio:.1f}x average)")
        
        # OBV Analysis
        if 'obv' in features and not pd.isna(features['obv']):
            obv = features['obv']
            if obv > 0:
                volume_signals.append("OBV positive - accumulation")
                volume_strength += 0.1
            else:
                volume_signals.append("OBV negative - distribution")
                volume_strength -= 0.1
        
        analysis['volume'] = {
            'signals': volume_signals,
            'strength': volume_strength,
            'interpretation': self._interpret_volume(volume_strength, action)
        }
        
        # VOLATILITY ANALYSIS
        volatility_signals = []
        volatility_strength = 0
        
        # Bollinger Bands Analysis
        if 'bb_position' in features and not pd.isna(features['bb_position']):
            bb_pos = features['bb_position']
            if bb_pos > 0.8:
                volatility_signals.append(f"Price near upper Bollinger Band ({bb_pos:.2f}) - potential reversal")
                volatility_strength -= 0.2
            elif bb_pos < 0.2:
                volatility_signals.append(f"Price near lower Bollinger Band ({bb_pos:.2f}) - potential reversal")
                volatility_strength += 0.2
            else:
                volatility_signals.append(f"Price in middle Bollinger Band range ({bb_pos:.2f})")
        
        # ATR Analysis
        if 'atr' in features and not pd.isna(features['atr']):
            atr = features['atr']
            volatility_signals.append(f"ATR at {atr:.2f} - volatility level")
        
        analysis['volatility'] = {
            'signals': volatility_signals,
            'strength': volatility_strength,
            'interpretation': self._interpret_volatility(volatility_strength, action)
        }
        
        # PATTERN ANALYSIS
        pattern_signals = []
        
        # Candlestick Patterns
        if 'doji' in features and features['doji'] == 1:
            pattern_signals.append("Doji pattern detected - indecision")
        
        if 'hammer' in features and features['hammer'] == 1:
            pattern_signals.append("Hammer pattern detected - potential reversal")
        
        if 'engulfing' in features and features['engulfing'] == 1:
            pattern_signals.append("Engulfing pattern detected - strong reversal signal")
        
        analysis['patterns'] = {
            'signals': pattern_signals,
            'interpretation': self._interpret_patterns(pattern_signals, action)
        }
        
        # SUPPORT/RESISTANCE ANALYSIS
        support_resistance_signals = []
        
        # Pivot Points
        if all(k in features for k in ['close', 'pivot', 'resistance_1', 'support_1']):
            close = features['close']
            pivot = features['pivot']
            res1 = features['resistance_1']
            sup1 = features['support_1']
            
            if close > res1:
                support_resistance_signals.append("Price above resistance - bullish breakout")
            elif close < sup1:
                support_resistance_signals.append("Price below support - bearish breakdown")
            elif close > pivot:
                support_resistance_signals.append("Price above pivot - bullish bias")
            else:
                support_resistance_signals.append("Price below pivot - bearish bias")
        
        analysis['support_resistance'] = {
            'signals': support_resistance_signals,
            'interpretation': self._interpret_support_resistance(support_resistance_signals, action)
        }
        
        return analysis
    
    def _analyze_sentiment(self, sentiment_data: Dict, action: str) -> Dict[str, Any]:
        """Analyze sentiment indicators"""
        
        sentiment_signals = []
        sentiment_strength = 0
        
        # Composite Sentiment
        if 'composite_sentiment' in sentiment_data:
            comp_sent = sentiment_data['composite_sentiment']
            if comp_sent > 0.3:
                sentiment_signals.append(f"Bullish sentiment ({comp_sent:.2f})")
                sentiment_strength += 0.3
            elif comp_sent < -0.3:
                sentiment_signals.append(f"Bearish sentiment ({comp_sent:.2f})")
                sentiment_strength -= 0.3
            else:
                sentiment_signals.append(f"Neutral sentiment ({comp_sent:.2f})")
        
        # Fear & Greed Index
        if 'fear_greed_index' in sentiment_data:
            fgi = sentiment_data['fear_greed_index']
            if fgi > 70:
                sentiment_signals.append(f"Extreme greed ({fgi:.0f}) - contrarian bearish")
                sentiment_strength -= 0.2
            elif fgi < 30:
                sentiment_signals.append(f"Extreme fear ({fgi:.0f}) - contrarian bullish")
                sentiment_strength += 0.2
            else:
                sentiment_signals.append(f"Moderate sentiment ({fgi:.0f})")
        
        # Put/Call Ratio
        if 'put_call_ratio' in sentiment_data:
            pcr = sentiment_data['put_call_ratio']
            if pcr > 1.5:
                sentiment_signals.append(f"High put/call ratio ({pcr:.2f}) - bearish sentiment")
                sentiment_strength -= 0.15
            elif pcr < 0.7:
                sentiment_signals.append(f"Low put/call ratio ({pcr:.2f}) - bullish sentiment")
                sentiment_strength += 0.15
        
        # Short Interest
        if 'short_interest' in sentiment_data:
            si = sentiment_data['short_interest']
            if si > 0.4:
                sentiment_signals.append(f"High short interest ({si:.1%}) - potential squeeze")
                sentiment_strength += 0.1
            elif si < 0.1:
                sentiment_signals.append(f"Low short interest ({si:.1%}) - limited upside")
                sentiment_strength -= 0.1
        
        return {
            'signals': sentiment_signals,
            'strength': sentiment_strength,
            'interpretation': self._interpret_sentiment(sentiment_strength, action)
        }
    
    def _assess_risk(self, features: Dict, sentiment_data: Dict = None) -> Dict[str, Any]:
        """Assess risk factors"""
        
        risk_factors = []
        risk_level = 0
        
        # Volatility Risk
        if 'volatility_20' in features and not pd.isna(features['volatility_20']):
            vol = features['volatility_20']
            if vol > 0.05:  # 5% daily volatility
                risk_factors.append(f"High volatility ({vol:.1%}) - increased risk")
                risk_level += 0.3
            elif vol < 0.01:  # 1% daily volatility
                risk_factors.append(f"Low volatility ({vol:.1%}) - reduced risk")
                risk_level -= 0.1
        
        # Volume Risk
        if 'volume_ratio' in features and not pd.isna(features['volume_ratio']):
            vol_ratio = features['volume_ratio']
            if vol_ratio < 0.3:
                risk_factors.append(f"Very low volume ({vol_ratio:.1f}x) - liquidity risk")
                risk_level += 0.2
        
        # Sentiment Risk
        if sentiment_data and 'sentiment_volatility' in sentiment_data:
            sent_vol = sentiment_data['sentiment_volatility']
            if sent_vol > 0.5:
                risk_factors.append(f"High sentiment volatility ({sent_vol:.2f}) - emotional risk")
                risk_level += 0.2
        
        return {
            'factors': risk_factors,
            'level': risk_level,
            'interpretation': self._interpret_risk(risk_level)
        }
    
    def _generate_detailed_reasoning(self, technical: Dict, sentiment: Dict, risk: Dict, 
                                   action: str, score: float) -> str:
        """Generate comprehensive reasoning"""
        
        reasoning_parts = []
        
        # Action justification
        reasoning_parts.append(f"**{action.upper()} RECOMMENDATION** (Score: {score:.3f})")
        reasoning_parts.append("")
        
        # Technical Analysis Summary
        reasoning_parts.append("**TECHNICAL ANALYSIS:**")
        
        # Momentum
        if technical['momentum']['signals']:
            reasoning_parts.append(f"• Momentum: {technical['momentum']['interpretation']}")
            for signal in technical['momentum']['signals'][:2]:  # Top 2 signals
                reasoning_parts.append(f"  - {signal}")
        
        # Trend
        if technical['trend']['signals']:
            reasoning_parts.append(f"• Trend: {technical['trend']['interpretation']}")
            for signal in technical['trend']['signals'][:2]:
                reasoning_parts.append(f"  - {signal}")
        
        # Volume
        if technical['volume']['signals']:
            reasoning_parts.append(f"• Volume: {technical['volume']['interpretation']}")
            for signal in technical['volume']['signals']:
                reasoning_parts.append(f"  - {signal}")
        
        # Volatility
        if technical['volatility']['signals']:
            reasoning_parts.append(f"• Volatility: {technical['volatility']['interpretation']}")
            for signal in technical['volatility']['signals']:
                reasoning_parts.append(f"  - {signal}")
        
        # Patterns
        if technical['patterns']['signals']:
            reasoning_parts.append(f"• Patterns: {technical['patterns']['interpretation']}")
            for signal in technical['patterns']['signals']:
                reasoning_parts.append(f"  - {signal}")
        
        # Support/Resistance
        if technical['support_resistance']['signals']:
            reasoning_parts.append(f"• Support/Resistance: {technical['support_resistance']['interpretation']}")
            for signal in technical['support_resistance']['signals']:
                reasoning_parts.append(f"  - {signal}")
        
        # Sentiment Analysis
        if sentiment and sentiment['signals']:
            reasoning_parts.append("")
            reasoning_parts.append("**SENTIMENT ANALYSIS:**")
            reasoning_parts.append(f"• Market Sentiment: {sentiment['interpretation']}")
            for signal in sentiment['signals'][:3]:  # Top 3 signals
                reasoning_parts.append(f"  - {signal}")
        
        # Risk Assessment
        if risk['factors']:
            reasoning_parts.append("")
            reasoning_parts.append("**RISK ASSESSMENT:**")
            reasoning_parts.append(f"• Risk Level: {risk['interpretation']}")
            for factor in risk['factors']:
                reasoning_parts.append(f"  - {factor}")
        
        return "\n".join(reasoning_parts)
    
    def _extract_key_factors(self, analysis: Dict) -> List[str]:
        """Extract key factors driving the prediction"""
        
        key_factors = []
        
        # Technical factors
        if analysis['technical_analysis']['momentum']['strength'] > 0.2:
            key_factors.append("Strong momentum indicators")
        elif analysis['technical_analysis']['momentum']['strength'] < -0.2:
            key_factors.append("Weak momentum indicators")
        
        if analysis['technical_analysis']['trend']['strength'] > 0.3:
            key_factors.append("Strong trend alignment")
        elif analysis['technical_analysis']['trend']['strength'] < -0.3:
            key_factors.append("Strong trend reversal")
        
        if analysis['technical_analysis']['volume']['strength'] > 0.2:
            key_factors.append("High volume confirmation")
        
        # Sentiment factors
        if analysis['sentiment_analysis'] and analysis['sentiment_analysis']['strength'] > 0.2:
            key_factors.append("Positive sentiment")
        elif analysis['sentiment_analysis'] and analysis['sentiment_analysis']['strength'] < -0.2:
            key_factors.append("Negative sentiment")
        
        # Risk factors
        if analysis['risk_assessment']['level'] > 0.3:
            key_factors.append("Elevated risk factors")
        
        return key_factors
    
    def _find_contradictory_signals(self, analysis: Dict) -> List[str]:
        """Find contradictory signals"""
        
        contradictions = []
        
        # Check momentum vs trend
        momentum_strength = analysis['technical_analysis']['momentum']['strength']
        trend_strength = analysis['technical_analysis']['trend']['strength']
        
        if (momentum_strength > 0.2 and trend_strength < -0.2) or \
           (momentum_strength < -0.2 and trend_strength > 0.2):
            contradictions.append("Momentum and trend signals conflict")
        
        # Check sentiment vs technical
        if analysis['sentiment_analysis']:
            sentiment_strength = analysis['sentiment_analysis']['strength']
            combined_technical = momentum_strength + trend_strength
            
            if (sentiment_strength > 0.2 and combined_technical < -0.2) or \
               (sentiment_strength < -0.2 and combined_technical > 0.2):
                contradictions.append("Sentiment and technical signals conflict")
        
        return contradictions
    
    def _get_recommendation_strength(self, analysis: Dict) -> str:
        """Determine recommendation strength"""
        
        # Calculate overall strength
        momentum = analysis['technical_analysis']['momentum']['strength']
        trend = analysis['technical_analysis']['trend']['strength']
        volume = analysis['technical_analysis']['volume']['strength']
        
        sentiment = analysis['sentiment_analysis'].get('strength', 0) if analysis['sentiment_analysis'] else 0
        
        overall_strength = abs(momentum + trend + volume + sentiment)
        
        if overall_strength > 0.8:
            return "VERY STRONG"
        elif overall_strength > 0.5:
            return "STRONG"
        elif overall_strength > 0.3:
            return "MODERATE"
        elif overall_strength > 0.1:
            return "WEAK"
        else:
            return "VERY WEAK"
    
    def _get_confidence_level(self, score: float) -> str:
        """Get confidence level based on score"""
        
        if score > 0.7 or score < 0.3:
            return "HIGH"
        elif score > 0.6 or score < 0.4:
            return "MEDIUM"
        else:
            return "LOW"
    
    # Interpretation helper methods
    def _interpret_momentum(self, strength: float, action: str) -> str:
        if strength > 0.3:
            return "Strong bullish momentum" if action == "long" else "Momentum bullish but overridden"
        elif strength < -0.3:
            return "Strong bearish momentum" if action == "short" else "Momentum bearish but overridden"
        else:
            return "Neutral momentum"
    
    def _interpret_trend(self, strength: float, action: str) -> str:
        if strength > 0.3:
            return "Strong uptrend" if action == "long" else "Uptrend but other factors override"
        elif strength < -0.3:
            return "Strong downtrend" if action == "short" else "Downtrend but other factors override"
        else:
            return "Mixed trend signals"
    
    def _interpret_volume(self, strength: float, action: str) -> str:
        if strength > 0.2:
            return "High volume confirmation"
        elif strength < -0.2:
            return "Low volume - weak conviction"
        else:
            return "Normal volume levels"
    
    def _interpret_volatility(self, strength: float, action: str) -> str:
        if abs(strength) > 0.2:
            return "High volatility - increased risk"
        else:
            return "Normal volatility levels"
    
    def _interpret_patterns(self, signals: List[str], action: str) -> str:
        if not signals:
            return "No significant patterns detected"
        else:
            return f"{len(signals)} pattern(s) detected"
    
    def _interpret_support_resistance(self, signals: List[str], action: str) -> str:
        if not signals:
            return "No clear support/resistance signals"
        else:
            return signals[0] if signals else "Neutral support/resistance"
    
    def _interpret_sentiment(self, strength: float, action: str) -> str:
        if strength > 0.2:
            return "Bullish sentiment" if action == "long" else "Sentiment bullish but overridden"
        elif strength < -0.2:
            return "Bearish sentiment" if action == "short" else "Sentiment bearish but overridden"
        else:
            return "Neutral sentiment"
    
    def _interpret_risk(self, level: float) -> str:
        if level > 0.5:
            return "HIGH RISK"
        elif level > 0.3:
            return "MEDIUM RISK"
        elif level > 0.1:
            return "LOW RISK"
        else:
            return "VERY LOW RISK"
