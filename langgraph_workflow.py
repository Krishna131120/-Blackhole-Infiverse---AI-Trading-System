"""
LangGraph Workflow for Krishna Prediction Agent
Replaces n8n automation with programmatic workflow orchestration.
Integrates with MCP endpoints and provides automated prediction → validation → execution pipeline.
"""

from typing import TypedDict, Annotated, Sequence, Literal
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
# from langgraph.checkpoint.sqlite import SqliteSaver  # Not used for now
import operator
import asyncio
import httpx
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ===================== STATE DEFINITIONS =====================
class WorkflowState(TypedDict):
    """State maintained across workflow execution"""
    messages: Annotated[Sequence[dict], operator.add]
    symbols: list[str]
    predictions: list[dict]
    validated_predictions: list[dict]
    karma_scores: dict[str, float]
    execution_decisions: list[dict]
    errors: list[str]
    workflow_stage: str
    api_token: str
    config: dict


# ===================== API CLIENT =====================
class MCPClient:
    """Client for Krishna Prediction Agent MCP endpoints"""
    
    def __init__(self, base_url: str = "http://localhost:8000", token: str = None):
        self.base_url = base_url
        self.token = token
        self.client = httpx.AsyncClient(timeout=30.0)
        
    async def authenticate(self, username: str = "demo", password: str = "demo"):
        """Get JWT token"""
        try:
            response = await self.client.post(
                f"{self.base_url}/auth/token",
                json={"username": username, "password": password}
            )
            response.raise_for_status()
            data = response.json()
            self.token = data["access_token"]
            return self.token
        except httpx.ConnectError as e:
            logger.error(f"❌ Cannot connect to API server at {self.base_url}")
            logger.error("💡 Make sure to start the API server first:")
            logger.error("   python api/server.py")
            raise ConnectionError(f"API server not running at {self.base_url}. Start server first with: python api/server.py")
        except Exception as e:
            logger.error(f"❌ Authentication failed: {e}")
            raise
    
    def _headers(self):
        return {"Authorization": f"Bearer {self.token}"}
    
    async def predict(self, symbols: list[str], horizon: str = "daily", risk_profile: dict = None):
        """Call /tools/predict"""
        payload = {
            "symbols": symbols,
            "horizon": horizon,
            "risk_profile": risk_profile
        }
        response = await self.client.post(
            f"{self.base_url}/prediction_agent/tools/predict",
            json=payload,
            headers=self._headers()
        )
        response.raise_for_status()
        return response.json()
    
    async def scan_all(self, top_k: int = 20, horizon: str = "daily", min_score: float = 0.0):
        """Call /tools/scan_all"""
        payload = {
            "top_k": top_k,
            "horizon": horizon,
            "min_score": min_score
        }
        response = await self.client.post(
            f"{self.base_url}/prediction_agent/tools/scan_all",
            json=payload,
            headers=self._headers()
        )
        response.raise_for_status()
        return response.json()
    
    async def analyze(self, symbols: list[str], horizon: str = "daily"):
        """Call /tools/analyze"""
        payload = {
            "symbols": symbols,
            "horizon": horizon,
            "detailed": True
        }
        response = await self.client.post(
            f"{self.base_url}/prediction_agent/tools/analyze",
            json=payload,
            headers=self._headers()
        )
        response.raise_for_status()
        return response.json()
    
    async def feedback(self, symbol: str, predicted_action: str, user_feedback: str):
        """Call /tools/feedback"""
        payload = {
            "symbol": symbol,
            "predicted_action": predicted_action,
            "user_feedback": user_feedback
        }
        response = await self.client.post(
            f"{self.base_url}/prediction_agent/tools/feedback",
            json=payload,
            headers=self._headers()
        )
        response.raise_for_status()
        return response.json()
    
    async def train_rl(self, agent_type: str = "linucb", rounds: int = 50, top_k: int = 20):
        """Call /tools/train_rl"""
        payload = {
            "agent_type": agent_type,
            "rounds": rounds,
            "top_k": top_k,
            "horizon": "daily"
        }
        response = await self.client.post(
            f"{self.base_url}/prediction_agent/tools/train_rl",
            json=payload,
            headers=self._headers()
        )
        response.raise_for_status()
        return response.json()
    
    async def fetch_data(self, symbols: list[str] = None, period: str = "6mo"):
        """Call /tools/fetch_data"""
        payload = {
            "symbols": symbols or [],
            "period": period,
            "interval": "1d"
        }
        response = await self.client.post(
            f"{self.base_url}/prediction_agent/tools/fetch_data",
            json=payload,
            headers=self._headers()
        )
        response.raise_for_status()
        return response.json()
    
    async def close(self):
        await self.client.aclose()


# ===================== KARMA SYSTEM INTEGRATION =====================
class KarmaTracker:
    """
    Integration with Karma Tracker (Siddhesh's component)
    Provides ethical/behavioral weighting for trade decisions
    """
    
    def __init__(self, karma_file: str = "logs/karma_scores.json"):
        self.karma_file = Path(karma_file)
        self.karma_file.parent.mkdir(parents=True, exist_ok=True)
        self.scores = self._load_scores()
    
    def _load_scores(self) -> dict:
        """Load karma scores from file"""
        if self.karma_file.exists():
            try:
                with open(self.karma_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load karma scores: {e}")
        return {"default": 1.0}
    
    def _save_scores(self):
        """Save karma scores to file"""
        try:
            with open(self.karma_file, 'w') as f:
                json.dump(self.scores, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save karma scores: {e}")
    
    def get_karma_score(self, symbol: str) -> float:
        """Get karma score for symbol (0.0 to 1.0)"""
        return self.scores.get(symbol, self.scores.get("default", 1.0))
    
    def update_karma(self, symbol: str, outcome: str, confidence: float):
        """
        Update karma score based on trade outcome
        
        Args:
            symbol: Trading symbol
            outcome: 'positive', 'negative', or 'neutral'
            confidence: Confidence of original prediction
        """
        current = self.get_karma_score(symbol)
        
        if outcome == 'positive':
            adjustment = 0.05 * confidence
        elif outcome == 'negative':
            adjustment = -0.05 * confidence
        else:
            adjustment = 0.0
        
        # Keep karma score in [0.0, 1.0] range
        new_score = max(0.0, min(1.0, current + adjustment))
        self.scores[symbol] = new_score
        self._save_scores()
        
        logger.info(f"Karma updated for {symbol}: {current:.3f} → {new_score:.3f}")
    
    def apply_karma_weighting(self, predictions: list[dict]) -> list[dict]:
        """
        Apply karma weighting to predictions
        Reduces scores for symbols with poor karma
        """
        weighted = []
        for pred in predictions:
            symbol = pred.get('symbol')
            karma = self.get_karma_score(symbol)
            
            # Apply karma weighting to score and confidence
            weighted_pred = pred.copy()
            weighted_pred['score'] = pred['score'] * karma
            weighted_pred['confidence'] = pred['confidence'] * karma
            weighted_pred['karma_score'] = karma
            weighted_pred['original_score'] = pred['score']
            
            weighted.append(weighted_pred)
        
        return weighted


# ===================== WORKFLOW NODES =====================
class WorkflowNodes:
    """Node implementations for LangGraph workflow"""
    
    def __init__(self, mcp_client: MCPClient, karma_tracker: KarmaTracker):
        self.mcp = mcp_client
        self.karma = karma_tracker
    
    async def fetch_universe(self, state: WorkflowState) -> WorkflowState:
        """Node 1: Load trading universe"""
        logger.info("🔄 Node: Fetching Universe")
        
        try:
            # Load universe from file or use default
            universe_file = Path("universe.txt")
            if universe_file.exists():
                with open(universe_file, 'r') as f:
                    symbols = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            else:
                symbols = state.get('symbols', ['AAPL', 'MSFT', 'BTC/USD', 'ETH/USD'])
            
            state['symbols'] = symbols
            state['workflow_stage'] = 'universe_loaded'
            state['messages'].append({
                "role": "system",
                "content": f"Loaded {len(symbols)} symbols from universe"
            })
            
            logger.info(f"✅ Loaded {len(symbols)} symbols")
            
        except Exception as e:
            logger.error(f"❌ Universe fetch failed: {e}")
            state['errors'].append(f"Universe fetch: {str(e)}")
        
        return state
    
    async def generate_predictions(self, state: WorkflowState) -> WorkflowState:
        """Node 2: Generate predictions using Krishna agent"""
        logger.info("🔄 Node: Generating Predictions")
        
        try:
            node_start = datetime.now()
            # Use scan_all for universe-wide ranking
            config = state.get('config', {})
            top_k = config.get('top_k', 20)
            min_score = config.get('min_score', 0.0)
            
            response = await self.mcp.scan_all(
                top_k=top_k,
                horizon='daily',
                min_score=min_score
            )
            node_elapsed = (datetime.now() - node_start).total_seconds()
            logger.info(f"⏱️ scan_all latency: {node_elapsed:.3f}s for top_k={top_k}")
            
            if response.get('success'):
                predictions = response.get('data', [])
                logger.info(f"📊 Debug: Raw predictions received: {len(predictions)}")
                if predictions:
                    logger.info(f"📊 Debug: First prediction: {predictions[0]}")
                
                # Attach latency metadata if available
                state['messages'].append({
                    "role": "system",
                    "content": f"scan_all latency {node_elapsed:.3f}s"
                })
                state['predictions'] = predictions
                state['workflow_stage'] = 'predictions_generated'
                state['messages'].append({
                    "role": "assistant",
                    "content": f"Generated {len(predictions)} predictions"
                })
                
                logger.info(f"✅ Generated {len(predictions)} predictions")
            else:
                error = response.get('error', 'Unknown error')
                state['errors'].append(f"Prediction generation: {error}")
                logger.error(f"❌ Prediction failed: {error}")
        
        except Exception as e:
            logger.error(f"❌ Prediction error: {e}")
            state['errors'].append(f"Prediction generation: {str(e)}")
        
        return state
    
    async def apply_karma_filter(self, state: WorkflowState) -> WorkflowState:
        """Node 3: Apply Karma weighting (BHIV ethics layer)"""
        logger.info("🔄 Node: Applying Karma Filter")
        
        try:
            predictions = state.get('predictions', [])
            logger.info(f"📊 Debug: Applying karma to {len(predictions)} predictions")
            
            if not predictions:
                logger.warning("⚠️  No predictions to filter")
                return state
            
            # Apply karma weighting
            weighted_predictions = self.karma.apply_karma_weighting(predictions)
            
            # Re-sort by weighted score
            weighted_predictions.sort(key=lambda x: x['score'], reverse=True)
            
            state['validated_predictions'] = weighted_predictions
            state['workflow_stage'] = 'karma_applied'
            state['messages'].append({
                "role": "system",
                "content": f"Applied karma weighting to {len(weighted_predictions)} predictions"
            })
            
            # Log karma impact
            for pred in weighted_predictions[:5]:
                logger.info(
                    f"  {pred['symbol']}: "
                    f"Score {pred['original_score']:.3f} → {pred['score']:.3f} "
                    f"(Karma: {pred['karma_score']:.3f})"
                )
            
            logger.info("✅ Karma filtering complete")
        
        except Exception as e:
            logger.error(f"❌ Karma filter error: {e}")
            state['errors'].append(f"Karma filter: {str(e)}")
        
        return state
    
    async def validate_predictions(self, state: WorkflowState) -> WorkflowState:
        """Node 4: Cross-validation with Karan executor"""
        logger.info("🔄 Node: Validating Predictions")
        
        try:
            validated = state.get('validated_predictions', [])
            logger.info(f"📊 Debug: Starting validation with {len(validated)} predictions")
            
            if not validated:
                logger.warning("⚠️  No predictions to validate")
                return state
            
            # Get top predictions for detailed analysis
            top_symbols = [p['symbol'] for p in validated[:10]]
            
            # Call analyze endpoint for detailed validation
            analysis_response = await self.mcp.analyze(
                symbols=top_symbols,
                horizon='daily'
            )
            
            if analysis_response.get('success'):
                analyses = analysis_response.get('data', [])
                
                # Merge analysis data into predictions
                analysis_map = {a['symbol']: a for a in analyses}
                
                for pred in validated:
                    symbol = pred['symbol']
                    if symbol in analysis_map:
                        pred['technical_analysis'] = analysis_map[symbol].get('signals', {})
                        pred['validation_reason'] = analysis_map[symbol].get('reason', '')
                
                state['workflow_stage'] = 'validated'
                state['messages'].append({
                    "role": "assistant",
                    "content": f"Validated {len(analyses)} predictions with technical analysis"
                })
                
                logger.info(f"✅ Validated {len(analyses)} predictions")
            else:
                logger.warning("⚠️  Validation endpoint returned error")
        
        except Exception as e:
            logger.error(f"❌ Validation error: {e}")
            state['errors'].append(f"Validation: {str(e)}")
        
        return state
    
    async def make_execution_decisions(self, state: WorkflowState) -> WorkflowState:
        """Node 5: Make final execution decisions"""
        logger.info("🔄 Node: Making Execution Decisions")
        
        try:
            validated = state.get('validated_predictions', [])
            config = state.get('config', {})
            
            # Debug logging
            logger.info(f"📊 Debug: Found {len(validated)} validated predictions")
            if validated:
                logger.info(f"📊 Debug: First prediction: {validated[0]}")
            
            # Decision thresholds - Updated to match MCP adapter
            min_confidence = config.get('min_confidence', 0.60)  # Higher threshold for clearer signals
            min_karma = config.get('min_karma', 0.5)
            max_positions = config.get('max_positions', 5)
            
            logger.info(f"📊 Debug: Thresholds - min_confidence: {min_confidence}, min_karma: {min_karma}, max_positions: {max_positions}")
            
            execution_decisions = []
            
            for i, pred in enumerate(validated):
                logger.info(f"📊 Debug: Processing prediction {i+1}: {pred.get('symbol', 'unknown')}")
                logger.info(f"📊 Debug: Confidence: {pred.get('confidence', 'missing')}, Karma: {pred.get('karma_score', 'missing')}")
                
                # Apply decision filters
                if pred.get('confidence', 0) < min_confidence:
                    logger.info(f"📊 Debug: Skipping {pred.get('symbol')} - confidence {pred.get('confidence')} < {min_confidence}")
                    continue
                
                if pred.get('karma_score', 1.0) < min_karma:
                    logger.info(f"📊 Debug: Skipping {pred.get('symbol')} - karma {pred.get('karma_score')} < {min_karma}")
                    continue
                
                # Temporarily allow hold actions for testing - comment out this filter
                # if pred.get('action') == 'hold':
                #     logger.info(f"📊 Debug: Skipping {pred.get('symbol')} - hold action (no execution needed)")
                #     continue
                
                # Determine position size based on confidence and karma
                confidence = pred.get('confidence', 0)
                karma = pred.get('karma_score', 1.0)
                
                # Position size: 0.5% to 2.5% of capital
                position_size_pct = 0.5 + (2.0 * confidence * karma)
                
                decision = {
                    'symbol': pred.get('symbol'),
                    'action': pred.get('action'),
                    'score': pred.get('score'),
                    'confidence': confidence,
                    'karma_score': karma,
                    'position_size_pct': round(position_size_pct, 2),
                    'predicted_price': pred.get('predicted_price'),
                    'timestamp': datetime.now().isoformat(),
                    'execution_approved': True
                }
                
                execution_decisions.append(decision)
                logger.info(f"📊 Debug: Added decision for {pred.get('symbol')}")
                
                # Limit to max positions
                if len(execution_decisions) >= max_positions:
                    logger.info(f"📊 Debug: Reached max positions limit ({max_positions})")
                    break
            
            logger.info(f"📊 Debug: Final execution decisions: {len(execution_decisions)}")
            state['execution_decisions'] = execution_decisions
            state['workflow_stage'] = 'decisions_made'
            state['messages'].append({
                "role": "assistant",
                "content": f"Approved {len(execution_decisions)} trades for execution"
            })
            
            # Log decisions
            logger.info(f"✅ Execution decisions: {len(execution_decisions)} approved")
            for dec in execution_decisions:
                logger.info(
                    f"  {dec['symbol']}: {dec['action']} "
                    f"(Conf: {dec['confidence']:.2f}, "
                    f"Karma: {dec['karma_score']:.2f}, "
                    f"Size: {dec['position_size_pct']:.2f}%)"
                )
        
        except Exception as e:
            logger.error(f"❌ Decision error: {e}")
            state['errors'].append(f"Execution decisions: {str(e)}")
        
        return state
    
    async def log_pipeline(self, state: WorkflowState) -> WorkflowState:
        """Node 6: Log entire pipeline execution"""
        logger.info("🔄 Node: Logging Pipeline")
        
        try:
            log_dir = Path("logs/trade_pipeline")
            log_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"pipeline_{timestamp}.json"
            
            pipeline_log = {
                'timestamp': datetime.now().isoformat(),
                'workflow_stage': state.get('workflow_stage'),
                'symbols_processed': len(state.get('symbols', [])),
                'predictions_generated': len(state.get('predictions', [])),
                'predictions_validated': len(state.get('validated_predictions', [])),
                'execution_decisions': state.get('execution_decisions', []),
                'errors': state.get('errors', []),
                'messages': state.get('messages', [])
            }
            
            with open(log_file, 'w') as f:
                json.dump(pipeline_log, f, indent=2)
            
            logger.info(f"✅ Pipeline logged to {log_file}")
            
            state['messages'].append({
                "role": "system",
                "content": f"Pipeline logged to {log_file}"
            })
        
        except Exception as e:
            logger.error(f"❌ Logging error: {e}")
            state['errors'].append(f"Logging: {str(e)}")
        
        return state
    
    async def handle_feedback(self, state: WorkflowState) -> WorkflowState:
        """Node 7: Process feedback (incremental only, no full retrain)"""
        logger.info("🔄 Node: Handling Feedback")
        
        try:
            # Check for feedback file
            feedback_file = Path("logs/pending_feedback.json")
            
            if not feedback_file.exists():
                logger.info("ℹ️  No pending feedback")
                return state
            
            # Load feedback
            with open(feedback_file, 'r') as f:
                feedback_items = json.load(f)
            
            # Process each feedback
            for item in feedback_items:
                symbol = item['symbol']
                predicted_action = item['predicted_action']
                user_feedback = item['user_feedback']
                
                # Send to MCP feedback endpoint
                await self.mcp.feedback(symbol, predicted_action, user_feedback)
                
                # Update karma based on feedback
                outcome = 'positive' if user_feedback == 'correct' else 'negative'
                confidence = item.get('confidence', 0.5)
                self.karma.update_karma(symbol, outcome, confidence)
                
                logger.info(f"✅ Processed feedback for {symbol}: {user_feedback}")
            
            # Clear processed feedback
            feedback_file.unlink()
            
            # Incremental updates only; no full retraining here
            state['messages'].append({
                "role": "system",
                "content": f"Processed {len(feedback_items)} feedback items (incremental updates applied)"
            })
            
            state['workflow_stage'] = 'feedback_processed'
        
        except Exception as e:
            logger.error(f"❌ Feedback handling error: {e}")
            state['errors'].append(f"Feedback: {str(e)}")
        
        return state


# ===================== WORKFLOW GRAPH BUILDER =====================
class WorkflowBuilder:
    """Builds the LangGraph workflow"""
    
    def __init__(self, mcp_client: MCPClient, karma_tracker: KarmaTracker):
        self.nodes = WorkflowNodes(mcp_client, karma_tracker)
        self.graph = None
    
    def build(self) -> StateGraph:
        """Build the workflow graph"""
        # Create graph
        workflow = StateGraph(WorkflowState)
        
        # Add nodes
        workflow.add_node("fetch_universe", self.nodes.fetch_universe)
        workflow.add_node("generate_predictions", self.nodes.generate_predictions)
        workflow.add_node("apply_karma", self.nodes.apply_karma_filter)
        workflow.add_node("validate", self.nodes.validate_predictions)
        workflow.add_node("make_decisions", self.nodes.make_execution_decisions)
        workflow.add_node("log_pipeline", self.nodes.log_pipeline)
        workflow.add_node("handle_feedback", self.nodes.handle_feedback)
        
        # Define flow
        workflow.set_entry_point("fetch_universe")
        
        # Linear pipeline flow
        workflow.add_edge("fetch_universe", "generate_predictions")
        workflow.add_edge("generate_predictions", "apply_karma")
        workflow.add_edge("apply_karma", "validate")
        workflow.add_edge("validate", "make_decisions")
        workflow.add_edge("make_decisions", "log_pipeline")
        workflow.add_edge("log_pipeline", "handle_feedback")
        workflow.add_edge("handle_feedback", END)
        
        # Compile without checkpointing (simplified for now)
        self.graph = workflow.compile()
        
        return self.graph
    
    def visualize(self, output_file: str = "workflow_graph.png"):
        """Generate workflow visualization"""
        try:
            from IPython.display import Image, display
            display(Image(self.graph.get_graph().draw_mermaid_png()))
        except Exception as e:
            logger.warning(f"Could not visualize graph: {e}")


# ===================== SCHEDULER =====================
class WorkflowScheduler:
    """Schedules workflow execution"""
    
    def __init__(self, workflow_builder: WorkflowBuilder):
        self.builder = workflow_builder
        self.graph = workflow_builder.graph
        self.running = False
    
    async def run_once(self, config: dict = None) -> dict:
        """Run workflow once"""
        logger.info("=" * 60)
        logger.info("STARTING WORKFLOW EXECUTION")
        logger.info("=" * 60)
        
        # Initialize state
        initial_state = WorkflowState(
            messages=[],
            symbols=[],
            predictions=[],
            validated_predictions=[],
            karma_scores={},
            execution_decisions=[],
            errors=[],
            workflow_stage='initialized',
            api_token='',
            config=config or {}
        )
        
        # Execute workflow
        final_state = await self.graph.ainvoke(initial_state)
        
        logger.info("=" * 60)
        logger.info("WORKFLOW EXECUTION COMPLETE")
        logger.info("=" * 60)
        
        return final_state
    
    async def run_scheduled(self, interval_minutes: int = 60, config: dict = None):
        """Run workflow on schedule"""
        logger.info(f"Starting scheduled workflow (every {interval_minutes} minutes)")
        self.running = True
        
        while self.running:
            try:
                await self.run_once(config)
                await asyncio.sleep(interval_minutes * 60)
            except KeyboardInterrupt:
                logger.info("Scheduler stopped by user")
                self.running = False
                break
            except Exception as e:
                logger.error(f"Workflow error: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry
    
    def stop(self):
        """Stop scheduler"""
        self.running = False


# ===================== MAIN EXECUTION =====================
async def main():
    """Main execution function"""
    # Configuration
    api_base_url = os.getenv("API_BASE_URL", "http://localhost:8000")
    
    # Initialize components
    logger.info("Initializing components...")
    mcp_client = MCPClient(base_url=api_base_url)
    karma_tracker = KarmaTracker()
    
    # Authenticate
    logger.info("Authenticating...")
    try:
        await mcp_client.authenticate(username="demo", password="demo")
        logger.info("✅ Authenticated")
    except ConnectionError as e:
        logger.error(f"❌ {e}")
        logger.error("🔄 Please start the API server first, then run LangGraph again")
        return
    except Exception as e:
        logger.error(f"❌ Authentication failed: {e}")
        return
    
    # Build workflow
    logger.info("Building workflow graph...")
    builder = WorkflowBuilder(mcp_client, karma_tracker)
    workflow_graph = builder.build()
    logger.info("✅ Workflow graph built")
    
    # Create scheduler
    scheduler = WorkflowScheduler(builder)
    
    # Run mode
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "scheduled":
        # Scheduled mode
        interval = int(sys.argv[2]) if len(sys.argv) > 2 else 60
        config = {
            'top_k': 20,
            'min_score': 0.0,
            'min_confidence': 0.60,  # Higher threshold for clearer signals
            'min_karma': 0.5,
            'max_positions': 5
        }
        await scheduler.run_scheduled(interval_minutes=interval, config=config)
    else:
        # One-time run
        config = {
            'top_k': 20,
            'min_score': 0.0,
            'min_confidence': 0.60,  # Higher threshold for clearer signals
            'min_karma': 0.5,
            'max_positions': 5
        }
        final_state = await scheduler.run_once(config)
        
        # Print results
        print("\n" + "=" * 60)
        print("EXECUTION RESULTS")
        print("=" * 60)
        print(f"Workflow Stage: {final_state.get('workflow_stage')}")
        print(f"Predictions Generated: {len(final_state.get('predictions', []))}")
        print(f"Execution Decisions: {len(final_state.get('execution_decisions', []))}")
        print(f"Errors: {len(final_state.get('errors', []))}")
        
        if final_state.get('execution_decisions'):
            print("\nApproved Trades:")
            for dec in final_state['execution_decisions']:
                print(f"  - {dec['symbol']}: {dec['action']} "
                      f"(Conf: {dec['confidence']:.2f}, Size: {dec['position_size_pct']:.1f}%)")
    
    # Cleanup
    await mcp_client.close()


if __name__ == "__main__":
    asyncio.run(main())
