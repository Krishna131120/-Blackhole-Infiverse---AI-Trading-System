#!/usr/bin/env python3
"""
Train RL Agent Individually - FIXED VERSION
Simple script to train the RL agent directly with proper API integration
Ensures high accuracy and seamless feedback integration
"""

import sys
from pathlib import Path
import argparse
import logging
from datetime import datetime
import json

# Fix Unicode for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from core.models.rl_agent import LinUCBAgent, ThompsonSamplingAgent, DQNAgent, RLTrainer
from core.enhanced_features import EnhancedFeaturePipeline
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Train RL Agent Individually")
    parser.add_argument("--agent", choices=["linucb", "thompson", "dqn"], default="linucb",
                       help="RL agent type to train")
    parser.add_argument("--rounds", type=int, default=100, help="Training rounds")
    parser.add_argument("--top-k", type=int, default=20, help="Top-k symbols for training")
    parser.add_argument("--test-feedback", action="store_true", help="Test feedback system")
    args = parser.parse_args()
    
    print("="*60)
    print("TRAIN RL AGENT INDIVIDUALLY - FIXED VERSION")
    print("="*60)
    print(f"Agent Type: {args.agent.upper()}")
    print(f"Training Rounds: {args.rounds}")
    print(f"Top-K Selection: {args.top_k}")
    
    # Load feature store
    print("\n[1] Loading feature store...")
    pipeline = EnhancedFeaturePipeline()
    
    try:
        feature_dict = pipeline.load_feature_store()
        print(f"[OK] Loaded features for {len(feature_dict)} symbols")
    except FileNotFoundError:
        print("[ERROR] Feature store not found!")
        print("Please run: python core/features.py")
        return
    
    # Get feature dimensions
    sample_df = next(iter(feature_dict.values()))
    exclude_cols = [
        'open', 'high', 'low', 'close', 'volume', 'adj_close',
        'symbol', 'source', 'fetch_timestamp', 'date', 'Date',
        'target', 'target_return', 'target_direction', 'target_binary',
        'dividends', 'stock splits', 'capital gains'  # Additional metadata columns
    ]
    feature_cols = [c for c in sample_df.columns if c not in exclude_cols]
    n_features = len(feature_cols)
    
    print(f"[OK] Feature count: {n_features}")
    print(f"[OK] Training symbols: {len(feature_dict)}")
    
    # Validate feature store has sufficient data
    valid_symbols = 0
    for symbol, df in feature_dict.items():
        if len(df) >= 2:  # Need at least 2 rows for historical return calculation
            valid_symbols += 1
    
    if valid_symbols == 0:
        print("[ERROR] No valid symbols found in feature store!")
        print("Each symbol needs at least 2 rows of data")
        return
    print(f"[OK] Valid symbols: {valid_symbols}/{len(feature_dict)}")
    
    # Create agent based on type
    print(f"\n[2] Creating {args.agent.upper()} agent...")
    if args.agent == "linucb":
        agent = LinUCBAgent(n_features=n_features, alpha=1.0, model_dir="./models")
        model_file = "linucb_agent.pkl"
    elif args.agent == "thompson":
        agent = ThompsonSamplingAgent(n_features=n_features, model_dir="./models")
        model_file = "thompson_agent.pkl"
    else:  # dqn
        import os
        device = "cuda" if os.getenv("GPU_DEVICE", "cpu") != "cpu" else "cpu"
        agent = DQNAgent(
            state_dim=n_features,
            action_dim=3,
            device=device,
            model_dir="./models"
        )
        model_file = "dqn_agent.pt"
    
    # Create trainer with proper parameters
    print("\n[3] Creating RL trainer...")
    trainer = RLTrainer(agent, feature_dict, agent_type=args.agent)
    
    try:
        # Train the agent
        print(f"\n[4] Training {args.agent.upper()} agent...")
        print(f"Training for {args.rounds} rounds with top-{args.top_k} selection...")
        
        if args.agent in ["linucb", "thompson"]:
            # Bandit training
            stats = trainer.train_bandit(
                n_rounds=args.rounds, 
                top_k=min(args.top_k, valid_symbols),
                horizon=1
            )
            
            print(f"\n[5] Training Results:")
            print(f"  Cumulative Reward: {stats['cumulative_reward']:.4f}")
            print(f"  Average Reward: {stats['avg_reward']:.4f}")
            print(f"  Total Rounds: {stats['total_rounds']}")
            
            # Show reward distribution
            if 'rewards_per_round' in stats and stats['rewards_per_round']:
                rewards = stats['rewards_per_round']
                print(f"  Reward Range: [{min(rewards):.2f}, {max(rewards):.2f}]")
                print(f"  Reward Std Dev: {np.std(rewards):.4f}")
        else:
            # DQN training
            stats = trainer.train_dqn(
                n_episodes=args.rounds,
                max_steps=min(50, valid_symbols),
                horizon=1
            )
            
            print(f"\n[5] Training Results:")
            print(f"  Average Episode Reward: {stats['avg_reward']:.4f}")
            print(f"  Total Episodes: {stats['total_episodes']}")
            
            if 'episode_rewards' in stats and stats['episode_rewards']:
                rewards = stats['episode_rewards']
                print(f"  Reward Range: [{min(rewards):.2f}, {max(rewards):.2f}]")
        
        # Save the agent
        print(f"\n[6] Saving {args.agent.upper()} agent...")
        agent.save(model_file)
        print(f"[OK] Agent saved to: ./models/{model_file}")
        
        # Evaluate the agent
        print("\n[7] Evaluating agent...")
        eval_metrics = trainer.evaluate(top_k=min(args.top_k, valid_symbols))
        
        print(f"\n[8] Evaluation Results:")
        print(f"  Mean Return: {eval_metrics['mean_return']:.4f}")
        print(f"  Sharpe Proxy: {eval_metrics['sharpe_proxy']:.4f}")
        print(f"  Win Rate: {eval_metrics['win_rate']:.2%}")
        
        if eval_metrics.get('top_symbols'):
            print(f"  Top Symbols: {', '.join(eval_metrics['top_symbols'][:5])}")
        
        # Save evaluation results
        eval_path = Path("logs") / f"evaluation_{args.agent}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        eval_path.parent.mkdir(exist_ok=True)
        
        eval_data = {
            'agent_type': args.agent,
            'training_rounds': args.rounds,
            'training_stats': stats,
            'evaluation_metrics': eval_metrics,
            'n_features': n_features,
            'valid_symbols': valid_symbols,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(eval_path, 'w') as f:
            json.dump(eval_data, f, indent=2)
        
        print(f"[OK] Evaluation saved: {eval_path}")
        
        # Test feedback system if requested
        if args.test_feedback:
            print("\n[9] Testing feedback system...")
            test_feedback_system(agent, trainer, feature_dict, feature_cols)
        
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "="*60)
    print("RL AGENT TRAINING COMPLETE!")
    print("="*60)
    print("Results Summary:")
    print(f"  Agent Type: {args.agent.upper()}")
    print(f"  Training Rounds: {args.rounds}")
    print(f"  Features Used: {n_features}")
    print(f"  Valid Symbols: {valid_symbols}")
    print(f"  Mean Return: {eval_metrics['mean_return']:.4f}")
    print(f"  Win Rate: {eval_metrics['win_rate']:.2%}")
    print()
    print("Next steps:")
    print("  1. Start API server: python api/server.py")
    print("  2. Test predictions via Postman")
    print("  3. Provide feedback to improve agent")
    print("  4. Monitor logs/evaluation_*.json for performance")

def test_feedback_system(agent, trainer, feature_dict, feature_cols):
    """Test the feedback system with sample data"""
    print("Testing feedback integration...")
    
    # Get a sample symbol for testing
    sample_symbol = next(iter(feature_dict.keys()))
    sample_df = feature_dict[sample_symbol]
    
    if len(sample_df) < 2:
        print(f"[WARNING] {sample_symbol} has insufficient data for feedback test")
        return
    
    # Get latest features
    features = sample_df[feature_cols].iloc[-1].values
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Test positive feedback
    print(f"  Testing positive feedback for {sample_symbol}...")
    try:
        adjustment = trainer.incorporate_feedback(
            symbol=sample_symbol,
            predicted_action="long",
            user_feedback="correct",
            confidence=0.8
        )
        print(f"    [OK] Positive feedback adjustment: {adjustment:.4f}")
    except Exception as e:
        print(f"    [ERROR] Positive feedback failed: {e}")
    
    # Test negative feedback
    print(f"  Testing negative feedback for {sample_symbol}...")
    try:
        adjustment = trainer.incorporate_feedback(
            symbol=sample_symbol,
            predicted_action="short",
            user_feedback="incorrect",
            confidence=0.7
        )
        print(f"    [OK] Negative feedback adjustment: {adjustment:.4f}")
    except Exception as e:
        print(f"    [ERROR] Negative feedback failed: {e}")
    
    # Get feedback stats
    try:
        stats = trainer.get_feedback_stats()
        print(f"    [OK] Feedback stats: {stats['total_feedback']} total, {stats['correct_rate']:.2%} correct rate")
    except Exception as e:
        print(f"    [ERROR] Feedback stats failed: {e}")
    
    print("[OK] Feedback system test complete!")

if __name__ == "__main__":
    main()
