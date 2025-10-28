"""
Lightweight Reinforcement Learning Agent - FIXED VERSION
Critical fixes:
1. Reward calculation now uses PAST returns (not future NaN values)
2. Feedback properly integrates with agent learning
3. Proper feature alignment and validation
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
from pathlib import Path
import joblib
import json
import logging
from typing import Dict, List, Tuple, Optional
import random
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class FeedbackMemory:
    """Manages user feedback for RL agent learning"""
    
    def __init__(self, memory_file: str = "feedback_memory.json", log_file: str = "logs/feedback_loop.json"):
        self.memory_file = Path(memory_file)
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.feedback_data = self._load_feedback()
    
    def _load_feedback(self) -> Dict:
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load feedback memory: {e}")
        return {"feedback_history": [], "reward_adjustments": {}}
    
    def _save_feedback(self):
        try:
            with open(self.memory_file, 'w') as f:
                json.dump(self.feedback_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save feedback memory: {e}")
    
    def add_feedback(self, symbol: str, predicted_action: str, user_feedback: str, 
                    confidence: float = 0.0, features: Dict = None):
        feedback_entry = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "predicted_action": predicted_action,
            "user_feedback": user_feedback,
            "confidence": confidence,
            "features": features or {}
        }
        
        self.feedback_data["feedback_history"].append(feedback_entry)
        reward_adjustment = self._calculate_reward_adjustment(predicted_action, user_feedback, confidence)
        
        if symbol not in self.feedback_data["reward_adjustments"]:
            self.feedback_data["reward_adjustments"][symbol] = []
        
        self.feedback_data["reward_adjustments"][symbol].append({
            "timestamp": feedback_entry["timestamp"],
            "adjustment": reward_adjustment,
            "action": predicted_action,
            "feedback": user_feedback
        })
        
        self._log_feedback(feedback_entry, reward_adjustment)
        self._save_feedback()
        
        logger.info(f"Feedback recorded for {symbol}: {predicted_action} -> {user_feedback} (adjustment: {reward_adjustment})")
        return reward_adjustment
    
    def _calculate_reward_adjustment(self, predicted_action: str, user_feedback: str, confidence: float) -> float:
        if user_feedback.lower() == "correct":
            return confidence * 10.0  # Increased from 0.1
        elif user_feedback.lower() == "incorrect":
            return -confidence * 10.0  # Increased from -0.1
        return 0.0
    
    def _log_feedback(self, feedback_entry: Dict, reward_adjustment: float):
        log_entry = {
            "timestamp": feedback_entry["timestamp"],
            "symbol": feedback_entry["symbol"],
            "predicted_action": feedback_entry["predicted_action"],
            "user_feedback": feedback_entry["user_feedback"],
            "confidence": feedback_entry["confidence"],
            "reward_adjustment": reward_adjustment,
            "action": "feedback_recorded"
        }
        
        try:
            if self.log_file.exists():
                with open(self.log_file, 'r') as f:
                    logs = json.load(f)
            else:
                logs = {"feedback_logs": []}
            
            logs["feedback_logs"].append(log_entry)
            
            if len(logs["feedback_logs"]) > 1000:
                logs["feedback_logs"] = logs["feedback_logs"][-1000:]
            
            with open(self.log_file, 'w') as f:
                json.dump(logs, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to log feedback: {e}")
    
    def get_recent_feedback(self, symbol: str = None, limit: int = 10) -> List[Dict]:
        history = self.feedback_data["feedback_history"]
        if symbol:
            history = [entry for entry in history if entry["symbol"] == symbol]
        return history[-limit:]
    
    def get_reward_adjustments(self, symbol: str) -> List[float]:
        if symbol in self.feedback_data["reward_adjustments"]:
            return [adj["adjustment"] for adj in self.feedback_data["reward_adjustments"][symbol]]
        return []
    
    def get_feedback_stats(self) -> Dict:
        history = self.feedback_data["feedback_history"]
        if not history:
            return {"total_feedback": 0, "correct_rate": 0.0, "symbols": []}
        
        total = len(history)
        correct = sum(1 for entry in history if entry["user_feedback"].lower() == "correct")
        symbols = list(set(entry["symbol"] for entry in history))
        
        return {
            "total_feedback": total,
            "correct_rate": correct / total if total > 0 else 0.0,
            "symbols": symbols,
            "recent_feedback": history[-5:] if history else []
        }


class LinUCBAgent:
    """Linear Upper Confidence Bound (LinUCB) Contextual Bandit"""
    
    def __init__(self, n_features: int, alpha: float = 1.0, model_dir: str = "./models"):
        self.n_features = n_features
        self.alpha = alpha
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.arms = {}
        self.total_reward = 0
        self.n_rounds = 0
        self.feedback_memory = FeedbackMemory()
        
    def incorporate_feedback(self, symbol: str, predicted_action: str, user_feedback: str, 
                           confidence: float = 0.0, features: np.ndarray = None):
        reward_adjustment = self.feedback_memory.add_feedback(
            symbol, predicted_action, user_feedback, confidence, 
            features.tolist() if features is not None else None
        )
        
        if features is not None and symbol in self.arms:
            if user_feedback.lower() == "correct":
                reward = confidence * 10.0
            elif user_feedback.lower() == "incorrect":
                reward = -confidence * 10.0
            else:
                reward = 0.0
            
            self.update(symbol, features, reward)
            logger.info(f"LinUCB updated for {symbol} with reward {reward}")
        
        return reward_adjustment
    
    def get_feedback_stats(self) -> Dict:
        return self.feedback_memory.get_feedback_stats()
    
    def _init_arm(self, arm_id: str):
        self.arms[arm_id] = {
            'A': np.identity(self.n_features),
            'b': np.zeros(self.n_features)
        }
    
    def select_action(self, context: np.ndarray, arm_id: str) -> Tuple[float, float]:
        if arm_id not in self.arms:
            self._init_arm(arm_id)
        
        if len(context) != self.n_features:
            logger.warning(f"Context size mismatch: expected {self.n_features}, got {len(context)}")
            context = np.pad(context, (0, max(0, self.n_features - len(context))), mode='constant')[:self.n_features]
        
        context = np.nan_to_num(context, nan=0.0, posinf=0.0, neginf=0.0)
        
        A = self.arms[arm_id]['A']
        b = self.arms[arm_id]['b']
        
        try:
            A_inv = np.linalg.inv(A + np.eye(self.n_features) * 1e-6)
            theta = A_inv.dot(b)
            confidence = self.alpha * np.sqrt(context.dot(A_inv).dot(context))
            score = theta.dot(context) + confidence
        except np.linalg.LinAlgError:
            logger.warning(f"Matrix inversion failed for {arm_id}, returning default")
            score = 0.0
            confidence = 1.0
        
        return float(score), float(confidence)
    
    def update(self, arm_id: str, context: np.ndarray, reward: float):
        if arm_id not in self.arms:
            self._init_arm(arm_id)
        
        if len(context) != self.n_features:
            context = np.pad(context, (0, max(0, self.n_features - len(context))), mode='constant')[:self.n_features]
        
        context = np.nan_to_num(context, nan=0.0, posinf=0.0, neginf=0.0)
        reward = float(np.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0))
        
        self.arms[arm_id]['A'] += np.outer(context, context)
        self.arms[arm_id]['b'] += reward * context
        
        self.total_reward += reward
        self.n_rounds += 1
    
    def rank_symbols(self, contexts: Dict[str, np.ndarray], top_k: Optional[int] = None) -> List[Tuple[str, float, float]]:
        rankings = []
        
        for symbol, context in contexts.items():
            try:
                score, confidence = self.select_action(context, symbol)
                rankings.append((symbol, score, confidence))
            except Exception as e:
                logger.error(f"Error ranking {symbol}: {e}")
                continue
        
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        if top_k:
            rankings = rankings[:top_k]
        
        return rankings
    
    def save(self, filename: str = "linucb_agent.pkl"):
        save_path = self.model_dir / filename
        state = {
            'arms': self.arms,
            'n_features': self.n_features,
            'alpha': self.alpha,
            'total_reward': self.total_reward,
            'n_rounds': self.n_rounds
        }
        joblib.dump(state, save_path)
        logger.info(f"LinUCB agent saved to {save_path}")
    
    def load(self, filename: str = "linucb_agent.pkl"):
        load_path = self.model_dir / filename
        if not load_path.exists():
            raise FileNotFoundError(f"Model not found: {load_path}")
        
        state = joblib.load(load_path)
        self.arms = state['arms']
        self.n_features = state['n_features']
        self.alpha = state['alpha']
        self.total_reward = state['total_reward']
        self.n_rounds = state['n_rounds']
        logger.info(f"LinUCB agent loaded from {load_path}")


class ThompsonSamplingAgent:
    """Thompson Sampling Contextual Bandit"""
    
    def __init__(self, n_features: int, lambda_: float = 1.0, v: float = 1.0, model_dir: str = "./models"):
        self.n_features = n_features
        self.lambda_ = lambda_
        self.v = v
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.arms = {}
        self.total_reward = 0
        self.n_rounds = 0
        self.feedback_memory = FeedbackMemory()
        
        logger.info(f"Thompson Sampling Agent initialized with {n_features} features")
    
    def incorporate_feedback(self, symbol: str, predicted_action: str, user_feedback: str, 
                           confidence: float = 0.0, features: np.ndarray = None):
        reward_adjustment = self.feedback_memory.add_feedback(
            symbol, predicted_action, user_feedback, confidence, 
            features.tolist() if features is not None else None
        )
        
        if features is not None and symbol in self.arms:
            if user_feedback.lower() == "correct":
                reward = confidence * 10.0
            elif user_feedback.lower() == "incorrect":
                reward = -confidence * 10.0
            else:
                reward = 0.0
            
            self.update(symbol, features, reward)
            logger.info(f"Thompson Sampling updated for {symbol} with reward {reward}")
        
        return reward_adjustment
    
    def get_feedback_stats(self) -> Dict:
        return self.feedback_memory.get_feedback_stats()
    
    def _init_arm(self, arm_id: str):
        self.arms[arm_id] = {
            'B': self.lambda_ * np.identity(self.n_features),
            'mu': np.zeros(self.n_features),
            'f': np.zeros(self.n_features)
        }
    
    def select_action(self, context: np.ndarray, arm_id: str) -> Tuple[float, float]:
        if arm_id not in self.arms:
            self._init_arm(arm_id)
        
        if len(context) != self.n_features:
            context = np.pad(context, (0, max(0, self.n_features - len(context))), mode='constant')[:self.n_features]
        
        context = np.nan_to_num(context, nan=0.0, posinf=0.0, neginf=0.0)
        
        B = self.arms[arm_id]['B']
        f = self.arms[arm_id]['f']
        
        try:
            B_inv = np.linalg.inv(B + np.eye(self.n_features) * 1e-6)
            mu = B_inv.dot(f)
            
            # Ensure covariance matrix is positive definite
            cov_matrix = self.v * B_inv
            cov_matrix = (cov_matrix + cov_matrix.T) / 2  # Make symmetric
            cov_matrix += np.eye(self.n_features) * 1e-6  # Add regularization
            
            # Check if matrix is positive definite
            try:
                np.linalg.cholesky(cov_matrix)
            except np.linalg.LinAlgError:
                # If not positive definite, use identity matrix
                cov_matrix = np.eye(self.n_features) * 0.1
            
            theta_sample = np.random.multivariate_normal(mu, cov_matrix)
            score = theta_sample.dot(context)
            confidence = np.sqrt(context.dot(B_inv).dot(context))
        except (np.linalg.LinAlgError, ValueError) as e:
            logger.warning(f"Posterior sampling failed for {arm_id}: {e}")
            score = 0.0
            confidence = 1.0
        
        return float(score), float(confidence)
    
    def update(self, arm_id: str, context: np.ndarray, reward: float):
        if arm_id not in self.arms:
            self._init_arm(arm_id)
        
        if len(context) != self.n_features:
            context = np.pad(context, (0, max(0, self.n_features - len(context))), mode='constant')[:self.n_features]
        
        context = np.nan_to_num(context, nan=0.0, posinf=0.0, neginf=0.0)
        reward = float(np.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0))
        
        self.arms[arm_id]['B'] += np.outer(context, context)
        self.arms[arm_id]['f'] += reward * context
        
        try:
            B_inv = np.linalg.inv(self.arms[arm_id]['B'] + np.eye(self.n_features) * 1e-6)
            self.arms[arm_id]['mu'] = B_inv.dot(self.arms[arm_id]['f'])
        except np.linalg.LinAlgError:
            pass
        
        self.total_reward += reward
        self.n_rounds += 1
    
    def rank_symbols(self, contexts: Dict[str, np.ndarray], top_k: Optional[int] = None, n_samples: int = 100) -> List[Tuple[str, float, float]]:
        rankings = {}
        
        for _ in range(n_samples):
            for symbol, context in contexts.items():
                try:
                    score, confidence = self.select_action(context, symbol)
                    if symbol not in rankings:
                        rankings[symbol] = {'scores': [], 'confidences': []}
                    rankings[symbol]['scores'].append(score)
                    rankings[symbol]['confidences'].append(confidence)
                except Exception as e:
                    logger.error(f"Error sampling {symbol}: {e}")
                    continue
        
        results = []
        for symbol, data in rankings.items():
            if data['scores']:
                avg_score = np.mean(data['scores'])
                avg_confidence = np.mean(data['confidences'])
                results.append((symbol, float(avg_score), float(avg_confidence)))
        
        results.sort(key=lambda x: x[1], reverse=True)
        
        if top_k:
            results = results[:top_k]
        
        return results
    
    def save(self, filename: str = "thompson_agent.pkl"):
        save_path = self.model_dir / filename
        state = {
            'arms': self.arms,
            'n_features': self.n_features,
            'lambda_': self.lambda_,
            'v': self.v,
            'total_reward': self.total_reward,
            'n_rounds': self.n_rounds
        }
        joblib.dump(state, save_path)
        logger.info(f"Thompson Sampling agent saved to {save_path}")
    
    def load(self, filename: str = "thompson_agent.pkl"):
        load_path = self.model_dir / filename
        if not load_path.exists():
            raise FileNotFoundError(f"Model not found: {load_path}")
        
        state = joblib.load(load_path)
        self.arms = state['arms']
        self.n_features = state['n_features']
        self.lambda_ = state['lambda_']
        self.v = state['v']
        self.total_reward = state['total_reward']
        self.n_rounds = state['n_rounds']
        logger.info(f"Thompson Sampling agent loaded from {load_path}")


class ReplayBuffer:
    """Experience replay buffer for DQN"""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
        next_state = np.nan_to_num(next_state, nan=0.0, posinf=0.0, neginf=0.0)
        reward = float(np.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0))
        self.buffer.append(Experience(state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> List[Experience]:
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)


class DQNNetwork(nn.Module):
    """Simple DQN network for Q-value estimation"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(DQNNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x):
        return self.network(x)


class DQNAgent:
    """Deep Q-Network Agent for ranking"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int = 3,
        hidden_dim: int = 128,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_capacity: int = 10000,
        batch_size: int = 32,
        device: str = 'cpu',
        model_dir: str = "./models"
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.q_network = DQNNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network = DQNNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        self.total_reward = 0
        self.episode_rewards = []
        self.losses = []
        self.feedback_memory = FeedbackMemory()
        
        logger.info(f"DQN Agent initialized on {self.device}")
    
    def incorporate_feedback(self, symbol: str, predicted_action: str, user_feedback: str, 
                           confidence: float = 0.0, features: np.ndarray = None):
        reward_adjustment = self.feedback_memory.add_feedback(
            symbol, predicted_action, user_feedback, confidence, 
            features.tolist() if features is not None else None
        )
        
        if features is not None:
            if user_feedback.lower() == "correct":
                reward = confidence * 10.0
            elif user_feedback.lower() == "incorrect":
                reward = -confidence * 10.0
            else:
                reward = 0.0
            
            action_idx = {"long": 2, "short": 0, "hold": 1}.get(predicted_action.lower(), 1)
            self.replay_buffer.push(features, action_idx, reward, features, True)
            self.train_step()
            
            logger.info(f"DQN feedback recorded for {symbol}: {predicted_action} -> {user_feedback} (reward: {reward})")
        
        return reward_adjustment
    
    def get_feedback_stats(self) -> Dict:
        return self.feedback_memory.get_feedback_stats()
    
    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
        
        if not evaluate and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax(dim=1).item()
    
    def get_action_scores(self, state: np.ndarray) -> np.ndarray:
        state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.cpu().numpy()[0]
    
    def store_transition(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train_step(self) -> Optional[float]:
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        batch = self.replay_buffer.sample(self.batch_size)
        
        states = torch.FloatTensor([e.state for e in batch]).to(self.device)
        actions = torch.LongTensor([e.action for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in batch]).to(self.device)
        dones = torch.FloatTensor([e.done for e in batch]).to(self.device)
        
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def rank_symbols(self, contexts: Dict[str, np.ndarray], top_k: Optional[int] = None) -> List[Tuple[str, float, int, float]]:
        rankings = []
        
        for symbol, context in contexts.items():
            try:
                q_values = self.get_action_scores(context)
                action = int(q_values.argmax())
                score = float(q_values[action])
                
                q_exp = np.exp(q_values - q_values.max())
                confidence = float(q_exp[action] / q_exp.sum())
                
                rankings.append((symbol, score, action, confidence))
            except Exception as e:
                logger.error(f"Error ranking {symbol}: {e}")
                continue
        
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        if top_k:
            rankings = rankings[:top_k]
        
        return rankings
    
    def save(self, filename: str = "dqn_agent.pt"):
        save_path = self.model_dir / filename
        checkpoint = {
            'q_network_state': self.q_network.state_dict(),
            'target_network_state': self.target_network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'total_reward': self.total_reward,
            'episode_rewards': self.episode_rewards,
            'losses': self.losses,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim
        }
        torch.save(checkpoint, save_path)
        logger.info(f"DQN agent saved to {save_path}")
    
    def load(self, filename: str = "dqn_agent.pt"):
        load_path = self.model_dir / filename
        if not load_path.exists():
            raise FileNotFoundError(f"Model not found: {load_path}")
        
        checkpoint = torch.load(load_path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state'])
        self.target_network.load_state_dict(checkpoint['target_network_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.epsilon = checkpoint['epsilon']
        self.total_reward = checkpoint['total_reward']
        self.episode_rewards = checkpoint['episode_rewards']
        self.losses = checkpoint['losses']
        logger.info(f"DQN agent loaded from {load_path}")


class RLTrainer:
    """Training loop for RL agents - FIXED VERSION"""
    
    def __init__(self, agent, feature_store: Dict[str, pd.DataFrame], agent_type: str = "linucb"):
        self.agent = agent
        self.feature_store = feature_store
        self.agent_type = agent_type
        self.training_history = []
        self.feedback_memory = FeedbackMemory()
        
        logger.info(f"RL Trainer initialized for {agent_type} agent")
        logger.info(f"Feature store contains {len(feature_store)} symbols")
    
    def incorporate_feedback(self, symbol: str, predicted_action: str, user_feedback: str, confidence: float = 0.0):
        features = None
        if symbol in self.feature_store:
            df = self.feature_store[symbol]
            if not df.empty:
                feature_cols = self._get_feature_columns(df)
                if feature_cols:
                    features = df[feature_cols].iloc[-1].values
                    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        if hasattr(self.agent, 'incorporate_feedback'):
            return self.agent.incorporate_feedback(symbol, predicted_action, user_feedback, confidence, features)
        else:
            return self.feedback_memory.add_feedback(symbol, predicted_action, user_feedback, confidence, 
                                                   features.tolist() if features is not None else None)
    
    def get_feedback_stats(self) -> Dict:
        if hasattr(self.agent, 'get_feedback_stats'):
            return self.agent.get_feedback_stats()
        else:
            return self.feedback_memory.get_feedback_stats()
    
    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Extract feature columns (exclude OHLCV, metadata, targets, and timestamp columns)"""
        exclude_cols = [
            'open', 'high', 'low', 'close', 'volume', 'adj_close',
            'symbol', 'source', 'fetch_timestamp', 'date', 'Date',
            'target', 'target_return', 'target_direction', 'target_binary',
            'dividends', 'stock splits', 'capital gains'  # Additional metadata columns
        ]
        return [col for col in df.columns if col not in exclude_cols]
    
    def _compute_reward(self, symbol: str, action: int, horizon: int = 1) -> float:
        """
        FIXED: Compute reward based on HISTORICAL returns (not future NaN)
        
        Uses the PAST price movement to calculate reward for training.
        For bandit agents: positive return = positive reward
        For DQN: reward based on action-return alignment
        """
        df = self.feature_store[symbol]
        
        # Check if we have enough data
        if len(df) < 2:
            return 0.0
        
        # Use HISTORICAL return (past price movement)
        # This is the return that JUST HAPPENED (from previous row to current)
        current_close = df['close'].iloc[-1] if 'close' in df.columns else df['Close'].iloc[-1]
        previous_close = df['close'].iloc[-2] if 'close' in df.columns else df['Close'].iloc[-2]
        
        # Calculate the actual return that occurred
        historical_return = (current_close - previous_close) / previous_close
        
        # Handle NaN
        if pd.isna(historical_return):
            return 0.0
        
        if self.agent_type == "dqn":
            # DQN: reward based on whether action aligns with what happened
            if action == 2:  # long
                reward = historical_return if historical_return > 0 else -abs(historical_return)
            elif action == 0:  # short
                reward = -historical_return if historical_return < 0 else -abs(historical_return)
            else:  # hold
                reward = 0.01 if abs(historical_return) < 0.01 else -abs(historical_return) * 0.5
        else:
            # Bandit: direct return as reward
            reward = historical_return
        
        # Scale reward to reasonable range (multiply by 100 to get percentage points)
        return float(reward * 100)
    
    def train_bandit(self, n_rounds: int = 100, top_k: int = 20, horizon: int = 1) -> Dict:
        """Train contextual bandit agent"""
        logger.info(f"Training {self.agent_type} agent for {n_rounds} rounds")
        
        cumulative_reward = 0
        rewards_per_round = []
        
        for round_num in range(n_rounds):
            # Prepare contexts for all symbols
            contexts = {}
            for symbol, df in self.feature_store.items():
                if len(df) > 1:  # Need at least 2 rows for historical return
                    feature_cols = self._get_feature_columns(df)
                    context = df[feature_cols].iloc[-1].values
                    context = np.nan_to_num(context, nan=0.0, posinf=0.0, neginf=0.0)
                    contexts[symbol] = context
            
            if not contexts:
                logger.warning("No valid contexts available")
                break
            
            # Rank and select top-k
            rankings = self.agent.rank_symbols(contexts, top_k=top_k)
            
            # Update based on observed rewards
            round_reward = 0
            for symbol, score, confidence in rankings:
                reward = self._compute_reward(symbol, action=2, horizon=horizon)
                self.agent.update(symbol, contexts[symbol], reward)
                round_reward += reward
            
            cumulative_reward += round_reward
            rewards_per_round.append(round_reward)
            
            if (round_num + 1) % 10 == 0:
                avg_reward = np.mean(rewards_per_round[-10:])
                logger.info(f"Round {round_num + 1}/{n_rounds} - Avg Reward: {avg_reward:.4f}")
        
        stats = {
            'cumulative_reward': cumulative_reward,
            'avg_reward': cumulative_reward / n_rounds if n_rounds > 0 else 0,
            'rewards_per_round': rewards_per_round,
            'total_rounds': n_rounds
        }
        
        self.training_history.append(stats)
        logger.info(f"Training complete. Avg reward: {stats['avg_reward']:.4f}")
        
        return stats
    
    def train_dqn(self, n_episodes: int = 100, max_steps: int = 50, target_update_freq: int = 10, horizon: int = 1) -> Dict:
        """Train DQN agent"""
        logger.info(f"Training DQN agent for {n_episodes} episodes")
        
        episode_rewards = []
        losses = []
        
        for episode in range(n_episodes):
            episode_reward = 0
            
            # Sample random symbols for this episode
            symbols = list(self.feature_store.keys())
            if not symbols:
                logger.warning("No symbols available for training")
                break
            
            # Filter symbols with enough data
            valid_symbols = [s for s in symbols if len(self.feature_store[s]) > 1]
            if not valid_symbols:
                logger.warning("No valid symbols with sufficient data")
                break
            
            episode_symbols = random.sample(valid_symbols, min(max_steps, len(valid_symbols)))
            
            for step, symbol in enumerate(episode_symbols):
                if step >= max_steps:
                    break
                
                try:
                    df = self.feature_store[symbol]
                    if df.empty or len(df) < 2:
                        continue
                    
                    # Get features
                    feature_cols = self._get_feature_columns(df)
                    if not feature_cols:
                        continue
                    
                    state = df[feature_cols].iloc[-1].values
                    state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    # Select action
                    action = self.agent.select_action(state)
                    
                    # Compute reward (FIXED: uses historical return)
                    reward = self._compute_reward(symbol, action, horizon)
                    
                    # Store transition
                    next_state = state  # Simplified
                    done = step == len(episode_symbols) - 1
                    
                    self.agent.store_transition(state, action, reward, next_state, done)
                    
                    # Train
                    loss = self.agent.train_step()
                    if loss is not None:
                        losses.append(loss)
                    
                    episode_reward += reward
                    
                except Exception as e:
                    logger.error(f"Error in episode {episode}, step {step}: {e}")
                    continue
            
            episode_rewards.append(episode_reward)
            
            # Update target network
            if episode % target_update_freq == 0:
                self.agent.update_target_network()
            
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                logger.info(f"Episode {episode + 1}/{n_episodes} - Avg Reward: {avg_reward:.4f}")
        
        stats = {
            'avg_reward': np.mean(episode_rewards) if episode_rewards else 0,
            'episode_rewards': episode_rewards,
            'losses': losses,
            'total_episodes': n_episodes
        }
        
        self.training_history.append(stats)
        logger.info(f"DQN training complete. Avg reward: {stats['avg_reward']:.4f}")
        
        return stats
    
    def evaluate(self, top_k: int = 20) -> Dict:
        """Evaluate agent performance on feature store"""
        logger.info(f"Evaluating {self.agent_type} agent")
        
        # Prepare contexts for all symbols
        contexts = {}
        symbol_data = {}
        
        for symbol, df in self.feature_store.items():
            if df.empty or len(df) < 2:
                continue
            
            try:
                feature_cols = self._get_feature_columns(df)
                if not feature_cols:
                    continue
                
                context = df[feature_cols].iloc[-1].values
                context = np.nan_to_num(context, nan=0.0, posinf=0.0, neginf=0.0)
                
                contexts[symbol] = context
                symbol_data[symbol] = df.iloc[-1]
                
            except Exception as e:
                logger.error(f"Error preparing {symbol}: {e}")
                continue
        
        if not contexts:
            logger.warning("No valid contexts for evaluation")
            return {
                'mean_return': 0.0,
                'sharpe_proxy': 0.0,
                'win_rate': 0.0,
                'top_symbols': []
            }
        
        # Rank symbols
        try:
            if hasattr(self.agent, 'action_dim'):
                ranked = self.agent.rank_symbols(contexts, top_k=top_k)
            else:
                ranked = self.agent.rank_symbols(contexts, top_k=top_k)
        except Exception as e:
            logger.error(f"Ranking failed: {e}")
            return {
                'mean_return': 0.0,
                'sharpe_proxy': 0.0,
                'win_rate': 0.0,
                'top_symbols': []
            }
        
        # Evaluate top symbols using HISTORICAL returns
        returns = []
        top_symbols = []
        
        for item in ranked[:top_k]:
            try:
                if hasattr(self.agent, 'action_dim'):
                    symbol, score, action_idx, confidence = item
                else:
                    symbol, score, confidence = item
                
                # Get actual historical return
                if symbol in self.feature_store:
                    df = self.feature_store[symbol]
                    if len(df) >= 2:
                        close_col = 'close' if 'close' in df.columns else 'Close'
                        current = df[close_col].iloc[-1]
                        previous = df[close_col].iloc[-2]
                        actual_return = (current - previous) / previous
                        
                        if not pd.isna(actual_return):
                            returns.append(actual_return)
                            top_symbols.append(symbol)
                
            except Exception as e:
                logger.error(f"Error evaluating {symbol}: {e}")
                continue
        
        if not returns:
            logger.warning("No valid returns for evaluation")
            return {
                'mean_return': 0.0,
                'sharpe_proxy': 0.0,
                'win_rate': 0.0,
                'top_symbols': []
            }
        
        # Calculate metrics
        mean_return = np.mean(returns)
        sharpe_proxy = mean_return / (np.std(returns) + 1e-10)
        win_rate = np.mean([r > 0 for r in returns])
        
        metrics = {
            'mean_return': float(mean_return),
            'sharpe_proxy': float(sharpe_proxy),
            'win_rate': float(win_rate),
            'top_symbols': top_symbols[:10]
        }
        
        logger.info(f"Evaluation complete:")
        logger.info(f"  Mean Return: {metrics['mean_return']:.4f}")
        logger.info(f"  Sharpe Proxy: {metrics['sharpe_proxy']:.4f}")
        logger.info(f"  Win Rate: {metrics['win_rate']:.2%}")
        
        return metrics