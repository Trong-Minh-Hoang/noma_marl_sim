# agents/d3qn_agent.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
from typing import List, Tuple, Dict

# Prioritized Experience Replay
from utils.prioritized_replay import PrioritizedReplayBuffer

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class DuelingDQN(nn.Module):
    """
    Dueling DQN architecture (Eq. 15 in paper).
    Matches: Section III.C, "Dueling Double DQN Architecture".
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [256, 256, 128, 128]):
        super(DuelingDQN, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Shared backbone
        layers = []
        input_dim = state_dim
        for hidden in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden))
            layers.append(nn.BatchNorm1d(hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_dim = hidden
        self.backbone = nn.Sequential(*layers)
        
        # Value stream V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dims[-1], 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Advantage stream A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dims[-1], 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        # Eq. (15): Q(s,a) = V(s) + [A(s,a) - mean(A(s,a))]
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

class D3QNAgent:
    """
    Independent Q-Learning (IQL) agent with Dueling Double DQN (D3QN-IQL).
    Matches: Section III.D, Algorithm 1, and convergence mechanisms.
    """
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 agent_id: int,
                 lr: float = 1e-3,
                 gamma: float = 0.9995,
                 tau: float = 0.005,        # Polyak averaging (Section III.D)
                 buffer_size: int = 100000,
                 batch_size: int = 128,     # Updated from 64 → 128
                 grad_max_norm: float = 10.0,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay_episodes: int = 30000):
        
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.grad_max_norm = grad_max_norm
        self.epsilon_decay_episodes = epsilon_decay_episodes
        
        # Networks
        self.q_net = DuelingDQN(state_dim, action_dim)
        self.target_q_net = DuelingDQN(state_dim, action_dim)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        
        # Prioritized Replay Buffer (Section III.D, Mechanism 1)
        self.replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=0.6)
        
        # Epsilon decay (Section III.D, Mechanism 3)
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        
        # Tracking for Figure 2
        self.entropy_history = []
        self.td_error_history = []
        self.episode_rewards = []
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net.to(self.device)
        self.target_q_net.to(self.device)
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """ε-greedy action selection (Algorithm 1, line 13–16)."""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        
        # Track policy entropy for Figure 2 (Mechanism: softmax over Q)
        temperature = 1.0  # Can be tuned
        probs = F.softmax(q_values / temperature, dim=1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1).mean().item()
        self.entropy_history.append(entropy)
        
        return q_values.max(1)[1].item()
    
    def store_transition(self, 
                        state: np.ndarray, 
                        action: int, 
                        reward: float, 
                        next_state: np.ndarray, 
                        done: bool):
        """Store experience in prioritized buffer."""
        self.replay_buffer.add(state, action, reward, next_state, done)
        self.episode_rewards.append(reward)
    
    def compute_reward_variance(self, window: int = 100) -> float:
        """Track reward variance every 100 episodes (Figure 2 requirement)."""
        if len(self.episode_rewards) < window:
            return np.var(self.episode_rewards) if self.episode_rewards else 0.0
        return np.var(self.episode_rewards[-window:])
    
    def update_epsilon(self, episode: int):
        """Linear decay (Section III.D, Mechanism 3)."""
        decay_rate = (self.epsilon_start - self.epsilon_end) / self.epsilon_decay_episodes
        self.epsilon = max(self.epsilon_end, self.epsilon_start - decay_rate * episode)
    
    def train(self) -> Tuple[float, float]:
        """Train one step (Algorithm 1, line 28–36)."""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0, 0.0
        
        # Sample batch with priorities
        batch, indices, weights = self.replay_buffer.sample(self.batch_size)
        weights = torch.FloatTensor(weights).to(self.device)
        
        state_batch = torch.FloatTensor(np.array([b.state for b in batch])).to(self.device)
        action_batch = torch.LongTensor([b.action for b in batch]).to(self.device)
        reward_batch = torch.FloatTensor([b.reward for b in batch]).to(self.device)
        next_state_batch = torch.FloatTensor(np.array([b.next_state for b in batch])).to(self.device)
        done_batch = torch.BoolTensor([b.done for b in batch]).to(self.device)
        
        # Current Q values
        current_q_values = self.q_net(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
        
        # Double DQN target: use main net to select, target net to evaluate
        with torch.no_grad():
            next_actions = self.q_net(next_state_batch).max(1)[1]
            next_q_values = self.target_q_net(next_state_batch).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q_values = reward_batch + (self.gamma * next_q_values * (~done_batch))
        
        # TD Error (for tracking and priority update)
        td_errors = target_q_values - current_q_values
        td_error_abs = torch.abs(td_errors)
        mean_td_error = td_error_abs.mean().item()
        self.td_error_history.append(mean_td_error)
        
        # Huber loss (smooth L1) with importance sampling weights
        loss = (weights * F.smooth_l1_loss(current_q_values, target_q_values, reduction='none')).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.grad_max_norm)  # Gradient clipping
        self.optimizer.step()
        
        # Update priorities
        self.replay_buffer.update_priorities(indices, td_error_abs.detach().cpu().numpy() + 1e-5)
        
        return loss.item(), mean_td_error
    
    def update_target_network(self):
        """Polyak averaging (Section III.D, Mechanism 2)."""
        for target_param, param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
    
    def get_metrics(self) -> Dict[str, float]:
        """Return metrics for Figure 2."""
        return {
            'policy_entropy': np.mean(self.entropy_history[-100:]) if self.entropy_history else 0.0,
            'td_error': np.mean(self.td_error_history[-100:]) if self.td_error_history else 0.0,
            'reward_variance': self.compute_reward_variance()
        }