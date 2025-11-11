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
                 tau: float = 0.005,
                 buffer_size: int = 100000,
                 batch_size: int = 128,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: int = 30000,
                 device: str = 'cpu'):
        """
        Initialize D3QN-IQL agent.
        
        Args:
            state_dim: Observation dimension
            action_dim: Action space size
            agent_id: Agent identifier
            lr: Initial learning rate
            gamma: Discount factor
            tau: Soft update coefficient (Polyak averaging)
            buffer_size: Replay buffer capacity
            batch_size: Training batch size
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Episodes for epsilon decay
            device: 'cpu' or 'cuda'
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.agent_id = agent_id
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = device
        
        # Exploration schedule
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start
        
        # Q-networks
        self.q_net = DuelingDQN(state_dim, action_dim).to(device)
        self.target_q_net = DuelingDQN(state_dim, action_dim).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.target_q_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.lr = lr
        
        # Replay buffer (Prioritized Experience Replay)
        self.replay_buffer = PrioritizedReplayBuffer(buffer_size)
        
        # Training statistics
        self.train_steps = 0
        self.episode_count = 0
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using ε-greedy policy.
        
        Args:
            state: Current state (observation)
            training: If True, use exploration; if False, use greedy
        
        Returns:
            Action index
        """
        if training and np.random.rand() < self.epsilon:
            # Exploration: random action
            return np.random.randint(self.action_dim)
        
        # Exploitation: greedy action
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_net(state_tensor)
            action = q_values.argmax(dim=1).item()
        
        return action
    
    def store_transition(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool):
        """Store transition in replay buffer."""
        transition = Transition(state, action, reward, next_state, done)
        self.replay_buffer.add(transition)
    
    def update_epsilon(self, episode: int):
        """
        Update exploration rate with linear decay.
        ε(t) = max(ε_end, ε_start - (ε_start - ε_end) * t / ε_decay)
        """
        decay_rate = (self.epsilon_start - self.epsilon_end) / self.epsilon_decay
        self.epsilon = max(self.epsilon_end, self.epsilon_start - decay_rate * episode)
    
    def update_target_network(self):
        """
        Soft update target network using Polyak averaging (Eq. 3.3).
        θ_target ← τ * θ + (1-τ) * θ_target
        """
        for param, target_param in zip(self.q_net.parameters(), 
                                       self.target_q_net.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
    
    def train_step(self) -> float:
        """
        Perform one training step using prioritized experience replay.
        
        Returns:
            TD loss value
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        # Sample batch with priorities
        transitions, indices, weights = self.replay_buffer.sample(self.batch_size)
        
        # Unpack transitions
        states = torch.FloatTensor(np.array([t.state for t in transitions])).to(self.device)
        actions = torch.LongTensor(np.array([t.action for t in transitions])).to(self.device)
        rewards = torch.FloatTensor(np.array([t.reward for t in transitions])).to(self.device)
        next_states = torch.FloatTensor(np.array([t.next_state for t in transitions])).to(self.device)
        dones = torch.FloatTensor(np.array([t.done for t in transitions])).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Current Q-values
        q_values = self.q_net(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Double DQN: use online network to select action, target network to evaluate
        with torch.no_grad():
            next_q_values_online = self.q_net(next_states)
            next_actions = next_q_values_online.argmax(dim=1)
            
            next_q_values_target = self.target_q_net(next_states)
            next_q_values = next_q_values_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            
            # Bellman target
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # TD loss with importance sampling weights
        td_error = q_values - target_q_values
        weighted_loss = (weights * td_error.pow(2)).mean()
        
        # Backward pass
        self.optimizer.zero_grad()
        weighted_loss.backward()
        
        # Gradient clipping (norm ≤ 10)
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        
        self.optimizer.step()
        
        # Update priorities in replay buffer
        priorities = np.abs(td_error.detach().cpu().numpy()) + 1e-6
        self.replay_buffer.update_priorities(indices, priorities)
        
        self.train_steps += 1
        
        return weighted_loss.item()
    
    def set_learning_rate(self, lr: float):
        """Update learning rate."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def save_checkpoint(self, path: str):
        """Save agent checkpoint."""
        checkpoint = {
            'q_net': self.q_net.state_dict(),
            'target_q_net': self.target_q_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_steps': self.train_steps,
            'episode_count': self.episode_count
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """Load agent checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(checkpoint['q_net'])
        self.target_q_net.load_state_dict(checkpoint['target_q_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.train_steps = checkpoint['train_steps']
        self.episode_count = checkpoint['episode_count']
