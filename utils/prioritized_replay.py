# utils/prioritized_replay.py
import random
import numpy as np
from collections import deque, namedtuple
from typing import List, Tuple, Any

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class SegmentTree:
    """Segment tree for efficient sum queries and updates O(log N)."""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity)
        self.data = np.zeros(capacity, dtype=object)
    
    def _propagate(self, idx: int, change: float):
        """Propagate change up the tree."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx: int, s: float) -> int:
        """Retrieve leaf index for cumulative sum s."""
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def add(self, priority: float, data: Any):
        """Add new element with priority."""
        idx = self.capacity + len(self.data)
        if idx >= len(self.tree):
            # Expand tree
            self.tree = np.concatenate([self.tree, np.zeros(len(self.tree))])
        
        self.set(idx - self.capacity, priority, data)
    
    def set(self, idx: int, priority: float, data: Any):
        """Set priority and data at index."""
        tree_idx = idx + self.capacity
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self.data[idx] = data
        self._propagate(tree_idx, change)
    
    def sum(self) -> float:
        """Get total sum of priorities."""
        return self.tree[1]
    
    def sample(self, s: float) -> Tuple[int, float, Any]:
        """Sample index based on priority."""
        idx = self._retrieve(1, s)
        data_idx = idx - self.capacity
        return data_idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer.
    Samples transitions with probability proportional to TD-error.
    """
    
    def __init__(self, capacity: int = 100000, alpha: float = 0.6, beta: float = 0.4):
        """
        Initialize prioritized replay buffer.
        
        Args:
            capacity: Maximum buffer size
            alpha: Priority exponent (0=uniform, 1=full prioritization)
            beta: Importance sampling exponent (0=no correction, 1=full correction)
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = 0.001  # Anneal beta towards 1.0
        
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.max_priority = 1.0
        
    def add(self, transition: Transition):
        """
        Add transition with maximum priority.
        
        Args:
            transition: Transition namedtuple
        """
        self.buffer.append(transition)
        self.priorities.append(self.max_priority)
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """
        Update priorities for sampled transitions.
        
        Args:
            indices: Indices of transitions
            priorities: New priority values (typically TD-errors)
        """
        for idx, priority in zip(indices, priorities):
            if idx < len(self.priorities):
                self.priorities[idx] = priority
                self.max_priority = max(self.max_priority, priority)
    
    def sample(self, batch_size: int) -> Tuple[List[Transition], np.ndarray, np.ndarray]:
        """
        Sample batch with prioritization and importance sampling weights.
        
        Args:
            batch_size: Number of samples
        
        Returns:
            transitions: List of sampled transitions
            indices: Indices of sampled transitions
            weights: Importance sampling weights
        """
        if len(self.buffer) == 0:
            return [], np.array([]), np.array([])
        
        # Compute sampling probabilities
        priorities = np.array(list(self.priorities))
        priorities = np.power(priorities, self.alpha)
        probabilities = priorities / priorities.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), size=batch_size, p=probabilities)
        
        # Compute importance sampling weights
        weights = np.power(len(self.buffer) * probabilities[indices], -self.beta)
        weights /= weights.max()  # Normalize by max weight
        
        # Anneal beta towards 1.0
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Get transitions
        transitions = [self.buffer[i] for i in indices]
        
        return transitions, indices, weights
    
    def __len__(self) -> int:
        """Return buffer size."""
        return len(self.buffer)
