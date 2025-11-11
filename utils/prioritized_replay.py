# utils/prioritized_replay.py
import random
import numpy as np
from collections import deque, namedtuple
from typing import List, Tuple, Any

# Cấu trúc transition
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class SegmentTree:
    """Cây đoạn (segment tree) cho truy vấn tổng và cập nhật hiệu quả O(log N)."""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity)
    
    def _propagate(self, idx: int, change: float):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def update(self, idx: int, val: float):
        change = val - self.tree[idx]
        self.tree[idx] = val
        if idx > 0:
            self._propagate(idx, change)
    
    def total(self) -> float:
        return self.tree[0]
    
    def get(self, s: float) -> int:
        idx = self._retrieve(0, s)
        return idx

class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer (PER).
    Matches: Section III.D, Mechanism 1 & Algorithm 1.
    References: Schaul et al., "Prioritized Experience Replay", arXiv:1511.05952.
    """
    def __init__(self, 
                 capacity: int, 
                 alpha: float = 0.6, 
                 beta: float = 0.4, 
                 beta_increment: float = 1e-5,
                 epsilon: float = 1e-5):
        """
        Args:
            capacity: kích thước bộ đệm (paper: 100,000)
            alpha: hệ số ưu tiên (paper: α=0.6)
            beta: hệ số IS correction (ban đầu thấp, tăng dần)
            beta_increment: tốc độ tăng beta mỗi lần sample
            epsilon: tránh chia cho 0
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        
        self.buffer = deque(maxlen=capacity)
        self.priorities = SegmentTree(capacity)
        self.pos = 0  # Con trỏ ghi
    
    def add(self, state, action, reward, next_state, done):
        """Thêm transition mới với ưu tiên cao nhất."""
        max_priority = self.priorities.tree.max()
        if max_priority == 0:
            max_priority = 1.0
        
        self.buffer.append(Transition(state, action, reward, next_state, done))
        idx = self.pos + self.capacity
        self.priorities.update(idx, max_priority)
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple[List[Transition], List[int], np.ndarray]:
        """Lấy mẫu batch với ưu tiên và tính trọng số IS."""
        priorities = self.priorities.tree[self.capacity: self.capacity + len(self.buffer)]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        batch = [self.buffer[idx] for idx in indices]
        
        # Tính trọng số Importance Sampling (IS)
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Chuẩn hóa để ổn định
        weights = np.array(weights, dtype=np.float32)
        
        # Tăng beta dần (giảm bias theo thời gian)
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return batch, indices.tolist(), weights
    
    def update_priorities(self, indices: List[int], td_errors: np.ndarray):
        """Cập nhật ưu tiên dựa trên TD error mới."""
        for idx, td_err in zip(indices, td_errors):
            priority = (td_err + self.epsilon) ** self.alpha
            tree_idx = idx + self.capacity
            self.priorities.update(tree_idx, priority)
    
    def __len__(self):
        return len(self.buffer)