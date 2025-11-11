# env/noma_udn_env.py
import numpy as np
from typing import Dict, List, Tuple, Any
import math

class NOMA_UDN_Env:
    """
    Hexagonal ultra-dense network (UDN) environment for cooperative NOMA.
    Matches: Section II (System Model) and Section III (Problem Formulation) in the paper.
    """
    
    def __init__(self, config: Dict[str, Any]):
        # Load system parameters from config (Section II, Table I)
        self.P_max_dBm = config['P_max_dBm']      # = 10 dBm
        self.P_total_dBm = config['P_total_dBm']  # = 15 dBm
        self.R_min = config['R_min_bpsHz']        # = 0.5 bps/Hz
        self.I_th_dBm = config['I_th_dBm']        # = -50 dBm
        self.B = config['B_Hz']                   # = 1e6 Hz
        self.sigma2_dBm = config['sigma2_dBm']    # = -100 dBm
        self.alpha_L = config['alpha_L']          # = 0.01
        self.beta_L = config['beta_L']            # = 1.0
        self.d0 = config['d0_m']                  # Cooperation threshold (e.g., 10.0)
        
        # Convert dBm to linear scale (W)
        self.P_max = 10**(self.P_max_dBm / 10) / 1000
        self.P_total = 10**(self.P_total_dBm / 10) / 1000
        self.I_th = 10**(self.I_th_dBm / 10) / 1000
        self.sigma2 = 10**(self.sigma2_dBm / 10) / 1000
        
        # Network topology
        self.num_cells = 7  # Hexagonal layout
        self.num_users_per_cell = 2  # 1 near, 1 far
        
        # Action space discretization (from paper: 25 power levels → ~300 action pairs)
        self.power_levels_dBm = np.linspace(-10, 10, 25)  # -10 to 10 dBm
        self.power_levels = 10**(self.power_levels_dBm / 10) / 1000  # Linear (W)
        self.action_space = []
        for Pc in self.power_levels:
            for Pe in self.power_levels:
                if 0 < Pc < Pe <= self.P_max and (Pc + Pe) <= self.P_total:
                    self.action_space.append((Pc, Pe))
        self.num_actions = len(self.action_space)  # ≈ 300 as in paper
        
        # Random projection matrices for observation (Eq. 20)
        # QR decomposition ensures orthonormal basis
        self.Phi_pos = np.linalg.qr(np.random.randn(3, 3))[0]
        self.Phi_ch = np.linalg.qr(np.random.randn(3, 3))[0]
        
        # State tracking
        self.user_positions = {}
        self.channel_gains = {}
        self.current_powers = {}  # Track last action
        
    def reset(self) -> Dict[int, np.ndarray]:
        """Initialize user positions and channels (start of episode)."""
        self.user_positions = {}  # {cell_id: [(x1,y1), (x2,y2)]}
        self.channel_gains = {}   # {cell_id: [g1, g2, g12]}
        self.current_powers = {}  # {cell_id: (Pc, Pe)}
        
        for j in range(self.num_cells):
            # Random user positions within hex cell (simplified: circular radius 50m)
            users = []
            for _ in range(self.num_users_per_cell):
                r = 50 * np.sqrt(np.random.rand())  # Uniform in circle
                theta = 2 * np.pi * np.random.rand()
                users.append((r * np.cos(theta), r * np.sin(theta)))
            self.user_positions[j] = users
            
            # Rayleigh fading: g ~ Exp(1) → channel power gain
            g1 = np.random.exponential(1)
            g2 = np.random.exponential(1)
            g12 = np.random.exponential(1)
            self.channel_gains[j] = [g1, g2, g12]
            
            # Initialize powers
            self.current_powers[j] = (self.power_levels[5], self.power_levels[10])
        
        return self._get_observations()
    
    def _get_path_loss(self, distance: float) -> float:
        """
        Stretched Path Loss model (Section II).
        L(d) = exp(-α_L * d^β_L)
        """
        return np.exp(-self.alpha_L * (distance ** self.beta_L))
    
    def _get_observations(self) -> Dict[int, np.ndarray]:
        """
        Compute observation for each cell (Eq. 20).
        Φ_pos ∈ R^(3×3), Φ_ch ∈ R^(3×3) are random projection matrices.
        Observation: [Φ_pos @ d_vector; Φ_ch @ g_vector] ∈ R^6
        """
        obs = {}
        
        for j in range(self.num_cells):
            # Extract distances
            x1, y1 = self.user_positions[j][0]  # Near user
            x2, y2 = self.user_positions[j][1]  # Far user
            
            d1 = np.sqrt(x1**2 + y1**2)  # Distance near user to BS
            d2 = np.sqrt(x2**2 + y2**2)  # Distance far user to BS
            d12 = np.sqrt((x2-x1)**2 + (y2-y1)**2)  # Inter-user distance
            
            # Distance vector
            d_vec = np.array([d1, d2, d12], dtype=np.float32)
            
            # Channel gain vector
            g_vec = np.array(self.channel_gains[j], dtype=np.float32)
            
            # Apply random projections (Eq. 20)
            obs_pos = self.Phi_pos @ d_vec
            obs_ch = self.Phi_ch @ g_vec
            
            # Concatenate: [Φ_pos @ d; Φ_ch @ g] ∈ R^6
            obs[j] = np.concatenate([obs_pos, obs_ch]).astype(np.float32)
        
        return obs
    
    def _compute_sinrs(self, powers: Dict[int, Tuple[float, float]]) -> Dict[int, Tuple[float, float]]:
        """
        Compute SINR for both users in each cell (Eq. 6).
        
        Args:
            powers: {cell_id: (Pc, Pe)} power allocation
        
        Returns:
            {cell_id: (SINR1, SINR2)} SINRs
        """
        sinrs = {}
        
        for i in range(self.num_cells):
            Pc_i, Pe_i = powers[i]
            x1_i, y1_i = self.user_positions[i][0]
            x2_i, y2_i = self.user_positions[i][1]
            d1_i = np.sqrt(x1_i**2 + y1_i**2)
            d2_i = np.sqrt(x2_i**2 + y2_i**2)
            d12_i = np.sqrt((x2_i-x1_i)**2 + (y2_i-y1_i)**2)
            
            g1_i, g2_i, g12_i = self.channel_gains[i]
            L1_i = self._get_path_loss(d1_i)
            L2_i = self._get_path_loss(d2_i)
            L12_i = self._get_path_loss(d12_i)
            
            # Desired signal power (Eq. 1-3)
            P1_desired = Pc_i * g1_i * L1_i
            P2_direct = Pe_i * g2_i * L2_i
            P2_relay = Pe_i * g1_i * g12_i * L12_i * L1_i * (1 if d12_i < self.d0 else 0)
            P2_desired = P2_direct + P2_relay
            
            # Interference from other cells
            I1_total = self.sigma2
            I2_total = self.sigma2
            I_relay = 0  # Relay interference
            
            for j in range(self.num_cells):
                if j == i:
                    continue
                
                Pc_j, Pe_j = powers[j]
                x1_j, y1_j = self.user_positions[j][0]
                x2_j, y2_j = self.user_positions[j][1]
                d1_ij = np.sqrt((x1_j - x1_i)**2 + (y1_j - y1_i)**2)
                d2_ij = np.sqrt((x2_j - x1_i)**2 + (y2_j - y1_i)**2)
                d12_ij = np.sqrt((x2_j - x1_j)**2 + (y2_j - y1_j)**2)
                
                g1_ij = np.random.exponential(1)  # New fading realization
                g2_ij = np.random.exponential(1)
                g12_ij = np.random.exponential(1)
                
                L1_ij = self._get_path_loss(d1_ij)
                L2_ij = self._get_path_loss(d2_ij)
                L12_ij = self._get_path_loss(d12_ij)
                
                # Interference to near user (Eq. 4)
                I1_total += Pc_j * g1_ij * L1_ij + Pe_j * g2_ij * L2_ij
                
                # Interference to far user (Eq. 4)
                I2_total += Pc_j * g1_ij * L1_ij + Pe_j * g2_ij * L2_ij
                
                # Relay interference (Eq. 5)
                if d12_ij < self.d0:
                    I_relay += Pe_j * g1_ij * g12_ij * L12_ij * L1_ij
            
            I2_total += I_relay
            
            # SINR (Eq. 6)
            sinr1 = P1_desired / I1_total if I1_total > 0 else 0
            sinr2 = P2_desired / I2_total if I2_total > 0 else 0
            
            sinrs[i] = (sinr1, sinr2)
        
        return sinrs
    
    def _compute_rates(self, sinrs: Dict[int, Tuple[float, float]]) -> Dict[int, Tuple[float, float]]:
        """
        Compute throughput rates from SINRs.
        R = B * log2(1 + SINR)
        """
        rates = {}
        for i in range(self.num_cells):
            sinr1, sinr2 = sinrs[i]
            r1 = self.B * np.log2(1 + sinr1)
            r2 = self.B * np.log2(1 + sinr2)
            rates[i] = (r1, r2)
        return rates
    
    def _compute_interference_caused(self, powers: Dict[int, Tuple[float, float]]) -> Dict[int, float]:
        """
        Compute interference caused by each cell to neighbors (Eq. 12, C4).
        I_caused[j] = sum over neighbors of interference from cell j
        """
        I_caused = {j: 0.0 for j in range(self.num_cells)}
        
        for i in range(self.num_cells):
            Pc_i, Pe_i = powers[i]
            x1_i, y1_i = self.user_positions[i][0]
            x2_i, y2_i = self.user_positions[i][1]
            d12_i = np.sqrt((x2_i-x1_i)**2 + (y2_i-y1_i)**2)
            
            for j in range(self.num_cells):
                if j == i:
                    continue
                
                x1_j, y1_j = self.user_positions[j][0]
                x2_j, y2_j = self.user_positions[j][1]
                
                d1_ij = np.sqrt((x1_i - x1_j)**2 + (y1_i - y1_j)**2)
                d2_ij = np.sqrt((x2_i - x2_j)**2 + (y2_i - y2_j)**2)
                d12_ij = np.sqrt((x2_i - x1_i)**2 + (y2_i - y1_i)**2)
                
                g1_ij = np.random.exponential(1)
                g2_ij = np.random.exponential(1)
                g12_ij = np.random.exponential(1)
                
                L1_ij = self._get_path_loss(d1_ij)
                L2_ij = self._get_path_loss(d2_ij)
                L12_ij = self._get_path_loss(d12_ij)
                
                # Direct interference
                I_direct = Pc_i * g1_ij * L1_ij + Pe_i * g2_ij * L2_ij
                
                # Relay interference
                I_relay = Pe_i * g1_ij * g12_ij * L12_ij * L1_ij * (1 if d12_i < self.d0 else 0)
                
                I_caused[i] += I_direct + I_relay
        
        return I_caused
    
    def _validate_constraints(self, powers: Dict[int, Tuple[float, float]]) -> bool:
        """
        Validate constraints C1-C5 (Eq. 9-13).
        
        Returns:
            True if all constraints satisfied, False otherwise
        """
        for j in range(self.num_cells):
            Pc, Pe = powers[j]
            
            # C1: 0 < Pc < Pe ≤ P_max
            if not (0 < Pc < Pe <= self.P_max):
                return False
            
            # C2: Pc + Pe ≤ P_total
            if Pc + Pe > self.P_total:
                return False
        
        return True
    
    def compute_reward(self,
                      rates: Dict[int, Tuple[float, float]],
                      I_caused: Dict[int, float],
                      phase: int = 1,
                      lambda_I: float = 0.1,
                      lambda_Q: float = 2.0,
                      lambda_var: float = 0.5) -> Dict[int, float]:
        """
        Compute reward for each cell based on phase (Eq. 15-16).
        
        Phase 1 (Eq. 15): Maximize sum-rate with interference penalty
        Phase 2 (Eq. 16): Maximize Jain's Fairness Index
        
        Args:
            rates: {cell_id: (R1, R2)} throughput rates
            I_caused: {cell_id: I} interference caused to neighbors
            phase: Training phase (1 or 2)
            lambda_I: Interference penalty weight
            lambda_Q: QoS penalty weight
            lambda_var: Variance penalty weight (Phase 2)
        
        Returns:
            {cell_id: reward} reward for each cell
        """
        rewards = {}
        
        for j in range(self.num_cells):
            R1, R2 = rates[j]
            I_j = I_caused[j]
            
            if phase == 1:
                # Eq. 15: Sum-rate maximization with QoS constraints
                sum_rate = R1 + R2
                
                # QoS penalty: penalize if rates below minimum
                qos_penalty = max(0, self.R_min - R1) + max(0, self.R_min - R2)
                
                # Reward = sum_rate - λ_I * I_caused - λ_Q * QoS_penalty
                reward = sum_rate - lambda_I * I_j - lambda_Q * qos_penalty
            
            else:  # phase == 2
                # Eq. 16: Fairness optimization (Jain's Fairness Index)
                # JFI = (R1 + R2)^2 / (R1^2 + R2^2)
                numerator = (R1 + R2) ** 2
                denominator = R1**2 + R2**2 + 1e-10  # Avoid division by zero
                jfi = numerator / denominator
                
                # Variance penalty: encourage equal rates
                rate_var = np.var([R1, R2])
                
                # Reward = JFI - λ_I * I_caused - λ_var * variance
                reward = jfi - lambda_I * I_j - lambda_var * rate_var
            
            rewards[j] = reward
        
        return rewards
    
    def step(self, actions: Dict[int, int]) -> Tuple[Dict, bool, Dict, Dict, Dict]:
        """
        Execute one step in environment.
        
        Args:
            actions: {cell_id: action_idx} power allocation actions
        
        Returns:
            next_obs: Next observations for all cells
            done: Episode termination flag
            rates: Throughput rates {cell_id: (R1, R2)}
            I_caused: Interference caused {cell_id: I}
            info: Additional info dict
        """
        # Convert action indices to power pairs
        powers = {}
        for j in range(self.num_cells):
            action_idx = actions[j]
            powers[j] = self.action_space[action_idx]
        
        # Validate constraints
        if not self._validate_constraints(powers):
            # Fallback to safe action
            for j in range(self.num_cells):
                powers[j] = self.action_space[0]
        
        # Store current powers
        self.current_powers = powers
        
        # Compute SINRs and rates
        sinrs = self._compute_sinrs(powers)
        rates = self._compute_rates(sinrs)
        
        # Compute interference caused
        I_caused = self._compute_interference_caused(powers)
        
        # Update channel gains for next step (Rayleigh fading)
        for j in range(self.num_cells):
            self.channel_gains[j] = [
                np.random.exponential(1),
                np.random.exponential(1),
                np.random.exponential(1)
            ]
        
        # Get next observations
        next_obs = self._get_observations()
        
        # Episode termination (fixed length episodes)
        done = False
        
        # Info dict
        info = {
            'sinrs': sinrs,
            'powers': powers,
            'I_caused': I_caused
        }
        
        return next_obs, done, rates, I_caused, info
