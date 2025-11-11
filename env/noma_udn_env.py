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

    def reset(self) -> Dict[int, np.ndarray]:
        """Initialize user positions and channels (start of episode)."""
        self.user_positions = {}  # {cell_id: [(x1,y1), (x2,y2)]}
        self.channel_gains = {}   # {cell_id: [g1, g2, g12]}
        
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
        
        return self._get_observations()
    
    def _get_observations(self) -> Dict[int, np.ndarray]:
        """Local observation per BS (Section III.B: POMDP formulation)."""
        obs = {}
        for j in range(self.num_cells):
            pos = self.user_positions[j]
            d1 = np.linalg.norm(pos[0])  # Distance near user to BS
            d2 = np.linalg.norm(pos[1])  # Distance far user to BS
            d12 = np.linalg.norm(np.array(pos[0]) - np.array(pos[1]))  # Inter-user dist
            
            g1, g2, g12 = self.channel_gains[j]
            
            # Observation vector (8D as in paper: d_s=8)
            obs[j] = np.array([
                d1, d2, d12,      # 3 distances
                g1, g2, g12,      # 3 channel gains
                0.0, 0.0          # Placeholder for P_c, P_e (will be updated after action)
            ], dtype=np.float32)
        return obs

    def step(self, actions: Dict[int, int]) -> Tuple[Dict, Dict, Dict, Dict]:
        """
        Execute joint actions, compute rewards, SINR, and next state.
        Matches: Eq.(1)-(6), Eq.(8)-(11), and reward definitions (Eq.13-14).
        """
        rewards = {}
        infos = {}
        
        # Store chosen powers
        powers = {}
        for j, action_idx in actions.items():
            Pc, Pe = self.action_space[action_idx]
            powers[j] = (Pc, Pe)
        
        # Precompute all signal powers and interference
        S1 = {}  # Near user signal
        S2 = {}  # Far user combined signal
        I_total = {}  # Total interference per user
        I_caused = {}  # Interference caused by BS j
        
        for j in range(self.num_cells):
            d12 = np.linalg.norm(
                np.array(self.user_positions[j][0]) - np.array(self.user_positions[j][1])
            )
            g1, g2, g12 = self.channel_gains[j]
            Pc, Pe = powers[j]
            
            # Path losses (Stretched Path Loss model: L = exp(-α d^β))
            L1 = np.exp(-self.alpha_L * (np.linalg.norm(self.user_positions[j][0]))**self.beta_L)
            L2 = np.exp(-self.alpha_L * (np.linalg.norm(self.user_positions[j][1]))**self.beta_L)
            L12 = np.exp(-self.alpha_L * d12**self.beta_L)
            
            # Signal powers (Eq. 1, 3)
            S1[j] = Pc * g1 * L1
            coop_gain = Pe * g1 * g12 * L12 * L1 if d12 < self.d0 else 0.0
            S2[j] = Pe * g2 * L2 + coop_gain
            
            # Total interference caused by BS j (for C4)
            I_caused[j] = 0.0
        
        # Compute interference at each BS (Eq. 4, 5)
        for i in range(self.num_cells):  # Target BS
            I_direct = 0.0
            I_relay = 0.0
            for j in range(self.num_cells):  # Interfering BS
                if i == j:
                    continue
                # Distance from interferers to target BS i
                # Simplified: assume fixed inter-BS distance = 100m (2 * cell radius)
                d_to_i = 100.0
                L1_to_i = np.exp(-self.alpha_L * d_to_i**self.beta_L)
                L2_to_i = L1_to_i
                
                Pc_j, Pe_j = powers[j]
                g1_to_i = np.random.exponential(1)  # New fading to BS i
                g2_to_i = np.random.exponential(1)
                
                # Direct interference (Eq. 4)
                I_direct += Pc_j * g1_to_i * L1_to_i + Pe_j * g2_to_i * L2_to_i
                
                # Relay interference (Eq. 5)
                d12_j = np.linalg.norm(
                    np.array(self.user_positions[j][0]) - np.array(self.user_positions[j][1])
                )
                if d12_j < self.d0:
                    g12_to_i = np.random.exponential(1)
                    L12_to_i = np.exp(-self.alpha_L * d12_j**self.beta_L)
                    I_relay += Pe_j * g1_to_i * g12_to_i * L12_to_i * L1_to_i
            
            I_total[i] = I_direct + I_relay
            
            # Accumulate I_caused for each BS j
            for j in range(self.num_cells):
                if i != j:
                    # Same as above but from j's perspective
                    d_to_j = 100.0
                    L1_to_j = np.exp(-self.alpha_L * d_to_j**self.beta_L)
                    L2_to_j = L1_to_j
                    Pc_j, Pe_j = powers[j]
                    g1_to_j = np.random.exponential(1)
                    g2_to_j = np.random.exponential(1)
                    I_dir = Pc_j * g1_to_j * L1_to_j + Pe_j * g2_to_j * L2_to_j
                    
                    d12_j = np.linalg.norm(
                        np.array(self.user_positions[j][0]) - np.array(self.user_positions[j][1])
                    )
                    I_rel = 0.0
                    if d12_j < self.d0:
                        g12_to_j = np.random.exponential(1)
                        L12_to_j = np.exp(-self.alpha_L * d12_j**self.beta_L)
                        I_rel = Pe_j * g1_to_j * g12_to_j * L12_to_j * L1_to_j
                    I_caused[j] += I_dir + I_rel
        
        # Compute SINR and rates (Eq. 6, 7)
        rates = {}
        sinrs = {}
        for j in range(self.num_cells):
            total_interference = I_total[j]
            sinr1 = S1[j] / (total_interference + self.sigma2)
            sinr2 = S2[j] / (total_interference + self.sigma2)
            sinrs[j] = (sinr1, sinr2)
            
            R1 = self.B * np.log2(1 + sinr1)
            R2 = self.B * np.log2(1 + sinr2)
            rates[j] = (R1, R2)
        
        # Return observations, rewards, etc.
        next_obs = self._get_next_observations(powers)
        
        return next_obs, rewards, rates, I_caused, sinrs  # Will compute rewards externally

    def _get_next_observations(self, powers: Dict[int, Tuple[float, float]]) -> Dict[int, np.ndarray]:
        """Update observation with current powers and new channels."""
        next_obs = {}
        for j in range(self.num_cells):
            # Update channel gains (Rayleigh fading per frame)
            g1 = np.random.exponential(1)
            g2 = np.random.exponential(1)
            g12 = np.random.exponential(1)
            self.channel_gains[j] = [g1, g2, g12]
            
            pos = self.user_positions[j]
            d1 = np.linalg.norm(pos[0])
            d2 = np.linalg.norm(pos[1])
            d12 = np.linalg.norm(np.array(pos[0]) - np.array(pos[1]))
            
            Pc, Pe = powers[j]
            next_obs[j] = np.array([
                d1, d2, d12,
                g1, g2, g12,
                Pc, Pe  # Now include actual chosen powers
            ], dtype=np.float32)
        return next_obs

    def compute_reward(self, 
                      rates: Dict[int, Tuple[float, float]], 
                      I_caused: Dict[int, float],
                      phase: int,
                      lambda_I: float = 0.1,
                      lambda_Q: float = 2.0,
                      lambda_var: float = 0.5) -> Dict[int, float]:
        """
        Compute phase-dependent rewards (Eq. 13, 14).
        """
        rewards = {}
        for j in range(self.num_cells):
            R1, R2 = rates[j]
            I_j = I_caused[j]
            
            # QoS penalty
            qos_penalty = max(0, self.R_min - R1) + max(0, self.R_min - R2)
            
            if phase == 1:  # Throughput optimization
                reward = (R1 + R2) - lambda_I * I_j - lambda_Q * qos_penalty
            else:  # Phase 2: Fairness optimization
                # Jain's Fairness Index (Eq. 12)
                jfi = 0.5 * (R1 + R2)**2 / (R1**2 + R2**2 + 1e-10)
                rate_var = np.var([R1, R2])
                reward = jfi - lambda_I * I_j - lambda_var * rate_var
            
            rewards[j] = reward
        return rewards