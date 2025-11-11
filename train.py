# train.py
import os
import yaml
import numpy as np
import torch
import random
from collections import defaultdict
from env.noma_udn_env import NOMA_UDN_Env
from agents.d3qn_agent import D3QNAgent
from utils.reward_calculator import get_blended_reward, get_phase, get_learning_rate
import argparse
import logging
from datetime import datetime

def load_yaml(path):
    """Load YAML configuration file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def setup_logging(run_id: int, ablation: str):
    """Setup logging configuration."""
    log_dir = f"logs/run_{run_id}_{ablation}"
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{log_dir}/train.log"),
            logging.StreamHandler()
        ]
    )
    
    return log_dir

def main(run_id: int = 0, ablation: str = "full", num_episodes: int = 40000):
    """
    Main training loop.
    
    Args:
        run_id: Run identifier
        ablation: Ablation type ('full', 'no_interference', 'no_qos', 'single_phase', 'abrupt_switch')
        num_episodes: Number of training episodes
    """
    
    # 🔧 Setup
    set_seed(run_id)
    log_dir = setup_logging(run_id, ablation)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logging.info(f"Starting training: run_id={run_id}, ablation={ablation}, device={device}")
    
    # 📋 Load configurations
    system_config = load_yaml("config/system.yaml")
    hparams = load_yaml("config/hparams.yaml")
    
    # 🔄 Apply ablation settings
    if ablation == "no_interference":
        hparams['lambda_I'] = 0.0
        logging.info("Ablation: Disabled interference penalty")
    elif ablation == "no_qos":
        hparams['lambda_Q'] = 0.0
        logging.info("Ablation: Disabled QoS penalty")
    elif ablation == "single_phase":
        hparams['smooth_start'] = num_episodes + 1  # Never transition
        logging.info("Ablation: Single phase (Phase 1 only)")
    
    smooth_transition = (ablation != "abrupt_switch")
    
    # 🌐 Initialize environment and agents
    env = NOMA_UDN_Env(system_config)
    state_dim = 6  # Observation dimension from random projections
    action_dim = env.num_actions
    
    logging.info(f"Environment: {env.num_cells} cells, {action_dim} actions per agent")
    
    agents = {}
    for j in range(env.num_cells):
        agents[j] = D3QNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            agent_id=j,
            lr=hparams['lr_phase1'],
            gamma=hparams['gamma'],
            tau=hparams['tau'],
            buffer_size=hparams['replay_buffer_size'],
            batch_size=hparams['batch_size'],
            epsilon_start=hparams['epsilon_start'],
            epsilon_end=hparams['epsilon_end'],
            epsilon_decay=hparams['epsilon_decay'],
            device=device
        )
    
    # 📊 Metrics tracking
    metrics = defaultdict(list)
    episode_rewards = {j: [] for j in range(env.num_cells)}
    episode_rates = {j: [] for j in range(env.num_cells)}
    
    # 🎮 Training loop
    for episode in range(num_episodes):
        # Reset environment
        obs = env.reset()
        episode_done = False
        step_count = 0
        max_steps = 100  # Fixed episode length
        
        # Episode metrics
        ep_rewards = {j: 0.0 for j in range(env.num_cells)}
        ep_rates = {j: 0.0 for j in range(env.num_cells)}
        ep_losses = {j: 0.0 for j in range(env.num_cells)}
        
        # 🔄 Episode loop
        while step_count < max_steps and not episode_done:
            # Select actions
            actions = {}
            for j in range(env.num_cells):
                actions[j] = agents[j].select_action(obs[j], training=True)
            
            # Step environment
            next_obs, done, rates, I_caused, info = env.step(actions)
            
            # Determine phase and compute rewards
            phase = get_phase(episode, hparams['smooth_start'])
            rewards_phase1 = env.compute_reward(rates, I_caused, phase=1, 
                                               lambda_I=hparams.get('lambda_I', 0.1),
                                               lambda_Q=hparams.get('lambda_Q', 2.0))
            rewards_phase2 = env.compute_reward(rates, I_caused, phase=2,
                                               lambda_I=hparams.get('lambda_I', 0.1),
                                               lambda_var=hparams.get('lambda_var', 0.5))
            
            # Blend rewards if smooth transition enabled
            if smooth_transition:
                rewards = {}
                for j in range(env.num_cells):
                    rewards[j] = get_blended_reward(
                        episode,
                        rewards_phase1[j],
                        rewards_phase2[j],
                        hparams['smooth_start'],
                        hparams['smooth_end']
                    )
            else:
                rewards = rewards_phase1 if phase == 1 else rewards_phase2
            
            # Store transitions and accumulate metrics
            for j in range(env.num_cells):
                agents[j].store_transition(
                    obs[j], actions[j], rewards[j], next_obs[j], done
                )
                ep_rewards[j] += rewards[j]
                ep_rates[j] += rates[j][0] + rates[j][1]  # Sum of both users
            
            # Training step
            for j in range(env.num_cells):
                loss = agents[j].train_step()
                ep_losses[j] += loss
                agents[j].update_target_network()
            
            # Update observations
            obs = next_obs
            step_count += 1
        
        # 📈 Update epsilon and learning rate
        for j in range(env.num_cells):
            agents[j].update_epsilon(episode)
            agents[j].episode_count = episode
        
        # Update learning rate based on phase
        phase = get_phase(episode, hparams['smooth_start'])
        if phase == 1:
            lr = get_learning_rate(episode, 1, hparams['lr_phase1'])
        else:
            lr = hparams['lr_phase2']
        
        for j in range(env.num_cells):
            agents[j].set_learning_rate(lr)
        
        # 📊 Log metrics
        avg_reward = np.mean([ep_rewards[j] for j in range(env.num_cells)])
        avg_rate = np.mean([ep_rates[j] for j in range(env.num_cells)])
        avg_loss = np.mean([ep_losses[j] for j in range(env.num_cells)])
        
        metrics['episode'].append(episode)
        metrics['avg_reward'].append(avg_reward)
        metrics['avg_rate'].append(avg_rate)
        metrics['avg_loss'].append(avg_loss)
        metrics['phase'].append(phase)
        
        # Logging
        if (episode + 1) % 100 == 0:
            logging.info(
                f"Episode {episode+1}/{num_episodes} | "
                f"Phase: {phase} | "
                f"Avg Reward: {avg_reward:.4f} | "
                f"Avg Rate: {avg_rate:.2f} | "
                f"Avg Loss: {avg_loss:.6f} | "
                f"LR: {lr:.2e}"
            )
        
        # Checkpoint saving
        if (episode + 1) % 1000 == 0:
            for j in range(env.num_cells):
                agents[j].save_checkpoint(
                    f"{log_dir}/agent_{j}_ep{episode+1}.pt"
                )
            logging.info(f"Checkpoint saved at episode {episode+1}")
    
    # 💾 Save final metrics
    np.save(f"{log_dir}/metrics.npy", metrics)
    logging.info(f"Training completed. Results saved to {log_dir}")
    
    return metrics, agents

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train D3QN-IQL agents for NOMA-UDN")
    parser.add_argument('--run_id', type=int, default=0, help='Run identifier')
    parser.add_argument('--ablation', type=str, default='full',
                       choices=['full', 'no_interference', 'no_qos', 'single_phase', 'abrupt_switch'],
                       help='Ablation type')
    parser.add_argument('--episodes', type=int, default=40000, help='Number of episodes')
    parser.add_argument('--device', type=str, default='auto', help='Device: cpu/cuda/auto')
    
    args = parser.parse_args()
    
    metrics, agents = main(
        run_id=args.run_id,
        ablation=args.ablation,
        num_episodes=args.episodes
    )
