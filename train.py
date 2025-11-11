# train.py
import os
import yaml
import numpy as np
import torch
import random
from collections import defaultdict
from env.noma_udn_env import NOMA_UDN_Env
from agents.d3qn_agent import D3QNAgent
from utils.reward_calculator import get_blended_reward
import argparse

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main(run_id: int = 0, ablation: str = "full"):
    # 🔹 TẢI CẤU HÌNH TỪ 2 FILE YAML
    system_config = load_yaml("config/system.yaml")
    hparams = load_yaml("config/hparams.yaml")
    
    # Áp dụng ablation bằng cách ghi đè hparams
    if ablation == "no_interference":
        hparams['lambda_I'] = 0.0
    elif ablation == "no_qos":
        hparams['lambda_Q'] = 0.0
    elif ablation == "single_phase":
        single_phase_flag = True
    else:
        single_phase_flag = False
    
    smooth_transition = (ablation != "abrupt_switch")
    
    # Thiết lập môi trường và agent
    env = NOMA_UDN_Env(system_config)
    state_dim = 8
    action_dim = env.num_actions
    
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
            grad_max_norm=hparams['grad_max_norm'],
            epsilon_start=hparams['epsilon_start'],
            epsilon_end=hparams['epsilon_end'],
            epsilon_decay_episodes=hparams['epsilon_decay_episodes']
        )
    
    # Theo dõi kết quả
    episode_returns = []
    convergence_metrics = {
        'entropy': [],
        'td_error': [],
        'reward_var': []
    }
    
    total_episodes = hparams['total_episodes']
    phase1_end = hparams['phase1_episodes']
    smooth_start = hparams['smooth_transition_start']
    smooth_end = hparams['smooth_transition_end']
    
    # Vòng huấn luyện chính
    for episode in range(total_episodes):
        # Xác định pha
        if single_phase_flag:
            phase = 1
        elif episode < phase1_end:
            phase = 1
        else:
            phase = 2
        
        # Cập nhật learning rate & epsilon cho Phase 2
        if phase == 2:
            for agent in agents.values():
                for param_group in agent.optimizer.param_groups:
                    param_group['lr'] = hparams['lr_phase2']
                agent.epsilon = max(hparams['epsilon_end'], 0.1 * hparams['epsilon_start'])
        
        # Reset môi trường
        obs = env.reset()
        episode_reward = 0.0
        
        # Mỗi episode có T bước (giả sử T=100 như trong nhiều tác phẩm MARL)
        for t in range(100):
            actions = {j: agents[j].select_action(obs[j], training=True) for j in agents}
            next_obs, _, rates, I_caused, _ = env.step(actions)
            
            # Tính phần thưởng theo từng pha
            rewards_p1 = env.compute_reward(rates, I_caused, phase=1,
                                           lambda_I=hparams['lambda_I'],
                                           lambda_Q=hparams['lambda_Q'])
            rewards_p2 = env.compute_reward(rates, I_caused, phase=2,
                                           lambda_I=hparams['lambda_I'],
                                           lambda_var=hparams['lambda_var'])
            
            # Áp dụng blending (nếu không phải ablation abrupt)
            if smooth_transition and not single_phase_flag:
                final_rewards = {
                    j: get_blended_reward(episode, rewards_p1[j], rewards_p2[j], smooth_start, smooth_end)
                    for j in agents
                }
            else:
                final_rewards = rewards_p1 if phase == 1 else rewards_p2
            
            # Lưu transition và cộng dồn phần thưởng
            for j in agents:
                agents[j].store_transition(obs[j], actions[j], final_rewards[j], next_obs[j], False)
                episode_reward += final_rewards[j]
            
            # Huấn luyện
            for agent in agents.values():
                if len(agent.replay_buffer) >= agent.batch_size:
                    agent.train()
                    agent.update_target_network()
            
            obs = next_obs
        
        # Cập nhật epsilon (chỉ Phase 1)
        if phase == 1:
            for agent in agents.values():
                agent.update_epsilon(episode)
        
        # Ghi nhận kết quả
        avg_reward = episode_reward / (env.num_cells * 100)
        episode_returns.append(avg_reward)
        
        # Ghi chỉ số hội tụ mỗi 100 episodes
        if episode % 100 == 0:
            ent = np.mean([agents[j].get_metrics()['policy_entropy'] for j in agents])
            td = np.mean([agents[j].get_metrics()['td_error'] for j in agents])
            var = np.mean([agents[j].compute_reward_variance() for j in agents])
            convergence_metrics['entropy'].append(ent)
            convergence_metrics['td_error'].append(td)
            convergence_metrics['reward_var'].append(var)
    
    # Lưu kết quả
    os.makedirs(f"results/{ablation}", exist_ok=True)
    np.save(f"results/{ablation}/episode_returns_run{run_id}.npy", np.array(episode_returns))
    np.save(f"results/{ablation}/convergence_metrics_run{run_id}.npy", 
            {k: np.array(v) for k, v in convergence_metrics.items()})
    
    print(f"✅ Hoàn thành: ablation={ablation}, run={run_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=int, default=0,
                        help="Seed index for independent runs (for Figure 1)")
    parser.add_argument("--ablation", type=str, default="full",
                        choices=["full", "no_interference", "no_qos", "single_phase", "abrupt_switch"],
                        help="Ablation study configuration (for Table 2)")
    args = parser.parse_args()
    
    set_seed(42 + args.run_id)
    main(args.run_id, args.ablation)