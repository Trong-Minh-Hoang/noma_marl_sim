# utils/reward_calculator.py
"""
Blended reward function for two-phase training.
Matches: Section III.D (convergence mechanisms)
"""

def get_blended_reward(
    episode: int, 
    reward_phase1: float, 
    reward_phase2: float,
    smooth_start: int = 28000,
    smooth_end: int = 32000
) -> float:
    """
    Smooth transition between Phase 1 (sum-rate) and Phase 2 (fairness).
    
    Args:
        episode: Current episode number
        reward_phase1: Sum-rate reward (Phase 1 objective)
        reward_phase2: Fairness reward (Phase 2 objective)
        smooth_start: Episode to start blending (default: 28000)
        smooth_end: Episode to complete transition (default: 32000)
    
    Returns:
        Blended reward value
    """
    if episode < smooth_start:
        # Phase 1: Pure sum-rate maximization
        return reward_phase1
    
    elif episode > smooth_end:
        # Phase 2: Pure fairness optimization
        return reward_phase2
    
    else:
        # Smooth transition: Linear interpolation
        alpha = (episode - smooth_start) / (smooth_end - smooth_start)
        blended = (1 - alpha) * reward_phase1 + alpha * reward_phase2
        return blended


def get_phase(episode: int, smooth_start: int = 28000) -> int:
    """
    Determine current training phase.
    
    Args:
        episode: Current episode number
        smooth_start: Episode to start Phase 2
    
    Returns:
        1 for Phase 1, 2 for Phase 2
    """
    return 1 if episode < smooth_start else 2


def get_learning_rate(
    episode: int,
    phase: int,
    lr_base: float,
    smooth_start: int = 28000,
    smooth_end: int = 32000
) -> float:
    """
    Compute learning rate with phase-dependent scheduling.
    
    Phase 1: Cosine annealing from lr_base to 0.1*lr_base over 30000 episodes
    Phase 2: Constant lr_base (reduced by factor in train.py)
    
    Args:
        episode: Current episode
        phase: Training phase (1 or 2)
        lr_base: Base learning rate
        smooth_start: Phase 2 start episode
        smooth_end: Phase 2 end episode
    
    Returns:
        Learning rate for current episode
    """
    import numpy as np
    
    if phase == 1:
        # Cosine annealing: lr(t) = 0.5 * lr_base * (1 + cos(π*t/30000))
        progress = min(episode / 30000.0, 1.0)
        lr = lr_base * 0.5 * (1 + np.cos(np.pi * progress))
        return max(lr, 0.1 * lr_base)  # Floor at 0.1*lr_base
    
    else:  # phase == 2
        # Constant learning rate (already reduced in train.py)
        return lr_base
