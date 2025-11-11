# utils/reward_calculator.py

def get_blended_reward(
    episode: int, 
    reward_phase1: float, 
    reward_phase2: float,
    smooth_start: int = 28000,
    smooth_end: int = 32000
) -> float:
    """
    Blend rewards between Phase 1 (throughput) and Phase 2 (fairness) 
    during smooth transition window.
    
    Matches: Section IV, Figure 1 caption, and Algorithm 1 (Phase Transition block).
    
    Args:
        episode: Current training episode
        reward_phase1: Reward from throughput objective (Eq. 13)
        reward_phase2: Reward from fairness objective (Eq. 14)
        smooth_start: Start of blending window (default: 28,000)
        smooth_end: End of blending window (default: 32,000)
    
    Returns:
        Blended reward r_t = (1 - α) * r1 + α * r2
    """
    if episode < smooth_start:
        # Pure Phase 1
        return reward_phase1
    elif episode >= smooth_end:
        # Pure Phase 2
        return reward_phase2
    else:
        # Linear interpolation: α from 0 → 1
        alpha = (episode - smooth_start) / (smooth_end - smooth_start)
        return (1.0 - alpha) * reward_phase1 + alpha * reward_phase2