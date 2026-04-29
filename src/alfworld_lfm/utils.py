# convert the trajectory (past 20 steps) into the input string for the policy
def format_context(instruction, trajectory, current_obs, cumulative_reward=None):
    """Format the observation with most recent steps"""
    if cumulative_reward is None:
        cumulative_reward = sum(step.get('reward', 0) or 0 for step in trajectory)
    
    if not trajectory:
        return f"Task: {instruction}\nScore: {cumulative_reward}\n\nCurrent observation: {current_obs}"
    
    context_str = ""
    for i, step in enumerate(trajectory):
        # Truncate long observations
        obs_short = step['obs'][:300] + "..." if len(step['obs']) > 300 else step['obs']
        context_str += f"Step {i+1}: {obs_short}\nAction: {step['action']}\nReward: {step['reward']}\n\n"
    
    # Truncate current observation
    obs_short = current_obs[:300] + "..." if len(current_obs) > 300 else current_obs
    
    return f"Task: {instruction}\nScore: {cumulative_reward}\n\nPrevious steps:\n{context_str}\nCurrent observation: {obs_short}"