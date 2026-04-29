# convert the trajectory (past 20 steps) into the input string for the policy
def format_context(instruction, trajectory, current_obs, cumulative_reward=None):
    """Format the observation with most recent steps"""
    if cumulative_reward is None:
        cumulative_reward = sum(step.get('reward', 0) for step in trajectory)
    
    if not trajectory:
        return f"Task: {instruction}\nScore: {cumulative_reward}\n\nCurrent observation: {current_obs}"
        # return f"Task: {instruction}\n\nCurrent observation: {current_obs}"
    
    context_str = ""
    for i, step in enumerate(trajectory):
        # Truncate long observations
        obs_short = step['obs'][:300] + "..." if len(step['obs']) > 300 else step['obs']
        context_str += f"Step {i+1}: {obs_short}\nAction: {step['action']}\nReward: {step['reward']}\n\n"
        # context_str += f"Step {i+1}: {obs_short}\nAction: {step['action']}\n\n"
    
    # Truncate current observation
    obs_short = current_obs[:300] + "..." if len(current_obs) > 300 else current_obs
    
    return f"Task: {instruction}\nScore: {cumulative_reward}\n\nPrevious steps:\n{context_str}\nCurrent observation: {obs_short}"
    # return f"Task: {instruction}\n\nPrevious steps:\n{context_str}\nCurrent observation: {obs_short}"

def convert_trajectories(trajectories, limit=None):
    data = []
    for idx, t in enumerate(trajectories):
        traj = t['traj']
        for step, x in enumerate(traj):
            if step == 0:
                task = x['after']['obs']['instruction']
                before = x['after']['obs']['observation']
                continue
            assert before is not None
            assert task is not None
            after = x['after']['obs']['observation']
            action = x['action']
            if isinstance(action, dict):
                action = action['key']
            y = dict(
                task=task,
                before=before,
                after=after,
                action=action,
                label=False,
                feedback='',
                step=step,
                orig=x,
                traj_idx=idx
            )
            before = after
            data.append(y)
            if limit and len(data) >= limit:
                break
    return data