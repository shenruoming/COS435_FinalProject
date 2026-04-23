from environment import VerbalizedALFWorld
from utils import format_context
from feedback_collector import get_all_feedback
from feedback_data import FeedbackDataset
def test_rollout(env):
    max_steps = 50
    context_window = 20
    instruction, obs, actions = env.reset()
    
    # Flatten actions if needed
    if actions and isinstance(actions[0], list):
        actions = actions[0]
    
    # Store trajectory history for context window
    traj = []
    done = False
    step = 0
    
    traj.append({
        'after': {
            'obs': {
                'instruction': instruction,
                'observation': obs
            },
            'reward': None,
            'terminated': None,
        },
        'action': None,
        'action_scores': None,
        'prompt': None,
    })
    
    trajectory_steps = [] 
    cumulative_reward = 0.0
    won = False
    
    for step in range(max_steps):
        # Build input with cumulative reward
        recent_context = trajectory_steps[-context_window:] if len(trajectory_steps) >= context_window else trajectory_steps
        inp = format_context(instruction, recent_context, obs, cumulative_reward)
        
        # Use policy to choose action
        action = env.get_expert_action()
        
        # Take action
        instruction, obs, reward, done, actions = env.step(action)
        
        # Update cumulative reward
        cumulative_reward += reward  
        
        # Store step in trajectory
        traj.append({
            'after': {
                'obs': {
                    'instruction': instruction,
                    'observation': obs
                },
                'reward': reward,
                'terminated': done,
            },
            'action': action,
            'action_scores': 0,
            'prompt': inp,
        })

        # Store for context window
        trajectory_steps.append({'obs': obs, 'action': action, 'reward': reward})  # ← Store reward for tracking
        
        if done:
            if reward > 0:
                won = True
            break
    
    return traj, trajectory_steps
    print(trajectory_steps)
    print(len(traj))
    # print(trajectory_steps)

def test_load_dataset():
    dataset = FeedbackDataset(feedback_data=None,load_path='./lfm/feedback_data')
    print(len(dataset.train_examples))
    train_loader = dataset.get_train_loader(2)
    for input in train_loader:
        if input[0]['target'] == 'Yes':
            print(f"Input: {input[0]['input']}, target {input[0]['target']}")

if __name__ == "__main__":
    # trajectories = []
    # env = VerbalizedALFWorld()
    # for _ in range(10):
    #     traj, traj_steps = test_rollout(env)
    #     trajectories.append(traj)
    # feedback = get_all_feedback(trajectories)
    # print(feedback)
    test_load_dataset()