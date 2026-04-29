from policy import Policy
from environment import VerbalizedALFWorld
from utils import format_context

policy = Policy("./models/bc_alfworld_best")
env = VerbalizedALFWorld(split='train')

# Run 3 full episodes
for episode in range(3):
    instruction, obs, actions = env.reset()
    trajectory_steps = []
    cumulative_reward = 0.0
    won = False
    
    print(f"\n{'='*50}")
    print(f"Episode {episode+1}")
    print(f"Instruction: {instruction}")
    print(f"{'='*50}")
    
    for step in range(50):
        recent_context = trajectory_steps[-20:]
        inp = format_context(instruction, recent_context, obs, cumulative_reward)
        
        action, scores = policy.choose_action(inp, actions)
        
        print(f"  Step {step+1}: chose '{action}'")
        
        instruction, obs, reward, done, actions = env.step(action)
        cumulative_reward += reward
        trajectory_steps.append({'obs': obs, 'action': action, 'reward': reward})
        
        if done:
            if reward > 0:
                won = True
                print(f"  ✓ COMPLETED in {step+1} steps")
            else:
                print(f"  ✗ Failed")
            break
    
    if not won:
        print(f"  ✗ Did not complete in 50 steps")
    
    # Show if model kept picking same actions
    from collections import Counter
    action_counts = Counter(s['action'] for s in trajectory_steps)
    print(f"  Action variety: {len(action_counts)} unique actions")
    print(f"  Most repeated: {action_counts.most_common(3)}")