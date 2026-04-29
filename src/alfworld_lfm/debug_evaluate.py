"""
Debug evaluate pipeline locally without GPU or trained model.
Tests: environment reset, expert rollout, reward detection, trajectory saving.
Run with: python debug_evaluate.py
"""

import os
import bz2
import json
from pathlib import Path
from environment import VerbalizedALFWorld
from utils import format_context
# from evaluate import load_trajectories

def debug_evaluate(num_episodes=3, max_steps=50, output_dir='./debug_trajectories'):
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Creating environment...")
    env = VerbalizedALFWorld(split='train')  # train loads faster than eval
    
    outcomes = []
    
    for episode_idx in range(num_episodes):
        print(f"\n--- Episode {episode_idx+1}/{num_episodes} ---")
        
        instruction, obs, actions = env.reset()
        print(f"Instruction: {instruction[:80]}...")
        print(f"Actions available: {len(actions)}")
        
        traj = [{
            'after': {
                'obs': {'instruction': instruction, 'observation': obs},
                'reward': None,
                'terminated': None,
            },
            'action': None,
            'action_scores': None,
            'prompt': None,
        }]
        
        trajectory_steps = []
        cumulative_reward = 0.0
        won = False
        
        for step in range(max_steps):
            # Use expert instead of trained model -- no GPU needed
            action = env.get_expert_action()
            
            if action is None:
                print(f"  Step {step}: No expert action, taking first admissible")
                action = actions[0] if actions else "look"
            
            recent_context = trajectory_steps[-20:]
            inp = format_context(instruction, recent_context, obs, cumulative_reward)
            
            instruction, obs, reward, done, actions = env.step(action)
            cumulative_reward += reward
            
            # Print every step so you can see exactly what's happening
            print(f"  Step {step+1}: action='{action}' | "
                  f"reward={reward} | done={done} | "
                  f"cumulative={cumulative_reward}")
            
            traj.append({
                'after': {
                    'obs': {'instruction': instruction, 'observation': obs},
                    'reward': reward,
                    'terminated': done,
                },
                'action': action,
                'action_scores': None,
                'prompt': inp,
            })
            
            trajectory_steps.append({
                'obs': obs, 'action': action, 'reward': reward
            })
            
            if done:
                print(f"  Episode done at step {step+1}")
                print(f"  Final reward: {reward}")
                print(f"  Cumulative reward: {cumulative_reward}")
                if reward > 0:
                    won = True
                    print("  ✓ TASK COMPLETED")
                else:
                    print("  ✗ Task failed")
                break
        
        outcomes.append(won)
        
        # Save trajectory
        setting_id = getattr(env, 'curr_setting_id', f'episode_{episode_idx}')
        fout = output_dir / f'{setting_id}.json.bz2'
        with bz2.open(fout, 'wt') as f:
            json.dump({
                'traj': traj,
                'setting_id': setting_id,
                'won': won,
            }, f, indent=2)
        print(f"  Saved trajectory to {fout}")
    
    # Summary
    completion_rate = sum(outcomes) / len(outcomes) if outcomes else 0
    print(f"\n{'='*50}")
    print(f"Episodes: {len(outcomes)}")
    print(f"Completed: {sum(outcomes)}")
    print(f"Completion rate: {completion_rate:.1%}")
    print(f"Trajectories saved to: {output_dir}")
    
    return outcomes

if __name__ == "__main__":
    outcomes = debug_evaluate(
        num_episodes=3,
        max_steps=50,
        output_dir='./debug_trajectories'
    )