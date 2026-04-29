"""
Evaluate trained BC/ACTPRED/LFM models on ALFWorld.
Computes task completion rate (paper's main metric from Table 3).
Saves trajectories for analysis.
"""

import os
import json
import bz2
import torch
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from environment import VerbalizedALFWorld
from policy import Policy
from utils import format_context


def evaluate_model(model_path, split='eval_out_of_distribution', 
                   max_steps=80, context_window=20, aggregation='mean',
                   num_episodes=None, save_trajectories=True, output_dir=None):
    """
    Evaluate a trained model on ALFWorld environments.
    
    Args:
        model_path: Path to saved model (e.g., "./models/bc_alfworld_best")
        split: Which environment split to use
        max_steps: Maximum steps per episode
        context_window: Number of previous steps to include (paper uses 20)
        aggregation: 'mean' for perplexity averaging
        num_episodes: Number of episodes to evaluate (None = all)
        save_trajectories: Whether to save trajectories to disk
        output_dir: Directory to save trajectories (default: ./eval_results/)
    
    Returns:
        task_completion_rate: float (e.g., 0.626 for 62.6%)
        outcomes: list of booleans (True = task completed)
    """
    
    # Setup output directory
    if save_trajectories:
        if output_dir is None:
            model_name = Path(model_path).name
            output_dir = Path(f"./eval_results/{model_name}_{split}")
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving trajectories to {output_dir}")
    
    # Load policy
    print(f"Loading model from {model_path}")
    policy = Policy(model_path)
    
    # Create environment
    print(f"Creating environment (split={split})")
    env = VerbalizedALFWorld(split=split)
    
    # Get number of episodes to evaluate
    if num_episodes is None:
        num_episodes = env.num_settings()
    print(f"Evaluating on {num_episodes} environments")
    
    outcomes = []
    trajectories = []
    progress_bar = tqdm(range(num_episodes), desc="Evaluating")
    
    for episode_idx in progress_bar:
        try:
            # Reset environment
            instruction, obs, actions = env.reset()
        except Exception as e:
            print(f"Error resetting environment: {e}")
            continue
        
        # Get setting ID (environment identifier)
        setting_id = getattr(env, 'curr_setting_id', f'episode_{episode_idx}')
        
        # Store trajectory for this episode
        traj = []
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
            action, scores = policy.choose_action(inp, actions, aggregation=aggregation, sample=False)
            print(f"Action chosen: {action}")
            # print(f"Scores: {scores}")
            
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
                'action_scores': scores,
                'prompt': inp,
            })

            # Store for context window
            trajectory_steps.append({'obs': obs, 'action': action, 'reward': reward})  # ← Store reward for tracking
            
            if done:
                if reward > 0:
                    won = True
                break
        
        outcomes.append(won)
        
        # Save trajectory to disk
        if save_trajectories:
            fout = output_dir / f'{setting_id}.json.bz2'
            with bz2.open(fout, 'wt') as f:
                json.dump({
                    'traj': traj,
                    'setting_id': setting_id,
                    'won': won,
                }, f, indent=2)
        
        trajectories.append({
            'setting_id': setting_id,
            'won': won,
            'num_steps': len(traj) - 1,
        })
        
        success_rate = sum(outcomes) / len(outcomes)
        progress_bar.set_description(f"Success: {success_rate:.3f} ({sum(outcomes)}/{len(outcomes)})")
    
    # Save summary file
    if save_trajectories:
        summary_path = output_dir / 'summary.json'
        with open(summary_path, 'wt') as f:
            json.dump({
                'model_path': str(model_path),
                'split': split,
                'num_episodes': len(outcomes),
                'tasks_completed': sum(outcomes),
                'task_completion_rate': sum(outcomes) / len(outcomes) if outcomes else 0.0,
                'outcomes': outcomes,
                'trajectories': trajectories,
            }, f, indent=2)
        print(f"Saved summary to {summary_path}")
    
    # Compute final metrics
    task_completion_rate = sum(outcomes) / len(outcomes) if outcomes else 0.0
    
    print(f"\n{'='*50}")
    print(f"Evaluation complete!")
    print(f"Environments evaluated: {len(outcomes)}")
    print(f"Tasks completed: {sum(outcomes)}")
    print(f"Task completion rate: {task_completion_rate:.3f} ({task_completion_rate*100:.1f}%)")
    if save_trajectories:
        print(f"Trajectories saved to: {output_dir}")
    print(f"{'='*50}")
    
    return task_completion_rate, outcomes

def load_trajectories(output_dir):
    """Load saved trajectories for analysis."""
    output_dir = Path(output_dir)
    trajectories = []
    
    for f in output_dir.glob('*.json.bz2'):
        with bz2.open(f, 'rt') as file:
            data = json.load(file)
            trajectories.append(data)
    
    return trajectories


def evaluate_all_models():
    """Evaluate all trained models and compare to paper's Table 3."""
    
    models = {
        "BC": "./models/bc_alfworld_best",
        "BC_final": "./models/bc_alfworld_final",
        # Add more as you train them:
        # "ACTPRED": "./models/actpred_best",
        # "LFM": "./models/lfm_best",
    }
    
    results = {}
    for name, path in models.items():
        if os.path.exists(path):
            print(f"\n{'='*50}")
            print(f"Evaluating {name}")
            print(f"{'='*50}")
            rate, _ = evaluate_model(path, save_trajectories=True)
            results[name] = rate
        else:
            print(f"Skipping {name} - model not found at {path}")
    
    # Print comparison table
    print("\n" + "="*60)
    print("RESULTS COMPARISON")
    print("="*60)
    print(f"{'Method':<15} {'Our Result':<15} {'Paper (Table 3)':<15}")
    print("-"*60)
    
    paper_results = {
        "BC": 0.626,
        "ACTPRED": 0.560,
        "LFM": 0.641,
        "LFMA": 0.746,
    }
    
    for name, our_result in results.items():
        paper = paper_results.get(name, 0.0)
        diff = our_result - paper
        status = "✓" if abs(diff) < 0.05 else "⚠️"
        print(f"{name:<15} {our_result:.3f} ({our_result*100:.1f}%)     {paper:.3f} ({paper*100:.1f}%)     {status}")
    
    print("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate a trained model')
    parser.add_argument('--model_path', type=str, default='./models/bc_alfworld_final',
                        help='Path to saved model')
    parser.add_argument('--split', type=str, default='eval_out_of_distribution',
                        help='Environment split (train/eval_out_of_distribution)')
    parser.add_argument('--num_episodes', type=int, default=None,
                        help='Number of episodes to evaluate (default: all)')
    parser.add_argument('--all', action='store_true',
                        help='Evaluate all trained models')
    parser.add_argument('--no_save', action='store_true',
                        help='Do not save trajectories')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save trajectories')
    
    args = parser.parse_args()
    
    if args.all:
        evaluate_all_models()
    else:
        evaluate_model(
            args.model_path, 
            split=args.split, 
            num_episodes=args.num_episodes,
            save_trajectories=not args.no_save,
            output_dir=args.output_dir
        )