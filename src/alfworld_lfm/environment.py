"""
ALFWorld environment wrapper
"""

import os
import sys
from pathlib import Path

class VerbalizedALFWorld:
    """
    Wrapper for ALFWorld, provides verbalized observations
    - ALFWorld already text-based
    - To verbalize we just return observation as is
    - Environment provides (instruction, observation, admissible_actions)
    """

    def __init__(self, split='train', max_episode_steps=50):
        self.split = split
        self.max_episode_steps = max_episode_steps
        self._expert_plan = None
        self._expert_step_idx = 0
        self._gold_action = None

        # Set data path
        # At the top of __init__ in environment.py
        cluster_path = os.path.expanduser('~/COS435_FinalProject/data')
        mac_path = os.path.expanduser('~/COS435/COS435_FinalProject/data')
        data_path = cluster_path if os.path.exists(cluster_path) else mac_path

        os.environ['ALFWORLD_DATA'] = data_path

        # Create data directory
        os.makedirs(data_path, exist_ok=True)
        
        # Path to your config file
        # Try cluster path first, fall back to Mac path
        cluster_path = Path.home() / 'COS435_FinalProject' / 'configs' / 'alfworld_config.yaml'
        mac_path = Path.home() / 'COS435' / 'COS435_FinalProject' / 'configs' / 'alfworld_config.yaml'
        config_path = cluster_path if cluster_path.exists() else mac_path
        
        # Set sys.argv for ALFWorld's config loader
        sys.argv = [sys.argv[0], str(config_path)]
        
        # Load config using ALFWorld's loader
        from alfworld.agents.modules import generic
        self.config = generic.load_config()
        print("Config keys:", self.config.keys())  
        print("Full config:", self.config)  

        # Manually resolve ${ALFWORLD_DATA} since config doesn't auto-expand it
        self.config['dataset']['data_path'] = f"{data_path}/json_2.1.1/train"
        self.config['dataset']['eval_id_data_path'] = f"{data_path}/json_2.1.1/valid_seen"
        self.config['dataset']['eval_ood_data_path'] = f"{data_path}/json_2.1.1/valid_unseen"

        
        # Override split-specific settings
        if split == 'train':
            self.config['env']['goal_desc_human_anns_prob'] = 0.0
        else:
            self.config['env']['goal_desc_human_anns_prob'] = 1.0
        
        # Create the environment
        from alfworld.agents.environment import get_environment
        env_type = self.config['env'].get('type', 'AlfredTWEnv')
        self._env = get_environment(env_type)(self.config, train_eval=split)
        self._env = self._env.init_env(batch_size=1)
        self.current_instruction = None

    def num_settings(self):
        """Return number of available environments in current split."""
        if hasattr(self._env, 'num_games'):
            return self._env.num_games
        elif self.split == 'train':
            return 3553
        elif self.split == 'eval_out_of_distribution':
            return 134
        else:
            return 500

    def reset(self):
        obs, info = self._env.reset()
        obs_text = obs[0]
        
        # Initialize expert plan from info
        raw_plan = info.get('extra.expert_plan', [])
        # Flatten: [['look']] -> ['look']
        if raw_plan and isinstance(raw_plan, list) and len(raw_plan) > 0:
            if isinstance(raw_plan[0], list):
                self._expert_plan = [item[0] for item in raw_plan if item]
            else:
                self._expert_plan = raw_plan
        else:
            self._expert_plan = []
        
        self._expert_step_idx = 0
        self._gold_action = self._expert_plan[self._expert_step_idx] if self._expert_plan else None
        
        # Parse instruction and observation
        lines = obs_text.strip().split('\n')
        instruction = ""
        verb_lines = []
        for line in lines:
            if 'Your task is to:' in line:
                instruction = line.replace('Your task is to: ', '')
            else:
                verb_lines.append(line)
        
        self.current_instruction = instruction
        verbalized_obs = '\n'.join(verb_lines)
        
        # Flatten admissible actions
        admissible_actions = info.get('admissible_commands', [])
        if admissible_actions and isinstance(admissible_actions[0], list):
            admissible_actions = admissible_actions[0]
        
        return self.current_instruction, verbalized_obs, admissible_actions

    def step(self, action):
        # Ensure action is a string
        if isinstance(action, list):
            action = action[0] if action else "look"
        
        obs, reward, done, info = self._env.step([action])
        obs_text = obs[0]
        
        # Update expert plan from info (flatten if needed)
        # ALFWorld updates expert_plan to be the REMAINING plan after each step
        # so we always read index 0 of the updated plan
        if 'extra.expert_plan' in info:
            raw_plan = info.get('extra.expert_plan', [])
            if raw_plan and isinstance(raw_plan, list) and len(raw_plan) > 0:
                if isinstance(raw_plan[0], list):
                    self._expert_plan = [item[0] for item in raw_plan if item]
                else:
                    self._expert_plan = raw_plan
            else:
                self._expert_plan = []
            
            # Always read index 0 since ALFWorld removes consumed actions from front
            self._gold_action = self._expert_plan[0] if self._expert_plan else None
        
        # Parse observation
        lines = obs_text.strip().split('\n')
        verb_lines = [line for line in lines if 'Your task is to:' not in line]
        verbalized_obs = '\n'.join(verb_lines)
        
        # Flatten admissible actions
        admissible_actions = info.get('admissible_commands', [])
        if admissible_actions and isinstance(admissible_actions[0], list):
            admissible_actions = admissible_actions[0]
        
        # Handle reward and done (batch returns)
        reward_value = reward[0] if isinstance(reward, tuple) else reward
        done_value = done[0] if isinstance(done, tuple) else done
        
        return self.current_instruction, verbalized_obs, float(reward_value), done_value, admissible_actions

    def verbalize(self, observation):
        # ALFWorld observations are already verbalized --> return directly
        return observation

    def get_expert_action(self):
        """Return the current expert action as a string"""
        if self._gold_action:
            # If it's a list of lists, extract the inner string
            if isinstance(self._gold_action, list):
                if len(self._gold_action) > 0:
                    if isinstance(self._gold_action[0], list):
                        return self._gold_action[0][0] if self._gold_action[0] else None
                    else:
                        return self._gold_action[0] if self._gold_action else None
            return self._gold_action
        return None

if __name__ == "__main__":
    print("Testing ALFWorld verbalization!")
    env = VerbalizedALFWorld(split='train')
    instruction, obs, admissible_actions = env.reset()
    print("Instruction:", instruction)
    print("Observation:", obs)
    print("Number of admissible actions:", len(admissible_actions))