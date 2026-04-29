"""
ACTPRED Baseline - uses Mistral LLM to predict actions directly
"""

import os
import re
import logging
import torch
import tenacity as T
from transformers import AutoTokenizer, AutoModelForCausalLM
from base_train import BaseExpertDataset, train_imitation_learning


class ACTPREDDataset(BaseExpertDataset):
    """ACTPRED: Uses Mistral LLM to predict actions."""
    
    def __init__(self, env, num_episodes=500, max_steps=50, context_window=20, 
                 load_path=None, llm_model_name="mistralai/Mistral-7B-Instruct-v0.2",
                 use_8bit=True):
        
        # Store env reference for action_to_string
        self.env = env
        
        # Load Mistral model
        print(f"Loading Mistral model: {llm_model_name}")
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
        
        if use_8bit:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                llm_model_name, quantization_config=bnb_config, 
                device_map="auto", torch_dtype=torch.float16
            )
        else:
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                llm_model_name, device_map="auto", torch_dtype=torch.float16
            )
        
        self.llm_model.eval()
        self.max_new_tokens = 50
        self.temperature = 0.0
        
        # Call parent to collect data
        super().__init__(env, num_episodes, max_steps, context_window, load_path)
    
    def _action_to_string(self, action):
        """Convert structured action to string"""
        if action is None:
            return None
        if isinstance(action, str):
            return action
        if isinstance(action, list):
            return ' '.join(action)
        return ' '.join(action) if isinstance(action, (tuple, list)) else str(action)
    
    @T.retry(stop=T.stop_after_attempt(20), wait=T.wait_fixed(10),
             after=lambda s: logging.error(f"Mistral query failed: {s}"))
    def _query_mistral_with_retry(self, prompt):
        """Query Mistral with retry logic"""
        formatted_prompt = f"<s>[INST] {prompt} [/INST]"
        inputs = self.llm_tokenizer(formatted_prompt, return_tensors="pt").to(self.llm_model.device)
        
        with torch.no_grad():
            outputs = self.llm_model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, temperature=self.temperature,
                do_sample=False, pad_token_id=self.llm_tokenizer.eos_token_id,
            )
        
        response = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split("[/INST]")[-1].strip()
    
    def _parse_action_from_response(self, response, admissible_actions):
        """Parse Mistral response into valid action string"""
        response_lower = response.lower().strip()
        
        for action in admissible_actions:
            if isinstance(action, tuple):
                action_str = self._action_to_string(action).lower()
            else:
                action_str = action.lower()
            
            if action_str in response_lower or response_lower in action_str:
                return action_str if isinstance(action, str) else self._action_to_string(action)
        
        # Fallback: return first admissible action
        if admissible_actions:
            first = admissible_actions[0]
            return first if isinstance(first, str) else self._action_to_string(first)
        
        return None
    
    def _get_expert_action(self, env, instruction, obs, actions, trajectory):
        """Query Mistral to predict next action"""
        prompt = self._build_actpred_prompt(instruction, obs, actions, trajectory)
        full_prompt = prompt + "\nYou decide to: "
        
        try:
            llm_response = self._query_mistral_with_retry(full_prompt)
        except Exception as e:
            print(f"Mistral query failed after retries: {e}")
            return actions[0] if actions and isinstance(actions[0], str) else None
        
        parsed_action = self._parse_action_from_response(llm_response, actions)
        return parsed_action if parsed_action else (actions[0] if actions else None)
    
    def _build_actpred_prompt(self, instruction, obs, actions, trajectory):
        """Build prompt matching original get_prompt()"""
        prompt_parts = [f"Your task is: {instruction}", ""]
        
        # Include ALL steps from trajectory
        for step in trajectory:
            prompt_parts.append(f"You see: {step['obs']}")
            prompt_parts.append(f"You decide to: {step['action']}")
            if step.get('reward') is not None:
                prompt_parts.append(f"You got a reward of: {step['reward']}")
            prompt_parts.append("")
        
        # Add current observation
        prompt_parts.append(f"You see: {obs}")
        prompt_parts.append("")
        
        # Add ALL admissible actions
        prompt_parts.append("Available actions:")
        for action in actions:
            action_str = action if isinstance(action, str) else self._action_to_string(action)
            prompt_parts.append(f"- {action_str}")
        
        prompt_parts.append("")
        prompt_parts.append("What do you decide to do?")
        prompt_parts.append("You decide to: ")
        
        return "\n".join(prompt_parts)
        
    def __del__(self):
        if hasattr(self, 'llm_model'):
            self.llm_model.to('cpu')
            del self.llm_model
            torch.cuda.empty_cache()


def train_actpred():
    train_imitation_learning(
        dataset_class=ACTPREDDataset,
        model_name="google/flan-t5-large",
        dataset_path="./src/alfworld_lfm/data/actpred_dataset_500eps.pkl",
        model_save_path="./models/actpred_alfworld",
        num_episodes=500,
        max_steps=50,
        context_window=20,
        num_epochs=20,
        batch_size=20,
        accumulate_grad_batches=10,
        learning_rate=5e-5,
        val_interval=200,
        grad_clip=5.0,
        max_len_input=2048,
        max_len_output=16,
    )


if __name__ == "__main__":
    train_actpred()