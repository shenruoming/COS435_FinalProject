"""
Base classes for imitation learning baselines (BC, ActPred, LFM)
We have this shared functionality to avoid code duplication
"""

import os
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup
from tqdm import tqdm
from environment import VerbalizedALFWorld
from utils import format_context

# BaseExpertDataset takes from bc_train.py but is now shared by all baselines

class BaseExpertDataset(Dataset):
    """
    Base class for collecting expert demonstrations.
    Subclasses must implement _get_expert_action() method.
    """
    
    def __init__(self, env, num_episodes=20, max_steps=50, context_window=20,
                 load_path=None, include_feedback=False, feedback_data=None):
        self.context_window = context_window
        
        if load_path and os.path.exists(load_path):
            print(f"Loading existing dataset from {load_path}")
            self.load(load_path)
            if include_feedback:
                self.train_examples.extend(feedback_data)
            return
        
        self.examples = []
        
        print(f"Collecting {num_episodes} expert demonstrations")
        for episode in tqdm(range(num_episodes)):
            instruction, obs, actions = env.reset()
            
            if actions and isinstance(actions[0], list):
                actions = actions[0]
            
            trajectory = []
            done = False
            step = 0
            reward = 0.0
            
            while not done and step < max_steps:
                # Subclasses implement this method
                expert_action = self._get_expert_action(env, instruction, obs, actions, trajectory)
                
                if expert_action and expert_action in actions:
                    # Add current step to trajectory BEFORE taking action
                    # trajectory.append({'obs': obs, 'action': expert_action})
                    
                    # Use most recent context_window steps as input
                    recent_context = trajectory[-context_window:] if len(trajectory) >= context_window else trajectory
                    
                    # Format observation with context
                    context_obs = format_context(instruction, recent_context, obs)
                    
                    self.examples.append({
                        'input': context_obs,
                        'target': expert_action
                    })
                                    
                    # Add current step to trajectory BEFORE taking action
                    trajectory.append({'obs': obs, 'action': expert_action, 'reward': reward})
                    
                    # Take the expert action
                    instruction, obs, reward, done, actions = env.step(expert_action)
                    
                    if actions and isinstance(actions[0], list):
                        actions = actions[0]
                    
                    step += 1
                    continue
                
                # Take random action if expert action not available/invalid
                if actions:
                    print("expert action not available, taking random")
                    first_action = actions[0] if isinstance(actions[0], str) else actions[0][0]
                    instruction, obs, reward, done, actions = env.step(first_action)
                    if actions and isinstance(actions[0], list):
                        actions = actions[0]
                    step += 1
                else:
                    break
        
        # Split into train/val (80/20)
        # See 5.3 in paper - Experiment Details
        np.random.seed(42)
        indices = np.random.permutation(len(self.examples))
        split_idx = int(0.8 * len(self.examples))
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        self.train_examples = [self.examples[i] for i in train_indices]
        self.val_examples = [self.examples[i] for i in val_indices]
        
        print(f"Collected {len(self.examples)} examples")
        print(f"Train: {len(self.train_examples)}, Val: {len(self.val_examples)}")
    
    def _get_expert_action(self, env, instruction, obs, actions, trajectory):
        """
        We implement this in our subclasses 
        BC: returns env.get_expert_action()
        ActPred: returns Mistral-7B query result
        LFM: returns LFM prediction
        """
        raise NotImplementedError("Subclasses must implement _get_expert_action()")
    
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'train_examples': self.train_examples,
                'val_examples': self.val_examples,
                'context_window': self.context_window
            }, f)
        print(f"Saved dataset to {path}")
    
    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.train_examples = data['train_examples']
        self.val_examples = data['val_examples']
        self.examples = self.train_examples + self.val_examples
        self.context_window = data.get('context_window', 20)
        print(f"Loaded {len(self.examples)} examples from {path}")
        print(f"Train: {len(self.train_examples)}, Val: {len(self.val_examples)}")
    
    def get_train_loader(self, batch_size):
        class SimpleDataset(Dataset):
            def __init__(self, data):
                self.data = data
            def __len__(self):
                return len(self.data)
            def __getitem__(self, idx):
                return self.data[idx]
        
        def collate_fn(batch):
            return batch
        
        return DataLoader(SimpleDataset(self.train_examples), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    def get_val_loader(self, batch_size):
        class SimpleDataset(Dataset):
            def __init__(self, data):
                self.data = data
            def __len__(self):
                return len(self.data)
            def __getitem__(self, idx):
                return self.data[idx]
                
        def collate_fn(batch):
            return batch
        
        return DataLoader(SimpleDataset(self.val_examples), batch_size=batch_size, collate_fn=collate_fn)

# For these default configs see conf / bc.yaml in original codebase
def train_imitation_learning(dataset_class, model_name, dataset_path, model_save_path, 
                              num_episodes=500, max_steps=50, context_window=20,
                              num_epochs=20, batch_size=20, accumulate_grad_batches=10,
                              learning_rate=5e-5, val_interval=200, grad_clip=5.0,
                              max_len_input=2048, max_len_output=16):
    """
    Generic training function for imitation learning baselines (BC, ActPred, LFM)
    """
    
    env = VerbalizedALFWorld(split='train')
    
    dataset = dataset_class(env, num_episodes=num_episodes, max_steps=max_steps,
                            context_window=context_window, load_path=dataset_path)
    dataset.save(dataset_path)
    
    train_loader = dataset.get_train_loader(batch_size)
    val_loader = dataset.get_val_loader(batch_size)
    
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0, 
        num_training_steps=num_epochs * (len(train_loader) // accumulate_grad_batches)
    )

    # Training loop
    print(f"Starting training for {num_epochs} epochs")
    model.train()
    
    best_val_loss = float('inf')
    global_step = 0

    for epoch in range(num_epochs):
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            if not isinstance(batch, list):
                batch = list(batch)
            
            inputs_text = [b['input'] for b in batch]
            targets_text = [b['target'] for b in batch]
            
            inputs = tokenizer(inputs_text, padding=True, truncation=True, 
                              max_length=max_len_input, return_tensors='pt').to(device)
            targets = tokenizer(targets_text, padding=True, truncation=True, 
                               max_length=max_len_output, return_tensors='pt').to(device)
            
            labels = targets['input_ids'].clone()
            labels[labels == tokenizer.pad_token_id] = -100
            
            outputs = model(input_ids=inputs['input_ids'], 
                           attention_mask=inputs['attention_mask'], 
                           labels=labels)
            
            loss = outputs.loss / accumulate_grad_batches
            loss.backward()
            
            epoch_loss += loss.item() * accumulate_grad_batches
            
            if (batch_idx + 1) % accumulate_grad_batches == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                if global_step % val_interval == 0 and len(val_loader) > 0:
                    model.eval()
                    val_losses = []
                    with torch.no_grad():
                        for val_batch in val_loader:
                            if not isinstance(val_batch, list):
                                val_batch = list(val_batch)
                            
                            val_inputs = tokenizer([b['input'] for b in val_batch], 
                                                   padding=True, truncation=True, 
                                                   max_length=max_len_input, return_tensors='pt').to(device)
                            val_targets = tokenizer([b['target'] for b in val_batch], 
                                                    padding=True, truncation=True, 
                                                    max_length=max_len_output, return_tensors='pt').to(device)
                            val_labels = val_targets['input_ids'].clone()
                            val_labels[val_labels == tokenizer.pad_token_id] = -100
                            
                            val_outputs = model(input_ids=val_inputs['input_ids'], 
                                               attention_mask=val_inputs['attention_mask'], 
                                               labels=val_labels)
                            val_losses.append(val_outputs.loss.item())
                    
                    avg_val_loss = np.mean(val_losses)
                    progress_bar.write(f"Step {global_step}: Val loss = {avg_val_loss:.4f}")
                    
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        os.makedirs(model_save_path, exist_ok=True)
                        model.save_pretrained(f"{model_save_path}_best")
                        tokenizer.save_pretrained(f"{model_save_path}_best")
                        progress_bar.write(f"  saved new best model")
                    
                    model.train()
            
            progress_bar.set_postfix({'loss': f'{loss.item() * accumulate_grad_batches:.4f}'})
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} average loss: {avg_epoch_loss:.4f}")
        os.makedirs(model_save_path, exist_ok=True)

    model.save_pretrained(f"{model_save_path}_final")
    tokenizer.save_pretrained(f"{model_save_path}_final")
    print(f"\nBest val loss: {best_val_loss:.4f}")