"""
Behavioral Cloning baseline from paper Section 5.3
Paper: 10k steps, batch 20, lr 5e-5, early stopping
"""

import os
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
from environment import VerbalizedALFWorld
import glob


class ExpertDataset(Dataset):
    """
    Collect expert demonstrations with 20-step window context
    Paper: "Verbalized observations v contain the most recent 20 steps"
    """
    def __init__(self, env, num_episodes=500, max_steps=50, context_window=20, load_path=None):
        self.context_window = context_window
        
        # Try to load from disk first
        if load_path and os.path.exists(load_path):
            print(f"Loading existing dataset from {load_path}")
            self.load(load_path)
            return
        
        # Otherwise collect new data
        self.examples = []
        
        print(f"Collecting {num_episodes} expert demonstrations...")
        for episode in tqdm(range(num_episodes)):
            instruction, obs, actions = env.reset()
            
            # Flatten actions if needed
            if actions and isinstance(actions[0], list):
                actions = actions[0]
            
            # Store trajectory history for context window
            trajectory = []
            done = False
            step = 0
            
            while not done and step < max_steps:
                expert_action = env.get_expert_action()
                
                # Handle expert action (could be list or string)
                if expert_action:
                    if isinstance(expert_action, list):
                        expert_action = expert_action[0] if expert_action else None
                    
                    if expert_action and expert_action in actions:
                        # Add current step to trajectory BEFORE taking action
                        trajectory.append({'obs': obs, 'action': expert_action})
                        
                        # Use most recent context_window steps as input
                        recent_context = trajectory[-context_window:] if len(trajectory) >= context_window else trajectory
                        
                        # Format observation with context
                        context_obs = self._format_context(instruction, recent_context, obs)
                        
                        self.examples.append({
                            'input': context_obs,
                            'target': expert_action
                        })
                        
                        # Take the expert action
                        instruction, obs, reward, done, actions = env.step(expert_action)
                        
                        # Flatten actions after step
                        if actions and isinstance(actions[0], list):
                            actions = actions[0]
                        
                        step += 1
                        continue
                
                # Fallback: take random action if expert action not available/invalid
                if actions:
                    # Take first action
                    first_action = actions[0] if isinstance(actions[0], str) else actions[0][0]
                    instruction, obs, reward, done, actions = env.step(first_action)
                    if actions and isinstance(actions[0], list):
                        actions = actions[0]
                    step += 1
                else:
                    break
        
        # Split into train/val (80/20) for early stopping
        np.random.seed(42)
        indices = np.random.permutation(len(self.examples))
        split_idx = int(0.8 * len(self.examples))
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        self.train_examples = [self.examples[i] for i in train_indices]
        self.val_examples = [self.examples[i] for i in val_indices]
        
        print(f"Collected {len(self.examples)} examples")
        print(f"Train: {len(self.train_examples)}, Val: {len(self.val_examples)}")
    
    def _format_context(self, instruction, trajectory, current_obs):
        """Format the observation with most recent steps"""
        if not trajectory:
            return f"Task: {instruction}\n\nCurrent observation: {current_obs}"
        
        context_str = ""
        for i, step in enumerate(trajectory):
            # Truncate long observations
            obs_short = step['obs'][:300] + "..." if len(step['obs']) > 300 else step['obs']
            context_str += f"Step {i+1}: {obs_short}\nAction: {step['action']}\n\n"
        
        # Truncate current observation
        obs_short = current_obs[:300] + "..." if len(current_obs) > 300 else current_obs
        
        return f"Task: {instruction}\n\nPrevious steps:\n{context_str}\nCurrent observation: {obs_short}"
    
    def save(self, path):
        """Save collected examples to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'train_examples': self.train_examples,
                'val_examples': self.val_examples,
                'context_window': self.context_window
            }, f)
        print(f"Saved dataset to {path}")
    
    def load(self, path):
        """Load saved examples from disk"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.train_examples = data['train_examples']
        self.val_examples = data['val_examples']
        self.examples = self.train_examples + self.val_examples
        self.context_window = data.get('context_window', 20)
        print(f"Loaded {len(self.examples)} examples from {path}")
        print(f"Train: {len(self.train_examples)}, Val: {len(self.val_examples)}")
    
    def get_train_loader(self, batch_size):
        """Return DataLoader for training examples"""
        class SimpleDataset(Dataset):
            def __init__(self, data):
                self.data = data
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                item = self.data[idx]
                return {'input': item['input'], 'target': item['target']}
        
        def collate_fn(batch):
            return batch  # Return list of dicts as-is
        
        return DataLoader(
            SimpleDataset(self.train_examples), 
            batch_size=batch_size, 
            shuffle=True,
            collate_fn=collate_fn
        )

    def get_val_loader(self, batch_size):
        """Return DataLoader for validation examples"""
        class SimpleDataset(Dataset):
            def __init__(self, data):
                self.data = data
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                item = self.data[idx]
                return {'input': item['input'], 'target': item['target']}
        
        def collate_fn(batch):
            return batch
        
        return DataLoader(
            SimpleDataset(self.val_examples), 
            batch_size=batch_size,
            collate_fn=collate_fn
        )


def train_bc():
    """Train BC baseline with paper hyperparameters"""
    
    # Paper hyperparameters (Section 5.3)
    # See config files from authors' GitHub
    MODEL_NAME = "google/flan-t5-large"  # 770M
    BATCH_SIZE = 20  
    ACCUMULATE_GRAD_BATCHES = 10
    LEARNING_RATE = 5e-5
    NUM_EPOCHS = 20
    NUM_EPISODES = 500  
    CONTEXT_WINDOW = 20
    VAL_INTERVAL = 200  # Validate every 200 steps
    GRAD_CLIP = 5.0
    MAX_LEN_INPUT = 2048
    MAX_LEN_OUTPUT = 16
    
    CHECKPOINT_DIR = "./src/alfworld_lfm/models/checkpoints"
    CHECKPOINT_EVERY = 500
    RESUME_CHECKPOINT = "./src/alfworld_lfm/models/checkpoints/latest.pt"  
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Path for saving dataset
    DATASET_PATH = "./src/alfworld_lfm/data/bc_dataset_500eps.pkl"
    
    # Create environment
    print("Initializing environment...")
    env = VerbalizedALFWorld(split='train')
    
    # Load or collect dataset
    dataset = ExpertDataset(
        env, 
        num_episodes=NUM_EPISODES, 
        context_window=CONTEXT_WINDOW,
        load_path=DATASET_PATH  # This will load if exists
    )
    
    # Save dataset for future use
    dataset.save(DATASET_PATH)
    
    # Create data loaders
    train_loader = dataset.get_train_loader(BATCH_SIZE)
    val_loader = dataset.get_val_loader(BATCH_SIZE)
    
    # Load model
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Paper uses learning rate scheduler
    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0, 
        num_training_steps=NUM_EPOCHS * (len(train_loader) // ACCUMULATE_GRAD_BATCHES)
    )

    # Training loop
    print(f"Starting training for {NUM_EPOCHS} epochs...")
    model.train()
    
    start_epoch = 0
    best_val_loss = float('inf')
    patience_counter = 0
    global_step = 0

    # Resume from checkpoint if exists
    if os.path.exists(RESUME_CHECKPOINT):
        print(f"Loading checkpoint from {RESUME_CHECKPOINT}")
        checkpoint = torch.load(RESUME_CHECKPOINT)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        start_epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']
        best_val_loss = checkpoint['best_val_loss']
        print(f"Resumed from epoch {start_epoch}, step {global_step}")

    for epoch in range(start_epoch, NUM_EPOCHS):
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Ensure batch is list of dicts
            if not isinstance(batch, list):
                batch = list(batch)
            
            # Extract inputs and targets
            inputs_text = [b['input'] for b in batch]
            targets_text = [b['target'] for b in batch]
            
            # Tokenize inputs (max_len=2048 from config)
            inputs = tokenizer(
                inputs_text,
                padding=True,
                truncation=True,
                max_length=MAX_LEN_INPUT,
                return_tensors='pt'
            ).to(device)
            
            # Tokenize targets (max_len=16 from config)
            targets = tokenizer(
                targets_text,
                padding=True,
                truncation=True,
                max_length=MAX_LEN_OUTPUT,
                return_tensors='pt'
            ).to(device)
            
            # Set labels (ignore padding)
            labels = targets['input_ids']
            labels[labels == tokenizer.pad_token_id] = -100
            
            # Forward pass
            outputs = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                labels=labels
            )
            
            loss = outputs.loss
            loss = loss / ACCUMULATE_GRAD_BATCHES  # Normalize for accumulation
            loss.backward()
            
            epoch_loss += loss.item() * ACCUMULATE_GRAD_BATCHES
            
            # Gradient accumulation
            if (batch_idx + 1) % ACCUMULATE_GRAD_BATCHES == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                # Validation
                if global_step % VAL_INTERVAL == 0 and len(val_loader) > 0:
                    model.eval()
                    val_losses = []
                    with torch.no_grad():
                        for val_batch in val_loader:
                            if not isinstance(val_batch, list):
                                val_batch = list(val_batch)
                            
                            val_inputs = tokenizer(
                                [b['input'] for b in val_batch],
                                padding=True, truncation=True, max_length=MAX_LEN_INPUT, return_tensors='pt'
                            ).to(device)
                            val_targets = tokenizer(
                                [b['target'] for b in val_batch],
                                padding=True, truncation=True, max_length=MAX_LEN_OUTPUT, return_tensors='pt'
                            ).to(device)
                            val_labels = val_targets['input_ids']
                            val_labels[val_labels == tokenizer.pad_token_id] = -100
                            
                            val_outputs = model(
                                input_ids=val_inputs['input_ids'],
                                attention_mask=val_inputs['attention_mask'],
                                labels=val_labels
                            )
                            val_losses.append(val_outputs.loss.item())
                    
                    avg_val_loss = np.mean(val_losses)
                    progress_bar.write(f"Step {global_step}: Val loss = {avg_val_loss:.4f}")
                    
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        os.makedirs("./models", exist_ok=True)
                        model.save_pretrained("./models/bc_alfworld_best")
                        tokenizer.save_pretrained("./models/bc_alfworld_best")
                        progress_bar.write(f"  New best model saved!")
                    
                    model.train()
            
                # Save checkpoint
                if global_step % CHECKPOINT_EVERY == 0:
                    checkpoint = {
                        'epoch': epoch,
                        'global_step': global_step,
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'scheduler_state': scheduler.state_dict(),
                        'best_val_loss': best_val_loss
                    }
                    torch.save(checkpoint, RESUME_CHECKPOINT)
                    progress_bar.write(f"  Checkpoint saved at step {global_step}")
            
            progress_bar.set_postfix({'loss': f'{loss.item() * ACCUMULATE_GRAD_BATCHES:.4f}'})
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} average loss: {avg_epoch_loss:.4f}")
        
    # Save final model
    os.makedirs("./models", exist_ok=True)
    model.save_pretrained("./models/bc_alfworld_final")
    tokenizer.save_pretrained("./models/bc_alfworld_final")
    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    train_bc()