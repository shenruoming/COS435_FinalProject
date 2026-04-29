from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup
from feedback_data import FeedbackDataset
import torch
from torch.optim import AdamW
import os
from tqdm import tqdm
import numpy as np
from evaluate_model import load_trajectories
from torch.nn import functional
from utils import convert_trajectories
from collections import defaultdict
import bz2
from pathlib import Path
import json
import argparse
from evaluate import load


MODEL_NAME = 'google/flan-t5-large'
BATCH_SIZE = 2
DATASET_PATH = './lfm/feedback_data_new'
LFM_MODEL_PATH = './models/lfm_final'
LFM_MODEL_SAVE_PATH = './models/lfm_best'
OUTPUT_PATH = './inferred_feedback/dev'
TRAJECTORIES_PATH = './eval_results/bc_alfworld_final_eval_out_of_distribution'
LEARNING_RATE = 5e-5
NUM_EPOCHS = 2
ACCUMULATE_GRAD_BATCHES = 8
MAX_LEN_INPUT = 512
MAX_LEN_OUTPUT = 32
VAL_INTERVAL = 100
GRAD_CLIP = 5.0

CHECKPOINT_DIR = "./src/alfworld_lfm/models/lfm/checkpoints"
CHECKPOINT_EVERY = 500
RESUME_CHECKPOINT = "./src/alfworld_lfm/models/lfm/checkpoints/latest.pt"  
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

F1 = load('f1')
Precision = load('precision')
Recall = load('recall')

class LanguageFeedbackModel:
    def __init__(self, model_path=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if model_path is not None:
            print(f"Loading model: {MODEL_NAME}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
    
            self.model.to(self.device)
            return

        print(f"Loading feedback dataset from {DATASET_PATH}")
        self.dataset = FeedbackDataset(
            load_path=DATASET_PATH  # This will load if exists
        )
        print("Dataset fetched")

        # Load model
        print(f"Loading model: {MODEL_NAME}")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
        self.model.to(self.device)


    def train(self):
        # Training loop
        print(f"Starting training for {NUM_EPOCHS} epochs...")
        self.model.train()

        # Create data loaders
        train_loader = self.dataset.get_train_loader(BATCH_SIZE)
        val_loader = self.dataset.get_val_loader(BATCH_SIZE)

        optimizer = AdamW(self.model.parameters(), lr=LEARNING_RATE)
    
        # Paper uses learning rate scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=0, 
            num_training_steps=NUM_EPOCHS * (len(train_loader) // ACCUMULATE_GRAD_BATCHES)
        )
        
        start_epoch = 0
        best_val_loss = float('inf')
        global_step = 0

        # Resume from checkpoint if exists
        if os.path.exists(RESUME_CHECKPOINT):
            print(f"Loading checkpoint from {RESUME_CHECKPOINT}")
            checkpoint = torch.load(RESUME_CHECKPOINT)
            self.model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            scheduler.load_state_dict(checkpoint['scheduler_state'])
            start_epoch = checkpoint['epoch']
            global_step = checkpoint['global_step']
            best_val_loss = checkpoint['best_val_loss']
            print(f"Resumed from epoch {start_epoch}, step {global_step}")
        
        for epoch in range(start_epoch, NUM_EPOCHS):
            global_step, best_val_loss = self.train_one_epoch(epoch, global_step, best_val_loss, train_loader, val_loader, optimizer, scheduler)

         # Save final model
        os.makedirs("./models", exist_ok=True)
        self.model.save_pretrained(LFM_MODEL_PATH)
        self.tokenizer.save_pretrained(LFM_MODEL_PATH)
        print(f"\nTraining complete! Best val loss: {best_val_loss}")
        print(f"Final model sdaved at {LFM_MODEL_PATH}")

    
    def train_one_epoch(self, epoch, global_step, best_val_loss, train_loader, val_loader, optimizer, scheduler):
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Ensure batch is list of dicts
            if not isinstance(batch, list):
                batch = list(batch)
            
            # Extract inputs and targets
            inputs_text = [b['input'] for b in batch]
            targets_text = [b['target'] for b in batch]
            
            # Tokenize input
            inputs = self.tokenizer(
                inputs_text,
                padding=True,
                truncation=True,
                max_length=MAX_LEN_INPUT,
                return_tensors='pt'
            ).to(self.device)
            
            # Tokenize targets
            targets = self.tokenizer(
                targets_text,
                padding=True,
                truncation=True,
                max_length=MAX_LEN_OUTPUT,
                return_tensors='pt'
            ).to(self.device)
            
            # Set labels (ignore padding)
            labels = targets['input_ids'].clone()
            labels[labels == self.tokenizer.pad_token_id] = -100
            
            # Forward pass
            outputs = self.model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                labels=labels,
                decoder_attention_mask=targets['attention_mask'],
            )
            
            loss = outputs.loss
            loss = loss / ACCUMULATE_GRAD_BATCHES  # Normalize for accumulation
            loss.backward()
            
            epoch_loss += loss.item() * ACCUMULATE_GRAD_BATCHES
            
            # Gradient accumulation
            if (batch_idx + 1) % ACCUMULATE_GRAD_BATCHES == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                # Validation
                if global_step % VAL_INTERVAL == 0 and len(val_loader) > 0:
                    best_val_loss = self.val_step(val_loader, best_val_loss, progress_bar, global_step)
            
                # Save checkpoint
                if global_step % CHECKPOINT_EVERY == 0:
                    checkpoint = {
                        'epoch': epoch,
                        'global_step': global_step,
                        'model_state': self.model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'scheduler_state': scheduler.state_dict(),
                        'best_val_loss': best_val_loss
                    }
                    torch.save(checkpoint, RESUME_CHECKPOINT)
                    progress_bar.write(f"  Checkpoint saved at step {global_step}")
            
            progress_bar.set_postfix({'loss': f'{loss.item() * ACCUMULATE_GRAD_BATCHES:.4f}'})
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} average loss: {avg_epoch_loss:.4f}")
        return global_step, best_val_loss
    
    def val_step(self, val_loader, best_val_loss, progress_bar, global_step):
        self.model.eval()
        val_losses = []
        with torch.no_grad():
            for val_batch in val_loader:
                if not isinstance(val_batch, list):
                    val_batch = list(val_batch)

                # Extract inputs and targets
                inputs_text = [b['input'] for b in val_batch]
                targets_text = [b['target'] for b in val_batch]
                
                # Tokenize input
                inputs = self.tokenizer(
                    inputs_text,
                    padding=True,
                    truncation=True,
                    max_length=MAX_LEN_INPUT,
                    return_tensors='pt'
                ).to(self.device)
            
                # Tokenize targets
                targets = self.tokenizer(
                    targets_text,
                    padding=True,
                    truncation=True,
                    max_length=MAX_LEN_OUTPUT,
                    return_tensors='pt'
                ).to(self.device)
                
                # Set labels (ignore padding)
                labels = targets['input_ids'].clone()
                labels[labels == self.tokenizer.pad_token_id] = -100
                
                # Forward pass
                outputs = self.model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    labels=labels,
                    decoder_attention_mask=targets['attention_mask'],
                )
                
                val_losses.append(outputs.loss.item())


                pred, _ = self.choose_action(val_batch)
                gold = [v['target'] == 'Yes' for v in val_batch]
                # gold = val_batch['label_boolean'].tolist()
                pred = [1 if x else 0 for x in pred]
                gold = [1 if x else 0 for x in gold]
                # print('val_f1', F1.compute(predictions=pred, references=gold, average='binary', pos_label=1)['f1'])
                # print('val_p', Precision.compute(predictions=pred, references=gold)['precision'])
                # print('val_r', Recall.compute(predictions=pred, references=gold)['recall'])

        avg_val_loss = np.mean(val_losses)
        progress_bar.write(f"Step {global_step}: Val loss = {avg_val_loss:.4f}")
                    
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(LFM_MODEL_SAVE_PATH, exist_ok=True)
            self.model.save_pretrained(f"{LFM_MODEL_SAVE_PATH}_best")
            self.tokenizer.save_pretrained(f"{LFM_MODEL_SAVE_PATH}_best")
            progress_bar.write(f"  New best model saved!")

        self.model.train()
        return best_val_loss


    def choose_action(self, batch, max_len_output=16, aggregation='mean'):
        # Extract inputs and targets
        inputs_text = [b['input'] for b in batch]
        targets_text = [b['target'] for b in batch]

        # Tokenize input
        inputs = self.tokenizer(
            inputs_text,
            padding=True,
            truncation=True,
            max_length=MAX_LEN_INPUT,
            return_tensors='pt'
        ).to(self.device)
        
        # Tokenize targets
        targets = self.tokenizer(
            targets_text,
            padding=True,
            truncation=True,
            max_length=MAX_LEN_OUTPUT,
            return_tensors='pt'
        ).to(self.device)

        # inputs = batch['input']
        yes_tokenized_targets = self.tokenizer(
            ['Yes'] * len(targets_text), max_length=max_len_output,
            truncation=True,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        no_tokenized_targets = self.tokenizer(
            ['No'] * len(targets_text), max_length=max_len_output,
            truncation=True,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            labels = yes_tokenized_targets['input_ids']
            labels[labels == 0] = -100
            label_mask = yes_tokenized_targets['attention_mask']
            outputs = self.model.forward(
                input_ids=inputs['input_ids'],
                attention_mask=inputs["attention_mask"],
                decoder_attention_mask=label_mask,
                labels=labels,
            )
            losses = functional.cross_entropy(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1), reduction='none').reshape(outputs.logits.size(0), -1)

            yes_loss = losses[:, 0]

            labels = no_tokenized_targets['input_ids']
            labels[labels == 0] = -100
            label_mask = no_tokenized_targets['attention_mask']
            outputs = self.model.forward(
                input_ids=inputs['input_ids'],
                attention_mask=inputs["attention_mask"],
                decoder_attention_mask=label_mask,
                labels=labels,
            )
            losses = functional.cross_entropy(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1), reduction='none').reshape(outputs.logits.size(0), -1)

            no_loss = losses[:, 0]

            pred = (yes_loss < no_loss).tolist()
            scores = (no_loss - yes_loss).tolist()
        return pred, scores

    def generate_action(self, batch, max_len_output=16, num_beams=4):
        with torch.no_grad():
            inputs = batch['input']
            beam_outputs = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=max_len_output,
                early_stopping=True,
                num_beams=num_beams,
                num_return_sequences=1,
            )
            action = self.tokenizer.batch_decode(
                beam_outputs,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
        return action

    def infer_feedback(self, trajectories):

        processed_trajectories = convert_trajectories(trajectories)
        test_dataset = FeedbackDataset(feedback_data=processed_trajectories, test_only=True)

        test_loader = test_dataset.get_test_loader(BATCH_SIZE)
        
        predictions = []

        progress_bar = tqdm(test_loader, desc=f"Predicting")
        for batch in progress_bar:
            preds = self.predict(batch)
            predictions.extend(preds)

        all_traj_steps = defaultdict(list)

        assert len(predictions) == len(test_dataset.test_examples), 'got {} predictions for {} input examples'.format(len(predictions), len(test_dataset.test_examples))

        idx = 0
        for ex, pred in zip(processed_trajectories, predictions):
            output = ex['orig'].copy()
            output['llm_pred'] = pred
            output['step'] = ex['step']

            all_traj_steps[ex['traj_idx']].append(output)

        for traj in all_traj_steps.values():
            traj.sort(key=lambda x: x['step'])

        if not os.path.isdir(OUTPUT_PATH):
            os.makedirs(OUTPUT_PATH)

        for fname, traj in all_traj_steps.items():
            with bz2.open('{}/trajectory_{}'.format(OUTPUT_PATH, fname), 'wt') as f:
                json.dump(traj, f, indent=2)

    def predict(self, batch):
        self.model.eval()
        # predicted_actions = self.generate_action(batch)
        predicted_actions, _ = self.choose_action(batch)
        # gold = self.tokenizer.batch_decode(
        #     batch['target']['input_ids'],
        #     skip_special_tokens=True,
        #     clean_up_tokenization_spaces=True,
        # )
        # context = self.tokenizer.batch_decode(
        #     batch['input']['input_ids'],
        #     skip_special_tokens=True,
        #     clean_up_tokenization_spaces=True,
        # )
        # preds = []
        # for ctx, g, p in zip(context, gold, predicted_actions):
        #     preds.append(dict(context=ctx, gold=g, pred=p))
        self.model.train()
        return predicted_actions
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or use a LFM')
    parser.add_argument('--train', action='store_true',
                        help='Train a LFM')
    parser.add_argument('--use', action='store_true',
                        help='Use a LFM to get feedback')
    
    args = parser.parse_args()
    if args.train:
        lfm = LanguageFeedbackModel()
        lfm.train()
    elif args.use:
        lfm = LanguageFeedbackModel(model_path=LFM_MODEL_PATH)
        print("Loading trajectories")
        trajectories = load_trajectories(TRAJECTORIES_PATH)
        print("Generating feedback")
        lfm.infer_feedback(trajectories)


