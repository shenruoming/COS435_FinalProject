import os
import re
import bz2
import random
import numpy as np
import json
from pathlib import Path
from tqdm import auto as tqdm
from collections import Counter
from collections import defaultdict
import pickle
from torch.utils.data import Dataset, DataLoader


class FeedbackDataset(Dataset):

    def __init__(self, feedback_data=None, p_pos=0.5, max_len_input=512, max_len_output=32, load_path=None, test_only=False):
        self.max_len_input = max_len_input
        self.max_len_output = max_len_output

        if test_only:
            self.test_examples = []
            for r in feedback_data:
                prompt = 'Task: {}\nBefore: {}\nAction: {}\nAfter: {}\nQuestion: was this helpful?\nAnswer:'.format(r['task'], r['before'], r['action'], r['after'])
                feedback = 'Yes' if r['label'] else 'No'
                self.test_examples.append({
                    'input': prompt,
                    'target': feedback
                })
            return



        # Try to load from disk first
        if load_path and os.path.exists(load_path):
            print(f"Loading existing dataset from {load_path}")
            self.load(load_path)
            return

        task_re = re.compile(r'Task: ([^\n]+)\n')
        before_re = re.compile(r'Before: ([^\n]+)\n')
        step_re = re.compile(r'Step (\d+)\nYour action: ([^\n]+?)\nResult: (.+?)\-\-\-', re.DOTALL)

        good_feedback_step_re = re.compile(r'Step (\d+)')

        proc = []

        example_idx = 0
        for ex in tqdm.tqdm(feedback_data, desc=f"Example {example_idx+1}"):
            example_idx += 1

            task = task_re.findall(ex['prompt'])[0].rstrip('. ')
            before = before_re.findall(ex['prompt'])[0].rstrip('. ')
            feedback = defaultdict(list)

            for step_str in good_feedback_step_re.findall(ex['response']):
                feedback[step_str].append("")

            steps = []
            for step_id, action, after in step_re.findall(ex['prompt']):
                steps.append(dict(
                    task=task,
                    step=step_id,
                    action=action,
                    before=before,
                    after=after,
                    label=step_id in feedback,
                    feedback=feedback[step_id],
                ))
                before = after

            proc.extend(steps)
        # if p_pos:
        #     pos = [x for x in proc if x['label']]
        #     neg = [x for x in proc if not x['label']]
        #     random.Random(0).shuffle(neg)
        #     cat = []
        #     # TODO: fix this
        #     cat.extend(pos)
        #     cat.extend(neg)
        #     # cat.extend(neg[:int(len(pos) / p_pos) - len(pos)])
        #     # cat = pos + neg[:int(len(pos) / p_pos) - len(pos)]
        #     random.Random(0).shuffle(cat)



        self.data = []
        for r in proc:
            prompt = 'Task: {}\nBefore: {}\nAction: {}\nAfter: {}\nQuestion: was this helpful?\nAnswer:'.format(r['task'], r['before'], r['action'], r['after'])
            feedback = 'Yes' if r['label'] else 'No'
            self.data.append({
                'input': prompt,
                'target': feedback
            })

        np.random.seed(42)
        indices = np.random.permutation(len(self.data))
        print(len(self.data))
        split_idx = int(0.8 * len(self.data))
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        self.train_examples = [self.data[i] for i in train_indices]
        self.val_examples = [self.data[i] for i in val_indices]


    # def __getitem__(self, key):
    #     return self.data[key]
    
    def save(self, path):
        """Save collected examples to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'train_examples': self.train_examples,
                'val_examples': self.val_examples,
            }, f)
        print(f"Saved dataset to {path}")
    
    def load(self, path):
        """Load saved examples from disk"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.train_examples = data['train_examples']
        self.val_examples = data['val_examples']

    def __len__(self):
        return len(self.data)
    
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
    
    def get_test_loader(self, batch_size):
        """Return DataLoader for test examples"""
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
            SimpleDataset(self.test_examples), 
            batch_size=batch_size,
            collate_fn=collate_fn
        )

def make_dataset(input_path):
    data = []
    for fname in input_path.glob('*.json.bz2'):
        with bz2.open(fname, 'rt') as f:
            data.append(json.load(f))
    return FeedbackDataset(feedback_data=data)

if __name__ == "__main__":
    LLM_INPUT_PATH = Path('./llm_feedback_new')
    DATASET_OUTPUT_PATH = './lfm/feedback_data_new'
    dataset = make_dataset(LLM_INPUT_PATH)
    dataset.save(DATASET_OUTPUT_PATH)