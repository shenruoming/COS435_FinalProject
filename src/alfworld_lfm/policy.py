"""
Policy module for BC agent inference.
Exact replication of LMAgent from original codebase.
"""

import torch
import torch.nn.functional as F
import numpy as np
import random
import copy
from typing import List, Dict, Tuple, Optional
from torch.distributions import Categorical
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import editdistance
from pathlib import Path

class Policy:
    
    def __init__(
        self,
        model_path,
        max_len_input=2048,
        max_len_output=16,
        device: Optional[str] = None,
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_len_input = max_len_input
        self.max_len_output = max_len_output
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
    
    def choose_action(
        self,
        inp: str,
        admissible_actions: List[str],
        aggregation: str = 'mean',
        sample: bool = False,
        eval_batch_size: int = 100,
    ) -> Tuple[str, List[float]]:
        """
        Score each action and select one.
        """
        if admissible_actions and isinstance(admissible_actions[0], dict):
            admissible_actions = [a['key'] for a in admissible_actions]
        
        all_losses = []
        all_labels = []
        
        for i in range(0, len(admissible_actions), eval_batch_size):
            aa = admissible_actions[i:i+eval_batch_size]
            
            inputs = self.tokenizer(
                        [inp] * len(aa),
                        max_length=self.max_len_input,
                        truncation=True,
                        padding='max_length',
                        return_tensors="pt"
                    )
            targets = self.tokenizer(
                        aa,
                        max_length=self.max_len_output,
                        truncation=True,
                        padding='max_length',
                        return_tensors="pt"
                    )
            
            
            source_ids = inputs["input_ids"]
            target_ids = targets["input_ids"]
            src_mask = inputs["attention_mask"]
            target_mask = targets["attention_mask"]
            
            labels = copy.deepcopy(target_ids)
            labels[labels == 0] = -100
            all_labels.append(labels)
            
            with torch.no_grad():
                outputs = self.model(
                    input_ids=source_ids,
                    attention_mask=src_mask,
                    decoder_attention_mask=target_mask,
                    labels=labels,
                )
                
                # Per-token loss computation
                losses = F.cross_entropy(
                    outputs.logits.view(-1, outputs.logits.size(-1)),
                    labels.view(-1),
                    reduction='none'
                ).reshape(outputs.logits.size(0), -1)
                
            all_losses.append(losses)
        
        # Concatenate and normalize
        losses = torch.cat(all_losses, dim=0)
        labels = torch.cat(all_labels, dim=0)
        
        norm_losses = losses.sum(dim=1)
        if aggregation == 'mean':
            norm_losses /= (labels != -100).sum(dim=1).float()
        
        scores = -norm_losses
        
        # Select action
        if sample:
            p = Categorical(F.softmax(scores, dim=0))
            max_arg = p.sample().item()
        else:
            max_score, max_arg = scores.max(0)
            max_arg = max_arg.item()
        
        action = admissible_actions[max_arg]
        return action, scores.tolist()
    