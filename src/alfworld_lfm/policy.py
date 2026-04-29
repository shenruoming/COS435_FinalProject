import torch
import torch.nn.functional as F
import numpy as np
import random
import copy
from typing import List, Dict, Tuple, Optional
from torch.distributions import Categorical
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pathlib import Path

# bc policy (parallels LMAgent from original codebase)
class Policy:
    
    def __init__(
        self,
        model_path,
        max_len_input=512,
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

        with torch.no_grad():
            # Encode input once instead of repeating per batch
            encoder_inputs = self.tokenizer(
                inp,
                max_length=self.max_len_input,
                truncation=True,
                return_tensors="pt"
            )
            encoder_inputs = {k: v.to(self.device) for k, v in encoder_inputs.items()}

            encoder_outputs = self.model.encoder(
                input_ids=encoder_inputs["input_ids"],
                attention_mask=encoder_inputs["attention_mask"],
            )

            for i in range(0, len(admissible_actions), eval_batch_size):
                aa = admissible_actions[i:i+eval_batch_size]
                batch_size = len(aa)

                # Expand encoder outputs to batch size
                expanded_hidden = encoder_outputs.last_hidden_state.expand(batch_size, -1, -1)
                expanded_mask = encoder_inputs["attention_mask"].expand(batch_size, -1)

                targets = self.tokenizer(
                    aa,
                    max_length=self.max_len_output,
                    truncation=True,
                    padding=True,  # pad to longest in batch, not max_length
                    return_tensors="pt"
                )
                targets = {k: v.to(self.device) for k, v in targets.items()}

                labels = targets["input_ids"].clone()
                labels[labels == self.tokenizer.pad_token_id] = -100
                all_labels.append(labels)

                outputs = self.model(
                    attention_mask=expanded_mask,
                    decoder_attention_mask=targets["attention_mask"],
                    encoder_outputs=(expanded_hidden,),
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
            max_arg = scores.argmax(0).item()

        print(f"max index {max_arg}")
        action = admissible_actions[max_arg]
        return action, scores.tolist()