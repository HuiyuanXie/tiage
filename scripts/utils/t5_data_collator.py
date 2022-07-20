"""
DataCollator for data preparation for T5.
"""

import dataclasses
from dataclasses import dataclass, field
from transformers import DataCollator
import torch
from typing import Dict, List
import ipdb as pdb

@dataclass
class T5DataCollator:
    # def collate_batch(self, batch: List) -> Dict[str, torch.Tensor]:
    def __call__(self, batch):
        """
        Take a list of samples from a Dataset and collate them into a batch.
        Returns:
            A dictionary of tensors
        """
        # pdb.set_trace()
        input_ids = torch.stack([example['input_ids'] for example in batch])
        lm_labels = torch.stack([example['target_ids'] for example in batch])
        lm_labels[lm_labels[:, :] == 0] = -100 # for pad tokens
        attention_mask = torch.stack([example['attention_mask'] for example in batch])
        decoder_attention_mask = torch.stack([example['target_attention_mask'] for example in batch])
        return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                # 'lm_labels': lm_labels,
                'labels': lm_labels,
                'decoder_attention_mask': decoder_attention_mask,
        }
