"""
Prepare PersonaChat data for finetuning T5.
"""

import os
import ipdb as pdb
import random
import torch
from torch.utils.data import Dataset


class PCDataset(Dataset):
    def __init__(self, tokenizer, contexts, responses, max_len, target_max_len):
        self.tokenizer = tokenizer
        self.contexts = contexts
        self.responses = responses
        self.max_len = max_len
        self.target_max_len = target_max_len
    
    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx):
        context = self.contexts[idx]
        response = self.responses[idx]
        context_encoding = self.tokenizer.encode(context, padding='max_length', max_length=self.max_len, truncation=True)
        response_encoding = self.tokenizer.encode(response, padding='max_length', max_length=self.target_max_len, truncation=True)
        input_ids = torch.as_tensor(context_encoding)
        target_ids = torch.as_tensor(response_encoding)
        encoding = {
            'input_ids': input_ids,
            # 'attention_mask': attention_mask,
            'target_ids': target_ids,
            # 'target_attention_mask': target_attention_mask
        }
        
        return encoding


def prepare_PC_data(tokenizer, datafile, max_len, target_max_len):
    contexts, responses = [], []
    with open(datafile, 'r') as f:
        for line in f:
            sents = line.strip().split('\t')
            context = sents[0]
            response = sents[1]
            contexts.append(context)
            responses.append(response)
            
    return PCDataset(tokenizer, contexts, responses, max_len, target_max_len)

def read_PC_file(infile):
    lines = []
    with open(infile, 'r') as inf:
        for line in inf:
            lines.append(line.strip())
    return lines

