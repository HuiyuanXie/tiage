"""
Dataset and DataLoader for WIKI-727K.
"""

import csv
import numpy as np
import pandas as pd
import ipdb as pdb
import torch
from torch.utils.data import Dataset, DataLoader


class WikiDataset(Dataset):
    def __init__(self, sentences, next_sentences, targets, tokenizer, max_len=128):
        self.sentences = sentences
        self.next_sentences = next_sentences
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        next_sentence = self.next_sentences[idx]
        target = self.targets[idx]

        encoding = self.tokenizer.encode_plus(
            sentence,
            next_sentence,
            add_special_tokens=True,
            return_token_type_ids=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            truncation=True
        )
        
        input_ids = torch.as_tensor(encoding['input_ids'])
        # pdb.set_trace()
        attention_mask = torch.as_tensor(encoding['attention_mask'])
        token_type_ids = torch.as_tensor(encoding['token_type_ids'])
        target = torch.as_tensor(target, dtype=torch.long)

        return (sentence, next_sentence, input_ids, attention_mask, token_type_ids, target)

class WikiDataLoader(DataLoader):
    def __init__(self, **kwargs):
        if 'datafile' not in kwargs:
            raise ValueError("Must give datafile path")
        datafile_path = kwargs.pop('datafile')
        if 'tokenizer' not in kwargs:
            raise ValueError("Must give tokenizer")
        tokenizer = kwargs.pop('tokenizer')
        max_len = kwargs.pop('max_len', None)
        df = pd.read_csv(datafile_path, sep='\t', header=None)
        # pdb.set_trace()
        targets = df[1]
        sentences = df[2]
        next_sentences = df[3]
        self.dataset = WikiDataset(sentences, next_sentences, targets, tokenizer, max_len)
        super(WikiDataLoader, self).__init__(self.dataset, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

