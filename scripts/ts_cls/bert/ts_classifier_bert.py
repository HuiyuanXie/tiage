"""
Topic shift classification model with BERT.
"""

import os
import sys
from transformers import BertModel
import torch
import torch.nn as nn

class TSClassifier(nn.Module):
    def __init__(self, bert, dropout=0.5, n_classes=2):
        super(TSClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes) 

    def forward(self, input_ids, attention_mask, token_type_ids):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        output = self.drop(pooled_output)
        return self.out(output)

