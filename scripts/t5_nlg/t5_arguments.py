"""
ModelArguments and DataTrainingArguments for finetuning T5.
"""

import dataclasses
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to finetune from.
    """
    model_name_or_path: str = field(
            metadata={"help": "Path to pretrained model or model identifier"}
    )
    tokenizer_name: Optional[str] = field(
            default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
            default=None, metadata={"help": "Path to the dir to store the pretrained models downloaded"}
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    train_file_path: Optional[str] = field(
            default='data/train_data.pt',
            metadata={"help": "Path for cached train dataset"}
    )
    valid_file_path: Optional[str] = field(
            default='data/valid_data.pt',
            metadata={"help": "Path for cached valid dataset"}
    )
    max_len: Optional[int] = field(
            default=512,
            metadata={"help": "Max input length for the source text"}
    )
    target_max_len: Optional[int] = field(
            default=32,
            metadata={"help": "Max input length for the target text"}
    )

