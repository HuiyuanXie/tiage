"""
Finetune DialoGPT on custom data.
"""

import os
import sys
import torch
import json
import argparse
import logging
import shutil
import ipdb as pdb
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import (
        HfArgumentParser,
        # Trainer,
        TrainingArguments,
        set_seed,
)
from typing import Dict
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from scripts.baselines.T5.t5_arguments import ModelArguments, DataTrainingArguments
from scripts.datasets.PersonaChat.pc_data_t5 import PCDataset, prepare_PC_data
from scripts.utils.trainer_utils import CustomTrainer

parser = argparse.ArgumentParser()

parser.add_argument('--timestamp', default='')
parser.add_argument('--exp_name', default='')
parser.add_argument('--model_name_or_path', default='', help='Path to pretrained model')
parser.add_argument('--train_file_path', default='data/train.tsv', help='Path to training file')
parser.add_argument('--valid_file_path', default='data/dev.tsv', help='Path to validation file')
parser.add_argument('--num_train_epochs', type=int, default=10, help='Total number of training epochs')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for model training (default: 5e-5)')
parser.add_argument('--logging_steps', type=int, default=1699, help='Number of update steps between two logs')
parser.add_argument('--eval_steps', type=int, default=1699, help='Number of update steps between two evaluations')
parser.add_argument('--save_steps', type=int, default=3398, help='Number of update steps between two checkpoints saved')
# fixed arguments
parser.add_argument('--output_dir', default='data/saved_models', help='Path to the output root dir')
parser.add_argument('--logging_dir', default='data/logs', help='Path to the tsb root dir')
parser.add_argument('--max_len', type=int, default=512, help='Max length of contexts')
parser.add_argument('--target_max_len', type=int, default=32, help='Max length of responses')
parser.add_argument('--do_train', default=True, help='If finetune the model or not')
parser.add_argument('--do_eval', default=True, help='If evaluate the model while training')
parser.add_argument('--evaluation_strategy', default='steps', choices=['no', 'steps', 'epoch'], help='Evaluation strategy')
parser.add_argument('--per_device_train_batch_size', type=int, default=16)
parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
parser.add_argument('--per_device_eval_batch_size', type=int, default=16)
parser.add_argument('--eval_accumulation_steps', type=int, default=10)
parser.add_argument('--overwrite_output_dir', default=True)
parser.add_argument('--prediction_loss_only', default=True)
parser.add_argument('--dataloader_num_workers', type=int, default=4)
parser.add_argument('--warmup_steps', type=int, default=0)
parser.add_argument('--weight_decay', type=int, default=0)


def train_dialogpt(args):
    logger = logging.getLogger(__name__)
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    exp_id = args.timestamp + '_' + args.exp_name
    output_dir = os.path.join(args.output_dir, exp_id)
    logging_dir = os.path.join(args.logging_dir, args.exp_name, exp_id)
    # write to args file
    args_dict = {
        'model_name_or_path': args.model_name_or_path,
        'train_file_path': args.train_file_path,
        'valid_file_path': args.valid_file_path,
        'num_train_epochs': args.num_train_epochs,
        'learning_rate': args.learning_rate,
        'logging_steps': args.logging_steps,
        'eval_steps': args.eval_steps,
        'save_steps': args.save_steps,
        'output_dir': output_dir,
        'logging_dir': logging_dir,
        'max_len': args.max_len,
        'target_max_len': args.target_max_len,
        'do_train': args.do_train,
        'do_eval': args.do_eval,
        'evaluation_strategy': args.evaluation_strategy,
        'per_device_train_batch_size': args.per_device_train_batch_size,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'per_device_eval_batch_size': args.per_device_eval_batch_size,
        'eval_accumulation_steps': args.eval_accumulation_steps,
        'overwrite_output_dir': args.overwrite_output_dir,
        'prediction_loss_only': args.prediction_loss_only,
        'dataloader_num_workers': args.dataloader_num_workers,
        'warmup_steps': args.warmup_steps,
        'weight_decay': args.weight_decay,
    }
    if not os.path.exists(args_dict['output_dir']):
        os.makedirs(args_dict['output_dir'])
    args_file = os.path.join(output_dir, 'args.json')
    with open(args_file, 'w') as f:
        json.dump(args_dict, f, indent=4)

    model_args, data_args, training_args = parser.parse_json_file(json_file=args_file)
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    if os.path.exists(args_dict['logging_dir']):
        logger.info("Removing old logging directory... :)")
        try:
            shutil.rmtree(args_dict['logging_dir'])
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    
    # Set seed
    set_seed(training_args.seed)

    logger.info("Download the T5 model and the tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    logger.info("Tokenizer loaded.")
    print("Initial vocab size:", len(tokenizer))
    # add special tokens
    sep_tok = '<sep>'
    eos_tok = '<eos>'
    n_toks = tokenizer.add_tokens([sep_tok, eos_tok], special_tokens=True)
    sep_tok_id = tokenizer.encode('<sep>')
    eos_tok_id = tokenizer.encode('<eos>')
    print("Added %d new tokens. %s: %s; %s: %s." % (n_toks, sep_tok, str(sep_tok_id), eos_tok, str(eos_tok_id)))
    tokenizer.pad_token = tokenizer.eos_token
    print("New vocab size:", len(tokenizer))
    # pdb.set_trace()
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)
    logger.info("Model loaded.")
    model.resize_token_embeddings(len(tokenizer))
    
    # prepare datasets from orginal data file
    logger.info("Prepare PersonaChat training data ...")
    train_dataset = prepare_PC_data(tokenizer, data_args.train_file_path, data_args.max_len, data_args.target_max_len)
    valid_dataset = prepare_PC_data(tokenizer, data_args.valid_file_path, data_args.max_len, data_args.target_max_len)
    logger.info("Data loaded.")

    # initialize Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        # data_collator=DialoGPTDataCollator(),
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        # compute_metrics=compute_metrics,
    )

    # training
    if training_args.do_train:
        trainer.train()
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval and training_args.local_rank in [-1, 0]:
        logger.info("*** Evaluate ***")

        eval_output = trainer.evaluate()

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(eval_output.keys()):
                logger.info("  %s = %s", key, str(eval_output[key]))
                writer.write("%s = %s\n" % (key, str(eval_output[key])))

        results.update(eval_output)

    return results

if __name__ == '__main__':
    args = parser.parse_args()
    train_dialogpt(args)
