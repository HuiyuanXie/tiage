"""
Finetune T5 on custom data.
"""

import os
import sys
import torch
import json
import logging
import ipdb as pdb
import shutil
from transformers import T5ForConditionalGeneration, T5Tokenizer, EvalPrediction
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
from scripts.utils.t5_data_collator import T5DataCollator
from scripts.datasets.PersonaChat.pc_data_t5 import PCDataset, prepare_PC_data
from scripts.utils.trainer_utils import CustomTrainer


def compute_metrics(p: EvalPrediction) -> Dict:
    ts_id = 1465  # 'positive'
    nts_id = 2841  # 'negative'
    # ts_id = 32102
    # nts_id = 32103
    legit_ids = [ts_id, nts_id]
    preds = p.predictions
    labels = p.label_ids
    preds = np.argmax(preds[0], axis=-1) 
    assert len(preds) == len(labels), "The length of model predictions does not match that of labels!" 
    pred_ids = [pred[0] for pred in preds]
    label_ids = [label[0] for label in labels] 
    total_len = len(label_ids)
    # sanity check
    for label_id in label_ids:
        assert label_id in legit_ids, label_id
    transferred = True
    for pred_id in pred_ids:
        if not pred_id in legit_ids:
            acc, precision, recall, f1 = 0, 0, 0, 0
            transferred = False
            break
    if transferred:  
        acc_count = 0
        for i in range(total_len):
            if pred_ids[i] == label_ids[i]:
                acc_count += 1
        acc = acc_count / total_len
        precision, recall, f1, _ = precision_recall_fscore_support(label_ids, pred_ids, average='binary', pos_label=ts_id)
    
    return {'eval_acc': acc, 'eval_precision': precision, 'eval_recall': recall, 'eval_f1': f1}


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    # write to args file
    args_dict = {
        'model_name_or_path': 't5-base', # from scratch
        'output_dir': 'data/saved_models',
        'logging_dir': 'data/logs',
        'overwrite_output_dir': True,
        'train_file_path': 'data/ts_train.tsv',
        'valid_file_path': 'data/ts_dev.tsv',
        'max_len': 512,
        'target_max_len': 2,
        'per_device_train_batch_size': 32,
        'gradient_accumulation_steps': 2,
        'per_device_eval_batch_size': 8,
        'eval_accumulation_steps': 10,
        'num_train_epochs': 10,
        'learning_rate': 1e-4,
        'weight_decay': 0,
        'warmup_steps': 0,
        'do_train': True,
        'do_eval': True,
        'evaluation_strategy': 'steps',
        # 'logging_first_step': True,
        'logging_steps': 91,
        'eval_steps': 91,
        'save_steps': 91000,
        # 'prediction_loss_only': True,
        'dataloader_num_workers': 4,
    }
    if not os.path.isdir(args_dict['output_dir']):
        os.makedirs(args_dict['output_dir'])
    if os.path.isdir(args_dict['logging_dir']):
        shutil.rmtree(args_dict['logging_dir'])
    args_file = os.path.join(args_dict['output_dir'], 'train_args.json')
    with open(args_file, 'w') as f:
        json.dump(args_dict, f)
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
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARNING,
    )
    logger = logging.getLogger(__name__)
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
    tokenizer = T5Tokenizer.from_pretrained(model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path)
    logger.info("Tokenizer loaded.")
    # add special tokens
    sep_tok = '<sep>'
    eos_tok = '<eos>'
    num_added_toks = tokenizer.add_tokens([sep_tok, eos_tok], special_tokens=True)
    sep_tok_id = tokenizer.encode(sep_tok)
    eos_tok_id = tokenizer.encode(eos_tok)
    logger.info("Added %d special tokens to vocab; tokens: %s %s, IDs: %s %s" % (num_added_toks, sep_tok, eos_tok, str(sep_tok_id), str(eos_tok_id)))
    pos_id = tokenizer.encode('positive')
    neg_id = tokenizer.encode('negative')
    logger.info("Pos_id: %s, neg_id: %s", str(pos_id), str(neg_id))
    model = T5ForConditionalGeneration.from_pretrained(model_args.model_name_or_path)
    logger.info("Model loaded.")
    # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e., the length of the tokenizer.
    model.resize_token_embeddings(len(tokenizer))
        
    # prepare datasets from orginal data file
    logger.info("Prepare the training data ...")
    train_dataset = prepare_PC_data(tokenizer, data_args.train_file_path, data_args.max_len, data_args.target_max_len)
    valid_dataset = prepare_PC_data(tokenizer, data_args.valid_file_path, data_args.max_len, data_args.target_max_len)
    logger.info("Data loaded.")

    # initialize Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=T5DataCollator(),
        compute_metrics=compute_metrics,
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
    main()
