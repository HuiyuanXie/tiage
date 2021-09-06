"""
Finetuning the BERT model for topic shift classification on WIKI_727K.
"""

import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
import ipdb as pdb
import time
from tqdm import tqdm
from collections import defaultdict
import transformers
from transformers import BertModel, BertTokenizer, AdamW
from transformers.optimization import get_linear_schedule_with_warmup
import torch
from torch import nn
import torch.nn.functional as F
from scripts.datasets.WIKI_727K.wiki_data_bert import WikiDataLoader
from scripts.ts_classification.bert.ts_classifier_bert import TSClassifier

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()

# data loading options
parser.add_argument('--train_file', default='data')
parser.add_argument('--val_file', default='data')
parser.add_argument('--loader_num_workers', default=0, type=int)
parser.add_argument('--shuffle_train_data', default=1, type=int)
# training options
parser.add_argument('--time', default=1, type=int) # record epoch pass time during training
parser.add_argument('--num_epochs', default=3, type=int)
parser.add_argument('--checkpoint_path', default='data/saved_models')
# model options
parser.add_argument('--pretrained_model_name', default='bert-base-uncased', choices=['bert-base-uncased', 'bert-base-cased'])
parser.add_argument('--max_len', default=128, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--learning_rate', default=2e-5, type=float)
parser.add_argument('--dropout', default=0.5, type=float)

def main(args):
    # set up pretrained model
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_name)
    print("Loading the model ...")
    model =  TSClassifier(args.pretrained_model_name, dropout=args.dropout)
    model = model.to(device)
    # load data
    train_loader_kwargs = {
        'datafile': args.train_file,
        'tokenizer': tokenizer,
        'max_len': args.max_len,
        'batch_size': args.batch_size,
        'num_workers': args.loader_num_workers,
    }
    val_loader_kwargs = {
        'datafile': args.val_file,
        'tokenizer': tokenizer,
        'max_len': args.max_len,
        'batch_size': args.batch_size,
        'num_workers': args.loader_num_workers,
    }

    print("Loading training data ...")
    train_loader = WikiDataLoader(**train_loader_kwargs)
    num_train_samples = len(train_loader.dataset)
    print("Train_loader has %d samples" % num_train_samples)
    print("Loading validation data ...")
    val_loader = WikiDataLoader(**val_loader_kwargs)
    num_val_samples = len(val_loader.dataset)
    print("Val_loader has %d samples" % num_val_samples)

    # train the model
    print("Start training the model ...")
    epoch = 0
    epoch_start_time, epoch_total_time = 0.0, 0.0
    loss_fn = nn.CrossEntropyLoss().to(device)
    total_training_steps = len(train_loader) * args.num_epochs
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_training_steps
    )
    stats = defaultdict(list)
    best_val_acc = 0.0
    
    while epoch < args.num_epochs:
        if epoch > 0 and args.time == 1: # record epoch pass time
            epoch_time = time.time() - epoch_start_time
            epoch_total_time += epoch_time
            print("Current epoch pass time: %.0f" % float(epoch_time))
            print("Epoch pass average time: %.0f" % float(epoch_total_time / epoch))
        epoch_start_time = time.time()
        epoch += 1
        print()
        print("Epoch %d/%d" % (epoch, args.num_epochs))
        print('-' * 10)
        train_acc, train_loss = train_epoch(model, train_loader, loss_fn, optimizer, scheduler, num_train_samples)
        print("Train acc: %.3f, loss: %.6f" % (train_acc, train_loss))
        val_acc, val_loss = eval_epoch(model, val_loader, loss_fn, num_val_samples)
        print("Val acc: %.3f, loss: %.6f" % (val_acc, val_loss))
        stats['train_acc'].append(train_acc)
        stats['train_loss'].append(train_loss)
        stats['val_acc'].append(val_acc)
        stats['val_loss'].append(val_loss)
        if val_acc > best_val_acc:
            print("Saving model ...")
            torch.save(model.state_dict(), args.checkpoint_path)
            best_val_acc = val_acc
            
def train_epoch(model, data_loader, loss_fn, optimizer, scheduler, num_samples):
    model.train() # set to training mode
    epoch_losses = []
    epoch_correct_preds = 0
    for batch in tqdm(data_loader):
        sentences, next_sentences, input_ids, attention_masks, token_type_ids, targets = batch
        # pdb.set_trace()
        input_ids = input_ids.to(device)
        attention_masks = attention_masks.to(device)
        token_type_ids = token_type_ids.to(device)
        targets = targets.to(device)

        outputs = model(input_ids, attention_masks, token_type_ids)
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)
        epoch_losses.append(loss.item())
        epoch_correct_preds += torch.sum(preds == targets)
        optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm_(model.paramerters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
    train_accuracy = float(epoch_correct_preds) / num_samples
    avg_train_loss = np.mean(epoch_losses)
    return train_accuracy, avg_train_loss

def eval_epoch(model, data_loader, loss_fn, num_samples):
    model.eval()
    eval_losses = []
    correct_preds = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader):
            _, _, input_ids, attention_masks, token_type_ids, targets = batch
            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)
            token_type_ids = token_type_ids.to(device)
            targets = targets.to(device)
            outputs = model(
                input_ids = input_ids,
                attention_mask = attention_masks,
                token_type_ids = token_type_ids
            )
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)
            eval_losses.append(loss.item())
            correct_preds += torch.sum(preds == targets)
    eval_accuracy = float(correct_preds) / num_samples
    avg_eval_loss = np.mean(eval_losses)
    return eval_accuracy, avg_eval_loss


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

