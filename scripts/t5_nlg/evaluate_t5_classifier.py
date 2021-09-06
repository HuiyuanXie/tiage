"""
Evaluate finetuned T5 on PersonaChat.
"""

import os
import ipdb as pdb
from transformers import T5ForConditionalGeneration, T5Tokenizer
from tqdm import tqdm
import re
import argparse
import torch
from scripts.datasets.PersonaChat.pc_data_t5 import prepare_PC_data
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', default='data/saved_models', help='model path to T5 TS classifier')
parser.add_argument('--all_test_file', default='data/test_t5_classifier.tsv')
parser.add_argument('--all_hyp_file', default='data/t5_classifier_predictions.tsv')


def generate_t5_predictions(model_path, testfile, hypfile):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Load finetuned model from %s" % model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    model = model.to(device)
    # add special tokens
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    eos_tok = tokenizer.encode('<sep>')
    sep_tok = tokenizer.encode('<eos>')   
    print("Model loaded.")
    print("Load test data ...")
    test_dataset = prepare_PC_data(tokenizer, testfile, max_len=512, target_max_len=2)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
    print("Test data loaded.")
    answers = []
    if os.path.exists(hypfile):
        print("Removing old prediction file ...")
        os.remove(hypfile)
    print("Predicting answers ...")
    for batch in tqdm(test_loader):
        outs = model.generate(
            input_ids=batch['input_ids'].to(device),
            attention_mask=batch['attention_mask'].to(device),
            max_length=4,
            # max_length=32,
            early_stopping=True,
            )
        outs = [tokenizer.decode(ids) for ids in outs]
        answers.extend(outs)

    print("Cache model predictions ...")
    with open(hypfile, 'w') as f:
        for ans in answers:
            print(ans)
            f.write(ans)
            f.write('\n')


if __name__ == '__main__':
    args = parser.parse_args()
    generate_t5_predictions(args.model_name_or_path, args.all_test_file, args.all_hyp_file)
