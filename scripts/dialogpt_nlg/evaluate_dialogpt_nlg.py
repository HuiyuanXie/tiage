"""
Evaluate finetuned DialoGPT on PersonaChat.
"""

import os
import ipdb as pdb
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import re
import argparse
import torch
from scripts.datasets.PersonaChat.pc_data_t5 import prepare_PC_data
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', default='')
parser.add_argument('--test_dir', default='data')
parser.add_argument('--all_test_file', default='test.tsv')
parser.add_argument('--hyp_dir', default='data/model_predictions')
parser.add_argument('--all_hyp_file', default='hyp_all.tsv')

def generate(model_path, testfile, hypfile):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Load finetuned model from %s" % model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model = model.to(device)
    print("Model loaded.")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # add special tokens
    print("Initial #vocab:", len(tokenizer))
    sep_tok = '<sep>'
    eos_tok = '<eos>'
    n_toks = tokenizer.add_tokens([sep_tok, eos_tok], special_tokens=True)
    sep_tok_id = tokenizer.encode('<sep>')
    eos_tok_id = tokenizer.encode('<eos>')
    print("Added %d new tokens. %s: %s; %s: %s." % (n_toks, sep_tok, str(sep_tok_id), eos_tok, str(eos_tok_id)))
    tokenizer.pad_token = tokenizer.eos_token
    print("New #vocab:", len(tokenizer))
    # pdb.set_trace()
    model.resize_token_embeddings(len(tokenizer))
    
    print("Load test data ...")
    test_dataset = prepare_PC_data(tokenizer, testfile, max_len=512, target_max_len=32)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    print("Test data loaded.")
    answers = []
    if os.path.exists(hypfile):
        print("Removing old prediction file ...")
        os.remove(hypfile)
    print("Predict answers ...")
    for batch in tqdm(test_loader):
        chat_history_ids = model.generate(
            input_ids=batch['input_ids'].to(device),
            max_length=600,
            pad_token_id=tokenizer.eos_token_id,
            num_beams=5,
            no_repeat_ngram_size=3,
            early_stopping=True,
            )
        # outs = [tokenizer.decode(ids) for ids in outs]
        outs = tokenizer.decode(chat_history_ids[:, batch['input_ids'].shape[-1]:][0], skip_special_tokens=True)
        outs = outs.split('<eos>')[0]
        answers.append(outs)

    print("Cache model predictions ...")
    with open(hypfile, 'w') as f:
        for ans in answers:
            f.write(ans)
            f.write('\n')


if __name__ == '__main__':
    args = parser.parse_args()
    assert os.path.exists(args.test_dir), "Cannot find test_dir %s" % args.test_dir
    if not os.path.exists(args.hyp_dir):
        os.makedirs(args.hyp_dir)
    all_test_file = os.path.join(args.test_dir, args.all_test_file)
    all_hyp_file = os.path.join(args.hyp_dir, args.all_hyp_file)
    generate(args.model_name_or_path, all_test_file, all_hyp_file)
