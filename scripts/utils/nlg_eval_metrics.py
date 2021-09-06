"""
Evaluate NLG model predictions using nlg-eval.
"""

import os
import ipdb as pdb
from nlgeval import compute_metrics
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('--timestamp', default='')
parser.add_argument('--exp_name', default='')
parser.add_argument('--hyp_root_dir', default='data/model_predictions')
parser.add_argument('--all_hyp_file', default='hyp_all.tsv')
parser.add_argument('--ref_root_dir', default='data')
parser.add_argument('--all_ref_file', default='ts_ref.tsv')
parser.add_argument('--all_tsv_file', default='data/results/all_results.tsv')
parser.add_argument('--all_latex_file', default='data/results/all_latex_results.tsv')

def remove_hts_labels(infile, outfile):
    with open(infile, 'r') as inf, open(outfile, 'w') as outf:
        for line in inf:
            new_line = ' '.join(line.split()[1:]) + '\n'
            outf.write(new_line)

def evaluate_metrics(hyp_file, ref_file):
    scores = compute_metrics(hypothesis=hyp_file, references=[ref_file])
    bleu_1 = scores['Bleu_1']
    bleu_2 = scores['Bleu_2']
    bleu_3 = scores['Bleu_3']
    bleu_4 = scores['Bleu_4']
    meteor = scores['METEOR']
    rouge_l = scores['ROUGE_L']
    cider = scores['CIDEr']
    return {'BLEU_1': bleu_1, 'BLEU_2': bleu_2, 'BLEU_3': bleu_3, 'BLEU_4': bleu_4, 'METEOR': meteor, 'ROUGE_L': rouge_l, 'CIDEr': cider}

def write_results_to_files(args, results, results_type):
    exp_id = args.timestamp + '_' + args.exp_name
    assert results_type in ['all', 'ts']
    if results_type == 'all':
        tsv_file = args.all_tsv_file
        latex_file = args.all_latex_file
    else:
        tsv_file = args.ts_tsv_file
        latex_file = args.ts_latex_file
    
    with open(tsv_file, 'a') as tsvf, open(latex_file, 'a') as latexf:
        bleu_1 = results['BLEU_1']
        bleu_2 = results['BLEU_2']
        bleu_3 = results['BLEU_3']
        bleu_4 = results['BLEU_4']
        meteor = results['METEOR']
        rouge_l = results['ROUGE_L']
        cider = results['CIDEr']
        tsvf.write('%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n' % (exp_id, bleu_1, bleu_2, bleu_3, bleu_4, meteor, rouge_l, cider))
        latexf.write('%s & %.3f & %.3f & %.3f %.3f & %.3f & %.3f & %.3f\n' % (exp_id, bleu_1, bleu_2, bleu_3, bleu_4, meteor, rouge_l, cider))

if __name__ == '__main__':
    args = parser.parse_args()
    exp_id = args.timestamp + '_' + args.exp_name
    hyp_dir = args.hyp_root_dir
    ref_dir = args.ref_root_dir
    ori_all_hyp_file = os.path.join(hyp_dir, args.all_hyp_file)
    ori_all_ref_file = os.path.join(ref_dir, args.all_ref_file)
    if 'multitask' in args.exp_name:
        all_hyp_file = os.path.join(hyp_dir, 'hyp_all_without_label.tsv')
        all_ref_file = os.path.join(ref_dir, 'ref_all_multitask.tsv')
        remove_hts_labels(ori_all_hyp_file, all_hyp_file)
        remove_hts_labels(ori_all_ref_file, all_ref_file)
    else:
        all_hyp_file = ori_all_hyp_file
        all_ref_file = ori_all_ref_file
    print(exp_id)
    print("Evaluate all test instances ...")
    all_results = evaluate_metrics(all_hyp_file, all_ref_file)
    write_results_to_files(args, all_results, results_type='all')
    print("All_results written into files.")