# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
import json
import os
import argparse
from pathlib import Path
import numpy as np
import util

def select_examples_NQ(data, index, passages, passages_index):
    selected_data = []
    for i, k in enumerate(index):
        ctxs = [
                {
                    'id': idx,
                    'title': passages[idx][1],
                    'text': passages[idx][0],
                }
                for idx in passages_index[str(i)]
            ]
        dico = {
            'question': data[k]['question'],
            'answers': data[k]['answer'],
            'ctxs': ctxs,
        }
        selected_data.append(dico)

    return selected_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess NQ data')
    parser.add_argument('--num_questions', type=int, default=None,
                       help='Number of questions to process (default: all)')
    parser.add_argument('--passages_file', type=str, default='open_domain_data/psgs_w100_subset.tsv',
                       help='Passages file name (default: psgs_w100_subset.tsv)')
    args = parser.parse_args()
    
    num_questions = args.num_questions
    passages_file =  args.passages_file
    
    print(f"Loading passages from {passages_file}")
    passages = util.load_passages(passages_file)
    passages = {p[0]: (p[1], p[2]) for p in passages}
    print(f"Loaded {len(passages)} passages")
    
    #load NQ question idx
    NQ_idx = {}
    NQ_passages = {}
    for split in ['test']:
        with open('open_domain_data/download/NQ.' + split + '.idx.json', 'r') as fin:
            NQ_idx[split] = json.load(fin)
        with open('open_domain_data/download/nq_passages/' + (split + '.json'), 'r') as fin:
            NQ_passages[split] = json.load(fin)

    originaldev = []
    with open('open_domain_data/download/NQ-open.dev.jsonl') as fin:
        for k, example in enumerate(fin):
            example = json.loads(example)
            originaldev.append(example)
    
    print(f"Selecting test examples")
    NQ_test = select_examples_NQ(originaldev, NQ_idx['test'][:num_questions], passages, NQ_passages['test'])
    
    NQ_save_path = 'open_domain_data/NQ'
    os.makedirs(NQ_save_path, exist_ok=True)

    with open(NQ_save_path+'/test.json', 'w') as fout:
        json.dump(NQ_test, fout, indent=4)