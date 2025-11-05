# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import argparse
from pathlib import Path


unique_passages_ids = set()

def select_examples_NQ(index, passages_index):
    for i in range(len(index)):
        for idx in passages_index[str(i)]:
            unique_passages_ids.add(idx)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Select question subset from NQ data')
    parser.add_argument('--data_dir', type=str, help='Directory containing data files')
    parser.add_argument('--num_questions', type=int, default=1000,
                       help='Number of questions to select (default: 1000)')
    args = parser.parse_args()
    
    dir_path = Path(args.data_dir)
    num_questions = args.num_questions
    
    #load NQ question idx
    NQ_idx = {}
    NQ_passages = {}
    for split in ['test']:
        with open(dir_path/('NQ.' + split + '.idx.json'), 'r') as fin:
            NQ_idx[split] = json.load(fin)
        with open(dir_path/'nq_passages' /  (split + '.json'), 'r') as fin:
            NQ_passages[split] = json.load(fin)
    
    select_examples_NQ(NQ_idx['test'][:num_questions], NQ_passages['test'])
    print(unique_passages_ids)

   