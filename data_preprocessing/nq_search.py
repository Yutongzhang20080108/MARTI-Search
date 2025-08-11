# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the nq dataset to parquet format
"""

import re
import os
import json
import shutil
import subprocess
from typing import Optional

import datasets

import argparse


def make_prefix(dp, template_type):
    question = dp['question']

    # NOTE: also need to change reward_score/countdown.py
    if template_type == 'base':
        """This works for any base model"""
        prefix = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""
    else:
        raise NotImplementedError
    return prefix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='/data/local/search_r1')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--template_type', type=str, default='base')

    args = parser.parse_args()

    data_source = 'nq'

    dataset = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', 'nq')

    train_dataset = dataset['train']
    test_dataset = dataset['test']

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            example['question'] = example['question'].strip()
            if example['question'][-1] != '?':
                example['question'] += '?'
            question = make_prefix(example, template_type=args.template_type)
            solution = {
                "target": example['golden_answers'],
            }

            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "fact-reasoning",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    # Ensure local output directory exists
    os.makedirs(local_dir, exist_ok=True)

    # Write parquet as before
    train_parquet = os.path.join(local_dir, 'train.parquet')
    test_parquet = os.path.join(local_dir, 'test.parquet')
    train_dataset.to_parquet(train_parquet)
    test_dataset.to_parquet(test_parquet)

    # Also write JSONL containing the resulting prompts and metadata
    def write_jsonl(ds: datasets.Dataset, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            for row in ds:
                # Ensure only JSON-serializable content
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    train_jsonl = os.path.join(local_dir, 'train.jsonl')
    test_jsonl = os.path.join(local_dir, 'test.jsonl')
    write_jsonl(train_dataset, train_jsonl)
    write_jsonl(test_dataset, test_jsonl)

    # Optional: copy results to hdfs_dir or another local dir without verl
    def _has_cmd(cmd: str) -> bool:
        return shutil.which(cmd) is not None

    def _copy_to_local_dst(src_dir: str, dst_dir: str):
        os.makedirs(dst_dir, exist_ok=True)
        for name in os.listdir(src_dir):
            src_path = os.path.join(src_dir, name)
            dst_path = os.path.join(dst_dir, name)
            if os.path.isdir(src_path):
                # Copy directories recursively (skip if already exist)
                if not os.path.exists(dst_path):
                    shutil.copytree(src_path, dst_path)
            else:
                shutil.copy2(src_path, dst_path)

    def _copy_local_to_hdfs(src_dir: str, hdfs_dst: str):
        # Requires Hadoop CLI (hdfs)
        if not _has_cmd('hdfs'):
            print(f"[WARN] 'hdfs' CLI not found. Skipping copy to {hdfs_dst}.")
            return
        try:
            subprocess.run(['hdfs', 'dfs', '-mkdir', '-p', hdfs_dst], check=True)
            # Put all files from src_dir into hdfs_dst
            subprocess.run(['hdfs', 'dfs', '-put', '-f', os.path.join(src_dir, '*'), hdfs_dst], check=True, shell=False)
        except subprocess.CalledProcessError as e:
            print(f"[WARN] Failed to copy to HDFS ({hdfs_dst}): {e}")

    if hdfs_dir is not None:
        if isinstance(hdfs_dir, str) and hdfs_dir.startswith('hdfs://'):
            _copy_local_to_hdfs(local_dir, hdfs_dir)
        else:
            _copy_to_local_dst(local_dir, hdfs_dir)

    print(f"Saved parquet: {train_parquet}, {test_parquet}")
    print(f"Saved jsonl: {train_jsonl}, {test_jsonl}")
