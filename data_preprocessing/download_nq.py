#!/usr/bin/env python3
import os
import json
import argparse
from typing import Dict
import datasets


def write_json_array(ds: datasets.Dataset, path: str):
    """Stream a dataset to a single JSON file containing a list of objects.

    This avoids loading the entire dataset into memory.
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write("[")
        first = True
        for row in ds:
            if not first:
                f.write(",\n")
            f.write(json.dumps(row, ensure_ascii=False))
            first = False
        f.write("]\n")


def main():
    parser = argparse.ArgumentParser(description="Download FlashRAG dataset locally and export splits as JSON arrays.")
    parser.add_argument('--local_dir', default='/data/local/search_r1',
                        help='Output directory to store the downloaded dataset and exports.')
    parser.add_argument('--repo_id', default='RUC-NLPIR/FlashRAG_datasets',
                        help='Hugging Face dataset repository ID.')
    parser.add_argument('--section', '--config', dest='section', default='nq',
                        help='Dataset section/config name, e.g., nq, musique, bamboogle.')
    args = parser.parse_args()

    local_dir = args.local_dir
    os.makedirs(local_dir, exist_ok=True)

    # Download dataset
    print(f"Loading dataset: {args.repo_id} (config: {args.section})...")
    try:
        ds_dict: Dict[str, datasets.Dataset] = datasets.load_dataset(args.repo_id, args.section)
    except Exception as e:
        raise SystemExit(f"Failed to load dataset {args.repo_id} with config {args.section}: {e}")

    # Export each split as a single JSON file (array of objects)
    for split, ds in ds_dict.items():
        json_path = os.path.join(local_dir, f"{split}.json")
        print(f"Writing {split} to {json_path}")
        write_json_array(ds, json_path)

    print("Done.")


if __name__ == '__main__':
    main()