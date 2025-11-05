#!/usr/bin/env python3
"""
Create a subset of passages that:
1. Contains ALL passages specified in passage.idx (required)
2. Contains 1% of total passages from the original file
3. Adds random passages to reach 1% target
Preserves exact bytes — CRLF, tabs, and quotes — from the original file.
"""

import ast
import argparse
from pathlib import Path
from tqdm import tqdm
import random


def load_passage_idx(passage_idx_file: str):
    """Load required passage IDs from passage.idx."""
    print(f"Loading required passage IDs from {passage_idx_file}...")
    with open(passage_idx_file, "r", encoding="utf-8") as f:
        data = ast.literal_eval(f.read())
    required_ids = {str(i) for i in data}
    print(f"Loaded {len(required_ids)} required passage IDs")
    return required_ids


def count_total_lines(file_path: str) -> int:
    """Count total lines in file (excluding header)."""
    with open(file_path, "rb") as f:
        total = sum(1 for _ in f) - 1
    return total


def create_subset(passage_idx_file: str, passages_file: str, output_file: str, seed: int = 42):
    random.seed(seed)
    required_ids = load_passage_idx(passage_idx_file)
    total_lines = count_total_lines(passages_file)
    target_count = max(1, total_lines // 100)

    print(f"\nTarget subset size: {target_count} passages (1% of {total_lines})")

    required_lines = []
    remaining_lines = []

    print("Scanning passages file (byte-preserving)...")
    with open(passages_file, "rb") as fin:
        header = fin.readline()
        for line in tqdm(fin, total=total_lines, unit="lines", desc="Reading lines"):
            id_bytes = line.split(b"\t", 1)[0]
            try:
                id_str = id_bytes.decode("utf-8")
            except UnicodeDecodeError:
                continue  # skip malformed lines
            if id_str in required_ids:
                required_lines.append(line)
            else:
                remaining_lines.append(line)

    found_ids = {line.split(b"\t", 1)[0].decode("utf-8") for line in required_lines}
    missing = required_ids - found_ids
    if missing:
        print(f"⚠️  Missing {len(missing)} required IDs (first 10: {list(missing)[:10]})")

    print(f"\nRequired passages found: {len(required_lines)}")
    needed_extra = max(0, target_count - len(required_lines))
    print(f"Additional random passages needed: {needed_extra}")

    if needed_extra > 0:
        sampled_lines = random.sample(remaining_lines, min(needed_extra, len(remaining_lines)))
    else:
        sampled_lines = []

    final_lines = required_lines + sampled_lines

    print(f"\n{'='*60}")
    print("SUBSET SUMMARY")
    print(f"{'='*60}")
    print(f"Total passages in original: {total_lines:,}")
    print(f"Required IDs: {len(required_ids):,}")
    print(f"Required passages found: {len(required_lines):,}")
    print(f"Final subset size: {len(final_lines):,}")
    print(f"Target (1%): {target_count:,}")
    print(f"Percentage of original: {len(final_lines)/total_lines*100:.2f}%")
    print(f"{'='*60}")

    print(f"\nSaving subset to {output_file} (byte-identical lines)...")
    with open(output_file, "wb") as fout:
        fout.write(header)
        fout.writelines(final_lines)
    print(f"✓ Saved {len(final_lines)} lines to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Create byte-identical subset of passages")
    parser.add_argument("--passage_idx", type=str, required=True,
                        help="Path to passage.idx file containing required passage IDs")
    parser.add_argument("--passages_file", type=str, default="psgs_w100.tsv",
                        help="Path to original passages file (default: psgs_w100.tsv)")
    parser.add_argument("--output_file", type=str, default="psgs_w100_subset.tsv",
                        help="Output file path (default: psgs_w100_subset.tsv)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    if not Path(args.passage_idx).exists():
        print(f"Error: Passage index file {args.passage_idx} not found!")
        return
    if not Path(args.passages_file).exists():
        print(f"Error: Passages file {args.passages_file} not found!")
        return

    create_subset(args.passage_idx, args.passages_file, args.output_file, args.seed)


if __name__ == "__main__":
    main()
