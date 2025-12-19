#!/usr/bin/env python3
"""Evaluate ShortGPT on shortest path data."""

import argparse
import torch
from tqdm import tqdm

from src.config import ShortGPTConfig, DataConfig
from src.tokenizer import ShortGPTTokenizer
from src.model import ShortGPT
from src.rl.reward import compute_path_reward
from data import get_splits


def evaluate(model, rows, tokenizer, device, max_new_tokens=64):
    """Evaluate model and return metrics."""
    model.eval()
    model.to(device)

    exact, valid, optimal, total_reward = 0, 0, 0, 0.0

    for row in tqdm(rows, desc="Evaluating"):
        generated = model.generate(tokenizer, row, device, max_new_tokens)
        reward = compute_path_reward(row, generated, tokenizer)

        total_reward += reward
        if reward > 0:
            valid += 1
        if reward == 1.0:
            optimal += 1

        # Check exact match
        gt = row["serialized_path"]
        if "<START_PATH>" in generated and "<END_PATH>" in generated:
            start = generated.index("<START_PATH>")
            end = generated.index("<END_PATH>") + len("<END_PATH>")
            if generated[start:end] == gt:
                exact += 1

    n = len(rows)
    return {
        "exact_match": exact / n,
        "valid_path": valid / n,
        "optimal": optimal / n,
        "avg_reward": total_reward / n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to JSONL data")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--compare", type=str, default=None, help="Second checkpoint to compare")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model_config = ShortGPTConfig()
    data_config = DataConfig()
    tokenizer = ShortGPTTokenizer()

    # Load data with SAME splits
    train_rows, val_rows, test_rows = get_splits(args.data, data_config)
    rows = {"train": train_rows, "val": val_rows, "test": test_rows}[args.split]
    print(f"Evaluating on {len(rows)} {args.split} examples")

    # Evaluate first model
    print(f"\n=== {args.checkpoint} ===")
    model1 = ShortGPT.from_pretrained(args.checkpoint, model_config)
    results1 = evaluate(model1, rows, tokenizer, device)

    print(f"Exact match: {results1['exact_match']:.2%}")
    print(f"Valid path:  {results1['valid_path']:.2%}")
    print(f"Optimal:     {results1['optimal']:.2%}")
    print(f"Avg reward:  {results1['avg_reward']:.4f}")

    # Optionally compare
    if args.compare:
        print(f"\n=== {args.compare} ===")
        model2 = ShortGPT.from_pretrained(args.compare, model_config)
        results2 = evaluate(model2, rows, tokenizer, device)

        print(f"Exact match: {results2['exact_match']:.2%}")
        print(f"Valid path:  {results2['valid_path']:.2%}")
        print(f"Optimal:     {results2['optimal']:.2%}")
        print(f"Avg reward:  {results2['avg_reward']:.4f}")

        print(f"\n=== Difference (model2 - model1) ===")
        print(f"Exact match: {results2['exact_match'] - results1['exact_match']:+.2%}")
        print(f"Valid path:  {results2['valid_path'] - results1['valid_path']:+.2%}")
        print(f"Optimal:     {results2['optimal'] - results1['optimal']:+.2%}")
        print(f"Avg reward:  {results2['avg_reward'] - results1['avg_reward']:+.4f}")


if __name__ == "__main__":
    main()
