#!/usr/bin/env python3
"""Evaluate model on test set with detailed per-example metrics."""

import argparse
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm

from src.config import ShortGPTConfig, DataConfig
from src.tokenizer import ShortGPTTokenizer
from src.model import ShortGPT
from data import get_splits


def check_valid_structure(generated: str, tokenizer: ShortGPTTokenizer) -> tuple[bool, list[int] | None]:
    """
    Check if output has valid structure: <START_PATH>node<TO>...<TO>node<END_PATH>

    Returns:
        (is_valid, node_list or None)
    """
    try:
        tokens = tokenizer.tokenize_string(generated)
    except ValueError:
        return False, None

    try:
        start_idx = tokens.index("<START_PATH>")
        end_idx = tokens.index("<END_PATH>")
    except ValueError:
        return False, None

    if end_idx <= start_idx:
        return False, None

    internal = tokens[start_idx + 1:end_idx]

    # Need at least one node
    if len(internal) < 1:
        return False, None

    # For paths with edges, need odd number: node, <TO>, node, ...
    if len(internal) > 1 and len(internal) % 2 != 1:
        return False, None

    # Extract nodes and verify structure
    nodes = []
    for j, tok in enumerate(internal):
        if j % 2 == 0:
            # Must be a node token (not a special token)
            if tok.startswith("<"):
                return False, None
            try:
                node_id = int(tok)
                nodes.append(node_id)
            except ValueError:
                return False, None
        else:
            # Must be <TO>
            if tok != "<TO>":
                return False, None

    if len(nodes) < 1:
        return False, None

    return True, nodes


def check_valid_path(nodes: list[int], row: dict) -> bool:
    """
    Check if node sequence is a valid path on the graph.
    Requires: correct origin, correct destination, all edges exist.
    """
    origin = row["origin"]
    destination = row["destination"]
    adl = row["adl"]

    # Check endpoints
    if nodes[0] != origin or nodes[-1] != destination:
        return False

    # Helper to get neighbors
    def neighbors(u: int) -> set:
        raw = adl.get(str(u)) or adl.get(u) or []
        return {int(n) for n in raw}

    # Check all edges exist
    for u, v in zip(nodes[:-1], nodes[1:]):
        if v not in neighbors(u):
            return False

    return True


def check_optimal(nodes: list[int], row: dict) -> bool:
    """Check if path length equals optimal shortest path length."""
    generated_length = len(nodes) - 1  # Number of edges
    return generated_length == row["shortest_path_length"]


def evaluate_example(generated: str, row: dict, tokenizer: ShortGPTTokenizer) -> dict:
    """Evaluate a single example and return metrics."""
    valid_structure, nodes = check_valid_structure(generated, tokenizer)

    if not valid_structure:
        return {
            "valid_structure": False,
            "valid_path": False,
            "optimal": False,
        }

    valid_path = check_valid_path(nodes, row)

    if not valid_path:
        return {
            "valid_structure": True,
            "valid_path": False,
            "optimal": False,
        }

    optimal = check_optimal(nodes, row)

    return {
        "valid_structure": True,
        "valid_path": True,
        "optimal": optimal,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to JSONL data")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    # Auto-generate output filename from checkpoint
    checkpoint_name = Path(args.checkpoint).stem
    output_path = f"logs/eval_{checkpoint_name}.jsonl"

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Load model
    model_config = ShortGPTConfig()
    tokenizer = ShortGPTTokenizer()
    model = ShortGPT.from_pretrained(args.checkpoint, model_config)
    model.to(device)
    model.eval()
    print(f"Loaded model from {args.checkpoint}")

    # Load test set
    data_config = DataConfig()
    _, _, test_rows = get_splits(args.data, data_config)
    print(f"Evaluating on {len(test_rows)} test examples")

    # Evaluate
    results = []
    counts = {"valid_structure": 0, "valid_path": 0, "optimal": 0}

    os.makedirs("logs", exist_ok=True)

    with open(output_path, "w") as f:
        for idx, row in enumerate(tqdm(test_rows, desc="Evaluating")):
            generated = model.generate(tokenizer, row, device, max_new_tokens=64)
            metrics = evaluate_example(generated, row, tokenizer)

            result = {
                "idx": idx,
                "num_nodes": row["num_nodes"],
                "shortest_path_length": row["shortest_path_length"],
                "valid_structure": metrics["valid_structure"],
                "valid_path": metrics["valid_path"],
                "optimal": metrics["optimal"],
                "generated": generated,
            }

            f.write(json.dumps(result) + "\n")

            # Update counts
            for key in counts:
                if metrics[key]:
                    counts[key] += 1

    # Print summary
    n = len(test_rows)
    print(f"\n{'='*50}")
    print(f"Results saved to {output_path}")
    print(f"{'='*50}")
    print(f"Valid Structure: {counts['valid_structure']:6d} / {n} ({counts['valid_structure']/n:.2%})")
    print(f"Valid Path:      {counts['valid_path']:6d} / {n} ({counts['valid_path']/n:.2%})")
    print(f"Optimal:         {counts['optimal']:6d} / {n} ({counts['optimal']/n:.2%})")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
