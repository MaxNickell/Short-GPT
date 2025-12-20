#!/usr/bin/env python3
"""Add degree and choice metrics to eval JSONL files."""

import argparse
import json

from data import get_splits
from src.config import DataConfig


def compute_sum_degree(adl: dict) -> int:
    """Compute sum of degrees from adjacency list."""
    return sum(len(v) for v in adl.values())


def compute_avg_degree(adl: dict) -> float:
    """Compute average degree from adjacency list."""
    if not adl:
        return 0.0
    return compute_sum_degree(adl) / len(adl)


def compute_choices(adl: dict, shortest_path: list) -> tuple[int, float]:
    """
    Compute sum and avg choices along the shortest path.

    For each step (node except destination), count the number of edges (choices).
    """
    if len(shortest_path) < 2:
        return 0, 0.0

    sum_choices = 0
    for node in shortest_path[:-1]:  # Exclude destination
        neighbors = adl.get(str(node)) or adl.get(node) or []
        sum_choices += len(neighbors)

    num_steps = len(shortest_path) - 1  # Number of edges
    avg_choices = sum_choices / num_steps

    return sum_choices, avg_choices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to original JSONL data")
    parser.add_argument("--eval", type=str, required=True, help="Path to eval JSONL to augment")
    args = parser.parse_args()

    # Load test set with same splits
    data_config = DataConfig()
    _, _, test_rows = get_splits(args.data, data_config)
    print(f"Loaded {len(test_rows)} test rows")

    # Load eval results
    with open(args.eval, "r") as f:
        eval_rows = [json.loads(line) for line in f]
    print(f"Loaded {len(eval_rows)} eval rows from {args.eval}")

    # Augment with avg_degree
    for eval_row in eval_rows:
        idx = eval_row["idx"]
        test_row = test_rows[idx]

        # Sanity check: verify match
        if eval_row["num_nodes"] != test_row["num_nodes"]:
            raise ValueError(
                f"Mismatch at idx {idx}: eval num_nodes={eval_row['num_nodes']}, "
                f"test num_nodes={test_row['num_nodes']}"
            )
        if eval_row["shortest_path_length"] != test_row["shortest_path_length"]:
            raise ValueError(
                f"Mismatch at idx {idx}: eval shortest_path_length={eval_row['shortest_path_length']}, "
                f"test shortest_path_length={test_row['shortest_path_length']}"
            )

        # Compute and add degree metrics
        adl = test_row["adl"]
        eval_row["avg_degree"] = compute_avg_degree(adl)
        eval_row["sum_degree"] = compute_sum_degree(adl)

        # Compute and add choice metrics
        shortest_path = test_row["shortest_path"]
        sum_choices, avg_choices = compute_choices(adl, shortest_path)
        eval_row["sum_choices"] = sum_choices
        eval_row["avg_choices"] = avg_choices

    # Rewrite file
    with open(args.eval, "w") as f:
        for row in eval_rows:
            f.write(json.dumps(row) + "\n")

    print(f"Added degree and choice metrics to {len(eval_rows)} rows in {args.eval}")


if __name__ == "__main__":
    main()
