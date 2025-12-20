#!/usr/bin/env python3
"""Generate evaluation plots comparing pretrained vs RL models."""

import argparse
import json
from collections import defaultdict

import matplotlib.pyplot as plt


# Professional color palette
COLORS = {
    "pretrained": "#2E86AB",  # Steel blue
    "rl": "#E94F37",          # Vermillion red
}


def load_results(path: str) -> list[dict]:
    """Load evaluation results from JSONL file."""
    results = []
    with open(path, "r") as f:
        for line in f:
            results.append(json.loads(line.strip()))
    return results


def compute_metrics_by_key(results: list[dict], key: str) -> dict:
    """
    Compute metrics grouped by a key (num_nodes or shortest_path_length).

    Returns:
        {key_value: {"valid_structure": rate, "valid_path": rate, "optimal": rate, "count": n}}
    """
    groups = defaultdict(lambda: {"valid_structure": 0, "valid_path": 0, "optimal": 0, "count": 0})

    for r in results:
        k = r[key]
        groups[k]["count"] += 1
        if r["valid_structure"]:
            groups[k]["valid_structure"] += 1
        if r["valid_path"]:
            groups[k]["valid_path"] += 1
        if r["optimal"]:
            groups[k]["optimal"] += 1

    # Convert to rates
    metrics = {}
    for k, v in groups.items():
        n = v["count"]
        metrics[k] = {
            "valid_structure": v["valid_structure"] / n * 100,
            "valid_path": v["valid_path"] / n * 100,
            "optimal": v["optimal"] / n * 100,
            "count": n,
        }

    return metrics


def compute_overall_metrics(results: list[dict]) -> dict:
    """Compute overall metrics across all examples."""
    n = len(results)
    return {
        "valid_structure": sum(r["valid_structure"] for r in results) / n * 100,
        "valid_path": sum(r["valid_path"] for r in results) / n * 100,
        "optimal": sum(r["optimal"] for r in results) / n * 100,
        "count": n,
    }


def plot_metric_comparison(pt_metrics: dict, rl_metrics: dict, metric: str, xlabel: str, ylabel: str, title: str, output_path: str):
    """Plot a single metric comparing pretrained vs RL."""
    # Get all keys from both
    all_keys = sorted(set(pt_metrics.keys()) | set(rl_metrics.keys()))

    pt_values = [pt_metrics.get(k, {}).get(metric, 0) for k in all_keys]
    rl_values = [rl_metrics.get(k, {}).get(metric, 0) for k in all_keys]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(all_keys, pt_values, label="Pretrained", color=COLORS["pretrained"], linewidth=2)
    ax.plot(all_keys, rl_values, label="RL Finetuned", color=COLORS["rl"], linewidth=2)

    ax.set_xlabel(xlabel, fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.set_title(title, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Set y-axis to better show differences (with padding)
    all_values = pt_values + rl_values
    y_min = max(0, min(all_values) - 5)
    y_max = min(100, max(all_values) + 5)
    ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def write_csv(pt_overall: dict, rl_overall: dict, pt_by_nodes: dict, rl_by_nodes: dict, pt_by_length: dict, rl_by_length: dict, output_path: str):
    """Write comparison CSV for paper reporting."""
    with open(output_path, "w") as f:
        # Overall Summary
        f.write("# Overall Summary\n")
        f.write("model,valid_structure,valid_path,optimal,count\n")
        f.write(f"pretrained,{pt_overall['valid_structure']:.2f},{pt_overall['valid_path']:.2f},{pt_overall['optimal']:.2f},{pt_overall['count']}\n")
        f.write(f"rl,{rl_overall['valid_structure']:.2f},{rl_overall['valid_path']:.2f},{rl_overall['optimal']:.2f},{rl_overall['count']}\n")
        delta_vs = rl_overall['valid_structure'] - pt_overall['valid_structure']
        delta_vp = rl_overall['valid_path'] - pt_overall['valid_path']
        delta_opt = rl_overall['optimal'] - pt_overall['optimal']
        f.write(f"delta,{delta_vs:+.2f},{delta_vp:+.2f},{delta_opt:+.2f},\n")
        f.write("\n")

        # By Graph Size
        f.write("# By Graph Size\n")
        f.write("num_nodes,pt_valid_structure,pt_valid_path,pt_optimal,rl_valid_structure,rl_valid_path,rl_optimal,count\n")
        all_nodes = sorted(set(pt_by_nodes.keys()) | set(rl_by_nodes.keys()))
        for k in all_nodes:
            pt = pt_by_nodes.get(k, {"valid_structure": 0, "valid_path": 0, "optimal": 0, "count": 0})
            rl = rl_by_nodes.get(k, {"valid_structure": 0, "valid_path": 0, "optimal": 0, "count": 0})
            f.write(f"{k},{pt['valid_structure']:.2f},{pt['valid_path']:.2f},{pt['optimal']:.2f},{rl['valid_structure']:.2f},{rl['valid_path']:.2f},{rl['optimal']:.2f},{pt['count']}\n")
        f.write("\n")

        # By Path Length
        f.write("# By Path Length\n")
        f.write("path_length,pt_valid_structure,pt_valid_path,pt_optimal,rl_valid_structure,rl_valid_path,rl_optimal,count\n")
        all_lengths = sorted(set(pt_by_length.keys()) | set(rl_by_length.keys()))
        for k in all_lengths:
            pt = pt_by_length.get(k, {"valid_structure": 0, "valid_path": 0, "optimal": 0, "count": 0})
            rl = rl_by_length.get(k, {"valid_structure": 0, "valid_path": 0, "optimal": 0, "count": 0})
            f.write(f"{k},{pt['valid_structure']:.2f},{pt['valid_path']:.2f},{pt['optimal']:.2f},{rl['valid_structure']:.2f},{rl['valid_path']:.2f},{rl['optimal']:.2f},{pt['count']}\n")

    print(f"Saved: {output_path}")


def print_summary_table(name: str, overall: dict, by_nodes: dict, by_length: dict):
    """Print summary table to console."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Overall ({overall['count']} examples):")
    print(f"    Valid Structure: {overall['valid_structure']:.2f}%")
    print(f"    Valid Path:      {overall['valid_path']:.2f}%")
    print(f"    Optimal:         {overall['optimal']:.2f}%")

    print(f"\n  By Graph Size:")
    print(f"    {'Nodes':<8} {'Valid Struct':<14} {'Valid Path':<14} {'Optimal':<14} {'Count':<8}")
    print(f"    {'-'*56}")
    for k in sorted(by_nodes.keys()):
        m = by_nodes[k]
        print(f"    {k:<8} {m['valid_structure']:<14.1f} {m['valid_path']:<14.1f} {m['optimal']:<14.1f} {m['count']:<8}")

    print(f"\n  By Shortest Path Length:")
    print(f"    {'Length':<8} {'Valid Struct':<14} {'Valid Path':<14} {'Optimal':<14} {'Count':<8}")
    print(f"    {'-'*56}")
    for k in sorted(by_length.keys()):
        m = by_length[k]
        print(f"    {k:<8} {m['valid_structure']:<14.1f} {m['valid_path']:<14.1f} {m['optimal']:<14.1f} {m['count']:<8}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", type=str, required=True, help="Path to pretrained eval JSONL")
    parser.add_argument("--rl", type=str, required=True, help="Path to RL eval JSONL")
    args = parser.parse_args()

    # Load results
    pt_results = load_results(args.pretrained)
    rl_results = load_results(args.rl)

    # Compute metrics
    pt_by_nodes = compute_metrics_by_key(pt_results, "num_nodes")
    pt_by_length = compute_metrics_by_key(pt_results, "shortest_path_length")
    pt_overall = compute_overall_metrics(pt_results)

    rl_by_nodes = compute_metrics_by_key(rl_results, "num_nodes")
    rl_by_length = compute_metrics_by_key(rl_results, "shortest_path_length")
    rl_overall = compute_overall_metrics(rl_results)

    # Print summary tables
    print_summary_table("Pretrained Model", pt_overall, pt_by_nodes, pt_by_length)
    print_summary_table("RL Finetuned Model", rl_overall, rl_by_nodes, rl_by_length)

    # Generate plots by graph size
    print("\n")
    plot_metric_comparison(
        pt_by_nodes, rl_by_nodes, "valid_structure",
        "Graph Size", "Valid Structure (%)", "Valid Structure by Graph Size",
        "logs/valid_structure_by_graph_size.png"
    )
    plot_metric_comparison(
        pt_by_nodes, rl_by_nodes, "valid_path",
        "Graph Size", "Valid Path (%)", "Valid Path by Graph Size",
        "logs/valid_path_by_graph_size.png"
    )
    plot_metric_comparison(
        pt_by_nodes, rl_by_nodes, "optimal",
        "Graph Size", "Optimal (%)", "Optimal by Graph Size",
        "logs/optimal_by_graph_size.png"
    )

    # Generate plots by path length
    plot_metric_comparison(
        pt_by_length, rl_by_length, "valid_structure",
        "Shortest Path Length", "Valid Structure (%)", "Valid Structure by Path Length",
        "logs/valid_structure_by_path_length.png"
    )
    plot_metric_comparison(
        pt_by_length, rl_by_length, "valid_path",
        "Shortest Path Length", "Valid Path (%)", "Valid Path by Path Length",
        "logs/valid_path_by_path_length.png"
    )
    plot_metric_comparison(
        pt_by_length, rl_by_length, "optimal",
        "Shortest Path Length", "Optimal (%)", "Optimal by Path Length",
        "logs/optimal_by_path_length.png"
    )

    # Write CSV for paper
    write_csv(pt_overall, rl_overall, pt_by_nodes, rl_by_nodes, pt_by_length, rl_by_length, "logs/eval_comparison.csv")


if __name__ == "__main__":
    main()
