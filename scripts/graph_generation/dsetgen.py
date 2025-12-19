"""
Dataset generation script for ShortGPT.

Generates training examples from graph pickle files.
Each example contains a graph representation and a shortest path query.
"""

import json
import random
import argparse
import itertools
import pickle
import time
from pathlib import Path
from typing import Dict, List

import networkx as nx


def generate_adjacency_list(G: nx.Graph) -> Dict[int, List[int]]:
    """Return adjacency list as a dictionary."""
    adj_list = {node: [] for node in G.nodes()}
    for u, v in G.edges():
        adj_list[u].append(v)
        adj_list[v].append(u)
    return adj_list


def serialize_graph(G: nx.Graph) -> str:
    """Serialize graph using the ShortGPT vocabulary."""
    edges = list(G.edges())
    if len(edges) == 0:
        return ""
    s = ""
    for u, v in edges:
        s += f"<EDGE>{u}<BD>{v}"
    return s


def shortest_path_repr(G: nx.Graph, s: int, t: int) -> List[int]:
    """Return the shortest path via BFS."""
    return nx.shortest_path(G, source=s, target=t)


def serialize_shortest_path(path: List[int]) -> str:
    """Serialize path using the ShortGPT vocabulary."""
    s = "<START_PATH>"
    s += str(path[0])
    for node in path[1:]:
        s += f"<TO>{node}"
    s += "<END_PATH>"
    return s


def connected_graph(G: nx.Graph) -> bool:
    """Check if the graph is connected."""
    return nx.is_connected(G)


def build_example(id: int, cnt: int, G: nx.Graph) -> Dict:
    """Build a full training example."""
    nodes = list(G.nodes())
    s, t = random.sample(nodes, 2)
    try:
        path = shortest_path_repr(G, s, t)
    except nx.NetworkXNoPath:
        path = []
    path_len = len(path) - 1 if path else -1
    serialized_path = serialize_shortest_path(path) if path else "<NO_PATH>"
    example = {
        "id": id,
        "nid": cnt,
        "num_nodes": len(nodes),
        "adl": generate_adjacency_list(G),
        "graph_repr": serialize_graph(G),
        "origin": s,
        "destination": t,
        "shortest_path": path,
        "shortest_path_length": path_len,
        "serialized_path": serialized_path,
    }
    return example


def exhash(example: Dict) -> int:
    """Create a hash for a training example based on its content."""
    hsh = hash((
        example["num_nodes"],
        example["graph_repr"],
        example["origin"],
        example["destination"],
        example["shortest_path_length"],
        example["serialized_path"],
    ))
    return hsh


def relabel_graph(G: nx.Graph) -> nx.Graph:
    """Relabel the nodes of a graph to use nodes 1-15."""
    nodes = list(G.nodes())
    rnodes = list(itertools.combinations(range(1, 16), len(nodes)))[0]
    new_nodes = list(rnodes)
    mapping = {nodes[idx]: new_nodes[idx] for idx in range(len(nodes))}
    relabeled_graph = nx.relabel_nodes(G, mapping)
    return relabeled_graph


def runner(start: int, end: int, graphs_dir: Path, output_dir: Path):
    """
    Generate dataset examples for graphs with node counts from start to end.

    Args:
        start: Minimum number of nodes
        end: Maximum number of nodes
        graphs_dir: Directory containing graph pickle files
        output_dir: Directory to save output JSONL files
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    files = []
    id = 1

    for i in range(start, end + 1):
        cnt = 1
        start_time = time.time()

        graph_file = graphs_dir / f"graphs_{i}.pkl"
        if not graph_file.exists():
            print(f"Warning: {graph_file} not found, skipping n={i}")
            continue

        with open(graph_file, "rb") as f:
            graphs = pickle.load(f)
            op = output_dir / f"graph_{i}_ex.jsonl"

            with open(op, "w") as out:
                print(f"Processing {len(graphs)} graphs for n={i}")

                if i < 6:
                    # For small graphs, enumerate all node label combinations
                    for j in range(len(graphs)):
                        nodes = list(graphs[j].nodes)
                        pairs = itertools.combinations(range(1, 16), i)
                        for k in pairs:
                            new_nodes = list(k)
                            mapping = {nodes[idx]: new_nodes[idx] for idx in range(len(nodes))}
                            relabeled_graph = nx.relabel_nodes(graphs[j], mapping)
                            ex = build_example(id, cnt, relabeled_graph)
                            id += 1
                            out.write(json.dumps(ex) + "\n")
                            cnt += 1
                    end_time = time.time()
                    print(f"Generated {cnt-1} examples for n={i} in {end_time - start_time:.4f} seconds.")

                elif i >= 6 and i <= 8:
                    # For medium graphs, sample up to max_limit examples
                    max_limit = 100000
                    while cnt < max_limit:
                        for g in graphs:
                            if cnt >= max_limit:
                                break
                            nodes = list(g.nodes())
                            new_nodes = random.sample(range(1, 16), len(nodes))
                            mapping = {nodes[idx]: new_nodes[idx] for idx in range(len(nodes))}
                            relabeled_graph = nx.relabel_nodes(g, mapping)
                            ex = build_example(id, cnt, relabeled_graph)
                            out.write(json.dumps(ex) + "\n")
                            id += 1
                            cnt += 1
                    end_time = time.time()
                    print(f"Generated {cnt-1} examples for n={i} in {end_time - start_time:.4f} seconds.")

                else:  # i > 8
                    # For large graphs, use each graph once with relabeling
                    for graph in graphs:
                        if i != 15:
                            relabeled_graph = relabel_graph(graph)
                        else:
                            relabeled_graph = graph
                        ex = build_example(id, cnt, relabeled_graph)
                        id += 1
                        cnt += 1
                        out.write(json.dumps(ex) + "\n")
                    end_time = time.time()
                    print(f"Generated {cnt-1} examples for n={i} in {end_time - start_time:.4f} seconds.")

                print(f"Examples saved to {op}")
                files.append(op)

        print(f"Generated {id-1} examples in total so far.")

    return files


def merge_files(files: List[Path], output_file: Path):
    """Merge multiple JSONL files into one."""
    with open(output_file, "w") as f:
        for file in files:
            with open(file, "r") as f2:
                for line in f2:
                    f.write(line)
    print(f"Merged {len(files)} files to {output_file}.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate dataset examples from graph pickle files."
    )
    parser.add_argument(
        "start",
        type=int,
        help="Minimum number of nodes in graphs",
    )
    parser.add_argument(
        "end",
        type=int,
        help="Maximum number of nodes in graphs",
    )
    parser.add_argument(
        "--graphs-dir",
        type=Path,
        default=Path("data/raw/graphs"),
        help="Directory containing graph pickle files (default: data/raw/graphs)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory to save output files (default: data/processed)",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge all generated files into one",
    )

    args = parser.parse_args()

    files = runner(args.start, args.end, args.graphs_dir, args.output_dir)

    if args.merge and files:
        merged_file = args.output_dir / f"merged_{args.start}_{args.end}.jsonl"
        merge_files(files, merged_file)


if __name__ == "__main__":
    main()
