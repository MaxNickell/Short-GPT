import json
import random
import networkx as nx
from typing import Dict, List
import sys
import time
import pickle
import itertools


def generate_adjacency_list(G: nx.Graph) -> Dict[int, List[int]]:
    """Return adjacency list as a dictionary."""
    adj_list = {node: [] for node in G.nodes()}
    for u, v in G.edges():
        adj_list[u].append(v)
        adj_list[v].append(u)
    return adj_list

def serialize_graph(G: nx.Graph) -> str:
    """Serialize using your proposal's vocabulary."""
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
    """Serialize path using your vocabulary."""
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
        tuple(example["nodes"]),
        example["graph_repr"],
        example["origin"],
        example["destination"],
        example["shortest_path_length"],
        example["serialized_path"],
    ))
    return hsh

def relabel_graph(G: nx.Graph) -> nx.Graph:
    """Relabel the nodes of a graph."""
    nodes = list(G.nodes())
    rnodes = list(itertools.combinations(range(1, 16), len(nodes)))[0]
    new_nodes = list(rnodes)
    mapping = {nodes[idx]: new_nodes[idx] for idx in range(len(nodes))}
    relabeled_graph = nx.relabel_nodes(G, mapping)
    return relabeled_graph

def runner(start, end):
    files = []
    id = 1
    for i in range(start, end + 1):
        cnt = 1
        start_time = time.time()
        with open('/Users/sivab/Documents/purdue/fall2025/nlp/project/antigrav/graphs/' + f"graphs_{i}.pkl", "rb") as f:
            graphs = pickle.load(f)
            op = f"graph_{i}_ex.jsonl"
            with open(op, "w") as out:
                print(f"Processing {len(graphs)} graphs for n={i}")
                if i < 6:
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
                    print(f"Examples saved to {op}")
                elif i>=6 and i<=8:
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
                    print(f"Generated {cnt} examples for n={i} in {end_time - start_time:.4f} seconds.")
                    print(f"Examples saved to {op}")
                if i > 8:
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
        print(f"Generated {id-1} examples in total.")
    return files

def trunner(start, end):
    for i in range(start, end + 1):
        with open('/Users/sivab/Documents/purdue/fall2025/nlp/project/antigrav/graphs/' + f"graphs_{i}.pkl", "rb") as f:
            graphs = pickle.load(f)
            print(f"Processing {len(graphs)} graphs for n={i}")
            pairs = itertools.combinations(range(1, 16), i)
            pairs = list(pairs)
            print(f"Generated {len(pairs)} combinations for n={i}")

if __name__ == "__main__":
    start = int(sys.argv[1])
    end = int(sys.argv[2])
    files = runner(start, end)
    # merge the files
    print(f"Merging {len(files)} files.")
    merged_file = f"merged_{start}_{end}.jsonl"
    with open(merged_file, "w") as f:
        for file in files:
            with open(file, "r") as f2:
                for line in f2:
                    f.write(line)
    print(f"Merged {len(files)} files to {merged_file}.")

# if __name__ == "__main__":
#     start = int(sys.argv[1])
#     end = int(sys.argv[2])
#     trunner(start, end)
