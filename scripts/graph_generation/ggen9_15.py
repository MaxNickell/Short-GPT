import random
import time
import pickle
import sys
import networkx as nx

max_limit = 100000  # you can lower this if things get too slow / big


def random_connected_graph(n, extra_edge_prob=0.2):
    """
    Generate a random connected graph on n nodes by:
    1. Starting with a random spanning tree (always connected)
    2. Adding extra edges with some probability
    """
    # Start with a random tree
    G = nx.random_labeled_tree(n)

    # Add extra random edges to make it denser
    nodes = list(G.nodes())
    for i in range(n):
        for j in range(i + 1, n):
            if not G.has_edge(i, j) and random.random() < extra_edge_prob:
                G.add_edge(i, j)

    return G


def graph_key(G):
    """
    Cheap invariant key to bucket graphs:
    - here we use the sorted degree sequence.
    Graphs that are not isomorphic almost always differ here,
    so we only run nx.is_isomorphic inside each bucket.
    """
    deg_seq = tuple(sorted(d for _, d in G.degree()))
    return deg_seq


def generate_random_connected_graphs(n, limit, extra_edge_prob=0.2):
    """
    Generate up to `limit` pairwise non-isomorphic connected graphs
    on n nodes, using random generation + invariant bucketing.
    """
    print(f"Generating random graphs for N={n} with limit={limit}...")

    non_isomorphic_graphs = []
    # buckets: invariant_key -> list of previously seen graphs with that key
    buckets = {}

    count = 0
    tries = 0

    while count < limit:
        tries += 1

        # Always generate a connected graph
        G = random_connected_graph(n, extra_edge_prob=extra_edge_prob)

        key = graph_key(G)
        bucket = buckets.get(key, [])

        # Only compare isomorphism against graphs with the same key
        is_iso = False
        for existing_G in bucket:
            if nx.is_isomorphic(G, existing_G):
                is_iso = True
                break

        if is_iso:
            continue

        # New non-isomorphic graph
        bucket.append(G)
        buckets[key] = bucket
        non_isomorphic_graphs.append(G)
        count += 1

        if count % 100 == 0:
            print(f"Found {count} non-isomorphic graphs after {tries} tries...")

    print(f"Finished N={n}: {count} non-isomorphic graphs (generated {tries} candidates).")
    return non_isomorphic_graphs


def runner(start, end, limit=max_limit, extra_edge_prob=0.2):
    limiter = {
        2: 1,
        3: 2,
        4: 6,
        5: 21,
        6: 112,
        7: 853,
        8: 11117,
        9: 261080,
        10: 11716571,
    }


    for i in range(start, end + 1):
        start_time = time.time()
        lmt = min(limit, limiter.get(i, limit))
        graphs = generate_random_connected_graphs(i, lmt, extra_edge_prob=extra_edge_prob)
        end_time = time.time()

        print(f"N={i}: Generated {len(graphs)} graphs in {end_time - start_time:.4f} seconds.")

        # Save graphs for this N
        with open(f"graphsalt_{i}.pkl", "wb") as f:
            pickle.dump(graphs, f)
        print(f"Saved graphs for N={i} to graphs_{i}.pkl")

if __name__ == "__main__":
    start = int(sys.argv[1])
    end = int(sys.argv[2])
    runner(start, end)