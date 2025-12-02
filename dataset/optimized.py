import networkx as nx
import time
from itertools import combinations
import pickle
import random
import sys

max_limit = 100000

def generate_connected_graphs_naive(n, limit):
    nodes = range(n)
    possible_edges = list(combinations(nodes, 2))
    non_isomorphic_graphs = []
    total_possible_edges = len(possible_edges)
    
    print(f"Generating graphs for N={n}...")
    count = 0
    
    for i in range(n - 1, total_possible_edges + 1):
        for edges in combinations(possible_edges, i):
            G = nx.Graph()
            G.add_nodes_from(nodes)
            G.add_edges_from(edges)
            
            if nx.is_connected(G):
                is_iso = False
                for existing_G in non_isomorphic_graphs:
                    if nx.is_isomorphic(G, existing_G):
                        is_iso = True
                        break
                
                if not is_iso:
                    non_isomorphic_graphs.append(G)
                    count += 1
                    if count % 10 == 0:
                        print(f"Found {count} non-isomorphic graphs so far...")
                    if count == limit:
                        return non_isomorphic_graphs
                        
    return non_isomorphic_graphs

def generate_random_connected_graphs(n, limit):
    nodes = range(n)
    non_isomorphic_graphs = []
    
    print(f"Generating random graphs for N={n} with limit={limit}...")
    count = 0
    
    while count < limit:
        # Randomly choose number of edges between n-1 (min for connected) and n*(n-1)/2 (complete)
        min_edges = n - 1
        max_edges = n * (n - 1) // 2
        num_edges = random.randint(min_edges, max_edges)
        
        # Generate a random graph
        G = nx.gnm_random_graph(n, num_edges)
        
        if nx.is_connected(G):
            is_iso = False
            for existing_G in non_isomorphic_graphs:
                if nx.is_isomorphic(G, existing_G):
                    is_iso = True
                    break
            
            if not is_iso:
                non_isomorphic_graphs.append(G)
                count += 1
                if count % 100 == 0:
                    print(f"Found {count} non-isomorphic graphs so far...")
                    
    return non_isomorphic_graphs

def runner(start, end):
    for i in range(start, end + 1):
        start_time = time.time()
        graphs = generate_connected_graphs_naive(i, max_limit)
        end_time = time.time()
        print(f"Generated {len(graphs)} graphs in {end_time - start_time:.4f} seconds.")

        # save the graphs to a file
        with open(f"graphs_{i}.pkl", "wb") as f:
            pickle.dump(graphs, f)

if __name__ == "__main__":
    # take input for command line
    start = int(sys.argv[1])
    end = int(sys.argv[2])
    runner(start, end)