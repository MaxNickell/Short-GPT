import networkx as nx
import time
from itertools import combinations
import multiprocessing
from collections import deque, defaultdict
import pickle

def check_connectivity_batch(args):
    """
    Worker function to check connectivity for a batch of edge combinations.
    args: (n, nodes, edge_combinations_batch)
    Returns a list of connected graphs (as edge lists).
    Uses BFS for faster connectivity check (approx 3.4x faster than NetworkX).
    """
    n, nodes, edge_combinations = args
    connected_graphs_edges = []
    
    # Pre-allocate visited array and queue to avoid repeated allocation if possible,
    # but inside the loop is cleaner for parallel safety.
    
    for edges in edge_combinations:
        # Fast path for small N
        if n <= 1:
            connected_graphs_edges.append(edges)
            continue
            
        # Build adjacency list
        # Using a fixed size list of lists is faster than dict
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # BFS
        visited_count = 0
        # We can use a simple list as stack for DFS or queue for BFS. 
        # DFS might be slightly faster due to list.pop() being O(1) vs deque.popleft()
        # Let's use DFS with a list stack.
        stack = [0]
        visited = [False] * n
        visited[0] = True
        visited_count = 1
        
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    visited_count += 1
                    stack.append(v)
        
        if visited_count == n:
            connected_graphs_edges.append(edges)
            
    return connected_graphs_edges

def get_degree_sequence_hash(G):
    """Returns a hashable tuple representing the degree sequence."""
    degree_sequence = sorted((d for n, d in G.degree()), reverse=True)
    return tuple(degree_sequence)

def filter_isomorphic_graphs(connected_edge_lists, n):
    """
    Filters a list of graph edge lists for isomorphism.
    Uses degree sequence bucketing to speed up comparisons.
    """
    unique_graphs = []
    # Bucket by degree sequence to reduce comparisons
    buckets = defaultdict(list)
    
    nodes = range(n)
    
    total_graphs = len(connected_edge_lists)
    processed_count = 0
    last_print_time = time.time()
    start_time = time.time()
    
    for edges in connected_edge_lists:
        processed_count += 1
        
        current_time = time.time()
        if current_time - last_print_time >= 10:
            elapsed = current_time - start_time
            print(f"  [Filtering] Processed {processed_count}/{total_graphs} graphs ({processed_count/total_graphs*100:.1f}%) "
                  f"in {elapsed:.1f}s. Found {len(unique_graphs)} unique so far.")
            last_print_time = current_time

        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        
        deg_seq = get_degree_sequence_hash(G)
        
        is_iso = False
        # Only check against graphs with the same degree sequence
        for existing_G in buckets[deg_seq]:
            if nx.is_isomorphic(G, existing_G):
                is_iso = True
                break
        
        if not is_iso:
            buckets[deg_seq].append(G)
            unique_graphs.append(G)
            
    return unique_graphs

def generate_connected_graphs_parallel(n, num_processes=None, max_graphs=100000):
    """
    Generates all non-isomorphic connected graphs of size n using multiprocessing.
    Stops if max_graphs connected graphs are found.
    """
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
        
    nodes = list(range(n))
    possible_edges = list(combinations(nodes, 2))
    total_possible_edges = len(possible_edges)
    
    all_connected_edge_lists = []
    
    print(f"Generating graphs for N={n} using {num_processes} cores (limit={max_graphs} unique)...")
    
    # Buckets for isomorphism checking
    buckets = defaultdict(list)
    unique_graphs = []
    
    # Create a pool of workers
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Iterate through all possible number of edges
        # Minimum edges for connected graph is n-1
        for i in range(n - 1, total_possible_edges + 1):
            # Generate all combinations of edges for this number of edges
            edge_combs = list(combinations(possible_edges, i))
            
            if not edge_combs:
                continue
                
            # Split into chunks for parallel processing
            chunk_size = max(1, len(edge_combs) // (num_processes * 4))
            chunks = [edge_combs[j:j + chunk_size] for j in range(0, len(edge_combs), chunk_size)]
            
            # Prepare arguments for workers
            tasks = [(n, nodes, chunk) for chunk in chunks]
            
            # Run tasks in parallel using imap_unordered to track progress
            total_chunks = len(chunks)
            processed_chunks = 0
            last_print_time = time.time()
            start_time = time.time()
            
            # We process results as they come in and filter immediately
            for res in pool.imap_unordered(check_connectivity_batch, tasks):
                processed_chunks += 1
                
                # 'res' is a list of edge lists (connected graphs)
                for edges in res:
                    G = nx.Graph()
                    G.add_nodes_from(nodes)
                    G.add_edges_from(edges)
                    
                    deg_seq = get_degree_sequence_hash(G)
                    
                    is_iso = False
                    # Only check against graphs with the same degree sequence
                    for existing_G in buckets[deg_seq]:
                        if nx.is_isomorphic(G, existing_G):
                            is_iso = True
                            break
                    
                    if not is_iso:
                        buckets[deg_seq].append(G)
                        unique_graphs.append(G)
                
                current_time = time.time()
                if current_time - last_print_time >= 10:  # Print every 10 seconds
                    elapsed = current_time - start_time
                    print(f"  [Progress] Processed {processed_chunks}/{total_chunks} chunks ({processed_chunks/total_chunks*100:.1f}%) "
                          f"in {elapsed:.1f}s. Found {len(unique_graphs)} unique graphs so far.")
                    last_print_time = current_time
                
                if len(unique_graphs) >= max_graphs:
                    print(f"  [Limit Reached] Found {len(unique_graphs)} unique graphs. Stopping generation.")
                    break
            
            if len(unique_graphs) >= max_graphs:
                break
                
    return unique_graphs

def runner_parallel(start, end, max_graphs=100000):
    for i in range(start, end + 1):
        start_time = time.time()
        graphs = generate_connected_graphs_parallel(i, max_graphs=max_graphs)
        end_time = time.time()
        print(f"Generated {len(graphs)} non-isomorphic graphs for N={i} in {end_time - start_time:.4f} seconds.")
        # save the graphs to a file
        with open(f"graphs_{i}.pkl", "wb") as f:
            pickle.dump(graphs, f)

if __name__ == "__main__":
    # Example usage
    runner_parallel(9, 9, max_graphs=100000)
