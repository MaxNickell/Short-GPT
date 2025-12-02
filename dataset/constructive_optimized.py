import networkx as nx
import time
from itertools import combinations, chain
import multiprocessing
from collections import defaultdict
import pickle

def get_degree_sequence_hash(G):
    """Returns a hashable tuple representing the degree sequence."""
    degree_sequence = sorted((d for n, d in G.degree()), reverse=True)
    return tuple(degree_sequence)

def generate_candidates_chunk(args):
    """
    Worker function to generate candidates for size N from a chunk of graphs of size N-1.
    args: (prev_graphs_chunk, n_prev)
    Returns: list of new graphs (as edge lists) of size n_prev + 1
    """
    prev_graphs_edges, n_prev = args
    new_candidates = []
    
    # The new node index is n_prev
    new_node = n_prev
    existing_nodes = list(range(n_prev))
    
    # Pre-calculate all possible connection subsets (non-empty)
    # We can iterate 1 to n_prev edges connecting to the new node
    # For n_prev=8, 2^8-1 = 255 combinations.
    
    # Optimization: Instead of generating all subsets for every graph, 
    # we can generate them once if n_prev is small.
    # But we need to yield them.
    
    all_subsets = []
    for r in range(1, n_prev + 1):
        all_subsets.extend(combinations(existing_nodes, r))
        
    for edges in prev_graphs_edges:
        # edges is a list of tuples
        base_edges = list(edges)
        
        for subset in all_subsets:
            # Create new edges connecting new_node to nodes in subset
            new_edges = [(u, new_node) for u in subset]
            candidate_edges = base_edges + new_edges
            new_candidates.append(candidate_edges)
            
    return new_candidates

def filter_isomorphic_graphs(candidate_edge_lists, n):
    """
    Filters a list of graph edge lists for isomorphism.
    """
    unique_graphs = []
    buckets = defaultdict(list)
    nodes = range(n)
    
    # We can parallelize the graph creation/hashing if needed, 
    # but let's keep it simple first.
    
    for edges in candidate_edge_lists:
        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        
        deg_seq = get_degree_sequence_hash(G)
        
        is_iso = False
        for existing_G in buckets[deg_seq]:
            if nx.is_isomorphic(G, existing_G):
                is_iso = True
                break
        
        if not is_iso:
            buckets[deg_seq].append(G)
            unique_graphs.append(G)
            
    return unique_graphs

def generate_constructive(target_n, num_processes=None, max_graphs=100000, initial_graphs_file=None):
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
        
    start_n = 2
    current_graphs = [[]] # Base case for N=1 (0 edges)
    
    if initial_graphs_file:
        print(f"Loading initial graphs from {initial_graphs_file}...")
        with open(initial_graphs_file, "rb") as f:
            current_graphs = pickle.load(f)
        
        if not current_graphs:
            print("Error: Loaded graph list is empty.")
            return []
            
        # Check if loaded graphs are nx.Graph objects or edge lists
        if isinstance(current_graphs[0], nx.Graph):
            print("Loaded nx.Graph objects. Converting to edge lists...")
            current_graphs = [list(G.edges()) for G in current_graphs]
            
        # Determine size of loaded graphs
        # current_graphs is now a list of edge lists. 
        # We need to find N. Since they are connected graphs of size N, 
        # the max node index should be N-1.
        # But be careful, for N=1, edges=[], max index is undefined.
        if not current_graphs[0]:
            # Empty edge list could be N=1 or N=0? 
            # Our base case for N=1 is [].
            loaded_n = 1
        else:
            # Find max node index in the first graph
            max_node = 0
            for u, v in current_graphs[0]:
                max_node = max(max_node, u, v)
            loaded_n = max_node + 1
            
        print(f"Resuming from N={loaded_n} with {len(current_graphs)} graphs.")
        start_n = loaded_n + 1
    else:
        print(f"Starting constructive generation for target N={target_n}...")
    
    for n in range(start_n, target_n + 1):
        prev_n = n - 1
        print(f"Generating graphs for N={n} from {len(current_graphs)} graphs of size {prev_n}...")
        
        start_time = time.time()
        
        # Prepare chunks
        chunk_size = max(1, len(current_graphs) // (num_processes * 4))
        chunks = [current_graphs[i:i + chunk_size] for i in range(0, len(current_graphs), chunk_size)]
        
        tasks = [(chunk, prev_n) for chunk in chunks]
        
        all_candidates = []
        
        # 1. Generate Candidates in Parallel
        with multiprocessing.Pool(processes=num_processes) as pool:
            # We can use imap to track generation progress if needed, 
            # but generation is usually fast compared to filtering.
            results = pool.map(generate_candidates_chunk, tasks)
            for res in results:
                all_candidates.extend(res)
                
        print(f"  Generated {len(all_candidates)} candidates. Filtering for isomorphism...")
        
        # 2. Filter Isomorphism
        # For the final layer, we can respect max_graphs
        if n == target_n:
            # We might want to parallelize filtering if candidates are too many
            # But for now let's use the sequential bucket approach which is robust
            unique_graphs = []
            buckets = defaultdict(list)
            nodes = range(n)
            
            total_candidates = len(all_candidates)
            processed = 0
            last_print = time.time()
            
            for edges in all_candidates:
                processed += 1
                if time.time() - last_print > 10:
                    print(f"  [Filtering] {processed}/{total_candidates} ({processed/total_candidates*100:.1f}%) - Found {len(unique_graphs)} unique")
                    last_print = time.time()
                
                G = nx.Graph()
                G.add_nodes_from(nodes)
                G.add_edges_from(edges)
                
                deg_seq = get_degree_sequence_hash(G)
                is_iso = False
                for existing_G in buckets[deg_seq]:
                    if nx.is_isomorphic(G, existing_G):
                        is_iso = True
                        break
                
                if not is_iso:
                    buckets[deg_seq].append(G)
                    unique_graphs.append(G)
                    
                if len(unique_graphs) >= max_graphs:
                    print(f"  [Limit Reached] Found {len(unique_graphs)} unique graphs.")
                    break
            
            current_graphs = [list(G.edges()) for G in unique_graphs] # Store as edges for consistency if we were to continue
            final_graphs = unique_graphs
            
        else:
            # Intermediate layers: must find ALL unique graphs to ensure correctness of next layer
            current_graphs_objs = filter_isomorphic_graphs(all_candidates, n)
            current_graphs = [list(G.edges()) for G in current_graphs_objs]
            print(f"  Found {len(current_graphs)} unique graphs for N={n} in {time.time() - start_time:.2f}s")
            
            # If we are just continuing, we might want to save intermediate results?
            # For now, just keep going.
            final_graphs = current_graphs_objs # In case loop ends here (target_n reached)
            
    return final_graphs

if __name__ == "__main__":
    start_total = time.time()
    graphs = generate_constructive(11, max_graphs=100000)
    print(f"Total time: {time.time() - start_total:.2f}s")
    print(f"Found {len(graphs)} graphs.")
    with open("graphs_constructive_11.pkl", "wb") as f:
        pickle.dump(graphs, f)
