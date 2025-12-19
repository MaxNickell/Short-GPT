"""Dense reward function for ShortGPT RL training."""

from src.tokenizer import ShortGPTTokenizer


def compute_path_reward(
    row: dict,
    generated_str: str,
    tokenizer: ShortGPTTokenizer,
) -> float:
    """
    Compute a dense reward for a generated path sequence.

    Three cases:
      Case 1 - Invalid structure (reward = -1.0):
        Output doesn't follow <START_PATH>node<TO>node<TO>...<TO>node<END_PATH> format

      Case 2 - Valid structure, invalid path (reward in [-1.0, 0.0)):
        Correct format but uses non-existent edges.
        R = -0.5 + (valid_edges/total_edges) * 0.5
            - 0.25 if first node != origin
            - 0.25 if last node != destination

      Case 3 - Valid path (reward in (1.0, 2.0]):
        All edges exist and path connects origin to destination.
        R = 1 + L*/L
        where L* = optimal path length, L = generated path length
    """
    origin = row["origin"]
    destination = row["destination"]
    adl = row["adl"]
    L_opt = row["shortest_path_length"]

    # ------------------------
    # 1) Tokenize and extract path segment
    # ------------------------
    try:
        tokens = tokenizer.tokenize_string(generated_str)
    except ValueError:
        # Malformed string (contains tokens not in vocabulary)
        return -1.0

    # Find <START_PATH> and <END_PATH>
    try:
        start_idx = tokens.index("<START_PATH>")
        end_idx = tokens.index("<END_PATH>")
    except ValueError:
        # Missing one of the markers
        return -1.0

    if end_idx <= start_idx:
        # End comes before start
        return -1.0

    # Path segment: [<START_PATH>, ..., <END_PATH>]
    path_tokens = tokens[start_idx:end_idx + 1]

    # Must start and end correctly
    if path_tokens[0] != "<START_PATH>" or path_tokens[-1] != "<END_PATH>":
        return -1.0

    # Internal tokens between start and end
    internal = path_tokens[1:-1]

    # We expect pattern: num (<TO> num)* => length >= 1 and odd
    # Minimum valid: single node (but that means 0 edges, only valid if origin=dest)
    if len(internal) < 1:
        return -1.0

    # For a path with edges, need odd number of tokens: node, <TO>, node, <TO>, node...
    if len(internal) > 1 and (len(internal) % 2) != 1:
        return -1.0

    # ------------------------
    # 2) Check structural pattern and extract node sequence
    # ------------------------
    node_tokens = []

    for j, tok in enumerate(internal):
        if j % 2 == 0:
            # Even positions: must be a node token "0".."15"
            if tok not in tokenizer.tokens or tok.startswith("<"):
                return -1.0
            try:
                node_id = int(tok)
            except ValueError:
                return -1.0
            node_tokens.append(node_id)
        else:
            # Odd positions: must be <TO>
            if tok != "<TO>":
                return -1.0

    if len(node_tokens) < 1:
        return -1.0

    # Special case: single node path (0 edges)
    if len(node_tokens) == 1:
        # Only valid if origin == destination and the single node matches
        if origin == destination and node_tokens[0] == origin:
            return 2.0  # Optimal for L*=0 case
        else:
            # Invalid structure for a path that should have edges
            return -1.0

    # ------------------------
    # 3) Check edges and compute reward
    # ------------------------
    def neighbors(u: int) -> set:
        """Get neighbors as a set of integers for consistent comparison."""
        raw_neighbors = adl.get(str(u)) or adl.get(u) or []
        return {int(n) for n in raw_neighbors}

    # Count valid and total edges
    total_edges = len(node_tokens) - 1
    valid_edges = 0

    for u, v in zip(node_tokens[:-1], node_tokens[1:]):
        if v in neighbors(u):
            valid_edges += 1

    # Check if path is fully valid (all edges exist)
    if valid_edges == total_edges:
        # Also need correct start and end for a valid path
        if node_tokens[0] == origin and node_tokens[-1] == destination:
            # Case 3: Valid path
            L = total_edges
            if L_opt <= 0:
                # Edge case: origin == destination, any valid path is optimal
                return 2.0
            return 1.0 + L_opt / L
        # Fall through to Case 2 (all edges valid but wrong endpoints)

    # Case 2: Valid structure, invalid path
    # Base reward based on edge validity
    if total_edges > 0:
        R_base = -0.5 + (valid_edges / total_edges) * 0.5
    else:
        R_base = -0.5

    # Endpoint penalties
    start_penalty = 0.25 if node_tokens[0] != origin else 0.0
    end_penalty = 0.25 if node_tokens[-1] != destination else 0.0

    reward = R_base - start_penalty - end_penalty

    return float(reward)
