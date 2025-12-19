import math
from src.tokenizer import ShortGPTTokenizer

def compute_path_reward(
    row: dict,
    generated_str: str,
    tokenizer: ShortGPTTokenizer,
) -> float:
    """
    Compute a scalar reward in [0, 1] for a generated sequence.

    Rules:
      1) If the output does NOT have a well-formed path structure:
           <START_PATH> num (<TO> num)* <END_PATH>
         → reward = 0.0  (invalid)

      2) Else, if the node sequence is not a valid path on the graph
         (wrong start, wrong end, or invalid edges)
         → reward = 0.0  (invalid)

      3) Else (valid path), compare its length L to the optimal shortest
         path length L*. We use a relative length penalty:

           rel_extra = max(L - L*, 0) / L*
           reward = max(1.0 - rel_extra, 0.0)

         So:
           - exact shortest path: L = L* → rel_extra = 0 → reward = 1.0
           - slightly longer path: reward in (0,1)
           - much longer path: reward may go toward 0.0
    """
    # ------------------------
    # 1) Tokenize and extract path segment
    # ------------------------
    tokens = tokenizer.tokenize_string(generated_str)

    # Find <START_PATH> and <END_PATH>
    try:
        start_idx = tokens.index("<START_PATH>")
        end_idx = tokens.index("<END_PATH>")
    except ValueError:
        # Missing one of the markers → structurally invalid
        return 0.0

    if end_idx <= start_idx:
        # End comes before start → invalid
        return 0.0

    # Path segment: [<START_PATH>, ..., <END_PATH>]
    path_tokens = tokens[start_idx:end_idx + 1]

    # Must start and end correctly
    if path_tokens[0] != "<START_PATH>" or path_tokens[-1] != "<END_PATH>":
        return 0.0

    # Internal tokens between start and end
    internal = path_tokens[1:-1]  # everything between START and END

    # We expect pattern: num (<TO> num)*  => length >= 1 and odd
    if len(internal) < 1 or (len(internal) % 2) != 1:
        # e.g., just "<START_PATH><END_PATH>" or wrong alternation
        return 0.0

    # ------------------------
    # 2) Check structural pattern and extract node sequence
    # ------------------------
    node_tokens = []

    for j, tok in enumerate(internal):
        if j % 2 == 0:
            # Even positions: must be a node token "0".."15"
            if tok not in tokenizer.tokens or tok.startswith("<"):
                return 0.0
            try:
                node_id = int(tok)
            except ValueError:
                return 0.0
            node_tokens.append(node_id)
        else:
            # Odd positions: must be <TO>
            if tok != "<TO>":
                return 0.0

    # Now node_tokens is [v0, v1, ..., vK]
    if len(node_tokens) < 1:
        return 0.0

    origin = row["origin"]
    destination = row["destination"]

    # ------------------------
    # 3) Graph validity checks
    # ------------------------

    # Check start and end
    if node_tokens[0] != origin:
        return 0.0
    if node_tokens[-1] != destination:
        return 0.0

    # Adjacency list may have string keys and string/int values; handle both
    adl = row["adl"]

    def neighbors(u: int) -> set:
        """Get neighbors as a set of integers for consistent comparison."""
        raw_neighbors = adl.get(str(u)) or adl.get(u) or []
        # Convert all neighbors to int for consistent comparison
        return {int(n) for n in raw_neighbors}

    # Check each edge v_i -> v_{i+1}
    for u, v in zip(node_tokens[:-1], node_tokens[1:]):
        if v not in neighbors(u):
            return 0.0  # invalid edge

    # If we get here, the path is structurally correct AND graph-valid
    # ------------------------
    # 4) Optimality / length-based reward
    # ------------------------
    L_opt = row["shortest_path_length"]  # optimal number of edges

    # Generated path length in edges: len(nodes) - 1
    L = max(len(node_tokens) - 1, 0)

    if L_opt <= 0:
        # Degenerate case; just treat valid path as 1.0
        return 1.0

    # In a correct dataset, L < L_opt shouldn't happen. If it does, clamp.
    extra = max(L - L_opt, 0)
    rel_extra = extra / L_opt  # relative extra steps

    # Reward in [0,1]: 1 when optimal, decreasing as rel_extra grows
    reward = 1.0 - rel_extra
    if reward < 0.0:
        reward = 0.0

    return float(reward)
