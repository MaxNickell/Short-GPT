Path lengths 14 are just lienar chain
Path Lengths 13 are less of a linear chain
Paths of length 12 are less less of linear chain


Degree Sequence Sum
For each degree sequence sum how does performance change

def graph_key(G):
    """
    Cheap invariant key to bucket graphs:
    - here we use the sorted degree sequence.
    Graphs that are not isomorphic almost always differ here,
    so we only run nx.is_isomorphic inside each bucket.
    """
    deg_seq = tuple(sorted(d for _, d in G.degree()))
    return deg_seq