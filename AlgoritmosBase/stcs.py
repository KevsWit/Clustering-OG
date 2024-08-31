#pip install networkx
#pip install itertools
import networkx as nx
import itertools

# Truss decomposition function
def truss_decomposition(G):
    support = {}
    for u, v in G.edges():
        support[(u, v)] = len(list(nx.common_neighbors(G, u, v)))
    trusses = {}
    k = 2
    while support:
        new_support = {}
        for u, v in support:
            if support[(u, v)] >= k - 2:
                new_support[(u, v)] = support[(u, v)]
        if not new_support:
            break
        trusses[k] = list(new_support.keys())
        support = new_support
        k += 1
    return trusses

# Initialize subgraph function
def initialize_subgraph(G, q, l, h):
    trusses = truss_decomposition(G)
    for k in sorted(trusses.keys(), reverse=True):
        subgraph = G.edge_subgraph(trusses[k])
        for component in nx.connected_components(subgraph):
            if q in component and l <= len(component) <= h:
                return subgraph.subgraph(component)
    return None

# ST-Base algorithm
def st_base(G, q, l, h):
    initial_subgraph = initialize_subgraph(G, q, l, h)
    return initial_subgraph

# ST-Heu algorithm
def st_heu(G, q, l, h):
    initial_subgraph = initialize_subgraph(G, q, l, h)
    if not initial_subgraph:
        return None

    def heuristic_expand(subgraph):
        candidates = set(G.neighbors(q)) - set(subgraph.nodes())
        while len(subgraph.nodes()) < h and candidates:
            candidate = max(candidates, key=lambda node: G.degree(node))
            subgraph.add_node(candidate)
            subgraph.add_edges_from((candidate, neighbor) for neighbor in G.neighbors(candidate) if neighbor in subgraph)
            candidates = set(G.neighbors(candidate)) - set(subgraph.nodes())
        return subgraph

    return heuristic_expand(initial_subgraph)

# Branch and bound function
def branch_and_bound(G, q, l, h, initial_subgraph):
    best_subgraph = initial_subgraph
    best_min_support = min([G.degree(v) for v in initial_subgraph.nodes()])

    def recurse(subgraph, candidates, min_support):
        nonlocal best_subgraph, best_min_support
        if len(subgraph.nodes()) > h:
            return
        if len(subgraph.nodes()) >= l and min_support > best_min_support:
            best_subgraph = subgraph.copy()
            best_min_support = min_support
        for candidate in candidates:
            new_subgraph = subgraph.copy()
            new_subgraph.add_node(candidate)
            new_subgraph.add_edges_from((candidate, neighbor) for neighbor in G.neighbors(candidate) if neighbor in new_subgraph)
            new_candidates = set(G.neighbors(candidate)) - set(new_subgraph.nodes())
            new_min_support = min(min_support, min([new_subgraph.degree(v) for v in new_subgraph.nodes()]))
            recurse(new_subgraph, new_candidates, new_min_support)

    initial_candidates = set(G.neighbors(q)) - set(initial_subgraph.nodes())
    recurse(initial_subgraph, initial_candidates, best_min_support)
    return best_subgraph

# ST-B&B algorithm
def st_bb(G, q, l, h):
    initial_subgraph = initialize_subgraph(G, q, l, h)
    if initial_subgraph:
        return branch_and_bound(G, q, l, h, initial_subgraph)
    return None

# BrandCheck algorithm
def brandcheck(G, q, l, h):
    initial_subgraph = initialize_subgraph(G, q, l, h)
    if not initial_subgraph:
        return None

    def check_structure(subgraph):
        for node in subgraph.nodes():
            neighbors = set(G.neighbors(node))
            if len(neighbors.intersection(subgraph.nodes())) < l:
                return False
        return True

    if check_structure(initial_subgraph):
        return initial_subgraph
    return None

# ST-B&BP algorithm
def st_bbp(G, q, l, h):
    initial_subgraph = initialize_subgraph(G, q, l, h)
    if not initial_subgraph:
        return None

    def branch_and_bound_with_priorities(subgraph, candidates, min_support):
        nonlocal best_subgraph, best_min_support
        if len(subgraph.nodes()) > h:
            return
        if len(subgraph.nodes()) >= l and min_support > best_min_support:
            best_subgraph = subgraph.copy()
            best_min_support = min_support
        for candidate in sorted(candidates, key=lambda x: G.degree(x), reverse=True):
            new_subgraph = subgraph.copy()
            new_subgraph.add_node(candidate)
            new_subgraph.add_edges_from((candidate, neighbor) for neighbor in G.neighbors(candidate) if neighbor in new_subgraph)
            new_candidates = set(G.neighbors(candidate)) - set(new_subgraph.nodes())
            new_min_support = min(min_support, min([new_subgraph.degree(v) for v in new_subgraph.nodes()]))
            branch_and_bound_with_priorities(new_subgraph, new_candidates, new_min_support)

    best_subgraph = initial_subgraph
    best_min_support = min([G.degree(v) for v in initial_subgraph.nodes()])
    initial_candidates = set(G.neighbors(q)) - set(initial_subgraph.nodes())
    branch_and_bound_with_priorities(initial_subgraph, initial_candidates, best_min_support)
    return best_subgraph

# ST-Exa algorithm
def st_exa(G, q, l, h):
    nodes = list(G.nodes())
    best_subgraph = None
    best_min_support = -1

    for subset in itertools.combinations(nodes, l):
        subgraph = G.subgraph(subset)
        if q in subgraph.nodes() and l <= len(subgraph.nodes()) <= h:
            min_support = min([G.degree(v) for v in subgraph.nodes()])
            if min_support > best_min_support:
                best_subgraph = subgraph
                best_min_support = min_support
    return best_subgraph

# Example usage for all algorithms
G = nx.karate_club_graph()
q = 0
l = 8
h = 11

community = st_exa(G, q, l, h)
if community:
    print(f"ST-Exa Vertices: {community.nodes()}")
    print(f"ST-Exa Edges: {community.edges()}")
else:
    print("ST-Exa: No community found.")