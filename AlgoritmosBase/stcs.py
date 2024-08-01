#pip install networkx
import networkx as nx

# Definición de la función truss_decomposition
# Se calcula el truss con el que se va a contener el vertice u y el vertice v, o
# el limite (u ,v)
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

# Definición de la función initialize_subgraph
# Obtencion del subgrafo contemplando los soportes tomados como trusses o 
# confianzas
def initialize_subgraph(G, q, l, h):
    trusses = truss_decomposition(G)
    for k in sorted(trusses.keys(), reverse=True):
        subgraph = G.edge_subgraph(trusses[k])
        for component in nx.connected_components(subgraph):
            if q in component and l <= len(component) <= h:
                return subgraph.subgraph(component)
    return None

# Definición de la función branch_and_bound
# De cada mejor subgrafo inicializado se obtienen candidatos hasta tener 
# el mejor soporte que sea mínimo
def branch_and_bound(G, q, l, h, initial_subgraph):
    best_subgraph = initial_subgraph
    best_min_support = min([G.degree(v) for v in initial_subgraph.nodes])
    
    def recurse(subgraph, candidates, min_support):
        nonlocal best_subgraph, best_min_support
        if len(subgraph.nodes) > h:
            return
        if len(subgraph.nodes) >= l and min_support > best_min_support:
            best_subgraph = subgraph.copy()
            best_min_support = min_support
        for candidate in candidates:
            new_subgraph = subgraph.copy()
            new_subgraph.add_node(candidate)
            new_subgraph.add_edges_from((candidate, neighbor) for neighbor in G.neighbors(candidate) if neighbor in new_subgraph)
            new_candidates = set(G.neighbors(candidate)) - set(new_subgraph.nodes)
            new_min_support = min(min_support, min([new_subgraph.degree(v) for v in new_subgraph.nodes]))
            recurse(new_subgraph, new_candidates, new_min_support)

    initial_candidates = set(G.neighbors(q)) - set(initial_subgraph.nodes)
    recurse(initial_subgraph, initial_candidates, best_min_support)
    return best_subgraph

# Se inicializa la función para obtener subgrafos y mejores ramas y límites
def stcs(G, q, l, h):
    initial_subgraph = initialize_subgraph(G, q, l, h)
    if initial_subgraph:
        return branch_and_bound(G, q, l, h, initial_subgraph)
    return None

# Ejemplo de uso
G = nx.karate_club_graph()
q = 0
l = 6
h = 11
community = stcs(G, q, l, h)
print(f"Vertices de la comunidad: {community.nodes}")
print(f"Aristas de la comunidad: {community.edges}")

# NOTA, COMMUNITY retorna un grafo que cumpla con la query y límites 
# establecidos si dentro de estos límites se encontró un mejor límite mínimo 
# de confianza o truss. Caso contrario retorna None.