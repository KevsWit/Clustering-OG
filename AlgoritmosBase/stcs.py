# Original source in c++:
# https://github.com/harrycoder17/Size-constrained-Community-Search

import networkx as nx
from itertools import combinations

def truss_decomposition(G):
    """Truss decomposition that returns the trussness of each edge."""
    trussness = {}
    # Código para calcular la descomposición de truss, que actualizará 'trussness'
    # Utiliza la función de truss decomposition de NetworkX o implementa la tuya
    return nx.core_number(G)  # Ejemplo con core_number para simplificar

def find_cliques(S, h):
    """Encuentra cliques diversificados en el subgrafo S."""
    # Encuentra todos los cliques en el subgrafo S
    cliques = list(nx.find_cliques(S))
    # Filtra los cliques que tienen un tamaño dentro del límite h
    cliques = [clique for clique in cliques if len(clique) <= h]
    # Ordenar los cliques por tamaño (puede ser un criterio de diversificación)
    cliques.sort(key=len, reverse=True)
    
    # Aquí podemos aplicar una heurística para diversificar, por ejemplo:
    diversified_cliques = []
    for clique in cliques:
        if not any(set(clique).issubset(set(d)) for d in diversified_cliques):
            diversified_cliques.append(clique)
    
    return diversified_cliques[:2]  # Retornar los top 2 cliques diversificados

def ST_Base(G, q, l, h, trussness):
    """ST-Base heuristic algorithm."""
    if trussness[q] > h:
        k_star = h
        C = {q}
    else:
        k_star = max([k for k in range(2, max(trussness.values())+1) 
                      if len([v for v in nx.node_connected_component(G, q) if trussness[v] >= k]) >= l])
        if len([v for v in nx.node_connected_component(G, q) if trussness[v] >= k_star]) <= h:
            return nx.subgraph(G, nx.node_connected_component(G, q)), k_star
        C = set([v for v in nx.node_connected_component(G, q) if trussness[v] >= k_star+1]) | {q}
    
    R = set([v for v in G.nodes if trussness[v] >= k_star]) - C
    
    while len(C) < h:
        v = max(R, key=lambda x: (len(set(G.neighbors(x)) & C), trussness[x]))
        C.add(v)
        R.remove(v)
        R.update(set([u for u in set(G.neighbors(v)) if trussness[u] >= k_star]) - C)  # Corregido

    H = nx.subgraph(G, C)
    k_prime = max([k for k in range(2, max(trussness.values())+1) 
                   if len([v for v in nx.node_connected_component(H, q) if trussness[v] >= k]) >= l])
    
    return H, k_prime

def ST_Heu(G, q, l, h, trussness):
    """ST-Heu: Advanced heuristic algorithm to address slow start and branch trap."""
    H, k_star = ST_Base(G, q, l, h, trussness)
    
    if H.nodes != {q}:
        return H, k_star
    
    D = set()
    max_neighbors = lambda vi: len(set(G.neighbors(vi)) & set(G.neighbors(q)))
    
    candidates = sorted([v for v in G.nodes if v != q and trussness[v] >= k_star and v not in D], key=max_neighbors, reverse=True)
    
    for vi in candidates:
        S = nx.subgraph(G, set(G.neighbors(vi)) & set(G.neighbors(q)))
        L = find_cliques(S, h)
        D.update(L)
        k_prime = 0
        
        for L_clique in L:
            C = {q, vi} | set(L_clique)
            Hi, k_prime_i = ST_Base(nx.subgraph(G, C), q, l, h, trussness)
            if k_prime_i == k_star:
                return Hi, k_prime_i
            elif k_prime_i > k_prime:
                k_prime = k_prime_i
                H = Hi
    
    return H, k_prime

def BranchCheck(C, R, G, l, h, k_prime, trussness):
    """BranchCheck: Budget-Cost Based Bounding."""
    for u in C:
        budget_u = min(h - len(C), len(set(G.neighbors(u)) & R))
        cost_u = max(k_prime + 1 - trussness[u], 0)
        if budget_u < cost_u:
            return False
    
    b_min = min([min(h - len(C), len(set(G.neighbors(u)) & R)) for u in C])
    c_max = max([max(k_prime + 1 - trussness[u], 0) for u in C])
    
    if b_min >= 2 * c_max:
        return True
    
    for u in C:
        if cost_u > 0:
            A = set(G.neighbors(u)) & R
            for x in C - A:
                budget_x = min(h - len(C) - cost_u, len(set(G.neighbors(x)) & R))
                if budget_x < cost_u:
                    return False
    
    return True

def ST_BandB(C, R, G, l, h, k_prime, trussness):
    """ST-B&B: Basic Branch and Bound algorithm."""
    if k_prime == k_star:
        return H
    if len(C) >= l:
        k_hat = max([k for k in range(2, max(trussness.values())+1) 
                     if len([v for v in nx.node_connected_component(G, q) if trussness[v] >= k]) >= l])
        if k_hat > k_prime:
            k_prime = k_hat
            H = nx.subgraph(G, C)
    
    if len(C) < h and len(R) > 0:
        v_star = max(R, key=lambda x: (len(set(G.neighbors(x)) & C), trussness[x]))
        ST_BandB(C | {v_star}, R - {v_star}, G, l, h, k_prime, trussness)
        ST_BandB(C, R - {v_star}, G, l, h, k_prime, trussness)
    
    return H

def ST_BandBP(G, q, l, h, k_star, k_prime, C, R, trussness):
    """ST-B&BP: Optimized Branch and Bound with Pruning."""
    H = None  # Inicializamos H como None para evitar UnboundLocalError
    
    if k_prime == k_star:
        return nx.subgraph(G, C)  # Retorna el subgrafo actual si es óptimo

    if len(C) == h or (len(C) >= l and len(R) == 0):
        k_hat = max([k for k in range(2, max(trussness.values())+1) 
                     if len([v for v in nx.node_connected_component(G, q) if trussness[v] >= k]) >= l])
        if k_hat > k_prime:
            k_prime = k_hat
            H = nx.subgraph(G, C)
    
    if len(C) < h and len(R) > 0 and BranchCheck(C, R, G, l, h, k_prime, trussness):
        R = {v for v in R if len(set(G.neighbors(v)) & C) + h - len(C) - 1 >= k_prime}
        v_star = max(R, key=lambda x: (len(set(G.neighbors(x)) & C), trussness[x]))
        V_star = {v_star} | {u for u in set(G.neighbors(v_star)) if trussness[u] >= k_prime}
        if V_star:
            H = ST_BandBP(G, q, l, h, k_star, k_prime, C | V_star, R - V_star, trussness)
    
    return H if H is not None else nx.subgraph(G, C)  # Retorna H o el subgrafo actual


def ST_Exa(G, q, l, h, trussness):
    """ST-Exa: Final exact algorithm."""
    H, k_star = ST_Heu(G, q, l, h, trussness)
    k_prime = min([trussness[v] for v in H.nodes])
    
    if k_prime != k_star:
        C = {q}
        R = set([v for v in G.nodes if trussness[v] >= k_prime + 1])
        H = ST_BandBP(G, q, l, h, k_star, k_prime, C, R, trussness)
    
    return H

# Ejemplo de uso
G = nx.karate_club_graph()  # Usando un grafo de ejemplo de NetworkX
trussness = truss_decomposition(G)
q = 0  # Nodo de consulta
l, h = 10, 12  # Restricciones de tamaño

H = ST_Exa(G, q, l, h, trussness)
print("Subgrafo resultante:", H.nodes)


# Problemas con ciertos límites
# el algoritmo coloca nodos en la comunidad a pesar de sobrepasar h, modificación de la anterior versión
# se completa los nodos a la cantidad de la comunidad más óptima, ej: de 11 y 12 a la comunidad de 13 nodos