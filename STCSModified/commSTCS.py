# Original source in c++:
# https://github.com/harrycoder17/Size-constrained-Community-Search

import networkx as nx
from itertools import combinations

def truss_decomposition(G):
    """Truss decomposition that returns the trussness of each edge."""
    trussness = {}
    
    # Calcula el k-truss para diferentes valores de k utilizando NetworkX
    max_k = max(nx.core_number(G).values())  # Esto nos da un límite superior aproximado para k
    
    for k in range(2, max_k + 1):
        k_truss_graph = nx.k_truss(G, k)
        for edge in k_truss_graph.edges():
            # Asegura que las aristas se almacenan como tuplas ordenadas
            trussness[tuple(sorted(edge))] = k
    
    return trussness

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

def node_trussness(G, trussness):
    """Calcula el trussness de cada nodo basado en las aristas que lo conectan."""
    node_truss = {}
    for node in G.nodes:
        connected_edges = list(G.edges(node))
        if connected_edges:
            node_truss[node] = min(trussness[tuple(sorted(edge))] for edge in connected_edges)
        else:
            node_truss[node] = 0  # Si no hay aristas conectadas, trussness es 0
    return node_truss

def ST_Base(G, q, l, h, trussness):
    """ST-Base heuristic algorithm."""
    node_truss = node_trussness(G, trussness)  # Obtener trussness por nodo
    
    if node_truss[q] > h:
        k_star = h
        C = {q}
    else:
        k_values = [
            k for k in range(2, max(trussness.values()) + 1) 
            if len([v for v in nx.node_connected_component(G, q) if node_truss[v] >= k]) >= l
        ]
        
        if k_values:  # Verifica si la lista no está vacía
            k_star = max(k_values)
        else:
            k_star = 2  # Asignar un valor predeterminado, por ejemplo, el mínimo k posible

        if len([v for v in nx.node_connected_component(G, q) if node_truss[v] >= k_star]) <= h:
            return nx.subgraph(G, nx.node_connected_component(G, q)), k_star
        
        C = set([v for v in nx.node_connected_component(G, q) if node_truss[v] >= k_star + 1]) | {q}
    
    R = set([v for v in G.nodes if node_truss[v] >= k_star]) - C
    
    while len(C) < h:
        v = max(R, key=lambda x: (len(set(G.neighbors(x)) & C), node_truss[x]))
        C.add(v)
        R.remove(v)
        R.update(set([u for u in set(G.neighbors(v)) if node_truss[u] >= k_star]) - C)

    H = nx.subgraph(G, C)
    
    k_prime_values = [
        k for k in range(2, max(trussness.values()) + 1) 
        if len([v for v in nx.node_connected_component(H, q) if node_truss[v] >= k]) >= l
    ]
    
    if k_prime_values:
        k_prime = max(k_prime_values)
    else:
        k_prime = 2  # Asignar un valor predeterminado si la lista está vacía

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
    node_truss = node_trussness(G, trussness)  # Asegúrate de calcular el trussness por nodo
    
    for u in C:
        budget_u = min(h - len(C), len(set(G.neighbors(u)) & R))
        
        # Calcula el trussness de nodo como el mínimo trussness de sus aristas conectadas
        connected_edges = [tuple(sorted(edge)) for edge in G.edges(u)]
        min_truss_u = min(trussness[edge] for edge in connected_edges if edge in trussness)
        
        cost_u = max(k_prime + 1 - min_truss_u, 0)
        
        if budget_u < cost_u:
            return False
    
    b_min = min([min(h - len(C), len(set(G.neighbors(u)) & R)) for u in C])
    c_max = max([max(k_prime + 1 - min_truss_u, 0) for u in C])
    
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

    # Asegurarse de que k_hat solo se calcule si hay valores posibles
    k_values = [
        k for k in range(2, max(trussness.values()) + 1)
        if len([v for v in nx.node_connected_component(G, q)
                if (tuple(sorted((v, q))) in trussness and trussness[tuple(sorted((v, q)))] >= k)]) >= l
    ]

    if k_values:  # Verifica si la lista no está vacía
        k_hat = max(k_values)
        if k_hat > k_prime:
            k_prime = k_hat
            H = nx.subgraph(G, C)
    else:
        k_hat = None  # O algún valor predeterminado si la lista está vacía
    
    if len(C) < h and len(R) > 0 and BranchCheck(C, R, G, l, h, k_prime, trussness):
        R = {v for v in R if len(set(G.neighbors(v)) & C) + h - len(C) - 1 >= k_prime}
        
        def node_min_truss(node):
            """Calcula el trussness mínimo de las aristas conectadas a un nodo."""
            connected_edges = [tuple(sorted(edge)) for edge in G.edges(node)]
            return min((trussness[edge] for edge in connected_edges if edge in trussness), default=float('inf'))
        
        v_star = max(R, key=lambda x: (len(set(G.neighbors(x)) & C), node_min_truss(x)))
        
        V_star = {v_star} | {u for u in set(G.neighbors(v_star)) if node_min_truss(u) >= k_prime}
        if V_star:
            H = ST_BandBP(G, q, l, h, k_star, k_prime, C | V_star, R - V_star, trussness)
    
    return H if H is not None else nx.subgraph(G, C)  # Retorna H o el subgrafo actual





def ST_Exa(G, q, l, h, trussness):
    """ST-Exa: Final exact algorithm."""
    H, k_star = ST_Heu(G, q, l, h, trussness)
    
    # Calcula k_prime como el mínimo trussness de las aristas en el subgrafo H
    k_prime = min(trussness[tuple(sorted(edge))] for edge in H.edges)
    
    if k_prime != k_star:
        C = {q}
        R = set(
            v for v in G.nodes
            if (q, v) in trussness or (v, q) in trussness and 
               trussness[tuple(sorted((v, q)))] >= k_prime + 1
        )
        H = ST_BandBP(G, q, l, h, k_star, k_prime, C, R, trussness)
    
    return H

def multi_cluster_STCS(G, l, h, trussness):
    """Genera múltiples clusters utilizando el algoritmo STCS."""
    all_clusters = []
    assigned_nodes = set()  # Para llevar un registro de los nodos ya asignados a un cluster

    for q in G.nodes:
        if q not in assigned_nodes:
            # Ejecuta ST_Exa para el nodo q que aún no está asignado
            H = ST_Exa(G, q, l, h, trussness)
            
            # Verifica si el cluster cumple con las restricciones de tamaño
            if l <= len(H.nodes) <= h:
                # Añadir los nodos del cluster H a la lista de nodos asignados
                assigned_nodes.update(H.nodes)
                
                # Añadir el subgrafo H a la lista de clusters
                all_clusters.append(H)
    
    return all_clusters

# Ejemplo de uso
G = nx.karate_club_graph()  # Usando un grafo de ejemplo de NetworkX
trussness = truss_decomposition(G)
l, h = 3,7   # Restricciones de tamaño

clusters = multi_cluster_STCS(G, l, h, trussness)

# Mostrar los clusters
for i, cluster in enumerate(clusters):
    print(f"Cluster {i + 1}: {cluster.nodes}")

#problemas:
# se repiten nodos en varios clusters (por corregir)