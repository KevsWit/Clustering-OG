import networkx as nx
from itertools import combinations
import matplotlib.pyplot as plt

def visualize_clusters(G, clusters):
    pos = nx.spring_layout(G)  # Layout para los nodos
    cmap = plt.get_cmap('viridis')  # Colormap para los colores

    # Asignar un color único a cada cluster
    for i, cluster in enumerate(clusters):
        color = cmap(i / len(clusters))
        nx.draw_networkx_nodes(G, pos, nodelist=list(cluster.nodes), node_size=300, node_color=[color] * len(cluster.nodes))

    # Dibujar aristas y etiquetas
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos)

    plt.show()

def truss_decomposition(G):
    """Truss decomposition that returns the trussness of each edge."""
    trussness = {}
    
    max_k = max(nx.core_number(G).values())
    
    for k in range(2, max_k + 1):
        k_truss_graph = nx.k_truss(G, k)
        for edge in k_truss_graph.edges():
            trussness[tuple(sorted(edge))] = k
    
    return trussness

def find_cliques(S, h):
    """Encuentra cliques diversificados en el subgrafo S."""
    cliques = list(nx.find_cliques(S))
    cliques = [clique for clique in cliques if len(clique) <= h]
    cliques.sort(key=len, reverse=True)
    
    diversified_cliques = []
    for clique in cliques:
        if not any(set(clique).issubset(set(d)) for d in diversified_cliques):
            diversified_cliques.append(clique)
    
    return diversified_cliques[:2]

def node_trussness(G, trussness):
    """Calcula el trussness de cada nodo basado en las aristas que lo conectan."""
    node_truss = {}
    for node in G.nodes:
        connected_edges = list(G.edges(node))
        if connected_edges:
            node_truss[node] = min(trussness[tuple(sorted(edge))] for edge in connected_edges)
        else:
            node_truss[node] = 0
    return node_truss

def ST_Base(G, q, l, h, trussness):
    """ST-Base heuristic algorithm."""
    node_truss = node_trussness(G, trussness)
    
    if node_truss[q] > h:
        k_star = h
        C = {q}
    else:
        k_values = [
            k for k in range(2, max(trussness.values()) + 1) 
            if len([v for v in nx.node_connected_component(G, q) if node_truss[v] >= k]) >= l
        ]
        
        if k_values:
            k_star = max(k_values)
        else:
            k_star = 2

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
        k_prime = 2

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
    node_truss = node_trussness(G, trussness)
    
    for u in C:
        budget_u = min(h - len(C), len(set(G.neighbors(u)) & R))
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
        k_hat = max([k for k in range(2, max(trussness.values()) + 1) 
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
    H = None
    
    if k_prime == k_star:
        return nx.subgraph(G, C)

    k_values = [
        k for k in range(2, max(trussness.values()) + 1)
        if len([v for v in nx.node_connected_component(G, q)
                if (tuple(sorted((v, q))) in trussness and trussness[tuple(sorted((v, q)))] >= k)]) >= l
    ]

    if k_values:
        k_hat = max(k_values)
        if k_hat > k_prime:
            k_prime = k_hat
            H = nx.subgraph(G, C)
    else:
        k_hat = None
    
    if len(C) < h and len(R) > 0 and BranchCheck(C, R, G, l, h, k_prime, trussness):
        R = {v for v in R if len(set(G.neighbors(v)) & C) + h - len(C) - 1 >= k_prime}
        
        def node_min_truss(node):
            connected_edges = [tuple(sorted(edge)) for edge in G.edges(node)]
            return min((trussness[edge] for edge in connected_edges if edge in trussness), default=float('inf'))

        if R:  # Verificamos si R no está vacío antes de usar max
            v_star = max(R, key=lambda x: (len(set(G.neighbors(x)) & C), node_min_truss(x)))
        
            V_star = {v_star} | {u for u in set(G.neighbors(v_star)) if node_min_truss(u) >= k_prime}
            if V_star:
                H = ST_BandBP(G, q, l, h, k_star, k_prime, C | V_star, R - V_star, trussness)
    
    return H if H is not None else nx.subgraph(G, C)


def ST_Exa(G, q, l, h, trussness):
    """ST-Exa: Final exact algorithm."""
    H, k_star = ST_Heu(G, q, l, h, trussness)
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

def combine_small_clusters(clusters, l, h):
    """Combina clusters pequeños para cumplir con las restricciones de tamaño."""
    combined_clusters = []
    remaining_clusters = []

    for cluster in clusters:
        if len(cluster) < l:
            remaining_clusters.append(cluster)
        else:
            combined_clusters.append(cluster)

    while remaining_clusters:
        cluster_to_combine = remaining_clusters.pop(0)
        for i, cluster in enumerate(remaining_clusters):
            combined_cluster = cluster_to_combine.union(cluster)
            if l <= len(combined_cluster) <= h:
                remaining_clusters.pop(i)
                combined_clusters.append(combined_cluster)
                break
        else:
            combined_clusters.append(cluster_to_combine)

    return combined_clusters

def split_large_clusters(clusters, h):
    """Divide clusters grandes en subclusters más pequeños que respeten el tamaño máximo permitido."""
    new_clusters = []
    
    for cluster in clusters:
        nodes = list(cluster)
        while len(nodes) > h:
            new_clusters.append(nodes[:h])
            nodes = nodes[h:]
        if nodes:
            new_clusters.append(nodes)
    
    return [set(cluster) for cluster in new_clusters if len(cluster) > 0]  # Filtrar clusters vacíos

def multi_cluster_STCS(G, l, h, trussness):
    """Genera múltiples clusters utilizando el algoritmo STCS y garantiza que respeten las restricciones de tamaño."""
    all_clusters = []
    assigned_nodes = set()

    for q in G.nodes:
        if q not in assigned_nodes:
            H = ST_Exa(G, q, l, h, trussness)
            H_nodes_filtered = {n for n in H.nodes if n not in assigned_nodes}
            
            if len(H_nodes_filtered) >= l:
                assigned_nodes.update(H_nodes_filtered)
                all_clusters.append(H_nodes_filtered)
    
    # Combina clusters pequeños
    combined_clusters = combine_small_clusters(all_clusters, l, h)
    
    # Divide clusters grandes
    final_clusters = []
    for cluster in combined_clusters:
        final_clusters.extend(split_large_clusters([cluster], h))
    
    # Convertimos de nuevo a subgrafos
    final_clusters = [G.subgraph(cluster) for cluster in final_clusters if len(cluster) > 0]  # Filtrar clusters vacíos

    return final_clusters

# Ejemplo de uso
G = nx.les_miserables_graph()  # Usando un grafo de ejemplo de NetworkX
trussness = truss_decomposition(G)
l, h = 3, 7  # Restricciones de tamaño

clusters = multi_cluster_STCS(G, l, h, trussness)

# Mostrar los clusters
for i, cluster in enumerate(clusters):
    print(f"Cluster {i + 1}: {cluster.nodes}")

# Dibujar el grafo con las comunidades detectadas
visualize_clusters(G, clusters)

# Observaciones:
# - varios nodos sin cluster
# - posible mejor clusterización de ciertos nodos