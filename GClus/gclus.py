# -*- coding: utf-8 -*-

import networkx as nx
from itertools import combinations
import matplotlib.pyplot as plt
from networkx.algorithms.cuts import conductance
import pymetis

__author__ = """Kevin Castillo (kev.gcastillo@outlook.com)"""
#    Copyright (C) 2024 by
#    Kevin Castillo <kev.gcastillo@outlook.com>
#    All rights reserved.


def visualize_clusters(G, clusters):
    """Visualiza el grafo en base a los clusters generados."""
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
    """Descomposición de Truss que devuelve el trussness de cada arista."""
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
            # Utilice únicamente los bordes que existen en trussness, con un valor predeterminado si no se encuentra
            node_truss[node] = min(trussness.get(tuple(sorted(edge)), 2) for edge in connected_edges)
        else:
            node_truss[node] = 0
    return node_truss


def GC_Base(G, q, l, h, trussness):
    """GC-Base heuristic algorithm."""
    node_truss = node_trussness(G, trussness)
    
    if not trussness:  # Verificar si no hay trussness
        return nx.subgraph(G, []), 2  # Retornar un subgrafo vacío y k_star=2 (default value)

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


def GC_Heuristic(G, q, l, h, trussness):
    """GC-Heuristic: Advanced heuristic algorithm to address slow start and branch trap."""
    H, k_star = GC_Base(G, q, l, h, trussness)
    
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
            Hi, k_prime_i = GC_Base(nx.subgraph(G, C), q, l, h, trussness)
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


def GC_BranchBound(C, R, G, l, h, k_prime, trussness):
    """GC-B&B: Basic Branch and Bound algorithm."""
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
        GC_BranchBound(C | {v_star}, R - {v_star}, G, l, h, k_prime, trussness)
        GC_BranchBound(C, R - {v_star}, G, l, h, k_prime, trussness)
    
    return H


def GC_BranchBoundP(G, q, l, h, k_star, k_prime, C, R, trussness):
    """GC-B&BP: Optimized Branch and Bound with Pruning."""
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
                H = GC_BranchBoundP(G, q, l, h, k_star, k_prime, C | V_star, R - V_star, trussness)
    
    return H if H is not None else nx.subgraph(G, C)


def GC_Final(G, q, l, h, trussness):
    """GC-Final: Final algorithm."""
    H, k_star = GC_Heuristic(G, q, l, h, trussness)
    if H.edges:
        k_prime = min(trussness[tuple(sorted(edge))] for edge in H.edges)
    else:
        k_prime = k_star  # Or some other default value
    
    if k_prime != k_star:
        C = {q}
        R = set(
            v for v in G.nodes
            if (q, v) in trussness or (v, q) in trussness and 
               trussness[tuple(sorted((v, q)))] >= k_prime + 1
        )
        if R:
            H = GC_BranchBoundP(G, q, l, h, k_star, k_prime, C, R, trussness)
    
    return H


def combine_small_clusters(clusters, l, h, G):
    """Combina pequeños grupos en función de la similitud para formar grupos dentro de las restricciones de tamaño."""
    combined_clusters = []
    remaining_clusters = []

    for cluster in clusters:
        if len(cluster) < l:
            remaining_clusters.append(cluster)
        else:
            combined_clusters.append(cluster)

    while remaining_clusters:
        # Elegir el cluster a fusionar
        cluster_to_combine = remaining_clusters.pop(0)
        best_merge = None
        best_similarity = -float('inf')

        for i, cluster in enumerate(remaining_clusters):
            # Calcular la similaridad entre dos clusters usando la conductancia como indicador de fusión. 
            similarity = conductance(G, cluster_to_combine, cluster)
            if similarity > best_similarity:
                best_similarity = similarity
                best_merge = i
        
        if best_merge is not None:
            # Combinar los dos clusters más similares
            merged_cluster = cluster_to_combine.union(remaining_clusters.pop(best_merge))
            if l <= len(merged_cluster) <= h:
                combined_clusters.append(merged_cluster)
        else:
            combined_clusters.append(cluster_to_combine)

    return combined_clusters


def split_large_clusters(clusters, h, G):
    """Divide clusters grandes en función de la conectividad interna para formar subclústeres dentro de las restricciones de tamaño."""
    new_clusters = []
    
    for cluster in clusters:
        if len(cluster) > h:
            # Dividir el clúster grande utilizando la estrategia de corte interno
            adjacency_list = []
            nodes = list(cluster)
            for p in nodes:
                adjacency_list.append([nodes.index(nei) for nei in G.neighbors(p) if nei in cluster])

            # Realizar el split
            edgecuts, parts = pymetis.part_graph(2, adjacency_list)

            cluster1 = set(nodes[i] for i in range(len(parts)) if parts[i] == 0)
            cluster2 = set(nodes[i] for i in range(len(parts)) if parts[i] == 1)

            if len(cluster1) > 0:
                new_clusters.append(cluster1)
            if len(cluster2) > 0:
                new_clusters.append(cluster2)
        else:
            new_clusters.append(cluster)

    return new_clusters


def assign_unclustered_nodes(G, all_clusters, l, h):
    """Asignar nodos no agrupados a clústeres existentes en función de la conductancia y la conexión directa."""
    all_clustered_nodes = set.union(*[set(cluster) for cluster in all_clusters])
    unclustered_nodes = set(G.nodes) - all_clustered_nodes

    for node in unclustered_nodes:
        best_cluster = None
        best_conductance = float('inf')  # Comenzar con una conductancia alta (peor de los casos)
        
        # Evaluar la conductancia del nodo con respecto a cada grupo
        for cluster in all_clusters:
            if len(cluster) < h:  # Asegurar que el clúster no haya excedido el límite de tamaño
                
                # Comprobar si el nodo tiene vecinos en el clúster
                neighbors_in_cluster = [neighbor for neighbor in G.neighbors(node) if neighbor in cluster]
                if neighbors_in_cluster:  # Solo considerar los clústeres donde el nodo tenga vecinos directos
                    # Calcular la conductancia entre el nodo y el cluster
                    cond = conductance(G, cluster, {node})
                    
                    if cond < best_conductance:
                        best_conductance = cond
                        best_cluster = cluster

        # Assign the node to the best connected cluster
        if best_cluster is not None:
            best_cluster.add(node)
        else:
            # Si no se encuentra ningún clúster adecuado (caso muy raro), asígnarlo a un clúster con al menos una conexión directa
            # Evitamos crear nuevos clústeres innecesariamente, por lo que elegimos un clúster con vecinos directos.
            for cluster in all_clusters:
                if any(neighbor in cluster for neighbor in G.neighbors(node)):
                    best_cluster = cluster
                    best_cluster.add(node)
                    break
    
    # Combinar grupos pequeños si es necesario
    combined_clusters = combine_small_clusters(all_clusters, l, h, G)
    return combined_clusters


def multi_cluster_STCS(G, l, h, max_iterations=5):
    """Genera múltiples clusters utilizando el algoritmo STCS y garantiza que respeten las restricciones de tamaño."""
    iteration = 0
    final_clusters = []
    
    all_clusters = []
    assigned_nodes = set()
    din_G = G.copy()  # Hacemos una copia del grafo original

    # Paso 1: Generar los clusters iniciales
    for q in G.nodes:  # Convertimos a lista para evitar problemas de modificación durante la iteración
        if q in din_G.nodes:
            trussness = truss_decomposition(din_G)
            H = GC_Final(din_G, q, l, h, trussness)
            H_nodes_filtered = {n for n in H.nodes if n not in assigned_nodes}

            if len(H_nodes_filtered) >= l:
                assigned_nodes.update(H_nodes_filtered)
                all_clusters.append(H_nodes_filtered)
                din_G.remove_nodes_from(H_nodes_filtered)

    while iteration < max_iterations:
        iteration += 1

        # Paso 2: Combina clusters pequeños
        combined_clusters = combine_small_clusters(all_clusters, l, h, G)

        # Paso 3: Divide clusters grandes
        final_clusters = []
        for cluster in combined_clusters:
            final_clusters.extend(split_large_clusters([cluster], h, G))

        # Paso 4: Asigna nodos no clusterizados
        final_clusters = assign_unclustered_nodes(G, final_clusters, l, h)

        # Paso 5: Verificar si todos los nodos están asignados a un cluster
        all_clustered_nodes = set.union(*[set(cluster) for cluster in final_clusters])
        unclustered_nodes = set(G.nodes()) - all_clustered_nodes

        if not unclustered_nodes:
            # Verificar si todos los nodos están asignados a un solo cluster
            num_clusters = len([cluster for cluster in final_clusters if len(cluster) > 0])
            if num_clusters == 1:
                print("Todos los nodos han sido asignados a un solo cluster. Repitiendo combinación y partición.")
            else:
                # Si no quedan nodos sin cluster y hay más de un cluster, salimos del ciclo
                break

        print(f"Iteración {iteration}: Nodos sin cluster: {len(unclustered_nodes)}. Repitiendo combinación y partición.")

    # Si después de las iteraciones quedan nodos sin asignar, retornamos el mejor esfuerzo
    if unclustered_nodes:
        print("Algunos nodos no fueron asignados a clusters respetando los tamaños mínimos.")

    # Convertimos de nuevo a subgrafos y filtramos clusters vacíos
    final_clusters = [G.subgraph(cluster) for cluster in final_clusters if len(cluster) > 0]

    return final_clusters