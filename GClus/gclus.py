# -*- coding: utf-8 -*-

import networkx as nx
from itertools import combinations
import matplotlib.pyplot as plt
from networkx.algorithms.cuts import conductance
import pymetis
import numpy as np
import copy
from networkx.algorithms.community import modularity

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


def select_pivots(G, num_pivots):
    """Selecciona nodos pivote en base a la centralidad de grado o alguna otra métrica de importancia."""
    centrality = nx.degree_centrality(G)
    sorted_nodes = sorted(centrality, key=centrality.get, reverse=True)
    return sorted_nodes[:num_pivots]


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


def combine_small_clusters(clusters, l_values, h_values, G, pivots):
    """Combina clusters pequeños respetando las restricciones dinámicas de tamaño."""
    combined_clusters = []
    remaining_clusters = []
    assigned_nodes = set()

    # Separar clusters pequeños
    for cluster in clusters:
        if len(cluster) < min(l_values) and not any(node in pivots for node in cluster):
            remaining_clusters.append(cluster)
        else:
            if not cluster & assigned_nodes:
                combined_clusters.append(cluster)
                assigned_nodes.update(cluster)

    # Combinar clusters pequeños
    while remaining_clusters:
        cluster_to_combine = remaining_clusters.pop(0)
        best_merge = None
        best_conductance = float('inf')

        for idx, cluster in enumerate(remaining_clusters):
            cond = conductance(G, cluster_to_combine, cluster)
            if cond < best_conductance:
                best_conductance = cond
                best_merge = idx

        if best_merge is not None:
            merged_cluster = cluster_to_combine.union(remaining_clusters.pop(best_merge))
            merged_cluster -= assigned_nodes
            if any(l <= len(merged_cluster) <= h for l, h in zip(l_values, h_values)):
                combined_clusters.append(merged_cluster)
                assigned_nodes.update(merged_cluster)
        else:
            cluster_to_combine -= assigned_nodes
            if cluster_to_combine:
                combined_clusters.append(cluster_to_combine)
                assigned_nodes.update(cluster_to_combine)

    return combined_clusters


def split_large_clusters(clusters, h, G):
    """Divide clusters grandes para cumplir con las restricciones de tamaño."""
    new_clusters = []
    for cluster in clusters:
        if len(cluster) > h:
            adjacency_list = []
            nodes = list(cluster)
            for node in nodes:
                adjacency_list.append([nodes.index(neighbor) for neighbor in G.neighbors(node) if neighbor in cluster])

            _, parts = pymetis.part_graph(2, adjacency=adjacency_list)

            cluster1 = set(nodes[i] for i in range(len(parts)) if parts[i] == 0)
            cluster2 = set(nodes[i] for i in range(len(parts)) if parts[i] == 1)

            # Validar tamaño de los subclusters
            if len(cluster1) > 0:
                new_clusters.append(cluster1)
            if len(cluster2) > 0:
                new_clusters.append(cluster2)
        else:
            new_clusters.append(cluster)
    return new_clusters


def assign_unclustered_nodes(G, all_clusters, l, h, pivots, blocked_clusters):
    """
    Asignar nodos no agrupados utilizando conductancia, distancia al pivote y una segunda pasada para considerar caminos mínimos.
    Respeta los clusters bloqueados para evitar modificaciones.
    """
    # Ajustar el tamaño de blocked_clusters si no coincide con all_clusters
    if len(blocked_clusters) != len(all_clusters):
        blocked_clusters = blocked_clusters[:len(all_clusters)] + [False] * (len(all_clusters) - len(blocked_clusters))

    # Primera pasada: Asignar nodos no agrupados basándonos en conductancia
    all_clustered_nodes = set().union(*[set(cluster) for cluster in all_clusters])
    unclustered_nodes = set(G.nodes) - all_clustered_nodes

    for node in unclustered_nodes:
        best_cluster = None
        best_conductance = float('inf')

        for idx, cluster in enumerate(all_clusters):
            if idx >= len(blocked_clusters) or blocked_clusters[idx]:  # Saltar clusters bloqueados
                continue
            if len(cluster) < h:
                # Verificar volúmenes antes de calcular la conductancia
                if nx.volume(G, cluster) > 0 and nx.volume(G, {node}) > 0:
                    cond = conductance(G, cluster, {node})
                    if cond < best_conductance:
                        best_conductance = cond
                        best_cluster = cluster
                else:
                    print(f"Advertencia: Nodo {node} o cluster {idx} tiene volumen cero. Ignorando cálculo de conductancia.")

        if best_cluster is not None:
            best_cluster.add(node)

    # Segunda pasada: Reasignar nodos considerando caminos mínimos a los pivotes
    distances_to_pivots = {pivot: nx.single_source_shortest_path_length(G, pivot) for pivot in pivots}

    for cluster_idx, cluster in enumerate(all_clusters):
        if cluster_idx >= len(pivots):
            print(f"Advertencia: El índice {cluster_idx} excede la longitud de pivots. Verifica los datos de entrada.")
            continue  # Saltar clusters que no tienen pivote asociado

        nodes_to_check = list(cluster)  # Copia de los nodos para evitar modificar el conjunto mientras iteramos

        for node in nodes_to_check:
            best_cluster = cluster
            best_distance = distances_to_pivots.get(pivots[cluster_idx], {}).get(node, float('inf'))

            # Evaluar si pertenece a otro clúster
            for idx, pivot in enumerate(pivots):
                if idx != cluster_idx and idx < len(blocked_clusters) and not blocked_clusters[idx]:  # Validar índices
                    distance_to_pivot = distances_to_pivots.get(pivot, {}).get(node, float('inf'))
                    if distance_to_pivot < best_distance:
                        best_distance = distance_to_pivot
                        best_cluster = all_clusters[idx]

            # Reasignar nodo si el mejor clúster es diferente
            if best_cluster != cluster:
                cluster.remove(node)
                best_cluster.add(node)

    return all_clusters


# Ordenar clusters por tamaño y restricciones
def sort_clusters_and_restrictions(clusters, l_values, h_values):
    if len(clusters) != len(l_values) or len(clusters) != len(h_values):
        raise ValueError("Las longitudes de clusters, l_values y h_values no coinciden.")
    
    sorted_indices = sorted(range(len(clusters)), key=lambda i: len(clusters[i]))
    clusters = [clusters[i] for i in sorted_indices]
    l_values = [l_values[i] for i in sorted_indices]
    h_values = [h_values[i] for i in sorted_indices]
    return clusters, l_values, h_values


def multi_cluster_GCLUS(G, h_values, delta=0.2, q_list=None, max_iterations=5):
    """
    Genera múltiples clusters utilizando el algoritmo STCS y garantiza que respeten las restricciones de tamaño especificadas para cada cluster.
    """
    # Verificación de parámetros
    if not (0 < delta < 1):
        raise ValueError("El parámetro delta debe ser mayor a 0 y menor a 1.")
    total_nodes = G.number_of_nodes()
    if sum(h_values) != total_nodes:
        raise ValueError("La suma de h_values debe ser igual a la cantidad de nodos en el grafo.")

    # Configuración inicial
    l_values = [int(h - (h * delta)) for h in h_values]
    num_clusters = len(h_values)
    blocked_clusters = [False] * num_clusters
    q_list = q_list or select_pivots(G, num_clusters)

    # Inicialización de clusters usando GC_Final
    final_clusters = []
    trussness = truss_decomposition(G)
    for idx, q in enumerate(q_list):
        H = GC_Final(G, q, l_values[idx], h_values[idx], trussness)
        cluster_nodes = set(H.nodes)
        final_clusters.append(cluster_nodes)

    # Sincronización inicial
    while len(final_clusters) < num_clusters:
        final_clusters.append(set())
    while len(l_values) < num_clusters:
        l_values.append(min(l_values))
    while len(h_values) < num_clusters:
        h_values.append(max(h_values))

    # Refinamiento iterativo
    for iteration in range(max_iterations):
        print(f"Iteración {iteration + 1}")
        final_clusters, l_values, h_values = sort_clusters_and_restrictions(final_clusters, l_values, h_values)

        # Fase de splits
        split_clusters = []
        for cluster, h in zip(final_clusters, h_values):
            if len(cluster) > h:
                split_clusters.extend(split_large_clusters([cluster], h, G))
            else:
                split_clusters.append(cluster)
        final_clusters = split_clusters

        # Fase de combinaciones
        final_clusters = combine_small_clusters(final_clusters, l_values, h_values, G, q_list)

        # Sincronización
        while len(final_clusters) < len(h_values):
            final_clusters.append(set())
        while len(l_values) < len(final_clusters):
            l_values.append(min(l_values))
        while len(h_values) < len(final_clusters):
            h_values.append(max(h_values))

        # Asignar nodos no agrupados
        all_clustered_nodes = set.union(*[set(cluster) for cluster in final_clusters])
        unclustered_nodes = set(G.nodes) - all_clustered_nodes
        if unclustered_nodes:
            final_clusters = assign_unclustered_nodes(G, final_clusters, min(l_values), max(h_values), q_list, blocked_clusters)

        # Validar fin del refinamiento
        if all(l <= len(cluster) <= h for cluster, l, h in zip(final_clusters, l_values, h_values)):
            print("Restricciones cumplidas en todos los clusters.")
            break

    # Fase final de ajuste del número de clusters
    print("Ajustando el número de clusters...")

    # Combinar clusters para reducir el número de clusters
    while len(final_clusters) > num_clusters:
        clusters = sorted(final_clusters, key=len)
        cluster1 = clusters.pop(0)  # Cluster más pequeño
        cluster2 = clusters.pop(0)  # Siguiente más pequeño
        combined_cluster = cluster1.union(cluster2)  # Combinamos
        final_clusters = clusters + [combined_cluster]  # Actualizamos lista

    # Dividir clusters para aumentar el número de clusters
    while len(final_clusters) < num_clusters:
        largest_cluster = max(final_clusters, key=len)
        final_clusters.remove(largest_cluster)
        midpoint = len(largest_cluster) // 2
        split1 = set(list(largest_cluster)[:midpoint])
        split2 = set(list(largest_cluster)[midpoint:])
        final_clusters.extend([split1, split2])  # Añadimos los nuevos clusters

    # Convertir clusters a subgrafos
    final_clusters = [G.subgraph(cluster) for cluster in final_clusters if len(cluster) > 0]


    return final_clusters