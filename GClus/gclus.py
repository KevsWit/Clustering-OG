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


def combine_small_clusters(clusters, l, h, G, pivots):
    """Combina clusters pequeños respetando los pivotes para garantizar el número de clusters."""
    combined_clusters = []
    remaining_clusters = []
    assigned_nodes = set()  # Para rastrear nodos ya asignados

    for cluster in clusters:
        if len(cluster) < l and not any(node in pivots for node in cluster):
            remaining_clusters.append(cluster)
        else:
            if not cluster & assigned_nodes:  # Verificar que no haya nodos ya asignados
                combined_clusters.append(cluster)
                assigned_nodes.update(cluster)

    while remaining_clusters:
        cluster_to_combine = remaining_clusters.pop(0)
        best_merge = None
        best_conductance = -float('inf')

        for i, cluster in enumerate(remaining_clusters):
            if any(node in pivots for node in cluster):
                continue
            conductance_value = conductance(G, cluster_to_combine, cluster)
            if conductance_value > best_conductance:
                best_conductance = conductance_value
                best_merge = i

        if best_merge is not None:
            merged_cluster = cluster_to_combine.union(remaining_clusters.pop(best_merge))
            merged_cluster -= assigned_nodes  # Eliminar nodos ya asignados
            if l <= len(merged_cluster) <= h:
                combined_clusters.append(merged_cluster)
                assigned_nodes.update(merged_cluster)
        else:
            cluster_to_combine -= assigned_nodes  # Eliminar nodos ya asignados
            if cluster_to_combine:
                combined_clusters.append(cluster_to_combine)
                assigned_nodes.update(cluster_to_combine)

    return combined_clusters


def split_large_clusters(clusters, h, G):
    """Divide clusters grandes en función de la conectividad interna para formar subclústeres dentro de las restricciones de tamaño."""
    new_clusters = []
    assigned_nodes = set()  # Para rastrear nodos ya asignados

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

            # Eliminar nodos ya asignados
            cluster1 -= assigned_nodes
            cluster2 -= assigned_nodes

            if len(cluster1) > 0:
                new_clusters.append(cluster1)
                assigned_nodes.update(cluster1)
            if len(cluster2) > 0:
                new_clusters.append(cluster2)
                assigned_nodes.update(cluster2)
        else:
            # Verificar que no haya nodos ya asignados
            cluster -= assigned_nodes
            if cluster:
                new_clusters.append(cluster)
                assigned_nodes.update(cluster)

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
            if blocked_clusters[idx]:  # Saltar clusters bloqueados
                continue
            if len(cluster) < h:
                cond = conductance(G, cluster, {node})
                if cond < best_conductance:
                    best_conductance = cond
                    best_cluster = cluster

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
                if idx != cluster_idx and not blocked_clusters[idx]:  # No comparar con el clúster actual ni con bloqueados
                    distance_to_pivot = distances_to_pivots.get(pivot, {}).get(node, float('inf'))
                    if distance_to_pivot < best_distance:
                        best_distance = distance_to_pivot
                        best_cluster = all_clusters[idx]

            # Reasignar nodo si el mejor clúster es diferente
            if best_cluster != cluster:
                cluster.remove(node)
                best_cluster.add(node)

    return all_clusters


def adjust_clusters_to_h_values(G, clusters, h_values):
    """
    Ajusta los clusters para que se alineen con los tamaños especificados en h_values.
    Garantiza que los nodos reasignados mantengan conexiones directas al nuevo cluster.
    """
    # Asegurar que h_values coincida con la cantidad de clusters
    if len(h_values) != len(clusters):
        h_values = h_values[:len(clusters)] + [max(h_values)] * (len(clusters) - len(h_values))

    if len(h_values) == 2:
        for idx, h in enumerate(h_values):
            while len(clusters[idx]) > h:  # Si el cluster excede el tamaño deseado
                print(f"Ajustando cluster {idx} (tam: {len(clusters[idx])}) al tamaño objetivo {h}.")

                # Paso 1: Seleccionar nodo para extraer (menor conectividad interna)
                node_to_remove = min(
                    clusters[idx],
                    key=lambda n: len(set(G.neighbors(n)) & clusters[idx])
                )
                clusters[idx].remove(node_to_remove)

                # Paso 2: Primera pasada - Reasignar a clusters con conexión directa
                best_cluster = None
                best_distance = float('inf')

                for j, cluster in enumerate(clusters):
                    if j == idx or len(cluster) >= h_values[j]:  # No reasignar al mismo cluster ni a clusters llenos
                        continue
                    if any(neighbor in cluster for neighbor in G.neighbors(node_to_remove)):
                        distance = min(nx.shortest_path_length(G, node_to_remove, target) for target in cluster)
                        if distance < best_distance:
                            best_distance = distance
                            best_cluster = cluster

                # Si no se encuentra un cluster en la primera pasada, buscar por caminos mínimos
                if not best_cluster:
                    for j, cluster in enumerate(clusters):
                        if j == idx or len(cluster) >= h_values[j]:
                            continue
                        distance = min(nx.shortest_path_length(G, node_to_remove, target) for target in cluster)
                        if distance < best_distance:
                            best_distance = distance
                            best_cluster = cluster

                # Reasignar el nodo
                if best_cluster:
                    best_cluster.add(node_to_remove)
                    print(f"Nodo {node_to_remove} reasignado al cluster {clusters.index(best_cluster)}.")
                else:
                    # Si no hay un cluster adecuado, devolver al cluster original
                    print(f"No se encontró un cluster válido para el nodo {node_to_remove}. Retornando al cluster original.")
                    clusters[idx].add(node_to_remove)
                    break  # Salir para evitar bucles infinitos
    else:
        for idx, h in enumerate(h_values):
            while len(clusters[idx]) > h:  # Si el cluster excede el tamaño deseado
                print(f"Ajustando cluster {idx} (tam: {len(clusters[idx])}) al tamaño objetivo {h}.")

                # Paso 1: Seleccionar nodo para extraer (menor conectividad interna)
                node_to_remove = min(
                    clusters[idx],
                    key=lambda n: len(set(G.neighbors(n)) & clusters[idx])
                )
                clusters[idx].remove(node_to_remove)

                # Paso 2: Primera pasada - Reasignar a clusters con conexión directa
                best_cluster = None
                best_distance = float('inf')

                for j, cluster in enumerate(clusters):
                    if j == idx or len(cluster) >= h_values[j]:  # No reasignar al mismo cluster ni a clusters llenos
                        continue
                    # Verificar que haya una conexión directa al menos con un nodo en el cluster
                    if any(neighbor in cluster for neighbor in G.neighbors(node_to_remove)):
                        distance = min(nx.shortest_path_length(G, node_to_remove, target) for target in cluster)
                        if distance < best_distance:
                            best_distance = distance
                            best_cluster = cluster

                # Si no se encuentra un cluster en la primera pasada, ignorar la reasignación
                if not best_cluster:
                    print(f"No se encontró un cluster válido con conexión directa para el nodo {node_to_remove}. Retornando.")
                    clusters[idx].add(node_to_remove)  # Devolver al cluster original
                    break

                # Reasignar el nodo al cluster con conexión directa
                best_cluster.add(node_to_remove)
                print(f"Nodo {node_to_remove} reasignado al cluster {clusters.index(best_cluster)}.")
    return clusters


def multi_cluster_GCLUS(G, h_values, delta=0.2, q_list=None, max_iterations=5):
    """
    Genera múltiples clusters utilizando el algoritmo STCS y garantiza que respeten las restricciones de tamaño especificadas para cada cluster.
    Para q_list se seleccionan nodos iniciales automáticamente en función de una métrica.
    """
    # Verificación de que delta esté entre 0 y 1
    if not (0 < delta < 1):
        raise ValueError("El parámetro delta debe ser mayor a 0 y menor a 1.")
    
    # Verificación de que la suma de h_values sea igual a la cantidad de nodos en el grafo
    total_nodes = G.number_of_nodes()
    if sum(h_values) != total_nodes:
        raise ValueError("La suma de h_values debe ser igual a la cantidad de nodos en el grafo.")

    final_clusters = []
    assigned_nodes = set()
    din_G = G.copy()
    h_values = [round(h) for h in h_values]
    l_values = [int(h - (h * delta)) for h in h_values]  # Calcula l dinámicamente para cada h en h_values
    num_clusters = len(h_values)
    blocked_clusters = [False] * num_clusters  # Inicialmente, ningún cluster está bloqueado

    
    # Seleccionar pivotes si no se proporciona q_list
    q_list = select_pivots(G, num_clusters)

    # Paso 1: Generar clusters iniciales para cada nodo pivote
    for idx, q in enumerate(q_list):
        cluster_nodes = set([q])  # Incluir q en el cluster desde el inicio
        if q in din_G.nodes:  # Verificar si el nodo está en el grafo
            trussness = truss_decomposition(din_G)
            H = GC_Final(din_G, q, l_values[idx], h_values[idx], trussness)
            H_nodes_filtered = {n for n in H.nodes if n not in assigned_nodes}

            # Garantizar que el nodo q esté en su cluster
            if len(H_nodes_filtered) >= l_values[idx] or not H_nodes_filtered:
                # Añadir q solo si no está ya en H_nodes_filtered
                if q not in H_nodes_filtered:
                    H_nodes_filtered.add(q)
                assigned_nodes.update(H_nodes_filtered)
                cluster_nodes.update(H_nodes_filtered)
                din_G.remove_nodes_from(H_nodes_filtered)
            else:
                print(f"El cluster de q={q} es menor al tamaño mínimo requerido. Asignando al menos el nodo pivote.")

        # Aseguramos que el cluster tenga al menos el nodo q
        if not cluster_nodes:
            cluster_nodes.add(q)

        # Si aún no se asignaron suficientes nodos, forzar la asignación de vecinos más cercanos
        if len(cluster_nodes) < l_values[idx]:
            if q in din_G.nodes:  # Verificar nuevamente antes de acceder a los vecinos
                neighbors = set(din_G.neighbors(q)) - assigned_nodes
                needed_nodes = l_values[idx] - len(cluster_nodes)
                additional_nodes = list(neighbors)[:needed_nodes]  # Seleccionar nodos vecinos
                cluster_nodes.update(additional_nodes)
                assigned_nodes.update(additional_nodes)
                din_G.remove_nodes_from(additional_nodes)
            else:
                print(f"El nodo {q} ya no está en el grafo. No se pueden asignar más nodos vecinos.")

        # Asegurar que cada cluster tenga nodos y agregarlo a final_clusters
        final_clusters.append(cluster_nodes)

    # Validar que se crearon exactamente el número de clusters requeridos
    if len(final_clusters) < num_clusters:
        print(f"Advertencia: Solo se generaron {len(final_clusters)} clusters en el paso inicial. Creando clusters vacíos adicionales.")
        while len(final_clusters) < num_clusters:
            final_clusters.append(set())


    # Paso 2: Asignar nodos no clusterizados antes de las iteraciones
    all_clustered_nodes = set.union(*[set(cluster) for cluster in final_clusters])
    unclustered_nodes = set(G.nodes()) - all_clustered_nodes

    if unclustered_nodes:
        print(f"Asignando nodos no clusterizados iniciales: {len(unclustered_nodes)} nodos.")
        final_clusters = assign_unclustered_nodes(G, final_clusters, min(l_values), max(h_values), q_list, blocked_clusters)

    # Refinar clusters en iteraciones
    iteration = 0
    remaining_l_values = l_values.copy()  # Inicializar las restricciones dinámicas al inicio
    remaining_h_values = h_values.copy()
    

    while iteration < max_iterations:
        iteration += 1

        # Ajustar restricciones dinámicas al número actual de clusters
        remaining_l_values = remaining_l_values[:len(final_clusters)] + [min(l_values)] * (len(final_clusters) - len(remaining_l_values))
        remaining_h_values = remaining_h_values[:len(final_clusters)] + [max(h_values)] * (len(final_clusters) - len(remaining_h_values))


        refined_clusters = []
        combine_clusters = []

        # Crear listas temporales para las restricciones restantes
        temp_l_values = []
        temp_h_values = []

        # Refinar cada cluster para cumplir con los tamaños l y h específicos
        balance_needed = 0
        for idx, cluster in enumerate(final_clusters):
            print(f"Iteración {iteration}, Cluster {idx}: Tamaño actual: {len(cluster)}")
            
            # Verificar si el cluster ya cumple con el tamaño objetivo
            if remaining_l_values[idx] <= len(cluster) <= remaining_h_values[idx]:
                refined_clusters.append(cluster)
                temp_l_values.append(remaining_l_values[idx])
                temp_h_values.append(remaining_h_values[idx])
                blocked_clusters[idx] = True  # Bloquear el cluster
                print(f"Cluster {idx} bloqueado: cumple con el tamaño objetivo.")
                continue

            # Si el cluster es demasiado pequeño, intentar combinar o rellenar
            if len(cluster) < remaining_l_values[idx]:
                balance_needed = 1
                if len(refined_clusters) == 0:
                    refined_clusters.extend(
                        combine_small_clusters(
                            final_clusters, remaining_l_values[idx], remaining_h_values[idx], G, q_list
                        )
                    )
                else:
                    combine_clusters = refined_clusters.copy()
                    combine_clusters.append(cluster)
                    refined_clusters.extend(
                        combine_small_clusters(
                            combine_clusters, remaining_l_values[idx], remaining_h_values[idx], G, q_list
                        )
                    )
            # Si el cluster es demasiado grande, dividirlo
            elif len(cluster) > remaining_h_values[idx] and (len(final_clusters) < num_clusters and idx == 0):
                balance_needed = 1
                print("num ref clus: ", len(final_clusters), ", clusters: ", num_clusters)
                refined_clusters.extend(
                    split_large_clusters([cluster], remaining_h_values[idx], G)
                )
            else:
                # Cluster refinado, agregarlo a refined_clusters
                balance_needed = 1
                refined_clusters.append(cluster)
                temp_l_values.append(remaining_l_values[idx])
                temp_h_values.append(remaining_h_values[idx])

        # Ajustar blocked_clusters después de combinar o dividir clusters
        if len(refined_clusters) > len(blocked_clusters):
            blocked_clusters.extend([False] * (len(refined_clusters) - len(blocked_clusters)))
        elif len(refined_clusters) < len(blocked_clusters):
            blocked_clusters = blocked_clusters[:len(refined_clusters)]

        # Actualizar restricciones restantes
        remaining_l_values = temp_l_values
        remaining_h_values = temp_h_values

        # Llamar a assign_unclustered_nodes
        final_clusters = assign_unclustered_nodes(G, refined_clusters, min(l_values), max(h_values), q_list, blocked_clusters)
        
        # Asegurarse de tener la cantidad de clusters exacta
        if len(final_clusters) < num_clusters:
            final_clusters.extend([set()] * (num_clusters - len(final_clusters)))
        elif len(refined_clusters) > num_clusters:
            final_clusters = combine_small_clusters(final_clusters, min(l_values), max(h_values), G, q_list)

        all_clustered_nodes = set.union(*[set(cluster) for cluster in final_clusters])
        unclustered_nodes = set(G.nodes()) - all_clustered_nodes

        # Si no quedan nodos sin asignar, salir del bucle
        if not unclustered_nodes:
            break
        print(f"Iteración {iteration}: Nodos sin cluster: {len(unclustered_nodes)}.")



    # Paso 3: Asignar nodos restantes si es necesario
    if unclustered_nodes:
        print("Asignando nodos restantes a los clusters más pequeños.")
        for node in unclustered_nodes:
            smallest_cluster = min(final_clusters, key=lambda c: len(c))
            if len(smallest_cluster) < h_values[final_clusters.index(smallest_cluster)]:
                smallest_cluster.add(node)

    # Ajustar tamaños y reasignar nodos según los valores deseados en h_values
    final_clusters = adjust_clusters_to_h_values(G, final_clusters, h_values)


    # Convertimos los clusters a subgrafos y eliminamos clusters vacíos
    final_clusters = [G.subgraph(cluster) for cluster in final_clusters if len(cluster) > 0]

    # Asegurar la cantidad de clusters especificada
    while len(final_clusters) < num_clusters:
        final_clusters.append(G.subgraph(set()))

    # for idx, cluster in enumerate(final_clusters):
    #     if blocked_clusters[idx]:
    #         assert remaining_l_values[idx] <= len(cluster) <= remaining_h_values[idx], f"Cluster {idx} modificado indebidamente."


    return final_clusters