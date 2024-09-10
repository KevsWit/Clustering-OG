# Constrained Hierarchical Clustering via Graph Coarsening and Optimal Cuts

import networkx as nx
import matplotlib.pyplot as plt
import random

def coarsen_graph_with_constraints(G, constraint):
    # Coarsening step con restricciones
    nodes = list(G.nodes())
    random.shuffle(nodes)
    coarsened_graph = nx.Graph()
    for i in range(0, len(nodes), constraint):
        coarsened_graph.add_node(tuple(nodes[i:i+constraint]))
    return coarsened_graph

def optimal_cut_with_constraints(G, constraint):
    # Cortes óptimos con restricciones
    cut_value, partition = nx.minimum_cut(G, list(G.nodes())[0], list(G.nodes())[1])
    return partition

def constrained_hierarchical_clustering(G, constraint):
    clusters = [G]
    while len(clusters[-1].nodes()) > 1:
        coarsened_graph = coarsen_graph_with_constraints(clusters[-1], constraint)
        clusters.append(coarsened_graph)
    return clusters

def draw_clusters(clusters):
    for i, cluster in enumerate(clusters):
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(cluster)
        nx.draw(cluster, pos, with_labels=True, node_color=range(len(cluster.nodes())), cmap=plt.cm.rainbow)
        plt.title(f'Nivel {i} de la Jerarquía')
        plt.show()

def print_clusters(clusters):
    for i, cluster in enumerate(clusters):
        print(f'Nivel {i} de la Jerarquía:')
        for node in cluster.nodes():
            print(f'  Cluster: {node}')
        print()

# Grafo de karate
G = nx.karate_club_graph()
constraint = 2  # Ejemplo de restricción
clusters = constrained_hierarchical_clustering(G, constraint)
print("Número de niveles en la jerarquía:", len(clusters))
# draw_clusters(clusters)
print_clusters(clusters)