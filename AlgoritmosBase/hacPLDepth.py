# Hierarchical Agglomerative Graph Clustering in Poly-Logarithmic Depth
import networkx as nx
import matplotlib.pyplot as plt
import random

def coarsen_graph(G):
    # Coarsening step: agrupar nodos aleatoriamente
    nodes = list(G.nodes())
    random.shuffle(nodes)
    coarsened_graph = nx.Graph()
    for i in range(0, len(nodes), 2):
        if i + 1 < len(nodes):
            coarsened_graph.add_node((nodes[i], nodes[i+1]))
        else:
            coarsened_graph.add_node((nodes[i],))
    return coarsened_graph

def optimal_cut(G):
    # Cortes óptimos: usar un corte mínimo simple como ejemplo
    cut_value, partition = nx.minimum_cut(G, list(G.nodes())[0], list(G.nodes())[1])
    return partition

def hierarchical_clustering(G):
    clusters = [G]
    while len(clusters[-1].nodes()) > 1:
        coarsened_graph = coarsen_graph(clusters[-1])
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
clusters = hierarchical_clustering(G)
print("Número de niveles en la jerarquía:", len(clusters))
# draw_clusters(clusters)
print_clusters(clusters)