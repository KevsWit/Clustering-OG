import networkx as nx
import os
from sklearn.metrics import normalized_mutual_info_score  # NMI
from sklearn.metrics import adjusted_mutual_info_score  # AMI
from networkx.algorithms.community import modularity # Modularidad
from gclus import multi_cluster_GCLUS, visualize_clusters
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from smknn import cluster

# ### Aplicacion

# ########################### karate

# Define the base path to the test files
base_path = os.path.join('test')

# Load the Karate Club graph
karate_path = os.path.join(base_path, 'karate.gml')
G = nx.read_gml(karate_path)

########################### SMKNN

# Crear datos de entrada para el algoritmo SMKNN (usar embeddings o características de nodos)
# Aquí usamos una matriz de adyacencia como entrada y reducimos dimensionalidad con PCA
adj_matrix = nx.adjacency_matrix(G).toarray()
scaler = StandardScaler()
scaled_adj_matrix = scaler.fit_transform(adj_matrix)
pca = PCA(n_components=2)
data = pca.fit_transform(scaled_adj_matrix)

# Número de clusters deseados
K = 4

# Ejecutar el algoritmo SMKNN
clusters, labels = cluster(data, K)

# Crear el diccionario de clusters con las etiquetas únicas generadas por SMKNN
unique_labels = set(map(int, labels))  # Convertir las etiquetas a enteros y eliminar duplicados
clusters_smk = {label: set() for label in unique_labels}  # Crear un diccionario con esas claves

# Asignar nodos a los clusters
for node, label in zip(G.nodes(), labels):
    clusters_smk[int(label)].add(node)  # Convertir etiqueta a entero y agregar nodo

# Convertir el diccionario a una lista de conjuntos para calcular la modularidad
clusters_smk_list = list(clusters_smk.values())

# Calcular la modularidad
modularity_value = modularity(G, clusters_smk_list)
print(f"Modularidad de los clusters generados por SMKNN: {modularity_value:.4f}")



# Visualizar los resultados en el grafo original
pos = nx.spring_layout(G, seed=42)  # Layout para visualización
colors = [plt.cm.tab10(int(label)) for label in labels]

plt.figure(figsize=(10, 7))
nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=300, cmap='tab10')
nx.draw_networkx_edges(G, pos, alpha=0.5)
nx.draw_networkx_labels(G, pos, font_size=10)
plt.title("Clustering del grafo Karate Club con SMKNN")
plt.show()

########################### GCLUS

# Extract the ground truth labels from the 'gt' field in the GML file
ground_truth_labels = [G.nodes[node]['gt'] for node in G.nodes()]

# Configuración
# h_values = [10,5,9,10]
h_values = [7,7,10,10]
delta = 0.1

# Ejecutar la función
clusters = multi_cluster_GCLUS(G, h_values, delta)

# Assign each node to a cluster ID
node_to_cluster = {}
for i, cluster in enumerate(clusters):
    for node in cluster.nodes:
        node_to_cluster[node] = i

# Crear las comunidades como una lista de conjuntos
communities = [set(cluster.nodes) for cluster in clusters]

# Calcular la modularidad si la partición es válida
modularity_value = modularity(G, communities)
print(f"Modularidad de los clusters: {modularity_value}")

# Predicted labels based on the clusters
predicted_labels = [str(node_to_cluster[node] + 1) for node in G.nodes()]
print(predicted_labels)
print(ground_truth_labels)

# Count nodes in each predicted cluster and print the results
cluster_counts = {i + 1: len(cluster.nodes) for i, cluster in enumerate(clusters)}
print("\nCluster sizes:")
for cluster_id, count in cluster_counts.items():
    print(f"Cluster {cluster_id}: {count} nodes")

visualize_clusters(G, clusters)

########################### dolphins

# # Load the Dolphins graph
# dolphins_path = os.path.join(base_path, 'dolphins.gml')
# G = nx.read_gml(dolphins_path)

# # Extract the ground truth labels from the 'gt' field in the GML file
# ground_truth_labels = [G.nodes[node]['gt'] for node in G.nodes()]

# # Set your size constraints
# l, h = 20, 42  # Adjust your size constraints as needed
# clusters = multi_cluster_GCLUS(G, l, h)

# # Assign each node to a cluster ID
# node_to_cluster = {}
# for i, cluster in enumerate(clusters):
#     for node in cluster.nodes:
#         node_to_cluster[node] = i

# # Predicted labels based on the clusters
# predicted_labels = [str(node_to_cluster[node] + 1) for node in G.nodes()]
# print(predicted_labels)
# print(ground_truth_labels)

# # Compute AMI between ground truth and predicted clusters
# ami_dolphins = adjusted_mutual_info_score(ground_truth_labels, predicted_labels)
# print(f"AMI dolphins: {ami_dolphins}")

# # Compute NMI between ground truth and predicted clusters
# nmi_dolphins = normalized_mutual_info_score(ground_truth_labels, predicted_labels)
# print(f"NMI dolphins: {nmi_dolphins}")

# visualize_clusters(G, clusters)

# ########################### pol_books

# # Load the Political Books graph
# polbooks_path = os.path.join(base_path, 'polbooks.gml')
# G = nx.read_gml(polbooks_path)

# # Extract the ground truth labels from the 'gt' field in the GML file
# label_map = {'n': 0, 'c': 1, 'l': 2}
# ground_truth_labels = [label_map[G.nodes[node]['gt']] for node in G.nodes()]

# # Set your size constraints
# l, h = 13, 49  # Adjust your size constraints as needed
# clusters = multi_cluster_GCLUS(G, l, h)

# # Assign each node to a cluster ID
# node_to_cluster = {}
# for i, cluster in enumerate(clusters):
#     for node in cluster.nodes:
#         node_to_cluster[node] = i

# # Predicted labels based on the clusters
# predicted_labels = [node_to_cluster[node] for node in G.nodes()]
# print(predicted_labels)
# print(ground_truth_labels)

# # Compute AMI between ground truth and predicted clusters
# ami_pol_books = adjusted_mutual_info_score(ground_truth_labels, predicted_labels)
# print(f"AMI pol_books: {ami_pol_books}")

# # Compute NMI between ground truth and predicted clusters
# nmi_pol_books = normalized_mutual_info_score(ground_truth_labels, predicted_labels)
# print(f"NMI pol_books: {nmi_pol_books}")

# visualize_clusters(G, clusters)

# ########################### football

# # Load the Football graph
# football_path = os.path.join(base_path, 'football.gml')
# G = nx.read_gml(football_path)

# # Extract the ground truth labels from the 'gt' field in the GML file
# ground_truth_labels = [G.nodes[node]['gt'] for node in G.nodes()]

# # Set your size constraints
# l, h = 5, 13  # Adjust your size constraints as needed
# clusters = multi_cluster_GCLUS(G, l, h)

# # Assign each node to a cluster ID
# node_to_cluster = {}
# for i, cluster in enumerate(clusters):
#     for node in cluster.nodes:
#         node_to_cluster[node] = i

# # Predicted labels based on the clusters
# predicted_labels = [node_to_cluster[node] for node in G.nodes()]
# print(predicted_labels)
# print(ground_truth_labels)

# # Compute AMI between ground truth and predicted clusters
# ami_football = adjusted_mutual_info_score(ground_truth_labels, predicted_labels)
# print(f"AMI football: {ami_football}")

# # Compute NMI between ground truth and predicted clusters
# nmi_football = normalized_mutual_info_score(ground_truth_labels, predicted_labels)
# print(f"NMI football: {nmi_football}")

# visualize_clusters(G, clusters)


# #**************************************************************** Extras

# ########################### AS topology

# # Load the AS Internet topology graph
# as_path = os.path.join(base_path, 'as.gml')
# G = nx.read_gml(as_path)

# # Extract the ground truth labels from the 'gt' field in the GML file
# ground_truth_labels = [G.nodes[node]['gt'] for node in G.nodes()]

# # Set your size constraints
# l, h = 1, 7118  # Adjust your size constraints as needed
# clusters = multi_cluster_GCLUS(G, l, h)

# # Assign each node to a cluster ID
# node_to_cluster = {}
# for i, cluster in enumerate(clusters):
#     for node in cluster.nodes:
#         node_to_cluster[node] = i

# # Predicted labels based on the clusters
# predicted_labels = [str(node_to_cluster[node] + 1) for node in G.nodes()]
# # print(predicted_labels)
# # print(ground_truth_labels)

# # Compute AMI between ground truth and predicted clusters
# ami_as = adjusted_mutual_info_score(ground_truth_labels, predicted_labels)
# print(f"AMI AS: {ami_as}")

# # Compute NMI between ground truth and predicted clusters
# nmi_as = normalized_mutual_info_score(ground_truth_labels, predicted_labels)
# print(f"NMI AS: {nmi_as}")

# # visualize_clusters(G, clusters)


