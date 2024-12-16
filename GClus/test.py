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
import random

import random

def auto_h_values(k, G):
    """
    Genera un vector de h_values aleatorio para un grafo dado y un número de clusters k.

    Parámetros:
        k (int): Número de clusters.
        G (networkx.Graph): Grafo en el que se realizará el clustering.

    Retorna:
        list: Un vector de h_values de longitud k con tamaños de cluster distribuidos aleatoriamente.
    """
    total_nodes = G.number_of_nodes()
    if k > total_nodes:
        raise ValueError("El número de clusters no puede ser mayor que el número de nodos en el grafo.")
    
    # Inicializar los valores de h_values y los nodos restantes
    remaining_nodes = total_nodes
    h_values = []

    for i in range(k):
        # Asegurarse de dejar nodos suficientes para los clusters restantes
        min_size = 1
        max_size = remaining_nodes - (k - len(h_values) - 1)
        
        # Generar tamaños aleatorios evitando siempre que sean demasiado pequeños
        cluster_size = random.randint(min_size, max_size // (k - i))
        h_values.append(cluster_size)
        remaining_nodes -= cluster_size

    # Ajustar si queda algún nodo sin asignar
    if sum(h_values) < total_nodes:
        h_values[-1] += total_nodes - sum(h_values)

    return h_values


# Calcular la desviación de los tamaños solicitados
def calculate_deviation(h_values, cluster_counts):
    """
    Calcula la desviación total, promedio y porcentual entre los tamaños solicitados y los obtenidos.
    """
    requested_sizes = h_values.copy()
    obtained_sizes = list(cluster_counts.values())

    # Ordenar listas para emparejar las desviaciones más pequeñas
    requested_sizes.sort()
    obtained_sizes.sort()

    deviations = [abs(r - o) for r, o in zip(requested_sizes, obtained_sizes)]

    # Calcular métricas de desviación
    total_deviation = sum(deviations)
    average_deviation = total_deviation / len(h_values)
    percentage_deviation = (total_deviation / sum(h_values)) * 100

    return total_deviation, average_deviation, percentage_deviation

def gclus_desviaciones(k, G, delta=0.1, repetitions=10):
    """
    Calcula el promedio de desviaciones en tamaño de los clusters generados por multi_cluster_GCLUS.

    Parámetros:
        k (int): Número de clusters deseados.
        G (networkx.Graph): Grafo sobre el cual realizar el clustering.
        delta (float): Factor para calcular los tamaños mínimos de los clusters.
        repetitions (int): Número de veces que se repetirá el algoritmo.

    Retorna:
        float: Promedio de las desviaciones promedio obtenidas en las repeticiones.
    """
    total_average_deviation = 0

    for _ in range(repetitions):
        h_values = auto_h_values(k, G)
        clusters = multi_cluster_GCLUS(G, h_values, delta)

        # Contar nodos en cada cluster generado
        cluster_counts = {i + 1: len(cluster.nodes) for i, cluster in enumerate(clusters)}

        # Calcular desviaciones
        _, avg_dev, _ = calculate_deviation(h_values, cluster_counts)
        total_average_deviation += avg_dev

    # Promedio de desviaciones promedio
    return total_average_deviation / repetitions


def gclus_modularidades(k, G, delta=0.1, repetitions=10):
    """
    Calcula el promedio de modularidad de los clusters generados por multi_cluster_GCLUS.

    Parámetros:
        k (int): Número de clusters deseados.
        G (networkx.Graph): Grafo sobre el cual realizar el clustering.
        delta (float): Factor para calcular los tamaños mínimos de los clusters.
        repetitions (int): Número de veces que se repetirá el algoritmo.

    Retorna:
        float: Promedio de las modularidades obtenidas en las repeticiones.
    """
    total_modularity = 0

    for _ in range(repetitions):
        h_values = auto_h_values(k, G)
        clusters = multi_cluster_GCLUS(G, h_values, delta)

        # Crear las comunidades como una lista de conjuntos
        communities = [set(cluster.nodes) for cluster in clusters]

        # Calcular modularidad
        modularity_value = modularity(G, communities)
        total_modularity += modularity_value

    # Promedio de modularidades
    return total_modularity / repetitions


def analisis_modularidad(k_list, G, repetitions=10, delta=0.1):
    """
    Compara las modularidades obtenidas por SMKNN y GCLUS para diferentes valores de k en un grafo dado.

    Parámetros:
        k_list (list): Lista de valores de k (número de clusters) a probar.
        G (networkx.Graph): Grafo sobre el cual se realizarán las pruebas.
        repetitions (int): Número de repeticiones para GCLUS.
        delta (float): Parámetro delta para GCLUS.

    Retorna:
        None: Muestra un gráfico comparativo de las modularidades.
    """
    smknn_modularities = []
    gclus_avg_modularities = []

    # Crear datos de entrada para SMKNN
    adj_matrix = nx.adjacency_matrix(G).toarray()
    scaler = StandardScaler()
    scaled_adj_matrix = scaler.fit_transform(adj_matrix)
    pca = PCA(n_components=2)
    data = pca.fit_transform(scaled_adj_matrix)

    for k in k_list:
        # Ejecutar SMKNN
        clusters, labels = cluster(data, k)
        unique_labels = set(map(int, labels))
        clusters_smk = {label: set() for label in unique_labels}

        for node, label in zip(G.nodes(), labels):
            clusters_smk[int(label)].add(node)
        clusters_smk_list = list(clusters_smk.values())

        # Calcular modularidad para SMKNN
        modularity_smk = modularity(G, clusters_smk_list)
        smknn_modularities.append(modularity_smk)

        # Calcular modularidad promedio para GCLUS
        avg_modularity = gclus_modularidades(k=k, G=G, repetitions=repetitions, delta=delta)
        gclus_avg_modularities.append(avg_modularity)

        print(f"k={k}: SMKNN={modularity_smk:.4f}, GCLUS Promedio={avg_modularity:.4f}")

    # Graficar resultados
    plt.figure(figsize=(10, 6))
    plt.plot(k_list, smknn_modularities, marker='o', linestyle='-', label='SMKNN')
    plt.plot(k_list, gclus_avg_modularities, marker='o', linestyle='-', label='GCLUS Promedio')
    plt.title("Comparación de Modularidad entre SMKNN y GCLUS")
    plt.xlabel("Número de Clusters (k)")
    plt.ylabel("Modularidad")
    plt.legend()
    plt.grid(True)
    plt.show()


def analisis_desviaciones(k_list, G, repetitions=10, delta=0.1):
    """
    Analiza las desviaciones promedio obtenidas por GCLUS para diferentes valores de k en un grafo dado.

    Parámetros:
        k_list (list): Lista de valores de k (número de clusters) a probar.
        G (networkx.Graph): Grafo sobre el cual se realizarán las pruebas.
        repetitions (int): Número de repeticiones para GCLUS.
        delta (float): Parámetro delta para GCLUS.

    Retorna:
        None: Muestra un gráfico comparativo de las desviaciones promedio.
    """
    gclus_avg_deviations = []  # Lista para almacenar las desviaciones promedio por cada k

    for k in k_list:
        # Calcular desviación promedio usando gclus_desviaciones
        avg_deviation = gclus_desviaciones(k=k, G=G, repetitions=repetitions, delta=delta)
        gclus_avg_deviations.append(avg_deviation)

        print(f"k={k}: GCLUS Desviación Promedio={avg_deviation:.4f}")

    # Graficar resultados
    plt.figure(figsize=(10, 6))
    plt.plot(k_list, gclus_avg_deviations, marker='o', linestyle='-', color='tab:blue', label='GCLUS Desviación Promedio')
    plt.title("Desviación Promedio de GCLUS para Diferentes Valores de k")
    plt.xlabel("Número de Clusters (k)")
    plt.ylabel("Desviación Promedio")
    plt.xticks(k_list)
    plt.grid(True)
    plt.legend()
    plt.show()


# Define the base path to the test files
base_path = os.path.join('test')

# ### Aplicacion

# ########################### SMKNN

# # Crear datos de entrada para el algoritmo SMKNN (usar embeddings o características de nodos)
# # Aquí usamos una matriz de adyacencia como entrada y reducimos dimensionalidad con PCA
# karate_path = os.path.join(base_path, 'karate.gml')
# G = nx.read_gml(karate_path)
# adj_matrix = nx.adjacency_matrix(G).toarray()
# scaler = StandardScaler()
# scaled_adj_matrix = scaler.fit_transform(adj_matrix)
# pca = PCA(n_components=2)
# data = pca.fit_transform(scaled_adj_matrix)

# # Número de clusters deseados
# K = 2

# # Ejecutar el algoritmo SMKNN
# clusters, labels = cluster(data, K)

# # Crear el diccionario de clusters con las etiquetas únicas generadas por SMKNN
# unique_labels = set(map(int, labels))  # Convertir las etiquetas a enteros y eliminar duplicados
# clusters_smk = {label: set() for label in unique_labels}  # Crear un diccionario con esas claves

# # Asignar nodos a los clusters
# for node, label in zip(G.nodes(), labels):
#     clusters_smk[int(label)].add(node)  # Convertir etiqueta a entero y agregar nodo

# # Convertir el diccionario a una lista de conjuntos para calcular la modularidad
# clusters_smk_list = list(clusters_smk.values())

# # Visualizar los resultados en el grafo original
# pos = nx.spring_layout(G, seed=42)  # Layout para visualización
# colors = [plt.cm.tab10(int(label)) for label in labels]

# plt.figure(figsize=(10, 7))
# nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=300, cmap='tab10')
# nx.draw_networkx_edges(G, pos, alpha=0.5)
# nx.draw_networkx_labels(G, pos, font_size=10)
# plt.title("Clustering del grafo dolphins con SMKNN")
# plt.show()

# ########################### karate

print("########################### karate")

# Load the Karate Club graph
karate_path = os.path.join(base_path, 'karate.gml')
G = nx.read_gml(karate_path)

# Ejecutar la comparación de modularidad para k = [2, 3, 4, 5]
analisis_modularidad(k_list=[2, 3, 4, 5], G=G, repetitions=10)

# Ejecutar la función de análisis
analisis_desviaciones(k_list=[2, 3, 4, 5], G=G, repetitions=10)


# ########################### dolphins

print("########################### dolphins")

# Load the Dolphins graph
dolphins_path = os.path.join(base_path, 'dolphins.gml')
G = nx.read_gml(dolphins_path)

# Ejecutar la comparación de modularidad para k = [2, 3, 4, 5]
analisis_modularidad(k_list=[2, 3, 4, 5], G=G, repetitions=10)

# Ejecutar la función de análisis
analisis_desviaciones(k_list=[2, 3, 4, 5], G=G, repetitions=10)


# ########################### pol_books

print("########################### pol_books")

# Load the Political Books graph
polbooks_path = os.path.join(base_path, 'polbooks.gml')
G = nx.read_gml(polbooks_path)

# Ejecutar la comparación de modularidad para k = [2, 3, 4, 5]
analisis_modularidad(k_list=[2, 3, 4, 5], G=G, repetitions=10)

# Ejecutar la función de análisis
analisis_desviaciones(k_list=[2, 3, 4, 5], G=G, repetitions=10)


########################### les miserables

print("########################### les miserables")

# Load the miserables graph
miserables_path = os.path.join(base_path, 'lesmiserables.gml')
G = nx.read_gml(miserables_path)

# Ejecutar la comparación de modularidad para k = [2, 3, 4, 5]
analisis_modularidad(k_list=[2, 3, 4, 5], G=G, repetitions=10)

# Ejecutar la función de análisis
analisis_desviaciones(k_list=[2, 3, 4, 5], G=G, repetitions=10)


# ########################### SMKNN

# # Crear datos de entrada para el algoritmo SMKNN (usar embeddings o características de nodos)
# # Aquí usamos una matriz de adyacencia como entrada y reducimos dimensionalidad con PCA
# adj_matrix = nx.adjacency_matrix(G).toarray()
# scaler = StandardScaler()
# scaled_adj_matrix = scaler.fit_transform(adj_matrix)
# pca = PCA(n_components=2)
# data = pca.fit_transform(scaled_adj_matrix)

# # Número de clusters deseados
# K = 2

# # Ejecutar el algoritmo SMKNN
# clusters, labels = cluster(data, K)

# # Crear el diccionario de clusters con las etiquetas únicas generadas por SMKNN
# unique_labels = set(map(int, labels))  # Convertir las etiquetas a enteros y eliminar duplicados
# clusters_smk = {label: set() for label in unique_labels}  # Crear un diccionario con esas claves

# # Asignar nodos a los clusters
# for node, label in zip(G.nodes(), labels):
#     clusters_smk[int(label)].add(node)  # Convertir etiqueta a entero y agregar nodo

# # Convertir el diccionario a una lista de conjuntos para calcular la modularidad
# clusters_smk_list = list(clusters_smk.values())

# # Visualizar los resultados en el grafo original
# pos = nx.spring_layout(G, seed=42)  # Layout para visualización
# colors = [plt.cm.tab10(int(label)) for label in labels]

# plt.figure(figsize=(10, 7))
# nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=300, cmap='tab10')
# nx.draw_networkx_edges(G, pos, alpha=0.5)
# nx.draw_networkx_labels(G, pos, font_size=10)
# plt.title("Clustering del grafo football con SMKNN")
# plt.show()

# print('\n######### Resultados\n')
# # Calcular la modularidad
# modularity_value = modularity(G, clusters_smk_list)
# print(f"Modularidad de los clusters generados por SMKNN: {modularity_value:.4f} \n\n")

# ########################### GCLUS

# # Extract the ground truth labels from the 'gt' field in the GML file
# ground_truth_labels = [G.nodes[node]['gt'] for node in G.nodes()]

# # Configuración
# h_values = [7,17,10]
# delta = 0.1

# # Ejecutar la función
# clusters = multi_cluster_GCLUS(G, h_values, delta)

# # Assign each node to a cluster ID
# node_to_cluster = {}
# for i, cluster in enumerate(clusters):
#     for node in cluster.nodes:
#         node_to_cluster[node] = i

# # Crear las comunidades como una lista de conjuntos
# communities = [set(cluster.nodes) for cluster in clusters]

# # Etiquetas predichas basadas en el cluster
# predicted_labels = [str(node_to_cluster[node] + 1) for node in G.nodes()]
# print(predicted_labels)
# print(ground_truth_labels)

# # Contar nodos en cada cluster predicho
# cluster_counts = {i + 1: len(cluster.nodes) for i, cluster in enumerate(clusters)}
# print("\nCluster sizes:")
# for cluster_id, count in cluster_counts.items():
#     print(f"Cluster {cluster_id}: {count} nodes")

# visualize_clusters(G, clusters)

# print('\n######### Resultados\n')

# # Calcular la modularidad si la partición es válida
# modularity_value = modularity(G, communities)
# print(f"Modularidad de los clusters: {modularity_value}")

# # Calcular y mostrar las métricas de desviación
# total_dev, avg_dev, perc_dev = calculate_deviation(h_values, cluster_counts)




########### impresion
# karate_path = os.path.join(base_path, 'karate.gml')
# G = nx.read_gml(karate_path)

# # Configuración
# h_values = [7,17,10]
# delta = 0.1

# # Ejecutar la función
# clusters = multi_cluster_GCLUS(G, h_values, delta)

# # Contar nodos en cada cluster predicho
# cluster_counts = {i + 1: len(cluster.nodes) for i, cluster in enumerate(clusters)}
# print("\nCluster sizes:")
# for cluster_id, count in cluster_counts.items():
#     print(f"Cluster {cluster_id}: {count} nodes")

# visualize_clusters(G, clusters)