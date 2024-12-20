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
import time
import tracemalloc


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


# Calcular la variación de los tamaños solicitados
def calculate_variation(h_values, cluster_counts):
    """
    Calcula la variación total, promedio y porcentual entre los tamaños solicitados y los obtenidos.
    """
    requested_sizes = h_values.copy()
    obtained_sizes = list(cluster_counts.values())

    # Ordenar listas para emparejar las variaciones más pequeñas
    requested_sizes.sort()
    obtained_sizes.sort()

    variations = [abs(r - o) for r, o in zip(requested_sizes, obtained_sizes)]

    # Calcular métricas de variación
    total_variation = sum(variations)
    average_variation = total_variation / len(h_values)
    percentage_variation = (total_variation / sum(h_values)) * 100

    return total_variation, average_variation, percentage_variation

def gclus_variaciones(k, G, delta=0.1, repetitions=10):
    """
    Calcula el promedio de variaciones en tamaño de los clusters generados por multi_cluster_GCLUS.

    Parámetros:
        k (int): Número de clusters deseados.
        G (networkx.Graph): Grafo sobre el cual realizar el clustering.
        delta (float): Factor para calcular los tamaños mínimos de los clusters.
        repetitions (int): Número de veces que se repetirá el algoritmo.

    Retorna:
        float: Promedio de las variaciones promedio obtenidas en las repeticiones.
    """
    total_average_variation = 0

    for _ in range(repetitions):
        h_values = auto_h_values(k, G)
        clusters = multi_cluster_GCLUS(G, h_values, delta)

        # Contar nodos en cada cluster generado
        cluster_counts = {i + 1: len(cluster.nodes) for i, cluster in enumerate(clusters)}

        # Calcular variaciones
        _, avg_variation, _ = calculate_variation(h_values, cluster_counts)
        total_average_variation += avg_variation

    # Promedio de variaciones promedio
    return total_average_variation / repetitions


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


def analisis_variaciones(k_list, G, repetitions=10, delta=0.1):
    """
    Analiza las variaciones promedio obtenidas por GCLUS para diferentes valores de k en un grafo dado.

    Parámetros:
        k_list (list): Lista de valores de k (número de clusters) a probar.
        G (networkx.Graph): Grafo sobre el cual se realizarán las pruebas.
        repetitions (int): Número de repeticiones para GCLUS.
        delta (float): Parámetro delta para GCLUS.

    Retorna:
        None: Muestra un gráfico comparativo de las variaciones promedio.
    """
    gclus_avg_variations = []  # Lista para almacenar las variaciones promedio por cada k

    for k in k_list:
        # Calcular variación promedio usando gclus_variaciones
        avg_variation = gclus_variaciones(k=k, G=G, repetitions=repetitions, delta=delta)
        gclus_avg_variations.append(avg_variation)

        print(f"k={k}: GCLUS Variación Promedio={avg_variation:.4f}")

    # Graficar resultados
    plt.figure(figsize=(10, 6))
    plt.plot(k_list, gclus_avg_variations, marker='o', linestyle='-', color='tab:blue', label='GCLUS Variación Promedio')
    plt.title("Variación Promedio de GCLUS para Diferentes Valores de k")
    plt.xlabel("Número de Clusters (k)")
    plt.ylabel("Variación Promedio")
    plt.xticks(k_list)
    plt.grid(True)
    plt.legend()
    plt.show()


def analisis_tiempo(k_list, G, repetitions=10, delta=0.1):
    """
    Compara los tiempos de ejecución de SMKNN y GCLUS para diferentes valores de k en un grafo dado.

    Parámetros:
        k_list (list): Lista de valores de k (número de clusters) a probar.
        G (networkx.Graph): Grafo sobre el cual se realizarán las pruebas.
        repetitions (int): Número de repeticiones para GCLUS.
        delta (float): Parámetro delta para GCLUS.

    Retorna:
        dict: Un diccionario con los tiempos promedio de GClus y los tiempos únicos de SMKNN para cada valor de k.
    """
    smknn_times = []
    gclus_avg_times = []

    # Crear datos de entrada para SMKNN
    adj_matrix = nx.adjacency_matrix(G).toarray()
    scaler = StandardScaler()
    scaled_adj_matrix = scaler.fit_transform(adj_matrix)
    pca = PCA(n_components=2)
    data = pca.fit_transform(scaled_adj_matrix)

    for k in k_list:
        # Medir tiempo para SMKNN
        start_time_smknn = time.time()
        clusters, labels = cluster(data, k)
        smknn_time = (time.time() - start_time_smknn)*1000
        smknn_times.append(smknn_time)

        # Medir tiempos para GClus
        gclus_times = []
        for _ in range(repetitions):
            start_time_gclus = time.time()
            h_values = auto_h_values(k, G)
            multi_cluster_GCLUS(G, h_values, delta)
            gclus_times.append((time.time() - start_time_gclus)*1000)

        # Calcular promedio de tiempos para GClus
        avg_gclus_time = sum(gclus_times) / repetitions
        gclus_avg_times.append(avg_gclus_time)

        print(f"k={k}: SMKNN={smknn_time:.4f}s, GCLUS Promedio={avg_gclus_time:.4f}s")


def analisis_memoria(k_list, G, repetitions=10, delta=0.1):
    """
    Compara la memoria utilizada por SMKNN y GCLUS para diferentes valores de k en un grafo dado.

    Parámetros:
        k_list (list): Lista de valores de k (número de clusters) a probar.
        G (networkx.Graph): Grafo sobre el cual se realizarán las pruebas.
        repetitions (int): Número de repeticiones para GCLUS.
        delta (float): Parámetro delta para GCLUS.

    Retorna:
        dict: Un diccionario con las memorias promedio de GClus y las memorias únicas de SMKNN para cada valor de k.
    """
    smknn_memorias = []
    gclus_avg_memorias = []

    # Crear datos de entrada para SMKNN
    adj_matrix = nx.adjacency_matrix(G).toarray()
    scaler = StandardScaler()
    scaled_adj_matrix = scaler.fit_transform(adj_matrix)
    pca = PCA(n_components=2)
    data = pca.fit_transform(scaled_adj_matrix)

    for k in k_list:
        # Medir memoria para SMKNN
        tracemalloc.start()
        clusters, labels = cluster(data, k)
        current, peak_smknn = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        smknn_memorias.append(peak_smknn / 1024)  # Convertir a KB

        # Medir memoria para GClus
        gclus_memorias = []
        for _ in range(repetitions):
            tracemalloc.start()
            h_values = auto_h_values(k, G)
            multi_cluster_GCLUS(G, h_values, delta)
            current, peak_gclus = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            gclus_memorias.append(peak_gclus / 1024)  # Convertir a KB

        # Calcular promedio de memoria para GClus
        avg_gclus_memoria = sum(gclus_memorias) / repetitions
        gclus_avg_memorias.append(avg_gclus_memoria)

        print(f"k={k}: SMKNN Memoria={smknn_memorias[-1]:.4f}KB, GCLUS Promedio Memoria={avg_gclus_memoria:.4f}KB")


# Define the base path to the test files
base_path = os.path.join('test')

# ### Aplicacion

#*********************************** Gráficas

# # ########################### SMKNN
# # # Load the Karate Club graph
# karate_path = os.path.join(base_path, 'karate.gml')
# G = nx.read_gml(karate_path)

# # Crear datos de entrada para el algoritmo SMKNN (usar embeddings o características de nodos)
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

# # Convertir el diccionario a una lista de conjuntos para calcular la modularidad y visualización
# clusters_smk_list = list(clusters_smk.values())

# # Visualizar los resultados en el grafo interactivo con Pyvis
# visualize_clusters(G, clusters_smk_list, output_path="clusters_smknn_graph.html")

# # ########################### GCLUS

# # Configuración
# h_values = [16,18]
# delta = 0.1

# # Ejecutar la función
# clusters = multi_cluster_GCLUS(G, h_values, delta)

# # Contar nodos en cada cluster predicho
# cluster_counts = {i + 1: len(cluster.nodes) for i, cluster in enumerate(clusters)}
# print("\nCluster sizes:")
# for cluster_id, count in cluster_counts.items():
#     print(f"Cluster {cluster_id}: {count} nodes")

# visualize_clusters(G, clusters, output_path="clusters_gclus_graph.html")

#*********************************** AMI

# ########################### Karate

# # ########################### SMKNN
# # Load the Karate Club graph
# karate_path = os.path.join(base_path, 'karate.gml')
# G = nx.read_gml(karate_path)

# # Extract the ground truth labels from the 'gt' field in the GML file
# ground_truth_labels = [G.nodes[node]['gt'] for node in G.nodes()]

# # Crear datos de entrada para el algoritmo SMKNN (usar embeddings o características de nodos)
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

# # Convertir el diccionario a una lista de conjuntos para calcular la modularidad y visualización
# clusters_smk_list = list(clusters_smk.values())

# # Convertir las etiquetas de SMKNN a cadenas para mantener consistencia con GCLUS
# labels_str = [str(int(label)) for label in labels]  # Convertimos a entero y luego a cadena

# # Calcular el AMI para SMKNN
# ami_smknn = adjusted_mutual_info_score(ground_truth_labels, labels_str)
# print(f"AMI SMKNN karate: {ami_smknn}")


# # ########################### GCLUS

# # Configuración
# h_values = [16, 18]
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

# # Contar nodos en cada cluster predicho
# cluster_counts = {i + 1: len(cluster.nodes) for i, cluster in enumerate(clusters)}
# print("\nCluster sizes:")
# for cluster_id, count in cluster_counts.items():
#     print(f"Cluster {cluster_id}: {count} nodes")

# # Compute AMI between ground truth and predicted clusters
# ami_gclus = adjusted_mutual_info_score(ground_truth_labels, predicted_labels)
# print(f"AMI GCLUS karate: {ami_gclus}")

# ########################### Dolphins

# # ########################### SMKNN
# # Load the Karate Club graph
# dolphins_path = os.path.join(base_path, 'dolphins.gml')
# G = nx.read_gml(dolphins_path)

# # Extract the ground truth labels from the 'gt' field in the GML file
# ground_truth_labels = [G.nodes[node]['gt'] for node in G.nodes()]

# # Crear datos de entrada para el algoritmo SMKNN (usar embeddings o características de nodos)
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

# # Convertir el diccionario a una lista de conjuntos para calcular la modularidad y visualización
# clusters_smk_list = list(clusters_smk.values())

# # Convertir las etiquetas de SMKNN a cadenas para mantener consistencia con GCLUS
# labels_str = [str(int(label)) for label in labels]  # Convertimos a entero y luego a cadena

# # Calcular el AMI para SMKNN
# ami_smknn = adjusted_mutual_info_score(ground_truth_labels, labels_str)
# print(f"AMI SMKNN dolphins: {ami_smknn}")


# # ########################### GCLUS

# # Configuración
# h_values = [42, 20]
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

# # Contar nodos en cada cluster predicho
# cluster_counts = {i + 1: len(cluster.nodes) for i, cluster in enumerate(clusters)}
# print("\nCluster sizes:")
# for cluster_id, count in cluster_counts.items():
#     print(f"Cluster {cluster_id}: {count} nodes")

# # Compute AMI between ground truth and predicted clusters
# ami_gclus = adjusted_mutual_info_score(ground_truth_labels, predicted_labels)
# print(f"AMI GCLUS dolphins: {ami_gclus}")

# ########################### Polbooks

# # ########################### SMKNN
# # Load the Karate Club graph
# polbooks_path = os.path.join(base_path, 'polbooks.gml')
# G = nx.read_gml(polbooks_path)

# # Extract the ground truth labels from the 'gt' field in the GML file
# label_map = {'n': 0, 'c': 1, 'l': 2}
# ground_truth_labels = [label_map[G.nodes[node]['gt']] for node in G.nodes()]

# # Crear datos de entrada para el algoritmo SMKNN (usar embeddings o características de nodos)
# adj_matrix = nx.adjacency_matrix(G).toarray()
# scaler = StandardScaler()
# scaled_adj_matrix = scaler.fit_transform(adj_matrix)
# pca = PCA(n_components=2)
# data = pca.fit_transform(scaled_adj_matrix)

# # Número de clusters deseados
# K = 3

# # Ejecutar el algoritmo SMKNN
# clusters, labels = cluster(data, K)

# # Crear el diccionario de clusters con las etiquetas únicas generadas por SMKNN
# unique_labels = set(map(int, labels))  # Convertir las etiquetas a enteros y eliminar duplicados
# clusters_smk = {label: set() for label in unique_labels}  # Crear un diccionario con esas claves

# # Asignar nodos a los clusters
# for node, label in zip(G.nodes(), labels):
#     clusters_smk[int(label)].add(node)  # Convertir etiqueta a entero y agregar nodo

# # Convertir el diccionario a una lista de conjuntos para calcular la modularidad y visualización
# clusters_smk_list = list(clusters_smk.values())

# # Convertir las etiquetas de SMKNN a cadenas para mantener consistencia con GCLUS
# labels_str = [int(label) for label in labels]  # Convertimos a entero y luego a cadena

# # Calcular el AMI para SMKNN
# ami_smknn = adjusted_mutual_info_score(ground_truth_labels, labels_str)
# print(f"AMI SMKNN polbooks: {ami_smknn}")


# # ########################### GCLUS

# # Configuración
# h_values = [13,49,43]
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
# predicted_labels = [node_to_cluster[node] for node in G.nodes()]

# # Contar nodos en cada cluster predicho
# cluster_counts = {i + 1: len(cluster.nodes) for i, cluster in enumerate(clusters)}
# print("\nCluster sizes:")
# for cluster_id, count in cluster_counts.items():
#     print(f"Cluster {cluster_id}: {count} nodes")

# # Compute AMI between ground truth and predicted clusters
# ami_gclus = adjusted_mutual_info_score(ground_truth_labels, predicted_labels)
# print(f"AMI GCLUS polbooks: {ami_gclus}")

# #*********************************** Modularidad

# ########################### karate

# print("########################### karate")

# # Load the Karate Club graph
# karate_path = os.path.join(base_path, 'karate.gml')
# G = nx.read_gml(karate_path)

# # Ejecutar la comparación de modularidad para k = [2, 3, 4, 5]
# analisis_modularidad(k_list=[2, 3, 4, 5], G=G, repetitions=10)

# # Ejecutar la función de análisis
# analisis_variaciones(k_list=[2, 3, 4, 5], G=G, repetitions=10)

# # Ejecutar la función de tiempo
# analisis_tiempo(k_list=[2, 3, 4, 5], G=G, repetitions=10)

# # Ejecutar la función de memoria
# analisis_memoria(k_list=[2, 3, 4, 5], G=G, repetitions=10)

# ########################### dolphins

# print("########################### dolphins")

# # Load the Dolphins graph
# dolphins_path = os.path.join(base_path, 'dolphins.gml')
# G = nx.read_gml(dolphins_path)

# # Ejecutar la comparación de modularidad para k = [2, 3, 4, 5]
# analisis_modularidad(k_list=[2, 3, 4, 5], G=G, repetitions=10)

# # Ejecutar la función de análisis
# analisis_variaciones(k_list=[2, 3, 4, 5], G=G, repetitions=10)

# # Ejecutar la función de tiempo
# analisis_tiempo(k_list=[2, 3, 4, 5], G=G, repetitions=10)

# # Ejecutar la función de memoria
# analisis_memoria(k_list=[2, 3, 4, 5], G=G, repetitions=10)

# ########################### pol_books

# print("########################### pol_books")

# # Load the Political Books graph
# polbooks_path = os.path.join(base_path, 'polbooks.gml')
# G = nx.read_gml(polbooks_path)

# # Ejecutar la comparación de modularidad para k = [2, 3, 4, 5]
# analisis_modularidad(k_list=[2, 3, 4, 5], G=G, repetitions=10)

# # Ejecutar la función de análisis
# analisis_variaciones(k_list=[2, 3, 4, 5], G=G, repetitions=10)

# # Ejecutar la función de tiempo
# analisis_tiempo(k_list=[2, 3, 4, 5], G=G, repetitions=10)

# # Ejecutar la función de memoria
# analisis_memoria(k_list=[2, 3, 4, 5], G=G, repetitions=10)

# ########################### les miserables

# print("########################### les miserables")

# # Load the miserables graph
# G = nx.les_miserables_graph()

# # Ejecutar la comparación de modularidad para k = [2, 3, 4, 5]
# analisis_modularidad(k_list=[2, 3, 4, 5], G=G, repetitions=10)

# # Ejecutar la función de análisis
# analisis_variaciones(k_list=[2, 3, 4, 5], G=G, repetitions=10)

# # Ejecutar la función de tiempo
# analisis_tiempo(k_list=[2, 3, 4, 5], G=G, repetitions=10)

# # Ejecutar la función de memoria
# analisis_memoria(k_list=[2, 3, 4, 5], G=G, repetitions=10)

# # ******************************* graficas tiempo

# # Datos
# ids = ['1', '2', '3', '4']
# k_values = ['K=2', 'K=3', 'K=4', 'K=5']
# gclus_times = [
#     [39.991, 27.829, 32.025, 43.739],  # ID 1
#     [6.994, 7.779, 10.805, 9.992],     # ID 2
#     [135.562, 163.696, 143.115, 158.495],  # ID 3
#     [98.527, 62.108, 92.865, 90.471]   # ID 4
# ]
# smknn_times = [
#     [63.141, 27.325, 28.900, 37.742],  # ID 1
#     [6.022, 10.194, 7.979, 8.741],     # ID 2
#     [67.239, 208.246, 165.142, 218.056],  # ID 3
#     [43.167, 32.752, 39.904, 38.170]   # ID 4
# ]

# x = np.arange(len(ids))  # posiciones de los IDs
# width = 0.2  # ancho de las barras

# # Gráfico para GCLUS
# fig, ax = plt.subplots(figsize=(10, 6))
# for i, k in enumerate(k_values):
#     ax.bar(x - width/2 + i*width, [row[i] for row in gclus_times], width, label=f'GCLUS {k}')

# ax.set_xlabel("Dataset ID")
# ax.set_ylabel("Tiempo (ms)")
# ax.set_title("Comparación de Tiempos de Ejecución para GCLUS")
# ax.set_xticks(x)
# ax.set_xticklabels(ids)
# ax.legend()
# ax.grid(axis='y', linestyle='--', alpha=0.7)

# plt.tight_layout()
# plt.show()

# # Gráfico para SMKNN
# fig, ax = plt.subplots(figsize=(10, 6))
# for i, k in enumerate(k_values):
#     ax.bar(x - width/2 + i*width, [row[i] for row in smknn_times], width, label=f'SMKNN {k}')

# ax.set_xlabel("Dataset ID")
# ax.set_ylabel("Tiempo (ms)")
# ax.set_title("Comparación de Tiempos de Ejecución para SMKNN")
# ax.set_xticks(x)
# ax.set_xticklabels(ids)
# ax.legend()
# ax.grid(axis='y', linestyle='--', alpha=0.7)

# plt.tight_layout()
# plt.show()

# # ******************************* graficas memoria

# Datos
ids = ['1', '2', '3', '4']
k_values = ['K=2', 'K=3', 'K=4', 'K=5']
gclus_memory = [
    [172.308, 211.352, 214.187, 242.231],  # ID 1
    [131.561, 130.053, 135.156, 143.241],  # ID 2
    [564.083, 509.677, 712.744, 848.615],  # ID 3
    [586.421, 556.065, 671.332, 724.177]   # ID 4
]
smknn_memory = [
    [337.096, 306.775, 306.447, 306.565],  # ID 1
    [196.910, 176.434, 176.106, 176.106],  # ID 2
    [552.450, 513.028, 512.700, 512.700],  # ID 3
    [442.134, 395.618, 395.407, 395.407]   # ID 4
]

x = np.arange(len(ids))  # posiciones de los IDs
width = 0.2  # ancho de las barras

# Gráfico para GCLUS
fig, ax = plt.subplots(figsize=(10, 6))
for i, k in enumerate(k_values):
    ax.bar(x - width/2 + i*width, [row[i] for row in gclus_memory], width, label=f'GCLUS {k}')

ax.set_xlabel("Dataset ID")
ax.set_ylabel("Memoria (KB)")
ax.set_title("Comparación de Consumo de Memoria para GCLUS")
ax.set_xticks(x)
ax.set_xticklabels(ids)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# Gráfico para SMKNN
fig, ax = plt.subplots(figsize=(10, 6))
for i, k in enumerate(k_values):
    ax.bar(x - width/2 + i*width, [row[i] for row in smknn_memory], width, label=f'SMKNN {k}')

ax.set_xlabel("Dataset ID")
ax.set_ylabel("Memoria (KB)")
ax.set_title("Comparación de Consumo de Memoria para SMKNN")
ax.set_xticks(x)
ax.set_xticklabels(ids)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()