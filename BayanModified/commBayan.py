import networkx as nx
import pulp
import matplotlib.pyplot as plt

def bayan_algorithm(G, max_community_size=None, min_community_size=None):
    """
    Implementación simplificada del Algoritmo Bayan utilizando programación lineal entera.

    Parámetros:
    - G: Grafo de NetworkX.
    - max_community_size: Tamaño máximo permitido para una comunidad.
    - min_community_size: Tamaño mínimo permitido para una comunidad.

    Retorna:
    - Un diccionario que mapea nodos a comunidades.
    """

    # Número de nodos
    n = G.number_of_nodes()
    nodes = list(G.nodes())

    # Calcular la matriz de modularidad B
    m = G.size(weight='weight')
    degrees = dict(G.degree(weight='weight'))
    B = {}
    for i in nodes:
        for j in nodes:
            w = G.get_edge_data(i, j, default={'weight': 0})['weight']
            B[i, j] = w - degrees[i] * degrees[j] / (2 * m)

    # Crear el problema de optimización
    prob = pulp.LpProblem("Maximización de Modularidad", pulp.LpMaximize)

    # Variables de decisión: x_i_j = 1 si los nodos i y j están en la misma comunidad
    x = pulp.LpVariable.dicts("x", [(i, j) for i in nodes for j in nodes if i < j], cat='Binary')

    # Función objetivo: maximizar sumatoria de B_ij * x_ij
    prob += pulp.lpSum([B[i, j] * x[i, j] for i in nodes for j in nodes if i < j]), "Función Objetivo"

    # Restricciones:

    # Restricciones de transitividad: x_ij + x_jk - x_ik <= 1
    for i in nodes:
        for j in nodes:
            for k in nodes:
                if i < j and j < k and i < k:
                    prob += x[i, j] + x[j, k] - x[i, k] <= 1, f"Trans1_{i}_{j}_{k}"
                    prob += x[i, k] + x[j, k] - x[i, j] <= 1, f"Trans2_{i}_{j}_{k}"
                    prob += x[i, j] + x[i, k] - x[j, k] <= 1, f"Trans3_{i}_{j}_{k}"

    # Restricciones de tamaño (si se proporcionan)
    if max_community_size is not None or min_community_size is not None:
        # Esta implementación simplificada no incluye restricciones de tamaño debido a la complejidad añadida.
        # Para una implementación completa, se requieren variables y restricciones adicionales.
        pass

    # Resolver el problema
    solver = pulp.PULP_CBC_CMD(msg=False)
    prob.solve(solver)

    # Construir las comunidades a partir de la solución
    communities = {}
    for i in nodes:
        communities[i] = i  # Inicializar cada nodo en su propia comunidad

    # Unir nodos en comunidades basadas en las variables x_i_j
    for i in nodes:
        for j in nodes:
            if i < j:
                var_value = pulp.value(x[i, j])
                if var_value == 1:
                    # Unir las comunidades de los nodos i y j
                    com_i = communities[i]
                    com_j = communities[j]
                    # Actualizar etiquetas de comunidad
                    for node in communities:
                        if communities[node] == com_j:
                            communities[node] = com_i

    # Renumerar comunidades
    community_ids = {}
    new_id = 0
    for node in communities:
        com = communities[node]
        if com not in community_ids:
            community_ids[com] = new_id
            new_id += 1
        communities[node] = community_ids[com]

    return communities

# Ejemplo de uso con el grafo del club de karate de Zachary
G = nx.karate_club_graph()

# Ejecutar el algoritmo Bayan simplificado
partition = bayan_algorithm(G)

# Visualizar las comunidades
pos = nx.spring_layout(G)
cmap = plt.get_cmap('viridis')
nx.draw_networkx_nodes(G, pos, node_size=300, cmap=cmap, node_color=list(partition.values()))
nx.draw_networkx_edges(G, pos, alpha=0.5)
nx.draw_networkx_labels(G, pos)
plt.show()
