# -*- coding: utf-8 -*-
"""
Este módulo implementa detección de comunidades con restricciones de tamaño.
"""

import array
import numbers
import warnings
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

__author__ = """Kevin Castillo (kev.gcastillo@outlook.com), Thomas Aynaud (thomas.aynaud@lip6.fr)"""
#    Copyright (C) 2024 by
#    Kevin Castillo <kev.gcastillo@outlook.com>
#    All rights reserved.
#    BSD license.

__PASS_MAX = -1
__MIN = 0.0000001

# Definir las restricciones de tamaño para las comunidades
MIN_SIZE = 7   # Tamaño mínimo permitido para una comunidad
MAX_SIZE = 15  # Tamaño máximo permitido para una comunidad

class Status(object):
    """
    Para manejar varios datos en una estructura.

    Podría ser reemplazado por named tuple, pero no queremos depender de python 2.6
    """
    node2com = {}
    total_weight = 0
    internals = {}
    degrees = {}
    gdegrees = {}
    sizes = {}

    def __init__(self):
        self.node2com = dict([])
        self.total_weight = 0
        self.degrees = dict([])
        self.gdegrees = dict([])
        self.internals = dict([])
        self.loops = dict([])
        self.sizes = dict([])  # Nuevo atributo para rastrear el tamaño de las comunidades

    def __str__(self):
        return ("node2com : " + str(self.node2com) + " degrees : "
                + str(self.degrees) + " internals : " + str(self.internals)
                + " total_weight : " + str(self.total_weight))

    def copy(self):
        """Realizar una copia profunda de status"""
        new_status = Status()
        new_status.node2com = self.node2com.copy()
        new_status.internals = self.internals.copy()
        new_status.degrees = self.degrees.copy()
        new_status.gdegrees = self.gdegrees.copy()
        new_status.total_weight = self.total_weight
        new_status.sizes = self.sizes.copy()  # Copiar el tamaño de las comunidades
        return new_status

    def init(self, graph, weight, part=None):
        """Inicializar el estado de un grafo con cada nodo en una comunidad"""
        count = 0
        self.node2com = dict([])
        self.total_weight = 0
        self.degrees = dict([])
        self.gdegrees = dict([])
        self.internals = dict([])
        self.total_weight = graph.size(weight=weight)
        if part is None:
            for node in graph.nodes():
                self.node2com[node] = count
                deg = float(graph.degree(node, weight=weight))
                if deg < 0:
                    error = "Bad node degree ({})".format(deg)
                    raise ValueError(error)
                self.degrees[count] = deg
                self.gdegrees[node] = deg
                edge_data = graph.get_edge_data(node, node, default={weight: 0})
                self.loops[node] = float(edge_data.get(weight, 0))
                self.internals[count] = self.loops[node]
                self.sizes[count] = 1  # Inicializar tamaño de la comunidad
                count += 1
        else:
            for node in graph.nodes():
                com = part[node]
                self.node2com[node] = com
                deg = float(graph.degree(node, weight=weight))
                self.degrees[com] = self.degrees.get(com, 0) + deg
                self.gdegrees[node] = deg
                inc = 0.
                for neighbor, datas in graph[node].items():
                    edge_weight = datas.get(weight, 1)
                    if edge_weight <= 0:
                        error = "Bad graph type ({})".format(type(graph))
                        raise ValueError(error)
                    if part[neighbor] == com:
                        if neighbor == node:
                            inc += float(edge_weight)
                        else:
                            inc += float(edge_weight) / 2.
                self.internals[com] = self.internals.get(com, 0) + inc
                self.sizes[com] = self.sizes.get(com, 0) + 1  # Actualizar tamaño

def check_random_state(seed):
    """Convierte seed en una instancia de np.random.RandomState."""
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError("%r cannot be used to seed a numpy.random.RandomState"
                     " instance" % seed)

def partition_at_level(dendrogram, level):
    """Devuelve la partición de los nodos en el nivel dado del dendrograma."""
    partition = dendrogram[0].copy()
    for index in range(1, level + 1):
        for node, community in partition.items():
            partition[node] = dendrogram[index][community]
    return partition

def modularity(partition, graph, weight='weight'):
    """Calcula la modularidad de una partición de un grafo."""
    if graph.is_directed():
        raise TypeError("Bad graph type, use only non directed graph")

    inc = dict([])
    deg = dict([])
    links = graph.size(weight=weight)
    if links == 0:
        raise ValueError("A graph without link has an undefined modularity")

    for node in graph:
        com = partition[node]
        deg[com] = deg.get(com, 0.) + graph.degree(node, weight=weight)
        for neighbor, datas in graph[node].items():
            edge_weight = datas.get(weight, 1)
            if partition[neighbor] == com:
                if neighbor == node:
                    inc[com] = inc.get(com, 0.) + float(edge_weight)
                else:
                    inc[com] = inc.get(com, 0.) + float(edge_weight) / 2.

    res = 0.
    for com in set(partition.values()):
        res += (inc.get(com, 0.) / links) - \
               (deg.get(com, 0.) / (2. * links)) ** 2
    return res

def best_partition(graph,
                   partition=None,
                   weight='weight',
                   resolution=1.,
                   randomize=None,
                   random_state=None):
    """Calcula la partición de los nodos del grafo que maximiza la modularidad."""
    dendo = generate_dendrogram(graph,
                                partition,
                                weight,
                                resolution,
                                randomize,
                                random_state)
    return partition_at_level(dendo, len(dendo) - 1)

def generate_dendrogram(graph,
                        part_init=None,
                        weight='weight',
                        resolution=1.,
                        randomize=None,
                        random_state=None):
    """Encuentra comunidades en el grafo y devuelve el dendrograma asociado."""
    if graph.is_directed():
        raise TypeError("Bad graph type, use only non directed graph")

    if randomize is not None:
        warnings.warn("The `randomize` parameter will be deprecated in future "
                      "versions. Use `random_state` instead.", DeprecationWarning)
        if randomize is False:
            random_state = 0

    if randomize and random_state is not None:
        raise ValueError("`randomize` and `random_state` cannot be used at the "
                         "same time")

    random_state = check_random_state(random_state)

    if graph.number_of_edges() == 0:
        part = dict([])
        for i, node in enumerate(graph.nodes()):
            part[node] = i
        return [part]

    current_graph = graph.copy()
    status = Status()
    status.init(current_graph, weight, part_init)
    status_list = list()
    __one_level(current_graph, status, weight, resolution, random_state)
    new_mod = __modularity(status, resolution)
    partition = __renumber(status.node2com)
    status_list.append(partition)
    mod = new_mod
    current_graph = induced_graph(partition, current_graph, weight)
    status.init(current_graph, weight)

    while True:
        __one_level(current_graph, status, weight, resolution, random_state)
        new_mod = __modularity(status, resolution)
        if new_mod - mod < __MIN:
            break
        partition = __renumber(status.node2com)
        status_list.append(partition)
        mod = new_mod
        current_graph = induced_graph(partition, current_graph, weight)
        status.init(current_graph, weight)
    return status_list[:]

def induced_graph(partition, graph, weight="weight"):
    """Produce el grafo donde los nodos son las comunidades."""
    ret = nx.Graph()
    ret.add_nodes_from(set(partition.values()))

    for node1, node2, datas in graph.edges(data=True):
        edge_weight = datas.get(weight, 1)
        com1 = partition[node1]
        com2 = partition[node2]
        if ret.has_edge(com1, com2):
            ret[com1][com2][weight] += edge_weight
        else:
            ret.add_edge(com1, com2, **{weight: edge_weight})

    return ret

def __renumber(dictionary):
    """Renumerar los valores del diccionario de 0 a n."""
    values = set(dictionary.values())
    target = set(range(len(values)))

    if values == target:
        ret = dictionary.copy()
    else:
        renumbering = dict(zip(target.intersection(values),
                               target.intersection(values)))
        renumbering.update(dict(zip(values.difference(target),
                                    target.difference(values))))
        ret = {k: renumbering[v] for k, v in dictionary.items()}

    return ret

def load_binary(data):
    """Carga un grafo binario como se usa en la implementación cpp de este algoritmo."""
    data = open(data, "rb")

    reader = array.array("I")
    reader.fromfile(data, 1)
    num_nodes = reader.pop()
    reader = array.array("I")
    reader.fromfile(data, num_nodes)
    cum_deg = reader.tolist()
    num_links = reader.pop()
    reader = array.array("I")
    reader.fromfile(data, num_links)
    links = reader.tolist()
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    prec_deg = 0

    for index in range(num_nodes):
        last_deg = cum_deg[index]
        neighbors = links[prec_deg:last_deg]
        graph.add_edges_from([(index, int(neigh)) for neigh in neighbors])
        prec_deg = last_deg

    return graph

def __one_level(graph, status, weight_key, resolution, random_state):
    """Computa un nivel de comunidades."""
    modified = True
    nb_pass_done = 0
    cur_mod = __modularity(status, resolution)
    new_mod = cur_mod

    while modified and nb_pass_done != __PASS_MAX:
        cur_mod = new_mod
        modified = False
        nb_pass_done += 1

        for node in __randomize(graph.nodes(), random_state):
            com_node = status.node2com[node]
            degc_totw = status.gdegrees.get(node, 0.) / (status.total_weight * 2.)
            neigh_communities = __neighcom(node, graph, status, weight_key)

            # Costo de remover el nodo de su comunidad actual
            remove_cost = - neigh_communities.get(com_node, 0) + \
                resolution * (status.degrees.get(com_node, 0.) - status.gdegrees.get(node, 0.)) * degc_totw

            __remove(node, com_node, neigh_communities.get(com_node, 0.), status)

            best_com = com_node
            best_increase = 0

            # Lista de comunidades vecinas válidas que respetan las restricciones de tamaño
            for com, dnc in neigh_communities.items():
                if com == com_node:
                    continue
                new_size = status.sizes.get(com, 0) + 1
                if new_size > MAX_SIZE:
                    continue
                # Verificar si la comunidad original después de remover el nodo sigue siendo válida
                size_com_node = status.sizes.get(com_node, 0)
                if size_com_node < MIN_SIZE and size_com_node > 0:
                    continue

                incr = remove_cost + dnc - \
                       resolution * status.degrees.get(com, 0.) * degc_totw
                if incr > best_increase:
                    best_increase = incr
                    best_com = com

            # Si no hay mejoras, insertar de vuelta en la comunidad original
            if best_com == com_node:
                __insert(node, com_node, neigh_communities.get(com_node, 0.), status)
            else:
                __insert(node, best_com, neigh_communities.get(best_com, 0.), status)
                modified = True

        new_mod = __modularity(status, resolution)
        if new_mod - cur_mod < __MIN:
            break

def __neighcom(node, graph, status, weight_key):
    """
    Computa las comunidades en el vecindario de un nodo.
    """
    weights = {}
    for neighbor, datas in graph[node].items():
        if neighbor != node:
            edge_weight = datas.get(weight_key, 1)
            neighborcom = status.node2com[neighbor]
            weights[neighborcom] = weights.get(neighborcom, 0) + edge_weight

    return weights

def __remove(node, com, weight, status):
    """Remueve un nodo de una comunidad y modifica el estado."""
    status.degrees[com] = (status.degrees.get(com, 0.) -
                           status.gdegrees.get(node, 0.))
    status.internals[com] = float(status.internals.get(com, 0.) -
                                  weight - status.loops.get(node, 0.))
    status.sizes[com] -= 1  # Actualizar tamaño de la comunidad
    status.node2com[node] = -1

def __insert(node, com, weight, status):
    """Inserta un nodo en una comunidad y modifica el estado."""
    status.node2com[node] = com
    status.degrees[com] = (status.degrees.get(com, 0.) +
                           status.gdegrees.get(node, 0.))
    status.internals[com] = float(status.internals.get(com, 0.) +
                                  weight + status.loops.get(node, 0.))
    status.sizes[com] = status.sizes.get(com, 0) + 1  # Actualizar tamaño de la comunidad

def __modularity(status, resolution):
    """
    Computa rápidamente la modularidad de la partición del grafo usando
    el estado precomputado.
    """
    links = float(status.total_weight)
    result = 0.
    for community in set(status.node2com.values()):
        if community == -1:
            continue
        in_degree = status.internals.get(community, 0.)
        degree = status.degrees.get(community, 0.)
        if links > 0:
            result += in_degree * resolution / links - ((degree / (2. * links)) ** 2)
    return result

def __randomize(items, random_state):
    """Devuelve una lista con una permutación aleatoria de items."""
    randomized_items = list(items)
    random_state.shuffle(randomized_items)
    return randomized_items

# Crear el grafo del club de karate de Zachary
G = nx.karate_club_graph()

# Aplicar el algoritmo de Louvain con restricciones de tamaño
partition = best_partition(G)

# Dibujar el grafo con las comunidades detectadas
pos = nx.spring_layout(G)
cmap = plt.get_cmap('viridis')
nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=300, cmap=cmap, node_color=list(partition.values()))
nx.draw_networkx_edges(G, pos, alpha=0.5)
nx.draw_networkx_labels(G, pos)
plt.show()
