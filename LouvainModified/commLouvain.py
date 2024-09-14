# -*- coding: utf-8 -*-
"""
This module implements community detection.
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

class Status(object):
    """
    To handle several data in one struct.
    """
    node2com = {}
    total_weight = 0
    internals = {}
    degrees = {}
    gdegrees = {}

    def __init__(self):
        self.node2com = dict([])
        self.total_weight = 0
        self.degrees = dict([])
        self.gdegrees = dict([])
        self.internals = dict([])
        self.loops = dict([])

    def __str__(self):
        return ("node2com : " + str(self.node2com) + " degrees : "
                + str(self.degrees) + " internals : " + str(self.internals)
                + " total_weight : " + str(self.total_weight))

    def copy(self):
        """Perform a deep copy of status"""
        new_status = Status()
        new_status.node2com = self.node2com.copy()
        new_status.internals = self.internals.copy()
        new_status.degrees = self.degrees.copy()
        new_status.gdegrees = self.gdegrees.copy()
        new_status.total_weight = self.total_weight
        return new_status

    def init(self, graph, weight, part=None):
        """Initialize the status of a graph with every node in one community"""
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
                self.loops[node] = float(edge_data.get(weight, 1))
                self.internals[count] = self.loops[node]
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

def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance."""
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError("%r cannot be used to seed a numpy.random.RandomState"
                     " instance" % seed)

def partition_at_level(dendrogram, level):
    """Return the partition of the nodes at the given level"""
    partition = dendrogram[0].copy()
    for index in range(1, level + 1):
        for node, community in partition.items():
            partition[node] = dendrogram[index][community]
    return partition

def modularity(partition, graph, weight='weight'):
    """Compute the modularity of a partition of a graph"""
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
                   resolution=1.0,
                   randomize=None,
                   random_state=None,
                   min_cluster_size=3,
                   max_cluster_size=8):
    """Compute the partition of the graph nodes which maximises the modularity
    with size limits."""
    dendo = generate_dendrogram(graph,
                                partition,
                                weight,
                                resolution,
                                randomize,
                                random_state,
                                min_cluster_size,
                                max_cluster_size)
    return partition_at_level(dendo, len(dendo) - 1)

def generate_dendrogram(graph,
                        part_init=None,
                        weight='weight',
                        resolution=1.0,
                        randomize=None,
                        random_state=None,
                        min_cluster_size=3,
                        max_cluster_size=8):
    """Generate a dendrogram with cluster size limits"""
    if graph.is_directed():
        raise TypeError("Bad graph type, use only non directed graph")

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
    __one_level(current_graph, status, weight, resolution, random_state, min_cluster_size, max_cluster_size)
    new_mod = __modularity(status, resolution)
    partition = __renumber(status.node2com)
    status_list.append(partition)
    mod = new_mod
    current_graph = induced_graph(partition, current_graph, weight)
    status.init(current_graph, weight)

    while True:
        __one_level(current_graph, status, weight, resolution, random_state, min_cluster_size, max_cluster_size)
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
    """Produce the graph where nodes are the communities."""
    ret = nx.Graph()
    ret.add_nodes_from(partition.values())

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
    """Renumber the values of the dictionary from 0 to n."""
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

def __one_level(graph, status, weight_key, resolution, random_state, min_cluster_size, max_cluster_size):
    """Compute one level of communities with size limits."""
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
            degc_totw = status.gdegrees.get(node, 0.0) / (status.total_weight * 2.0)
            neigh_communities = __neighcom(node, graph, status, weight_key)
            remove_cost = -neigh_communities.get(com_node, 0.0) + resolution * (status.degrees.get(com_node, 0.0) - status.gdegrees.get(node, 0.0)) * degc_totw

            best_com = com_node
            best_increase = 0.0
            best_found = False  # Para indicar si se encontró una mejor comunidad

            # Evaluar si mover el nodo es beneficioso
            for com, dnc in __randomize(neigh_communities.items(), random_state):
                if com != com_node:  # No considerar la misma comunidad
                    incr = remove_cost + dnc - resolution * status.degrees.get(com, 0.0) * degc_totw
                    if incr > best_increase and __insert(node, com, dnc, status, max_cluster_size):
                        best_increase = incr
                        best_com = com
                        best_found = True  # Se encontró una comunidad mejor
                        break

            # Si se encontró una mejor comunidad, mover el nodo
            if best_found:
                modified = True
                __remove(node, com_node, neigh_communities.get(com_node, 0.0), status, min_cluster_size)
            else:
                # Mantener el nodo en su comunidad original si no se encontró mejor
                __insert(node, com_node, neigh_communities.get(com_node, 0.0), status, max_cluster_size)

        new_mod = __modularity(status, resolution)
        if new_mod - cur_mod < __MIN:
            break



def __insert(node, com, weight, status, max_cluster_size):
    """Insert node into community with size limits and modify status."""
    current_size = sum(1 for n in status.node2com.values() if n == com)

    if current_size < max_cluster_size:
        status.node2com[node] = com
        status.degrees[com] = status.degrees.get(com, 0.0) + status.gdegrees.get(node, 0.0)
        status.internals[com] = status.internals.get(com, 0.0) + weight + status.loops.get(node, 0.0)
        return True
    return False

def __remove(node, com, weight, status, min_cluster_size):
    """Remove node from community with size limits and modify status."""
    current_size = sum(1 for n in status.node2com.values() if n == com)
    if current_size - 1 >= min_cluster_size:
        status.degrees[com] = (status.degrees.get(com, 0.0) - status.gdegrees.get(node, 0.0))
        status.internals[com] = float(status.internals.get(com, 0.0) - weight - status.loops.get(node, 0.0))
        status.node2com[node] = -1  # Mark the node as removed
        return True
    return False

def __neighcom(node, graph, status, weight_key):
    """
    Compute the communities in the neighborhood of node in the graph given
    with the decomposition node2com.
    """
    weights = {}
    for neighbor, datas in graph[node].items():
        if neighbor != node:
            edge_weight = datas.get(weight_key, 1)
            neighborcom = status.node2com[neighbor]
            weights[neighborcom] = weights.get(neighborcom, 0) + edge_weight

    return weights

def __modularity(status, resolution):
    """
    Fast compute the modularity of the partition of the graph using
    status precomputed.
    """
    links = float(status.total_weight)
    result = 0.
    for community in set(status.node2com.values()):
        in_degree = status.internals.get(community, 0.)
        degree = status.degrees.get(community, 0.)
        if links > 0:
            result += in_degree * resolution / links -  ((degree / (2. * links)) ** 2)
    return result

def __randomize(items, random_state):
    """Returns a List containing a random permutation of items."""
    randomized_items = list(items)
    random_state.shuffle(randomized_items)
    return randomized_items

######### Código de prueba

# Crear el grafo del club de karate de Zachary
G = nx.les_miserables_graph()

# Aplicar el algoritmo de Louvain modificado con límites de tamaño de clúster
min_cluster_size = 5
max_cluster_size = 10
partition = best_partition(G, min_cluster_size=min_cluster_size, max_cluster_size=max_cluster_size)

# Dibujar el grafo con las comunidades detectadas
pos = nx.spring_layout(G)
cmap = plt.get_cmap('viridis')
nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=300, cmap=cmap, node_color=list(partition.values()))
nx.draw_networkx_edges(G, pos, alpha=0.5)
nx.draw_networkx_labels(G, pos)
plt.show()
