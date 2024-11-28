# Y. Wang, Y. Ma, H. Huang, B. Wang, and D. P. Acharjya, “A
# split–merge clustering algorithm based on the k-nearest neighbor graph,”
# Information Systems, vol. 111, p. 102124, 2023. [Online]. Available:
# https://www.sciencedirect.com/science/article/pii/S0306437922001028

# Code source: https://github.com/yian158/clustering-analysis-SMKNN

import scipy.spatial as spt
from scipy.spatial import distance
import networkx as nx
import numpy as np
import pymetis

def build_KNN_graph(data, N, k):
    points = [i for i in range(N)]
    edges = []

    tree = spt.cKDTree(data)
    kneighbor_indexs = []
    gneighbor_indexs = []
    distance_withneighbor = []
    k_dists = []

    for i in range(N):
        point = data[i]
        distances, indexs = tree.query(point, k*3 if k*3 < N else N)
        k_dists.append(distances[k])
        distance_withneighbor.append([d for d in distances[1:] if d <= distances[k]])
        kneighbor_indexs.append(indexs[1:len(distance_withneighbor[i])+1])
        gneighbor_indexs.append(kneighbor_indexs[i].tolist())
        edges.extend([(str(i), str(kneighbor_indexs[i][j]), distance_withneighbor[i][j]) for j in range(len(kneighbor_indexs[i]))])

    for p in range(N):
        for q in gneighbor_indexs[p]:
            if p not in gneighbor_indexs[q]:
                gneighbor_indexs[q].append(p)

    KNN_graph = nx.Graph()
    for node in points:
        KNN_graph.add_node(str(node))
    for edge in edges:
        KNN_graph.add_edge(edge[0], edge[1], weight=edge[2])

    return KNN_graph, gneighbor_indexs, distance_withneighbor, k_dists


def split_KNN_graph(KNN_graph, povit_points):
    KNN_graph_splitted = nx.Graph()
    for node in KNN_graph.nodes:
        if int(node) not in povit_points:
            KNN_graph_splitted.add_node(node)
    for edge in KNN_graph.edges:
        if int(edge[0]) not in povit_points and int(edge[1]) not in povit_points:
            KNN_graph_splitted.add_edge(edge[0], edge[1])

    return KNN_graph_splitted


def fun_gaussian(x, si):
    return np.exp((-1)*(x**2)/(2*(si**2)))


def fun_simi(c1, c2, graph, povit_points, neighbor_indexs, si):
    res = -1
    op = []
    p_c1 = []
    p_c2 = []
    ee_c1 = []
    ee_c2 = []
    for p in povit_points:
        nei_list = np.array(neighbor_indexs)[c1]
        p1s = [c1[index] for (index, item) in enumerate(nei_list) if p in item]

        nei_list = np.array(neighbor_indexs)[c2]
        p2s = [c2[index] for (index, item) in enumerate(nei_list) if p in item]

        if len(p1s) and len(p2s):
            op.append(p)

            p_c1.extend(p1s)
            p1_ws = [graph[str(p)][str(p1)] for p1 in p1s]
            ee_c1.extend([w['weight'] for w in p1_ws])

            p_c2.extend(p2s)
            p2_ws = [graph[str(p)][str(p2)] for p2 in p2s]
            ee_c2.extend([w['weight'] for w in p2_ws])

    if len(op) > 0:
        p_c1_set = list(set(p_c1))
        p_c2_set = list(set(p_c2))
        ie_c1 = []
        ie_c2 = []

        for p1 in set(c1).difference(p_c1_set):
            nei_list = np.array(neighbor_indexs)[p_c1_set]
            p3s = [p_c1_set[index] for (index, item) in enumerate(nei_list) if p1 in item]
            p3_ws = [graph[str(p1)][str(p3)] for p3 in p3s]
            ie_c1.extend([w['weight'] for w in p3_ws])

        for p2 in set(c2).difference(p_c2_set):
            nei_list = np.array(neighbor_indexs)[p_c2_set]
            p4s = [p_c2_set[index] for (index, item) in enumerate(nei_list) if p2 in item]
            p4_ws = [graph[str(p2)][str(p4)] for p4 in p4s]
            ie_c2.extend([w['weight'] for w in p4_ws])

        if len(ie_c1) or len(ie_c2):
            factor1 = fun_gaussian((np.mean(ee_c1) - np.mean(ie_c1)), si) if len(ie_c1) else np.inf
            factor2 = fun_gaussian((np.mean(ee_c2) - np.mean(ie_c2)), si) if len(ie_c2) else np.inf
            res = min(factor1, factor2)
        else:
            res = 0

    return res


def sec_screen(maxsi_index, res_cluster, data):
    n = len(maxsi_index[0])
    dm = np.full((n, 3), np.inf)
    for i in range(n):
        c1, c2 = maxsi_index[0][i], maxsi_index[1][i]
        diss = distance.cdist(np.array(data)[res_cluster[c1]], np.array(data)[res_cluster[c2]], metric='euclidean')
        dm[i, 0] = np.mean(diss)
        dm[i, 1] = np.amin(diss)
        dm[i, 2] = np.amax(diss)

    mindis = np.amin(dm[:, 1])
    mindis_index = np.where(dm[:, 1] <= mindis)

    if len(mindis_index[0]) <= 1:
        return maxsi_index[0][mindis_index[0][0]], maxsi_index[1][mindis_index[0][0]]
    else:
        mindis = np.amin(dm[:, 0])
        mindis_index = np.where(dm[:, 0] <= mindis)

        if len(mindis_index[0]) <= 1:
            return maxsi_index[0][mindis_index[0][0]], maxsi_index[1][mindis_index[0][0]]
        else:
            mindis = np.amin(dm[:, 2])
            mindis_index = np.where(dm[:, 2] <= mindis)
            return maxsi_index[0][mindis_index[0][0]], maxsi_index[1][mindis_index[0][0]]


def merge(graph, subgraph, K, si, povit_points, neighbor_indexs, data):
    res_cluster = []
    for item in subgraph:
        if len(item) > 1:
            res_cluster.append([int(p) for p in item])
        else:
            povit_points.append(item[0])

    if len(res_cluster) < K:
        return cut(res_cluster, K, neighbor_indexs, data)

    n = len(res_cluster)
    dm = np.ones((n, n)) * (-2)
    for i in range(n):
        for j in range(i + 1, n):
            dm[i][j] = fun_simi(res_cluster[i], res_cluster[j], graph, povit_points, neighbor_indexs, si)

    while len(res_cluster) > K:
        maxsi = np.amax(dm)
        maxsi_index = np.where(dm >= maxsi)
        rc_i, rc_j = (maxsi_index[0][0], maxsi_index[1][0]) if len(maxsi_index[0]) <= 1 else sec_screen(maxsi_index, res_cluster, data)
        res_cluster[rc_i].extend(res_cluster.pop(rc_j))
        print('.', end='')

        if len(res_cluster) <= K:
            break
        if rc_i >= rc_j:
            print('Error: rc_i >= rc_j')

        dm = np.delete(dm, rc_j, axis=0)
        dm = np.delete(dm, rc_j, axis=1)
        for i in range(len(res_cluster)):
            if i < rc_i:
                dm[i][rc_i] = fun_simi(res_cluster[i], res_cluster[rc_i], graph, povit_points, neighbor_indexs, si)
            elif i > rc_i:
                dm[rc_i][i] = fun_simi(res_cluster[i], res_cluster[rc_i], graph, povit_points, neighbor_indexs, si)


    return res_cluster


def fun_cutstd(subcluster, neighbor_indexs):
    if len(subcluster) <= 2:
        return np.inf, []

    adjacency_list = []
    for p in subcluster:
        adjacency_list.append([subcluster.index(str(nei)) for nei in neighbor_indexs[int(p)] if str(nei) in subcluster])
    edgecuts, parts = pymetis.part_graph(2, adjacency_list)

    return edgecuts, parts


def cut(graph, K, neighbor_indexs, data):
    res_cluster = [list(item) for item in graph]
    edgecuts = []
    partcuts = []
    for subcluster in res_cluster:
        edgecut, partcut = fun_cutstd(subcluster, neighbor_indexs)
        edgecuts.append(edgecut)
        partcuts.append(partcut)

    while len(res_cluster) < K:
        minec = np.amin(edgecuts)
        minec_index = np.where(edgecuts <= minec)

        if len(minec_index[0]) == 1:
            c_index = minec_index[0][0]
        else:
            ens = [len(subc) for index, subc in enumerate(res_cluster) if index in minec_index[0]]
            minen = np.amin(ens)
            minen_index = np.where(ens <= minen)
            c_index = minec_index[0][minen_index[0][0]]

        parts = partcuts[c_index]
        c = res_cluster[c_index]

        c1_index = np.where(np.array(parts) <= 0)[0]
        c1 = list(np.array(c)[c1_index])
        c2_index = np.where(np.array(parts) >= 1)[0]
        c2 = list(np.array(c)[c2_index])
        res_cluster[c_index] = c1
        res_cluster.append(c2)

        if len(res_cluster) >= K:
            break

        edgecut1, partcut1 = fun_cutstd(c1, neighbor_indexs)
        edgecut2, partcut2 = fun_cutstd(c2, neighbor_indexs)
        edgecuts[c_index] = edgecut1
        edgecuts.append(edgecut2)
        partcuts[c_index] = partcut1
        partcuts.append(partcut2)

    return [list(map(int, item)) for item in res_cluster]


def assign_povit_points(res_subgraph, povit_points, data):
    K = len(res_subgraph)
    while len(povit_points) > 0:
        dm = np.full((len(povit_points), K), np.inf)
        for i, op in enumerate(povit_points):
            dm[i, :] = [np.amin(distance.cdist([np.array(data)[op]], np.array(data)[gra], metric='euclidean')) for gra in res_subgraph]

        mindis = np.amin(dm)
        mindis_index = np.where(dm <= mindis)

        ops = []
        for i in range(len(mindis_index[0])):
            opi = mindis_index[0][i]
            l = mindis_index[1][i]
            if povit_points[opi] not in ops:
                res_subgraph[l].append(povit_points[opi])
                ops.append(povit_points[opi])
        for op in ops:
            povit_points.remove(op)
        if len(povit_points)%50 == 0:
            print('.', end='')

    return res_subgraph


def cluster(data, K, ka=10, la=1, si=(1/np.power(2, 0.5))):
    n_rows, n_cols = data.shape
    if K > n_rows:
        raise TypeError('Warning: K should not be larger than N.')

    k = ka
    if k > n_rows:
        print('Warning: N>k: k will be reset as N.')
        k = n_rows

    print('Clustering.', end='')
    points = [i for i in range(n_rows)]
    KNN_graph, neighbor_indexs, distance_withneighbor, k_dists = build_KNN_graph(data, n_rows, k)
    Dx = [(np.mean([k_dists[p] for p in neighbor_indexs[i]]) / k_dists[i]) for i in points]
    povit_points = [i for i in points if Dx[i] < la]
    KNN_graph_splitted = split_KNN_graph(KNN_graph, povit_points)
    subgraph = list(nx.connected_components(KNN_graph_splitted))

    if len(subgraph) == K:
        res_subgraph = [list(map(int, item)) for item in subgraph]
    else:
        res_subgraph = merge(KNN_graph, subgraph, K, si, povit_points, neighbor_indexs, data) if len(subgraph) > K else cut(subgraph, K, neighbor_indexs, data)
    res_cluster = assign_povit_points(res_subgraph, povit_points, data)
    cluster_label = np.zeros(n_rows)
    for index, clu in enumerate(res_cluster):
        cluster_label[clu] = (index+1)
    print('\nDone')

    return res_cluster, cluster_label