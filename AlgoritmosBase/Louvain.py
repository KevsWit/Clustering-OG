import networkx as nx
import community.community_louvain as community_louvain
import matplotlib.pyplot as plt

# Crear el grafo del club de karate de Zachary
G = nx.karate_club_graph()

# Aplicar el algoritmo de Louvain
partition = community_louvain.best_partition(G)

# Dibujar el grafo con las comunidades detectadas
pos = nx.spring_layout(G)
cmap = plt.get_cmap('viridis')
nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=300, cmap=cmap, node_color=list(partition.values()))
nx.draw_networkx_edges(G, pos, alpha=0.5)
nx.draw_networkx_labels(G, pos)
plt.show()