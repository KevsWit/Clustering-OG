import networkx as nx
import community.community_louvain as community_louvain
import matplotlib.pyplot as plt

# Función para aplicar Louvain y graficar
def aplicar_louvain_y_graficar(G, title):
    partition = community_louvain.best_partition(G)
    pos = nx.spring_layout(G)
    cmap = plt.get_cmap('viridis')
    nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=300, cmap=cmap, node_color=list(partition.values()))
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos)
    plt.title(title)
    plt.show()



# Grafo del club de karate de Zachary
G_karate = nx.read_gml('test\\karate.gml')
aplicar_louvain_y_graficar(G_karate, "Karate Club")

# Grafo de Dolphins
G_dolphins = nx.read_gml('test\\dolphins.gml')
aplicar_louvain_y_graficar(G_dolphins, "Dolphins")

# Grafo de Political Books
G_pol_books = nx.read_gml('test\\polbooks.gml')
aplicar_louvain_y_graficar(G_pol_books, "Political Books")

# Grafo de Football
G_football = nx.read_gml('test\\football.gml')
aplicar_louvain_y_graficar(G_football, "Football")
