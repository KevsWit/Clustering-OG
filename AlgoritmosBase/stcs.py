import networkx as nx

# Función de puntuación para evaluar candidatos
def score(G, node, community):
    # Número de vecinos comunes con la comunidad
    return len(set(G.neighbors(node)).intersection(community.nodes()))

# Algoritmo mejorado STCS para encontrar una comunidad
def improved_stcs(G, q, l, h, visited_nodes):
    community = nx.Graph()
    community.add_node(q)
    candidates = set(G.neighbors(q)) - visited_nodes
    visited = set()
    
    while len(community.nodes()) < h and candidates:
        # Seleccionar el mejor candidato basado en la función de puntuación
        candidate = max(candidates, key=lambda node: score(G, node, community))
        community.add_node(candidate)
        community.add_edges_from(
            (candidate, neighbor) for neighbor in G.neighbors(candidate) if neighbor in community.nodes()
        )
        visited.add(candidate)
        candidates.remove(candidate)
        # Actualizar candidatos con los vecinos del candidato recién agregado
        new_neighbors = set(G.neighbors(candidate)) - community.nodes() - visited - visited_nodes
        candidates.update(new_neighbors)
    
    return community

# Función para encontrar todas las comunidades en el grafo
def find_all_communities(G, l, h):
    visited_nodes = set()
    communities = []
    for q in G.nodes():
        if q not in visited_nodes:
            community = improved_stcs(G, q, l, h, visited_nodes)
            if len(community.nodes()) >= l:
                communities.append(community)
                visited_nodes.update(community.nodes())
            else:
                # Si la comunidad es demasiado pequeña, no se añade y el nodo queda sin asignar
                pass
    # Segunda pasada para asignar nodos no asignados
    unassigned_nodes = set(G.nodes()) - visited_nodes
    for node in unassigned_nodes:
        # Intentar asignar el nodo a la comunidad más cercana
        neighbor_communities = []
        for idx, community in enumerate(communities):
            if node in G:
                neighbors = set(G.neighbors(node))
                if neighbors.intersection(community.nodes()):
                    neighbor_communities.append((idx, len(neighbors.intersection(community.nodes()))))
        if neighbor_communities:
            # Asignar el nodo a la comunidad con la que tiene más conexiones
            best_community_idx = max(neighbor_communities, key=lambda x: x[1])[0]
            best_community = communities[best_community_idx]
            best_community.add_node(node)
            best_community.add_edges_from(
                (node, neighbor) for neighbor in G.neighbors(node) if neighbor in best_community.nodes()
            )
            visited_nodes.add(node)
        else:
            # Si el nodo no está conectado a ninguna comunidad, crear una nueva comunidad
            new_community = nx.Graph()
            new_community.add_node(node)
            visited_nodes.add(node)
            communities.append(new_community)
    return communities

# Ejemplo de uso
G = nx.karate_club_graph()
l = 3  # Tamaño mínimo de la comunidad
h = 7 # Tamaño máximo de la comunidad

communities = find_all_communities(G, l, h)
for idx, community in enumerate(communities):
    print(f"Comunidad {idx+1}:")
    print(f"  Nodos: {sorted(community.nodes())}")
    print(f"  Número de aristas: {community.number_of_edges()}")
