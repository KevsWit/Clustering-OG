import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities

# Create Karate Club graph
G = nx.karate_club_graph()

# Apply the fast greedy modularity algorithm
communities = greedy_modularity_communities(G)

# Print the clusters (communities)
for i, community in enumerate(communities):
    print(f"Community {i + 1}: {sorted(community)}")