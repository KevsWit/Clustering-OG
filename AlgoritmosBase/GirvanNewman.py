import networkx as nx
from networkx.algorithms.community import girvan_newman

# Create Karate Club graph
G = nx.karate_club_graph()

# Apply the Girvan-Newman algorithm
communities_generator = girvan_newman(G)
top_level_communities = next(communities_generator)

# Print the first set of communities
for i, community in enumerate(top_level_communities):
    print(f"Community {i + 1}: {sorted(community)}")