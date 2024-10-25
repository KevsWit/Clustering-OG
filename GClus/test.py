import networkx as nx
import os
from sklearn.metrics import normalized_mutual_info_score  # NMI
from sklearn.metrics import adjusted_mutual_info_score  # AMI
from gclus import multi_cluster_STCS, visualize_clusters

# Define the base path to the test files
base_path = os.path.join('test')

### Aplicacion

########################### karate

# Load the Karate Club graph
karate_path = os.path.join(base_path, 'karate.gml')
G = nx.read_gml(karate_path)

# Extract the ground truth labels from the 'gt' field in the GML file
ground_truth_labels = [G.nodes[node]['gt'] for node in G.nodes()]

# Set your size constraints
l, h = 15, 19  # Adjust your size constraints as needed
clusters = multi_cluster_STCS(G, l, h)

# Assign each node to a cluster ID
node_to_cluster = {}
for i, cluster in enumerate(clusters):
    for node in cluster.nodes:
        node_to_cluster[node] = i

# Predicted labels based on the clusters
predicted_labels = [str(node_to_cluster[node] + 1) for node in G.nodes()]
print(predicted_labels)
print(ground_truth_labels)

# Compute AMI between ground truth and predicted clusters
ami_karate = adjusted_mutual_info_score(ground_truth_labels, predicted_labels)
print(f"AMI karate: {ami_karate}")

# Compute NMI between ground truth and predicted clusters
nmi_karate = normalized_mutual_info_score(ground_truth_labels, predicted_labels)
print(f"NMI karate: {nmi_karate}")

visualize_clusters(G, clusters)

########################### dolphins

# Load the Dolphins graph
dolphins_path = os.path.join(base_path, 'dolphins.gml')
G = nx.read_gml(dolphins_path)

# Extract the ground truth labels from the 'gt' field in the GML file
ground_truth_labels = [G.nodes[node]['gt'] for node in G.nodes()]

# Set your size constraints
l, h = 20, 42  # Adjust your size constraints as needed
clusters = multi_cluster_STCS(G, l, h)

# Assign each node to a cluster ID
node_to_cluster = {}
for i, cluster in enumerate(clusters):
    for node in cluster.nodes:
        node_to_cluster[node] = i

# Predicted labels based on the clusters
predicted_labels = [str(node_to_cluster[node] + 1) for node in G.nodes()]
print(predicted_labels)
print(ground_truth_labels)

# Compute AMI between ground truth and predicted clusters
ami_dolphins = adjusted_mutual_info_score(ground_truth_labels, predicted_labels)
print(f"AMI dolphins: {ami_dolphins}")

# Compute NMI between ground truth and predicted clusters
nmi_dolphins = normalized_mutual_info_score(ground_truth_labels, predicted_labels)
print(f"NMI dolphins: {nmi_dolphins}")

visualize_clusters(G, clusters)

########################### pol_books

# Load the Political Books graph
polbooks_path = os.path.join(base_path, 'polbooks.gml')
G = nx.read_gml(polbooks_path)

# Extract the ground truth labels from the 'gt' field in the GML file
label_map = {'n': 0, 'c': 1, 'l': 2}
ground_truth_labels = [label_map[G.nodes[node]['gt']] for node in G.nodes()]

# Set your size constraints
l, h = 13, 49  # Adjust your size constraints as needed
clusters = multi_cluster_STCS(G, l, h)

# Assign each node to a cluster ID
node_to_cluster = {}
for i, cluster in enumerate(clusters):
    for node in cluster.nodes:
        node_to_cluster[node] = i

# Predicted labels based on the clusters
predicted_labels = [node_to_cluster[node] for node in G.nodes()]
print(predicted_labels)
print(ground_truth_labels)

# Compute AMI between ground truth and predicted clusters
ami_pol_books = adjusted_mutual_info_score(ground_truth_labels, predicted_labels)
print(f"AMI pol_books: {ami_pol_books}")

# Compute NMI between ground truth and predicted clusters
nmi_pol_books = normalized_mutual_info_score(ground_truth_labels, predicted_labels)
print(f"NMI pol_books: {nmi_pol_books}")

visualize_clusters(G, clusters)

########################### football

# Load the Football graph
football_path = os.path.join(base_path, 'football.gml')
G = nx.read_gml(football_path)

# Extract the ground truth labels from the 'gt' field in the GML file
ground_truth_labels = [G.nodes[node]['gt'] for node in G.nodes()]

# Set your size constraints
l, h = 5, 13  # Adjust your size constraints as needed
clusters = multi_cluster_STCS(G, l, h)

# Assign each node to a cluster ID
node_to_cluster = {}
for i, cluster in enumerate(clusters):
    for node in cluster.nodes:
        node_to_cluster[node] = i

# Predicted labels based on the clusters
predicted_labels = [node_to_cluster[node] for node in G.nodes()]
print(predicted_labels)
print(ground_truth_labels)

# Compute AMI between ground truth and predicted clusters
ami_football = adjusted_mutual_info_score(ground_truth_labels, predicted_labels)
print(f"AMI football: {ami_football}")

# Compute NMI between ground truth and predicted clusters
nmi_football = normalized_mutual_info_score(ground_truth_labels, predicted_labels)
print(f"NMI football: {nmi_football}")

visualize_clusters(G, clusters)


#**************************************************************** Extras

########################### AS topology

# Load the AS Internet topology graph
as_path = os.path.join(base_path, 'as.gml')
G = nx.read_gml(as_path)

# Extract the ground truth labels from the 'gt' field in the GML file
ground_truth_labels = [G.nodes[node]['gt'] for node in G.nodes()]

# Set your size constraints
l, h = 1, 7118  # Adjust your size constraints as needed
clusters = multi_cluster_STCS(G, l, h)

# Assign each node to a cluster ID
node_to_cluster = {}
for i, cluster in enumerate(clusters):
    for node in cluster.nodes:
        node_to_cluster[node] = i

# Predicted labels based on the clusters
predicted_labels = [str(node_to_cluster[node] + 1) for node in G.nodes()]
# print(predicted_labels)
# print(ground_truth_labels)

# Compute AMI between ground truth and predicted clusters
ami_as = adjusted_mutual_info_score(ground_truth_labels, predicted_labels)
print(f"AMI AS: {ami_as}")

# Compute NMI between ground truth and predicted clusters
nmi_as = normalized_mutual_info_score(ground_truth_labels, predicted_labels)
print(f"NMI AS: {nmi_as}")

# visualize_clusters(G, clusters)


