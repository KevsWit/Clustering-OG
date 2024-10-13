import networkx as nx

# Function to calculate the size of each class based on 'gt' attribute
def calcular_tamano_clases(G):
    clases = {}
    for node, data in G.nodes(data=True):
        gt = data.get('gt')
        if gt in clases:
            clases[gt] += 1
        else:
            clases[gt] = 1
    return clases

# Load the GML files
dolphins_gml_path = 'test\\dolphins.gml'
football_gml_path = 'test\\football.gml'
karate_gml_path = 'test\\karate.gml'
polbooks_gml_path = 'test\\polbooks.gml'

# Load the graphs
G_dolphins = nx.read_gml(dolphins_gml_path)
G_football = nx.read_gml(football_gml_path)
G_karate = nx.read_gml(karate_gml_path)
G_polbooks = nx.read_gml(polbooks_gml_path)

# Calculate the size of each class for all graphs
tamanos_dolphins = calcular_tamano_clases(G_dolphins)
tamanos_football = calcular_tamano_clases(G_football)
tamanos_karate = calcular_tamano_clases(G_karate)
tamanos_polbooks = calcular_tamano_clases(G_polbooks)

print("tam karate:\n",tamanos_karate, 
"\ntam dolphin:\n",tamanos_dolphins, 
"\ntam polbooks:\n",tamanos_polbooks,
"\ntam football:\n",tamanos_football)