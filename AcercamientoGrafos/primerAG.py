#pip install networkx matplotlib nltk
import networkx as nx
import matplotlib.pyplot as plt

from nltk.stem import SnowballStemmer
from collections import defaultdict
# Crear un grafo vacío
G = nx.Graph()

# Añadir nodos al grafo
palabras = ['casa', 'auto', 'árbol', 'libro', 'ciudad']
G.add_nodes_from(palabras)

# Añadir aristas entre algunos nodos
G.add_edge('casa', 'ciudad')
G.add_edge('auto', 'ciudad')
G.add_edge('libro', 'árbol')

# Dibujar el grafo
nx.draw(G, with_labels=True)
plt.show()

# Crear un stemmer en español
stemmer = SnowballStemmer('spanish')

# Lista de palabras
palabras = ['correr', 'corriendo', 'corredor', 'corral', 'corrección', 'corredores']

# Diccionario para agrupar las palabras por su raíz
grupos = defaultdict(list)

# Agrupar las palabras por su raíz
for palabra in palabras:
    raiz = stemmer.stem(palabra)
    grupos[raiz].append(palabra)

# Imprimir los grupos de palabras
for raiz, palabras in grupos.items():
    print(f"Raíz: {raiz}, Palabras: {palabras}")