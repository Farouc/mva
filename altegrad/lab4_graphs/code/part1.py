"""
Graph Mining - ALTEGRAD - Nov 2024
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


############## Task 1
file_path="..\datasets\CA-HepTh.txt"

G=nx.read_edgelist(file_path,
                   comments='#',
                   delimiter="\t",
                   create_using=nx.Graph,
                   nodetype=int)

num_nodes=G.number_of_nodes()
num_edges=G.number_of_edges()
print(f"Number of nodes: {num_nodes}")
print(f"Number of edges: {num_edges}")

############## Task 2
connected_components=list(nx.connected_components(G))

num_connected_components=len(connected_components)
print(f"Number of connected components: {num_connected_components}")

if num_connected_components>1:
    print("Graph is not connected. Let's extract its biggest connected component")
    largest_conn_comp=max(connected_components,key=len)
    G_cc=G.subgraph(largest_conn_comp)

    num_nodes_cc=G_cc.number_of_nodes()
    num_edges_cc=G_cc.number_of_edges()
    frac_nodes = num_nodes_cc / num_nodes
    frac_edges = num_edges_cc / num_edges
    print(f"Fraction of nodes in G_cc : {frac_nodes:.4f} ({frac_nodes * 100:.2f} %)")
    print(f"Fraction of edges in G_cc : {frac_edges:.4f} ({frac_edges * 100:.2f} %)")
else: #connected graph
    print("Graph is connected")
    G_cc=G
    


