"""
Graph Mining - ALTEGRAD - Nov 2024
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye
from random import randint
from sklearn.cluster import KMeans



############## Task 3
# Perform spectral clustering to partition graph G into k clusters
def spectral_clustering(G, k,random_state=0,n_init=10):
    #adjacency_matrix
    print("ectract adj matrix")
    A=nx.adjacency_matrix(G)
    m=G.number_of_nodes()
    degrees=[G.degree(node) for node in G.nodes()]
    D_inv=diags([1/d if d>0 else 0 for d in degrees])

    #Laplacian matrix
    print("computing laplacian")
    L_rw=eye(m)-D_inv@A

    # eigs returns (eigvals, eigvectors) of L_rw
    print("computing eigvectors")
    eigenvalues, eigenvectors = eigs(L_rw, k=k, which='SM')
    U=np.real(eigenvectors)

    #applying K-means to U
    print("kmeans")
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
    cluster_labels = kmeans.fit_predict(U)

    print("final part")
    #extracting clusters
    clustering={}
    nodes=list(G.nodes())
    for i in range(m):
        node = nodes[i]
        clustering[node] = int(cluster_labels[i])
    
    return clustering


############## Task 4

file_path="..\datasets\CA-HepTh.txt"
print("reading G ...")

G=nx.read_edgelist(file_path,
                   comments='#',
                   delimiter="\t",
                   create_using=nx.Graph,
                   nodetype=int)
print("starting clustering...")
clustering=spectral_clustering(G,k=50)
# print("Clustering: ",clustering)
# print(clustering.values())
print("\n Printing some sample results (10 first nodes) :")

sample_results = dict(list(clustering.items())[:10])
for node, cluster_id in sample_results.items():
    print(f"Nœud {node}: Cluster {cluster_id}")
    



############## Task 5
# Compute modularity value from graph G based on clustering
def modularity(G, clustering):
    m=G.number_of_edges()
    communities={}
    for node,c in clustering.items():
        if c not in communities:
            communities[c]=[]
        communities[c].append(node)
    Q=0
    #iterate over communities
    for c,nodes in communities.items():
        #lc: number of internal edges in community c
        lc=0
        #dc: sum of degrees of nodes in community c
        dc=0
        for u in nodes:
            dc+=G.degree(u)
            for v in G.neighbors(u):
                if v in nodes:
                    lc+=1
        lc=lc//2 #each edge is computed twice

        Q+=(lc/m) - (dc/(2*m))**2

    return Q

modularity_spectral=modularity(G,clustering)
print("Modularity of the spectral clustering: ", round(modularity_spectral,4))

############## Task 6

#Random partition into 50 clusters

nodes=list(G.nodes())
random_clustering={}
for node in nodes:
    random_clustering[node]=randint(0,49)

modularity_random=modularity(G,random_clustering)

print("Modularity of a random 50-way partition:     ", round(modularity_random, 4))

 
# For k=50, the spectral clustering result achieves a modularity of approximately 
# 0.084, whereas a purely random 50-way assignment yields a modularity close to zero (
# 0.0002).
# This confirms that spectral clustering extracts non-trivial community structure, although the
#  overall modularity remains low because the number of clusters (50) is far larger than the 
# natural community structure of the graph.





