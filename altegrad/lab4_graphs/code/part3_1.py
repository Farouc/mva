"""
Graph Mining - ALTEGRAD - Nov 2024
"""

import networkx as nx
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx

############## Task 7


#load Mutag dataset
def load_dataset():

    dataset=TUDataset(root=".",name="MUTAG")
    #converting each PyG to NetworkX
    Gs=[]
    for data in dataset:
        G_nx=to_networkx(data)
        G=nx.Graph(G_nx)
        Gs.append(G)

    y = [data.y.item() for data in dataset]
    return Gs, y


Gs,y = load_dataset()
print("Task 7 done.")
#Gs, y = create_dataset()
G_train, G_test, y_train, y_test = train_test_split(Gs, y, test_size=0.2, random_state=42)

# Compute the shortest path kernel
def shortest_path_kernel(Gs_train, Gs_test):    
    all_paths = dict()
    sp_counts_train = dict()
    
    for i,G in enumerate(Gs_train):
        sp_lengths = dict(nx.shortest_path_length(G))
        sp_counts_train[i] = dict()
        nodes = G.nodes()
        for v1 in nodes:
            for v2 in nodes:
                if v2 in sp_lengths[v1]:
                    length = sp_lengths[v1][v2]
                    if length in sp_counts_train[i]:
                        sp_counts_train[i][length] += 1
                    else:
                        sp_counts_train[i][length] = 1

                    if length not in all_paths:
                        all_paths[length] = len(all_paths)
                        
    sp_counts_test = dict()

    for i,G in enumerate(Gs_test):
        sp_lengths = dict(nx.shortest_path_length(G))
        sp_counts_test[i] = dict()
        nodes = G.nodes()
        for v1 in nodes:
            for v2 in nodes:
                if v2 in sp_lengths[v1]:
                    length = sp_lengths[v1][v2]
                    if length in sp_counts_test[i]:
                        sp_counts_test[i][length] += 1
                    else:
                        sp_counts_test[i][length] = 1

                    if length not in all_paths:
                        all_paths[length] = len(all_paths)

    phi_train = np.zeros((len(Gs_train), len(all_paths)))
    for i in range(len(Gs_train)):
        for length in sp_counts_train[i]:
            phi_train[i,all_paths[length]] = sp_counts_train[i][length]
    
  
    phi_test = np.zeros((len(Gs_test), len(all_paths)))
    for i in range(len(Gs_test)):
        for length in sp_counts_test[i]:
            phi_test[i,all_paths[length]] = sp_counts_test[i][length]

    K_train = np.dot(phi_train, phi_train.T)
    K_test = np.dot(phi_test, phi_train.T)

    return K_train, K_test



############## Task 8
# Compute the graphlet kernel
def graphlet_kernel(Gs_train, Gs_test, n_samples=200):
    graphlets = [nx.Graph(), nx.Graph(), nx.Graph(), nx.Graph()]
    
    #G1: empty graph
    graphlets[0].add_nodes_from(range(3))

    #G2: 1 edge
    graphlets[1].add_nodes_from(range(3))
    graphlets[1].add_edge(0,1)

    #G3: 2 edges
    graphlets[2].add_nodes_from(range(3))
    graphlets[2].add_edge(0,1)
    graphlets[2].add_edge(1,2)
    #G4: 3 edges
    graphlets[3].add_nodes_from(range(3))
    graphlets[3].add_edge(0,1)
    graphlets[3].add_edge(1,2)
    graphlets[3].add_edge(0,2)

    
    phi_train = np.zeros((len(G_train), 4))
    # Train set
    for i,G in enumerate(Gs_train):
        nodes=list(G.nodes())
        for _ in range(n_samples):
            #sample 3 nodes
            sampled=np.random.choice(nodes,size=3,replace=False)
            sub=G.subgraph(sampled)
            #compare to each graphlet
            for g_id,glet in enumerate(graphlets):
                if nx.is_isomorphic(sub,glet):
                    phi_train[i,g_id]+=1
                    break
    

    phi_test = np.zeros((len(G_test), 4))

    #Test set
    for i,G in enumerate(Gs_test):
        nodes=list(G.nodes())
        for _ in range(n_samples):
            #sample 3 nodes
            sampled=np.random.choice(nodes,size=3,replace=False)
            sub=G.subgraph(sampled)
            #compare to each graphlet
            for g_id,glet in enumerate(graphlets):
                if nx.is_isomorphic(sub,glet):
                    phi_train[i,g_id]+=1
                    break

    K_train = np.dot(phi_train, phi_train.T)
    K_test = np.dot(phi_test, phi_train.T)

    return K_train, K_test


K_train_sp, K_test_sp = shortest_path_kernel(G_train, G_test)



############## Task 9
K_train_glt, K_test_glt = graphlet_kernel(G_train, G_test)
print("Tasks 8,9 done.")

############## Task 10

#shortest path kernel
clf_sp=SVC(kernel="precomputed")
clf_sp.fit(K_train_sp,y_train)
y_pred_sp=clf_sp.predict(K_test_sp)
acc_sp=accuracy_score(y_test,y_pred_sp)
print("Accuracy (Shortest Path Kernel):", acc_sp)

#graphlet kernel
# ---- Graphlet Kernel ----
clf_glt = SVC(kernel='precomputed')
clf_glt.fit(K_train_glt, y_train)
y_pred_glt = clf_glt.predict(K_test_glt)
acc_glt = accuracy_score(y_test, y_pred_glt)
print("Accuracy (Graphlet Kernel):", acc_glt)

print("Task 10 done")

# The shortest-path kernel significantly outperforms the 3-graphlet kernel on MUTAG
# (89.5% vs. 68.4% accuracy).
# This is expected, since MUTAG molecules are better characterized by global structural
# information (captured by shortest path distances) rather than purely local subgraph
# patterns of size 3. Therefore, the shortest-path kernel is more effective for this dataset.