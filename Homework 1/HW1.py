"""
ACMS 80770-03: Deep Learning with Graphs
Instructor: Navid Shervani-Tabar
Fall 2022
University of Notre Dame

Homework 1: Programming assignment
"""

from re import A
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from networkx.algorithms import bipartite
from networkx.generators.random_graphs import erdos_renyi_graph
import copy


# -- Initialize graphs
seed = 30
G = nx.florentine_families_graph()
nodes = G.nodes()

layout = nx.spring_layout(G, seed=seed)

# -- compute jaccard's similarity
"""
    This example is using NetwrokX's native implementation to compute similarities.
    Write a code to compute Jaccard's similarity and replace with this function.
"""
def get_jaccard(G):
    node_list  = list(nodes)
    A = nx.adjacency_matrix(G)
    A= A.to_numpy_array()
    A_numer = np.matmul(A,A)
    A_deno = np.ones_like(A_numer)
    for i in range(len(A)):
        for j in range(len(A)):
            A_deno[i,j] = np.sum(A[i]) + np.sum(A[j]) - A_numer[i,j]
    
    Jaccard_matrix = A_numer/A_deno
    item = (node_list[i],node_list[j],Jaccard_matrix[i][j] for i in range(len(A)) for j in range(len(A)))

    return item

pred = get_jaccard(G)




# -- keep a copy of edges in the graph
old_edges = copy.deepcopy(G.edges())

# -- add new edges representing similarities.
new_edges, metric = [], []
for u, v, p in pred:
    G.add_edge(u, v)
    print(f"({u}, {v}) -> {p:.8f}")
    new_edges.append((u, v))
    metric.append(p)

# -- plot Florentine Families graph
nx.draw_networkx_nodes(G, nodelist=nodes, label=nodes, pos=layout, node_size=600)
nx.draw_networkx_edges(G, edgelist=old_edges, pos=layout, edge_color='gray', width=4)

# -- plot edges representing similarity
"""
    This example is randomly plotting similarities between 8 pairs of nodes in the graph. 
    Identify the ”Ginori”
"""
for i in range(len(A)):
   if new_edges[i][0] == "Ginori":
       break
    

ne = nx.draw_networkx_edges(G, edgelist=new_edges[i:i+len(A)], pos=layout, edge_color=np.asarray(metric[:8]), width=4, alpha=0.7)
plt.colorbar(ne)
plt.axis('off')
plt.show()
