import torch
import torch_geometric as pyg
import networkX_loader as nxl

graph = nxl.get_networkx_graph(remove_self_loops=True, remove_isolated_nodes=True, bipartite=True)

# print number of nodes and edges
print("Number of nodes: ", graph.number_of_nodes())
print("Number of edges: ", graph.number_of_edges())



# Convert networkX graph to pytorch geometric graph

pyg_graph = pyg.utils.from_networkx(graph)

# print out the graph
print(pyg_graph.edge_index)






