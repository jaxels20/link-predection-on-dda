import torch
from torch_geometric.data import Data
import torch.nn as nn
""" import torch.nn.functional as F
from torch_geometric.loader import LinkNeighborLoader
import networkx as nx
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
torch.manual_seed(5)


x = torch.rand(10, 1)
edge_index = torch.randint(0, 10, (2, 10))
print(edge_index)

data = Data(x=x, edge_index=edge_index, edge_label=torch.ones(10))

data.edge_index_label = edge_index

loader = LinkNeighborLoader(data, 
                            batch_size=5, 
                            shuffle=True, 
                            num_neighbors=[1], 
                            neg_sampling_ratio=1.0, 
                            edge_label_index=data.edge_index_label, 
                            edge_label=data.edge_label)

for x in loader:
    # Convert the Data object to a NetworkX graph
    print(f"label_index {x.edge_label_index}")
    print(f" edge_index {x.edge_index}")
    print(f"edge label {x.edge_label}")

    G = to_networkx(x, to_undirected=False)
    # Print the NetworkX graph

    # draw the network
    nx.draw_circular(G, with_labels=True)


    plt.show()
 """


import torch

# create a nested tensor with some values
my_tensor = torch.tensor([[1, 2], [3, 4]])



# apply the function to the nested tensor
log_tensor = torch.log(my_tensor)

# print the result
print(log_tensor)