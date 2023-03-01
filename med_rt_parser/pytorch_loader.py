import torch
import torch_geometric as pyg
from torch_geometric.data import HeteroData
from med_rt_parser.networkX_loader import get_networkx_graph
import networkx as nx

def get_pyg(bipartite=True):
    # Load the graph
    nx_graph = get_networkx_graph(bipartite=True)

    drug_nodes = [node for node in nx_graph.nodes if nx_graph.nodes[node]["type"] == "drug"]
    disease_nodes = [node for node in nx_graph.nodes if nx_graph.nodes[node]["type"] == "disease"]
    edge_attributes = nx.get_edge_attributes(nx_graph, "association_type")


    data = HeteroData()

    # Add the nodes
    data["drug"].node_id = torch.arange(len(drug_nodes))
    data["disease"].node_id = torch.arange(len(disease_nodes))


    # this is where to add the features (important that they are the same size and numerical)
    data["drug"].x = torch.zeros(len(drug_nodes), 10)
    data["disease"].x = torch.ones(len(disease_nodes), 10)



    # Add the edges
    for edge in nx_graph.edges:
        try:
            data["drug", str(edge_attributes[edge]), "disease"].edge_index = torch.cat((data["drug", str(edge_attributes[edge]), "disease"].edge_index, torch.tensor([[drug_nodes.index(edge[0])], [disease_nodes.index(edge[1])]])), dim=1)
        except:
            data["drug", str(edge_attributes[edge]), "disease"].edge_index = (torch.tensor([[drug_nodes.index(edge[0])], [disease_nodes.index(edge[1])]]))
    
    return data







