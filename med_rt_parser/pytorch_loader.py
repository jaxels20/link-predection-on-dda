import torch
import torch_geometric as pyg
from torch_geometric.data import HeteroData
import networkx as nx
import torch_geometric.transforms as T
import numpy 
from torch_geometric.utils import to_undirected

try: 
    from med_rt_parser.networkX_loader import get_networkx_graph
except:
    from networkX_loader import get_networkx_graph

def get_pyg(bipartite=True):
    # Load the graph
    nx_graph = get_networkx_graph(bipartite=True)


    drug_nodes = [node for node in nx_graph.nodes if nx_graph.nodes[node]["type"] == "drug"]
    disease_nodes = [node for node in nx_graph.nodes if nx_graph.nodes[node]["type"] == "disease"]
    """ 
    MoA_nodes = [node for node in nx_graph.nodes if nx_graph.nodes[node]["type"] == "MoA"]
    SC_nodes = [node for node in nx_graph.nodes if nx_graph.nodes[node]["type"] == "SC"]
    PE_nodes = [node for node in nx_graph.nodes if nx_graph.nodes[node]["type"] == "PE"]
    TC_nodes = [node for node in nx_graph.nodes if nx_graph.nodes[node]["type"] == "TC"] """
    

    edge_attributes = nx.get_edge_attributes(nx_graph, "association_type")

    data = HeteroData()

    # Add the nodes
    data["drug"].node_id = torch.arange(len(drug_nodes))
    data["drug"].node_type = torch.zeros(len(drug_nodes), dtype=torch.long)
    
    data["disease"].node_id = torch.arange(len(disease_nodes))
    data["disease"].node_type = torch.ones(len(disease_nodes), dtype=torch.long)

    """ 
    data["MoA"].node_id = torch.arange(len(MoA_nodes))
    data["SC"].node_id = torch.arange(len(SC_nodes))
    data["PE"].node_id = torch.arange(len(PE_nodes))
    data["TC"].node_id = torch.arange(len(TC_nodes)) """


    # this is where to add the features (important that they are the same size and numerical)
    data["drug"].x = torch.zeros(len(drug_nodes), 10)
    data["disease"].x = torch.ones(len(disease_nodes), 10)
    """ 
    data["MoA"].x = torch.ones(len(MoA_nodes), 10)
    data["SC"].x = torch.ones(len(SC_nodes), 10)
    data["PE"].x = torch.ones(len(PE_nodes), 10)
    data["TC"].x = torch.ones(len(TC_nodes), 10) """


    # Add the edges
    for edge in nx_graph.edges:
        try:
            data["drug", str(edge_attributes[edge]), "disease"].edge_index = torch.cat(
                (data["drug", str(edge_attributes[edge]), "disease"].edge_index, torch.tensor([[drug_nodes.index(edge[0])], [disease_nodes.index(edge[1])]])), 
                dim=1)
        except:
            data["drug", str(edge_attributes[edge]), "disease"].edge_index = (torch.tensor([[drug_nodes.index(edge[0])], [disease_nodes.index(edge[1])]]))
    
    return T.ToUndirected()(data)

def is_bipartite(pyg):
    for edge_type in pyg.edge_types:
        for edge in pyg[edge_type].edge_index.T:
            from_node = edge[0].item()
            to_node = edge[1].item()
            if edge_type == ("drug", "may_treat", "disease"):
                from_node_type = pyg["drug"]["node_type"][from_node].item()
                to_node_type = pyg["disease"]["node_type"][to_node].item()

            elif edge_type == ("disease", "rev_may_treat", "drug"):
                from_node_type = pyg["disease"]["node_type"][from_node].item()
                to_node_type = pyg["drug"]["node_type"][to_node].item()
            else:
                raise ValueError("Edge type not recognized")

            if from_node_type == to_node_type:
                print("Error: edge between nodes of the same type")
                return False
            
    return True


if __name__ == "__main__":
    pyg = get_pyg()
    
    print(is_bipartite(pyg))

    
        
    




