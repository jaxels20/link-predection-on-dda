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
    nx_graph = get_networkx_graph(bipartite=False)

    drug_nodes = [node for node in nx_graph.nodes if nx_graph.nodes[node]["type"] == "drug"]
    disease_nodes = [node for node in nx_graph.nodes if nx_graph.nodes[node]["type"] == "disease"]
    
    MoA_nodes = [node for node in nx_graph.nodes if nx_graph.nodes[node]["type"] == "MoA"]
    PE_nodes = [node for node in nx_graph.nodes if nx_graph.nodes[node]["type"] == "PE"]
    TC_nodes = [node for node in nx_graph.nodes if nx_graph.nodes[node]["type"] == "TC"]
    EPC_nodes = [node for node in nx_graph.nodes if nx_graph.nodes[node]["type"] == "EPC"]
    EXT_nodes = [node for node in nx_graph.nodes if nx_graph.nodes[node]["type"] == "EXT"]
    PK_nodes = [node for node in nx_graph.nodes if nx_graph.nodes[node]["type"] == "PK"]
    APC_nodes = [node for node in nx_graph.nodes if nx_graph.nodes[node]["type"] == "APC"]
    HC_nodes = [node for node in nx_graph.nodes if nx_graph.nodes[node]["type"] == "HC"]

    edge_attributes = nx.get_edge_attributes(nx_graph, "association_type")

    data = HeteroData()

    # Add the nodes
    data["drug"].node_id = torch.arange(len(drug_nodes))
    data["drug"].node_type = torch.zeros(len(drug_nodes), dtype=torch.long)
    
    data["disease"].node_id = torch.arange(len(disease_nodes))
    data["disease"].node_type = torch.ones(len(disease_nodes), dtype=torch.long)

    
    data["MoA"].node_id = torch.arange(len(MoA_nodes))
    data["PE"].node_id = torch.arange(len(PE_nodes))
    data["TC"].node_id = torch.arange(len(TC_nodes))
    data["EPC"].node_id = torch.arange(len(EPC_nodes))
    data["EXT"].node_id = torch.arange(len(EXT_nodes))
    data["PK"].node_id = torch.arange(len(PK_nodes))
    data["APC"].node_id = torch.arange(len(APC_nodes))
    data["HC"].node_id = torch.arange(len(HC_nodes))


    # this is where to add the features (important that they are the same size and numerical)
    data["drug"].x = torch.tensor([1,0,0,0,0,0,0,0,0,0]).to(torch.float32).repeat(len(drug_nodes), 1)
    data["disease"].x = torch.tensor([0,1,0,0,0,0,0,0,0,0]).to(torch.float32).repeat(len(disease_nodes), 1)

    data["MoA"].x = torch.tensor([0,0,1,0,0,0,0,0,0,0]).to(torch.float32).repeat(len(MoA_nodes), 1)
    data["PE"].x = torch.tensor([0,0,0,1,0,0,0,0,0,0]).to(torch.float32).repeat(len(PE_nodes), 1)
    data["TC"].x = torch.tensor([0,0,0,0,1,0,0,0,0,0]).to(torch.float32).repeat(len(TC_nodes), 1)
    data["EPC"].x = torch.tensor([0,0,0,0,0,1,0,0,0,0]).to(torch.float32).repeat(len(EPC_nodes), 1)
    data["EXT"].x = torch.tensor([0,0,0,0,0,0,1,0,0,0]).to(torch.float32).repeat(len(EXT_nodes), 1)
    data["PK"].x = torch.tensor([0,0,0,0,0,0,0,1,0,0]).to(torch.float32).repeat(len(PK_nodes), 1)
    data["APC"].x = torch.tensor([0,0,0,0,0,0,0,0,1,0]).to(torch.float32).repeat(len(APC_nodes), 1)
    data["HC"].x = torch.tensor([0,0,0,0,0,0,0,0,0,1]).to(torch.float32).repeat(len(HC_nodes), 1)

    data["MoA"].node_type = torch.ones(len(MoA_nodes), dtype=torch.long)
    data["PE"].node_type = torch.ones(len(PE_nodes), dtype=torch.long)
    data["TC"].node_type = torch.ones(len(TC_nodes), dtype=torch.long)
    data["EPC"].node_type = torch.ones(len(EPC_nodes), dtype=torch.long)
    data["EXT"].node_type = torch.ones(len(EXT_nodes), dtype=torch.long)
    data["PK"].node_type = torch.ones(len(PK_nodes), dtype=torch.long)
    data["APC"].node_type = torch.ones(len(APC_nodes), dtype=torch.long)
    data["HC"].node_type = torch.ones(len(HC_nodes), dtype=torch.long)
    



    # Add the edges
    for edge in nx_graph.edges:

        from_node_list = []
        to_node_list = []

        from_node_type = nx_graph.nodes[edge[0]]["type"]
        to_node_type = nx_graph.nodes[edge[1]]["type"]

        if from_node_type == "drug":
            from_node_list = drug_nodes
        elif from_node_type == "disease":
            from_node_list = disease_nodes
        elif from_node_type == "MoA":
            from_node_list = MoA_nodes
        elif from_node_type == "PE":
            from_node_list = PE_nodes
        elif from_node_type == "TC":
            from_node_list = TC_nodes
        elif from_node_type == "EPC":
            from_node_list = EPC_nodes
        elif from_node_type == "EXT":
            from_node_list = EXT_nodes
        elif from_node_type == "PK":
            from_node_list = PK_nodes
        elif from_node_type == "APC":
            from_node_list = APC_nodes
        elif from_node_type == "HC":
            from_node_list = HC_nodes

        else:
            raise ValueError("Node type not recognized")

        if to_node_type == "drug":
            to_node_list = drug_nodes
        elif to_node_type == "disease":
            to_node_list = disease_nodes
        elif to_node_type == "MoA":
            to_node_list = MoA_nodes
        elif to_node_type == "PE":
            to_node_list = PE_nodes
        elif to_node_type == "TC":
            to_node_list = TC_nodes
        elif to_node_type == "EPC":
            to_node_list = EPC_nodes
        elif to_node_type == "EXT":
            to_node_list = EXT_nodes
        elif to_node_type == "PK":
            to_node_list = PK_nodes
        elif to_node_type == "APC":
            to_node_list = APC_nodes
        elif to_node_type == "HC":
            to_node_list = HC_nodes
        else:
            raise ValueError("Node type not recognized")

        try:
            data[from_node_type, str(edge_attributes[edge]), to_node_type].edge_index = torch.cat(
                (data[from_node_type, str(edge_attributes[edge]), to_node_type].edge_index, torch.tensor([[from_node_list.index(edge[0])], [to_node_list.index(edge[1])]])), 
                dim=1)

        except:
            data[from_node_type, str(edge_attributes[edge]), to_node_type].edge_index = (torch.tensor([[from_node_list.index(edge[0])], [to_node_list.index(edge[1])]]))

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
    print(pyg)    
        
    




