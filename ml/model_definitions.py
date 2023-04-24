import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import to_hetero
import torch.nn as nn
from torch_geometric.utils import negative_sampling
import networkx as nx
import numpy as np
from torch_geometric.utils.convert import to_networkx
from torch_geometric.nn import MetaPath2Vec


class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, size_gnn):
        super().__init__()
        self.convs = torch.nn.ModuleList()

        for i in range(size_gnn):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))

        self.act = F.leaky_relu

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.act(x)

        return x
    
class Classifier(torch.nn.Module):
    def __init__(self, size_nn, hidden_channels):
        super().__init__()

        sizes = []
        sizes.append(2*hidden_channels)
        for i in range(size_nn):
            size = 2**(size_nn-i+2)

            if size > 2*hidden_channels:
                size = 2*hidden_channels


            sizes.append(size)

        sizes.append(1)

        self.fcs = torch.nn.ModuleList([torch.nn.Linear(sizes[i], sizes[i+1]) for i in range(len(sizes)-1)])
        self.act = F.leaky_relu



        self.act = F.leaky_relu

    def forward(self, x_drug, x_disease, edge_label_index):
        """
        This is the link prediction model. It takes in the node embeddings and the edge indices
        """
        # Convert node embeddings to edge-level representations:
        edge_feat_drug = x_drug[edge_label_index[0]]
        edge_feat_disease = x_disease[edge_label_index[1]]

        # Apply MLP to get a prediction per supervision edge:
        edge_feat = torch.cat([edge_feat_drug, edge_feat_disease], dim=-1)

        for fc in self.fcs[:-1]:
            edge_feat = self.act(fc(edge_feat))
        
        edge_feat = self.fcs[-1](edge_feat)
        
        # convert the tensor to a 1D array

        edge_feat = edge_feat.squeeze()

        return edge_feat
    
class Model(torch.nn.Module):
    def __init__(self, hidden_channels, pyg, size_nn, size_gnn, is_bipartite):
        super().__init__()

        self.disease_lin = torch.nn.Linear(10, hidden_channels)
        self.drug_emb = torch.nn.Embedding(pyg["drug"].num_nodes, hidden_channels)
        self.disease_emb = torch.nn.Embedding(pyg["disease"].num_nodes, hidden_channels)

        if not is_bipartite:
            # the first parameter is length of each input vector
            self.MoA_emb = torch.nn.Embedding(pyg["MoA"].num_nodes, hidden_channels)
            self.EPC_emb = torch.nn.Embedding(pyg["EPC"].num_nodes, hidden_channels)
            self.PE_emb = torch.nn.Embedding(pyg["PE"].num_nodes, hidden_channels)
            self.TC_emb = torch.nn.Embedding(pyg["TC"].num_nodes, hidden_channels)
            self.HC_emb = torch.nn.Embedding(pyg["HC"].num_nodes, hidden_channels)
            self.APC_emb = torch.nn.Embedding(pyg["APC"].num_nodes, hidden_channels)
            self.EXT_emb = torch.nn.Embedding(pyg["EXT"].num_nodes, hidden_channels)
            self.PK_emb = torch.nn.Embedding(pyg["PK"].num_nodes, hidden_channels)

        # Instantiate homogeneous GNN:
        self.gnn = GNN(hidden_channels, size_gnn)

        # Convert GNN model into a heterogeneous variant:
        self.gnn = to_hetero(self.gnn, metadata=pyg.metadata())
        self.classifier = Classifier(size_nn, hidden_channels)
        self.is_bipartite = is_bipartite


    def forward(self, data):
        if self.is_bipartite:
            x_dict = {
                "drug": self.drug_emb(data["drug"].node_id),
                "disease": self.disease_lin(data["disease"].x) + self.disease_emb(data["disease"].node_id),
            }
        else:
            x_dict = {
                "drug": self.drug_emb(data["drug"].node_id),
                "disease": self.disease_lin(data["disease"].x) + self.disease_emb(data["disease"].node_id),
                "MoA": self.MoA_emb(data["MoA"].node_id),
                "EPC": self.EPC_emb(data["EPC"].node_id),
                "PE": self.PE_emb(data["PE"].node_id),
                "TC": self.TC_emb(data["TC"].node_id),
                "HC": self.HC_emb(data["HC"].node_id),
                "APC": self.APC_emb(data["APC"].node_id),
                "EXT": self.EXT_emb(data["EXT"].node_id),
                "PK": self.PK_emb(data["PK"].node_id),
            }
        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        
        pred = self.classifier(
            x_dict["drug"],
            x_dict["disease"],
            data["drug", "may_treat", "disease"].edge_label_index,
        )
        
        return pred

class Generator(nn.Module):
    def __init__(self, input_size, num_drugs, num_diseases, embedder):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LeakyReLU(),
            nn.Linear(256, num_diseases),
            nn.Softmax(dim=-1),
        )
        self.num_drugs = num_drugs
        self.num_diseases = num_diseases
        self.input_size = input_size
        self.drug_emb = embedder("drug")


    def forward(self, batch_size):

        # Sample a random batch of drugs
        drug_idx = torch.randint(high=self.num_drugs, size=(batch_size,))

        drug_embeddings = self.drug_emb[drug_idx]

        # generate the probability distribution for diseases
        scores = self.net(drug_embeddings)
        
        # Sample indices for drugs and diseases separately
        disease_idx = torch.multinomial(scores, num_samples=1).squeeze()
        
        fake_edge_index = torch.stack([drug_idx, disease_idx], dim=0)

        # convert the edge_index to a float tensor
        fake_edge_index = fake_edge_index.type(torch.FloatTensor)

        return fake_edge_index, scores, disease_idx
    
        

    
# Define the discriminator network
class Discriminator(nn.Module):
    def __init__(self, input_size, embedder):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.Sigmoid(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.input_size = input_size
        self.drug_emb = embedder("drug")
        self.disease_emb = embedder("disease")

        
    def forward(self, edge_index):
        edge_index = edge_index.type(torch.LongTensor)

        drug_emb = self.drug_emb[edge_index[0]]
        disease_emb = self.disease_emb[edge_index[1]]

        x = torch.cat([drug_emb, disease_emb], dim=1)

        x = self.net(x)
        return x

# Define the GAN
class GAN(nn.Module):
    def __init__(self, emb_dim, num_drugs, num_diseases, pyg):
        super(GAN, self).__init__()
        self.embedder = self.get_emb(pyg, emb_dim)

        self.discriminator = Discriminator(input_size=2 * emb_dim, embedder=self.embedder)
        self.generator = Generator(emb_dim, num_drugs, num_diseases, embedder=self.embedder)
        self.num_drugs = num_drugs
        self.num_diseases = num_diseases
        self.emb_dim = emb_dim


    def forward(self, pyg):
        edge_index = pyg["drug", "may_treat", "disease"].edge_index
        x = {
            "drug": pyg["drug"].x,
            "disease": pyg["disease"].x,
        }

        noise = torch.randn(120, self.input_size)

        # produce the fake edges
        drug_idx, disease_idx = self.generator(noise)
        fake_edge_index = torch.stack([drug_idx, disease_idx], dim=0)

        # add the fake edges to the real edges
        all_edges = torch.cat([edge_index, fake_edge_index], dim=1)

        # Compute discriminator scores
        scores = self.discriminator(all_edges)

        # Split real and fake scores
        real_scores = scores[:edge_index.size(1)]
        fake_scores = scores[edge_index.size(1):]

    def get_emb(self, pyg, emb_dim):
        # Define Node2Vec model
        metapath = [
        ('drug', 'may_treat', 'disease'),
        ('disease', 'rev_may_treat', 'drug'),
        ]

        device = "cpu"

        model = MetaPath2Vec(pyg.edge_index_dict, embedding_dim=emb_dim,
                            metapath=metapath, walk_length=4, context_size=4,
                            walks_per_node=5, num_negative_samples=5,
                            sparse=True).to(device)

        loader = model.loader(batch_size=128, shuffle=True, num_workers=6)
        optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)
        
        for epoch in range(1, 20):
            total_loss = 0
            for i, (pos_rw, neg_rw) in enumerate(loader):
                optimizer.zero_grad()
                loss = model.loss(pos_rw.to(device), neg_rw.to(device))
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                """ if (i + 1) % 10 == 0:
                    print((f'Epoch: {epoch}, Step: {i + 1:05d}/{len(loader)}, '
                        f'Loss: {total_loss / 100:.4f}'))
                    total_loss = 0 """

        return model
        




        
