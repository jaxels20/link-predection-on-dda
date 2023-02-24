import sys
import networkx as nx
sys.path.append("..")
from torch_geometric.utils.convert import from_networkx, to_networkx
from torch_geometric.data import HeteroData
import torch 
from med_rt_parser.pytorch_loader import get_pyg
import torch_geometric.transforms as T
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import SAGEConv, to_hetero
import torch.nn.functional as F
import tqdm
from sklearn.metrics import roc_auc_score





pyg = get_pyg()

pyg = T.ToUndirected()(pyg)

transform = T.RandomLinkSplit(
    num_val=0.1,
    num_test=0.2,
    disjoint_train_ratio=0.3,
    neg_sampling_ratio=2.0,
    add_negative_train_samples=False,
    edge_types=("drug", "may_treat", "disease"),
    rev_edge_types=("disease", "rev_may_treat", "drug"), 
)
train_data, val_data, test_data = transform(pyg)

# Define seed edges:
edge_label_index = train_data["drug", "may_treat", "disease"].edge_label_index
edge_label = train_data["drug", "may_treat", "disease"].edge_label

train_loader = LinkNeighborLoader(
    data=train_data,
    num_neighbors=[10, 10],
    neg_sampling_ratio=2.0,
    edge_label_index=(("drug", "may_treat", "disease"), edge_label_index),
    edge_label=edge_label,
    batch_size=128,
    shuffle=True,
)


class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x
# Our final classifier applies the dot-product between source and destination
# node embeddings to derive edge-level predictions:
class Classifier(torch.nn.Module):
    def forward(self, x_user, x_movie, edge_label_index):
        # Convert node embeddings to edge-level representations:
        edge_feat_user = x_user[edge_label_index[0]]
        edge_feat_movie = x_movie[edge_label_index[1]]
        # Apply dot-product to get a prediction per supervision edge:
        return (edge_feat_user * edge_feat_movie).sum(dim=-1)

class Model(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        # Since the dataset does not come with rich features, we also learn two
        # embedding matrices for users and movies:

        # the first parameter is length of each input vector
        self.movie_lin = torch.nn.Linear(1, hidden_channels)
        self.user_emb = torch.nn.Embedding(pyg["drug"].num_nodes, hidden_channels)
        self.movie_emb = torch.nn.Embedding(pyg["disease"].num_nodes, hidden_channels)
        # Instantiate homogeneous GNN:
        self.gnn = GNN(hidden_channels)
        # Convert GNN model into a heterogeneous variant:
        self.gnn = to_hetero(self.gnn, metadata=pyg.metadata())
        self.classifier = Classifier()
    def forward(self, data: HeteroData):
        x_dict = {
          "drug": self.user_emb(data["drug"].node_id),
          "disease": self.movie_lin(data["disease"].x) + self.movie_emb(data["disease"].node_id),
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
        
model = Model(hidden_channels=256)   

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: '{device}'")
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for epoch in range(1, 6):
    total_loss = total_examples = 0
    for sampled_data in tqdm.tqdm(train_loader):
        optimizer.zero_grad()
        sampled_data.to(device)
        pred = model(sampled_data)
        ground_truth = sampled_data["drug", "may_treat", "disease"].edge_label
        loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * pred.numel()
        total_examples += pred.numel()
    print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")



# Define the validation seed edges:
edge_label_index = val_data["drug", "may_treat", "disease"].edge_label_index
edge_label = val_data["drug", "may_treat", "disease"].edge_label
val_loader = LinkNeighborLoader(
    data=val_data,
    num_neighbors=[20, 10],
    edge_label_index=(("drug", "may_treat", "disease"), edge_label_index),
    edge_label=edge_label,
    batch_size=3 * 128,
    shuffle=False,
)
sampled_data = next(iter(val_loader))

preds = []
ground_truths = []
for sampled_data in tqdm.tqdm(val_loader):
    with torch.no_grad():
        sampled_data.to(device)
        preds.append(model(sampled_data))
        ground_truths.append(sampled_data["drug", "may_treat", "disease"].edge_label)
pred = torch.cat(preds, dim=0).cpu().numpy()
ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
auc = roc_auc_score(ground_truth, pred)
print()
print(f"Validation AUC: {auc:.4f}")

















