import sys
import networkx as nx
sys.path.append("..")
from torch_geometric.data import HeteroData
import torch 
from med_rt_parser.pytorch_loader import get_pyg, is_bipartite
import torch_geometric.transforms as T
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import SAGEConv, to_hetero, GATv2Conv
from torch_geometric.utils import to_undirected
import torch.nn.functional as F
import tqdm
from sklearn.metrics import roc_auc_score, recall_score, accuracy_score, f1_score, precision_score
import numpy as np
import pandas as pd
import csv
import json
from datetime import datetime
import networkx as nx


HIDDEN_CHANNELS = 100
EPOCHS = 10
LEARNING_RATE = 0.001
NUM_VAL = 0.1
NUM_TEST = 0.2
DISJOINT_TRAIN_RATIO = 0.3
NEG_SAMPLING_RATIO = 1.0
ADD_NEGATIVE_TRAIN_SAMPLES = False
BATCH_SIZE = 128
NUM_NEIGHBORS = [20, 10]
SHUFFLE = False
NUM_HEADS = None
AGGR = 'mean'
DROPOUT = None
EARLY_STOPPING_PATIENCE = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = T.RandomLinkSplit(
    num_val=NUM_VAL,
    num_test=NUM_TEST,
    disjoint_train_ratio=DISJOINT_TRAIN_RATIO,
    neg_sampling_ratio=NEG_SAMPLING_RATIO,
    add_negative_train_samples=ADD_NEGATIVE_TRAIN_SAMPLES,
    edge_types=("drug", "may_treat", "disease"),
    rev_edge_types=("disease", "rev_may_treat", "drug"), 
)

def get_train_loader(train_data):
    edge_label_index = train_data["drug", "may_treat", "disease"].edge_label_index
    edge_label = train_data["drug", "may_treat", "disease"].edge_label

    train_loader = LinkNeighborLoader(
        data=train_data,
        num_neighbors=NUM_NEIGHBORS,
        neg_sampling_ratio=NEG_SAMPLING_RATIO,
        edge_label_index=(("drug", "may_treat", "disease"), edge_label_index),
        edge_label=edge_label,
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE,
    )
    return train_loader

def get_val_loader(val_data):
    edge_label_index = val_data["drug", "may_treat", "disease"].edge_label_index
    edge_label = val_data["drug", "may_treat", "disease"].edge_label

    val_loader = LinkNeighborLoader(
        data=val_data,
        num_neighbors=NUM_NEIGHBORS,
        neg_sampling_ratio=NEG_SAMPLING_RATIO,
        edge_label_index=(("drug", "may_treat", "disease"), edge_label_index),
        edge_label=edge_label,
        batch_size=3 * BATCH_SIZE,
        shuffle=SHUFFLE,
        neighbor_sampler=None,
    )
    return val_loader

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(hidden_channels, hidden_channels, aggr=AGGR)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels, aggr=AGGR)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels, aggr=AGGR)
        self.conv4 = SAGEConv(hidden_channels, hidden_channels, aggr=AGGR)

        self.act = F.leaky_relu

    def forward(self, x, edge_index):
        x = self.act(self.conv1(x, edge_index))
        x = self.act(self.conv2(x, edge_index))
        x = self.act(self.conv3(x, edge_index))
        x = self.act(self.conv4(x, edge_index))

        return x

class Classifier(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # create neural network layers

        self.fc1 = torch.nn.Linear(200, 128)
        self.fc2 = torch.nn.Linear(128, 32)
        self.fc3 = torch.nn.Linear(32, 16)
        self.fc4 = torch.nn.Linear(16, 1)



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
        edge_feat = self.act(self.fc1(edge_feat))
        edge_feat = self.act(self.fc2(edge_feat))
        edge_feat = self.act(self.fc3(edge_feat))
        
        edge_feat = self.fc4(edge_feat)
        
        # convert the tensor to a 1D array

        edge_feat = edge_feat.squeeze()

        return edge_feat

class Model(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()

        # the first parameter is length of each input vector
        self.disease_lin = torch.nn.Linear(10, hidden_channels)
        self.drug_emb = torch.nn.Embedding(pyg["drug"].num_nodes, hidden_channels)
        self.disease_emb = torch.nn.Embedding(pyg["disease"].num_nodes, hidden_channels)
        self.MoA_emb = torch.nn.Embedding(pyg["MoA"].num_nodes, hidden_channels)
        self.EPC_emb = torch.nn.Embedding(pyg["EPC"].num_nodes, hidden_channels)
        self.PE_emb = torch.nn.Embedding(pyg["PE"].num_nodes, hidden_channels)
        self.TC_emb = torch.nn.Embedding(pyg["TC"].num_nodes, hidden_channels)
        self.HC_emb = torch.nn.Embedding(pyg["HC"].num_nodes, hidden_channels)
        self.APC_emb = torch.nn.Embedding(pyg["APC"].num_nodes, hidden_channels)
        self.EXT_emb = torch.nn.Embedding(pyg["EXT"].num_nodes, hidden_channels)
        self.PK_emb = torch.nn.Embedding(pyg["PK"].num_nodes, hidden_channels)

        # Instantiate homogeneous GNN:
        self.gnn = GNN(hidden_channels)

        # Convert GNN model into a heterogeneous variant:
        self.gnn = to_hetero(self.gnn, metadata=pyg.metadata())
        self.classifier = Classifier()

    def forward(self, data: HeteroData):
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

def train_model(model, train_loader):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    for epoch in range(1, EPOCHS):
        total_loss = total_examples = 0
        for sampled_data in tqdm.tqdm(train_loader):
            optimizer.zero_grad()
            sampled_data.to(device)
            """ Calls the forward function of the model """
            pred = model(sampled_data)
            ground_truth = sampled_data["drug", "may_treat", "disease"].edge_label
            loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel()
        print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")

def early_stopping_train_model(model, train_loader, val_loader):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_val_loss = float('inf')
    epochs_since_improvement = 0
    for epoch in range(1, EPOCHS):
        total_loss = total_examples = 0
        for sampled_data in tqdm.tqdm(train_loader):
            optimizer.zero_grad()
            sampled_data.to(device)
            """ Calls the forward function of the model """
            pred = model(sampled_data)
            ground_truth = sampled_data["drug", "may_treat", "disease"].edge_label
            loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel()
        
        # Compute validation loss
        val_loss = 0
        with torch.no_grad():
            for sampled_data in val_loader:
                sampled_data.to(device)
                pred = model(sampled_data)
                ground_truth = sampled_data["drug", "may_treat", "disease"].edge_label
                loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
                val_loss += loss.item() * pred.numel()
        val_loss /= len(val_loader.dataset)
        
        # Check if validation loss has improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            
        # Check if we should stop early
        if epochs_since_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping after {epoch} epochs.")
            break
        
        print(f"Epoch: {epoch:03d}, Train Loss: {total_loss / total_examples:.4f}, Val Loss: {val_loss:.4f}")

def evaluate_model(model, data):
    # Define the validation seed edges:
    edge_label_index = data["drug", "may_treat", "disease"].edge_label_index
    edge_label = data["drug", "may_treat", "disease"].edge_label
    val_loader = LinkNeighborLoader(
        data=data,
        num_neighbors=[20, 10],
        edge_label_index=(("drug", "may_treat", "disease"), edge_label_index),
        edge_label=edge_label,
        batch_size=128,
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

    # Calculate AUC
    auc = roc_auc_score(ground_truth, pred)
    print(f"Validation AUC: {auc:.4f}")

    # Calculate recall
    threshold = 0.5
    pred_binary = np.where(pred > threshold, 1, 0)
    recall = recall_score(ground_truth, pred_binary)
    print(f"Validation Recall: {recall:.4f}")

    # Calculate accuracy
    accuracy = accuracy_score(ground_truth, pred_binary)
    print(f"Validation Accuracy: {accuracy:.4f}")

    # Calculate f1 score
    f1 = f1_score(ground_truth, pred_binary)
    print(f"Validation F1: {f1:.4f}")

    # Calculate precision
    precision = precision_score(ground_truth, pred_binary)
    print(f"Validation Precision: {precision:.4f}")

    return auc, recall, accuracy, f1, precision

def get_id():
    try:
        with open('model_results.csv', 'r') as f:
            reader = csv.reader(f)
            last_id = list(reader)[-1][0]
        return int(last_id) + 1
    except:
        return 1

def append_model_results_to_csv(ID, auc, recall, accuracy, f1, precision):
    with open('model_results.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([ID, auc, recall, accuracy, f1, precision])
    
def export_model_configuration(ID):
    gnn_config = {str(name): str(value) for name, value in model.gnn.named_children()}
    classifier_config = {str(name): str(value) for name, value in model.classifier.named_children()}

    model_config = {
        "ID": ID,
        "TIMESTAMP": str(datetime.now()),
        "EPOCHS": EPOCHS,
        "LEARNING_RATE": LEARNING_RATE,
        "HIDDEN_CHANNELS": HIDDEN_CHANNELS,
        "NUM_VAL": NUM_VAL,
        "NUM_TEST": NUM_TEST,
        "DISJOINT_TRAIN_RATIO": DISJOINT_TRAIN_RATIO,
        "NEG_SAMPLING_RATIO": NEG_SAMPLING_RATIO,
        "ADD_NEGATIVE_TRAIN_SAMPLES": ADD_NEGATIVE_TRAIN_SAMPLES,
        "BATCH_SIZE": BATCH_SIZE,
        "NUM_NEIGHBORS": NUM_NEIGHBORS,
        "SHUFFLE": SHUFFLE,
        "NUM_HEADS": NUM_HEADS,
        "AGGR": AGGR,
        "DROPOUT": DROPOUT,
        "gnn_config": gnn_config,
        "classifier_config": classifier_config
        
    }
    with open("models_json/model_{}.json".format(ID), "w") as f:
        json.dump(model_config, f, indent=4)


if __name__ == "__main__":
    pyg = get_pyg(True)

    train_data, val_data, test_data = transform(pyg)

    train_loader = get_train_loader(train_data)

    val_loader = get_val_loader(val_data)

    model = Model(hidden_channels=HIDDEN_CHANNELS)
    early_stopping_train_model(model, train_loader, val_loader)

    auc, recall, accuracy, f1, precision = evaluate_model(model, test_data)

    ID = get_id()
    append_model_results_to_csv(ID, auc, recall, accuracy, f1, precision)
    export_model_configuration(ID)








