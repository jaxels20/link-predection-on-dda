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
import optuna


#HIDDEN_CHANNELS = 100
EPOCHS = 10
#LEARNING_RATE = 0.001
NUM_VAL = 0.1
NUM_TEST = 0.2
#DISJOINT_TRAIN_RATIO = 0.3
#NEG_SAMPLING_RATIO = 1.0
ADD_NEGATIVE_TRAIN_SAMPLES = False
#BATCH_SIZE = 128
NUM_NEIGHBORS = [20, 10]
SHUFFLE = False
AGGR = 'max'
EARLY_STOPPING_PATIENCE = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_train_loader(train_data, negative_sample_ratio, batch_size ):
    edge_label_index = train_data["drug", "may_treat", "disease"].edge_label_index
    edge_label = train_data["drug", "may_treat", "disease"].edge_label

    train_loader = LinkNeighborLoader(
        data=train_data,
        num_neighbors=NUM_NEIGHBORS,
        neg_sampling_ratio=negative_sample_ratio,
        edge_label_index=(("drug", "may_treat", "disease"), edge_label_index),
        edge_label=edge_label,
        batch_size=batch_size,
        shuffle=SHUFFLE,
    )
    return train_loader

def get_val_loader(val_data, batch_size, negative_sample_ratio):
    edge_label_index = val_data["drug", "may_treat", "disease"].edge_label_index
    edge_label = val_data["drug", "may_treat", "disease"].edge_label

    val_loader = LinkNeighborLoader(
        data=val_data,
        num_neighbors=NUM_NEIGHBORS,
        neg_sampling_ratio=negative_sample_ratio,
        edge_label_index=(("drug", "may_treat", "disease"), edge_label_index),
        edge_label=edge_label,
        batch_size=3 * batch_size,
        shuffle=SHUFFLE,
        neighbor_sampler=None,
    )
    return val_loader

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
    def __init__(self, hidden_channels, pyg, size_nn, size_gnn):
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
        self.gnn = GNN(hidden_channels, size_gnn)

        # Convert GNN model into a heterogeneous variant:
        self.gnn = to_hetero(self.gnn, metadata=pyg.metadata())
        self.classifier = Classifier(size_nn, hidden_channels)

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

def train_model(model, train_loader, lr):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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

def early_stopping_train_model(model, train_loader, val_loader, lr):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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


def compute_validation_loss(model, val_loader):
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
        return val_loss

def train_and_eval_model(lr, hidden_channels, disjoint_train_ratio, neg_sampling_ratio, batch_size, size_gnn, size_nn):
    pyg = get_pyg(True)

    transform = T.RandomLinkSplit(
        num_val=NUM_VAL,
        num_test=NUM_TEST,
        disjoint_train_ratio=disjoint_train_ratio,
        neg_sampling_ratio=neg_sampling_ratio,
        add_negative_train_samples=ADD_NEGATIVE_TRAIN_SAMPLES,
        edge_types=("drug", "may_treat", "disease"),
        rev_edge_types=("disease", "rev_may_treat", "drug"), 
    )

    train_data, val_data, test_data = transform(pyg)

    train_loader = get_train_loader(train_data, batch_size=batch_size, negative_sample_ratio=neg_sampling_ratio)

    val_loader = get_val_loader(val_data, batch_size=batch_size, negative_sample_ratio=neg_sampling_ratio)

    model = Model(hidden_channels=hidden_channels, pyg=pyg, size_gnn=size_gnn, size_nn=size_nn)
    early_stopping_train_model(model, train_loader, val_loader, lr)

    val_loss = compute_validation_loss(model, val_loader)

    return val_loss

def objective(trail):

    lr = trail.suggest_float("lr", 1e-5, 1e-1)
    hidden_channels = trail.suggest_int("hidden_channels", 16, 128)
    disjoint_train_ratio = trail.suggest_float("disjoint_train_ratio", 0.1, 0.5)
    neg_sampling_ratio = trail.suggest_float("neg_sampling_ratio", 1, 3)
    batch_size = trail.suggest_int("batch_size", 64, 265)
    size_gnn = trail.suggest_int("size_gnn", 5, 15)
    size_nn = trail.suggest_int("size_nn", 5, 15)

    val_loss = train_and_eval_model(lr, hidden_channels, disjoint_train_ratio, neg_sampling_ratio, batch_size, size_gnn, size_nn)

    return val_loss








if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10, timeout=600)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))


    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    lr = trial.params["lr"]
    hidden_channels = trial.params["hidden_channels"]
    disjoint_train_ratio = trial.params["disjoint_train_ratio"]
    neg_sampling_ratio = trial.params["neg_sampling_ratio"]
    batch_size = trial.params["batch_size"]
    size_gnn = trial.params["size_gnn"]
    size_nn = trial.params["size_nn"]


    pyg = get_pyg(True)

    transform = T.RandomLinkSplit(
        num_val=NUM_VAL,
        num_test=NUM_TEST,
        disjoint_train_ratio=disjoint_train_ratio,
        neg_sampling_ratio=neg_sampling_ratio,
        add_negative_train_samples=ADD_NEGATIVE_TRAIN_SAMPLES,
        edge_types=("drug", "may_treat", "disease"),
        rev_edge_types=("disease", "rev_may_treat", "drug"), 
    )

    train_data, val_data, test_data = transform(pyg)

    train_loader = get_train_loader(train_data, batch_size=batch_size, negative_sample_ratio=neg_sampling_ratio)

    val_loader = get_val_loader(val_data, batch_size=batch_size, negative_sample_ratio=neg_sampling_ratio)

    model = Model(hidden_channels=hidden_channels, pyg=pyg, size_gnn=size_gnn, size_nn=size_nn)
    early_stopping_train_model(model, train_loader, val_loader, lr)

    auc, recall, accuracy, f1, precision = evaluate_model(model, test_data)

    """ ID = get_id()
    append_model_results_to_csv(ID, auc, recall, accuracy, f1, precision)
    export_model_configuration(ID)
    """














