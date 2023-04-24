import sys
sys.path.append("..")
import torch 
from med_rt_parser.pytorch_loader import get_pyg
import torch_geometric.transforms as T
from torch_geometric.loader import LinkNeighborLoader
import tqdm
from sklearn.metrics import roc_auc_score, recall_score, accuracy_score, f1_score, precision_score
import numpy as np
from model_definitions import Model, GAN
import torch.nn.functional as F
import export_model
from torch import nn
import pandas as pd
from torch_geometric.utils import degree
import draw_gan_performance as dgf

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_train_loader(train_data, negative_sample_ratio, batch_size, num_neighbors, shuffle=True):
    edge_label_index = train_data["drug", "may_treat", "disease"].edge_label_index
    edge_label = train_data["drug", "may_treat", "disease"].edge_label

    train_loader = LinkNeighborLoader(
        data=train_data,
        num_neighbors=num_neighbors,
        neg_sampling_ratio=negative_sample_ratio,
        edge_label_index=(("drug", "may_treat", "disease"), edge_label_index),
        edge_label=edge_label,
        batch_size=batch_size,
        shuffle=shuffle,
    )
    return train_loader

def get_val_loader(val_data, batch_size, negative_sample_ratio, num_neighbors, shuffle):
    edge_label_index = val_data["drug", "may_treat", "disease"].edge_label_index
    edge_label = val_data["drug", "may_treat", "disease"].edge_label

    val_loader = LinkNeighborLoader(
        data=val_data,
        num_neighbors=num_neighbors,
        neg_sampling_ratio=negative_sample_ratio,
        edge_label_index=(("drug", "may_treat", "disease"), edge_label_index),
        edge_label=edge_label,
        batch_size=3 * batch_size,
        shuffle=shuffle,
        neighbor_sampler=None,
    )
    return val_loader

def get_train_val_test_data(pyg, num_val, num_test, disjoint_train_ratio, neg_sampling_ratio, add_negative_train_samples):
    transform = T.RandomLinkSplit(
        num_val=num_val,
        num_test=num_test,
        disjoint_train_ratio=disjoint_train_ratio,
        neg_sampling_ratio=neg_sampling_ratio,
        add_negative_train_samples=add_negative_train_samples,
        edge_types=("drug", "may_treat", "disease"),
        rev_edge_types=("disease", "rev_may_treat", "drug")
    )

    return transform(pyg)

def early_stopping_train_model(model, train_loader, val_loader, lr, epochs, early_stopping_patience):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')
    epochs_since_improvement = 0
    for epoch in range(1, epochs):
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
        if epochs_since_improvement >= early_stopping_patience:
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

def train_and_eval_model_for_HPT(lr, hidden_channels, disjoint_train_ratio, neg_sampling_ratio, batch_size, size_gnn, size_nn):

    num_val = 0.1
    num_test = 0.2
    add_negative_train_samples = True
    num_neighbors = [20, 10]
    shuffle = True
    num_epochs = 4
    early_stopping_patience = 1
    is_bipartite = True

    pyg = get_pyg(is_bipartite)

    train_data, val_data, test_data = get_train_val_test_data(pyg, num_val, num_test, disjoint_train_ratio, neg_sampling_ratio, add_negative_train_samples)

    train_loader = get_train_loader(train_data, batch_size=batch_size, negative_sample_ratio=neg_sampling_ratio, num_neighbors=num_neighbors, shuffle=shuffle )

    val_loader = get_val_loader(val_data, batch_size=batch_size, negative_sample_ratio=neg_sampling_ratio, num_neighbors=num_neighbors, shuffle=shuffle)

    model = Model(hidden_channels=hidden_channels, pyg=pyg, size_gnn=size_gnn, size_nn=size_nn, is_bipartite=is_bipartite)
    early_stopping_train_model(model, train_loader, val_loader, lr, num_epochs, early_stopping_patience)

    val_loss = compute_validation_loss(model, val_loader)

    return val_loss

def train_and_eval_model_for_metrics(lr, hidden_channels, disjoint_train_ratio, neg_sampling_ratio, batch_size, size_gnn, size_nn, early_stopping_patience, num_epochs, is_bipartite, num_val, num_test, add_negative_train_samples, num_neighbors, shuffle):

    pyg = get_pyg(is_bipartite)

    train_data, val_data, test_data = get_train_val_test_data(pyg, num_val, num_test, disjoint_train_ratio, neg_sampling_ratio, add_negative_train_samples)

    train_loader = get_train_loader(train_data, batch_size=batch_size, negative_sample_ratio=neg_sampling_ratio, num_neighbors=num_neighbors, shuffle=shuffle )

    val_loader = get_val_loader(val_data, batch_size=batch_size, negative_sample_ratio=neg_sampling_ratio, num_neighbors=num_neighbors, shuffle=shuffle)

    model = Model(hidden_channels=hidden_channels, pyg=pyg, size_gnn=size_gnn, size_nn=size_nn, is_bipartite=is_bipartite)
    early_stopping_train_model(model, train_loader, val_loader, lr, num_epochs, early_stopping_patience)

    auc, recall, accuracy, f1, precision = evaluate_model(model, val_data)

    # export the model 

    ID = export_model.get_next_id()

    export_model.append_performance_metrics_to_csv(ID, auc, recall, accuracy, f1, precision)

    export_model.export_model_configuration(ID, model, num_epochs, lr, hidden_channels, num_val, num_test, disjoint_train_ratio, neg_sampling_ratio, add_negative_train_samples, batch_size, num_neighbors, shuffle, aggr)

def train_and_eval_model_for_metrics_gan(lr, hidden_channels, disjoint_train_ratio, batch_size, size_gnn, size_nn, early_stopping_patience, num_epochs, is_bipartite, num_val, num_test, add_negative_train_samples, num_neighbors, shuffle):
    
        pyg = get_pyg(is_bipartite)

        neg_sampling_ratio = 0
    
        train_data, val_data, test_data = get_train_val_test_data(pyg, num_val, num_test, disjoint_train_ratio, neg_sampling_ratio, add_negative_train_samples)
    
        train_loader = get_train_loader(train_data, batch_size=batch_size, negative_sample_ratio=neg_sampling_ratio, num_neighbors=num_neighbors, shuffle=shuffle )
    
        val_loader = get_val_loader(val_data, batch_size=batch_size, negative_sample_ratio=neg_sampling_ratio, num_neighbors=num_neighbors, shuffle=shuffle)
    
        model = Model(hidden_channels=hidden_channels, pyg=pyg, size_gnn=size_gnn, size_nn=size_nn, is_bipartite=is_bipartite)
        early_stopping_train_model(model, train_loader, val_loader, lr, num_epochs, early_stopping_patience)
    
        auc, recall, accuracy, f1, precision = evaluate_model(model, val_data)
    
        # export the model 
    
        ID = export_model.get_next_id()
    
        export_model.append_performance_metrics_to_csv(ID, auc, recall, accuracy, f1, precision)
    
        export_model.export_model_configuration(ID, model, num_epochs, lr, hidden_channels, num_val, num_test, disjoint_train_ratio, neg_sampling_ratio, add_negative_train_samples, batch_size, num_neighbors, shuffle, aggr)
    
def train_gan():
    g_data = {"overall_epoch": [], "sub_epoch": [], "loss": [], "acc": []}
    d_data = {"overall_epoch": [], "sub_epoch": [], "loss": [], "acc": []}
    print("train gan")
    num_epochs = 60
    batch_size = 2000
    d_epochs = 1
    g_epochs = 1

    pyg = get_pyg(True)

    train_data, val_data, test_data = get_train_val_test_data(pyg, num_val=0.1, num_test=0.2, disjoint_train_ratio=0.3, neg_sampling_ratio=0, add_negative_train_samples=False)

    train_loader = get_train_loader(train_data, batch_size=batch_size, negative_sample_ratio=0, num_neighbors=[20, 10], shuffle=False )

    val_loader = get_val_loader(val_data, batch_size=batch_size, negative_sample_ratio=0, num_neighbors=[20, 10], shuffle=False)

    num_disease = pyg["disease"]["node_id"].size(0)
    num_drug = pyg["drug"]["node_id"].size(0)
    
    real_edge_index = pyg["drug", "may_treat", "disease"]["edge_index"]

    gan = GAN(emb_dim=128, num_diseases=num_disease, num_drugs=num_drug, pyg=pyg)


    # test 
    for param in gan.discriminator.parameters():
        param.requires_grad = True
    
    # Define the loss function and optimizer for the discriminator
    d_criterion = nn.BCELoss()
    d_optimizer = torch.optim.Adam(gan.discriminator.parameters(), lr=0.001)

    # Define the optimizer for the generator
    g_optimizer = torch.optim.Adam(gan.generator.parameters(), lr=0.001)

    # Train the discriminator and generator in alternating steps
    for epoch in range(num_epochs):
        for i in range(d_epochs):
            # Train the discriminator
            gan.discriminator.train()
            gan.generator.eval()
            d_optimizer.zero_grad()
            fake_edge_index, _, _  = gan.generator(batch_size=batch_size)

            # Generate a batch of real samples
            real_edge_index_random_idx = torch.randint(low=0, high=real_edge_index.shape[1], size=(batch_size,))

            real_edge_index_batch = real_edge_index[:, real_edge_index_random_idx]



            # Compute the discriminator's predictions on the fake and real samples
            # maybe add detach() here to fake_edge_index to prevent backpropagation 
            d_fake = gan.discriminator(fake_edge_index) 
            d_real = gan.discriminator(real_edge_index_batch)
            rounded_d_fake = torch.round(d_fake)
            rounded_d_real = torch.round(d_real)

            # calculate the accuracy of the discriminator
            correct_fakes = torch.sum(rounded_d_fake == torch.zeros_like(d_fake)).item() 
            correct_reals = torch.sum(rounded_d_real == torch.ones_like(d_real)).item() 

            d_acc = (correct_fakes + correct_reals) / (d_fake.size(0) + d_real.size(0))

            # Compute the discriminator's loss
            d_fake_loss = d_criterion(d_fake, torch.zeros_like(d_fake))
            d_real_loss = d_criterion(d_real, torch.ones_like(d_real))
            

            d_loss = d_fake_loss + d_real_loss
            
            # Backpropagate and update the discriminator's parameters
            d_loss.backward()
            d_optimizer.step()
            print(f"d_loss: {d_loss}, d_acc: {d_acc}")
                


        for i in range(g_epochs):

            gan.generator.train()
            gan.discriminator.eval()
            # Generate a batch of fake samples
            fake_edge_index, policies, disease_idx = gan.generator(batch_size=batch_size)

            #log_policies = torch.log(policies)
            # Compute the discriminator's predictions on the fake samples
            d_out = gan.discriminator(fake_edge_index)

            # scale the reward so the reward has range -1 to 1
        
            scaled_d_out = d_out * 2 - 1
            
            # There are a total of 1308 diseases, so we need to convert the disease_idx to a one-hot vector
            temp = torch.zeros((len(disease_idx), 1308))

            # set the appropriate element in each row to 1
            for i, index in enumerate(disease_idx):
                temp[i, index] = 1

            temp = temp * scaled_d_out

            # find the dot product of the policies and the reward
            reward = torch.sum(temp * -policies, dim=1)

            rounded_d_fake = torch.round(d_out)

            # calculate the accuracy of the generatorF
            correct_fakes = int(torch.sum(rounded_d_fake).item())
            g_acc = correct_fakes / reward.size(0)
            print(f"correct_fakes: {correct_fakes}, g_acc: {g_acc}")
            from time import sleep
            sleep(0.5)

            
            # Compute the generator's loss
            g_loss = torch.mean(reward)
            
            
            # Backpropagate and update the generator's parameters
            g_optimizer.zero_grad()
            g_loss.backward()
            # print the gradients of the generator
            #print(gan.generator.net[0].weight.grad)
            g_optimizer.step()
            if i % 50 == 0:
                print(f"g_loss: {g_loss.item()} g_acc: {g_acc}")

                
        if epoch % 1 == 0:
            g_data["overall_epoch"].append(epoch)
            d_data["overall_epoch"].append(epoch)

            g_data["sub_epoch"].append(i)
            d_data["sub_epoch"].append(i)

            g_data["loss"].append(g_loss.item())
            d_data["loss"].append(d_loss.item())

            g_data["acc"].append(g_acc)
            d_data["acc"].append(d_acc)


        

        """ if epoch % 50 == 0:

            new_edge_index = gan.generator.generate_edges(edge_index, batch_size=len(edge_index[0]))

            x = new_edge_index[0]

            x = x.tolist()
            x = [int(i) for i in x]
            x = torch.tensor(x)

            y = new_edge_index[1]
            y = y.tolist()
            y = [int(i) for i in y]
            y = torch.tensor(y)


            drug_degree = degree(x, len(pyg["drug"]["node_id"]))
            disease_degree = degree(y, len(pyg["disease"]["node_id"]))

            dgf.draw_degree_dist([drug_degree.tolist(), disease_degree.tolist()], epoch=epoch, names=["drug", "disease"])
 """

    # convert the json array to csv and output it

    d_df = pd.DataFrame(d_data)
    g_df = pd.DataFrame(g_data)
    d_df.to_csv("d.csv")
    g_df.to_csv("g.csv")

if __name__ == "__main__":

    lr  = 0.01
    hidden_channels = 100
    disjoint_train_ratio = 0.3
    neg_sampling_ratio = 0
    batch_size = 64
    size_gnn = 5
    size_nn = 5
    num_epochs = 2
    num_val = 0.1
    num_test = 0.2
    add_negative_train_samples = False
    num_neighbors = [20, 10]
    shuffle = False
    aggr = 'max'
    early_stopping_patience = 2
    is_bipartite = True

    train_gan()

    """     pyg = get_pyg(True)


    drug_degree = degree(pyg["drug", "may_treat", "disease"].edge_index[0], len(pyg["drug"]["node_id"]))
    disease_degree = degree(pyg["drug", "may_treat", "disease"].edge_index[1], len(pyg["disease"]["node_id"]))



    dgf.draw_degree_dist([drug_degree.tolist(), disease_degree.tolist()]) """


    






















