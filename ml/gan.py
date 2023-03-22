import sys
sys.path.append("..")
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from med_rt_parser.pytorch_loader import get_pyg


class Generator(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, in_dim)

    def forward(self, z):
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

class Discriminator(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.fc1(x).squeeze()
        return x

def train_gan(train_data, num_epochs, batch_size, lr_G, lr_D, hidden_dim, noise_dim, NEG_SAMPLING_RATIO):
    # Initialize generator and discriminator
    generator = Generator(noise_dim, hidden_dim)
    discriminator = Discriminator(train_data.num_node_features["drug"], hidden_dim)

    # Initialize optimizers and loss criterion
    optimizer_G = optim.Adam(generator.parameters(), lr=lr_G, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr_D, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()

    # Create data loader
    
    # Get indices of drugs and diseases nodes
    drug_nodes = (train_data["node_type"] == "drug").nonzero(as_tuple=False).squeeze()
    disease_nodes = (train_data["node_type"] == "disease").nonzero(as_tuple=False).squeeze()

    # Training loop
    for epoch in range(num_epochs):
        # Sample negative edges between drugs and diseases
        num_neg_samples = int(train_data.size(0) * NEG_SAMPLING_RATIO)
        neg_drug_idx = torch.randint(0, drug_nodes.size(0), (num_neg_samples,))
        neg_disease_idx = torch.randint(0, disease_nodes.size(0), (num_neg_samples,))
        neg_edges = torch.stack([drug_nodes[neg_drug_idx], disease_nodes[neg_disease_idx]])

        # Train discriminator
        optimizer_D.zero_grad()
        real_labels = torch.ones(train_data.size(0))
        fake_labels = torch.zeros(neg_edges.size(1))
        pos_scores = discriminator(train_data.x, train_data.edge_index[:, train_data[1]])
        neg_scores = discriminator(train_data.x, neg_edges)
        real_loss = criterion(pos_scores, real_labels)
        fake_loss = criterion(neg_scores, fake_labels)
        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_D.step()

        # Train generator
        optimizer_G.zero_grad()
        z = torch.randn((batch_size, noise_dim))
        fake_edges = generator(z)
        fake_scores = discriminator(train_data.x, train_data.edge_index[:, fake_edges[1]])
        g_loss = criterion(fake_scores, real_labels)
        g_loss.backward()
        optimizer_G.step()

        # Print loss information
        print('Epoch [{}/{}], D_Loss: {:.4f}, G_Loss: {:.4f}'.format(epoch+1, num_epochs, d_loss.item(), g_loss.item()))

        
if __name__ == "__main__":
    # Hyperparameters
    num_epochs = 100
    batch_size = 64 
    lr_G = 0.0001
    lr_D = 0.0001
    hidden_dim = 128
    noise_dim = 128
    NEG_SAMPLING_RATIO = 1

    # Load data
    train_data = get_pyg()
    print(train_data)

    # Train GAN
    train_gan(train_data, num_epochs, batch_size, lr_G, lr_D, hidden_dim, noise_dim, NEG_SAMPLING_RATIO)


