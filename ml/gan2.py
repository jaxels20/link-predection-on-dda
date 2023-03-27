import sys
sys.path.append("..")
import torch
import torch.nn as nn
import torch.optim as optim
from med_rt_parser.pytorch_loader import get_pyg

num_epochs = 5
num_batches = 2
batch_size = 128
num_negative_edges = 2

# Define the generator and discriminator models
class Generator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Generator, self).__init__()
        self.linear1 = nn.Linear(input_size, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = torch.sigmoid(self.linear3(x))
        return x

class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        self.linear1 = nn.Linear(input_size, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = torch.sigmoid(self.linear3(x))
        return x

# Define the graph and positive edges
graph = get_pyg()
print(graph)

positive_edges = graph["drug", "may_treat", "disease"].edge_index # Generate positive edges using any suitable method
positive_edges = torch.tensor(list(zip(positive_edges[0], positive_edges[1])))
# Define the input noise vector size
noise_size = 10

# Define the generator and discriminator models
generator = Generator(noise_size, graph.num_nodes) # Output size should be the number of nodes in the graph
discriminator = Discriminator(graph.num_nodes) # Input size should be the number of nodes in the graph

# Define the loss function and optimizer
criterion = nn.BCELoss()
generator_optimizer = optim.Adam(generator.parameters(), lr=0.0002)
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)

# Train the GAN
for epoch in range(num_epochs):
    for i in range(num_batches):
        # Generate random noise vectors and use the generator to generate negative edges
        noise = torch.randn(batch_size, noise_size)
        negative_edges = generator(noise)

        print("positive_edges", positive_edges)
        print("negative_edges", negative_edges)
        
        # Concatenate the positive and negative edges and create labels for the discriminator
        edges = torch.cat((positive_edges, negative_edges))
        labels = torch.cat((torch.ones(batch_size), torch.zeros(batch_size)))

        # Train the discriminator
        discriminator_optimizer.zero_grad()
        outputs = discriminator(edges)
        discriminator_loss = criterion(outputs, labels)
        discriminator_loss.backward()
        discriminator_optimizer.step()

        # Train the generator
        generator_optimizer.zero_grad()
        noise = torch.randn(batch_size, noise_size)
        fake_edges = generator(noise)
        outputs = discriminator(fake_edges)
        generator_loss = criterion(outputs, torch.ones(batch_size))
        generator_loss.backward()
        generator_optimizer.step()

    # Print the loss for this epoch
    print(f"Epoch [{epoch+1}/{num_epochs}], Discriminator Loss: {discriminator_loss.item():.4f}, Generator Loss: {generator_loss.item():.4f}")

# Generate negative edges
with torch.no_grad():
    noise = torch.randn(num_negative_edges, noise_size)
    negative_edges = generator(noise)

# Add the negative edges to the graph
for i in range(num_negative_edges):
    src = int(negative_edges[i][0])
    dst = int(negative_edges[i][1])
    if not graph.has_edge(src, dst):
        graph.add_edge(src, dst)

# Print the number of edges in the graph after adding the negative edges
print(f"Number of edges in the graph: {graph.number_of_edges()}")