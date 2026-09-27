"""
Downstream task: predict U0, the internal energy at 0 K.
Split molecules by size.
Aligned pretraining set: molecules with similar molecular sizes to the downstream set.
Less-aligned pretraining set: molecules with substantially different sizes.
Same GCN.
Same number of pretraining molecules.
Same epochs.
Same downstream data.
Compare downstream MAE.
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCN, global_mean_pool, GCNConv
import matplotlib.pyplot as plt


SEED = 42
TARGET_INDEX = 7 # selected feature column "U0"to evaluate
PRETRAIN_EPOCHS = 5
FINETUNE_EPOCHS = 5

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load Dataset
dataset = QM9(
    root="dataset/QM9"
)

num_atoms = np.array([
    data.num_nodes
    for data in dataset
])


"""
downstream: 15-16 atoms   # for fine-tuning
aligned:    17-18 atoms   # for pretraining
misaligned: 22-23 atoms   # for pretraining
"""

# storing indices as candiates -> shuffle and trim candidates to 3000
downstream_candidates = [
    i for i, data in enumerate(dataset) 
    if 15 <= data.num_nodes <=16 
]

aligned_candidates = [
    i for i, data in enumerate(dataset) 
    if 17 <= data.num_nodes <=18 
]

misaligned_candidates = [
    i for i, data in enumerate(dataset) 
    if 22 <= data.num_nodes <=23 
]

print(len(downstream_candidates))
print(len(aligned_candidates))
print(len(misaligned_candidates))


# shuffle via random number generator
rng = np.random.default_rng(SEED)

rng.shuffle(downstream_candidates)
rng.shuffle(aligned_candidates)
rng.shuffle(misaligned_candidates)

N_DOWNSTREAM = 3000
N_PRETRAIN = 3000

downstream_indices = downstream_candidates[:N_DOWNSTREAM]
aligned_pretrain_indices = aligned_candidates[:N_PRETRAIN]
misaligned_pretrain_indices = misaligned_candidates[:N_PRETRAIN]

# creating splits
rng.shuffle(downstream_indices)

n_train = int(0.7 * len(downstream_indices))
n_val = int(0.15 * len(downstream_indices))

downstream_train_indices = downstream_indices[:n_train]

downstream_val_indices = downstream_indices[
    n_train:n_train + n_val
]

downstream_test_indices = downstream_indices[
    n_train + n_val:
]

print("Downstream train:", len(downstream_train_indices))
print("Downstream val:", len(downstream_val_indices))
print("Downstream test:", len(downstream_test_indices))

# creating datsets
aligned_pretrain_dataset = [dataset[i] for i in aligned_pretrain_indices]
misaligned_pretrain_dataset = [dataset[i] for i in misaligned_pretrain_indices]

downstream_train_dataset = [dataset[i] for i in downstream_train_indices]
downstream_val_dataset = [dataset[i] for i in downstream_val_indices]
downstream_test_dataset = [dataset[i] for i in downstream_test_indices]


# create dataloaders
BATCH_SIZE = 64
aligned_loader = DataLoader(aligned_pretrain_dataset, batch_size=BATCH_SIZE, shuffle=True )
misaligned_loader = DataLoader(misaligned_pretrain_dataset, batch_size=BATCH_SIZE, shuffle=True)

train_loader = DataLoader(downstream_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(downstream_val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(downstream_test_dataset, batch_size=BATCH_SIZE,shuffle=False)

# define GCN
class GCN(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128, embedding_dim=128):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, embedding_dim)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)

        # As it is a graph level task so we need mean pooling (converting node embeddings to graph eembedding)
        # batch here is 1D gaint graph, to achieve parallelization
        x = global_mean_pool(x, batch)
        return x


class MolecularModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, embedding_dim=128):
        super().__init__()
        self.gnn = GCN(input_dim=input_dim, hidden_dim=hidden_dim, embedding_dim=embedding_dim)
        self.head = nn.Linear(embedding_dim, 1)
    def forward(self, data):

        z = self.gnn(data.x, data.edge_index, data.batch)
        out = self.head(z)
        return out.squeeze(-1)

    



def pre_training(model, loader, optimizer):
    total_loss = 0
    total_examples = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)
        target = batch.y[:, TARGET_INDEX] # U0 column
        loss = F.mse_loss(pred, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        total_examples += batch.num_graphs
    return total_loss / total_examples

# Pretraining on aligned dataset
aligned_model = MolecularModel(input_dim=dataset.num_node_features).to(device)
optimizer = torch.optim.Adam(aligned_model.parameters(), lr=0.001)
aligned_pretrain_losses = []

for epoch in range(PRETRAIN_EPOCHS):
    loss = pre_training(aligned_model, aligned_loader, optimizer)
    aligned_pretrain_losses.append(loss)

    print(
        f"Aligned pretraining "
        f"Epoch {epoch:02d} | "
        f"Loss: {loss:.4f}"
    )


# Pretraining on misaligned dataset
misaligned_model = MolecularModel(input_dim=dataset.num_node_features).to(device)
optimizer = torch.optim.Adam(misaligned_model.parameters(), lr=0.001)
misaligned_pretrain_losses = []

for epoch in range(PRETRAIN_EPOCHS):
    loss = pre_training(misaligned_model, misaligned_loader, optimizer)
    misaligned_pretrain_losses.append(loss)

    print(
        f"Misaligned pretraining "
        f"Epoch {epoch:02d} | "
        f"Loss: {loss:.4f}"
    )


def finetune(model, loader, optimizer):
    model.train()
    total_loss = 0
    total_examples = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)
        target = batch.y[:, TARGET_INDEX]
        loss = F.mse_loss(pred, target)
        loss.backward()
        optimizer.step()

        total_loss += (loss.item() * batch.num_graphs)
        total_examples += batch.num_graphs

    return total_loss / total_examples

@torch.no_grad()
def evaluate(model, loader):
    model.eval()

    total_absolute_error = 0
    total_examples = 0

    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        target = batch.y[:, TARGET_INDEX]
        absolute_error = torch.abs(pred - target)
        total_absolute_error += (absolute_error.sum().item())
        total_examples += batch.num_graphs

    mae = (total_absolute_error / total_examples)
    return mae


# Fine-tune aligned model
aligned_optimizer = torch.optim.Adam(aligned_model.parameters(), lr=0.01)

aligned_history = {"train_loss": [], "val_mae": []}

for epoch in range(FINETUNE_EPOCHS):
    train_loss = finetune(aligned_model, train_loader, aligned_optimizer)

    val_mae = evaluate(aligned_model, val_loader)

    aligned_history["train_loss"].append(train_loss)
    aligned_history["val_mae"].append(val_mae)

    print(
        f"Aligned | "
        f"Epoch {epoch:02d} | "
        f"Loss: {train_loss:.4f} | "
        f"Val MAE: {val_mae:.4f}"
    )


# Fine-tune misaligned model
misaligned_optimizer = torch.optim.Adam(misaligned_model.parameters(), lr=0.01)

misaligned_history = {"train_loss": [], "val_mae": []}

for epoch in range(FINETUNE_EPOCHS):
    train_loss = finetune(misaligned_model, train_loader, misaligned_optimizer)

    val_mae = evaluate(misaligned_model, val_loader)

    misaligned_history["train_loss"].append(train_loss)
    misaligned_history["val_mae"].append(val_mae)

    print(
        f"MisAligned | "
        f"Epoch {epoch:02d} | "
        f"Loss: {train_loss:.4f} | "
        f"Val MAE: {val_mae:.4f}"
    )


aligned_test_mae = evaluate(
    aligned_model,
    test_loader
)

misaligned_test_mae = evaluate(
    misaligned_model,
    test_loader
)

print(
    f"Aligned pretraining MAE: "
    f"{aligned_test_mae:.4f}"
)

print(
    f"Misaligned pretraining MAE: "
    f"{misaligned_test_mae:.4f}"
)

difference = (
    misaligned_test_mae
    - aligned_test_mae
)

relative_difference = (
    difference
    / misaligned_test_mae
) * 100

print(
    f"Aligned MAE:     {aligned_test_mae:.4f}"
)

print(
    f"Misaligned MAE:   {misaligned_test_mae:.4f}"
)

print(
    f"Absolute difference: {difference:.4f}"
)

print(
    f"Relative difference: {relative_difference:.2f}%"
)

import matplotlib.pyplot as plt

epochs = range(
    0,
    FINETUNE_EPOCHS
)

plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    aligned_history["val_mae"],
    label="Aligned pretraining"
)

plt.plot(
    epochs,
    misaligned_history["val_mae"],
    label="Misaligned pretraining"
)

plt.xlabel("Fine-tuning epoch")
plt.ylabel("Validation MAE")
plt.title("Effect of pretraining-data alignment")
plt.legend()
plt.grid(True)
plt.savefig("figures/molecule_graphs/loss over time.png")
plt.show()


labels = [
    "Aligned",
    "Misaligned"
]

mae_values = [
    aligned_test_mae,
    misaligned_test_mae
]

plt.figure(figsize=(7, 5))

plt.bar(
    labels,
    mae_values
)

plt.ylabel("Test MAE")
plt.title("Downstream performance after pretraining")
plt.savefig("figures/molecule_graphs/downstream_performance_after_pretraining.png")
plt.show()


aligned_sizes = [
    dataset[i].num_nodes
    for i in aligned_pretrain_indices
]

misaligned_sizes = [
    dataset[i].num_nodes
    for i in misaligned_pretrain_indices
]

downstream_sizes = [
    dataset[i].num_nodes
    for i in downstream_indices
]

plt.figure(figsize=(8, 5))

plt.hist(
    aligned_sizes,
    bins=np.arange(
        min(num_atoms) - 0.5,
        max(num_atoms) + 1.5,
        1
    ),
    alpha=0.6,
    label="Aligned pretraining"
)

plt.hist(
    misaligned_sizes,
    bins=np.arange(
        min(num_atoms) - 0.5,
        max(num_atoms) + 1.5,
        1
    ),
    alpha=0.6,
    label="Misaligned pretraining"
)

plt.hist(
    downstream_sizes,
    bins=np.arange(
        min(num_atoms) - 0.5,
        max(num_atoms) + 1.5,
        1
    ),
    alpha=0.6,
    label="Downstream"
)

plt.xlabel("Number of atoms")
plt.ylabel("Number of molecules")
plt.title("Molecular-size distributions")
plt.legend()
plt.savefig("figures/molecule_graphs/molecule_size_distributions.png")
plt.show()

