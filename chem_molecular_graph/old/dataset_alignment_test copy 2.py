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

import enum
import glob
import random
import numpy as np
import torch
from torch import optim
from torch.cpu import is_available
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCN, global_mean_pool, GCNConv
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


SEED = 42
TARGET_INDEXES = list(range(19))  # iterate over all QM9 feature columns (0-18)
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

misaligned_small_candidates = [
    i for i, data in enumerate(dataset)
    if 8 <= data.num_nodes <= 9   # pick a range as far below downstream as your current misaligned (22-23) is above
]

print(len(downstream_candidates))
print(len(aligned_candidates))
print(len(misaligned_candidates))


# shuffle via random number generator
rng = np.random.default_rng(SEED)

rng.shuffle(downstream_candidates)
rng.shuffle(aligned_candidates)
rng.shuffle(misaligned_candidates)
rng.shuffle(misaligned_small_candidates)

N_DOWNSTREAM = 3000
N_PRETRAIN = 3000

downstream_indices = downstream_candidates[:N_DOWNSTREAM]
aligned_pretrain_indices = aligned_candidates[:N_PRETRAIN]
misaligned_pretrain_indices = misaligned_candidates[:N_PRETRAIN]
misaligned_small_pretrain_indices = misaligned_small_candidates[:N_PRETRAIN]


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
misaligned_small_pretrain_dataset = [dataset[i] for i in misaligned_small_pretrain_indices]

downstream_train_dataset = [dataset[i] for i in downstream_train_indices]
downstream_val_dataset = [dataset[i] for i in downstream_val_indices]
downstream_test_dataset = [dataset[i] for i in downstream_test_indices]


# create dataloaders
BATCH_SIZE = 64
aligned_loader = DataLoader(aligned_pretrain_dataset, batch_size=BATCH_SIZE, shuffle=True )
misaligned_loader = DataLoader(misaligned_pretrain_dataset, batch_size=BATCH_SIZE, shuffle=True)
misaligned_small_loader = DataLoader(misaligned_small_pretrain_dataset, batch_size=BATCH_SIZE, shuffle=True)

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

    



def pre_training(model, loader, optimizer, target_index, target_mean, target_std):
    total_loss = 0
    total_examples = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)
        target = (batch.y[:, target_index] - target_mean) / target_std # target column
        loss = F.mse_loss(pred, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        total_examples += batch.num_graphs
    return total_loss / total_examples

# Pretraining on aligned dataset
# aligned_model = MolecularModel(input_dim=dataset.num_node_features).to(device)
# optimizer = torch.optim.Adam(aligned_model.parameters(), lr=0.001)
# aligned_pretrain_losses = []

# for epoch in range(PRETRAIN_EPOCHS):
#     loss = pre_training(aligned_model, aligned_loader, optimizer)
#     aligned_pretrain_losses.append(loss)

#     print(
#         f"Aligned pretraining "
#         f"Epoch {epoch:02d} | "
#         f"Loss: {loss:.4f}"
#     )


# # Pretraining on misaligned dataset
# misaligned_model = MolecularModel(input_dim=dataset.num_node_features).to(device)
# optimizer = torch.optim.Adam(misaligned_model.parameters(), lr=0.001)
# misaligned_pretrain_losses = []

# for epoch in range(PRETRAIN_EPOCHS):
#     loss = pre_training(misaligned_model, misaligned_loader, optimizer)
#     misaligned_pretrain_losses.append(loss)

#     print(
#         f"Misaligned pretraining "
#         f"Epoch {epoch:02d} | "
#         f"Loss: {loss:.4f}"
#     )


def finetune(model, loader, optimizer, target_index, target_mean, target_std):
    model.train()
    total_loss = 0
    total_examples = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)
        target = (batch.y[:, target_index] - target_mean) / target_std
        loss = F.mse_loss(pred, target)
        loss.backward()
        optimizer.step()

        total_loss += (loss.item() * batch.num_graphs)
        total_examples += batch.num_graphs

    return total_loss / total_examples

@torch.no_grad()
def evaluate(model, loader, target_index, target_mean, target_std):
    model.eval()
    total_absolute_error = 0
    total_examples = 0

    for batch in loader:
        batch = batch.to(device)
        pred_norm = model(batch)
        pred_real = pred_norm * target_std + target_mean
        target_real = batch.y[:, target_index]
        absolute_error = torch.abs(pred_real - target_real)
        total_absolute_error += absolute_error.sum().item()
        total_examples += batch.num_graphs

    return total_absolute_error / total_examples


# Fine-tune aligned model
# aligned_optimizer = torch.optim.Adam(aligned_model.parameters(), lr=0.01)

# aligned_history = {"train_loss": [], "val_mae": []}

# for epoch in range(FINETUNE_EPOCHS):
#     train_loss = finetune(aligned_model, train_loader, aligned_optimizer)

#     val_mae = evaluate(aligned_model, val_loader)

#     aligned_history["train_loss"].append(train_loss)
#     aligned_history["val_mae"].append(val_mae)

#     print(
#         f"Aligned | "
#         f"Epoch {epoch:02d} | "
#         f"Loss: {train_loss:.4f} | "
#         f"Val MAE: {val_mae:.4f}"
#     )


# Fine-tune misaligned model
# misaligned_optimizer = torch.optim.Adam(misaligned_model.parameters(), lr=0.01)

# misaligned_history = {"train_loss": [], "val_mae": []}

# for epoch in range(FINETUNE_EPOCHS):
#     train_loss = finetune(misaligned_model, train_loader, misaligned_optimizer)

#     val_mae = evaluate(misaligned_model, val_loader)

#     misaligned_history["train_loss"].append(train_loss)
#     misaligned_history["val_mae"].append(val_mae)

#     print(
#         f"MisAligned | "
#         f"Epoch {epoch:02d} | "
#         f"Loss: {train_loss:.4f} | "
#         f"Val MAE: {val_mae:.4f}"
#     )


# aligned_test_mae = evaluate(
#     aligned_model,
#     test_loader
# )

# misaligned_test_mae = evaluate(
#     misaligned_model,
#     test_loader
# )

# print(
#     f"Aligned pretraining MAE: "
#     f"{aligned_test_mae:.4f}"
# )

# print(
#     f"Misaligned pretraining MAE: "
#     f"{misaligned_test_mae:.4f}"
# )

# difference = (
#     misaligned_test_mae
#     - aligned_test_mae
# )

# relative_difference = (
#     difference
#     / misaligned_test_mae
# ) * 100

# print(
#     f"Aligned MAE:     {aligned_test_mae:.4f}"
# )

# print(
#     f"Misaligned MAE:   {misaligned_test_mae:.4f}"
# )

# print(
#     f"Absolute difference: {difference:.4f}"
# )

# print(
#     f"Relative difference: {relative_difference:.2f}%"
# )

def run_experiment(pretrain_loader, condition_name, seed, target_index, target_mean, target_std):

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = MolecularModel(input_dim=dataset.num_node_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    pretrain_losses = []

    for epoch in range(PRETRAIN_EPOCHS):
        loss = pre_training(model, pretrain_loader, optimizer, target_index, target_mean, target_std)
        pretrain_losses.append(loss)

        print(
            f"[{condition_name} | seed {seed}] "
            f"Pretrain Epoch {epoch:02d} | "
            f"Loss: {loss:.4f}"
        )

    ft_optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    finetune_history = {"train_loss": [], "val_mae": []}

    for epoch in range(FINETUNE_EPOCHS):
        train_loss = finetune(model, train_loader, ft_optimizer, target_index, target_mean, target_std)
        val_mae = evaluate(model, val_loader, target_index, target_mean, target_std)

        finetune_history["train_loss"].append(train_loss)
        finetune_history["val_mae"].append(val_mae)

        print(
            f"[{condition_name} | seed {seed}] "
            f"Finetune Epoch {epoch:02d} | "
            f"Loss: {train_loss:.4f} | "
            f"Val MAE: {val_mae:.4f}"
        )

    test_mae = evaluate(model, test_loader, target_index, target_mean, target_std)

    print(
        f"[{condition_name} | seed {seed}] "
        f"Test MAE: {test_mae:.4f}\n"
    )

    return {
        "condition": condition_name,
        "seed": seed,
        "test_mae": test_mae,
        "pretrain_losses": pretrain_losses,
        "finetune_history": finetune_history,
    }


SEEDS = [42, 1, 7, 123, 2024]
conditions = {
    "aligned": aligned_loader,
    "misaligned_larger": misaligned_loader,
    "misaligned_smaller": misaligned_small_loader,
}

import os
import pandas as pd

cross_all = []
for target_index in TARGET_INDEXES:
    # fit scaler per target feature (mean/std over the downstream train split)
    train_targets = np.array([
        dataset[i].y[0, target_index].item()
        for i in downstream_train_indices
    ]).reshape(-1, 1)

    scaler = StandardScaler()
    scaler.fit(train_targets)

    target_mean = torch.tensor(scaler.mean_[0], dtype=torch.float32)
    target_std = torch.tensor(scaler.scale_[0], dtype=torch.float32)

    out_dir = f"figures/molecule_graphs/target_{target_index}"
    os.makedirs(out_dir, exist_ok=True)

    results = []
    for seed in SEEDS:
        for name, loader in conditions.items():
            result = run_experiment(loader, name, seed, target_index, target_mean, target_std)
            results.append(result)
            print(result)
            result["target_index"] = target_index

    df = pd.DataFrame(results)
    cross_all.append(df)
    summary = df.groupby("condition")["test_mae"].agg(["mean", "std"])
    print(summary)

    plt.figure(figsize=(7, 5))
    plt.bar(summary.index, summary["mean"], yerr=summary["std"], capsize=5)
    plt.ylabel("Test MAE")
    plt.title(f"Effect of Pretraining Alignment Direction - target {target_index} (mean ± std over 5 seeds)")
    plt.savefig(f"{out_dir}/alignment_direction_comparison.png")
    plt.show()

    df.boxplot(column="test_mae", by="condition")
    plt.title(f"Test MAE distribution by pretraining condition - target {target_index}")
    plt.suptitle("")
    plt.ylabel("Test MAE")
    plt.savefig(f"{out_dir}/alignment_boxplot.png")
    plt.show()

# cross-feature comparison graphs across all targets
all_df = pd.concat(cross_all, ignore_index=True)
cross_summary = all_df.groupby(["target_index", "condition"])["test_mae"].agg(["mean", "std"]).reset_index()
overall = cross_summary.pivot(index="target_index", columns="condition", values="mean")

plt.figure(figsize=(10, 6))
for cond in overall.columns:
    plt.plot(overall.index, overall[cond], marker="o", label=cond)
plt.xlabel("Target index")
plt.ylabel("Mean Test MAE")
plt.title("Cross-feature comparison: Test MAE by target index and condition")
plt.legend()
plt.grid(True)
plt.savefig("figures/molecule_graphs/cross_feature_comparison_line.png")
plt.show()

plt.figure(figsize=(12, 6))
overall.plot(kind="bar", rot=0, figsize=(12, 6))
plt.ylabel("Mean Test MAE")
plt.xlabel("Target index")
plt.title("Cross-feature comparison: Test MAE by target index and condition")
plt.legend(title="Condition")
plt.tight_layout()
plt.savefig("figures/molecule_graphs/cross_feature_comparison_bars.png")
plt.show()

# relabel indexes with QM9 property names for readability
qm9_props = ["mu", "alpha", "HOMO", "LUMO", "gap", "R2", "ZPVE", "U0", "U", "H", "G", "Cv", "U0_atom", "U_atom", "H_atom", "G_atom", "A", "B", "C"]
overall.index = [qm9_props[i] if i < len(qm9_props) else i for i in overall.index]
plt.figure(figsize=(12, 6))
overall.plot(kind="bar", rot=45, figsize=(12, 6))
plt.ylabel("Mean Test MAE")
plt.xlabel("Target property")
plt.title("Cross-feature comparison labeled by QM9 properties")
plt.legend(title="Condition")
plt.tight_layout()
plt.savefig("figures/molecule_graphs/cross_feature_comparison_labeled.png")
plt.show()

# import matplotlib.pyplot as plt

# epochs = range(
#     0,
#     FINETUNE_EPOCHS
# )

# plt.figure(figsize=(8, 5))

# plt.plot(
#     epochs,
#     aligned_history["val_mae"],
#     label="Aligned pretraining"
# )

# plt.plot(
#     epochs,
#     misaligned_history["val_mae"],
#     label="Misaligned pretraining"
# )

# plt.xlabel("Fine-tuning epoch")
# plt.ylabel("Validation MAE")
# plt.title("Effect of pretraining-data alignment")
# plt.legend()
# plt.grid(True)
# plt.savefig("figures/molecule_graphs/loss over time.png")
# plt.show()


# labels = [
#     "Aligned",
#     "Misaligned"
# ]

# mae_values = [
#     aligned_test_mae,
#     misaligned_test_mae
# ]

# plt.figure(figsize=(7, 5))

# plt.bar(
#     labels,
#     mae_values
# )

# plt.ylabel("Test MAE")
# plt.title("Downstream performance after pretraining")
# plt.savefig("figures/molecule_graphs/downstream_performance_after_pretraining.png")
# plt.show()


# aligned_sizes = [
#     dataset[i].num_nodes
#     for i in aligned_pretrain_indices
# ]

# misaligned_sizes = [
#     dataset[i].num_nodes
#     for i in misaligned_pretrain_indices
# ]

# downstream_sizes = [
#     dataset[i].num_nodes
#     for i in downstream_indices
# ]

# plt.figure(figsize=(8, 5))

# plt.hist(
#     aligned_sizes,
#     bins=np.arange(
#         min(num_atoms) - 0.5,
#         max(num_atoms) + 1.5,
#         1
#     ),
#     alpha=0.6,
#     label="Aligned pretraining"
# )

# plt.hist(
#     misaligned_sizes,
#     bins=np.arange(
#         min(num_atoms) - 0.5,
#         max(num_atoms) + 1.5,
#         1
#     ),
#     alpha=0.6,
#     label="Misaligned pretraining"
# )

# plt.hist(
#     downstream_sizes,
#     bins=np.arange(
#         min(num_atoms) - 0.5,
#         max(num_atoms) + 1.5,
#         1
#     ),
#     alpha=0.6,
#     label="Downstream"
# )

# plt.xlabel("Number of atoms")
# plt.ylabel("Number of molecules")
# plt.title("Molecular-size distributions")
# plt.legend()
# plt.savefig("figures/molecule_graphs/molecule_size_distributions.png")
# plt.show()

