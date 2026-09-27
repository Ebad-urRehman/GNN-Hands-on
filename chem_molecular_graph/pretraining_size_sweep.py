"""
Downstream task: predict a target property (iterate over all QM9 features).
Split molecules by size.
Pretraining groups sweep symmetrically around the downstream size (15-16 atoms):
    10_11 (10-11), 12_13 (12-13), 17_18 (17-18), 19_20 (19-20)
All pretraining groups use the SAME number of molecules.
Same GCN.
Same epochs.
Same downstream data.
Compare downstream MAE across the size-swept pretraining groups.
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
downstream: 15-16 atoms   # for fine-tuning (unchanged)
pretrain groups sweep around downstream:
    10_11: 10-11 atoms
    12_13: 12-13 atoms
    17_18: 17-18 atoms
    19_20: 19-20 atoms
    21_22: 21-22 atoms
    22_23: 22-23 atoms
    23_24: 23-24 atoms
    25_26: 25-26 atoms
"""

# storing indices as candidates -> shuffle and trim candidates to an equal size
downstream_candidates = [
    i for i, data in enumerate(dataset) 
    if 15 <= data.num_nodes <=16 
]

group10_11_candidates = [
    i for i, data in enumerate(dataset) 
    if 10 <= data.num_nodes <=11 
]

group12_13_candidates = [
    i for i, data in enumerate(dataset) 
    if 12 <= data.num_nodes <=13 
]

group17_18_candidates = [
    i for i, data in enumerate(dataset) 
    if 17 <= data.num_nodes <=18 
]

group19_20_candidates = [
    i for i, data in enumerate(dataset) 
    if 19 <= data.num_nodes <=20 
]

group21_22_candidates = [
    i for i, data in enumerate(dataset) 
    if 21 <= data.num_nodes <=22 
]

group22_23_candidates = [
    i for i, data in enumerate(dataset) 
    if 22 <= data.num_nodes <=23 
]

group23_24_candidates = [
    i for i, data in enumerate(dataset) 
    if 23 <= data.num_nodes <=24 
]

group25_26_candidates = [
    i for i, data in enumerate(dataset) 
    if 25 <= data.num_nodes <=26 
]

print(len(downstream_candidates))
print(len(group10_11_candidates))
print(len(group12_13_candidates))
print(len(group17_18_candidates))
print(len(group19_20_candidates))
print(len(group21_22_candidates))
print(len(group22_23_candidates))
print(len(group23_24_candidates))
print(len(group25_26_candidates))


# shuffle via random number generator
rng = np.random.default_rng(SEED)

rng.shuffle(downstream_candidates)
rng.shuffle(group10_11_candidates)
rng.shuffle(group12_13_candidates)
rng.shuffle(group17_18_candidates)
rng.shuffle(group19_20_candidates)
rng.shuffle(group21_22_candidates)
rng.shuffle(group22_23_candidates)
rng.shuffle(group23_24_candidates)
rng.shuffle(group25_26_candidates)

N_DOWNSTREAM = 3000
# cap every pretraining group to the smallest pool (10_11) so all groups are equal-sized
N_PRETRAIN = min(len(group10_11_candidates), len(group12_13_candidates),
                 len(group17_18_candidates), len(group19_20_candidates),
                 len(group21_22_candidates), len(group22_23_candidates),
                 len(group23_24_candidates), len(group25_26_candidates))

downstream_indices = downstream_candidates[:N_DOWNSTREAM]
group10_11_pretrain_indices = group10_11_candidates[:N_PRETRAIN]
group12_13_pretrain_indices = group12_13_candidates[:N_PRETRAIN]
group17_18_pretrain_indices = group17_18_candidates[:N_PRETRAIN]
group19_20_pretrain_indices = group19_20_candidates[:N_PRETRAIN]
group21_22_pretrain_indices = group21_22_candidates[:N_PRETRAIN]
group22_23_pretrain_indices = group22_23_candidates[:N_PRETRAIN]
group23_24_pretrain_indices = group23_24_candidates[:N_PRETRAIN]
group25_26_pretrain_indices = group25_26_candidates[:N_PRETRAIN]

# pretraining datasets keyed by group name (used in `conditions` below)
pretrain_datasets = {
    "10_11": [dataset[i] for i in group10_11_pretrain_indices],
    "12_13": [dataset[i] for i in group12_13_pretrain_indices],
    "17_18": [dataset[i] for i in group17_18_pretrain_indices],
    "19_20": [dataset[i] for i in group19_20_pretrain_indices],
    "21_22": [dataset[i] for i in group21_22_pretrain_indices],
    "22_23": [dataset[i] for i in group22_23_pretrain_indices],
    "23_24": [dataset[i] for i in group23_24_pretrain_indices],
    "25_26": [dataset[i] for i in group25_26_pretrain_indices],
}
print("N_PRETRAIN (equal per group):", N_PRETRAIN)


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
downstream_train_dataset = [dataset[i] for i in downstream_train_indices]
downstream_val_dataset = [dataset[i] for i in downstream_val_indices]
downstream_test_dataset = [dataset[i] for i in downstream_test_indices]


# create dataloaders
BATCH_SIZE = 64
pretrain_loaders = {
    name: DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    for name, ds in pretrain_datasets.items()
}

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
    "10_11": pretrain_loaders["10_11"],
    "12_13": pretrain_loaders["12_13"],
    "17_18": pretrain_loaders["17_18"],
    "19_20": pretrain_loaders["19_20"],
    "21_22": pretrain_loaders["21_22"],
    "22_23": pretrain_loaders["22_23"],
    "23_24": pretrain_loaders["23_24"],
    "25_26": pretrain_loaders["25_26"],
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
    plt.pause(3)
    plt.close()

    df.boxplot(column="test_mae", by="condition")
    plt.title(f"Test MAE distribution by pretraining condition - target {target_index}")
    plt.suptitle("")
    plt.ylabel("Test MAE")
    plt.savefig(f"{out_dir}/alignment_boxplot.png")
    plt.show()
    plt.pause(3)
    plt.close()

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



