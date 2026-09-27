import copy
import random
import numpy as np
import torch
import torch.nn as nn

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

import matplotlib.pyplot as plt



SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)



device = torch.device("cpu")

print("Device:", device)


dataset = QM9(root="./dataset/QM9")

print("Number of molecules:", len(dataset))
print("Number of node features:", dataset.num_features)
print("Number of targets:", dataset.num_classes)


TARGET = 7

print("Target index:", TARGET)

sample = dataset[0]

print("Target shape:", sample.y.shape)
print("U0:", sample.y[0, TARGET].item())

rng = torch.Generator().manual_seed(SEED)

all_indices = torch.randperm(
    len(dataset),
    generator=rng
).tolist()


# --------------------------------------------------
# Downstream dataset
# --------------------------------------------------

DOWNSTREAM_TRAIN_N = 300
DOWNSTREAM_VAL_N = 100
DOWNSTREAM_TEST_N = 100

downstream_train_idx = all_indices[
    :DOWNSTREAM_TRAIN_N
]

downstream_val_idx = all_indices[
    DOWNSTREAM_TRAIN_N:
    DOWNSTREAM_TRAIN_N + DOWNSTREAM_VAL_N
]

downstream_test_idx = all_indices[
    DOWNSTREAM_TRAIN_N + DOWNSTREAM_VAL_N:
    DOWNSTREAM_TRAIN_N + DOWNSTREAM_VAL_N + DOWNSTREAM_TEST_N
]


# --------------------------------------------------
# Candidate upstream pool
# --------------------------------------------------

candidate_start = (
    DOWNSTREAM_TRAIN_N
    + DOWNSTREAM_VAL_N
    + DOWNSTREAM_TEST_N
)

CANDIDATE_N = 3000

candidate_idx = all_indices[
    candidate_start:
    candidate_start + CANDIDATE_N
]


print("Downstream train:", len(downstream_train_idx))
print("Downstream val:", len(downstream_val_idx))
print("Downstream test:", len(downstream_test_idx))
print("Candidate pool:", len(candidate_idx))


def pyg_to_rdkit(data):
    mol = Chem.RWMol()

    # Add atoms
    for atomic_number in data.z.tolist():
        atom = Chem.Atom(int(atomic_number))
        mol.AddAtom(atom)

    # QM9 edge_attr:
    # 0 = single
    # 1 = double
    # 2 = triple
    # 3 = aromatic
    bond_types = [
        Chem.BondType.SINGLE,
        Chem.BondType.DOUBLE,
        Chem.BondType.TRIPLE,
        Chem.BondType.AROMATIC,
    ]

    added_bonds = set()

    for edge_idx in range(data.edge_index.shape[1]):

        src = int(data.edge_index[0, edge_idx])
        dst = int(data.edge_index[1, edge_idx])

        # Avoid adding both directions
        pair = tuple(sorted((src, dst)))

        if pair in added_bonds:
            continue

        added_bonds.add(pair)

        bond_type_idx = int(
            torch.argmax(data.edge_attr[edge_idx]).item()
        )

        mol.AddBond(
            src,
            dst,
            bond_types[bond_type_idx]
        )

    mol = mol.GetMol()

    try:
        Chem.SanitizeMol(mol)
    except:
        return None

    return mol


def molecular_fingerprint(data):
    mol = pyg_to_rdkit(data)

    if mol is None:
        return None

    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=2,
        nBits=1024
    )

    return fp


print("Creating downstream fingerprints...")

downstream_fps = []

for idx in downstream_train_idx:

    fp = molecular_fingerprint(dataset[idx])

    if fp is not None:
        downstream_fps.append(fp)


print(
    "Valid downstream fingerprints:",
    len(downstream_fps)
)


print("Creating candidate fingerprints...")

candidate_fps = []
valid_candidate_indices = []

for idx in candidate_idx:

    fp = molecular_fingerprint(dataset[idx])

    if fp is not None:
        candidate_fps.append(fp)
        valid_candidate_indices.append(idx)


print(
    "Valid candidate fingerprints:",
    len(candidate_fps)
)


TOP_K_SIMILAR = 5

alignment_scores = []

for candidate_fp in candidate_fps:

    similarities = []

    for downstream_fp in downstream_fps:

        score = DataStructs.TanimotoSimilarity(
            candidate_fp,
            downstream_fp
        )

        similarities.append(score)

    similarities.sort(reverse=True)

    top_scores = similarities[:TOP_K_SIMILAR]

    alignment_scores.append(
        np.mean(top_scores)
    )


alignment_scores = np.array(alignment_scores)

print("Alignment score statistics:")
print("Mean:", alignment_scores.mean())
print("Min :", alignment_scores.min())
print("Max :", alignment_scores.max())


PRETRAIN_N = 300

sorted_positions = np.argsort(
    alignment_scores
)


# Lowest similarity
misaligned_positions = sorted_positions[
    :PRETRAIN_N
]


# Highest similarity
aligned_positions = sorted_positions[
    -PRETRAIN_N:
]


aligned_idx = [
    valid_candidate_indices[i]
    for i in aligned_positions
]


misaligned_idx = [
    valid_candidate_indices[i]
    for i in misaligned_positions
]


print("Aligned molecules:", len(aligned_idx))
print("Misaligned molecules:", len(misaligned_idx))


print(
    "Mean aligned similarity:",
    alignment_scores[aligned_positions].mean()
)


print(
    "Mean misaligned similarity:",
    alignment_scores[misaligned_positions].mean()
)

plt.figure(figsize=(8, 5))

plt.hist(
    alignment_scores[aligned_positions],
    bins=20,
    alpha=0.7,
    label="Aligned"
)

plt.hist(
    alignment_scores[misaligned_positions],
    bins=20,
    alpha=0.7,
    label="Misaligned"
)

plt.xlabel("Structural similarity to downstream data")
plt.ylabel("Number of molecules")
plt.title("Pretraining Data Alignment")

plt.legend()
plt.tight_layout()
plt.show()

aligned_dataset = [
    dataset[i]
    for i in aligned_idx
]

misaligned_dataset = [
    dataset[i]
    for i in misaligned_idx
]

downstream_train = [
    dataset[i]
    for i in downstream_train_idx
]

downstream_val = [
    dataset[i]
    for i in downstream_val_idx
]

downstream_test = [
    dataset[i]
    for i in downstream_test_idx
]


print("Aligned:", len(aligned_dataset))
print("Misaligned:", len(misaligned_dataset))
print("Downstream train:", len(downstream_train))
print("Downstream val:", len(downstream_val))
print("Downstream test:", len(downstream_test))



train_targets = torch.tensor([
    float(data.y[0, TARGET])
    for data in downstream_train
])


target_mean = train_targets.mean()
target_std = train_targets.std()


print("Target mean:", target_mean.item())
print("Target std :", target_std.item())


def prepare_dataset(data_list):

    prepared = []

    for data in data_list:

        data = data.clone()

        y = data.y[0, TARGET]

        data.y_target = (
            y - target_mean
        ) / target_std

        prepared.append(data)

    return prepared


aligned_dataset = prepare_dataset(aligned_dataset)
misaligned_dataset = prepare_dataset(misaligned_dataset)

downstream_train = prepare_dataset(downstream_train)
downstream_val = prepare_dataset(downstream_val)
downstream_test = prepare_dataset(downstream_test)


BATCH_SIZE = 32

aligned_loader = DataLoader(
    aligned_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

misaligned_loader = DataLoader(
    misaligned_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

train_loader = DataLoader(
    downstream_train,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    downstream_val,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    downstream_test,
    batch_size=BATCH_SIZE,
    shuffle=False
)


class SmallGCN(nn.Module):

    def __init__(
        self,
        in_channels,
        hidden_channels=64
    ):

        super().__init__()

        self.conv1 = GCNConv(
            in_channels,
            hidden_channels
        )

        self.conv2 = GCNConv(
            hidden_channels,
            hidden_channels
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_channels, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x, edge_index, batch):

        x = self.conv1(
            x,
            edge_index
        )

        x = torch.relu(x)

        x = self.conv2(
            x,
            edge_index
        )

        x = torch.relu(x)

        x = global_mean_pool(
            x,
            batch
        )

        return self.head(x).squeeze(-1)


def train_one_epoch(
model,
loader,
optimizer
):

    model.train()

    total_loss = 0.0
    total_examples = 0

    for batch in loader:

        batch = batch.to(device)

        optimizer.zero_grad()

        pred = model(
            batch.x,
            batch.edge_index,
            batch.batch
        )

        target = batch.y_target.float()

        loss = nn.functional.mse_loss(
            pred,
            target
        )

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item()
            * batch.num_graphs
        )

        total_examples += batch.num_graphs

    return total_loss / total_examples



@torch.no_grad()
def evaluate(
    model,
    loader
):

    model.eval()

    predictions = []
    targets = []

    for batch in loader:

        batch = batch.to(device)

        pred = model(
            batch.x,
            batch.edge_index,
            batch.batch
        )

        predictions.append(
            pred.cpu()
        )

        targets.append(
            batch.y_target.cpu()
        )

    predictions = torch.cat(predictions)
    targets = torch.cat(targets)

    # Convert back to original U0 units
    predictions = (
        predictions * target_std
        + target_mean
    )

    targets = (
        targets * target_std
        + target_mean
    )

    mae = torch.mean(
        torch.abs(predictions - targets)
    )

    return mae.item()


torch.manual_seed(SEED)

base_model = SmallGCN(
    in_channels=dataset.num_features,
    hidden_channels=64
)

initial_state = copy.deepcopy(
    base_model.state_dict()
)

def pretrain(
    loader,
    initial_state,
    epochs=5
):

    model = SmallGCN(
        in_channels=dataset.num_features,
        hidden_channels=64
    ).to(device)

    model.load_state_dict(
        copy.deepcopy(initial_state)
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    history = []

    for epoch in range(epochs):

        loss = train_one_epoch(
            model,
            loader,
            optimizer
        )

        history.append(loss)

        print(
            f"Epoch {epoch:02d} | "
            f"Pretrain loss: {loss:.4f}"
        )

    return model, history


print("\n========== ALIGNED PRETRAINING ==========\n")

aligned_model, aligned_pretrain_history = pretrain(
    aligned_loader,
    initial_state,
    epochs=5
)

print("\n========== MISALIGNED PRETRAINING ==========\n")

misaligned_model, misaligned_pretrain_history = pretrain(
    misaligned_loader,
    initial_state,
    epochs=5
)

def finetune(
    pretrained_model,
    epochs=15
):

    model = SmallGCN(
        in_channels=dataset.num_features,
        hidden_channels=64
    ).to(device)

    model.load_state_dict(
        copy.deepcopy(
            pretrained_model.state_dict()
        )
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    history = []

    for epoch in range(epochs):

        loss = train_one_epoch(
            model,
            train_loader,
            optimizer
        )

        val_mae = evaluate(
            model,
            val_loader
        )

        history.append(
            (loss, val_mae)
        )

        print(
            f"Epoch {epoch:02d} | "
            f"Loss: {loss:.4f} | "
            f"Val MAE: {val_mae:.4f}"
        )

    test_mae = evaluate(
        model,
        test_loader
    )

    return model, history, test_mae


print("\n========== ALIGNED → QM9 ==========\n")

aligned_ft_model, aligned_ft_history, aligned_test_mae = finetune(
    aligned_model,
    epochs=15
)

print(
    "\nAligned test MAE:",
    aligned_test_mae
)

print("\n========== MISALIGNED → QM9 ==========\n")

misaligned_ft_model, misaligned_ft_history, misaligned_test_mae = finetune(
    misaligned_model,
    epochs=15
)

print(
    "\nMisaligned test MAE:",
    misaligned_test_mae
)

print("\n========== TRAIN FROM SCRATCH ==========\n")

scratch_model = SmallGCN(
    in_channels=dataset.num_features,
    hidden_channels=64
).to(device)

scratch_history = []

optimizer = torch.optim.Adam(
    scratch_model.parameters(),
    lr=0.001
)

for epoch in range(15):

    loss = train_one_epoch(
        scratch_model,
        train_loader,
        optimizer
    )

    val_mae = evaluate(
        scratch_model,
        val_loader
    )

    scratch_history.append(
        (loss, val_mae)
    )

    print(
        f"Epoch {epoch:02d} | "
        f"Loss: {loss:.4f} | "
        f"Val MAE: {val_mae:.4f}"
    )


scratch_test_mae = evaluate(
    scratch_model,
    test_loader
)

print(
    "\nScratch test MAE:",
    scratch_test_mae
)

print("\n========== FINAL RESULTS ==========\n")

print(
    f"Scratch:      {scratch_test_mae:.4f}"
)

print(
    f"Aligned:      {aligned_test_mae:.4f}"
)

print(
    f"Misaligned:   {misaligned_test_mae:.4f}"
)


aligned_vs_misaligned = (
    (misaligned_test_mae - aligned_test_mae)
    / misaligned_test_mae
) * 100


print(
    f"\nAligned improvement over misaligned: "
    f"{aligned_vs_misaligned:.2f}%"
)


aligned_val = [
    x[1]
    for x in aligned_ft_history
]

misaligned_val = [
    x[1]
    for x in misaligned_ft_history
]

scratch_val = [
    x[1]
    for x in scratch_history
]


plt.figure(figsize=(8, 5))

plt.plot(
    aligned_val,
    marker="o",
    label="Aligned pretraining"
)

plt.plot(
    misaligned_val,
    marker="o",
    label="Misaligned pretraining"
)

plt.plot(
    scratch_val,
    marker="o",
    label="From scratch"
)

plt.xlabel("Fine-tuning epoch")
plt.ylabel("Validation MAE")
plt.title("Downstream Transfer")

plt.legend()
plt.tight_layout()
plt.show()


labels = [
    "Scratch",
    "Aligned",
    "Misaligned"
]

values = [
    scratch_test_mae,
    aligned_test_mae,
    misaligned_test_mae
]


plt.figure(figsize=(7, 5))

plt.bar(
    labels,
    values
)

plt.ylabel("Test MAE")
plt.title("Effect of Pretraining Data Alignment")

plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 5))

plt.plot(
    aligned_pretrain_history,
    marker="o",
    label="Aligned"
)

plt.plot(
    misaligned_pretrain_history,
    marker="o",
    label="Misaligned"
)

plt.xlabel("Pretraining epoch")
plt.ylabel("MSE loss")
plt.title("Pretraining Loss")

plt.legend()
plt.tight_layout()
plt.show()