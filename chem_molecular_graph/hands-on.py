# graph level regression task

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

from torch_geometric.nn.models.schnet import qm9_target_dict

import matplotlib.pyplot as plt
import utils
print(QM9.raw_url)

dataset = QM9(root='dataset/QM9')

print(dataset)
print("Number of Molecules/Graphs: ", len(dataset))
print("Number of node features: ", dataset.num_node_features)
print("Number of classes: ", dataset.num_classes)

graph1 = dataset[0]

print(graph1)

print("Node Features shape: ", graph1.x.shape)
print("Edge index shape : ", graph1.edge_index.shape)
print("Edge attributes shape: ", graph1.edge_attr.shape)
print("Target shape ", graph1.y.shape) 

print(qm9_target_dict.items()) # first 12 property names (columns)
print("Target Values: ", graph1.y[0, :]) # 19 molecular target properties stored in the QM9 dataset

# features explanation
"""
Dipole Moment : 
Isotropic Polarizability :
"""

TARGET_INDEX = 7

print("U0:", graph1.y[0, TARGET_INDEX].item())


# print("Number of Graphs: ", dataset.num_graphs)
# Extracting number for every molecule
num_atoms = np.array([data.num_nodes for data in dataset])

num_atoms_with_smiles = [
    (dataset[i].smiles, dataset[i].num_nodes) for i in range(len(dataset))
]

# Statistics
sorted_mols = sorted(num_atoms_with_smiles, key=lambda x: x[1])

min_mol = sorted_mols[0]
max_mol = sorted_mols[-1]

print("Minimum atoms:", min_mol[1], "| SMILES:", min_mol[0])
print("Maximum atoms:", max_mol[1], "| SMILES:", max_mol[0])

mean_atoms = sum(num_atoms) / len(num_atoms)
print("Mean atoms:", mean_atoms)

median_mol = sorted_mols[len(sorted_mols) // 2]
print("Median-ish atoms:", median_mol[1], "| SMILES:", median_mol[0])


# atoms distribution
unique, counts = np.unique(
    num_atoms,
    return_counts=True
)
# print distribution
for atoms, count in zip(unique, counts):
    print(f"{atoms:2d} atoms -> {count:6d} molecules")

# atoms distribution
plt.hist(num_atoms, bins=30, edgecolor='black')
plt.xlabel("Number of Atoms")
plt.ylabel("Frequency")
plt.title("Distribution of Atoms")
plt.savefig("figures/molecule_graphs/Distribution of Atoms.png")
plt.show()



