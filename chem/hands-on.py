# graph level regression task

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

import matplotlib.pyplot as plt

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