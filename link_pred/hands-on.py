import torch
from ogb.linkproppred import PygLinkPropPredDataset
import torch.nn as nn
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr, GlobalStorage
import numpy
import os

torch.serialization.add_safe_globals([DataTensorAttr, DataEdgeAttr, GlobalStorage, numpy.core.multiarray._reconstruct])

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "true"

dataset = PygLinkPropPredDataset(name='ogbl-collab',
                                 root='dataset/')

graph = dataset[0]

print(len(dataset))

print(graph)
print(graph.num_node_features, graph.num_edges)
print("num of nodes, num of features : ", graph.x.shape)
print("edge weights [Edge 1-10] : ", graph.edge_weight[:10].squeeze())

# there are no classes, prediction is on either the edge exists or not

split_edge = dataset.get_edge_split()
print(split_edge.keys())

print(split_edge['train'].keys())
print(split_edge['train'])
print(split_edge['valid'].keys())

split = dataset.get_edge_split()
train_split = split_edge['train']
val_split = split_edge['valid']
test_split = split_edge['test']
print(train_split)

train_edges = train_split['edge']

print(train_edges.shape, train_edges[:5])

first_edge = train_edges[0]

u = first_edge[0]
v = first_edge[1]

print("nodes of first edge", u, v)

# SEAL (learning from subgraphs, Embeddings and attributes for link prediction)


# train has edges only, while validation has negative edges (representing absence of edges)
#  called negative injection 

