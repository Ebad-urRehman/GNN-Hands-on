from numpy import argmax
import torch
from ogb.linkproppred import PygLinkPropPredDataset
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr, GlobalStorage
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling
import os
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt

torch.serialization.add_safe_globals([DataTensorAttr, DataEdgeAttr, GlobalStorage])

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "true"


dataset = PygLinkPropPredDataset(name='ogbl-collab',
                                 root='dataset/')

graph = dataset[0]


class GCNEncoder(nn.Module):
    def __init__(self, num_features, hidden_dim):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)

        return x


num_features = graph.num_node_features
num_edges = graph.num_edges
hidden_dim = 128

model = GCNEncoder(num_features=num_features, hidden_dim=hidden_dim)
loss_fct = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.01)

print(model)

batch_size = 10

split = dataset.get_edge_split()
train_split = split['train']
val_split = split['valid']
test_split = split['test']
print(train_split)

# train_idx = [f'{edge[0]} {edge[1]}' for edge in train_split['edge']]
# val_idx = [f'{edge[0]} {edge[1]}' for edge in val_split['edge']]
# test_idx = [f'{edge[0]} {edge[1]}' for edge in test_split['edge']]

# print(train_idx)

train_edges = train_split['edge']
# train_neg_edges = train_split['edge_neg'] # this dataset has no negative edges
# generating artificially
train_neg_edges = negative_sampling(
    edge_index=graph.edge_index,
    num_nodes=graph.num_nodes,
    num_neg_samples=train_edges.size(0)
).t()

train_losses = []

for batch in range(batch_size):
    z = model(graph.x, graph.edge_index)

    edges = torch.cat([train_edges, train_neg_edges], dim=0)

    src = edges[:, 0] # first column source nodes
    dst = edges[:, 1] # 2nd column destination nodes

    # print(src.shape, dst.shape)

    src_z = z[src]
    dst_z = z[dst]

    # print(src_z.shape)
    # print(dst_z.shape)

    scores = (src_z * dst_z).sum(dim=1)
    # print(scores.shape)

    probabilities = torch.sigmoid(scores)

    pos_labels = torch.ones(train_edges.size(0), dtype=torch.float)
    neg_labels = torch.zeros(train_neg_edges.size(0), dtype=torch.float)

    labels = torch.cat([pos_labels, neg_labels])

    preds = (probabilities > 0.5).long()
    print(scores[:100])
    print(preds==0)
    print(labels==0)

    loss_fct = nn.BCEWithLogitsLoss()

    loss = loss_fct(scores, labels)
    train_losses.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(loss)


# plot loss function
plt.figure(figsize=[10,8])

plt.plot(train_losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Epochs vs Loss")
plt.savefig('Link Prediction in Collab dataset')
plt.show()


model.eval()

with torch.no_grad():
    z = model(graph.x, graph.edge_index)

    edges = torch.cat([train_edges, train_neg_edges], dim=0)

    src = edges[:, 0]
    dst = edges[:, 1]

    src_z = z[src]
    dst_z = z[dst]

    scores = (src_z * dst_z).sum(dim=1)

    probabilities = torch.sigmoid(scores)

    pos_labels = torch.ones(
        train_edges.size(0),
        dtype=torch.float
    )

    neg_labels = torch.zeros(
        train_neg_edges.size(0),
        dtype=torch.float
    )

    labels = torch.cat([pos_labels, neg_labels])

    preds = (probabilities > 0.5).long()