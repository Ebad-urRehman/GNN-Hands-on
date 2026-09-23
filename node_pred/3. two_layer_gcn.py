from numpy import argmax
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch_geometric.nn import GCNConv
from torch_geometric.data.data import DataTensorAttr, GlobalStorage, DataEdgeAttr
from ogb.nodeproppred import PygNodePropPredDataset
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

torch.serialization.add_safe_globals([DataTensorAttr, DataEdgeAttr, GlobalStorage])

dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='dataset')

num_features = dataset.num_node_features
num_classes = dataset.num_classes
graph = dataset[0]

class GCN(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()

        self.conv1 = GCNConv(num_features, 256)
        self.conv2 = GCNConv(256, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


# define model, loss, optimizer
model = GCN(num_features=num_features, num_classes=num_classes)
loss_fct = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.01)

epochs = 100

# idx splits
split_idx = dataset.get_idx_split()
train_idx = split_idx['train']
val_idx = split_idx['valid']
test_idx = split_idx['test']


train_losses = []
val_losses = []

for epoch in range(epochs):
    out = model(graph.x, graph.edge_index)

    train_loss = loss_fct(out[train_idx], 
                          graph.y[train_idx].squeeze())

    val_loss = loss_fct(out[val_idx], 
                              graph.y[val_idx].squeeze())

    train_losses.append(train_loss.item())
    val_losses.append(val_loss.item())

    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(epoch)

# draw losses over epochs
plt.figure(figsize=[12,8])

plt.plot(train_losses)
plt.plot(val_losses)

plt.xlabel("Epochs")
plt.ylabel("Losses")
plt.title("Losses over Epochs")
plt.savefig('Loss over epochs GCN.png')

plt.show()


# draw confusion matrix, and print classification matrix

model.eval()

with torch.no_grad():
    out = model(graph.x, graph.edge_index)

    y_pred = argmax(out, axis=1)
    y_true = graph.y.squeeze()

# find for split indexes
train_true = y_true[train_idx]
val_true = y_true[val_idx]
test_true = y_true[test_idx]

train_pred = y_pred[train_idx]
val_pred = y_pred[val_idx]
test_pred = y_pred[test_idx]


print('-'*30, "Train Report", '-'*30)
print(classification_report(
    train_true.cpu(),
    train_pred.cpu()
))

print('-'*30, "Validation Report", '-'*30)
print(classification_report(
    val_true.cpu(),
    val_pred.cpu()
))

print('-'*30, "Test Report", '-'*30)
print(classification_report(
    test_true.cpu(),
    test_pred.cpu()
))


# confusion matrices

def show_confusion_matrices(cm, split_name):
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(12, 10))

    plt.imshow(cm_normalized)

    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.title(f"{split_name} Confusion Matrix")

    plt.colorbar()
    plt.savefig(f'{split_name} GCN Confusion Matrix.png')

    plt.show()

train_cm = confusion_matrix(
    train_true.cpu(),
    train_pred.cpu()
)

val_cm = confusion_matrix(
    val_true.cpu(),
    val_pred.cpu()
)

test_cm = confusion_matrix(
    test_true.cpu(),
    test_pred.cpu()
)

show_confusion_matrices(train_cm, "Train")
show_confusion_matrices(val_cm, "Validation")
show_confusion_matrices(test_cm, "Test")