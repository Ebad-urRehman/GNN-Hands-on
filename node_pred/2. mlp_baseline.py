# How well can we classify arXiv papers using only their 128-dimensional node features?

# we don't utilize edge index in this video just node level predictions via MLP

import torch
from torch import relu_
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

import ogb
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric import data
from torch_geometric.data.data import DataTensorAttr, DataEdgeAttr, GlobalStorage

torch.serialization.add_safe_globals([DataEdgeAttr, DataTensorAttr, GlobalStorage])


dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='dataset/')

graph = dataset[0]
print(graph)

num_features = graph.num_node_features
num_classes = dataset.num_classes

print("Input Features : ", num_features)
print("Output Classes : ", num_classes)


# MLP 128 -> 256 (hidden layer) -> 40


class MLP(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()

        self.lin1 = nn.Linear(num_features, 256)
        self.lin2 = nn.Linear(256, num_classes)


    def forward(self, x):
        # forward prop
        x = self.lin1(x) # 128 -> 256
        x = F.relu(x)
        x = self.lin2(x) # 256 -> 40
        return x




model = MLP(num_features=num_features, num_classes=num_classes)
loss_func = CrossEntropyLoss()
optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01
    )

split_idx = dataset.get_idx_split()
print(split_idx.keys())
train_idx = split_idx['train']
valid_idx = split_idx['valid']
test_idx = split_idx['test']

epochs = 15


# lists for storing losses
train_losses = []
val_losses = []

for epoch in range(epochs):

    # print(model)
    out = model(graph.x) # logits

    # print(out.shape)
    # print(out[0], out[0].argmax())

    train_loss = loss_func(
        out[train_idx],
        graph.y[train_idx].squeeze()
    )
    val_loss = loss_func(
        out[valid_idx],
        graph.y[valid_idx].squeeze()
    )
    print("Train loss", train_loss, "\n", "Valid loss", val_loss)

    train_losses.append(train_loss.item())
    val_losses.append(val_loss.item())

    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()


plt.figure(figsize=(12, 8))

plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and validation loss")
plt.legend()
plt.savefig('Loss over epochs MLP.png')
plt.show()


# Classification Reports

model.eval()

# calculate all preds once
with torch.no_grad():
    out = model(graph.x)

y_pred = out.argmax(dim=1)
y_true = graph.y.squeeze()

# train
train_pred = y_pred[train_idx]
train_true = y_true[train_idx]

# validation
val_pred = y_pred[valid_idx]
val_true = y_true[valid_idx]

# test
test_pred = y_pred[test_idx]
test_true = y_true[test_idx]

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
    plt.savefig(f'{split_name} MLP Confusion Matrix.png')

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