import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import RGCNConv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

"""
HeteroData represents graph containing different node types and edge types (Dictionary like interfaces)
RGCNCovn : Relational Graph Conv Network Layer
We have hetrogenous relations in the dataset
"""

# import dataset
nodes = pd.read_csv("dataset/hetionet/hetionet-v1.0-nodes.tsv", sep="\t")

edges = pd.read_csv("dataset/hetionet/hetionet-v1.0-edges.sif.gz", sep="\t")

TARGET_RELATION = "CcSE"

target_edges = edges[edges["metaedge"] == TARGET_RELATION].copy()

print("CcSE edges:", len(target_edges))

train_edges, test_edges = train_test_split(target_edges, test_size=0.20, random_state=42)
train_edges, val_edges = train_test_split(train_edges, test_size=0.20, random_state=42)

print("Train:", len(train_edges))
print("Validation:", len(val_edges))
print("Test:", len(test_edges))


SUBGRAPH_RELATIONS = ["CcSE", "CbG", "CdG", "CuG"]

subgraph_edges = edges[edges["metaedge"].isin(SUBGRAPH_RELATIONS)].copy()

print("Relations:")
print(subgraph_edges["metaedge"].value_counts())

print("\nTotal subgraph edges:", len(subgraph_edges))

val_pairs = set(zip(val_edges["source"], val_edges["target"]))
test_pairs = set(zip(test_edges["source"], test_edges["target"]))

message_edges = []
for _, row in subgraph_edges.iterrows():
    if row["metaedge"] == TARGET_RELATION:
        pair = (row["source"], row["target"])
        if pair in val_pairs or pair in test_pairs:
            continue
    message_edges.append(row)

message_edges = pd.DataFrame(message_edges)

print("Subgraph edges before hiding:", len(subgraph_edges))
print("Message-passing edges:", len(message_edges))

all_nodes = pd.concat([subgraph_edges["source"], subgraph_edges["target"]]).unique()

node_to_id = {node: i for i, node in enumerate(all_nodes)}
id_to_node = {i: node for node, i in node_to_id.items()}
num_nodes = len(all_nodes)

print("Subgraph nodes:", num_nodes)

relations = subgraph_edges["metaedge"].unique()

relation_to_id = {relation: i for i, relation in enumerate(relations)}
num_relations = len(relations)

print(relation_to_id)


edge_index = torch.tensor([
        [
            node_to_id[node]
            for node in message_edges["source"]
        ],
        [
            node_to_id[node]
            for node in message_edges["target"]
        ]
        ], dtype=torch.long
)

edge_type = torch.tensor(
    [relation_to_id[relation] for relation in message_edges["metaedge"]], dtype=torch.long
)

print("edge_index:", edge_index.shape)
print("edge_type:", edge_type.shape)


reverse_edge_index = edge_index.flip(0)
reverse_edge_type = edge_type + num_relations

edge_index = torch.cat([edge_index, reverse_edge_index], dim=1)
edge_type = torch.cat([edge_type, reverse_edge_type])
num_message_relations = num_relations * 2
print("Message-passing edges:", edge_index.shape[1])

print("Message-passing relations:",num_message_relations)

compound_ids = torch.tensor(
    [node_to_id[node] for node in all_nodes if node.startswith("Compound::")], dtype=torch.long
)

side_effect_ids = torch.tensor(
    [node_to_id[node] for node in all_nodes if node.startswith("Side Effect::")], dtype=torch.long
)

print("Compounds:", len(compound_ids))
print("Side Effects:", len(side_effect_ids))


train_source = torch.tensor([node_to_id[node] for node in train_edges["source"]],
    dtype=torch.long)

train_target = torch.tensor([node_to_id[node] for node in train_edges["target"]],
    dtype=torch.long)
print("Training CcSE edges:", len(train_source))

positive_pairs = set(zip(target_edges["source"], target_edges["target"]))
print("Known positive CcSE pairs:", len(positive_pairs))


def negative_sample(train_edges):
    negative_source = []
    negative_target = []

    for _, row in train_edges.iterrows():
        compound = row["source"]
        while True:
            side_effect_id = side_effect_ids[
                torch.randint(len(side_effect_ids),(1,))
            ].item()

            side_effect = id_to_node[side_effect_id]

            if (compound, side_effect) not in positive_pairs:
                break

        negative_source.append(node_to_id[compound])
        negative_target.append(side_effect_id)

    return (torch.tensor(negative_source, dtype=torch.long),
        torch.tensor(negative_target, dtype=torch.long))


negative_source, negative_target = negative_sample(train_edges)

print("Negative samples:", len(negative_source))

class RGCN(nn.Module):
    def __init__(self, num_nodes, num_relations, embedding_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        self.conv1 = RGCNConv(embedding_dim, embedding_dim, num_relations)
        self.conv2 = RGCNConv(embedding_dim, embedding_dim, num_relations)

    def forward(self,edge_index,edge_type):
        x = self.embedding.weight
        x = self.conv1(x, edge_index,edge_type)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_type)
        return x



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

model = RGCN(num_nodes=num_nodes, num_relations=num_message_relations, embedding_dim=64).to(device)

edge_index = edge_index.to(device)
edge_type = edge_type.to(device)
train_source = train_source.to(device)
train_target = train_target.to(device)
negative_source = negative_source.to(device)
negative_target = negative_target.to(device)

relation_embedding = nn.Parameter(torch.randn(64, device=device))

optimizer = torch.optim.Adam(list(model.parameters()) + [relation_embedding], lr=0.01)


def score(head, tail):
    return torch.sum(head * relation_embedding * tail, dim=1)

losses = []
for epoch in range(50):
    model.train()
    optimizer.zero_grad()
    x = model(edge_index, edge_type)

    # Positive CcSE links
    positive_score = score(x[train_source], x[train_target])
    # Negative CcSE links
    negative_score = score(x[negative_source], x[negative_target])

    positive_loss = F.binary_cross_entropy_with_logits(positive_score, torch.ones_like(positive_score))
    negative_loss = F.binary_cross_entropy_with_logits(negative_score, torch.zeros_like(negative_score))
    loss = (positive_loss + negative_loss)
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if epoch % 5 == 0:
        print(
            f"Epoch {epoch:02d} | "
            f"Loss: {loss.item():.4f}"
        )



plt.figure(figsize=(7, 4))
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Compound-Gene Subgraph R-GCN Training Loss")

plt.show()


relation_counts = subgraph_edges["metaedge"].value_counts()

plt.figure(figsize=(7, 4))
relation_counts.plot(kind="bar")
plt.xlabel("Relation")
plt.ylabel("Number of edges")
plt.title("Relations in Compound–Gene Subgraph")
plt.xticks(rotation=0)
plt.show()


model.eval()
with torch.no_grad():
    embeddings = model(edge_index, edge_type).cpu().numpy()

pca = PCA(n_components=2)

embedding_2d = pca.fit_transform(embeddings)
compound_mask = np.array([node.startswith("Compound::") for node in all_nodes])
gene_mask = np.array([node.startswith("Gene::") for node in all_nodes])

plt.figure(figsize=(7, 6))
plt.scatter(embedding_2d[compound_mask, 0], embedding_2d[compound_mask, 1], alpha=0.5, label="Compound")

plt.scatter(embedding_2d[gene_mask, 0], embedding_2d[gene_mask, 1], alpha=0.5, label="Gene")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("R-GCN Embeddings: Compound–Gene Subgraph")
plt.legend()
plt.show()


