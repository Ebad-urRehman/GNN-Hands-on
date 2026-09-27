import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import HeteroData
from torch_geometric.nn import RGCNConv

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

"""
HeteroData represents graph containing different node types and edge types (Dictionary like interfaces)
RGCNCovn : Relational Graph Conv Network Layer
We have hetrogenous relations in the dataset
"""

# import dataset
nodes = pd.read_csv(
    "dataset/hetionet/hetionet-v1.0-nodes.tsv",
    sep="\t"
)

edges = pd.read_csv(
    "dataset/hetionet/hetionet-v1.0-edges.sif.gz",
    sep="\t"
)

# get node type
def get_node_type(node_id):
    return node_id.split("::")[0] # format Biological Process::GO:0071357

edges["source_type"] = edges["source"].apply(get_node_type)
edges["target_type"] = edges["target"].apply(get_node_type)


TARGET_RELATION = "CcSE"
target_edges = edges[edges["metaedge"] == TARGET_RELATION].copy() # 138044 edges
target_edges = target_edges.reset_index(drop=True)

# Train 88924, Validation 22231, Test 27789
train_edges, test_edges = train_test_split(target_edges, test_size=0.2, random_state=42)
train_edges, val_edges = train_test_split(train_edges, test_size=0.2, random_state=42)

all_nodes = pd.concat([edges["source"], edges["target"]]).unique()
print(all_nodes, len(all_nodes))

# create Hetrogeneous graph
# format {node_type : [nodes]}

node_types = {}

for node in all_nodes:
    node_type = get_node_type(node)

    if node_type not in node_types:
        node_types[node_type] = []

    node_types[node_type].append(node)

node_to_id = {}
for node_type, nodes_of_type in node_types.items():
    node_to_id[node_type] = {
        node: i
        for i, node in enumerate(nodes_of_type)
    } # node_to_id["Gene"]["Gene::9021"] = 0

data = HeteroData()

embedding_dim = 64

for node_type, nodes_of_type in node_types.items():
    data[node_type].x = torch.randn(len(nodes_of_type),embedding_dim) # 64 dim embedding for nodes
    data[node_type].num_nodes = len(nodes_of_type)

# convert KG to PyG's heterogeneous edge format
"""
In case of PyG's Hetrogeneous graph format edge index is like that:
data['compound', 'causes', 'side effect'].edge_index = torch.tensor([
    [0, 1, 2],  
    [1, 12, 42] 
], dtype=torch.long)
"""
for relation, group in edges.groupby("metaedge"):
    source_type = group["source_type"].iloc[0]
    target_type = group["target_type"].iloc[0]
    
    source_ids = [node_to_id[source_type][node] for node in group["source"]]
    target_ids = [node_to_id[target_type][node] for node in group["target"]]

    edge_index = torch.tensor([source_ids, target_ids], dtype=torch.long)

    data[source_type, relation, target_type].edge_index = edge_index

print(data)
print(data.node_types)
print(data.edge_types)

edge_type = data.edge_types[0]

print("Edge type:", edge_type)
print("Edge index shape:", data[edge_type].edge_index.shape)
print(data[edge_type].edge_index[:, :10])


# target edge selection
target_edges = edges[edges["metaedge"] == TARGET_RELATION].copy()
target_edges = target_edges.reset_index(drop=True)

print("Target relation:", TARGET_RELATION)
print("Number of target edges:", len(target_edges))

print(target_edges[["source", "metaedge", "target"]].head())

from sklearn.model_selection import train_test_split

target_edges = target_edges.reset_index(drop=True)

# split datasets
train_edges, test_edges = train_test_split(target_edges,test_size=0.20,random_state=42)
train_edges, val_edges = train_test_split(train_edges,test_size=0.20,random_state=42)

print("Train:", len(train_edges))
print("Validation:", len(val_edges))
print("Test:", len(test_edges))


relations = edges["metaedge"].unique()

relation_to_id = {
    relation: i
    for i, relation in enumerate(relations)
}

num_relations = len(relations)

print("Number of relations:", num_relations)

# remove validation and test CcSE edges
test_pairs = set(
    zip(test_edges["source"], test_edges["target"])
)

val_pairs = set(
    zip(val_edges["source"], val_edges["target"])
)

message_edges = []

for _, row in edges.iterrows():

    pair = (row["source"], row["target"])

    if row["metaedge"] == TARGET_RELATION:
        if pair in test_pairs or pair in val_pairs:
            continue

    message_edges.append(row)

message_edges = pd.DataFrame(message_edges)

print("Edges used for message passing:", len(message_edges))

# build the graph
edge_index = torch.tensor(
    [
        [node_to_id[x] for x in message_edges["source"]],
        [node_to_id[x] for x in message_edges["target"]]
    ],
    dtype=torch.long
)

edge_type = torch.tensor(
    [
        relation_to_id[x]
        for x in message_edges["metaedge"]
    ],
    dtype=torch.long
)

print(edge_index.shape)
print(edge_type.shape)

