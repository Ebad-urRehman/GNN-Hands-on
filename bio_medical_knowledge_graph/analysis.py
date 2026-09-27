import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import HeteroData
from torch_geometric.nn import RGCNConv

import pandas as pd
import numpy as np
import requests
from pathlib import Path

"""
HeteroData represents graph containing different node types and edge types
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

print("Number of nodes:", len(nodes))
print("Number of edges:", len(edges))

# print(nodes.head())
print(nodes.columns)
print("Node Names: ", nodes['name'].unique(), len(nodes['name'].unique()))
print("Node Types: ", nodes['kind'].unique(), len(nodes['kind'].unique()))

# print(edges.head())
print(edges.columns)
print("Edge Source: ", edges['source'].unique(), len(edges['source'].unique()))
print("Edge Meta Edge: ", edges['metaedge'].unique(), len(edges['metaedge'].unique()))
print("Edge Target: ", edges['target'].unique(), len(edges['target'].unique()))


"""
Hetrogenous graph
CtD Compound treats disease
CbG Compound binds Gene
DaG Disease associates Gene
"""
print(
    edges["metaedge"]
    .value_counts()
)

# get node type
def get_node_type(node_id):
    return node_id.split("::")[0] # format Biological Process::GO:0071357


edges["source_type"] = edges["source"].apply(get_node_type)
edges["target_type"] = edges["target"].apply(get_node_type)

print(edges["source_type"].unique())
print(edges[["source_type", "target_type"]].head())

print(pd.crosstab(edges["source_type"], edges["target_type"]))


print(edges[edges["source_type"].str.contains("Compound", case=False, na=False)]["metaedge"].value_counts())

# Create a separate ID mapping for each node type
all_nodes = pd.concat([edges["source"], edges["target"]]).unique()
node_to_id = {}
node_types = {}

for node in all_nodes:
    node_type = get_node_type(node)

    if node_type not in node_types:
        node_types[node_type] = []

    node_types[node_type].append(node)


# Create local IDs within each node type
for node_type, nodes_of_type in node_types.items():
    node_to_id[node_type] = {
        node: i
        for i, node in enumerate(nodes_of_type)
    }

for node, i in node_to_id.items()[:10]:
    print(node,i)

#  Example : Compound::DB00448 represents Lansoprazole     CcSE  Side Effect::C0032584 represents Polyp

# select CcSE "Compound causes Side Effect"
