import torch
import torch_geometric
import ogb
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr, GlobalStorage
from collections import Counter

print("PyTorch:", torch.__version__)
print("PyG:", torch_geometric.__version__)
print("OGB:", ogb.__version__)

from ogb.nodeproppred import PygNodePropPredDataset

torch.serialization.add_safe_globals([DataEdgeAttr, DataTensorAttr, GlobalStorage])

dataset = PygNodePropPredDataset(name="ogbn-arxiv", root="dataset/")
print(dataset)
print("Number of graphs : ", len(dataset))

graph = dataset[0]
print(graph)
print("x shape", graph.x.shape) # [num of nodes, no of features per node]
print(graph.x[0], len(graph.x[0])) # first instance, feature sizes


# edge index
print("edge_index shape", graph.edge_index.shape) 

# [2, number of edges] one row representing source 
# and other represents destination, 
print(graph.edge_index[0], len(graph.edge_index[0])) # source row, number of edges
print(graph.edge_index[1], len(graph.edge_index[1])) # destination row, number of edges


print("y shape", graph.y.shape) # number of rows, predication column

# number of classes
print("Number of classes", len(graph.y.unique()))
# print(dataset.num_classes)
# print(dataset.meta_info)

# counts per class
counts = Counter(graph.y.squeeze().tolist())
print("Category : instances count", counts)

# split index
split_idx = dataset.get_idx_split()
print(split_idx, split_idx.keys())

# print size of splits
for split, indices in split_idx.items():
    print(f"Size of {split}:", indices.shape)