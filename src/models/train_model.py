import math
import os.path as osp

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv
from torch.nn import LayerNorm, Linear

from src.data.hhh_graph import HHHGraph

train_root = osp.join(osp.dirname(osp.realpath(__file__)), "..", "..", "data/train")
val_root = osp.join(osp.dirname(osp.realpath(__file__)), "..", "..", "data/val")
test_root = osp.join(osp.dirname(osp.realpath(__file__)), "..", "..", "data/test")

train_dataset = HHHGraph(root=train_root, entry_start=0, entry_stop=1_000)
val_dataset = HHHGraph(root=val_root, entry_start=1_000, entry_stop=2_000)
test_dataset = HHHGraph(root=test_root, entry_start=2_000, entry_stop=3_000)

class TransformerModel(torch.nn.Module):
    def __init__(self, in_node_channels, in_edge_channels, num_classes, hidden_channels, num_layers, heads, dropout=0.3):
        super().__init__()

        self.convs = torch.nn.ModuleList((
            TransformerConv(in_node_channels, hidden_channels // heads, heads, concat=True, beta=True, dropout=dropout, edge_dim=in_edge_channels),
            TransformerConv(hidden_channels, hidden_channels // heads, heads, concat=True, beta=True, dropout=dropout, edge_dim=in_edge_channels),
            TransformerConv(hidden_channels, num_classes, heads, concat=False, beta=True, dropout=dropout, edge_dim=in_edge_channels),
        ))
        self.norms = torch.nn.ModuleList((
            LayerNorm(hidden_channels),
            LayerNorm(hidden_channels),
        ))

        self.lins = torch.nn.ModuleList((
            Linear(num_classes * 2 + in_edge_channels, num_classes),
        ))

    def forward(self, x, edge_index, edge_attr):
        for conv, norm in zip(self.convs, self.norms):
            x = norm(conv(x, edge_index, edge_attr)).relu()
        x = self.convs[-1](x, edge_index, edge_attr)

        edge_attr = torch.cat([x[edge_index[1]], x[edge_index[0]], edge_attr], dim=1)

        return self.lins[0](edge_attr)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_data = train_dataset.data.to(device)
val_data = val_dataset.data.to(device)

train_data.y = train_data.y.to(torch.long)
print(train_data.y[train_data.y < 0])

model = TransformerModel(in_node_channels=train_dataset.num_features, in_edge_channels=train_dataset.num_edge_features, num_classes=train_dataset.num_classes, hidden_channels=64, num_layers=3, heads=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)
print(model)

def train(data):  # How many labels to use for propagation
    model.train()

    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data.edge_attr)
    loss = F.cross_entropy(out, data.y)
    loss.backward()
    optimizer.step()

    return float(loss)


@torch.no_grad()
def test(data):
    model.eval()

    out = model(data.x, data.edge_index, data.edge_attr)
    pred = out.argmax(dim=-1)
    test_acc = int((pred == data.y).sum()) / pred.size(0)

    return test_acc


for epoch in range(1, 101):
    loss = train(train_data)
    train_acc = test(train_data)
    val_acc = test(val_data)
    print(f"Epoch: {epoch:03d}, Train Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
