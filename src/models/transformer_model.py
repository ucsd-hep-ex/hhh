import torch
from torch.nn import LayerNorm, Linear
from torch_geometric.nn import TransformerConv


class TransformerModel(torch.nn.Module):
    def __init__(self, in_node_channels, in_edge_channels, num_classes, hidden_channels, num_layers, heads, dropout=0.3):
        super().__init__()

        if num_layers < 2:
            raise RuntimeError(f"Need at least 2 layers, but got {num_layers}")

        self.convs = torch.nn.ModuleList(
            (
                TransformerConv(
                    in_node_channels,
                    hidden_channels // heads,
                    heads,
                    concat=True,
                    beta=True,
                    dropout=dropout,
                    edge_dim=in_edge_channels,
                ),
            )
        )
        for i in range(1, num_layers - 1):
            self.convs.append(
                TransformerConv(
                    hidden_channels,
                    hidden_channels // heads,
                    heads,
                    concat=True,
                    beta=True,
                    dropout=dropout,
                    edge_dim=in_edge_channels,
                )
            )
        self.convs.append(
            TransformerConv(
                hidden_channels, num_classes, heads, concat=False, beta=True, dropout=dropout, edge_dim=in_edge_channels
            ),
        )
        self.norms = torch.nn.ModuleList([LayerNorm(hidden_channels) for i in range(0, num_layers - 1)])

        self.lins = torch.nn.ModuleList((Linear(num_classes * 2 + in_edge_channels, num_classes),))

    def forward(self, x, edge_index, edge_attr):
        for conv, norm in zip(self.convs, self.norms):
            x = norm(conv(x, edge_index, edge_attr)).relu()
        x = self.convs[-1](x, edge_index, edge_attr)

        edge_attr = torch.cat([x[edge_index[1]], x[edge_index[0]], edge_attr], dim=1)

        return self.lins[0](edge_attr)
