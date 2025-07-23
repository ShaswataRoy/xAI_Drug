import torch
import torch_geometric
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GINConv
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool, global_add_pool

class simpleGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, heads=8):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout)
        self.conv3 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout)
        self.lin1 = Linear(hidden_channels * heads, 5*out_channels)
        self.lin2 = Linear(5*out_channels, out_channels)
        
    def forward(self, x, edge_index=None, batch=None, edge_weight=None):
        #x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()
        #x = self.conv3(x, edge_index).relu()
        x = global_mean_pool(x, batch)  # Aggregate node features to graph level
        x = self.lin1(x).relu()
        return self.lin2(x)

class simpleGIN(torch.nn.Module):
    """GIN"""
    def __init__(self, in_channels, dim_h, out_channels, dropout):
        super(simpleGIN, self).__init__()
        self.conv1 = GINConv(
            Sequential(Linear(in_channels, dim_h),
                       BatchNorm1d(dim_h), ReLU(),
                       Dropout(dropout),
                       Linear(dim_h, dim_h), ReLU()))
        self.conv2 = GINConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Dropout(dropout),
                       Linear(dim_h, dim_h), ReLU()))
        self.conv3 = GINConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Dropout(dropout),
                       Linear(dim_h, dim_h), ReLU()))
        self.lin1 = Linear(dim_h*3, dim_h*3)
        self.lin2 = Linear(dim_h*3, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index=None, batch=None, edge_weight=None):
        h1 = self.conv1(x, edge_index)
        h2 = self.conv2(h1, edge_index)
        h3 = self.conv3(h2, edge_index)

        h1 = global_add_pool(h1, batch)
        h2 = global_add_pool(h2, batch)
        h3 = global_add_pool(h3, batch)

        h = torch.cat((h1, h2, h3), dim=1)

        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        return self.lin2(h)