import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GINConv
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout, ModuleList
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool, global_add_pool

class simpleGAT(torch.nn.Module):
    def __init__(self, in_channels, dim_h, out_channels, num_layers, dropout, heads=8):
        super(simpleGAT,self).__init__()
        self.conv1 = GATConv(in_channels, dim_h, heads=heads, dropout=dropout)
        # self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout)
        # self.conv3 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout)

        self.GAT_layers = ModuleList([
            GATConv(dim_h * heads, dim_h, heads=heads) 
            for _ in range(num_layers-1)
        ])

        self.classifier = Sequential(
            Linear(dim_h * heads, dim_h),  # *2 for mean and max pooling
            ReLU(),
            Dropout(dropout),
            Linear(dim_h, dim_h // 2),
            ReLU(),
            Dropout(dropout),
            Linear(dim_h // 2, out_channels)
        )
        # self.lin1 = Linear(hidden_channels * heads, 5*out_channels)
        # self.lin2 = Linear(5*out_channels, out_channels)
        
    def forward(self, x=None, edge_index=None, edge_attr=None, batch=None, edge_weight=None, data=None):
        if x is None:
            x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index).relu()

        for layer in self.GAT_layers:
            x = layer(x, edge_index).relu()

        # x = self.conv2(x, edge_index).relu()
        # x = self.conv3(x, edge_index).relu()
        #x = self.conv3(x, edge_index).relu()
        graph_repr = global_mean_pool(x, batch)  # Aggregate node features to graph level
        # x = self.lin1(x).relu()
        return self.classifier(graph_repr)
    
    def get_emb(self, x, edge_index=None) -> torch.Tensor:
        
        post_conv = self.conv1(x, edge_index).relu()
        for layer in self.GAT_layers:
            post_conv = layer(post_conv, edge_index).relu()
            
        return post_conv

class simpleGIN(torch.nn.Module):
    """GIN"""
    def __init__(self, in_channels, dim_h, out_channels, num_layers, dropout):
        super(simpleGIN, self).__init__()
        self.conv1 = GINConv(
            Sequential(Linear(in_channels, dim_h),
                       BatchNorm1d(dim_h), ReLU(),
                       Dropout(dropout),
                       Linear(dim_h, dim_h), ReLU()))
        # self.conv2 = GINConv(
        #     Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
        #                Dropout(dropout),
        #                Linear(dim_h, dim_h), ReLU()))
        # self.conv3 = GINConv(
        #     Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
        #                Dropout(dropout),
        #                Linear(dim_h, dim_h), ReLU()))
        
        self.GIN_layers = ModuleList([
            GINConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       #Dropout(dropout),
                       Linear(dim_h, dim_h), ReLU())) 
            for _ in range(num_layers-1)
        ])

        self.classifier = Sequential(
            Linear(dim_h * num_layers, dim_h),  # *2 for mean and max pooling
            ReLU(),
            Dropout(dropout),
            Linear(dim_h, dim_h // 2),
            ReLU(),
            Dropout(dropout),
            Linear(dim_h // 2, out_channels)
        )
        
        # self.lin1 = Linear(dim_h*3, dim_h*3)
        # self.lin2 = Linear(dim_h*3, out_channels)
        # self.dropout = dropout

    def forward(self, x, edge_index=None, edge_attr=None, batch=None, edge_weight=None):
        h = self.conv1(x, edge_index)
        hlist = []
        hlist.append(h)

        for layer in self.GIN_layers:
            h = layer(h, edge_index)
            hlist.append(h)
        # h2 = self.conv2(h1, edge_index)
        # h3 = self.conv3(h2, edge_index)
        for i in range(len(hlist)):
            hlist[i] = global_add_pool(hlist[i], batch)
        # h1 = global_add_pool(h1, batch)
        # h2 = global_add_pool(h2, batch)
        # h3 = global_add_pool(h3, batch)

        # h = torch.cat((h1, h2, h3), dim=1)
        graph_repr = torch.cat(hlist, dim=1)

        return self.classifier(graph_repr)
        # h = self.lin1(h)
        # h = h.relu()
        # h = F.dropout(h, p=self.dropout, training=self.training)
        
        # return self.lin2(h)

    def get_emb(self, x, edge_index=None) -> torch.Tensor:
        
        post_conv = self.conv1(x, edge_index).relu()
        for layer in self.GIN_layers:
            post_conv = layer(post_conv, edge_index).relu()
            
        return post_conv
    
class MPNNLayer(MessagePassing):
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super(MPNNLayer, self).__init__(aggr='add')
        
        # Message function
        self.message_net = Sequential(
            Linear(2 * node_dim + edge_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim)
        )
        
        self.update_net = Sequential(
            Linear(node_dim + hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, node_dim)
        )
    
    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_i, x_j, edge_attr):
        msg_input = torch.cat([x_i, x_j, edge_attr], dim=1)
        return self.message_net(msg_input)
    
    def update(self, aggr_out, x):
        update_input = torch.cat([x, aggr_out], dim=1)
        return self.update_net(update_input)

class MPNNModel(torch.nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, num_layers=3, num_classes=2, dropout=0.2):
        super(MPNNModel, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        self.node_embedding = Linear(node_dim, hidden_dim)
        
        self.mpnn_layers = ModuleList([
            MPNNLayer(hidden_dim, edge_dim, hidden_dim) 
            for _ in range(num_layers)
        ])
        
        self.classifier = Sequential(
            Linear(hidden_dim * 2, hidden_dim),  # *2 for mean and max pooling
            ReLU(),
            Dropout(dropout),
            Linear(hidden_dim, hidden_dim // 2),
            ReLU(),
            Dropout(dropout),
            Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x, edge_index=None, edge_attr=None, batch=None, edge_weight=None):
        #x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        x = self.node_embedding(x)
        
        if edge_index.size(1) > 0:
            for layer in self.mpnn_layers:
                x = layer(x, edge_index, edge_attr)
                x = F.relu(x)
        else:
            x = F.relu(x)
        
        graph_mean = global_mean_pool(x, batch)
        graph_max = global_max_pool(x, batch)
        graph_repr = torch.cat([graph_mean, graph_max], dim=1)
        
        out = self.classifier(graph_repr)
        return out