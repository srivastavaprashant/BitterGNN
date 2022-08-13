import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, EdgePooling, GraphConv, JumpingKnowledge
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool

class BitterGCN_Baseline(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(BitterGCN_Baseline, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(20, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, 2)


    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x1 = self.conv1(x, edge_index)
        x1 = x1.relu()
        x2 = self.conv2(x1, edge_index)
        x2 = x2.relu()
        #x = F.dropout(x, p = 0.1, training=self.training)
        x3 = self.conv3(x2, edge_index)

        # 2. Readout layer [batch_size, hidden_channels]
        x = global_mean_pool(x3, batch)  

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.lin(x)
        
        return x

class BitterGCN_MixedPool(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(BitterGCN_MixedPool, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(20, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, 2)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x1 = self.conv1(x, edge_index)
        x1 = x1.relu()
        x2 = self.conv2(x1, edge_index)
        x2 = x2.relu()
        x3 = self.conv3(x2, edge_index)
        
        # 2. Readout layer [batch_size, hidden_channels]
        x = global_max_pool(x3, batch) + \
            global_add_pool(x3, batch) + \
            global_mean_pool(x3, batch)# [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.lin(x)
        
        return x

class BitterGCNEdgePooling(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(BitterGCNEdgePooling, self).__init__()
        self.conv1 = GraphConv(20, 16, aggr='mean')
        self.convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.convs.extend([
            GraphConv(hidden_channels, hidden_channels, aggr='mean')
            for i in range(3 - 1)
        ])
        self.pools.extend(
            [EdgePooling(hidden_channels) for i in range((3) // 2)])
        self.jump = JumpingKnowledge(mode='cat')
        self.lin1 = Linear(3 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 2)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        xs = [global_mean_pool(x, batch)]
        for i, conv in enumerate(self.convs):
            x = F.relu(conv(x, edge_index))
            xs += [global_mean_pool(x, batch)]
            if i % 2 == 0 and i < len(self.convs) - 1:
                pool = self.pools[i // 2]
                x, edge_index, batch, _ = pool(x, edge_index, batch=batch)
        x = self.jump(xs)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

class BitterGAT_Baseline(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(BitterGAT_Baseline, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GATConv(20, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.conv3 = GATConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, 2)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x1 = self.conv1(x, edge_index)
        x1 = x1.relu()
        x2 = self.conv2(x1, edge_index)
        x2 = x2.relu()
        x3 = self.conv3(x2, edge_index)

        # 2. Readout layer [batch_size, hidden_channels]
        x = global_mean_pool(x3, batch)  

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.lin(x)    
        return x

class BitterGAT_Edge(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(BitterGAT_Edge, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GATConv(20, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.conv3 = GATConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, 2)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x1 = self.conv1(x, edge_index)
        x1 = x1.relu()
        x2 = self.conv2(x1, edge_index)
        x2 = x2.relu()
        x3 = self.conv3(x2, edge_index)

        # 2. Readout layer [batch_size, hidden_channels]
        x = global_mean_pool(x3, batch)  

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.lin(x)    
        return x

class BitterGAT_MixedPool(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(BitterGAT_MixedPool, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GATConv(20, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.conv3 = GATConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, 2)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x1 = self.conv1(x, edge_index)
        x1 = x1.relu()
        x2 = self.conv2(x1, edge_index)
        x2 = x2.relu()
        x3 = self.conv3(x2, edge_index)

        # 2. Readout layer [batch_size, hidden_channels]
        x = global_max_pool(x3, batch) + \
            global_add_pool(x3, batch) + \
            global_mean_pool(x3, batch)# [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.lin(x)
        
        return x

class BitterGraphSAGE_Baseline(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(BitterGraphSAGE_Baseline, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = SAGEConv(20, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, 2)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x1 = self.conv1(x, edge_index)
        x1 = x1.relu()
        x2 = self.conv2(x1, edge_index)
        x2 = x2.relu()
        #x = F.dropout(x, p = 0.1, training=self.training)
        x3 = self.conv3(x2, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x3, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.lin(x)
        
        return x

class BitterGraphSAGE_MixedPool(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(BitterGraphSAGE_MixedPool, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = SAGEConv(20, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, 2)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x1 = self.conv1(x, edge_index)
        x1 = x1.relu()
        x2 = self.conv2(x1, edge_index)
        x2 = x2.relu()
        #x = F.dropout(x, p = 0.1, training=self.training)
        x3 = self.conv3(x2, edge_index)

        # 2. Readout layer [batch_size, hidden_channels]
        x = global_max_pool(x3, batch) + \
            global_add_pool(x3, batch) + \
            global_mean_pool(x3, batch)# [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.lin(x)
        
        return x

