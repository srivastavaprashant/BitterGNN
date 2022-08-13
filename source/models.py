import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GraphConv
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
        # x = F.dropout(x, p = 0.1, training=self.training)
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
        x = (
            global_max_pool(x3, batch)
            + global_add_pool(x3, batch)
            + global_mean_pool(x3, batch)
        )  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.lin(x)

        return x


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
        x = (
            global_max_pool(x3, batch)
            + global_add_pool(x3, batch)
            + global_mean_pool(x3, batch)
        )  # [batch_size, hidden_channels]

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
        # x = F.dropout(x, p = 0.1, training=self.training)
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
        # x = F.dropout(x, p = 0.1, training=self.training)
        x3 = self.conv3(x2, edge_index)

        # 2. Readout layer [batch_size, hidden_channels]
        x = (
            global_max_pool(x3, batch)
            + global_add_pool(x3, batch)
            + global_mean_pool(x3, batch)
        )  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.lin(x)

        return x
