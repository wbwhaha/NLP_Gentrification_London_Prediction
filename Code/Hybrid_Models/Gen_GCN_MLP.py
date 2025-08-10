import torch
from torch import nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

class GentrificationGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout_rate, num_classes):
        super().__init__()

        # First GCN layer over spatial graph
        self.gcn1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)  # BatchNorm after first GCN

        # Second GCN layer over spatial graph
        self.gcn2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)  # BatchNorm after second GCN

        # MLP Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, num_classes)
        )

    def forward(self, x, edge_index, edge_weight):
        """
        x: Node feature matrix [n_nodes, in_channels]
        edge_index: Graph edges (PyG format)
        edge_weight: Edge weights (e.g. shared boundary length)
        """

        # First GCN + BN + ReLU
        x = self.gcn1(x, edge_index, edge_weight)
        x = self.bn1(x)
        x = F.relu(x)

        # Second GCN + BN (Residual connection)
        x_res = self.gcn2(x, edge_index, edge_weight)
        x_res = self.bn2(x_res)
        x = x + x_res  # residual connection

        # Predict
        return self.classifier(x)