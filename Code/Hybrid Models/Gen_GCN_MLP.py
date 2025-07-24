import torch

from torch import nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

class GentrificationGCN(nn.Module):

    def __init__(self, in_channels, hidden_channels, dropout_rate, num_classes):
        super().__init__()
        self.gcn1 = GCNConv(in_channels, hidden_channels)
        self.gcn2 = GCNConv(hidden_channels, hidden_channels)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, num_classes)
        )

    def forward(self, x, edge_index, edge_weight):
        
        x = self.gcn1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.gcn2(x, edge_index, edge_weight)

        return self.classifier(x)