import torch

from torch import nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

class GentrificationGCN_LSTM(nn.Module):
    def __init__(self, input_dim, lstm_hidden, gcn_hidden, dropout_rate, num_classes):
        super().__init__()

        # LSTM over temporal text features (per LSOA)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden,
            batch_first=True
        )

        # GCN over spatial graph of LSOAs
        self.gcn1 = GCNConv(lstm_hidden, gcn_hidden)
        self.gcn2 = GCNConv(gcn_hidden, gcn_hidden)

        # Classifier
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(gcn_hidden, gcn_hidden),
            nn.ReLU(),
            nn.Linear(gcn_hidden, num_classes)
        )

    def forward(self, x_seq, edge_index, edge_weight):
        """
        x_seq: Tensor of shape [n_lsoa, T, input_dim]
        edge_index: Graph edges (PyG format)
        edge_weight: Edge weights (e.g. shared boundary length)
        """

        # LSTM per node
        lstm_out, _ = self.lstm(x_seq)  # [n_lsoa, T, lstm_hidden]
        x = lstm_out[:, -1, :]          # Take the last time step as the semantic evolution feature [n_lsoa, lstm_hidden]

        # GCN over spatial structure
        x = F.relu(self.gcn1(x, edge_index, edge_weight))
        x = self.gcn2(x, edge_index, edge_weight)

        # Classifier
        return self.classifier(x)
