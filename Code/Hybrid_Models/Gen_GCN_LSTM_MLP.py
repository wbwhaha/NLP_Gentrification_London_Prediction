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

        # GCN layers over spatial graph
        self.gcn1 = GCNConv(lstm_hidden, gcn_hidden)
        self.bn1 = nn.BatchNorm1d(gcn_hidden)  # BatchNorm after first GCN

        self.gcn2 = GCNConv(gcn_hidden, gcn_hidden)
        self.bn2 = nn.BatchNorm1d(gcn_hidden)  # BatchNorm after second GCN

        # MLP Classifier
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(gcn_hidden, gcn_hidden // 2),
            nn.ReLU(),
            nn.Linear(gcn_hidden // 2, num_classes)
        )

    def forward(self, x_seq, edge_index, edge_weight):
        """
        x_seq: Tensor of shape [n_lsoa, T, input_dim]
        edge_index: Graph edges (PyG format)
        edge_weight: Edge weights (e.g. shared boundary length)
        """

        # Temporal encoding per node via LSTM
        lstm_out, _ = self.lstm(x_seq)          # [n_lsoa, T, lstm_hidden]
        x = lstm_out[:, -1, :]                  # Last time step as temporal-semantic feature

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