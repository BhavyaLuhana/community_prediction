import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class TemporalGNN(nn.Module):
    def __init__(self, node_features=1, hidden_dim=16, num_classes=5):
        super().__init__()
        self.gcn = GCNConv(node_features, hidden_dim)
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, sequence):
        embeddings = []
        for data in sequence:
            x = self.gcn(data.x, data.edge_index)
            x = torch.relu(x)
            embeddings.append(x.mean(dim=0, keepdim=True))  # [1, hidden_dim]

        rnn_input = torch.stack(embeddings, dim=1)  # [1, time_steps, hidden_dim]
        _, h_n = self.rnn(rnn_input)  # h_n: [1, 1, hidden_dim]
        final_hidden = torch.clamp(h_n.squeeze(0), -10, 10)  # Prevent exploding values
        return self.classifier(final_hidden)  # [1, num_classes]
