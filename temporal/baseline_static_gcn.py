# baseline_static_gcn.py
# Evaluate a simple 2-layer GCN on Snapshot T10 as a baseline for temporal GNN comparison.

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from torch_geometric.nn import GCNConv
import community.community_louvain as community_louvain
from temporal.temporal_data import load_temporal_data

class StaticGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

def run_static_gcn_baseline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n Running Static GCN Baseline on device: {device}")

    all_snapshots = load_temporal_data()
    snapshot = all_snapshots[-1].to(device)

    model = StaticGCN(in_channels=snapshot.num_node_features, hidden_channels=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    dummy_label = torch.randint(0, 5, (snapshot.num_nodes,), dtype=torch.long).to(device)

    #  Training for 200 epochs
    model.train()
    for epoch in range(1, 201):
        optimizer.zero_grad()
        out = model(snapshot.x, snapshot.edge_index)
        loss = F.cross_entropy(out, dummy_label)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        embeddings = model(snapshot.x, snapshot.edge_index).cpu().numpy()

    pred_labels = KMeans(n_clusters=5, n_init=10, random_state=42).fit_predict(embeddings)

    #  Ground Truth via Louvain
    edge_index = snapshot.edge_index.cpu().numpy()
    G = nx.Graph()
    G.add_edges_from(zip(edge_index[0], edge_index[1]))
    partition = community_louvain.best_partition(G)
    true_labels = np.array([partition.get(n, -1) for n in range(snapshot.num_nodes)])
    valid = true_labels != -1

    ari = adjusted_rand_score(true_labels[valid], pred_labels[valid])
    nmi = normalized_mutual_info_score(true_labels[valid], pred_labels[valid])

    #  Final Results
    print("\n Static GCN Baseline Results (Snapshot T10)")
    print("--------------------------------------------------")
    print(f"Adjusted Rand Index (ARI): {ari:.4f}")
    print(f"Normalized Mutual Information (NMI): {nmi:.4f}")

if __name__ == "__main__":
    run_static_gcn_baseline()
