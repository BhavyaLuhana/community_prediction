import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import random
import networkx as nx
import numpy as np
import pynvml
from torch_geometric.utils import from_networkx
from temporal.temporal_model import TemporalGNN 
from sklearn.metrics import adjusted_rand_score
import community as community_louvain

# Synthetic SBM Generator 
def generate_sbm_snapshots(num_snapshots=10, num_nodes=10000, num_communities=5):
    snapshots = []
    for t in range(num_snapshots):
        sizes = [num_nodes // num_communities] * num_communities
        probs = [[0.01 if i == j else 0.001 for j in range(num_communities)] for i in range(num_communities)]
        G = nx.stochastic_block_model(sizes, probs, seed=random.randint(0, 9999))
        data = from_networkx(G)
        data.x = torch.randn((num_nodes, 32)) 
        snapshots.append(data)
    return snapshots

# Full-Batch Training
def train_temporal_gnn(snapshots, device):
    model = TemporalGNN(node_features=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    max_gpu_mem = 0
    start_time = time.time()

    for epoch in range(10):
        optimizer.zero_grad()
        out = model([s.to(device) for s in snapshots])
        label = torch.tensor([random.randint(0, 4)], dtype=torch.long).to(device)
        loss = loss_fn(out, label)
        loss.backward()
        optimizer.step()

        mem_used = pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1024**2
        max_gpu_mem = max(max_gpu_mem, mem_used)
        print(f"Epoch {epoch+1:02d} | Loss: {loss.item():.4f}")

    pynvml.nvmlShutdown()
    avg_time = (time.time() - start_time) / 10
    return model, avg_time, max_gpu_mem, loss.item()

def evaluate_on_last_snapshot(model, snapshot, device):
    model.eval()
    with torch.no_grad():
        x = model.gcn(snapshot.x.to(device), snapshot.edge_index.to(device))
        pred_labels = torch.argmax(x, dim=1).cpu().numpy()

    edge_index = snapshot.edge_index.cpu().numpy()
    G = nx.Graph()
    G.add_edges_from(zip(edge_index[0], edge_index[1]))
    partition = community_louvain.best_partition(G)
    true_labels = np.array([partition.get(n, -1) for n in range(snapshot.num_nodes)])
    valid = true_labels != -1
    ari = adjusted_rand_score(true_labels[valid], pred_labels[valid])
    return ari

def run_phase6_2():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Using device: {device}")

    print("\n Generating synthetic dynamic SBM graph (10 snapshots, 10k nodes)")
    snapshots = generate_sbm_snapshots()

    print("\n Training TemporalGNN with full-batch...")
    model, avg_time, max_gpu_mem, final_loss = train_temporal_gnn(snapshots, device)

    print("\n Evaluating ARI on final snapshot (T10)...")
    ari = evaluate_on_last_snapshot(model, snapshots[-1], device)

    print("\n Results")
    print("------------------------------------")
    print(f"Nodes: {snapshots[-1].num_nodes}")
    print(f"Epoch Time (s): {avg_time:.2f}")
    print(f"Max GPU (MB): {max_gpu_mem:.0f}")
    print(f"Final Loss: {final_loss:.4f}")
    print(f"ARI (T10): {ari:.4f}")

if __name__ == "__main__":
    run_phase6_2()
