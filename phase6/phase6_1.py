import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import networkx as nx
import random
import psutil
import pynvml
import numpy as np
from torch_geometric.utils import from_networkx
from torch_geometric.loader import DataLoader
from temporal.temporal_model import TemporalGNN
from temporal.temporal_data import load_temporal_data
from sklearn.metrics import adjusted_rand_score
import community as community_louvain

# Training loop full-batch
def train_full_batch(model, sequence, device, epochs=10):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    start_time = time.time()
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    max_gpu_mem = 0

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        out = model(sequence)
        target = torch.tensor([random.randint(0, 4)], dtype=torch.long).to(device)  
        loss = loss_fn(out, target)
        loss.backward()
        optimizer.step()
        print(f"\nEpoch {epoch:02d} | Loss: {loss.item():.4f}")

        mem = pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1024**2
        max_gpu_mem = max(max_gpu_mem, mem)

    end_time = time.time()
    avg_time = (end_time - start_time) / epochs
    pynvml.nvmlShutdown()

    return avg_time, max_gpu_mem, loss.item()

def run_full_batch():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = load_temporal_data()
    model = TemporalGNN(node_features=dataset[0].num_node_features).to(device)
    sequence = [snapshot.to(device) for snapshot in dataset]

    avg_time, max_gpu_mem, final_loss = train_full_batch(model, sequence, device)

    snapshot = dataset[-1]
    snapshot = snapshot.to(device)
    with torch.no_grad():
        x = model.gcn(snapshot.x, snapshot.edge_index)
        pred_labels = torch.argmax(x, dim=1).cpu().numpy()

    edge_index = snapshot.edge_index.cpu().numpy()
    G = nx.Graph()
    G.add_edges_from(zip(edge_index[0], edge_index[1]))
    partition = community_louvain.best_partition(G)
    true_labels = np.array([partition.get(n, -1) for n in range(snapshot.num_nodes)])
    valid = true_labels != -1
    ari = adjusted_rand_score(true_labels[valid], pred_labels[valid])

    print("\nPhase 6.1 Results")
    print("----------------------------")
    print(f"Nodes: {snapshot.num_nodes}")
    print(f"Epoch Time (s): {avg_time:.2f}")
    print(f"Max GPU (MB): {max_gpu_mem:.0f}")
    print(f"Final Loss: {final_loss:.4f}")
    print(f"ARI (T10): {ari:.4f}")

if __name__ == "__main__":
    run_full_batch()