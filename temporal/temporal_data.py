import numpy as np
import torch
from torch_geometric.data import Data
from pathlib import Path

def load_temporal_data():
    BASE_DIR = Path(__file__).parent
    adj_dir = BASE_DIR.parent / "data" / "adj_matrices"
    feat_dir = BASE_DIR.parent / "data" / "feat_matrices"

    snapshots = []

    for i in range(1, 11):
        adj = np.load(adj_dir / f"adj_snapshot_{i}.npy")
        feat = np.load(feat_dir / f"feat_snapshot_{i}.npy", allow_pickle=True)

        if feat.ndim == 1:
            feat = np.sum(adj, axis=1).reshape(-1, 1)

        feat = (feat - feat.mean()) / (feat.std() + 1e-5)
        edge_index = torch.tensor(np.stack(np.where(adj > 0)), dtype=torch.long)

        if edge_index.shape[1] == 0:
            print(f"[Warning] Snapshot {i} has no edges, adding dummy self-loop.")
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)

        x = torch.tensor(feat, dtype=torch.float)
        y = torch.tensor([i % 5], dtype=torch.long)  # Placeholder label

        snapshots.append(Data(x=x, edge_index=edge_index, y=y))
        print(f"Loaded snapshot {i} with {x.shape[0]} nodes")

    return snapshots
