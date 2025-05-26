import random
import os
import networkx as nx
import pickle
from pathlib import Path

BASE_DIR = Path(__file__).parent
EDGE_PATH = BASE_DIR.parent / "data" / "facebook_combined.txt"
SAVE_DIR = BASE_DIR.parent / "data" / "snapshots"

SAVE_DIR.mkdir(parents=True, exist_ok=True)

print(f"Input file location: {EDGE_PATH}")
print(f"Output directory: {SAVE_DIR}")

# Verify input file exists
if not EDGE_PATH.exists():
    raise FileNotFoundError(f"Input file not found at: {EDGE_PATH}")

# Load the full graph
try:
    G_full = nx.read_edgelist(str(EDGE_PATH), nodetype=int)
    print(f"Loaded graph with {len(G_full.nodes())} nodes and {len(G_full.edges())} edges")
except Exception as e:
    raise RuntimeError(f"Error loading graph: {str(e)}")

# Create snapshots
edges = list(G_full.edges())
random.shuffle(edges)

n_parts = 10
chunk_size = len(edges) // n_parts
snapshots = []

print("\nGenerating snapshots...")
for i in range(1, n_parts + 1):
    edge_subset = edges[:i * chunk_size]
    G_snapshot = nx.Graph()
    G_snapshot.add_edges_from(edge_subset)
    
    snapshot_path = SAVE_DIR / f"snapshot_{i}.gpickle"
    with open(snapshot_path, "wb") as f:
        pickle.dump(G_snapshot, f)
    print(f"Created snapshot {i} with {len(G_snapshot.edges())} edges")
    snapshots.append(G_snapshot)

print(f"\n Successfully saved {n_parts} snapshots to {SAVE_DIR}")
print("Files created:", [f.name for f in SAVE_DIR.glob("*.gpickle")])