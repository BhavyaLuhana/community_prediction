import os
import pickle
import numpy as np
from pathlib import Path

def main():
    BASE_DIR = Path(__file__).parent
    SNAP_DIR = BASE_DIR.parent / "data" / "snapshots"
    SAVE_DIR = BASE_DIR.parent / "data" / "adj_matrices"
    
    if not SNAP_DIR.exists():
        raise FileNotFoundError(f"Snapshots directory not found at: {SNAP_DIR}")
    
    # Create output directory
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get all snapshot files sorted numerically
    snapshot_files = sorted(SNAP_DIR.glob("snapshot_*.gpickle"), 
                          key=lambda x: int(x.stem.split("_")[-1]))
    
    if not snapshot_files:
        raise FileNotFoundError(f"No snapshot files found in {SNAP_DIR}")
    
    print("\n=== Processing Snapshots ===")
    node_index = {}
    
    for i, snapshot_path in enumerate(snapshot_files, 1):
        try:
            print(f"\nProcessing {snapshot_path.name}...")
            
            # Load graph
            with open(snapshot_path, "rb") as f:
                G = pickle.load(f)
            
            # Create node index from snapshot
            if not node_index:
                nodes = sorted(G.nodes())
                node_index = {node: idx for idx, node in enumerate(nodes)}
                print(f"Created node index with {len(node_index)} nodes")
            
            # Initialize adjacency matrix
            adj = np.zeros((len(node_index), len(node_index)))
            
            # Fill adjacency matrix
            for u, v in G.edges():
                if u in node_index and v in node_index:
                    i, j = node_index[u], node_index[v]
                    adj[i][j] = 1
                    adj[j][i] = 1  
            
            # Save matrix
            output_path = SAVE_DIR / f"adj_snapshot_{i}.npy"
            np.save(output_path, adj)
            print(f"Saved adjacency matrix to {output_path.name}")
            
        except Exception as e:
            print(f"Error processing {snapshot_path.name}: {str(e)}")
            continue
    
    print("\nProcessing complete!")
    print(f"Saved {len(snapshot_files)} adjacency matrices to {SAVE_DIR}")

if __name__ == "__main__":
    main()