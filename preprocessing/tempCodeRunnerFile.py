import os
import pickle
import numpy as np
from pathlib import Path

def main():
    BASE_DIR = Path(__file__).parent
    FEAT_DIR = BASE_DIR.parent / "data" / "facebook"
    SNAP_DIR = BASE_DIR.parent / "data" / "snapshots"
    SAVE_DIR = BASE_DIR.parent / "data" / "feat_matrices"
    
    # output directory
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Verify feature directory exists
    if not FEAT_DIR.exists():
        raise FileNotFoundError(
            f"Facebook feature directory not found at: {FEAT_DIR}\n"
            "Please ensure you have downloaded the Facebook dataset files.\n"
            "Expected files: [0-9]+.feat and [0-9]+.egofeat in this directory."
        )
    
    # Build global node-to-feature map
    node_feat_map = {}
    print("\n=== Processing Feature Files ===")
    
    for feat_file in FEAT_DIR.glob("*.feat"):
        try:
            ego_node = int(feat_file.stem)
            features = np.loadtxt(feat_file)
            for row in features:
                node_id = int(row[0])
                node_feat_map[node_id] = row[1:]  # skip index
            print(f"Processed {feat_file.name} with {len(features)} nodes")
        except Exception as e:
            print(f"Error processing {feat_file.name}: {str(e)}")
            continue
    
    if not node_feat_map:
        raise ValueError("No valid feature files found in the directory")
    
    # Process snapshots
    print("\n=== Creating Feature Matrices ===")
    for idx in range(1, 11):
        try:
            snap_path = SNAP_DIR / f"snapshot_{idx}.gpickle"
            with open(snap_path, "rb") as f:
                G = pickle.load(f)
            
            nodes = sorted(G.nodes())
            feat_dim = len(next(iter(node_feat_map.values())))
            feat_matrix = np.zeros((len(nodes), feat_dim))
            
            for i, node in enumerate(nodes):
                feat_matrix[i] = node_feat_map.get(node, np.zeros(feat_dim))
            
            output_path = SAVE_DIR / f"feat_snapshot_{idx}.npy"
            np.save(output_path, feat_matrix)
            print(f"Created feature matrix for snapshot {idx} with shape {feat_matrix.shape}")
            
        except Exception as e:
            print(f"Error processing snapshot {idx}: {str(e)}")
            continue
    
    print("\n Feature matrices saved successfully!")

if __name__ == "__main__":
    main()