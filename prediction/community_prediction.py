import sys
import os
import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, f1_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from collections import defaultdict

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from temporal.temporal_data import load_temporal_data
from temporal.temporal_model import TemporalGNN

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # === Step 4.1: Load Data and Generate Node Embeddings ===
    print("Loading temporal snapshots and generating embeddings...")
    dataset = load_temporal_data()
    
    # Initialize model
    snapshot_0 = dataset[0].to(device)
    model = TemporalGNN(node_features=snapshot_0.num_node_features).to(device)
    
    # Load trained model weights (assuming you saved them during training)
    try:
        model.load_state_dict(torch.load("temporal_model.pt"))
        print("Loaded trained model weights")
    except FileNotFoundError:
        print("Warning: No trained model found. Using random initialization")
    
    model.eval()
    
    # Store all embeddings and predictions
    all_embeddings = []
    all_pred_labels = []
    all_true_labels = []
    
    # Process each snapshot
    for i, snapshot in enumerate(dataset):
        snapshot = snapshot.to(device)
        
        # Generate embeddings
        with torch.no_grad():
            x = model.gcn(snapshot.x, snapshot.edge_index)
            x = torch.relu(x)
            node_embeddings = x.cpu().numpy()
            all_embeddings.append(node_embeddings)
            
            # Simulate ground truth (5 communities)
            true_labels = np.array([node_id % 5 for node_id in range(snapshot.num_nodes)])
            all_true_labels.append(true_labels)
    
    # === Step 4.2: Cluster Nodes into Communities ===
    print("\nClustering nodes into communities...")
    n_clusters = 5  # Assuming 5 communities based on our simulation
    
    for i, embeddings in enumerate(all_embeddings):
        # Cluster using KMeans (as suggested in the PDF)
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        pred_labels = kmeans.fit_predict(embeddings)
        all_pred_labels.append(pred_labels)
        
        # === Step 4.4: Evaluate Cluster Quality ===
        ari = adjusted_rand_score(all_true_labels[i], pred_labels)
        nmi = normalized_mutual_info_score(all_true_labels[i], pred_labels)
        f1 = f1_score(all_true_labels[i], pred_labels, average='macro')
        
        print(f"\n📊 Snapshot {i+1} Community Clustering Evaluation:")
        print(f"Adjusted Rand Index (ARI): {ari:.4f}")
        print(f"Normalized Mutual Information (NMI): {nmi:.4f}")
        print(f"Macro F1 Score: {f1:.4f}")
    
    # === Visualization (Step 4.4 Bonus) ===
    print("\nGenerating visualizations...")
    visualize_embeddings(all_embeddings[-1], all_pred_labels[-1], "final_snapshot")
    
    # === Temporal Consistency Check (Step 4.4.3) ===
    analyze_temporal_consistency(all_pred_labels)

def visualize_embeddings(embeddings, labels, name):
    """Visualize embeddings using t-SNE"""
    tsne = TSNE(n_components=2, random_state=42)
    embedding_2d = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    plt.title(f"t-SNE of Node Embeddings ({name}) with Community Labels")
    scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], 
                         c=labels, cmap='tab10', s=15, alpha=0.7)
    plt.colorbar(scatter, label="Community")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{name}_communities.png", dpi=300)
    plt.close()
    print(f"Saved visualization: {name}_communities.png")

def analyze_temporal_consistency(all_pred_labels):
    """Analyze how communities evolve over time"""
    print("\nAnalyzing temporal community evolution...")
    
    # Track community sizes over time
    community_sizes = []
    for labels in all_pred_labels:
        unique, counts = np.unique(labels, return_counts=True)
        community_sizes.append(dict(zip(unique, counts)))
    
    # Plot community size evolution
    plt.figure(figsize=(10, 6))
    for comm_id in range(5):  # Assuming 5 communities
        sizes = [sizes.get(comm_id, 0) for sizes in community_sizes]
        plt.plot(range(1, len(sizes)+1), sizes, label=f"Community {comm_id}")
    
    plt.title("Community Size Evolution Over Time")
    plt.xlabel("Snapshot")
    plt.ylabel("Number of Nodes")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("community_evolution.png", dpi=300)
    plt.close()
    print("Saved community evolution plot: community_evolution.png")
    
    # Calculate Jaccard similarity between consecutive snapshots
    jaccard_similarities = []
    for i in range(len(all_pred_labels)-1):
        # Create sets of nodes for each community
        prev_comms = defaultdict(set)
        curr_comms = defaultdict(set)
        
        for node_id, comm_id in enumerate(all_pred_labels[i]):
            prev_comms[comm_id].add(node_id)
            
        for node_id, comm_id in enumerate(all_pred_labels[i+1]):
            curr_comms[comm_id].add(node_id)
        
        # Calculate max Jaccard similarity between current and previous communities
        max_similarities = []
        for curr_comm, curr_nodes in curr_comms.items():
            similarities = []
            for prev_comm, prev_nodes in prev_comms.items():
                intersection = len(curr_nodes & prev_nodes)
                union = len(curr_nodes | prev_nodes)
                similarities.append(intersection / union if union > 0 else 0)
            max_similarities.append(max(similarities) if similarities else 0)
        
        avg_similarity = np.mean(max_similarities)
        jaccard_similarities.append(avg_similarity)
    
    print("\nCommunity Similarity Between Consecutive Snapshots:")
    for i, sim in enumerate(jaccard_similarities, 1):
        print(f"Snapshot {i} to {i+1}: {sim:.3f}")

if __name__ == "__main__":
    main()