import sys
import os
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, f1_score
from sklearn.manifold import TSNE
import community as community_louvain  # pip install python-louvain

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from temporal.temporal_data import load_temporal_data
from temporal.temporal_model import TemporalGNN

metrics_data = []  # For saving ARI/NMI/F1
jaccard_data = []  # For saving Jaccard similarity

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading temporal snapshots and generating embeddings...")
    dataset = load_temporal_data()

    snapshot_0 = dataset[0].to(device)
    model = TemporalGNN(node_features=snapshot_0.num_node_features).to(device)

    # Load trained model
    try:
        model.load_state_dict(torch.load("temporal_model.pt"))
        print("✅ Loaded trained model weights")
    except FileNotFoundError:
        print("⚠️ Warning: No trained model found. Using random initialization")

    model.eval()

    all_embeddings, all_pred_labels, all_true_labels = [], [], []

    for i, snapshot in enumerate(dataset):
        snapshot = snapshot.to(device)
        with torch.no_grad():
            x = model.gcn(snapshot.x, snapshot.edge_index)
            x = torch.relu(x)
            node_embeddings = x.cpu().numpy()
            all_embeddings.append(node_embeddings)

        # Louvain community detection
        edge_index = snapshot.edge_index.cpu().numpy()
        G = nx.Graph()
        G.add_edges_from(zip(edge_index[0], edge_index[1]))
        partition = community_louvain.best_partition(G)

        # Assign community or fallback to -1
        true_labels = []
        for n in range(snapshot.num_nodes):
            if n in partition:
                true_labels.append(partition[n])
            else:
                true_labels.append(-1)
        true_labels = np.array(true_labels)
        all_true_labels.append(true_labels)

    print("\nClustering nodes into communities...")
    n_clusters = 5

    for i, embeddings in enumerate(all_embeddings):
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        pred_labels = kmeans.fit_predict(embeddings)
        all_pred_labels.append(pred_labels)

        # Filter out nodes with -1 label
        valid_indices = all_true_labels[i] != -1
        filtered_true = all_true_labels[i][valid_indices]
        filtered_pred = pred_labels[valid_indices]

        ari = adjusted_rand_score(filtered_true, filtered_pred)
        nmi = normalized_mutual_info_score(filtered_true, filtered_pred)
        f1 = f1_score(filtered_true, filtered_pred, average='macro')

        metrics_data.append({
            "Snapshot": i + 1,
            "ARI": ari,
            "NMI": nmi,
            "F1": f1
        })

        print(f"\n📊 Snapshot {i+1} Evaluation:")
        print(f"ARI: {ari:.4f} | NMI: {nmi:.4f} | F1: {f1:.4f}")

    print("\n📉 Generating t-SNE visualization for final snapshot...")
    visualize_embeddings(all_embeddings[-1], all_pred_labels[-1], "final_snapshot")

    print("📊 Analyzing community evolution...")
    analyze_temporal_consistency(all_pred_labels)

    # Save CSVs
    metrics_df = pd.DataFrame(metrics_data)
    jaccard_df = pd.DataFrame(jaccard_data)
    metrics_df.to_csv("snapshot_metrics.csv", index=False)
    jaccard_df.to_csv("jaccard_similarities.csv", index=False)
    print("📁 CSVs saved: 'snapshot_metrics.csv' and 'jaccard_similarities.csv'")


def visualize_embeddings(embeddings, labels, name):
    tsne = TSNE(n_components=2, random_state=42)
    embedding_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    plt.title(f"t-SNE of Node Embeddings - {name}")
    scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=labels, cmap='tab10', s=15, alpha=0.7)
    plt.colorbar(scatter, label="Community")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{name}_communities.png", dpi=300)
    plt.close()
    print(f"✅ Saved: {name}_communities.png")


def analyze_temporal_consistency(all_pred_labels):
    community_sizes = []
    for labels in all_pred_labels:
        unique, counts = np.unique(labels, return_counts=True)
        community_sizes.append(dict(zip(unique, counts)))

    plt.figure(figsize=(10, 6))
    for comm_id in range(5):
        sizes = [cs.get(comm_id, 0) for cs in community_sizes]
        plt.plot(range(1, len(sizes)+1), sizes, label=f"Community {comm_id}")

    plt.title("Community Size Over Time")
    plt.xlabel("Snapshot")
    plt.ylabel("Node Count")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("community_evolution.png", dpi=300)
    plt.close()
    print("✅ Saved: community_evolution.png")

    jaccard_similarities = []
    for i in range(len(all_pred_labels)-1):
        prev_comms, curr_comms = defaultdict(set), defaultdict(set)
        for node_id, comm_id in enumerate(all_pred_labels[i]):
            prev_comms[comm_id].add(node_id)
        for node_id, comm_id in enumerate(all_pred_labels[i+1]):
            curr_comms[comm_id].add(node_id)

        max_similarities = []
        for curr_nodes in curr_comms.values():
            similarities = []
            for prev_nodes in prev_comms.values():
                intersection = len(curr_nodes & prev_nodes)
                union = len(curr_nodes | prev_nodes)
                similarities.append(intersection / union if union > 0 else 0)
            max_similarities.append(max(similarities) if similarities else 0)

        avg_similarity = np.mean(max_similarities)
        jaccard_similarities.append(avg_similarity)

        jaccard_data.append({
            "From Snapshot": i + 1,
            "To Snapshot": i + 2,
            "Jaccard Similarity": avg_similarity
        })

    print("\n📈 Jaccard Similarity Between Snapshots:")
    for i, sim in enumerate(jaccard_similarities):
        print(f"Snapshot {i+1} → {i+2}: {sim:.3f}")


if __name__ == "__main__":
    main()