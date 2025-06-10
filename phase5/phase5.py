import sys
import os
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.manifold import TSNE
import community as community_louvain  # pip install python-louvain

# Add parent directory to Python path to access 'temporal' package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from temporal.temporal_data import load_temporal_data
from temporal.temporal_model import TemporalGNN

# Set up output folder
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

metrics_data = []

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("🔁 Loading temporal snapshots...")
    dataset = load_temporal_data()

    snapshot_0 = dataset[0].to(device)
    model = TemporalGNN(node_features=snapshot_0.num_node_features).to(device)

    try:
        model.load_state_dict(torch.load(os.path.join("..", "temporal_model.pt")))
        print("✅ Loaded trained model weights")
    except FileNotFoundError:
        print("⚠️ Warning: No trained model found. Using random weights")

    model.eval()

    for i, snapshot in enumerate(dataset):
        snapshot = snapshot.to(device)
        with torch.no_grad():
            x = model.gcn(snapshot.x, snapshot.edge_index)
            x = torch.relu(x)
            node_embeddings = x.cpu().numpy()

        # KMeans Clustering
        kmeans = KMeans(n_clusters=5, n_init=10, random_state=42)
        pred_labels = kmeans.fit_predict(node_embeddings)

        # Louvain Ground Truth
        edge_index = snapshot.edge_index.cpu().numpy()
        G = nx.Graph()
        G.add_edges_from(zip(edge_index[0], edge_index[1]))
        partition = community_louvain.best_partition(G)
        true_labels = np.array([partition.get(n, -1) for n in range(snapshot.num_nodes)])

        # Filter valid nodes
        valid_indices = true_labels != -1
        filtered_true = true_labels[valid_indices]
        filtered_pred = pred_labels[valid_indices]

        # Evaluation Metrics
        ari = adjusted_rand_score(filtered_true, filtered_pred)
        nmi = normalized_mutual_info_score(filtered_true, filtered_pred)

        # Modularity using predicted labels (with safe fallback)
        pred_communities = defaultdict(list)
        present_nodes = set(G.nodes)
        for node, comm in enumerate(pred_labels):
            if node in present_nodes:
                pred_communities[comm].append(node)

        # Prepare valid community list
        assigned_nodes = set()
        community_list = []
        for comm_nodes in pred_communities.values():
            filtered_nodes = [n for n in comm_nodes if n in present_nodes]
            if filtered_nodes:
                community_list.append(filtered_nodes)
                assigned_nodes.update(filtered_nodes)

        if assigned_nodes != present_nodes:
            print(f"⚠️ Warning: Skipping modularity for T{i+1} due to partial node coverage.")
            modularity = np.nan
        else:
            modularity = nx.community.modularity(G, community_list)

        print(f"\n📊 Snapshot T{i+1}: ARI={ari:.3f}, NMI={nmi:.3f}, Modularity={modularity:.3f}")
        metrics_data.append({
            "Snapshot": f"T{i+1}",
            "ARI": ari,
            "NMI": nmi,
            "Modularity": modularity
        })

    # Save CSV
    df = pd.DataFrame(metrics_data)
    csv_path = os.path.join(OUTPUT_DIR, "evaluation_scores_table.csv")
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved: {csv_path}")

    # Plot scores over time
    plot_scores(df)

    # Generate discussion summary
    generate_discussion(df)

def plot_scores(df):
    plt.figure(figsize=(10, 6))
    plt.plot(df["Snapshot"], df["ARI"], marker='o', label="ARI")
    plt.plot(df["Snapshot"], df["NMI"], marker='o', label="NMI")
    plt.plot(df["Snapshot"], df["Modularity"], marker='o', label="Modularity")
    plt.title("Community Evaluation Over Time")
    plt.xlabel("Snapshot")
    plt.ylabel("Score")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(OUTPUT_DIR, "community_evaluation_plot.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"✅ Saved: {plot_path}")

def generate_discussion(df):
    lines = ["📄 **Discussion Notes**\n"]
    trends = []

    for metric in ["ARI", "NMI", "Modularity"]:
        values = df[metric].values
        trend = "increasing" if np.all(np.diff(values) > 0) else (
                "decreasing" if np.all(np.diff(values) < 0) else "fluctuating")
        trends.append(f"- {metric} is {trend} over time.")

    lines += trends
    lines.append("\n📌 Notable Observations:")
    for i in range(1, len(df)):
        drop = False
        for metric in ["ARI", "NMI", "Modularity"]:
            if not pd.isna(df[metric][i]) and not pd.isna(df[metric][i-1]):
                if df[metric][i] < df[metric][i-1] - 0.1:
                    lines.append(f"- Sudden drop in {metric} from T{i} to T{i+1}.")
                    drop = True
        if not drop:
            lines.append(f"- No major drop from T{i} to T{i+1}.")

    summary_path = os.path.join(OUTPUT_DIR, "discussion_notes.txt")
    with open(summary_path, "w", encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f"✅ Saved: {summary_path}")

if __name__ == "__main__":
    main()
