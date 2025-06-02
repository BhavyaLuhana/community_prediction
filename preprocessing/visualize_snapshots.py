import pickle
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path

def build_consistent_layout(snapshot_dir, num_snapshots, seed=42):
    """
    Build a consistent node layout across all snapshots using the union of all nodes.
    """
    all_nodes = set()
    for i in range(1, num_snapshots + 1):
        with open(snapshot_dir / f"snapshot_{i}.gpickle", "rb") as f:
            G = pickle.load(f)
            all_nodes.update(G.nodes())

    base_graph = nx.Graph()
    base_graph.add_nodes_from(all_nodes)
    layout = nx.spring_layout(base_graph, seed=seed)
    return layout

def visualize_snapshots(snapshot_dir, num_snapshots=10, layout_type="consistent"):
    snapshot_dir = Path(snapshot_dir)
    assert snapshot_dir.exists(), f"Snapshot directory not found: {snapshot_dir}"

    print("Generating layout...")
    layout_global = None
    if layout_type == "consistent":
        layout_global = build_consistent_layout(snapshot_dir, num_snapshots)

    for i in range(1, num_snapshots + 1):
        snapshot_path = snapshot_dir / f"snapshot_{i}.gpickle"
        if not snapshot_path.exists():
            print(f"[Warning] Snapshot {i} not found, skipping.")
            continue

        with open(snapshot_path, "rb") as f:
            G = pickle.load(f)

        plt.figure(figsize=(6, 5))
        plt.title(f"Snapshot {i} - Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

        # Choose layout
        if layout_type == "consistent":
            pos = {node: layout_global[node] for node in G.nodes() if node in layout_global}
        else:
            pos = nx.spring_layout(G, seed=42)

        nx.draw_networkx(
            G, pos,
            node_size=20,
            with_labels=False,
            edge_color="gray"
        )

        plt.axis("off")
        plt.tight_layout()
        save_path = Path(f"snapshot_{i}.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"✅ Saved snapshot {i} to {save_path}")

if __name__ == "__main__":
    snapshot_folder = Path(__file__).parent.parent / "data" / "snapshots"
    visualize_snapshots(snapshot_folder, num_snapshots=10, layout_type="consistent")
