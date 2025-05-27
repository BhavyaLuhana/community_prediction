import networkx as nx
import matplotlib.pyplot as plt

def visualize_snapshot(snapshot_id):
    with open(f"data/snapshots/snapshot_{snapshot_id}.gpickle", "rb") as f:
        G = pickle.load(f)
    nx.draw(G, with_labels=True)
    plt.title(f"Snapshot {snapshot_id}")
    plt.show()