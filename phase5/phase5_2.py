import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns
import json

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(os.path.join(OUTPUT_DIR, "all_pred_labels.json"), "r") as f:
    all_pred_labels = json.load(f)

num_snapshots = len(all_pred_labels)
all_pred_labels = [np.array(labels) for labels in all_pred_labels] 

community_events = []
node_migrations = []

jaccard_threshold = 0.3

def jaccard(a, b):
    return len(a & b) / len(a | b) if len(a | b) > 0 else 0

def analyze_transitions():
    for t in range(num_snapshots - 1):
        curr_labels = all_pred_labels[t]
        next_labels = all_pred_labels[t + 1]

        curr_comms = defaultdict(set)
        next_comms = defaultdict(set)

        for node_id, label in enumerate(curr_labels):
            curr_comms[label].add(node_id)
        for node_id, label in enumerate(next_labels):
            next_comms[label].add(node_id)

        matched_curr = set()
        matched_next = set()

        births, deaths, splits, merges = 0, 0, 0, 0

        # Match communities
        for nc_id, nc_nodes in next_comms.items():
            max_jacc = 0
            matched_id = None
            for cc_id, cc_nodes in curr_comms.items():
                sim = jaccard(nc_nodes, cc_nodes)
                if sim > max_jacc:
                    max_jacc = sim
                    matched_id = cc_id
            if max_jacc > jaccard_threshold:
                matched_curr.add(matched_id)
                matched_next.add(nc_id)
            else:
                births += 1

        for cc_id in curr_comms:
            if cc_id not in matched_curr:
                deaths += 1

        # Splits and merges
        for cc_id, cc_nodes in curr_comms.items():
            overlap_count = 0
            for nc_nodes in next_comms.values():
                if jaccard(cc_nodes, nc_nodes) > jaccard_threshold:
                    overlap_count += 1
            if overlap_count > 1:
                splits += 1

        for nc_id, nc_nodes in next_comms.items():
            overlap_count = 0
            for cc_nodes in curr_comms.values():
                if jaccard(nc_nodes, cc_nodes) > jaccard_threshold:
                    overlap_count += 1
            if overlap_count > 1:
                merges += 1

        # Node migration (hanfling unequal snapshots)
        min_len = min(len(curr_labels), len(next_labels))
        migrated = np.sum(curr_labels[:min_len] != next_labels[:min_len])
        percent = 100.0 * migrated / min_len

        community_events.append({
            "Snapshot": f"T{t+1}-T{t+2}",
            "Births": births,
            "Deaths": deaths,
            "Splits": splits,
            "Merges": merges
        })

        node_migrations.append({
            "Snapshot": f"T{t+1}-T{t+2}",
            "# Migrated Nodes": migrated,
            "% Migrated": round(percent, 2)
        })

def save_outputs():
    pd.DataFrame(community_events).to_csv(os.path.join(OUTPUT_DIR, "community_events_table.csv"), index=False)
    pd.DataFrame(node_migrations).to_csv(os.path.join(OUTPUT_DIR, "node_migration_summary.csv"), index=False)
    print("✅ Saved: community_events_table.csv, node_migration_summary.csv")

    sankey_data = np.zeros((len(set(all_pred_labels[0])), len(set(all_pred_labels[-1]))))
    min_len = min(len(all_pred_labels[0]), len(all_pred_labels[-1]))
    for node_id in range(min_len):
        sankey_data[all_pred_labels[0][node_id]][all_pred_labels[-1][node_id]] += 1

    plt.figure(figsize=(8, 6))
    sns.heatmap(sankey_data, annot=True, fmt=".0f", cmap="Blues")
    plt.title("Community Flow from T1 to T10")
    plt.xlabel("T10 Clusters")
    plt.ylabel("T1 Clusters")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "sankey_community_flows.png"))
    plt.close()
    print("✅ Saved: sankey_community_flows.png")

    with open(os.path.join(OUTPUT_DIR, "discussion_notes.txt"), "w") as f:
        f.write("Community Dynamics Summary\n\n")
        for row in community_events:
            f.write(f"{row['Snapshot']}: Births={row['Births']}, Deaths={row['Deaths']}, Splits={row['Splits']}, Merges={row['Merges']}\n")
        f.write("\nNode Migration Summary:\n")
        for row in node_migrations:
            f.write(f"{row['Snapshot']}: {row['# Migrated Nodes']} nodes migrated ({row['% Migrated']}%)\n")
    print("✅ Saved: discussion_notes.txt")

if __name__ == "__main__":
    analyze_transitions()
    save_outputs()
