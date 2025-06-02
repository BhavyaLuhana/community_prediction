import torch
import torch.nn.functional as F
import numpy as np
from temporal_data import load_temporal_data
from temporal_model import TemporalGNN
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, f1_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.autograd.set_detect_anomaly(True)

dataset = load_temporal_data()
for i in range(len(dataset)):
    dataset[i] = dataset[i].to(device)

model = TemporalGNN(node_features=dataset[0].num_node_features).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

loss_history = []

def train():
    model.train()
    for epoch in range(50):
        optimizer.zero_grad()
        out = model(dataset)  # shape: [1, num_classes]
        label = dataset[-1].y  # shape: [1]

        if torch.isnan(out).any() or torch.isnan(label).any():
            print("[Error] NaN encountered!")
            break

        loss = F.cross_entropy(out, label)
        if torch.isnan(loss):
            print("[Error] Loss became NaN!")
            break

        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

def evaluate():
    model.eval()
    with torch.no_grad():
        embeddings = []
        labels_true = []

        for data in dataset:
            x = model.gcn(data.x, data.edge_index)
            x = torch.relu(x)
            embeddings.append(x.mean(dim=0).cpu().numpy())  # one embedding per snapshot
            labels_true.append(data.y.item())  # dummy label

        node_embeddings = np.stack(embeddings)  # shape: [10, hidden_dim]
        pred_labels = KMeans(n_clusters=5, n_init=10).fit_predict(node_embeddings)

        ari = adjusted_rand_score(labels_true, pred_labels)
        nmi = normalized_mutual_info_score(labels_true, pred_labels)
        f1 = f1_score(labels_true, pred_labels, average='macro')

        print(f"\nEvaluation Metrics:")
        print(f"Adjusted Rand Index (ARI): {ari:.4f}")
        print(f"Normalized Mutual Information (NMI): {nmi:.4f}")
        print(f"Macro F1-Score: {f1:.4f}")

if __name__ == "__main__":
    train()
    evaluate()
