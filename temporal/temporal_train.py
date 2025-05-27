import torch
import torch.nn.functional as F
from temporal_data import load_temporal_data
from temporal_model import TemporalGNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.autograd.set_detect_anomaly(True)

# Load dataset (list of PyG Data objects)
dataset = load_temporal_data()

# Move all Data objects to device upfront
for i in range(len(dataset)):
    dataset[i] = dataset[i].to(device)

model = TemporalGNN(node_features=dataset[0].num_node_features).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train():
    model.train()
    for epoch in range(50):
        optimizer.zero_grad()

        out = model(dataset)  # forward pass

        label = dataset[-1].y  # get last snapshot label, shape: [1]

        # Check for NaNs
        if torch.isnan(out).any():
            print("[Error] NaN in model output!")
            break
        if torch.isnan(label).any():
            print("[Error] NaN in label!")
            break

        loss = F.cross_entropy(out, label)
        if torch.isnan(loss):
            print("[Error] Loss became NaN!")
            break

        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

def evaluate():
    model.eval()
    with torch.no_grad():
        out = model(dataset)
        preds = out.argmax(dim=1)  # predictions, shape [batch_size=1]
        labels = dataset[-1].y

        print(f"Predictions: {preds.cpu().numpy()}")
        print(f"Ground Truth: {labels.cpu().numpy()}")

        correct = (preds == labels).sum().item()
        total = labels.size(0)
        accuracy = correct / total if total > 0 else 0.0
        print(f"Accuracy: {correct}/{total} = {accuracy:.4f}")

if __name__ == "__main__":
    train()
    evaluate()
