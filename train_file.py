import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from pointnet_plus import Pointnet_plus


class RadarScenesDataset(Dataset):
    def __init__(self, root, max_sequences=5):
        self.root = Path(root)
        self.sequence_folders = sorted(self.root.glob("sequence_*"))[:max_sequences]

        self.points = []
        self.labels = []

        print("Loading dataset from:", self.root)

        for folder in self.sequence_folders:
            print("Loading:", folder)

            points = np.load(folder / "points.npy")   # (S, 512, 4)
            labels = np.load(folder / "labels.npy")   # (S, 512, 1)

            self.points.append(points)
            self.labels.append(labels)

        self.points = np.concatenate(self.points, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)

        print("Loaded points:", self.points.shape)
        print("Loaded labels:", self.labels.shape)

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        x = self.points[idx]                    # (512, 4)
        y_points = self.labels[idx].squeeze()   # (512,)

        y_points = y_points.astype(np.int64)

        # Scene label = majority point label
        y = np.bincount(y_points).argmax()

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        return x, y


def train_small():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = RadarScenesDataset(
        "/space/kkoukakis/RadarScenes_Clutter/train_datasets",
        max_sequences=5
    )

    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0
    )

    model = Pointnet_plus(dim_in=4, num_classes=4).to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 3

    train_losses = []
    epoch_losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            logits = model(x)
            loss = loss_fn(logits, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            train_losses.append(loss.item())

            if batch_idx % 50 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"Batch {batch_idx} | "
                    f"Loss: {loss.item():.4f}"
                )

            # keep this small for testing
            if batch_idx == 300:
                break

        avg_loss = total_loss / (batch_idx + 1)
        epoch_losses.append(avg_loss)

        print(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")

    # -------------------------
    # Save loss plots
    # -------------------------
    plt.figure()
    plt.plot(train_losses)
    plt.xlabel("Batch step")
    plt.ylabel("Loss")
    plt.title("Training loss per batch")
    plt.savefig("batch_loss.png")
    plt.close()

    plt.figure()
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Average loss")
    plt.title("Average training loss per epoch")
    plt.savefig("epoch_loss.png")
    plt.close()

    pd.DataFrame({
        "batch_step": range(len(train_losses)),
        "loss": train_losses
    }).to_csv("batch_loss.csv", index=False)

    pd.DataFrame({
        "epoch": range(1, len(epoch_losses) + 1),
        "avg_loss": epoch_losses
    }).to_csv("epoch_loss.csv", index=False)

    print("Saved plots: batch_loss.png and epoch_loss.png")
    print("Saved CSVs: batch_loss.csv and epoch_loss.csv")

    # -------------------------
    # Evaluation on same loader
    # -------------------------
    model.eval()

    all_preds = []
    all_labels = []

    eval_loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0
    )

    with torch.no_grad():
        for x, y in eval_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    print("\nAccuracy:", acc)

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))

    # Save confusion matrix as CSV
    pd.DataFrame(cm).to_csv("confusion_matrix.csv", index=False)

    # Plot confusion matrix
    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.colorbar()

    num_classes = cm.shape[0]
    plt.xticks(range(num_classes))
    plt.yticks(range(num_classes))

    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()

    print("Saved confusion matrix to confusion_matrix.png and confusion_matrix.csv")

    torch.save(model.state_dict(), "pointnet_plus_small_test.pth")
    print("Saved model to pointnet_plus_small_test.pth")


if __name__ == "__main__":
    train_small()