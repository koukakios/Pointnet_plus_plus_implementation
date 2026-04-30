import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from pointnet_plus import Pointnet_plus


class RadarScenesDataset(Dataset):
    def __init__(self, root, max_sequences=2):
        self.root = Path(root)
        self.sequence_folders = sorted(self.root.glob("sequence_*"))[:max_sequences]

        self.points = []
        self.labels = []

        print("Loading dataset from:", self.root)

        for folder in self.sequence_folders:
            print("Loading:", folder)
            points = np.load(folder / "points.npy")
            labels = np.load(folder / "labels.npy")

            self.points.append(points)
            self.labels.append(labels)

        self.points = np.concatenate(self.points, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)

        print("Loaded points:", self.points.shape)
        print("Loaded labels:", self.labels.shape)

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        x = self.points[idx]
        y_points = self.labels[idx].squeeze().astype(np.int64)

        # scene label = majority point label
        y = np.bincount(y_points).argmax()

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        return x, y


def train_test_run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = RadarScenesDataset(
        "/space/kkoukakis/RadarScenes_Clutter/train_datasets",
        max_sequences=2
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0
    )

    model = Pointnet_plus(dim_in=4, num_classes=4).to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()

    for batch_idx, (x, y) in enumerate(loader):
        print("\nBatch:", batch_idx)
        print("x before device:", x.shape)
        print("y before device:", y.shape, y)

        x = x.to(device)
        y = y.to(device)

        print("x on device:", x.shape)

        optimizer.zero_grad()

        logits = model(x)
        print("logits:", logits.shape)

        loss = loss_fn(logits, y)
        print("loss:", loss.item())

        loss.backward()
        optimizer.step()

        print("backward + optimizer step worked")

        if batch_idx == 1:
            break

    print("\nTEST RUN FINISHED SUCCESSFULLY")


if __name__ == "__main__":
    train_test_run()