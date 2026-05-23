import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from pointnet_plus import Pointnet_plus


class SequenceDataset(Dataset):
    def __init__(self, sequence_dir, data_filename="points.npy", label_filename="labels.npy"):
        sequence_dir = Path(sequence_dir)

        data = np.load(sequence_dir / data_filename)
        labels = np.load(sequence_dir / label_filename)

        self.data = torch.from_numpy(data).float()
        self.labels = torch.from_numpy(labels.squeeze(-1)).long()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_root", default="/space/kkoukakis/RadarScenes_Clutter/train_datasets")
    parser.add_argument("--val_root", default="/space/kkoukakis/RadarScenes_Clutter/test_datasets")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--data_filename", default="points.npy")
    parser.add_argument("--label_filename", default="labels.npy")
    return parser.parse_args()


def find_sequence_dirs(root):
    return sorted([path for path in Path(root).iterdir() if path.is_dir()])


def make_loaders(sequence_dirs, args, shuffle):
    loaders = []

    for sequence_dir in sequence_dirs:
        dataset = SequenceDataset(
            sequence_dir,
            data_filename=args.data_filename,
            label_filename=args.label_filename,
        )
        loaders.append(
            DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=shuffle,
                num_workers=args.num_workers,
            )
        )

    return loaders


def format_logits_for_loss(logits, labels):
    num_points = labels.shape[1]

    if logits.shape[1] == num_points:
        return logits.transpose(1, 2)

    if logits.shape[2] == num_points:
        return logits

    raise ValueError(f"Bad logits shape {logits.shape} for labels shape {labels.shape}")


def run_loaders(model, loaders, loss_fn, device, optimizer=None):
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_points = 0

    context = torch.enable_grad() if is_training else torch.no_grad()

    with context:
        for loader in loaders:
            for x, y in loader:
                x = x.to(device)
                y = y.to(device)

                logits = model(x)
                logits = format_logits_for_loss(logits, y)
                loss = loss_fn(logits, y)

                if is_training:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                predictions = logits.argmax(dim=1)
                num_points = y.numel()

                total_loss += loss.item() * num_points
                total_correct += (predictions == y).sum().item()
                total_points += num_points

    return total_loss / total_points, total_correct / total_points


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_sequences = find_sequence_dirs(args.train_root)
    print(f"Found {len(train_sequences)} train sequences.")

    train_loaders = make_loaders(train_sequences, args, shuffle=True)

    val_loaders = None
    if args.val_root:
        val_sequences = find_sequence_dirs(args.val_root)
        print(f"Found {len(val_sequences)} validation sequences.")
        val_loaders = make_loaders(val_sequences, args, shuffle=False)

    model = Pointnet_plus().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.num_epochs):
        epoch_train_loaders = train_loaders[:]
        random.shuffle(epoch_train_loaders)

        train_loss, train_acc = run_loaders(
            model,
            epoch_train_loaders,
            loss_fn,
            device,
            optimizer=optimizer,
        )

        message = (
            f"Epoch {epoch + 1}/{args.num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f}"
        )

        if val_loaders is not None:
            val_loss, val_acc = run_loaders(model, val_loaders, loss_fn, device)
            message += f" | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"

        print(message)

    torch.save(model.state_dict(), "pointnet_plus_model.pth")
    print("Saved model to pointnet_plus_model.pth")


if __name__ == "__main__":
    main()
