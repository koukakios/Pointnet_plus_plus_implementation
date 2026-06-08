import inspect
import math
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
)
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.pointnet_plus import Pointnet_plus

try:
    from torchinfo import summary
    HAS_TORCHINFO = True
except ImportError:
    HAS_TORCHINFO = False


def print_model_summary(model, device):
    print("Model source:", inspect.getfile(type(model)))
    print(
        "Segmentation head source:",
        inspect.getfile(type(model.unitPointnetSegm)),
    )
    print("\nModel:")
    print(model)

    if HAS_TORCHINFO:
        try:
            print("\nModel Summary:")
            summary(model, input_size=(1, 512, 5), device=device)
        except Exception as e:
            print("Model summary failed:", e)
    else:
        print("\nInstall torchinfo for model summary:")
        print("pip install torchinfo")


def update_metrics(preds, batch_y, y_true_list, y_pred_list):
    """
    preds shape: [B, 3, N]
    batch_y shape: [B, N]
    """

    pred_classes = preds.argmax(dim=1)  # [B, N]

    correct = (pred_classes == batch_y).sum().item()
    total = batch_y.numel()

    y_true_list.extend(batch_y.detach().cpu().numpy().reshape(-1))
    y_pred_list.extend(pred_classes.detach().cpu().numpy().reshape(-1))

    return correct, total


def save_training_outputs(y_true, y_pred, loss_history, class_names):
    print("\nClassification Report:")
    print(classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=class_names,
        digits=4,
        zero_division=0
    ))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    print("\nConfusion Matrix:")
    print(cm)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    disp.plot()
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(range(1, len(loss_history) + 1), loss_history, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.savefig("loss_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("\nSaved:")
    print("- confusion_matrix.png")
    print("- loss_curve.png")


def run(begin_seq, end_seq, path_train_sequences):

    train_root = Path(path_train_sequences)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = Pointnet_plus(5, 3)
    model = model.to(device)

    print_model_summary(model, device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 20
    batch_size = 32

    sequence_paths = [
        train_root / f"sequence_{sequence}.npy"
        for sequence in range(begin_seq, end_seq)
        if (train_root / f"sequence_{sequence}.npy").exists()
    ]

    if not sequence_paths:
        print("No training data found. Check your sequence paths.")
        return

    total_batches = sum(
        math.ceil(np.load(path, mmap_mode="r").shape[0] / batch_size)
        for path in sequence_paths
    )

    loss_history = []

    final_y_true = []
    final_y_pred = []

    for epoch in range(num_epochs):
        epoch_start = perf_counter()
        model.train()

        total_loss = 0.0
        total_length = 0

        correct = 0
        total_points = 0

        epoch_y_true = []
        epoch_y_pred = []

        progress = tqdm(
            total=total_batches,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            unit="batch",
            dynamic_ncols=True,
        )

        for path in sequence_paths:
            dataset_npy = np.load(path)

            y = torch.from_numpy(dataset_npy[:, :, -1]).long()
            X = torch.from_numpy(dataset_npy[:, :, :-1]).float()


            dataset = TensorDataset(X, y)
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device, dtype=torch.long)

                preds = model(batch_X)

                # Your model likely outputs [B, N, 3]
                # CrossEntropyLoss needs [B, 3, N]
                preds = preds.permute(0, 2, 1)

                loss = loss_fn(preds, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * batch_X.size(0)
                total_length += batch_X.size(0)

                batch_correct, batch_total = update_metrics(
                    preds,
                    batch_y,
                    epoch_y_true,
                    epoch_y_pred
                )

                correct += batch_correct
                total_points += batch_total

                progress.set_postfix(
                    batch_loss=f"{loss.item():.4f}",
                    avg_loss=f"{total_loss / total_length:.4f}",
                    accuracy=f"{correct / total_points:.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.1e}",
                    sequence=path.stem,
                )
                progress.update(1)

        progress.close()
        avg_loss = total_loss / total_length
        acc = correct / total_points
        precision, recall, f1, _ = precision_recall_fscore_support(
            epoch_y_true,
            epoch_y_pred,
            average="macro",
            zero_division=0,
        )
        epoch_seconds = perf_counter() - epoch_start

        loss_history.append(avg_loss)

        tqdm.write(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"loss={avg_loss:.4f} | accuracy={acc:.4f} | "
            f"precision={precision:.4f} | recall={recall:.4f} | "
            f"f1={f1:.4f} | time={epoch_seconds:.1f}s"
        )

        final_y_true = epoch_y_true
        final_y_pred = epoch_y_pred

    class_names = ["class_0", "class_1", "class_2"]

    save_training_outputs(
        final_y_true,
        final_y_pred,
        loss_history,
        class_names
    )


def main():
    print("Starting training...")

    begin_seq = 0
    end_seq = 158

    path_train_sequences = "/users/kkoukakis/workspace/RadarScenes_Clutter/processed/train"

    run(begin_seq, end_seq, path_train_sequences)


if __name__ == "__main__":
    main()
