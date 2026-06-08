import argparse
import bisect
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

for import_path in (PROJECT_ROOT, MODELS_DIR):
    import_path_text = str(import_path)
    if import_path_text not in sys.path:
        sys.path.insert(0, import_path_text)

from models.pointnet_plus import Pointnet_plus


SPLIT_CANDIDATES = ("train", "validation", "val", "test")


class RadarScenesNpyDataset(Dataset):
    """
    Loads preprocessed sequence files saved as:

        split/sequence_*.npy -> (num_scenes, num_points, num_features + 1)

    The last column is the per-point label by default.
    """

    def __init__(
        self,
        files,
        label_index=-1,
        feature_dims=None,
        max_scenes=None,
    ):
        self.files = [Path(file_path) for file_path in files]
        self.label_index = label_index
        self.feature_dims = feature_dims
        self._arrays = None

        if not self.files:
            raise ValueError("Dataset received no .npy files")

        self.lengths = []
        self.shapes = []

        for file_path in self.files:
            array = np.load(file_path, mmap_mode="r")
            if array.ndim != 3:
                raise ValueError(
                    f"{file_path} must have shape "
                    f"(num_scenes, num_points, num_columns), got {array.shape}"
                )

            if array.shape[-1] < 2:
                raise ValueError(f"{file_path} needs at least one feature column and one label column")

            self.lengths.append(int(array.shape[0]))
            self.shapes.append(tuple(array.shape))
            del array

        self.num_points = self.shapes[0][1]
        self.num_columns = self.shapes[0][2]
        self.resolved_label_index = self._resolve_label_index(self.num_columns)
        self.num_features = self._resolve_num_features()

        for file_path, shape in zip(self.files, self.shapes):
            if shape[1] != self.num_points:
                raise ValueError(
                    f"All files in a split must use the same point count. "
                    f"{file_path} has {shape[1]}, expected {self.num_points}"
                )

            if shape[2] != self.num_columns:
                raise ValueError(
                    f"All files in a split must use the same column count. "
                    f"{file_path} has {shape[2]}, expected {self.num_columns}"
                )

        self.cumulative_lengths = np.cumsum(self.lengths).tolist()

        if max_scenes is not None:
            self.total_scenes = min(int(max_scenes), self.cumulative_lengths[-1])
        else:
            self.total_scenes = self.cumulative_lengths[-1]

        if self.total_scenes <= 0:
            raise ValueError("Dataset has no scenes")

    def _resolve_label_index(self, num_columns):
        label_index = self.label_index
        if label_index < 0:
            label_index = num_columns + label_index

        if label_index < 0 or label_index >= num_columns:
            raise ValueError(f"Label index {self.label_index} is invalid for {num_columns} columns")

        return label_index

    def _resolve_num_features(self):
        available_features = self.num_columns - 1

        if self.feature_dims is None:
            return available_features

        if self.feature_dims <= 0 or self.feature_dims > available_features:
            raise ValueError(
                f"--feature-dims must be between 1 and {available_features}, got {self.feature_dims}"
            )

        return self.feature_dims

    @property
    def arrays(self):
        if self._arrays is None:
            self._arrays = [np.load(file_path, mmap_mode="r") for file_path in self.files]
        return self._arrays

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_arrays"] = None
        return state

    def __len__(self):
        return self.total_scenes

    def __getitem__(self, index):
        if index < 0:
            index += self.total_scenes

        if index < 0 or index >= self.total_scenes:
            raise IndexError(index)

        file_index = bisect.bisect_right(self.cumulative_lengths, index)
        previous_total = 0 if file_index == 0 else self.cumulative_lengths[file_index - 1]
        scene_index = index - previous_total

        scene = np.asarray(self.arrays[file_index][scene_index])
        labels = scene[:, self.resolved_label_index].astype(np.int64)

        if self.resolved_label_index == self.num_columns - 1:
            features = scene[:, : self.resolved_label_index]
        else:
            features = np.delete(scene, self.resolved_label_index, axis=1)

        features = features[:, : self.num_features].astype(np.float32)

        return torch.from_numpy(features), torch.from_numpy(labels)

    def describe(self):
        return {
            "files": len(self.files),
            "scenes": len(self),
            "points_per_scene": self.num_points,
            "features_per_point": self.num_features,
            "columns_per_point": self.num_columns,
        }


def get_args():
    parser = argparse.ArgumentParser(
        description="Train PointNet++ on preprocessed RadarScenes .npy sequence files."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=(
            "Root produced by Data_preprocessing.py, containing split folders like "
            "train/, validation/, and test/. If omitted, the script checks "
            "RADARSCENES_PROCESSED_ROOT, ./tests, and ./processed."
        ),
    )
    parser.add_argument("--train-split", default="train")
    parser.add_argument(
        "--val-split",
        default="auto",
        help="Validation split name. Use 'auto' to prefer validation/val/test, or 'none' to disable.",
    )
    parser.add_argument("--label-index", type=int, default=-1)
    parser.add_argument(
        "--feature-dims",
        type=int,
        default=None,
        help="Number of feature columns to feed the model. Defaults to all non-label columns.",
    )
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--ignore-label", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument(
        "--fps-mode",
        default="auto",
        choices=("auto", "torch-geometric", "fallback"),
        help="Use torch-geometric FPS, a deterministic fallback, or auto-detect.",
    )
    parser.add_argument("--checkpoint-path", type=Path, default=Path("pointnet_plus_model.pth"))
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=None,
        help="Path for JSON metrics history. Defaults to <checkpoint-name>.metrics.json.",
    )
    parser.add_argument("--max-train-scenes", type=int, default=None)
    parser.add_argument("--max-val-scenes", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Load data/model and exit before training.")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_data_root(data_root):
    if data_root is not None:
        if not data_root.exists():
            raise FileNotFoundError(f"--data-root does not exist: {data_root}")
        return data_root

    candidates = []
    env_root = os.environ.get("RADARSCENES_PROCESSED_ROOT")
    if env_root:
        candidates.append(Path(env_root))

    candidates.extend(
        [
            Path.cwd() / "tests",
            Path.cwd() / "processed",
            PROJECT_ROOT / "tests",
            PROJECT_ROOT / "processed",
        ]
    )

    for candidate in candidates:
        if candidate.exists() and any((candidate / split).is_dir() for split in SPLIT_CANDIDATES):
            return candidate

    checked = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        "Could not auto-detect the preprocessed data root. "
        "Pass --data-root explicitly. Checked: "
        f"{checked}"
    )


def find_split_files(data_root, split_name):
    split_dir = data_root / split_name
    if not split_dir.is_dir():
        return []

    return sorted(path for path in split_dir.rglob("*.npy") if path.is_file())


def resolve_val_split(data_root, requested_split):
    if requested_split.lower() == "none":
        return None

    if requested_split.lower() != "auto":
        return requested_split

    for split_name in ("validation", "val", "test"):
        if find_split_files(data_root, split_name):
            return split_name

    return None


def make_loader(dataset, batch_size, shuffle, num_workers, device):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )


def infer_label_bounds(datasets, ignore_label):
    min_label = None
    max_label = None

    for dataset in datasets:
        for file_path in dataset.files:
            array = np.load(file_path, mmap_mode="r")
            labels = np.asarray(array[..., dataset.resolved_label_index]).reshape(-1)
            labels = labels.astype(np.int64)

            if ignore_label is not None:
                labels = labels[labels != ignore_label]

            if labels.size == 0:
                continue

            file_min = int(labels.min())
            file_max = int(labels.max())
            min_label = file_min if min_label is None else min(min_label, file_min)
            max_label = file_max if max_label is None else max(max_label, file_max)
            del array

    if min_label is None or max_label is None:
        raise ValueError("Could not infer labels from the selected datasets")

    return min_label, max_label


def get_set_abstraction_modules():
    modules = []

    for module_name in ("SetAbstraction", "models.SetAbstraction"):
        module = sys.modules.get(module_name)
        if module is not None and module not in modules:
            modules.append(module)

    if modules:
        return modules

    import SetAbstraction as sa_module

    return [sa_module]


def install_fps_fallback():
    def fallback_fps(x, batch, ratio):
        sampled_indices = []

        for batch_id in batch.unique(sorted=True):
            point_indices = (batch == batch_id).nonzero(as_tuple=False).view(-1)
            num_samples = max(1, int(point_indices.numel() * ratio))

            if num_samples >= point_indices.numel():
                sampled_indices.append(point_indices)
                continue

            positions = torch.linspace(
                0,
                point_indices.numel() - 1,
                steps=num_samples,
                device=point_indices.device,
            ).round().long()
            sampled_indices.append(point_indices[positions])

        return torch.cat(sampled_indices, dim=0)

    for sa_module in get_set_abstraction_modules():
        sa_module.fps = fallback_fps


def configure_fps(mode, device):
    if mode == "fallback":
        install_fps_fallback()
        print("Using fallback FPS sampler.")
        return

    if mode == "torch-geometric":
        print("Using torch-geometric FPS sampler.")
        return

    sa_module = get_set_abstraction_modules()[0]

    try:
        x = torch.randn(16, 3, device=device)
        batch = torch.zeros(16, dtype=torch.long, device=device)
        sa_module.fps(x, batch=batch, ratio=0.5)
        print("Using torch-geometric FPS sampler.")
    except Exception as error:
        install_fps_fallback()
        print(f"Using fallback FPS sampler because torch-geometric FPS failed: {error}")


def format_logits_for_loss(logits, labels):
    if logits.ndim != 3:
        raise ValueError(f"Expected logits with 3 dims, got {logits.shape}")

    num_points = labels.shape[1]

    if logits.shape[1] == num_points:
        return logits.transpose(1, 2)

    if logits.shape[2] == num_points:
        return logits

    raise ValueError(f"Bad logits shape {logits.shape} for labels shape {labels.shape}")


def safe_divide(numerator, denominator):
    if denominator == 0:
        return None

    return float(numerator / denominator)


def mean_defined(values):
    defined_values = [value for value in values if value is not None]
    if not defined_values:
        return 0.0

    return float(sum(defined_values) / len(defined_values))


def compute_metrics(total_loss, total_points, confusion_matrix):
    confusion = confusion_matrix.numpy().astype(np.int64)
    true_positive = np.diag(confusion)
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)

    per_class = {}
    precisions = []
    recalls = []
    f1_scores = []
    ious = []

    for class_id in range(confusion.shape[0]):
        tp = int(true_positive[class_id])
        fp = int(predicted[class_id] - true_positive[class_id])
        fn = int(support[class_id] - true_positive[class_id])

        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)

        if precision is None or recall is None or precision + recall == 0:
            f1_score = None
        else:
            f1_score = float(2 * precision * recall / (precision + recall))

        iou = safe_divide(tp, tp + fp + fn)

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)
        ious.append(iou)

        per_class[str(class_id)] = {
            "support": int(support[class_id]),
            "predicted": int(predicted[class_id]),
            "precision": precision,
            "recall": recall,
            "f1": f1_score,
            "iou": iou,
        }

    total_correct = int(true_positive.sum())

    return {
        "loss": float(total_loss / total_points),
        "accuracy": float(total_correct / total_points),
        "macro_precision": mean_defined(precisions),
        "macro_recall": mean_defined(recalls),
        "macro_f1": mean_defined(f1_scores),
        "mean_iou": mean_defined(ious),
        "total_points": int(total_points),
        "confusion_matrix": confusion.tolist(),
        "per_class": per_class,
    }


def format_metric_summary(name, metrics):
    return (
        f"{name} Loss: {metrics['loss']:.4f} | "
        f"{name} Acc: {metrics['accuracy']:.4f} | "
        f"{name} Macro F1: {metrics['macro_f1']:.4f} | "
        f"{name} mIoU: {metrics['mean_iou']:.4f}"
    )


def default_metrics_path(checkpoint_path):
    return checkpoint_path.with_suffix(".metrics.json")


def write_metrics(metrics_path, payload):
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    with open(metrics_path, "w", encoding="utf-8") as metrics_file:
        json.dump(payload, metrics_file, indent=2)


def run_epoch(model, loader, loss_fn, device, ignore_label, num_classes, optimizer=None):
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss = 0.0
    total_points = 0
    confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.long)

    context = torch.enable_grad() if is_training else torch.no_grad()

    with context:
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x)
            logits = format_logits_for_loss(logits, y)
            loss = loss_fn(logits, y)

            if is_training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            predictions = logits.argmax(dim=1)
            valid_mask = torch.ones_like(y, dtype=torch.bool)

            if ignore_label is not None:
                valid_mask = y != ignore_label

            valid_points = int(valid_mask.sum().item())
            if valid_points == 0:
                continue

            total_loss += loss.item() * valid_points
            total_points += valid_points

            valid_labels = y[valid_mask].detach()
            valid_predictions = predictions[valid_mask].detach()
            confusion_ids = valid_labels * num_classes + valid_predictions
            batch_confusion = torch.bincount(
                confusion_ids,
                minlength=num_classes * num_classes,
            ).reshape(num_classes, num_classes)
            confusion_matrix += batch_confusion.cpu()

    if total_points == 0:
        raise ValueError("No valid labels found during epoch")

    return compute_metrics(total_loss, total_points, confusion_matrix)


def main():
    args = get_args()
    set_seed(args.seed)

    data_root = resolve_data_root(args.data_root)
    val_split = resolve_val_split(data_root, args.val_split)

    train_files = find_split_files(data_root, args.train_split)
    if not train_files:
        raise FileNotFoundError(f"No .npy files found in {data_root / args.train_split}")

    val_files = find_split_files(data_root, val_split) if val_split else []

    train_dataset = RadarScenesNpyDataset(
        train_files,
        label_index=args.label_index,
        feature_dims=args.feature_dims,
        max_scenes=args.max_train_scenes,
    )

    val_dataset = None
    if val_files:
        val_dataset = RadarScenesNpyDataset(
            val_files,
            label_index=args.label_index,
            feature_dims=args.feature_dims,
            max_scenes=args.max_val_scenes,
        )

        if val_dataset.num_features != train_dataset.num_features:
            raise ValueError("Train and validation feature dimensions do not match")

        if val_dataset.num_points != train_dataset.num_points:
            raise ValueError("Train and validation point counts do not match")

    datasets_for_labels = [train_dataset]
    if val_dataset is not None:
        datasets_for_labels.append(val_dataset)

    min_label, max_label = infer_label_bounds(datasets_for_labels, args.ignore_label)
    num_classes = args.num_classes if args.num_classes is not None else max_label + 1

    if min_label < 0 and min_label != args.ignore_label:
        raise ValueError(
            f"Found negative label {min_label}. "
            "Use --ignore-label for ignored labels or fix preprocessing."
        )

    if max_label >= num_classes:
        raise ValueError(f"Max label {max_label} is outside num_classes={num_classes}")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    configure_fps(args.fps_mode, device)

    train_loader = make_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        device=device,
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = make_loader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            device=device,
        )

    model = Pointnet_plus(dim_in=train_dataset.num_features, num_classes=num_classes).to(device)
    loss_kwargs = {}
    if args.ignore_label is not None:
        loss_kwargs["ignore_index"] = args.ignore_label
    loss_fn = nn.CrossEntropyLoss(**loss_kwargs)
    optimizer = Adam(model.parameters(), lr=args.lr)
    metrics_path = args.metrics_path or default_metrics_path(args.checkpoint_path)

    print(f"Data root: {data_root}")
    print(f"Train split: {args.train_split} | {train_dataset.describe()}")
    if val_dataset is not None:
        print(f"Validation split: {val_split} | {val_dataset.describe()}")
    else:
        print("Validation split: disabled or not found")
    print(f"Label range: {min_label}..{max_label} | num_classes: {num_classes}")
    print(f"Device: {device}")
    print(f"Metrics path: {metrics_path}")

    if args.dry_run:
        print("Dry run complete. No training started.")
        return

    metrics_payload = {
        "config": {
            "data_root": str(data_root),
            "train_split": args.train_split,
            "val_split": val_split,
            "checkpoint_path": str(args.checkpoint_path),
            "feature_dims": train_dataset.num_features,
            "label_index": args.label_index,
            "num_classes": num_classes,
            "ignore_label": args.ignore_label,
            "batch_size": args.batch_size,
            "num_epochs": args.num_epochs,
            "lr": args.lr,
            "device": str(device),
            "fps_mode": args.fps_mode,
            "max_train_scenes": args.max_train_scenes,
            "max_val_scenes": args.max_val_scenes,
        },
        "dataset": {
            "train": train_dataset.describe(),
            "validation": val_dataset.describe() if val_dataset is not None else None,
        },
        "epochs": [],
    }

    for epoch in range(args.num_epochs):
        train_metrics = run_epoch(
            model,
            train_loader,
            loss_fn,
            device,
            args.ignore_label,
            num_classes,
            optimizer=optimizer,
        )

        epoch_metrics = {
            "epoch": epoch + 1,
            "train": train_metrics,
            "validation": None,
        }

        message = f"Epoch {epoch + 1}/{args.num_epochs} | {format_metric_summary('Train', train_metrics)}"

        if val_loader is not None:
            val_metrics = run_epoch(
                model,
                val_loader,
                loss_fn,
                device,
                args.ignore_label,
                num_classes,
            )
            epoch_metrics["validation"] = val_metrics
            message += f" | {format_metric_summary('Val', val_metrics)}"

        print(message)
        metrics_payload["epochs"].append(epoch_metrics)
        write_metrics(metrics_path, metrics_payload)

    args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_classes": num_classes,
            "feature_dims": train_dataset.num_features,
            "label_index": args.label_index,
            "data_root": str(data_root),
            "train_split": args.train_split,
            "val_split": val_split,
            "metrics_path": str(metrics_path),
            "last_epoch_metrics": metrics_payload["epochs"][-1] if metrics_payload["epochs"] else None,
        },
        args.checkpoint_path,
    )
    print(f"Saved checkpoint to {args.checkpoint_path}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
