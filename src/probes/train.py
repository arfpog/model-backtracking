#!/usr/bin/env python
"""
Probe training module for correctness prediction.

Replicates the methodology from Zhang et al. (2025) "Verifier Probing":
- Linear and MLP probes on hidden states
- Class-weighted BCE loss for imbalanced labels
- Early stopping on validation loss
- Evaluation: accuracy, precision, recall, F1, ROC-AUC
"""

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .base_probe import HS_DICT, load_probe


@dataclass
class TrainConfig:
    """Training configuration matching paper defaults."""

    epochs: int = 200
    batch_size: int = 64
    lr: float = 1e-5
    weight_decay: float = 0.001
    hidden_size: int = 0  # 0 = linear probe
    alpha_imbalance: float = 2.0  # class weight multiplier
    patience: int = 10  # early stopping patience
    threshold: float = 0.5  # classification threshold
    seed: int = 42
    device: str = "auto"


@dataclass
class TrainResult:
    """Results from training run."""

    best_val_loss: float
    best_epoch: int
    final_metrics: Dict[str, float]
    config: TrainConfig
    history: List[Dict[str, float]] = field(default_factory=list)


class ProbeDataset(Dataset):
    """Dataset for probe training from .pt shard files."""

    def __init__(
        self,
        shard_paths: List[Path],
        label_key: str = "is_correct",
        filter_none: bool = True,
    ):
        self.hidden_states: List[torch.Tensor] = []
        self.labels: List[int] = []
        self.metadata: List[Dict] = []

        for path in shard_paths:
            data = torch.load(path, weights_only=False)
            hs = data["hidden_states"]
            meta = data["meta"]

            for i, m in enumerate(meta):
                label = m.get(label_key)
                if filter_none and label is None:
                    continue
                self.hidden_states.append(hs[i])
                self.labels.append(int(label) if label is not None else 0)
                self.metadata.append(m)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.hidden_states[idx], self.labels[idx]

    def get_class_weights(self) -> torch.Tensor:
        """Compute pos_weight for BCE loss based on class distribution."""
        labels = torch.tensor(self.labels, dtype=torch.float)
        num_pos = labels.sum().item()
        num_neg = len(labels) - num_pos
        if num_pos == 0:
            return torch.tensor(1.0)
        return torch.tensor(num_neg / num_pos)


def select_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def compute_metrics(
    y_true: List[int], y_pred: List[int], y_prob: List[float]
) -> Dict[str, float]:
    """Compute all evaluation metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    # ROC-AUC requires both classes present
    if len(set(y_true)) > 1:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    else:
        metrics["roc_auc"] = float("nan")
    return metrics


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch, return average loss."""
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x = x.to(device)
        y = y.float().to(device).unsqueeze(1)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> Tuple[float, Dict[str, float]]:
    """Evaluate model, return (loss, metrics_dict)."""
    model.eval()
    total_loss = 0.0
    all_y, all_pred, all_prob = [], [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y_t = y.float().to(device).unsqueeze(1)

            logits = model(x)
            loss = criterion(logits, y_t)
            total_loss += loss.item() * x.size(0)

            probs = torch.sigmoid(logits).squeeze(1).cpu().tolist()
            preds = [1 if p >= threshold else 0 for p in probs]

            all_y.extend(y.tolist())
            all_pred.extend(preds)
            all_prob.extend(probs)

    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics(all_y, all_pred, all_prob)
    return avg_loss, metrics


def train_probe(
    train_dataset: ProbeDataset,
    val_dataset: ProbeDataset,
    input_size: int,
    config: TrainConfig,
) -> Tuple[nn.Module, TrainResult]:
    """
    Train a probe with early stopping.

    Returns:
        (trained_model, train_result)
    """
    torch.manual_seed(config.seed)
    device = select_device(config.device)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # Initialize model
    model = load_probe(input_size, config.hidden_size)
    model.to(device)

    # Class-weighted loss
    pos_weight = train_dataset.get_class_weights() * config.alpha_imbalance
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    # Training loop with early stopping
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    history = []
    patience_counter = 0

    for epoch in range(config.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = evaluate(
            model, val_loader, criterion, device, config.threshold
        )

        epoch_data = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(epoch_data)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    # Final evaluation
    _, final_metrics = evaluate(model, val_loader, criterion, device, config.threshold)

    result = TrainResult(
        best_val_loss=best_val_loss,
        best_epoch=best_epoch,
        final_metrics=final_metrics,
        config=config,
        history=history,
    )

    return model, result


def load_shards(data_dir: Path, split: str = "train") -> List[Path]:
    """Load shard paths from a directory."""
    pattern = f"part-*.pt"
    shards = sorted(data_dir.glob(pattern))
    if not shards:
        raise FileNotFoundError(f"No shards found in {data_dir}")
    return shards


def train_val_split(
    shards: List[Path], val_ratio: float = 0.2, seed: int = 42
) -> Tuple[List[Path], List[Path]]:
    """Split shards into train/val."""
    import random

    random.seed(seed)
    shards = list(shards)
    random.shuffle(shards)
    split_idx = int(len(shards) * (1 - val_ratio))
    return shards[:split_idx], shards[split_idx:]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train correctness probe.")
    parser.add_argument(
        "--data-dir", type=Path, required=True, help="Directory with .pt shards."
    )
    parser.add_argument(
        "--label-key",
        default="is_correct",
        choices=["is_correct", "is_stable", "is_backtrack_start"],
        help="Which label to predict.",
    )
    parser.add_argument("--model-name", help="Model name for input dim lookup.")
    parser.add_argument("--input-size", type=int, help="Hidden state dimension (auto-detected if not set).")
    parser.add_argument("--hidden-size", type=int, default=0, help="MLP hidden size (0=linear).")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--alpha-imbalance", type=float, default=2.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, help="Save trained model weights.")
    parser.add_argument("--results-json", type=Path, help="Save results to JSON.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    # Load shards and split
    shards = load_shards(args.data_dir)
    train_shards, val_shards = train_val_split(shards, args.val_ratio, args.seed)

    print(f"Train shards: {len(train_shards)}, Val shards: {len(val_shards)}")

    # Create datasets
    train_dataset = ProbeDataset(train_shards, label_key=args.label_key)
    val_dataset = ProbeDataset(val_shards, label_key=args.label_key)

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Determine input size
    if args.input_size:
        input_size = args.input_size
    elif args.model_name and args.model_name in HS_DICT:
        input_size = HS_DICT[args.model_name]
    else:
        # Auto-detect from first sample
        sample_hs, _ = train_dataset[0]
        input_size = sample_hs.shape[-1]
    print(f"Input size: {input_size}")

    # Build config
    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_size=args.hidden_size,
        alpha_imbalance=args.alpha_imbalance,
        patience=args.patience,
        seed=args.seed,
        device=args.device,
    )

    # Train
    model, result = train_probe(train_dataset, val_dataset, input_size, config)

    # Print results
    print(f"\nBest epoch: {result.best_epoch}, Best val loss: {result.best_val_loss:.4f}")
    print("Final metrics:")
    for k, v in result.final_metrics.items():
        print(f"  {k}: {v:.4f}")

    # Save model
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.output)
        print(f"Saved model to {args.output}")

    # Save results
    if args.results_json:
        args.results_json.parent.mkdir(parents=True, exist_ok=True)
        results_data = {
            "best_val_loss": result.best_val_loss,
            "best_epoch": result.best_epoch,
            "final_metrics": result.final_metrics,
            "config": {
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "lr": config.lr,
                "weight_decay": config.weight_decay,
                "hidden_size": config.hidden_size,
                "alpha_imbalance": config.alpha_imbalance,
                "patience": config.patience,
                "seed": config.seed,
            },
            "label_key": args.label_key,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
        }
        with open(args.results_json, "w") as f:
            json.dump(results_data, f, indent=2)
        print(f"Saved results to {args.results_json}")


if __name__ == "__main__":
    main()
