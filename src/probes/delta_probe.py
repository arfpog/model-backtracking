#!/usr/bin/env python
"""
Delta correctness prediction: p(correctness_t+1) - p(correctness_t)

This module extends the basic correctness probe to predict whether the
*upcoming* answer will be correct, and how correctness changes over time.

Key idea from the paper: models' hidden states encode correctness of future
answers, enabling early prediction before the intermediate answer is formulated.

We implement three prediction tasks:
1. Predict p(correct_t+1) from hidden state at chunk t (look-ahead)
2. Predict delta = p(correct_t+1) - p(correct_t) (correctness change)
3. Predict if upcoming chunk will change the answer (answer stability)
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .base_probe import load_probe
from .train import TrainConfig, compute_metrics, evaluate, select_device, train_epoch


@dataclass
class ChunkPair:
    """A pair of adjacent chunks for delta prediction."""

    example_id: str
    chunk_idx: int
    hidden_state: torch.Tensor  # hidden state at chunk t
    is_correct_t: Optional[bool]  # correctness at current chunk
    is_correct_t1: Optional[bool]  # correctness at next chunk
    answer_t: Optional[str]  # answer at current chunk
    answer_t1: Optional[str]  # answer at next chunk


class DeltaProbeDataset(Dataset):
    """
    Dataset for predicting upcoming correctness or correctness changes.

    Modes:
    - "lookahead": predict is_correct_{t+1} from hidden state at t
    - "delta": predict change in correctness (1 if improves, 0 if same/worsens)
    - "stability": predict if answer will change in next chunk
    """

    def __init__(
        self,
        shard_paths: List[Path],
        mode: str = "lookahead",
        filter_incomplete: bool = True,
    ):
        self.mode = mode
        self.pairs: List[ChunkPair] = []

        # Group chunks by example_id
        examples: Dict[str, List[Tuple[int, torch.Tensor, Dict]]] = {}

        for path in shard_paths:
            data = torch.load(path, weights_only=False)
            hs = data["hidden_states"]
            meta = data["meta"]

            for i, m in enumerate(meta):
                ex_id = m.get("example_id", "")
                chunk_idx = m.get("chunk_idx", 0)

                if ex_id not in examples:
                    examples[ex_id] = []

                examples[ex_id].append(
                    (
                        chunk_idx,
                        hs[i],
                        {
                            "is_correct": m.get("is_correct"),
                            "is_stable": m.get("is_stable"),
                            "intermediate_answer": m.get("intermediate_answer"),
                        },
                    )
                )

        # Create pairs from adjacent chunks
        for ex_id, chunks in examples.items():
            # Sort by chunk index
            chunks.sort(key=lambda x: x[0])

            for i in range(len(chunks) - 1):
                idx_t, hs_t, meta_t = chunks[i]
                idx_t1, hs_t1, meta_t1 = chunks[i + 1]

                # Verify consecutive
                if idx_t1 != idx_t + 1:
                    continue

                pair = ChunkPair(
                    example_id=ex_id,
                    chunk_idx=idx_t,
                    hidden_state=hs_t,
                    is_correct_t=meta_t.get("is_correct"),
                    is_correct_t1=meta_t1.get("is_correct"),
                    answer_t=meta_t.get("intermediate_answer"),
                    answer_t1=meta_t1.get("intermediate_answer"),
                )

                # Filter based on mode requirements
                if filter_incomplete:
                    if mode == "lookahead" and pair.is_correct_t1 is None:
                        continue
                    if mode == "delta" and (
                        pair.is_correct_t is None or pair.is_correct_t1 is None
                    ):
                        continue
                    if mode == "stability" and (
                        pair.answer_t is None and pair.answer_t1 is None
                    ):
                        continue

                self.pairs.append(pair)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        pair = self.pairs[idx]

        if self.mode == "lookahead":
            # Predict next chunk's correctness
            label = int(pair.is_correct_t1) if pair.is_correct_t1 is not None else 0

        elif self.mode == "delta":
            # Predict improvement: 1 if went from wrong to right, 0 otherwise
            was_wrong = pair.is_correct_t is False
            now_right = pair.is_correct_t1 is True
            label = 1 if (was_wrong and now_right) else 0

        elif self.mode == "stability":
            # Predict if answer will change
            if pair.answer_t is None or pair.answer_t1 is None:
                label = 0  # Can't determine
            else:
                label = 0 if pair.answer_t == pair.answer_t1 else 1

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return pair.hidden_state, label

    def get_class_weights(self) -> torch.Tensor:
        """Compute pos_weight for BCE loss."""
        labels = [self[i][1] for i in range(len(self))]
        labels = torch.tensor(labels, dtype=torch.float)
        num_pos = labels.sum().item()
        num_neg = len(labels) - num_pos
        if num_pos == 0:
            return torch.tensor(1.0)
        return torch.tensor(num_neg / num_pos)

    def get_stats(self) -> Dict[str, int]:
        """Get dataset statistics."""
        labels = [self[i][1] for i in range(len(self))]
        return {
            "total": len(labels),
            "positive": sum(labels),
            "negative": len(labels) - sum(labels),
        }


def train_delta_probe(
    train_dataset: DeltaProbeDataset,
    val_dataset: DeltaProbeDataset,
    input_size: int,
    config: TrainConfig,
) -> Tuple[nn.Module, Dict]:
    """Train a delta/lookahead probe."""
    torch.manual_seed(config.seed)
    device = select_device(config.device)

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    model = load_probe(input_size, config.hidden_size)
    model.to(device)

    pos_weight = train_dataset.get_class_weights() * config.alpha_imbalance
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    patience_counter = 0
    history = []

    for epoch in range(config.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = evaluate(
            model, val_loader, criterion, device, config.threshold
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                **val_metrics,
            }
        )

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

    if best_state:
        model.load_state_dict(best_state)
        model.to(device)

    _, final_metrics = evaluate(model, val_loader, criterion, device, config.threshold)

    result = {
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "final_metrics": final_metrics,
        "mode": train_dataset.mode,
        "train_stats": train_dataset.get_stats(),
        "val_stats": val_dataset.get_stats(),
    }

    return model, result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train delta/lookahead probe.")
    parser.add_argument(
        "--data-dir", type=Path, required=True, help="Directory with .pt shards."
    )
    parser.add_argument(
        "--mode",
        default="lookahead",
        choices=["lookahead", "delta", "stability"],
        help="Prediction mode.",
    )
    parser.add_argument("--hidden-size", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--alpha-imbalance", type=float, default=2.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, help="Save model weights.")
    parser.add_argument("--results-json", type=Path, help="Save results.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    # Load shards
    shards = sorted(args.data_dir.glob("part-*.pt"))
    if not shards:
        raise FileNotFoundError(f"No shards in {args.data_dir}")

    # Split shards
    import random

    random.seed(args.seed)
    shards = list(shards)
    random.shuffle(shards)
    split_idx = int(len(shards) * (1 - args.val_ratio))
    train_shards, val_shards = shards[:split_idx], shards[split_idx:]

    print(f"Mode: {args.mode}")
    print(f"Train shards: {len(train_shards)}, Val shards: {len(val_shards)}")

    train_dataset = DeltaProbeDataset(train_shards, mode=args.mode)
    val_dataset = DeltaProbeDataset(val_shards, mode=args.mode)

    print(f"Train pairs: {len(train_dataset)}, Val pairs: {len(val_dataset)}")
    print(f"Train stats: {train_dataset.get_stats()}")
    print(f"Val stats: {val_dataset.get_stats()}")

    if len(train_dataset) == 0:
        print("No valid pairs found. Check your data.")
        return

    # Auto-detect input size
    sample_hs, _ = train_dataset[0]
    input_size = sample_hs.shape[-1]
    print(f"Input size: {input_size}")

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

    model, result = train_delta_probe(train_dataset, val_dataset, input_size, config)

    print(f"\nBest epoch: {result['best_epoch']}")
    print(f"Best val loss: {result['best_val_loss']:.4f}")
    print("Final metrics:")
    for k, v in result["final_metrics"].items():
        print(f"  {k}: {v:.4f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.output)
        print(f"Saved model to {args.output}")

    if args.results_json:
        args.results_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.results_json, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"Saved results to {args.results_json}")


if __name__ == "__main__":
    main()
