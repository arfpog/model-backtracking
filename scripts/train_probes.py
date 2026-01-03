#!/usr/bin/env python
"""
Grid search training script for correctness probes.

Replicates hyperparameter search from Zhang et al. (2025):
- Learning rates: [1e-5, 1e-4, 1e-3]
- Hidden sizes: [0, 16, 32, 1024] (0 = linear)
- Weight decay: [0.001, 0.01]
- Alpha imbalance: [1.0, 2.0, 4.0]
"""

import argparse
import itertools
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.probes.train import (
    ProbeDataset,
    TrainConfig,
    load_shards,
    train_probe,
    train_val_split,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Grid search for probe training.")
    parser.add_argument(
        "--data-dir", type=Path, required=True, help="Directory with .pt shards."
    )
    parser.add_argument(
        "--label-key",
        default="is_correct",
        choices=["is_correct", "is_stable", "is_backtrack_start"],
        help="Which label to predict.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/grid_search"))
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--quick", action="store_true", help="Use reduced hyperparameter grid."
    )
    return parser


def get_hyperparam_grid(quick: bool = False):
    """Return hyperparameter combinations to search."""
    if quick:
        return {
            "lr": [1e-4],
            "hidden_size": [0, 32],
            "weight_decay": [0.001],
            "alpha_imbalance": [2.0],
        }
    return {
        "lr": [1e-5, 1e-4, 1e-3],
        "hidden_size": [0, 16, 32, 1024],
        "weight_decay": [0.001, 0.01],
        "alpha_imbalance": [1.0, 2.0, 4.0],
    }


def main() -> None:
    args = build_parser().parse_args()

    # Load and split data
    shards = load_shards(args.data_dir)
    train_shards, val_shards = train_val_split(shards, args.val_ratio, args.seed)

    train_dataset = ProbeDataset(train_shards, label_key=args.label_key)
    val_dataset = ProbeDataset(val_shards, label_key=args.label_key)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Auto-detect input size
    sample_hs, _ = train_dataset[0]
    input_size = sample_hs.shape[-1]
    print(f"Input size: {input_size}")

    # Grid search
    grid = get_hyperparam_grid(args.quick)
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))

    print(f"\nRunning {len(combos)} hyperparameter combinations...")

    results = []
    best_result = None
    best_val_loss = float("inf")

    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        print(f"\n[{i+1}/{len(combos)}] {params}")

        config = TrainConfig(
            lr=params["lr"],
            hidden_size=params["hidden_size"],
            weight_decay=params["weight_decay"],
            alpha_imbalance=params["alpha_imbalance"],
            seed=args.seed,
            device=args.device,
        )

        model, result = train_probe(train_dataset, val_dataset, input_size, config)

        entry = {
            "params": params,
            "best_val_loss": result.best_val_loss,
            "best_epoch": result.best_epoch,
            "metrics": result.final_metrics,
        }
        results.append(entry)

        print(f"  val_loss={result.best_val_loss:.4f}, acc={result.final_metrics['accuracy']:.4f}, "
              f"roc_auc={result.final_metrics.get('roc_auc', float('nan')):.4f}")

        if result.best_val_loss < best_val_loss:
            best_val_loss = result.best_val_loss
            best_result = entry
            best_model = model

    # Sort by val loss
    results.sort(key=lambda x: x["best_val_loss"])

    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_path = args.output_dir / f"grid_search_{args.label_key}_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "label_key": args.label_key,
                "train_samples": len(train_dataset),
                "val_samples": len(val_dataset),
                "input_size": input_size,
                "results": results,
                "best": best_result,
            },
            f,
            indent=2,
        )
    print(f"\nSaved results to {results_path}")

    # Save best model
    import torch

    model_path = args.output_dir / f"best_probe_{args.label_key}_{timestamp}.pt"
    torch.save(best_model.state_dict(), model_path)
    print(f"Saved best model to {model_path}")

    print(f"\n=== Best Result ===")
    print(f"Params: {best_result['params']}")
    print(f"Val loss: {best_result['best_val_loss']:.4f}")
    for k, v in best_result["metrics"].items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
