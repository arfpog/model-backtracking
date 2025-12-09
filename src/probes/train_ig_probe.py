#!/usr/bin/env python
"""
Probe for information-gain / productivity signals (RQ3).

Expected input:
- hidden state tensors (e.g., data/reps/<model>/<dataset>/)
- labels for productivity/backtracking value or entropy targets

Expected output:
- results/probes/ig/<run_id>/
"""

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train information-gain/productivity probe.")
    parser.add_argument("--reps", type=Path, required=True, help="Directory of hidden state shards.")
    parser.add_argument("--labels", type=Path, required=True, help="Labeled chunks JSONL.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/probes/ig"),
        help="Directory to save probe checkpoints/metrics.",
    )
    parser.add_argument("--hidden-dim", type=int, default=0, help="Hidden dim (0 = linear probe).")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raise SystemExit(
        "Not implemented: load reps/labels, train probe for information-gain/productivity targets. "
        f"Write metrics to {args.output_dir}."
    )


if __name__ == "__main__":
    main()
