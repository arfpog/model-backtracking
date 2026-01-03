#!/usr/bin/env python
"""
Integration test for probe training pipeline.

Creates synthetic data and verifies training works end-to-end.
"""

import tempfile
from pathlib import Path

import pytest
import torch


def create_synthetic_shards(output_dir: Path, num_examples: int = 20, hidden_dim: int = 64):
    """Create synthetic hidden state shards for testing."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create two shards
    for shard_idx in range(2):
        hidden_states = []
        meta = []

        for i in range(num_examples):
            example_id = f"ex_{shard_idx * num_examples + i}"

            # Create 3-5 chunks per example
            num_chunks = 3 + (i % 3)

            for chunk_idx in range(num_chunks):
                # Synthetic hidden state
                hs = torch.randn(hidden_dim)

                # Make correctness somewhat predictable from hidden state
                # (first dimension > 0 means correct)
                is_correct = hs[0].item() > 0

                # Stability: answer matches final if last chunk
                is_stable = chunk_idx == num_chunks - 1

                # Backtracking: random
                is_backtrack = chunk_idx > 0 and (i + chunk_idx) % 3 == 0

                hidden_states.append(hs)
                meta.append({
                    "example_id": example_id,
                    "chunk_idx": chunk_idx,
                    "is_correct": is_correct,
                    "is_stable": is_stable,
                    "is_backtrack_start": is_backtrack,
                    "intermediate_answer": str(i + chunk_idx),
                })

        tensor = torch.stack(hidden_states)
        path = output_dir / f"part-{shard_idx:04d}.pt"
        torch.save({"hidden_states": tensor, "meta": meta}, path)

    return output_dir


class TestProbeDataset:
    def test_load_shards(self):
        """Test loading synthetic shards."""
        from src.probes.train import ProbeDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = create_synthetic_shards(Path(tmpdir))
            shards = sorted(data_dir.glob("part-*.pt"))

            dataset = ProbeDataset(shards, label_key="is_correct")

            assert len(dataset) > 0
            hs, label = dataset[0]
            assert hs.shape == (64,)
            assert label in [0, 1]

    def test_class_weights(self):
        """Test class weight computation."""
        from src.probes.train import ProbeDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = create_synthetic_shards(Path(tmpdir))
            shards = sorted(data_dir.glob("part-*.pt"))

            dataset = ProbeDataset(shards, label_key="is_correct")
            weights = dataset.get_class_weights()

            assert isinstance(weights, torch.Tensor)
            assert weights.item() > 0


class TestProbeTraining:
    def test_train_linear_probe(self):
        """Test training a linear probe."""
        from src.probes.train import ProbeDataset, TrainConfig, train_probe

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = create_synthetic_shards(Path(tmpdir))
            shards = sorted(data_dir.glob("part-*.pt"))

            train_shards = shards[:1]
            val_shards = shards[1:]

            train_dataset = ProbeDataset(train_shards)
            val_dataset = ProbeDataset(val_shards)

            config = TrainConfig(
                epochs=5,
                batch_size=8,
                lr=1e-3,
                hidden_size=0,  # linear
                patience=3,
                device="cpu",
            )

            model, result = train_probe(train_dataset, val_dataset, 64, config)

            assert result.best_epoch >= 0
            assert "accuracy" in result.final_metrics
            assert "roc_auc" in result.final_metrics

    def test_train_mlp_probe(self):
        """Test training an MLP probe."""
        from src.probes.train import ProbeDataset, TrainConfig, train_probe

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = create_synthetic_shards(Path(tmpdir))
            shards = sorted(data_dir.glob("part-*.pt"))

            train_shards = shards[:1]
            val_shards = shards[1:]

            train_dataset = ProbeDataset(train_shards)
            val_dataset = ProbeDataset(val_shards)

            config = TrainConfig(
                epochs=5,
                batch_size=8,
                lr=1e-3,
                hidden_size=16,  # MLP with 16-dim hidden
                patience=3,
                device="cpu",
            )

            model, result = train_probe(train_dataset, val_dataset, 64, config)

            assert result.best_epoch >= 0


class TestDeltaProbe:
    def test_delta_dataset_lookahead(self):
        """Test creating lookahead dataset."""
        from src.probes.delta_probe import DeltaProbeDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = create_synthetic_shards(Path(tmpdir))
            shards = sorted(data_dir.glob("part-*.pt"))

            dataset = DeltaProbeDataset(shards, mode="lookahead")

            assert len(dataset) > 0
            hs, label = dataset[0]
            assert hs.shape == (64,)
            assert label in [0, 1]

    def test_delta_dataset_stability(self):
        """Test creating stability dataset."""
        from src.probes.delta_probe import DeltaProbeDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = create_synthetic_shards(Path(tmpdir))
            shards = sorted(data_dir.glob("part-*.pt"))

            dataset = DeltaProbeDataset(shards, mode="stability")
            stats = dataset.get_stats()

            assert stats["total"] > 0

    def test_train_delta_probe(self):
        """Test training a delta probe."""
        from src.probes.delta_probe import DeltaProbeDataset, train_delta_probe
        from src.probes.train import TrainConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = create_synthetic_shards(Path(tmpdir))
            shards = sorted(data_dir.glob("part-*.pt"))

            train_shards = shards[:1]
            val_shards = shards[1:]

            train_dataset = DeltaProbeDataset(train_shards, mode="lookahead")
            val_dataset = DeltaProbeDataset(val_shards, mode="lookahead")

            config = TrainConfig(
                epochs=5,
                batch_size=8,
                lr=1e-3,
                hidden_size=0,
                patience=3,
                device="cpu",
            )

            model, result = train_delta_probe(train_dataset, val_dataset, 64, config)

            assert result["best_epoch"] >= 0
            assert "accuracy" in result["final_metrics"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
