import json
import os
import sys
import tempfile
from pathlib import Path

import torch

# Ensure imports work when running via pytest
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import dump_hidden as dh  # noqa: E402


def run_dump_hidden(tmpdir: Path) -> Path:
    input_path = tmpdir / "toy.jsonl"
    output_dir = tmpdir / "reps"

    # Minimal labeled chunk record
    record = {
        "example_id": "ex1",
        "question": "What is 2+2?",
        "full_cot": "Let me think. 2+2=4.",
        "final_answer": "4",
        "chunks": [
            {
                "chunk_idx": 0,
                "text": "Let me think. 2+2=4.",
                "intermediate_answer": "4",
                "is_correct": True,
                "is_stable": True,
                "is_backtrack_start": False,
            }
        ],
    }
    with open(input_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    # Use a tiny model to keep runtime/memory low
    model_id = "distilgpt2"
    argv = [
        "dump_hidden",
        "--input",
        str(input_path),
        "--model",
        model_id,
        "--dataset",
        "toy",
        "--batch-size",
        "1",
        "--shard-size",
        "10",
        "--max-length",
        "64",
        "--position-mode",
        "last",
        "--output-dir",
        str(output_dir),
    ]

    sys_argv_old = sys.argv
    sys.argv = argv
    try:
        dh.main()
    finally:
        sys.argv = sys_argv_old

    shards = list(output_dir.glob("part-*.pt"))
    assert shards, "No shards written"
    return shards[0]


def test_dump_hidden_integration_writes_hidden_states():
    with tempfile.TemporaryDirectory() as tmp:
        shard_path = run_dump_hidden(Path(tmp))
        payload = torch.load(shard_path)
        assert "hidden_states" in payload and "meta" in payload
        hs = payload["hidden_states"]
        meta = payload["meta"]
        assert hs.ndim == 2  # (rows, hidden_dim)
        assert hs.shape[0] == len(meta) == 1
        assert meta[0]["example_id"] == "ex1"
        assert meta[0]["chunk_idx"] == 0
