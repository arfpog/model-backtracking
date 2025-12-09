#!/usr/bin/env python
import argparse
import shutil
import sys
from pathlib import Path

import datasets
import yaml


def download_dataset(entry: dict, raw_dir: Path) -> None:
    name = entry.get("name") or "unnamed_dataset"
    hf_id = entry.get("hf_id")
    subset = entry.get("subset")
    split = entry.get("split", "train")
    output_name = entry.get("output_name") or f"{name}.jsonl"
    local_path = entry.get("local_path")
    skip_download = entry.get("skip_download", False)

    output_path = raw_dir / output_name

    if skip_download:
        print(
            f"Skipping {name} (skip_download=True). If applicable, place file at {output_path}."
        )
        return

    if local_path:
        local_src = Path(local_path)
        if local_src.exists():
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(local_src, output_path)
            print(f"Copied {local_src} -> {output_path}")
            return
        else:
            print(
                f"Local path {local_src} not found; falling back to HF download if hf_id is set."
            )

    if not hf_id:
        print(f"Skipping {name}: missing hf_id and no valid local_path.")
        return

    print(f"Downloading {name} from {hf_id} (split={split}, subset={subset or 'default'})...")
    ds = datasets.load_dataset(hf_id, subset, split=split)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    ds.to_json(str(output_path))
    print(f"Saved to {output_path} ({len(ds)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download datasets to JSONL")
    parser.add_argument("--config", default="configs/datasets.yaml", help="Path to datasets YAML")
    parser.add_argument("--out", default="data/raw", help="Output directory for JSONL files")
    args = parser.parse_args()

    config_path = Path(args.config)
    raw_dir = Path(args.out)
    raw_dir.mkdir(parents=True, exist_ok=True)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    datasets_cfg = cfg.get("datasets", [])
    if not datasets_cfg:
        sys.exit("No datasets listed in datasets.yaml")

    for entry in datasets_cfg:
        download_dataset(entry, raw_dir)

    print("Data download complete.")


if __name__ == "__main__":
    main()
