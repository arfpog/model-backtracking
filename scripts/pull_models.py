#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download


def download_model(entry: dict, base_dir: Path) -> None:
    name = entry.get("name")
    hf_id = entry.get("hf_id")
    revision = entry.get("revision")
    if not name or not hf_id:
        print(f"Skipping entry with missing name/hf_id: {entry}")
        return

    target_dir = base_dir / name
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {hf_id} -> {target_dir}")
    snapshot_download(
        repo_id=hf_id,
        revision=revision,
        local_dir=target_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download models listed in models.yaml")
    parser.add_argument("--config", default="configs/models.yaml", help="Path to models YAML")
    parser.add_argument("--out", default="models", help="Directory to store downloaded models")
    args = parser.parse_args()

    config_path = Path(args.config)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    models_cfg = cfg.get("models", [])
    if not models_cfg:
        sys.exit("No models listed in models.yaml")

    for entry in models_cfg:
        download_model(entry, out_dir)

    print("All models downloaded.")


if __name__ == "__main__":
    main()
