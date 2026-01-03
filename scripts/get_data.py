#!/usr/bin/env python
import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

import datasets
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.utils import convert_answer_index_to_letter, format_question_with_choices


def extract_gsm8k_answer(answer_text: str) -> str:
    """Extract the final numeric answer from GSM8K format (after ####)."""
    # GSM8K answers are formatted as: "... #### <number>"
    match = re.search(r"####\s*(-?\d+(?:[,\d]*)?(?:\.\d+)?)", answer_text)
    if match:
        # Remove commas from numbers like "1,000"
        return match.group(1).replace(",", "")
    # Fallback: try to get the last number
    numbers = re.findall(r"-?\d+(?:,\d+)*(?:\.\d+)?", answer_text)
    if numbers:
        return numbers[-1].replace(",", "")
    return answer_text.strip()


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

    # Check if we need to process MMLU-style data with choices
    choices_field = entry.get("choices_field")
    text_field = entry.get("text_field", "question")
    answer_field = entry.get("answer_field", "answer")
    answer_type = entry.get("answer_type")

    if choices_field and answer_type == "letter":
        # Process MMLU-style: format question with choices, convert answer to letter
        with open(output_path, "w", encoding="utf-8") as f:
            for row in ds:
                processed = dict(row)
                # Format question with choices
                if choices_field in row and text_field in row:
                    processed[text_field] = format_question_with_choices(
                        row[text_field], row[choices_field]
                    )
                # Convert answer index to letter
                if answer_field in row:
                    processed[answer_field] = convert_answer_index_to_letter(row[answer_field])
                # Add answer_type for downstream processing
                processed["answer_type"] = answer_type
                f.write(json.dumps(processed, ensure_ascii=False) + "\n")
        print(f"Saved to {output_path} ({len(ds)} rows, processed for multiple-choice)")
    elif "gsm8k" in name.lower():
        # Process GSM8K: extract just the final number from "#### <number>" format
        with open(output_path, "w", encoding="utf-8") as f:
            for row in ds:
                processed = dict(row)
                if answer_field in row:
                    processed[answer_field] = extract_gsm8k_answer(row[answer_field])
                processed["answer_type"] = "numeric"
                f.write(json.dumps(processed, ensure_ascii=False) + "\n")
        print(f"Saved to {output_path} ({len(ds)} rows, processed for GSM8K)")
    else:
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
