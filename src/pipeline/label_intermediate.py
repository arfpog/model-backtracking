#!/usr/bin/env python
"""
Stage 03: extract and label intermediate answers within chunks.

Expected output:
- data/labeled/labeled_intermediate_<dataset>.jsonl
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

from .utils import (
    answers_match,
    detect_backtrack,
    extract_intermediate_answer,
    normalize_answer,
    read_jsonl,
    write_jsonl,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Label intermediate answers.")
    parser.add_argument("--input", type=Path, required=True, help="Chunked CoT JSONL.")
    parser.add_argument("--dataset", help="Dataset name (for naming outputs).")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSONL path. Defaults to data/labeled/labeled_intermediate_<dataset>.jsonl",
    )
    return parser


def compute_outcome(prev_correct: Optional[bool], curr_correct: Optional[bool]) -> Optional[str]:
    if prev_correct is None or curr_correct is None:
        return None
    if prev_correct is False and curr_correct is True:
        return "successful"
    if prev_correct is False and curr_correct is False:
        return "unsuccessful"
    if prev_correct is True and curr_correct is False:
        return "harmful"
    if prev_correct is True and curr_correct is True:
        return "unnecessary"
    return None


def label_record(record: Dict[str, Any]) -> Dict[str, Any]:
    gt_answer = record.get("answer")
    final_answer = record.get("final_answer") or extract_intermediate_answer(record.get("full_cot", ""))

    chunks = record.get("chunks", [])
    prev_answer = None
    prev_correct = None
    prev_idx = None
    backtracking_events: List[Dict[str, Any]] = []

    for chunk in chunks:
        text = chunk.get("text", "")
        intermediate = chunk.get("intermediate_answer") or extract_intermediate_answer(text)
        chunk["intermediate_answer"] = intermediate

        is_correct = answers_match(intermediate, gt_answer) if gt_answer else None
        chunk["is_correct"] = is_correct

        is_stable = answers_match(intermediate, final_answer) if intermediate and final_answer else None
        chunk["is_stable"] = is_stable

        backtrack_flag = detect_backtrack(text)
        if prev_answer is not None and intermediate is not None and normalize_answer(intermediate) != normalize_answer(prev_answer):
            backtrack_flag = True
        chunk["is_backtrack_start"] = backtrack_flag

        if backtrack_flag and prev_idx is not None:
            outcome = compute_outcome(prev_correct, is_correct)
            backtracking_events.append(
                {
                    "chunk_before": prev_idx,
                    "chunk_after": chunk.get("chunk_idx", len(backtracking_events)),
                    "outcome": outcome,
                    "is_productive": outcome == "successful",
                }
            )

        if intermediate is not None:
            prev_answer = intermediate
            prev_correct = is_correct
            prev_idx = chunk.get("chunk_idx", prev_idx)

    record["final_answer"] = final_answer
    record["chunks"] = chunks
    record["backtracking_events"] = backtracking_events
    return record


def main() -> None:
    args = build_parser().parse_args()
    dataset = args.dataset or args.input.stem
    default_out = Path("data/labeled") / f"labeled_intermediate_{dataset}.jsonl"
    output_path = args.output or default_out

    labeled = [label_record(record) for record in read_jsonl(args.input)]
    write_jsonl(output_path, labeled)
    print(f"Wrote {len(labeled)} labeled examples to {output_path}")


if __name__ == "__main__":
    main()
