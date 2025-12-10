#!/usr/bin/env python

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from .utils import (
        answers_match,
        detect_backtrack,
        extract_intermediate_answer,
        normalize_answer,
        read_jsonl,
        write_jsonl,
    )
except ImportError:
    # Allow running as a script: python src/pipeline/label_intermediate.py ...
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.pipeline.utils import (
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
    parser.add_argument(
        "--labeling-mode",
        choices=["regex", "gemini"],
        default="regex",
        help="Use regex-only labeling or Gemini API.",
    )
    parser.add_argument(
        "--gemini-model",
        default="gemini-2.0-flash",
        help="Gemini model name (only used if --labeling-mode=gemini).",
    )
    parser.add_argument(
        "--gemini-temperature",
        type=float,
        default=0.6,
        help="Gemini generation temperature.",
    )
    parser.add_argument(
        "--gemini-max-tokens",
        type=int,
        default=4000,
        help="Max output tokens for Gemini.",
    )
    parser.add_argument(
        "--gemini-api-key",
        default=None,
        help="Gemini API key (falls back to GEMINI_API_KEY env var).",
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


GEMINI_INSTRUCT = """Given several chunks of a reasoning trace and a ground-truth answer:
- For each chunk, if it contains an intermediate result, return it; else return null.
- If an intermediate result exists, compare it to the ground-truth answer:
  * true if it matches
  * false if it does not match
  * null if no intermediate answer is present.
Return a JSON array like:
[{"id": "1", "result": "6+9i" | null, "correctness": true | false | null}, ...]
Use 1-based ids matching the order of the chunks provided.
"""


def _maybe_bool(val: Any) -> Optional[bool]:
    if isinstance(val, bool) or val is None:
        return val
    if isinstance(val, str):
        low = val.strip().lower()
        if low in {"true", "yes"}:
            return True
        if low in {"false", "no"}:
            return False
    return None


def parse_gemini_labels(text: str) -> Optional[List[Dict[str, Any]]]:
    cleaned = text.strip()
    cleaned = re.sub(r"```(?:json)?", "", cleaned)
    cleaned = cleaned.replace("```", "")
    match = re.search(r"\[.*\]", cleaned, re.S)
    if match:
        cleaned = match.group(0)
    try:
        data = json.loads(cleaned)
    except Exception:
        return None
    if isinstance(data, list):
        return data
    return None


def gemini_label_chunks(record: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    # cfg: client, types, model, temperature, max_tokens
    gt_answer = record.get("answer")
    chunks = record.get("chunks", [])
    if not chunks or gt_answer is None:
        return {}

    client = cfg["client"]
    types = cfg["types"]

    reasoning_trace = [{"id": idx + 1, "chunk": c.get("text", "")} for idx, c in enumerate(chunks)]
    prompt = (
        GEMINI_INSTRUCT
        + "\nInput chunks: "
        + json.dumps(reasoning_trace, ensure_ascii=False)
        + f"\nGround-truth answer: {gt_answer}"
    )

    try:
        response = client.models.generate_content(
            model=cfg["model"],
            contents=[prompt],
            config=types.GenerateContentConfig(
                max_output_tokens=cfg["max_tokens"],
                temperature=cfg["temperature"],
            ),
        )
        parsed = parse_gemini_labels(response.text)
    except Exception as exc:  # pragma: no cover - network path
        print(f"[gemini] Error for example {record.get('example_id')}: {exc}")
        return {}

    if not parsed:
        print(f"[gemini] Failed to parse labels for example {record.get('example_id')}")
        return {}

    labels: Dict[int, Dict[str, Any]] = {}
    for entry in parsed:
        try:
            idx = int(entry.get("id"))
        except Exception:
            continue
        labels[idx - 1] = {
            "intermediate_answer": entry.get("result"),
            "is_correct": _maybe_bool(entry.get("correctness")),
        }
    return labels


def label_record(record: Dict[str, Any], mode: str = "regex", gemini_cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    gt_answer = record.get("answer")
    final_answer = record.get("final_answer") or extract_intermediate_answer(record.get("full_cot", ""))

    chunks = record.get("chunks", [])
    gemini_labels = gemini_label_chunks(record, gemini_cfg) if mode == "gemini" and gemini_cfg else {}
    prev_answer = None
    prev_correct = None
    prev_idx = None
    backtracking_events: List[Dict[str, Any]] = []

    for idx, chunk in enumerate(chunks):
        text = chunk.get("text", "")
        glabel = gemini_labels.get(idx, {}) if gemini_labels else {}

        intermediate = (
            glabel.get("intermediate_answer")
            or chunk.get("intermediate_answer")
            or extract_intermediate_answer(text)
        )
        chunk["intermediate_answer"] = intermediate

        is_correct = glabel.get("is_correct")
        if is_correct is None and gt_answer:
            is_correct = answers_match(intermediate, gt_answer)
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
    gemini_cfg = None
    if args.labeling_mode == "gemini":
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise SystemExit(
                "google-genai is required for --labeling-mode=gemini. Install via `pip install google-genai`."
            ) from exc
        api_key = args.gemini_api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise SystemExit("Set GEMINI_API_KEY or pass --gemini-api-key for Gemini labeling.")
        gemini_cfg = {
            "client": genai.Client(api_key=api_key),
            "types": types,
            "model": args.gemini_model,
            "temperature": args.gemini_temperature,
            "max_tokens": args.gemini_max_tokens,
        }

    labeled = [label_record(record, args.labeling_mode, gemini_cfg) for record in read_jsonl(args.input)]
    write_jsonl(output_path, labeled)
    print(f"Wrote {len(labeled)} labeled examples to {output_path}")


if __name__ == "__main__":
    main()
