#!/usr/bin/env python
"""
Stage 02: chunk CoT traces into intermediate reasoning segments.

Heuristics:
- First split on blank lines (DeepSeek often uses double newlines for steps).
- Start a new chunk when a transition keyword/phrase appears (similar to Zhang et al.).
- Fallback to spaCy sentence splits if no blank-line structure exists.

Expected output:
- data/chunks/segmented_<dataset>.jsonl (or model/dataset-specific path)
"""

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import spacy
from spacy.matcher import Matcher

from .utils import BACKTRACK_KEYWORDS, extract_intermediate_answer, read_jsonl, write_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chunk CoT rollouts into segments.")
    parser.add_argument("--input", type=Path, required=True, help="Input CoT JSONL.")
    parser.add_argument("--dataset", help="Dataset name (for naming outputs).")
    parser.add_argument(
        "--spacy-model",
        default="en_core_web_sm",
        help="spaCy model for sentence/chunk segmentation.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSONL path. Defaults to data/chunks/segmented_<dataset>.jsonl",
    )
    parser.add_argument("--max-chunks", type=int, default=None, help="Optional limit per example.")
    parser.add_argument(
        "--transition-markers",
        nargs="*",
        default=[
            "wait",
            "double-check",
            "alternatively",
            "make sure",
            "another way",
            "verify",
            "to confirm",
        ],
        help="Phrases that trigger a new chunk (lowercased). Defaults mirror Zhang et al. heuristics.",
    )
    return parser


def build_matcher(nlp, markers: Iterable[str]) -> Matcher:
    matcher = Matcher(nlp.vocab)
    for phrase in markers:
        if not phrase:
            continue
        pattern = [{"LOWER": w} for w in phrase.split()]
        matcher.add(phrase, [pattern])
    return matcher


def chunk_text(nlp, matcher: Matcher, text: str, max_chunks: int = None) -> List[str]:
    steps = [s.strip() for s in text.split("\n\n") if s.strip()]
    if not steps:
        steps = [sent.text.strip() for sent in nlp(text).sents if sent.text.strip()]
    if not steps and text.strip():
        steps = [text.strip()]

    chunks: List[str] = []
    current: List[str] = []
    for step in steps:
        doc = nlp(step)
        matches = matcher(doc)
        has_transition = bool(matches)
        if has_transition and current:
            chunks.append("\n\n".join(current))
            current = [step]
        else:
            current.append(step)
    if current:
        chunks.append("\n\n".join(current))

    if max_chunks:
        chunks = chunks[:max_chunks]
    return chunks


def main() -> None:
    args = build_parser().parse_args()
    dataset = args.dataset or args.input.stem
    default_out = Path("data/chunks") / f"segmented_{dataset}.jsonl"
    output_path = args.output or default_out

    nlp = spacy.load(args.spacy_model)
    matcher = build_matcher(nlp, set(args.transition_markers) | set(BACKTRACK_KEYWORDS))

    rows = []
    for record in read_jsonl(args.input):
        full_cot = record.get("full_cot", "")
        chunk_strings = chunk_text(nlp, matcher, full_cot, args.max_chunks)
        chunks: List[Dict[str, object]] = [
            {"chunk_idx": idx, "text": chunk} for idx, chunk in enumerate(chunk_strings)
        ]
        final_answer = record.get("final_answer") or extract_intermediate_answer(full_cot)
        rows.append(
            {
                **record,
                "chunks": chunks,
                "final_answer": final_answer,
            }
        )

    write_jsonl(output_path, rows)
    print(f"Wrote {len(rows)} examples to {output_path}")


if __name__ == "__main__":
    main()
