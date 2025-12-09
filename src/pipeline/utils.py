"""
Shared helpers for the trace pipeline.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

BACKTRACK_KEYWORDS = [
    "wait",
    "actually",
    "no,",
    "hmm",
    "let me reconsider",
    "that's not right",
    "i made a mistake",
    "going back",
    "let me try again",
    "alternatively",
    "on second thought",
    "hold on",
    "but wait",
    "that doesn't seem right",
]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True))
            f.write("\n")


def normalize_answer(ans: Optional[str]) -> Optional[str]:
    if ans is None:
        return None
    cleaned = ans.strip().lower()
    cleaned = cleaned.replace(",", "")
    cleaned = re.sub(r"\\boxed\s*\{([^}]*)\}", r"\1", cleaned)
    cleaned = re.sub(r"\\boxed\s*\(([^)]*)\)", r"\1", cleaned)
    cleaned = re.sub(r"\\boxed", "", cleaned)
    cleaned = re.sub(r"\\[a-z]+", "", cleaned)  # strip LaTeX commands
    cleaned = re.sub(r"[^0-9a-z\./+-]", "", cleaned)
    return cleaned


def answers_match(a: Optional[str], b: Optional[str]) -> bool:
    na, nb = normalize_answer(a), normalize_answer(b)
    return na is not None and nb is not None and na == nb


def extract_intermediate_answer(text: str) -> Optional[str]:
    boxed = re.findall(r"\\boxed\s*[({]?([^}\)]*)[)}]?", text)
    if boxed:
        return boxed[-1].strip()
    explicit = re.findall(r"(?i)answer\s*[:=]\s*([A-Za-z0-9\.\-/]+)", text)
    if explicit:
        return explicit[-1].strip()
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1]
    letters = re.findall(r"\\b([A-E])\\b", text)
    if letters:
        return letters[-1]
    return None


def detect_backtrack(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in BACKTRACK_KEYWORDS)
