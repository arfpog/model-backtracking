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

    # Handle GSM8K format: extract number after ####
    gsm_match = re.search(r"####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)", cleaned)
    if gsm_match:
        cleaned = gsm_match.group(1)

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


def extract_intermediate_answer(text: str, answer_type: str = "numeric") -> Optional[str]:
    """
    Extract intermediate answer from reasoning text.

    Args:
        text: The reasoning text to extract from.
        answer_type: "numeric" (GSM8K/MATH), "letter" (MMLU), or "auto".
            - "numeric": prioritizes boxed expressions and numbers
            - "letter": prioritizes letter choices (A-E)
            - "auto": tries all methods in order
    """
    # Common patterns
    boxed = re.findall(r"\\boxed\s*[({]?([^}\)]*)[)}]?", text)
    explicit_letter = re.findall(r"(?i)(?:the\s+)?answer\s+is\s*:?\s*([A-E])\b", text)
    explicit_general = re.findall(r"(?i)answer\s*[:=]\s*([A-Za-z0-9\.\-/]+)", text)
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    letters = re.findall(r"\b([A-E])\b", text)

    if answer_type == "letter":
        # For multiple choice: prioritize letter extraction
        if explicit_letter:
            return explicit_letter[-1].strip().upper()
        if letters:
            return letters[-1].upper()
        if boxed:
            # Check if boxed contains a letter
            boxed_content = boxed[-1].strip()
            if re.match(r"^[A-E]$", boxed_content, re.I):
                return boxed_content.upper()
            return boxed_content
        if explicit_general:
            return explicit_general[-1].strip()
        return None

    elif answer_type == "numeric":
        # For math problems: prioritize numeric extraction
        if boxed:
            return boxed[-1].strip()
        if explicit_general:
            return explicit_general[-1].strip()
        if numbers:
            return numbers[-1]
        if letters:
            return letters[-1].upper()
        return None

    else:  # auto - try everything
        if boxed:
            return boxed[-1].strip()
        if explicit_letter:
            return explicit_letter[-1].strip().upper()
        if explicit_general:
            return explicit_general[-1].strip()
        if numbers:
            return numbers[-1]
        if letters:
            return letters[-1].upper()
        return None


def detect_backtrack(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in BACKTRACK_KEYWORDS)


# MMLU-style answer formatting helpers
INDEX_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D"}


def format_question_with_choices(question: str, choices: List[str]) -> str:
    """Format a multiple-choice question with labeled choices."""
    formatted = question.strip() + "\n\n"
    for i, choice in enumerate(choices):
        letter = INDEX_TO_LETTER.get(i, chr(ord("A") + i))
        formatted += f"{letter}) {choice}\n"
    return formatted.strip()


def convert_answer_index_to_letter(answer: Any) -> str:
    """Convert numeric answer index (0-3) to letter (A-D)."""
    if isinstance(answer, int):
        return INDEX_TO_LETTER.get(answer, str(answer))
    if isinstance(answer, str) and answer.isdigit():
        return INDEX_TO_LETTER.get(int(answer), answer)
    return str(answer)
