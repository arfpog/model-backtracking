"""
Lightweight data structures for the canonical TraceBundle artifact.
These are intentionally permissive so early pipeline stages can fill fields
incrementally (generate CoT -> chunk -> label -> reps).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Chunk:
    chunk_idx: int
    text: str
    start_token: Optional[int] = None
    end_token: Optional[int] = None
    intermediate_answer: Optional[str] = None
    is_correct: Optional[bool] = None
    is_stable: Optional[bool] = None
    is_backtrack_start: Optional[bool] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktrackingEvent:
    chunk_before: int
    chunk_after: int
    outcome: Optional[str] = None
    is_productive: Optional[bool] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceBundle:
    example_id: str
    question: str
    answer: Optional[str] = None  # ground-truth
    final_answer: Optional[str] = None  # model's final answer
    model_name: Optional[str] = None
    dataset_name: Optional[str] = None
    full_cot: Optional[str] = None
    chunks: List[Chunk] = field(default_factory=list)
    backtracking_events: List[BacktrackingEvent] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

