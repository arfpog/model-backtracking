#!/usr/bin/env python
"""
Generate Chain-of-Thought rollouts using either local models or OpenRouter API.

Features:
- Multiple rollouts per question (--num-rollouts)
- Degenerate response detection and retry
- Configurable quality filters

Usage:
  # Local model
  python -m src.pipeline.generate_cot --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B ...

  # OpenRouter API
  python -m src.pipeline.generate_cot --model deepseek/deepseek-r1 --api openrouter ...
"""

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

try:
    from .utils import (
        convert_answer_index_to_letter,
        extract_intermediate_answer,
        format_question_with_choices,
        read_jsonl,
        write_jsonl,
    )
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.pipeline.utils import (
        convert_answer_index_to_letter,
        extract_intermediate_answer,
        format_question_with_choices,
        read_jsonl,
        write_jsonl,
    )

# Model-specific thinking tag patterns
THINK_PATTERNS = {
    "deepseek": (r"<think>", r"</think>"),
    "qwq": (r"<think>", r"</think>"),
    "default": (r"<think>", r"</think>"),
}


# =============================================================================
# Degenerate Response Detection
# =============================================================================


def is_degenerate(text: str, min_length: int = 50, max_repeat_ratio: float = 0.5) -> Tuple[bool, str]:
    """
    Check if a response is degenerate.

    Returns:
        (is_degenerate, reason)
    """
    if not text or not text.strip():
        return True, "empty"

    text = text.strip()

    # Too short
    if len(text) < min_length:
        return True, f"too_short ({len(text)} chars)"

    # Check for excessive repetition
    words = text.lower().split()
    if len(words) > 10:
        # Check for repeated phrases (n-grams)
        for n in [3, 4, 5]:
            if len(words) >= n * 2:
                ngrams = [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]
                unique_ratio = len(set(ngrams)) / len(ngrams)
                if unique_ratio < (1 - max_repeat_ratio):
                    return True, f"repetitive ({n}-grams unique ratio: {unique_ratio:.2f})"

    # Check for character-level repetition (e.g., "aaaaa" or "123123123")
    if len(text) > 100:
        # Check if any single character dominates
        char_counts = {}
        for c in text:
            char_counts[c] = char_counts.get(c, 0) + 1
        max_char_ratio = max(char_counts.values()) / len(text)
        if max_char_ratio > 0.3 and max(char_counts, key=char_counts.get) not in " \n":
            return True, f"char_repetition (max char ratio: {max_char_ratio:.2f})"

    # Check for repeated lines
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) > 5:
        unique_lines = set(lines)
        if len(unique_lines) / len(lines) < 0.3:
            return True, "repeated_lines"

    # Check for truncation indicators
    truncation_patterns = [
        r"\.\.\.$",  # Ends with ...
        r"I'll continue",  # Incomplete
        r"Let me finish",
        r"\[truncated\]",
        r"\[continued\]",
    ]
    for pattern in truncation_patterns:
        if re.search(pattern, text[-100:], re.IGNORECASE):
            return True, "truncated"

    return False, "ok"


def has_valid_reasoning(text: str, model_name: str = "") -> Tuple[bool, str]:
    """
    Check if response contains valid reasoning structure.

    Returns:
        (is_valid, reason)
    """
    # Check for think tags if expected
    model_lower = model_name.lower()
    if "deepseek" in model_lower or "qwq" in model_lower:
        if "<think>" in text and "</think>" not in text:
            return False, "unclosed_think_tag"

    # Check for some reasoning indicators
    reasoning_indicators = [
        r"\blet\s+me\b",
        r"\bfirst\b",
        r"\bstep\b",
        r"\bcalculat",
        r"\bsolv",
        r"\bthink",
        r"\bconsider",
        r"\bso\b",
        r"\btherefore\b",
        r"\bbecause\b",
        r"=",  # Math operations
    ]

    indicator_count = sum(1 for p in reasoning_indicators if re.search(p, text, re.IGNORECASE))
    if indicator_count < 2:
        return False, "no_reasoning_structure"

    return True, "ok"


def validate_response(
    text: str,
    model_name: str = "",
    min_length: int = 50,
    max_repeat_ratio: float = 0.5,
) -> Tuple[bool, str]:
    """
    Validate a response for quality.

    Returns:
        (is_valid, reason)
    """
    # Check for degenerate patterns
    is_degen, reason = is_degenerate(text, min_length, max_repeat_ratio)
    if is_degen:
        return False, f"degenerate:{reason}"

    # Check for valid reasoning
    is_valid, reason = has_valid_reasoning(text, model_name)
    if not is_valid:
        return False, f"invalid:{reason}"

    return True, "ok"


# =============================================================================
# Response Parsing
# =============================================================================


def parse_reasoning_output(text: str, model_name: str = "") -> Tuple[str, str, str]:
    """
    Parse model output into reasoning and answer portions.

    Returns:
        (full_output, reasoning_only, answer_portion)
    """
    model_lower = model_name.lower()
    if "deepseek" in model_lower:
        open_tag, close_tag = THINK_PATTERNS["deepseek"]
    elif "qwq" in model_lower:
        open_tag, close_tag = THINK_PATTERNS["qwq"]
    else:
        open_tag, close_tag = THINK_PATTERNS["default"]

    pattern = f"{open_tag}(.*?){close_tag}"
    match = re.search(pattern, text, re.DOTALL)

    if match:
        reasoning = match.group(1).strip()
        end_pos = match.end()
        answer_portion = text[end_pos:].strip()
        return text, reasoning, answer_portion

    # No think tags - try to split at answer indicators
    answer_indicators = [
        r"\\boxed\{",
        r"(?i)the\s+answer\s+is",
        r"(?i)final\s+answer",
        r"(?i)therefore,?\s+the\s+answer",
    ]

    for indicator in answer_indicators:
        match = re.search(indicator, text)
        if match:
            reasoning = text[: match.start()].strip()
            answer_portion = text[match.start() :].strip()
            return text, reasoning, answer_portion

    return text, text, ""


# =============================================================================
# Argument Parsing
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate CoT rollouts.")
    parser.add_argument("--model", required=True, help="Model name/id (HF or OpenRouter).")
    parser.add_argument("--dataset", required=True, help="Dataset name (for naming outputs).")
    parser.add_argument("--input", required=True, help="Dataset path (JSON/JSONL).")
    parser.add_argument("--question-field", default="question", help="Field name for question.")
    parser.add_argument("--answer-field", default="answer", help="Field name for ground-truth.")
    parser.add_argument("--choices-field", default=None, help="Field for MMLU-style choices.")
    parser.add_argument(
        "--answer-type",
        choices=["numeric", "letter", "auto"],
        default="numeric",
        help="Answer type: numeric (GSM8K/MATH), letter (MMLU), or auto.",
    )
    parser.add_argument("--id-field", default="id", help="Field name for example id.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling.")
    parser.add_argument("--max-new-tokens", type=int, default=8192, help="Max tokens to generate.")
    parser.add_argument("--max-examples", type=int, default=None, help="Max examples to process.")
    parser.add_argument(
        "--prompt-prefix",
        default=None,
        help="Instruction after the question. Defaults based on answer-type.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSONL path. Defaults to data/cot/<model>/<dataset>_rollouts.jsonl",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")

    # Multiple rollouts
    parser.add_argument(
        "--num-rollouts",
        type=int,
        default=1,
        help="Number of rollouts per question.",
    )

    # Retry and validation
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries for degenerate responses.",
    )
    parser.add_argument(
        "--min-response-length",
        type=int,
        default=50,
        help="Minimum response length (chars) to be valid.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip response validation (accept all responses).",
    )

    # API vs local options
    parser.add_argument(
        "--api",
        choices=["local", "openrouter"],
        default="local",
        help="API backend: local (HF transformers) or openrouter.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key (or set OPENROUTER_API_KEY env var).",
    )
    parser.add_argument(
        "--api-base",
        default="https://openrouter.ai/api/v1",
        help="API base URL for OpenRouter.",
    )
    parser.add_argument(
        "--rate-limit-delay",
        type=float,
        default=0.5,
        help="Delay between API calls in seconds.",
    )

    # Local-only options
    parser.add_argument("--device", default="auto", help="Device (local only).")
    parser.add_argument("--dtype", default=None, help="Torch dtype (local only).")
    parser.add_argument("--trust-remote-code", action="store_true", help="HF trust remote code.")

    return parser


def load_input(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return read_jsonl(path)
    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
    raise ValueError(f"Unsupported input format for {path}")


def build_prompt(question: str, prefix: str) -> str:
    return f"{question.strip()}\n\n{prefix.strip()}\n"


def get_default_prompt_prefix(answer_type: str) -> str:
    if answer_type == "letter":
        return "Please reason step by step, then provide your final answer as a single letter (A, B, C, or D)."
    return "Please reason step by step, and put your final answer within \\boxed{}."


# =============================================================================
# OpenRouter API Backend
# =============================================================================


def generate_openrouter(
    prompt: str,
    model: str,
    api_key: str,
    api_base: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> Optional[str]:
    """Generate completion using OpenRouter API."""
    import requests

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/model-backtracking",
    }

    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        # Include reasoning tokens (think tags) in output for reasoning models
        "include_reasoning": True,
        # Route to highest throughput providers for faster generation
        "provider": {
            "sort": "throughput",
            "allow_fallbacks": True,
        },
    }

    try:
        response = requests.post(
            f"{api_base}/chat/completions",
            headers=headers,
            json=data,
            timeout=300,
        )
        response.raise_for_status()
        result = response.json()
        message = result["choices"][0]["message"]
        content = message.get("content", "")
        # Some providers return reasoning in a separate field
        reasoning = message.get("reasoning", "") or message.get("reasoning_content", "")
        if reasoning and "<think>" not in content:
            # Combine reasoning and content with think tags
            return f"<think>\n{reasoning}\n</think>\n{content}"
        return content
    except requests.exceptions.RequestException as e:
        print(f"[openrouter] API error: {e}")
        return None
    except (KeyError, IndexError) as e:
        print(f"[openrouter] Parse error: {e}")
        return None


def run_openrouter(args, data: List[Dict], prompt_prefix: str) -> List[Dict[str, Any]]:
    """Run generation using OpenRouter API."""
    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENROUTER_API_KEY or pass --api-key for OpenRouter.")

    rows: List[Dict[str, Any]] = []
    stats = {"total": 0, "success": 0, "retries": 0, "failed": 0}

    total_iterations = len(data) * args.num_rollouts
    pbar = tqdm(total=total_iterations, desc="Generating (OpenRouter)")

    for idx, example in enumerate(data):
        question = example.get(args.question_field)
        gt_answer = example.get(args.answer_field)
        example_id = example.get(args.id_field, idx)
        if question is None:
            pbar.update(args.num_rollouts)
            continue

        if args.choices_field and args.choices_field in example:
            choices = example[args.choices_field]
            question = format_question_with_choices(str(question), choices)

        if args.answer_type == "letter":
            gt_answer = convert_answer_index_to_letter(gt_answer)

        prompt = build_prompt(str(question), prompt_prefix)

        # Generate multiple rollouts
        for rollout_idx in range(args.num_rollouts):
            stats["total"] += 1
            raw_output = None
            validation_reason = "no_response"

            # Retry loop for degenerate responses
            for retry in range(args.max_retries):
                raw_output = generate_openrouter(
                    prompt=prompt,
                    model=args.model,
                    api_key=api_key,
                    api_base=args.api_base,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_new_tokens,
                )

                if raw_output is None:
                    validation_reason = "api_error"
                    time.sleep(args.rate_limit_delay * 2)  # Extra delay on error
                    continue

                # Validate response
                if args.skip_validation:
                    break

                is_valid, validation_reason = validate_response(
                    raw_output,
                    args.model,
                    args.min_response_length,
                )

                if is_valid:
                    break

                # Log retry
                if retry < args.max_retries - 1:
                    stats["retries"] += 1
                    tqdm.write(
                        f"[retry {retry + 1}/{args.max_retries}] example={example_id} "
                        f"rollout={rollout_idx} reason={validation_reason}"
                    )
                    time.sleep(args.rate_limit_delay)

            # Check if we got a valid response
            if raw_output is None or (not args.skip_validation and not is_valid):
                stats["failed"] += 1
                tqdm.write(
                    f"[failed] example={example_id} rollout={rollout_idx} reason={validation_reason}"
                )
                pbar.update(1)
                time.sleep(args.rate_limit_delay)
                continue

            stats["success"] += 1
            full_output, reasoning, answer_portion = parse_reasoning_output(raw_output, args.model)
            final_answer = extract_intermediate_answer(raw_output, answer_type=args.answer_type)

            rows.append(
                {
                    "example_id": str(example_id),
                    "rollout_idx": rollout_idx,
                    "question": question,
                    "answer": gt_answer,
                    "answer_type": args.answer_type,
                    "model": args.model,
                    "dataset": args.dataset,
                    "prompt": prompt,
                    "full_output": full_output,
                    "reasoning": reasoning,
                    "answer_portion": answer_portion,
                    "final_answer": final_answer,
                    "metadata": {
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "max_new_tokens": args.max_new_tokens,
                        "api": "openrouter",
                    },
                }
            )

            pbar.update(1)
            time.sleep(args.rate_limit_delay)

    pbar.close()

    # Print stats
    print(f"\n[stats] total={stats['total']} success={stats['success']} "
          f"retries={stats['retries']} failed={stats['failed']}")

    return rows


# =============================================================================
# Local HuggingFace Backend
# =============================================================================


def run_local(args, data: List[Dict], prompt_prefix: str) -> List[Dict[str, Any]]:
    """Run generation using local HuggingFace model."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    def select_device(name: str) -> torch.device:
        if name == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(name)

    def decode_continuation(tokenizer, prompt_ids, generated_ids) -> str:
        prompt_len = prompt_ids.shape[1]
        continuation = generated_ids[:, prompt_len:]
        return tokenizer.batch_decode(continuation, skip_special_tokens=True)[0]

    torch.manual_seed(args.seed)
    dtype = getattr(torch, args.dtype) if args.dtype else None
    device = select_device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto" if args.device == "auto" else None,
        trust_remote_code=args.trust_remote_code,
    )
    if args.device != "auto":
        model.to(device)
    model.eval()

    rows: List[Dict[str, Any]] = []
    stats = {"total": 0, "success": 0, "retries": 0, "failed": 0}

    total_iterations = len(data) * args.num_rollouts
    pbar = tqdm(total=total_iterations, desc="Generating (local)")

    for idx, example in enumerate(data):
        question = example.get(args.question_field)
        gt_answer = example.get(args.answer_field)
        example_id = example.get(args.id_field, idx)
        if question is None:
            pbar.update(args.num_rollouts)
            continue

        if args.choices_field and args.choices_field in example:
            choices = example[args.choices_field]
            question = format_question_with_choices(str(question), choices)

        if args.answer_type == "letter":
            gt_answer = convert_answer_index_to_letter(gt_answer)

        prompt = build_prompt(str(question), prompt_prefix)
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate multiple rollouts
        for rollout_idx in range(args.num_rollouts):
            stats["total"] += 1
            raw_output = None
            validation_reason = "no_response"

            # Retry loop for degenerate responses
            for retry in range(args.max_retries):
                # Set different seed for each retry
                torch.manual_seed(args.seed + idx * 1000 + rollout_idx * 100 + retry)

                with torch.no_grad():
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=True,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )

                raw_output = decode_continuation(tokenizer, inputs["input_ids"], generated)

                # Validate response
                if args.skip_validation:
                    break

                is_valid, validation_reason = validate_response(
                    raw_output,
                    args.model,
                    args.min_response_length,
                )

                if is_valid:
                    break

                # Log retry
                if retry < args.max_retries - 1:
                    stats["retries"] += 1
                    tqdm.write(
                        f"[retry {retry + 1}/{args.max_retries}] example={example_id} "
                        f"rollout={rollout_idx} reason={validation_reason}"
                    )

            # Check if we got a valid response
            if raw_output is None or (not args.skip_validation and not is_valid):
                stats["failed"] += 1
                tqdm.write(
                    f"[failed] example={example_id} rollout={rollout_idx} reason={validation_reason}"
                )
                pbar.update(1)
                continue

            stats["success"] += 1
            full_output, reasoning, answer_portion = parse_reasoning_output(raw_output, args.model)
            final_answer = extract_intermediate_answer(raw_output, answer_type=args.answer_type)

            rows.append(
                {
                    "example_id": str(example_id),
                    "rollout_idx": rollout_idx,
                    "question": question,
                    "answer": gt_answer,
                    "answer_type": args.answer_type,
                    "model": args.model,
                    "dataset": args.dataset,
                    "prompt": prompt,
                    "full_output": full_output,
                    "reasoning": reasoning,
                    "answer_portion": answer_portion,
                    "final_answer": final_answer,
                    "metadata": {
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "max_new_tokens": args.max_new_tokens,
                        "api": "local",
                    },
                }
            )

            pbar.update(1)

    pbar.close()

    # Print stats
    print(f"\n[stats] total={stats['total']} success={stats['success']} "
          f"retries={stats['retries']} failed={stats['failed']}")

    return rows


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    args = build_parser().parse_args()
    random.seed(args.seed)

    prompt_prefix = args.prompt_prefix or get_default_prompt_prefix(args.answer_type)

    print(f"[generate_cot] Loading input from {args.input}")
    data = load_input(Path(args.input))
    print(f"[generate_cot] Loaded {len(data)} rows (dataset={args.dataset}, model={args.model})")

    if args.max_examples:
        data = data[: args.max_examples]
        print(f"[generate_cot] Truncated to {len(data)} rows due to --max-examples")

    print(f"[generate_cot] Generating {args.num_rollouts} rollout(s) per question")
    print(f"[generate_cot] Max retries for degenerate responses: {args.max_retries}")

    # Run generation
    if args.api == "openrouter":
        rows = run_openrouter(args, data, prompt_prefix)
    else:
        rows = run_local(args, data, prompt_prefix)

    # Save output
    model_slug = args.model.replace("/", "_")
    default_out = Path("data/cot") / model_slug / f"{args.dataset}_rollouts.jsonl"
    output_path = args.output or default_out

    write_jsonl(output_path, rows)
    print(f"[generate_cot] Wrote {len(rows)} rollouts to {output_path}")

    # Preview
    for sample in rows[:3]:
        reasoning_preview = sample["reasoning"][:200].replace("\n", " ")
        print(
            f"[preview] id={sample['example_id']} rollout={sample.get('rollout_idx', 0)} | "
            f"q={str(sample['question'])[:60]}... | reasoning={reasoning_preview}..."
        )


if __name__ == "__main__":
    main()
