#!/usr/bin/env python
"""
Generate Chain-of-Thought rollouts using either local models or OpenRouter API.

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
        return result["choices"][0]["message"]["content"]
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
    for idx, example in enumerate(tqdm(data, desc="Generating (OpenRouter)")):
        question = example.get(args.question_field)
        gt_answer = example.get(args.answer_field)
        example_id = example.get(args.id_field, idx)
        if question is None:
            continue

        if args.choices_field and args.choices_field in example:
            choices = example[args.choices_field]
            question = format_question_with_choices(str(question), choices)

        if args.answer_type == "letter":
            gt_answer = convert_answer_index_to_letter(gt_answer)

        prompt = build_prompt(str(question), prompt_prefix)

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
            print(f"[warning] Skipping example {example_id} due to API error")
            continue

        full_output, reasoning, answer_portion = parse_reasoning_output(raw_output, args.model)
        final_answer = extract_intermediate_answer(raw_output, answer_type=args.answer_type)

        rows.append(
            {
                "example_id": str(example_id),
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

        # Rate limiting
        time.sleep(args.rate_limit_delay)

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
    for idx, example in enumerate(tqdm(data, desc="Generating (local)")):
        question = example.get(args.question_field)
        gt_answer = example.get(args.answer_field)
        example_id = example.get(args.id_field, idx)
        if question is None:
            continue

        if args.choices_field and args.choices_field in example:
            choices = example[args.choices_field]
            question = format_question_with_choices(str(question), choices)

        if args.answer_type == "letter":
            gt_answer = convert_answer_index_to_letter(gt_answer)

        prompt = build_prompt(str(question), prompt_prefix)
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

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
        full_output, reasoning, answer_portion = parse_reasoning_output(raw_output, args.model)
        final_answer = extract_intermediate_answer(raw_output, answer_type=args.answer_type)

        rows.append(
            {
                "example_id": str(example_id),
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
    print(f"[generate_cot] Wrote {len(rows)} examples to {output_path}")

    # Preview
    for sample in rows[:3]:
        reasoning_preview = sample["reasoning"][:200].replace("\n", " ")
        print(
            f"[preview] id={sample['example_id']} | q={str(sample['question'])[:80]}... | reasoning={reasoning_preview}..."
        )


if __name__ == "__main__":
    main()
