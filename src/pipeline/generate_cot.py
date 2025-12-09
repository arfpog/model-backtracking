#!/usr/bin/env python
"""
Stage 01: generate chain-of-thought rollouts from a reasoning model.

Expected output:
- data/cot/<model>/<dataset>_rollouts.jsonl (one JSON object per example)
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .utils import extract_intermediate_answer, read_jsonl, write_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate CoT rollouts.")
    parser.add_argument("--model", required=True, help="Model name or HF id.")
    parser.add_argument("--dataset", required=True, help="Dataset name (for naming outputs).")
    parser.add_argument("--input", required=True, help="Dataset path (JSON/JSONL).")
    parser.add_argument("--question-field", default="question", help="Field name for question text.")
    parser.add_argument("--answer-field", default="answer", help="Field name for ground-truth answer.")
    parser.add_argument("--id-field", default="id", help="Field name for example id (optional).")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling.")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Max new tokens to generate.")
    parser.add_argument("--max-examples", type=int, default=None, help="Max examples to process.")
    parser.add_argument("--device", default="auto", help="Device: auto/cpu/cuda/mps.")
    parser.add_argument("--dtype", default=None, help="Optional torch dtype, e.g., float16 or bfloat16.")
    parser.add_argument(
        "--prompt-prefix",
        default="Please reason step by step, and put your final answer within \\boxed{}.",
        help="Instruction appended after the question.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSONL path. Defaults to data/cot/<model>/<dataset>_rollouts.jsonl",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Pass through to HF loaders.")
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


def select_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def build_prompt(question: str, prefix: str) -> str:
    return f"{question.strip()}\n\n{prefix.strip()}\n"


def decode_continuation(tokenizer, prompt_ids, generated_ids) -> str:
    prompt_len = prompt_ids.shape[1]
    continuation = generated_ids[:, prompt_len:]
    return tokenizer.batch_decode(continuation, skip_special_tokens=True)[0]


def main() -> None:
    args = build_parser().parse_args()
    random.seed(args.seed)
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

    data = load_input(Path(args.input))
    if args.max_examples:
        data = data[: args.max_examples]

    default_out = Path("data/cot") / args.model / f"{args.dataset}_rollouts.jsonl"
    output_path = args.output or default_out

    rows: List[Dict[str, Any]] = []
    for idx, example in enumerate(data):
        question = example.get(args.question_field)
        gt_answer = example.get(args.answer_field)
        example_id = example.get(args.id_field, idx)
        if question is None:
            continue

        prompt = build_prompt(str(question), args.prompt_prefix)
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

        cot = decode_continuation(tokenizer, inputs["input_ids"], generated)
        final_answer = extract_intermediate_answer(cot)

        rows.append(
            {
                "example_id": str(example_id),
                "question": question,
                "answer": gt_answer,
                "model": args.model,
                "dataset": args.dataset,
                "prompt": prompt,
                "full_cot": cot,
                "final_answer": final_answer,
                "metadata": {
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_new_tokens": args.max_new_tokens,
                },
            }
        )

    write_jsonl(output_path, rows)
    print(f"Wrote {len(rows)} examples to {output_path}")


if __name__ == "__main__":
    main()
