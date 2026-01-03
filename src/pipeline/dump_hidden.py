#!/usr/bin/env python

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .utils import read_jsonl
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.pipeline.utils import read_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dump hidden states for chunked traces.")
    parser.add_argument("--input", type=Path, required=True, help="Labeled chunk JSONL.")
    parser.add_argument("--model", required=True, help="Model name or HF id.")
    parser.add_argument("--dataset", help="Dataset name (for naming outputs).")
    parser.add_argument("--layer", type=int, default=-1, help="Layer to extract (default last).")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for forward pass.")
    parser.add_argument("--shard-size", type=int, default=512, help="Number of rows per shard.")
    parser.add_argument("--max-length", type=int, default=2048, help="Tokenizer max length.")
    parser.add_argument(
        "--prompt-prefix",
        default="Please reason step by step, and put your final answer within \\boxed{}.",
        help="Instruction appended after the question before chunk text.",
    )
    parser.add_argument(
        "--position-mode",
        choices=["last", "chunk_start"],
        default="last",
        help="Where to read hidden states: last token of prompt, or first token of current chunk.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for .pt shards. Defaults to data/reps/<model>/<dataset>/",
    )
    parser.add_argument("--device", default="auto", help="Device: auto/cpu/cuda/mps.")
    parser.add_argument("--dtype", default=None, help="Optional torch dtype, e.g., float16 or bfloat16.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Pass through to HF loaders.")
    return parser


def select_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def model_device(model) -> torch.device:
    if hasattr(model, "device"):
        return model.device
    if hasattr(model, "hf_device_map"):
        devices = {v for v in model.hf_device_map.values() if v not in ("disk", "cpu")}
        if devices:
            return torch.device(sorted(devices)[0])
    return next(model.parameters()).device


def build_prompt(question: str, prefix: str, merged_chunks: str) -> str:
    parts = [question.strip(), prefix.strip(), merged_chunks.strip()]
    return "\n\n".join(part for part in parts if part)


def compute_chunk_start(tokenizer, question: str, prefix: str, chunks: List[str], idx: int) -> int:
    """Returns the token index where chunk idx starts."""
    if idx == 0:
        base_prompt = build_prompt(question, prefix, "")
        return len(tokenizer(base_prompt, add_special_tokens=True).input_ids)
    else:
        prev_merged = "\n\n".join(chunks[:idx])
        prev_prompt = build_prompt(question, prefix, prev_merged)
        return len(tokenizer(prev_prompt, add_special_tokens=True).input_ids)


def iter_prompts(records: List[Dict], tokenizer, prefix: str, position_mode: str):
    for record in records:
        question = record.get("question", "")
        example_id = str(record.get("example_id"))
        chunk_texts = [c.get("text", "") for c in record.get("chunks", [])]

        for idx, chunk in enumerate(record.get("chunks", [])):
            prompt = build_prompt(question, prefix, "\n\n".join(chunk_texts[: idx + 1]))
            prompt_len = len(tokenizer(prompt, add_special_tokens=True).input_ids)

            if position_mode == "chunk_start":
                position = compute_chunk_start(tokenizer, question, prefix, chunk_texts, idx)
            else:  # "last"
                position = prompt_len - 1

            meta = {
                "example_id": example_id,
                "chunk_idx": idx,
                "intermediate_answer": chunk.get("intermediate_answer"),
                "is_correct": chunk.get("is_correct"),
                "is_stable": chunk.get("is_stable"),
                "is_backtrack_start": chunk.get("is_backtrack_start"),
            }
            yield prompt, position, meta


def gather_hidden_states(
    model, tokenizer, prompts: List[str], positions: List[int], max_length: int, layer_idx: int
) -> List[torch.Tensor]:
    """Extract hidden states at specified positions for each prompt."""
    tokenizer.padding_side = "left"
    encodings = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    device = model_device(model)
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

    layer = outputs.hidden_states[layer_idx]
    seq_lens = attention_mask.sum(dim=1)
    max_len = input_ids.shape[1]

    out_tensors: List[torch.Tensor] = []
    for b_idx, pos in enumerate(positions):
        seq_len = seq_lens[b_idx].item()
        pad_offset = max_len - seq_len  # account for left padding

        if pos < 0 or pos >= seq_len:
            pos = seq_len - 1  # fallback to last token
        adjusted_pos = pos + pad_offset
        out_tensors.append(layer[b_idx, adjusted_pos, :].detach().cpu())

    return out_tensors


def save_shard(output_dir: Path, shard_idx: int, tensors: List[torch.Tensor], meta: List[Dict]) -> None:
    if not tensors:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    tensor = torch.stack(tensors, dim=0)
    path = output_dir / f"part-{shard_idx:04d}.pt"
    torch.save({"hidden_states": tensor, "meta": meta}, path)
    print(f"Saved shard {shard_idx} -> {path} ({tensor.shape[0]} rows)")


def main() -> None:
    args = build_parser().parse_args()
    dataset = args.dataset or args.input.stem
    default_out_dir = Path("data/reps") / args.model / dataset
    output_dir = args.output_dir or default_out_dir

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

    records = read_jsonl(args.input)
    prompt_meta = list(iter_prompts(records, tokenizer, args.prompt_prefix, args.position_mode))
    if not prompt_meta:
        print("No chunks found in input; nothing to encode.")
        return

    buffer_tensors: List[torch.Tensor] = []
    buffer_meta: List[Dict] = []
    shard_idx = 0

    for start in tqdm(range(0, len(prompt_meta), args.batch_size), desc="Encoding"):
        batch = prompt_meta[start : start + args.batch_size]
        prompts, positions, metas = zip(*batch)
        tensors = gather_hidden_states(
            model,
            tokenizer,
            list(prompts),
            list(positions),
            args.max_length,
            args.layer,
        )
        for tensor, meta in zip(tensors, metas):
            buffer_tensors.append(tensor)
            buffer_meta.append(dict(meta))

        if len(buffer_tensors) >= args.shard_size:
            save_shard(output_dir, shard_idx, buffer_tensors, buffer_meta)
            buffer_tensors, buffer_meta = [], []
            shard_idx += 1

    if buffer_tensors:
        save_shard(output_dir, shard_idx, buffer_tensors, buffer_meta)

if __name__ == "__main__":
    main()
