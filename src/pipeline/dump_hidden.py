#!/usr/bin/env python

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

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
        choices=["last", "chunk_end", "chunk_start", "custom"],
        default="last",
        help="Where to read hidden states. Custom reads explicit positions JSON.",
    )
    parser.add_argument(
        "--positions-json",
        type=Path,
        help="JSON mapping example_id -> chunk_idx -> list of token positions (only for custom mode).",
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


def load_positions_map(path: Path) -> Dict[str, Dict[str, List[int]]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(ex): {str(k): v for k, v in chunk_map.items()} for ex, chunk_map in data.items()}


def compute_offsets(tokenizer, question: str, prefix: str, chunks: List[str], idx: int) -> Tuple[int, int, int]:
    """
    Returns (prompt_len, chunk_start_idx, chunk_end_idx) in token space for chunk idx,
    using the same prompt construction as the forward pass.
    """
    base_prompt = build_prompt(question, prefix, "")
    base_len = len(tokenizer(base_prompt, add_special_tokens=True).input_ids)

    merged_text = "\n\n".join(chunks[: idx + 1])
    prompt = build_prompt(question, prefix, merged_text)
    prompt_len = len(tokenizer(prompt, add_special_tokens=True).input_ids)

    if idx == 0:
        prev_len = base_len
    else:
        prev_merged = "\n\n".join(chunks[:idx])
        prev_prompt = build_prompt(question, prefix, prev_merged)
        prev_len = len(tokenizer(prev_prompt, add_special_tokens=True).input_ids)

    chunk_start = prev_len
    chunk_end = prompt_len - 1
    return prompt_len, chunk_start, chunk_end


def iter_prompts(records: List[Dict], tokenizer, prefix: str, position_mode: str, positions_map=None):
    for record in records:
        question = record.get("question", "")
        example_id = str(record.get("example_id"))
        chunk_texts = [c.get("text", "") for c in record.get("chunks", [])]

        for idx, chunk in enumerate(record.get("chunks", [])):
            prompt_len, chunk_start, chunk_end = compute_offsets(
                tokenizer, question, prefix, chunk_texts, idx
            )
            prompt = build_prompt(question, prefix, "\n\n".join(chunk_texts[: idx + 1]))

            if position_mode == "last":
                positions = [prompt_len - 1]
            elif position_mode == "chunk_start":
                positions = [chunk_start]
            elif position_mode == "chunk_end":
                positions = [chunk_end]
            elif position_mode == "custom" and positions_map:
                positions = positions_map.get(example_id, {}).get(str(idx), [])
                if not positions:
                    positions = [chunk_end]
            else:
                positions = [chunk_end]

            meta = {
                "example_id": example_id,
                "chunk_idx": idx,
                "intermediate_answer": chunk.get("intermediate_answer"),
                "is_correct": chunk.get("is_correct"),
                "is_stable": chunk.get("is_stable"),
                "is_backtrack_start": chunk.get("is_backtrack_start"),
                "positions": positions,
            }
            yield prompt, positions, meta


def gather_positions(model, tokenizer, prompts: List[str], positions: List[List[int]], max_length: int, layer_idx: int):
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
        hidden = outputs.hidden_states

    layer = hidden[layer_idx]
    out_tensors: List[torch.Tensor] = []
    out_indices: List[Tuple[int, int]] = []

    seq_lens = attention_mask.sum(dim=1)
    max_len = input_ids.shape[1]
    
    for b_idx, pos_list in enumerate(positions):
        seq_len = seq_lens[b_idx].item()
        pad_offset = max_len - seq_len  # FIX: account for left padding
        
        for pos in pos_list:
            if pos < 0 or pos >= seq_len:
                continue
            adjusted_pos = pos + pad_offset  # FIX: adjust index
            out_tensors.append(layer[b_idx, adjusted_pos, :].detach().cpu())
            out_indices.append((b_idx, pos))  # keep original pos for metadata

    return out_tensors, out_indices


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

    positions_map = None
    if args.position_mode == "custom":
        if not args.positions_json:
            raise SystemExit("Custom mode requires --positions-json.")
        positions_map = load_positions_map(args.positions_json)

    records = read_jsonl(args.input)
    prompt_meta = list(iter_prompts(records, tokenizer, args.prompt_prefix, args.position_mode, positions_map))
    if not prompt_meta:
        print("No chunks found in input; nothing to encode.")
        return

    buffer_tensors: List[torch.Tensor] = []
    buffer_meta: List[Dict] = []
    shard_idx = 0

    for start in tqdm(range(0, len(prompt_meta), args.batch_size), desc="Encoding"):
        batch = prompt_meta[start : start + args.batch_size]
        prompts, positions_batch, metas = zip(*batch)
        tensors, indices = gather_positions(
            model,
            tokenizer,
            list(prompts),
            list(positions_batch),
            args.max_length,
            args.layer,
        )
        for tensor, (b_idx, pos) in zip(tensors, indices):
            meta_entry = dict(metas[b_idx])
            meta_entry["position"] = pos
            buffer_tensors.append(tensor)
            buffer_meta.append(meta_entry)

        if len(buffer_tensors) >= args.shard_size:
            save_shard(output_dir, shard_idx, buffer_tensors, buffer_meta)
            buffer_tensors, buffer_meta = [], []
            shard_idx += 1

    if buffer_tensors:
        save_shard(output_dir, shard_idx, buffer_tensors, buffer_meta)

if __name__ == "__main__":
    main()
