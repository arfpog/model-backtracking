# Backtracking Interpretability

Goal: one pipeline that emits a canonical trace bundle (question, full CoT, chunks, intermediate answers, correctness/stability/backtrack tags, hidden states) so we can train probes for backtracking and information gain.

## Layout
- `configs/` model + dataset YAMLs
- `scripts/` env setup, data download, model download
- `src/pipeline/` generation, chunking, labeling, reps
- `src/probes/` base probe definitions (linear/MLP)
- `data/`, `results/`, `notebooks/`
- `third_party/` Zhang et al. repo (kept only for reference)

## Quickstart
Install deps and spaCy model:
```bash
bash scripts/setup_env.sh
```
Pull datasets/models (edit configs as needed):
```bash
bash scripts/get_data.sh
bash scripts/pull_models.sh
```

Run the pipeline on a small slice (example with a tiny model):
```bash
# 1) Generate CoT
python -m src.pipeline.generate_cot \
  --model gpt2 \
  --dataset toy \
  --input data/raw/toy.jsonl \
  --question-field question \
  --answer-field answer \
  --max-examples 5 \
  --max-new-tokens 128

# 2) Chunk
python -m src.pipeline.chunk_cot \
  --input data/cot/gpt2/toy_rollouts.jsonl \
  --dataset toy

# 3) Label
python -m src.pipeline.label_intermediate \
  --input data/chunks/segmented_toy.jsonl \
  --dataset toy

# 4) Hidden states (keep batch-size small for large models)
python -m src.pipeline.dump_hidden \
  --input data/labeled/labeled_intermediate_toy.jsonl \
  --model gpt2 \
  --dataset toy \
  --batch-size 2 \
  --max-length 512
```

Notes: Pipeline uses only local Hugging Face + spaCy; no third_party code is invoked. Labeling supports regex or Gemini (set `GEMINI_API_KEY`, install `google-genai`). Adjust batch sizes and max length for bigger models. Probes live under `src/probes/` and expect reps + labels.
