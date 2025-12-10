# Backtracking Interpretability

We want to understand when reasoning models change their mind, and whether that helps. Three questions: will the answer change (stability), will backtracking help (productive vs not), and how early we can tell.

Current pipeline:
- generate a chain-of-thought (full trace)
- split into chunks
- tag intermediate answers and correctness/stability/backtrack markers (regex or Gemini)
- grab hidden states at chunk boundaries
- train small probes on top

Quick start:
```bash
bash scripts/setup_env.sh
bash scripts/get_data.sh
bash scripts/pull_models.sh

python -m src.pipeline.generate_cot --model gpt2 --dataset toy --input data/raw/toy.jsonl --max-examples 5
python -m src.pipeline.chunk_cot --input data/cot/gpt2/toy_rollouts.jsonl --dataset toy
python -m src.pipeline.label_intermediate --input data/chunks/segmented_toy.jsonl --dataset toy
python -m src.pipeline.dump_hidden --input data/labeled/labeled_intermediate_toy.jsonl --model gpt2 --dataset toy
```

Gemini labeling is available with `--labeling-mode gemini` and `GEMINI_API_KEY`.
