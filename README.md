# Backtracking Interpretability

Shared pipeline for studying backtracking and information gain in reasoning models (e.g., DeepSeek-R1). The goal is to produce standardized reasoning traces + labels + hidden states, then plug RQ-specific probes on top.

## Repository layout
- `configs/`: YAML configs for models and datasets.
- `data/`: raw → chunked → labeled → representation artifacts.
- `scripts/`: environment setup, data pulls, model downloads.
- `src/`: pipeline and probe code (to be added).
- `third_party/reasoning_models_probing`: Zhang et al. repo as a submodule (trace generation utilities).
- `notebooks/`, `results/`: analysis and outputs.

## Setup
1) Create the conda env and install deps:
```bash
bash scripts/setup_env.sh  # optional: ENV_NAME=myenv PYTHON_VERSION=3.11 bash scripts/setup_env.sh
```
2) Pull datasets listed in `configs/datasets.yaml`:
```bash
bash scripts/get_data.sh  # or: python scripts/get_data.py --config configs/datasets.yaml --out data/raw
```
3) Download models listed in `configs/models.yaml` (HF auth may be required):
```bash
bash scripts/pull_models.sh  # or: python scripts/pull_models.py --config configs/models.yaml --out models
```

## Data / pipeline stages (planned)
1) Generate CoT rollouts from reasoning models → `data/cot/`
2) spaCy chunking of CoT → `data/chunks/` (see `src/pipeline/chunk_cot.py`)
3) Intermediate answer extraction + labels → `data/labeled/` (see `src/pipeline/label_intermediate.py`)
4) Hidden state dumps → `data/reps/` (see `src/pipeline/dump_hidden.py`)
5) Probe training (correctness, change, IG) → `results/probes/` (stubs in `src/probes/`)

## Notes
- Pipeline entrypoints under `src/pipeline` use local Hugging Face code (no third_party wiring).
- Zhang scripts remain in `third_party/` for reproducibility.
