# Local LLM — Build an AI Model from Scratch

> **Full detailed plan**: See [PLAN.md](PLAN.md) for the comprehensive implementation plan with architecture details, code snippets, dataset choices, and training configs.

## Project Goal
Build a modern AI language model from scratch for deep understanding, then fine-tune an existing small model into a local agentic coding assistant.

## Hardware
- **MacBook M4** (16GB RAM): Primary coding + local inference via MLX
- **ASUS ROG SCAR 17** (32GB RAM, RTX 3090 16GB VRAM): Heavy training (CUDA, fp16, Flash Attention 2)

## Architecture Choices (2025-era, NOT GPT-2)
- **RoPE** (not learned positional embeddings)
- **RMSNorm** (not LayerNorm)
- **SwiGLU** (not GELU)
- **Grouped Query Attention** (not standard MHA)
- **WSD schedule** (not cosine decay)
- **DPO** (not RLHF)

## Phase Checklist

| # | Phase | Status | Tests | Key Files |
|---|---|---|---|---|
| 1 | **Foundations** (numpy) | DONE | 29 | `phase1_foundations/01-04_*.ipynb` |
| 2 | **Tokenizer** (BPE) | DONE | 36 | `phase2_tokenizer/bpe_tokenizer.py` |
| 3 | **Transformer** (PyTorch) | DONE | 25 | `phase3_transformer/model.py` + 6 modules |
| 4 | **Training** pipeline | DONE | 18 | `phase4_training/trainer.py`, `train.py`, configs |
| 5 | **Generation** & eval | TODO | — | `phase5_generation/generate.py`, `kv_cache.py` |
| 6 | **Fine-tune** SmolLM2-360M | TODO | — | `phase6_finetune/lora.py`, `finetune.py` |
| 7 | **Agent** (coding assistant) | TODO | — | `phase7_agent/agent.py`, `cli.py` |

**Total tests passing: 108**

### What's been built
- Phase 1: 4 notebooks (NN, backprop, attention, modern components) with terminology glossaries
- Phase 2: Byte-level BPE tokenizer with train/encode/decode/save/load + special tokens
- Phase 3: Full Llama-style transformer — RoPE, RMSNorm, GQA, SwiGLU, ~15M params, weight tying
- Phase 4: Training pipeline — dataset loading (TinyStories/FineWeb-Edu), WSD scheduler, multi-device trainer (CUDA fp16 / MPS fp32), gradient accumulation, checkpointing

### How to train the model
```bash
# Smoke test (any hardware, ~30 seconds)
.venv/bin/python3 -m phase4_training.train --config phase4_training/configs/tiny.yaml

# Full training (ROG SCAR 17 recommended, ~2h)
.venv/bin/python3 -m phase4_training.train --config phase4_training/configs/small.yaml
```

### Next up: Phase 5 (Generation & Evaluation)
- `generate.py` — temperature, top-k, top-p decoding strategies
- `kv_cache.py` — cache K/V for fast autoregressive inference
- `evaluate.py` — perplexity measurement
- `interactive.py` — REPL for chatting with the model

## Development Conventions
- Phase 1: Jupyter notebooks (foundations/visualization)
- Phase 2-7: Python scripts
- **Always include tests** — every module gets a test file to ensure robustness
- Run tests: `python -m pytest tests/ -v`
- Virtual env: `.venv/` (created via `uv venv`)
- Use `.venv/bin/python3` to run scripts

## Testing Strategy
- Each phase has test files (e.g., `tests/test_phase1.py`, `tests/test_tokenizer.py`)
- Tests verify core logic works and prevent regressions
- Run `pytest` before moving to the next phase

## Key Commands
```bash
# Setup
uv venv && uv pip install -e ".[dev]"

# Run tests
.venv/bin/python3 -m pytest tests/ -v

# Run a notebook
.venv/bin/jupyter notebook phase1_foundations/

# Training (on ROG SCAR 17)
.venv/bin/python3 -m phase4_training.train --config phase4_training/configs/small.yaml
```

## File Structure
```
phase1_foundations/   — Jupyter notebooks (numpy only)
phase2_tokenizer/    — BPE tokenizer from scratch
phase3_transformer/  — Modern Llama-style architecture
phase4_training/     — Training pipeline
phase5_generation/   — Decoding strategies + KV-cache
phase6_finetune/     — LoRA + DPO on SmolLM2-360M
phase7_agent/        — ReAct agent with tool use
tests/               — Test files for all phases
```
