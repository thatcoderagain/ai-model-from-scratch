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

## Project Phases
1. **Foundations** (numpy only) — neural nets, backprop, attention, modern components
2. **Tokenizer** — byte-level BPE from scratch
3. **Transformer** — Llama-style 15M param model in PyTorch
4. **Training** — WSD schedule, AdamW, FineWeb-Edu dataset, train on 3090
5. **Generation** — KV-cache, top-k/top-p, interactive REPL
6. **Fine-tune** — LoRA on SmolLM2-360M, DPO, convert to MLX + quantize
7. **Agent** — ReAct + Reflection, function calling, sandboxed code exec, CLI

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
