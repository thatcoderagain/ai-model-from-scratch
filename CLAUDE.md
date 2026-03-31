# Local LLM — Build an AI Model from Scratch

> **Full detailed plan**: See [PLAN.md](PLAN.md) for architecture details, paper references, and design decisions.

## Project Goal
Build a modern AI language model from scratch for deep understanding, then fine-tune an existing small model into a local agentic coding assistant.

## Hardware
- **MacBook M4** (16GB RAM): Primary coding + local inference via MLX
- **ASUS ROG SCAR 17** (32GB RAM, RTX 3080 16GB VRAM): Heavy training (CUDA, fp16)

## Phase Checklist

| # | Phase | Status | Tests |
|---|---|---|---|
| 1 | **Foundations** (numpy) | DONE | 29 |
| 2 | **Tokenizer** (BPE) | DONE | 36 |
| 3 | **Transformer** (PyTorch) | DONE | 25 |
| 4 | **Training** pipeline | DONE | 18 |
| 5 | **Generation** & eval | DONE | 17 |
| 6 | **Fine-tune** SmolLM2-360M | DONE | 15 |
| 7 | **Agent** (coding assistant) | DONE | 27 |

**ALL PHASES COMPLETE — 167 tests passing**

---

## Quick Reference: Commands That Work

### 1. Setup — MacBook (one time)

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev,mlx]"

# Verify
python3 -c "import torch; print('MPS:', torch.backends.mps.is_available())"
```

### 2. Setup — ROG SCAR 17 (one time)

```bash
uv venv && source .venv/bin/activate

# IMPORTANT: default torch is CPU-only. Install CUDA version:
nvidia-smi                    # check CUDA version (top-right)
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv pip install -e ".[dev]"

# Verify
python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### 3. Train scratch model — ROG

```bash
# Smoke test (~30 seconds, any hardware)
python3 -m phase4_training.train --config phase4_training/configs/tiny.yaml

# Full training (~2h on ROG)
python3 -m phase4_training.train --config phase4_training/configs/small.yaml
```

### 4. Test scratch model — ROG or MacBook

```bash
python3 -m phase5_generation.interactive --checkpoint checkpoints/best.pt
```

### 5. Fine-tune SmolLM2 — ROG

```bash
# Download base model (pick one)
python3 -m phase6_finetune.download_model --model SmolLM2-360M
python3 -m phase6_finetune.download_model --model SmolLM2-1.7B
python3 -m phase6_finetune.download_model --list   # see all options

# Fine-tune with LoRA (~3-5h for 360M, longer for 1.7B)
python3 -m phase6_finetune.finetune --model SmolLM2-360M
python3 -m phase6_finetune.finetune --model SmolLM2-1.7B --batch-size 2 --grad-accum 8

# Quick test (~2 min)
python3 -m phase6_finetune.finetune --model SmolLM2-360M --max-examples 100 --max-steps 50
```

### 6. Deploy to MacBook

Copy `checkpoints/finetune/lora_best.pt` from ROG to MacBook, then:

```bash
# Reconstruct clean model from base + LoRA weights
# --base-model MUST match whatever you used in step 5
python3 -m phase6_finetune.reconstruct_model --lora checkpoints/finetune/lora_best.pt --base-model SmolLM2-360M

# Convert to MLX format
python3 -m phase6_finetune.convert_to_mlx --input checkpoints/finetune/merged_model

# Quantize to 4-bit (693MB → 198MB)
python3 -m phase6_finetune.quantize --input checkpoints/mlx_model --bits 4
```

### 7. Run the coding assistant — MacBook

```bash
# With fine-tuned MLX model (fastest, 102 tok/s)
python3 -m phase7_agent.cli --model checkpoints/mlx_model_q4

# With HuggingFace model (no MLX needed, slower)
python3 -m phase7_agent.cli --hf-model checkpoints/finetune/merged_model

# Demo mode (no model, tests the agent framework)
python3 -m phase7_agent.cli --demo
```

### 8. Run tests

```bash
python3 -m pytest tests/ -v        # all 167 tests
python3 -m pytest tests/test_phase1.py -v   # specific phase
```

### 9. Run notebooks

```bash
jupyter notebook phase1_foundations/
```

---

## Architecture (2025-era, NOT GPT-2)

| Component | What we use | Old approach |
|---|---|---|
| Position encoding | **RoPE** | Learned embeddings |
| Normalization | **RMSNorm** | LayerNorm |
| FFN activation | **SwiGLU** | GELU |
| Attention | **GQA** (6Q/2KV heads) | Standard MHA |
| LR schedule | **WSD** (warmup-stable-decay) | Cosine decay |
| Fine-tuning | **LoRA** (1% params) | Full fine-tuning |
| Alignment | **DPO** | RLHF |

## File Structure

```
phase1_foundations/   — 4 Jupyter notebooks (numpy): NN, backprop, attention, modern components
phase2_tokenizer/    — Byte-level BPE tokenizer from scratch
phase3_transformer/  — Llama-style model: config, rope, rmsnorm, attention, feedforward, block, model
phase4_training/     — Dataset loader, WSD scheduler, trainer (CUDA/MPS), CLI + YAML configs
phase5_generation/   — Top-k/top-p/temperature, KV-cache, perplexity eval, interactive REPL
phase6_finetune/     — SmolLM2 download, LoRA, instruction dataset, DPO, MLX convert, quantize
phase7_agent/        — ReAct+Reflection agent, tools (code/file/shell), function calling, memory, CLI
tests/               — 167 tests across 6 test files
```
