# Pre-Training an LLM from Scratch — Step-by-Step Guide

Build a working language model from absolute zero. No pre-trained weights, no HuggingFace models — everything from raw math to generating coherent text.

This covers **Phases 1-5** of the project.

---

## Table of Contents

1. [Overview: What Pre-Training Means](#1-overview-what-pre-training-means)
2. [Setup](#2-setup)
3. [Phase 1: Learn the Foundations](#3-phase-1-learn-the-foundations)
4. [Phase 2: Build a Tokenizer](#4-phase-2-build-a-tokenizer)
5. [Phase 3: Build the Transformer](#5-phase-3-build-the-transformer)
6. [Phase 4: Train the Model](#6-phase-4-train-the-model)
7. [Phase 5: Generate Text and Evaluate](#7-phase-5-generate-text-and-evaluate)
8. [Architecture Reference](#8-architecture-reference)
9. [Training Configurations](#9-training-configurations)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Overview: What Pre-Training Means

Pre-training is teaching a model to predict the next token from raw text. You start with random weights and end with a model that understands language.

```
Random weights                  → Trained model
"The cat" → "xK#2ñ" (gibberish)   "The cat" → "sat" (coherent)
```

**What happens during pre-training:**
1. Feed the model a sequence: `["The", "cat", "sat", "on"]`
2. At each position, the model predicts the next token
3. Compare predictions to actual next tokens → compute loss (cross-entropy)
4. Backpropagate gradients → update all weights
5. Repeat billions of times across the entire corpus

**What the model learns (emergently, not explicitly programmed):**
- Grammar and syntax
- Word meanings and relationships
- Facts about the world (from the training text)
- Reasoning patterns (from seeing logical text)
- Code syntax (from seeing code)

**Our model specs:**
- Architecture: Llama-style decoder-only transformer
- Parameters: ~15M (tiny by industry standards, but sufficient to demonstrate learning)
- Components: RoPE, RMSNorm, SwiGLU, GQA (2025-era, not GPT-2)
- Training data: TinyStories (~2.5M short stories) or FineWeb-Edu (educational web text)
- Training time: ~2h on RTX 3080, ~4-8h on MacBook M4 MPS

---

## 2. Setup

### MacBook M4

```bash
# Clone the project
git clone <your-repo-url>
cd model

# Create environment and install
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Verify PyTorch + MPS
python3 -c "import torch; print('MPS:', torch.backends.mps.is_available())"
# Should print: MPS: True
```

### ASUS ROG SCAR 17 (or any NVIDIA GPU machine)

```bash
# Clone the project
git clone <your-repo-url>
cd model

# Create environment
uv venv && source .venv/bin/activate

# IMPORTANT: Install CUDA-enabled PyTorch (default is CPU-only!)
nvidia-smi                    # Check CUDA version (top-right of output)
uv pip install torch --index-url https://download.pytorch.org/whl/cu124

# Install remaining deps
uv pip install -e ".[dev]"

# Verify CUDA
python3 -c "import torch; print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
# Should print: CUDA: True
```

### Verify everything works

```bash
python3 -m pytest tests/ -v
# All 167 tests should pass
```

---

## 3. Phase 1: Learn the Foundations

**Goal:** Understand every component before writing PyTorch code.

These are Jupyter notebooks using only numpy — no frameworks. Each includes a terminology glossary explaining every research term.

### Run the notebooks

```bash
jupyter notebook phase1_foundations/
```

### Notebook order

| # | Notebook | What you learn | Key concepts |
|---|---|---|---|
| 1 | `01_neural_network.ipynb` | What a neuron is, how layers work | Weights, bias, activation functions, forward pass |
| 2 | `02_backpropagation.ipynb` | How models learn | Chain rule, gradients, SGD, cross-entropy loss |
| 3 | `03_attention.ipynb` | The core of transformers | Q/K/V, scaled dot-product, causal masking, multi-head |
| 4 | `04_modern_components.ipynb` | 2025-era improvements | RoPE, RMSNorm, SwiGLU, GQA |

### What to look for

**Notebook 01** — Run the XOR network. Watch the loss curve drop. Visualize the decision boundary. Key insight: a single layer can't solve XOR, two layers can. This is why depth matters.

**Notebook 02** — Run gradient checking. It proves your backward pass is mathematically correct by comparing analytical gradients to numerical gradients (finite differences). This is how you debug backprop.

**Notebook 03** — Visualize the attention weight matrix. See how causal masking zeros out the upper triangle. Try changing a future token and verify that past outputs don't change.

**Notebook 04** — The most important notebook. Implement RoPE and see the relative position property. Compare RMSNorm vs LayerNorm output statistics. Understand why SwiGLU's gating mechanism helps.

### Verify

```bash
python3 -m pytest tests/test_phase1.py -v
# 29 tests covering activations, linear layers, backprop, XOR, cross-entropy, RoPE, RMSNorm, SwiGLU
```

---

## 4. Phase 2: Build a Tokenizer

**Goal:** Convert raw text into numbers the model can process.

### How BPE works

```
Starting vocabulary: 256 byte values (a=97, b=98, ... every possible byte)

Training (finding merges):
  Corpus: "the cat the cat the dog"
  Step 1: Most frequent pair is ('t','h') → merge into token 256 = "th"
  Step 2: Most frequent pair is ('th','e') → merge into token 257 = "the"
  Step 3: Most frequent pair is ('c','a') → merge into token 258 = "ca"
  ...continue until vocab reaches target size (4096)

Encoding "the cat":
  "the cat" → bytes → [116, 104, 101, 32, 99, 97, 116]
  Apply merge 256: [256, 101, 32, 99, 97, 116]     (t,h → th)
  Apply merge 257: [257, 32, 99, 97, 116]           (th,e → the)
  Apply merge 258: [257, 32, 258, 116]              (c,a → ca)
  ...final token IDs
```

### Train a tokenizer

```bash
# Train on sample English + Python corpus (vocab_size=4096)
python3 -m phase2_tokenizer.train_tokenizer --vocab-size 4096

# Output:
#   Saved tokenizer to phase2_tokenizer/vocab/
#   Compression ratio: ~3-4x for English text
```

### Test it interactively

```python
from phase2_tokenizer.bpe_tokenizer import BPETokenizer

tok = BPETokenizer.load("phase2_tokenizer/vocab")

# Encode
ids = tok.encode("Hello, world!")
print(ids)  # e.g., [72, 265, 352, 44, 285, 267]

# Decode
text = tok.decode(ids)
print(text)  # "Hello, world!"

# Roundtrip — must always be exact
assert tok.decode(tok.encode("Any text 🚀")) == "Any text 🚀"
```

### Key files

| File | Purpose |
|---|---|
| `phase2_tokenizer/bpe_tokenizer.py` | Full BPE implementation: train, encode, decode, save, load |
| `phase2_tokenizer/train_tokenizer.py` | Training script with sample corpus |
| `phase2_tokenizer/vocab/` | Saved vocabulary (vocab.json + merges.txt) |

### Verify

```bash
python3 -m pytest tests/test_tokenizer.py -v
# 36 tests covering roundtrip, unicode, emoji, code, special tokens, save/load
```

---

## 5. Phase 3: Build the Transformer

**Goal:** Implement the complete model architecture in PyTorch.

### The architecture (every file is one component)

```
Input token IDs
    │
    ▼
┌─────────────────────┐
│ Token Embedding      │  token_id → vector (n_embd dimensions)
│ (no position embed   │  RoPE handles position inside attention
│  — unlike GPT-2)     │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Transformer Block    │  ×6 (n_layer times)
│                      │
│  ┌─ RMSNorm ────┐   │  normalize before attention
│  ├─ GQA Attn ───┤   │  6 Q heads, 2 KV heads, with RoPE
│  ├─ + residual ─┤   │  x = x + attention(norm(x))
│  ├─ RMSNorm ────┤   │  normalize before FFN
│  ├─ SwiGLU FFN ─┤   │  gated feed-forward network
│  └─ + residual ─┘   │  x = x + ffn(norm(x))
│                      │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Final RMSNorm        │
├─────────────────────┤
│ LM Head              │  project to vocab_size → logits
│ (weight-tied with    │  shares weights with token embedding
│  token embedding)    │
└─────────────────────┘
          │
          ▼
    Logits (vocab_size predictions per position)
```

### Key files and what they implement

| File | Component | Paper |
|---|---|---|
| `config.py` | Model hyperparameters dataclass | — |
| `rope.py` | Rotary Position Embeddings | Su et al. 2021 |
| `rmsnorm.py` | RMS Normalization | Zhang & Sennrich 2019 |
| `attention.py` | Grouped Query Attention + RoPE + causal mask | Ainslie et al. 2023 |
| `feedforward.py` | SwiGLU FFN (3 projections, gated) | Shazeer 2020 |
| `block.py` | One transformer block (pre-norm + residuals) | — |
| `model.py` | Full model assembly + weight tying + generate() | — |

### Inspect the model

```python
from phase3_transformer.config import SMALL_CONFIG
from phase3_transformer.model import GPT

model = GPT(SMALL_CONFIG)
print(f"Parameters: {model.count_parameters():,}")
# ~15,000,000 parameters

# Print architecture
print(model)

# Test forward pass
import torch
idx = torch.randint(0, 4096, (2, 128))  # batch=2, seq_len=128
logits, loss = model(idx)
print(f"Output shape: {logits.shape}")  # (2, 128, 4096)
```

### Model configurations

| Config | Params | Layers | Heads | d_model | Use |
|---|---|---|---|---|---|
| `TINY_CONFIG` | ~100K | 2 | 2Q/1KV | 64 | Smoke tests (seconds) |
| `SMALL_CONFIG` | ~15M | 6 | 6Q/2KV | 384 | Real training (~2h on GPU) |

### Verify

```bash
python3 -m pytest tests/test_transformer.py -v
# 25 tests covering config, RoPE, RMSNorm, attention, FFN, block,
# full model, causal masking, weight tying, generation, backward pass
```

---

## 6. Phase 4: Train the Model

**Goal:** Take the randomly initialized model and train it to generate coherent text.

### Step 1: Understand what happens during training

```
Epoch 0:    Loss 8.3 (random — ln(4096) ≈ 8.3, guessing uniformly)
            Output: "xK#2ñ ¿¿ zW7 ..."

Epoch 1000: Loss 5.0 (learning common words)
            Output: "the the and was a little ..."

Epoch 5000: Loss 3.5 (learning grammar)
            Output: "Once there was a boy who liked to ..."

Epoch 10000: Loss 2.5 (learning narrative structure)
             Output: "Once upon a time, there was a little girl named Lily.
                      She loved to play in the park with her friends."
```

### Step 2: Run a smoke test (any hardware, ~30 seconds)

```bash
python3 -m phase4_training.train --config phase4_training/configs/tiny.yaml
```

This trains the tiny 100K model for 100 steps. Verifies:
- Dataset downloads and tokenizes correctly
- Training loop runs without errors
- Loss decreases
- Checkpoints save
- Generation produces output (will be gibberish — model is too small)

### Step 3: Run full training

**On RTX 3080/3090 (recommended):**
```bash
python3 -m phase4_training.train --config phase4_training/configs/small.yaml
```

Expected output:
```
Training on cuda | fp16
Steps: 10000 | Micro-batch: 16 | Grad accum: 4 | Effective batch: 64
Block size: 256 | Max LR: 0.0003
Params: 15,000,000
------------------------------------------------------------
Step    50/10000 | loss 7.8234 | lr 3.00e-04 | 85000 tok/s
Step   100/10000 | loss 6.4521 | lr 3.00e-04 | 92000 tok/s
...
Step 10000/10000 | loss 2.5123 | lr 3.00e-05 | 91000 tok/s
------------------------------------------------------------
Training complete: 10000 steps in 6842s (114min)
Best val loss: 2.4891
```

**On MacBook M4 MPS (slower, for debugging or if no GPU):**

Edit `phase4_training/configs/small.yaml` first:
```yaml
training:
  micro_batch_size: 4          # reduced from 16 (less VRAM)
  grad_accum_steps: 16         # increased to keep effective batch = 64
```

Then:
```bash
python3 -m phase4_training.train --config phase4_training/configs/small.yaml
```

Expect ~4-8 hours. Trains in fp32 (MPS doesn't support fp16 reliably).

### Step 4: Understand the training config

```yaml
# phase4_training/configs/small.yaml

model:
  vocab_size: 4096        # must match tokenizer
  n_layer: 6              # transformer depth
  n_head: 6               # query attention heads
  n_kv_head: 2            # key/value heads (GQA)
  n_embd: 384             # embedding dimension
  block_size: 256         # context window (max tokens the model sees at once)
  ffn_hidden: 1024        # SwiGLU intermediate size
  dropout: 0.1            # regularization (randomly zero out 10% of activations)

training:
  max_steps: 10000        # total training steps
  micro_batch_size: 16    # samples per forward pass (reduce if OOM)
  grad_accum_steps: 4     # accumulate gradients over 4 micro-batches
                          # effective batch = 16 × 4 = 64

  # Learning rate (WSD schedule)
  max_lr: 3.0e-4          # peak learning rate (during stable phase)
  min_lr: 3.0e-5          # end learning rate (after decay)
  warmup_fraction: 0.05   # first 5% of steps: LR ramps 0 → max_lr
  decay_fraction: 0.20    # last 20% of steps: LR decays max_lr → min_lr
                          # middle 75%: constant at max_lr

  weight_decay: 0.1       # penalize large weights (regularization)
  grad_clip: 1.0          # cap gradient magnitude (prevent explosions)

  # Logging
  log_interval: 50        # print loss every 50 steps
  eval_interval: 500      # run validation every 500 steps
  checkpoint_interval: 1000  # save checkpoint every 1000 steps

  # Data
  data_source: "tinystories"  # or "fineweb-edu" or path to .txt file
  max_chars: 10000000         # 10MB of text (set to null for full dataset)
```

### What each component does during training

| Component | Role |
|---|---|
| **Dataset** | Loads text → tokenizes → creates (input, target) chunks of `block_size` |
| **DataLoader** | Shuffles and batches the chunks |
| **Forward pass** | `model(input) → logits` then `cross_entropy(logits, target) → loss` |
| **Backward pass** | `loss.backward()` computes gradients for all parameters |
| **Gradient accumulation** | Accumulate gradients over N micro-batches before stepping |
| **Gradient clipping** | `clip_grad_norm_(params, 1.0)` prevents exploding gradients |
| **Optimizer** | `AdamW` updates weights using gradients + momentum + weight decay |
| **LR scheduler** | WSD: warmup → stable → decay. Controls step size over training. |
| **Mixed precision** | Forward/backward in fp16, optimizer in fp32. 2x faster on CUDA. |
| **Checkpointing** | Save model + optimizer state periodically. Resume after crash. |
| **Validation** | Compute loss on held-out data to detect overfitting. |

### Verify

```bash
python3 -m pytest tests/test_training.py -v
# 18 tests covering dataset, WSD schedule, trainer, loss decrease, checkpointing
```

---

## 7. Phase 5: Generate Text and Evaluate

**Goal:** Use the trained model to generate text and measure quality.

### Interactive generation

```bash
python3 -m phase5_generation.interactive --checkpoint checkpoints/best.pt
```

This starts a REPL where you type prompts and see completions:

```
> Once upon a time
Once upon a time, there was a little girl named Lily. She loved to play
in the garden. One day, she found a small box...

> The dog
The dog was very happy. He liked to run in the park with his friends.
One sunny day, the dog saw a big red ball...
```

### REPL commands

| Command | What it does |
|---|---|
| `:temp 0.5` | Lower temperature = more deterministic |
| `:temp 1.2` | Higher temperature = more creative/random |
| `:topk 20` | Only consider top 20 tokens |
| `:topp 0.95` | Nucleus sampling threshold |
| `:tokens 200` | Generate up to 200 tokens |
| `:settings` | Show current settings |
| `:quit` | Exit |

### Generation from a script

```python
import torch
from phase3_transformer.model import GPT
from phase3_transformer.config import SMALL_CONFIG
from phase2_tokenizer.bpe_tokenizer import BPETokenizer
from phase5_generation.generate import generate

# Load model
ckpt = torch.load("checkpoints/best.pt", map_location="cpu", weights_only=False)
config = SMALL_CONFIG
model = GPT(config)
model.load_state_dict(ckpt["model_state"])
model.eval()

# Load tokenizer
tokenizer = BPETokenizer.load("phase2_tokenizer/vocab")

# Generate
prompt = "Once upon a time"
prompt_ids = torch.tensor([tokenizer.encode(prompt)])

output_ids = generate(
    model, prompt_ids,
    max_new_tokens=100,
    temperature=0.8,    # 0 = greedy, 0.7-0.9 = balanced, >1 = creative
    top_k=40,           # only consider top 40 tokens
    top_p=0.9,          # nucleus sampling
    repetition_penalty=1.1,  # penalize repeated tokens
)

print(tokenizer.decode(output_ids[0].tolist()))
```

### Measure perplexity

```python
from phase5_generation.evaluate import compute_perplexity, evaluate_model
from phase4_training.dataset import TextDataset
import numpy as np

# Load tokens
tokens = np.load("data/tinystories_tokens.npy")
val_tokens = tokens[int(len(tokens) * 0.95):]  # last 5%
val_ds = TextDataset(val_tokens, block_size=256)

# Compute perplexity
ppl, loss = compute_perplexity(model, val_ds, max_batches=100)
print(f"Perplexity: {ppl:.2f}")
print(f"Loss: {loss:.4f}")

# Expected for well-trained 15M model on TinyStories:
#   Perplexity: 20-50
#   Loss: 3.0-3.9
```

### Understanding generation parameters

**Temperature** controls randomness:
```
temp=0.0:  "The cat sat on the mat. The cat sat on the mat."  (greedy, repetitive)
temp=0.7:  "The cat sat on the warm rug near the fireplace."  (balanced)
temp=1.0:  "The cat discovered a hidden portal behind the bookshelf."  (creative)
temp=1.5:  "The cat quantum-entangled with a philosophical cheese dream."  (chaotic)
```

**Top-k** filters out unlikely tokens:
```
top_k=1:   Greedy (same as temp=0)
top_k=10:  Very focused (only 10 options per step)
top_k=40:  Balanced (default)
top_k=100: More diverse
```

**Top-p (nucleus sampling)** adapts to confidence:
```
When model is confident (90% on one token): considers ~1-2 tokens
When model is uncertain (5% each on many): considers ~20+ tokens
top_p=0.9 means: keep tokens until cumulative probability reaches 90%
```

### Key files

| File | Purpose |
|---|---|
| `generate.py` | All decoding strategies: temperature, top-k, top-p, repetition penalty |
| `kv_cache.py` | KV-cache data structure for fast inference |
| `evaluate.py` | Perplexity computation on datasets |
| `interactive.py` | REPL with adjustable settings |

### Verify

```bash
python3 -m pytest tests/test_generation.py -v
# 17 tests covering greedy, temperature, top-k, top-p, repetition penalty,
# KV-cache, perplexity computation
```

---

## 8. Architecture Reference

### Why these specific components?

| Component | Old (GPT-2, 2019) | Ours (Llama-style, 2025) | Why we chose it |
|---|---|---|---|
| Position | Learned absolute embeddings | **RoPE** | Relative position, extrapolates to longer sequences |
| Normalization | LayerNorm (mean + variance) | **RMSNorm** (RMS only) | 7-64% faster, equally effective |
| FFN activation | GELU | **SwiGLU** (gated) | 1-2% better quality, gating filters information |
| Attention | MHA (6Q, 6KV) | **GQA** (6Q, 2KV) | 3x smaller KV-cache, same quality |
| Biases | Yes | **No** | Removing biases doesn't hurt, saves parameters |
| LR schedule | Cosine decay | **WSD** | Better final loss, simpler to tune |

### Parameter budget breakdown (SMALL_CONFIG, ~15M)

```
Token embedding:      4096 × 384          =  1,572,864  (10.5%)
                                              (shared with lm_head via weight tying)

Per transformer block (×6):
  RMSNorm ×2:         384 × 2             =        768
  Q projection:       384 × 384           =    147,456
  K projection:       384 × 128           =     49,152  (only 2 KV heads)
  V projection:       384 × 128           =     49,152
  O projection:       384 × 384           =    147,456
  SwiGLU W_gate:      384 × 1024          =    393,216
  SwiGLU W_up:        384 × 1024          =    393,216
  SwiGLU W_down:      1024 × 384          =    393,216
  Block total:                             =  1,573,632

6 blocks total:       1,573,632 × 6       =  9,441,792  (63%)
Final RMSNorm:        384                  =        384
LM head:              (tied with embedding, 0 extra)

TOTAL:                                     ≈ 11,015,040
```

---

## 9. Training Configurations

### Quick experiments (any hardware)

```yaml
# phase4_training/configs/tiny.yaml
# ~100K params, trains in 30 seconds
model:
  n_layer: 2
  n_head: 2
  n_kv_head: 1
  n_embd: 64
  block_size: 128
training:
  max_steps: 100
  micro_batch_size: 4
  max_chars: 500000    # 500KB
```

```bash
python3 -m phase4_training.train --config phase4_training/configs/tiny.yaml
```

### Real training (GPU recommended)

```yaml
# phase4_training/configs/small.yaml
# ~15M params, trains in ~2h on RTX 3080
model:
  n_layer: 6
  n_head: 6
  n_kv_head: 2
  n_embd: 384
  block_size: 256
training:
  max_steps: 10000
  micro_batch_size: 16     # reduce to 4-8 on MacBook
  grad_accum_steps: 4      # increase to 8-16 on MacBook
  max_chars: 10000000      # 10MB
```

```bash
python3 -m phase4_training.train --config phase4_training/configs/small.yaml
```

### Custom training on your own text

```bash
# Put your text in a file
echo "your training text..." > data/my_corpus.txt

# Edit the config (or create a new YAML):
#   data_source: "data/my_corpus.txt"

# Or modify small.yaml temporarily
python3 -m phase4_training.train --config phase4_training/configs/small.yaml
```

### Scaling up (if you have more compute)

For a ~50M param model (needs 24GB+ VRAM):

```yaml
model:
  n_layer: 12
  n_head: 12
  n_kv_head: 4
  n_embd: 768
  block_size: 512
  ffn_hidden: 2048
training:
  max_steps: 20000
  micro_batch_size: 8
  grad_accum_steps: 8
  max_lr: 2.0e-4           # slightly lower for larger model
```

---

## 10. Troubleshooting

### "Training on cpu | fp32" (expected CUDA)

PyTorch was installed without CUDA support:
```bash
uv pip uninstall torch
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### Out of memory (OOM)

Reduce `micro_batch_size` in the YAML config:
```yaml
micro_batch_size: 4         # was 16
grad_accum_steps: 16        # was 4 (keep effective batch = 64)
```

### Loss not decreasing

- Check learning rate isn't too high (try `max_lr: 1.0e-4`)
- Check data loaded correctly (should see "Train: X tokens")
- Make sure `block_size` isn't larger than most of your training examples

### Loss is NaN

- Gradient explosion: reduce `max_lr` or ensure `grad_clip: 1.0`
- Data issue: check for corrupt tokens in the dataset

### Tokenizer generates bad output

- Retrain tokenizer on more data: `python3 -m phase2_tokenizer.train_tokenizer --vocab-size 4096`
- The sample corpus may not cover your target domain well enough

### Model generates repetitive text

- Increase `temperature` (try 0.9-1.0)
- Enable `repetition_penalty` (try 1.1-1.2)
- Increase `top_k` or `top_p`
- Model may need more training steps

### Resume training from checkpoint

```bash
python3 -m phase4_training.train \
  --config phase4_training/configs/small.yaml \
  --resume checkpoints/step_005000.pt
```

### Training on a different dataset

Change `data_source` in the YAML:
```yaml
data_source: "tinystories"    # HuggingFace TinyStories
data_source: "fineweb-edu"    # HuggingFace FineWeb-Edu (higher quality, larger)
data_source: "data/my_file.txt"  # any local text file
```

---

## Summary: The Full Pre-Training Pipeline

```bash
# 1. Learn the math (Phase 1)
jupyter notebook phase1_foundations/

# 2. Build a tokenizer (Phase 2)
python3 -m phase2_tokenizer.train_tokenizer --vocab-size 4096

# 3. Verify the model architecture (Phase 3)
python3 -m pytest tests/test_transformer.py -v

# 4. Train the model (Phase 4)
python3 -m phase4_training.train --config phase4_training/configs/small.yaml

# 5. Talk to your model (Phase 5)
python3 -m phase5_generation.interactive --checkpoint checkpoints/best.pt

# 6. Measure quality
python3 -c "
from phase5_generation.evaluate import compute_perplexity
from phase4_training.dataset import TextDataset
from phase3_transformer.model import GPT
from phase3_transformer.config import SMALL_CONFIG
import torch, numpy as np
ckpt = torch.load('checkpoints/best.pt', map_location='cpu', weights_only=False)
model = GPT(SMALL_CONFIG); model.load_state_dict(ckpt['model_state']); model.eval()
tokens = np.load('data/tinystories_tokens.npy')
val_ds = TextDataset(tokens[int(len(tokens)*0.95):], 256)
ppl, loss = compute_perplexity(model, val_ds, max_batches=100)
print(f'Perplexity: {ppl:.1f}, Loss: {loss:.4f}')
"
```

**After this, you have a language model trained from absolute zero.** No pre-trained weights, no HuggingFace models — just math, data, and compute.

To turn it into a coding assistant, see [CLAUDE.md](CLAUDE.md) steps 5-7 (fine-tuning + agent).
