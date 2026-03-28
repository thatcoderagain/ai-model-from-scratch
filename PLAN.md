# Build an AI Model from Scratch → Agentic Coding Assistant

## Context

**Why**: Understand how modern AI language models work at every level — from raw math to a working local coding assistant. Not to replicate Claude Opus 4.6 (that requires billions in compute), but to build every component by hand, deeply understand the technology, then leverage existing open-source models to create something practical.

**Hardware (dual-machine setup)**:
- **MacBook M4** (16GB RAM): Primary coding, development, local inference via MLX
- **ASUS ROG SCAR 17** (32GB RAM, RTX 3090 16GB VRAM): Heavy training — CUDA, fp16, Flash Attention 2

**Approach**: Build from scratch for understanding (Phase 1-5), then fine-tune an existing small model for a capable agent (Phase 6-7)
**Framework**: PyTorch for learning → CUDA training on ROG → MLX for fast MacBook inference
**Format**: Jupyter notebooks for foundations, Python scripts for model code

### Dual-Machine Workflow
- **Code on MacBook** → push to git → **pull on ROG → train on 3090** → push checkpoint → **pull on MacBook → convert to MLX → run locally**
- Quick experiments/debugging: run on MacBook MPS
- Serious training: SSH into ROG or use shared git repo
- The 3090 unlocks: **fp16 mixed precision** (2x speed, half memory), **Flash Attention 2** (faster attention, less memory), **larger batch sizes**, and the ability to **fine-tune bigger models** (SmolLM2-360M or Gemma 3-1B instead of just 135M)

---

## What's Modern (2025-2026) vs Outdated

We build with the **Llama/Gemma-era architecture**, not GPT-2 (2019):

| Component | Outdated (GPT-2) | Modern (2025) | Why it matters |
|---|---|---|---|
| Position encoding | Learned absolute embeddings | **RoPE** (Rotary Position Embeddings) | Enables context length extension, better generalization |
| Normalization | LayerNorm | **RMSNorm** | 7-64% faster, equally stable |
| FFN activation | GELU | **SwiGLU** | 1-2% better performance, standard in all frontier models |
| Attention | Multi-Head (MHA) | **Grouped Query Attention (GQA)** | Faster inference, less memory, near-MHA quality |
| LR schedule | Cosine decay | **WSD** (Warmup-Stable-Decay) | Better final loss, standard for LLM training |
| Tokenizer | Basic BPE | **Byte-level BPE with byte-fallback** | Handles any text/code, no unknown tokens |
| Alignment | RLHF | **DPO** (Direct Preference Optimization) | Simpler, no reward model needed, works at small scale |
| Agent pattern | Basic ReAct | **ReAct + Reflection + Function Calling** | More reliable tool use, self-correction |

---

## Project Structure

```
/Users/gaurav/Desktop/Personal/claude/model/
├── pyproject.toml
├── .env                           # MPS/MLX environment vars
│
├── phase1_foundations/            # Jupyter notebooks — pure numpy
│   ├── 01_neural_network.ipynb
│   ├── 02_backpropagation.ipynb
│   ├── 03_attention.ipynb
│   └── 04_modern_components.ipynb  # RoPE, RMSNorm, SwiGLU from scratch
│
├── phase2_tokenizer/             # Python scripts
│   ├── bpe_tokenizer.py          # BPE from scratch with byte-fallback
│   ├── train_tokenizer.py
│   ├── test_tokenizer.py
│   └── vocab/
│
├── phase3_transformer/           # Python scripts — modern Llama-style arch
│   ├── config.py                 # Model config dataclass
│   ├── rope.py                   # Rotary Position Embeddings
│   ├── rmsnorm.py                # RMS Normalization
│   ├── attention.py              # Grouped Query Attention with RoPE
│   ├── feedforward.py            # SwiGLU FFN
│   ├── block.py                  # Transformer decoder block
│   ├── model.py                  # Full model assembly
│   └── test_model.py
│
├── phase4_training/              # Python scripts
│   ├── dataset.py                # Data loading + preparation
│   ├── trainer.py                # Training loop (MPS accelerated)
│   ├── lr_schedule.py            # WSD scheduler
│   ├── train.py                  # CLI entry point
│   └── configs/
│       ├── tiny.yaml             # ~1M params for quick smoke tests
│       └── small.yaml            # ~15M params target
│
├── phase5_generation/            # Python scripts
│   ├── generate.py               # Decoding strategies (top-k, top-p, temp)
│   ├── kv_cache.py               # KV-cache for fast inference
│   ├── evaluate.py               # Perplexity measurement
│   └── interactive.py            # REPL for testing
│
├── phase6_finetune/              # Fine-tune SmolLM2-135M or Gemma-3-270M
│   ├── download_model.py         # Download base model from HuggingFace
│   ├── lora.py                   # LoRA implementation from scratch
│   ├── instruction_dataset.py    # Instruction/code dataset prep
│   ├── finetune.py               # Fine-tuning script (LoRA on code data)
│   ├── dpo.py                    # DPO alignment (optional)
│   ├── convert_to_mlx.py         # Convert to MLX format
│   └── quantize.py               # Quantize for fast inference
│
├── phase7_agent/                 # Agentic coding assistant
│   ├── tools/
│   │   ├── base.py               # Tool interface
│   │   ├── code_executor.py      # Sandboxed Python execution
│   │   ├── file_ops.py           # Read/write files
│   │   └── shell.py              # Shell command execution
│   ├── agent.py                  # ReAct + Reflection loop
│   ├── function_calling.py       # Structured function call parser
│   ├── memory.py                 # Conversation context management
│   ├── inference.py              # MLX inference engine
│   └── cli.py                    # Terminal interface
│
├── checkpoints/
├── logs/
└── data/
```

---

## Phase 1: Foundations (Numpy Only) — 4-5 days

**Goal**: Understand neural network math from first principles before touching any framework.

### `01_neural_network.ipynb`
- Neurons, weights, biases as matrix multiplication
- Activation functions: ReLU, sigmoid, tanh — implement and visualize each
- Build a 2-layer network that learns XOR
- Visualize decision boundaries

### `02_backpropagation.ipynb`
- Chain rule of calculus with computational graphs
- Implement `.backward()` for each layer type
- SGD optimizer: `param -= lr * grad`
- Numerical gradient checking (finite differences)
- Train XOR network with manual backprop, visualize loss curve
- **Key insight**: This is the same algorithm PyTorch autograd does automatically

### `03_attention.ipynb`
- Why attention? Limitations of fixed-size representations
- Scaled dot-product attention: `softmax(QK^T / √d_k) · V`
- Causal masking — why LMs prevent attending to future tokens
- Multi-head attention: split, attend, concatenate, project
- Visualize attention weight matrices on sample sentences

### `04_modern_components.ipynb` ← **NEW, not in old tutorials**
- **RoPE** (Rotary Position Embeddings): encode position by rotating Q/K vectors in 2D pairs. Implement the rotation matrix, show why it captures relative position
- **RMSNorm**: `x / RMS(x) * γ` — simpler than LayerNorm, no mean subtraction. Implement and compare speed
- **SwiGLU**: `SwiGLU(x) = Swish(xW₁) ⊙ (xW₂)` — gated activation. Show why gating helps vs plain GELU
- **GQA concept**: fewer K/V heads than Q heads, diagram the grouping

---

## Phase 2: Tokenizer — 2-3 days

**Goal**: Build a byte-level BPE tokenizer from scratch.

### `bpe_tokenizer.py`
```
class BPETokenizer:
    train(text) → learn merges from corpus
    encode(text) → list[int]
    decode(ids) → str
    save/load vocabulary
```

**Modern approach** (byte-level with byte-fallback):
- Base vocabulary: 256 byte values (handles ANY input — no unknown tokens)
- Pre-tokenization regex (GPT-4 style): split on whitespace, punctuation, code tokens
- BPE merge loop: find most frequent adjacent pair → merge → repeat until `vocab_size`
- Special tokens: `<|begin_of_text|>`, `<|end_of_text|>`, `<|start_code|>`, `<|end_code|>`
- **Vocab size**: 4096 (educational) — real models use 32K-128K

### `train_tokenizer.py`
- Train on mixed corpus: some English text + Python code samples
- This ensures the tokenizer handles both natural language and code

### `test_tokenizer.py`
- Roundtrip tests: `decode(encode(text)) == text`
- Unicode, emoji, multilingual edge cases
- Code tokenization quality check
- Compression ratio measurement (~3-4x over raw bytes for English)

---

## Phase 3: Modern Transformer Architecture (PyTorch) — 5-7 days

**Goal**: Build a complete Llama-style decoder-only transformer. Every component as its own module.

### Target Architecture (~15M parameters)

```
vocab_size:  4096
n_layer:     6
n_head:      6         # query heads
n_kv_head:   2         # key/value heads (GQA: 3 query heads per KV head)
n_embd:      384
block_size:  256       # max context length
ffn_hidden:  1024      # SwiGLU intermediate size (≈ 2.67x n_embd)
dropout:     0.1
```

### `config.py`
- `@dataclass ModelConfig` with all hyperparameters
- Method to compute parameter count

### `rope.py` — Rotary Position Embeddings
- Precompute frequency table: `θᵢ = 10000^(-2i/d)` for each dimension pair
- Implement rotation: split embedding into pairs, rotate each by `position * θᵢ`
- Apply to Q and K (not V) inside attention
- **Key insight**: RoPE makes attention scores depend on *relative* position naturally

### `rmsnorm.py` — RMS Normalization
```python
class RMSNorm(nn.Module):
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight
```

### `attention.py` — Grouped Query Attention with RoPE
- Combined Q/K/V projection (but K/V have fewer heads than Q)
- Apply RoPE to Q and K after projection
- Causal mask (lower-triangular)
- Attention: `softmax(QK^T / √d_head) · V` with mask
- Output projection

**GQA detail**: With 6 query heads and 2 KV heads, every 3 query heads share 1 KV head. This saves memory and speeds up inference while maintaining quality.

### `feedforward.py` — SwiGLU
```python
class SwiGLU(nn.Module):
    # gate = swish(x @ W_gate)
    # up = x @ W_up
    # output = (gate * up) @ W_down
    # Three weight matrices instead of two, but intermediate size is 2/3 of 4x
```

### `block.py` — Transformer Block (Pre-norm)
```
x = x + attention(rmsnorm(x))
x = x + feedforward(rmsnorm(x))
```

### `model.py` — Full Model
- Token embeddings (no separate positional embedding — RoPE handles position)
- Stack of N transformer blocks
- Final RMSNorm
- Language model head (weight-tied with token embeddings)
- `forward(idx, targets=None) → (logits, loss)`

### `test_model.py`
- Verify output shape: `(batch, seq_len, vocab_size)`
- Verify parameter count (~15M)
- Verify causal masking: changing future tokens doesn't affect past outputs
- Test on MPS device
- Memory usage profiling

---

## Phase 4: Training Pipeline — 5-7 days + training time

**Goal**: Train the model to generate coherent text.

### Dataset: **FineWeb-Edu** (subset) ← modern choice
- High-quality educational web text curated by HuggingFace
- Better than TinyStories for learning real language patterns
- Download a ~500MB subset via HuggingFace `datasets` library
- Alternative: TinyStories for faster iteration (stories are simpler to learn)

### `dataset.py`
- Load text, tokenize with our BPE tokenizer
- Sliding window: input = tokens[i:i+256], target = tokens[i+1:i+257]
- Train/val split (95/5)
- Memory-mapped for efficiency

### `trainer.py`
- **Multi-device support**: auto-detect CUDA (ROG 3090) vs MPS (MacBook M4) vs CPU
- **On 3090 (CUDA)**: fp16 mixed precision via `torch.amp.autocast` + `GradScaler` — 2x faster, half memory
- **On 3090 (CUDA)**: Flash Attention 2 via `torch.nn.functional.scaled_dot_product_attention` with `attn_implementation="flash_attention_2"`
- **On M4 (MPS)**: Float32 training (MPS doesn't reliably support fp16)
- Gradient accumulation (effective batch = 64: micro-batch 16 on 3090, 8 on M4)
- Gradient clipping (max_norm = 1.0)
- Logging: loss, learning rate, tokens/sec
- Checkpoint saving every 1000 steps (device-agnostic: save to CPU tensors)
- Validation loss every 500 steps

### `lr_schedule.py` — **WSD Schedule** (modern, replaces cosine)
```
Warmup:  steps 0-500     (linear ramp 0 → 3e-4)
Stable:  steps 500-8000  (constant at 3e-4)
Decay:   steps 8000-10000 (linear decay 3e-4 → 3e-5)
```
**Why WSD > cosine**: Stable plateau allows maximum exploration; sharp decay at the end gives better convergence. This is the standard in 2025 LLM training.

### `train.py` — CLI entry point
```
Optimizer:     AdamW (β1=0.9, β2=0.95, weight_decay=0.1)
Learning rate: 3e-4 peak
Batch size:    8 (× 8 gradient accumulation = 64 effective)
Max steps:     10,000
Block size:    256
```

### Training expectations
- **On RTX 3090**: ~1.5-3 hours (fp16, Flash Attention 2, larger batches) ← **recommended**
- **On M4 MPS**: ~4-8 hours (fp32, slower but works for debugging)
- **Loss**: from ~8.3 (random, ln(4096)) → ~2.5-3.5
- **Output**: Coherent English paragraphs, not perfect but clearly learned language

### Environment: `.env`
```bash
# MacBook M4
PYTORCH_ENABLE_MPS_FALLBACK=1
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# ROG SCAR 17 (3090)
CUDA_VISIBLE_DEVICES=0
```

### `pyproject.toml`
```toml
[project]
name = "local-llm"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.2",
    "numpy>=1.26",
    "datasets>=2.16",
    "tqdm",
    "pyyaml",
    "matplotlib",
    "mlx>=0.18",
    "mlx-lm>=0.20",
    "huggingface-hub",
    "transformers",
]

[project.optional-dependencies]
dev = ["pytest", "jupyter", "ipykernel"]
```

---

## Phase 5: Generation & Evaluation — 2-3 days

**Goal**: Make the trained model generate text with modern decoding strategies.

### `generate.py`
- **Temperature scaling**: logits / temperature
- **Top-k sampling**: keep only k most likely tokens
- **Top-p (nucleus) sampling**: keep smallest set of tokens with cumulative prob ≥ p
- **Repetition penalty**: reduce probability of recently generated tokens
- Autoregressive loop with early stopping on `<|end_of_text|>`

### `kv_cache.py` ← **Modern optimization**
- Cache K/V tensors from previous positions
- Only compute attention for the new token, reuse cached K/V
- **Speedup**: O(1) per new token instead of O(n) recomputation
- This is how all production LLMs do inference

### `evaluate.py`
- Perplexity: `exp(avg_cross_entropy_loss)` on held-out validation set
- Target: perplexity ~20-50 for a well-trained 15M model

### `interactive.py`
- REPL: type a prompt, see the model's continuation
- Adjustable temperature, top-k, top-p from the command line

---

## Phase 6: Fine-tune a Real Model for Code — 7-10 days

**Goal**: Switch from our scratch-built 15M model to a pre-trained open-source model. Fine-tune it on code with LoRA, align with DPO, then convert to MLX for fast MacBook inference.

**Why switch**: Our 15M model proves we understand the architecture, but 15M params can't meaningfully write code. Pre-trained models already understand language/code from trillions of tokens — we add our skills on top.

**Model choice** (thanks to the RTX 3090):
| Model | Params | VRAM needed (LoRA fp16) | Quality |
|---|---|---|---|
| SmolLM2-135M | 135M | ~2GB | Decent for simple code |
| **SmolLM2-360M** | 360M | ~4GB | **Best bang for buck on 3090** ← recommended |
| Gemma 3 1B | 1B | ~8GB | Good quality, tight fit on 3090 |

With the 3090's 16GB VRAM + fp16, we can comfortably fine-tune **SmolLM2-360M** (or even try Gemma 3 1B). On MacBook alone, we'd be limited to 135M.

### `download_model.py`
- Download SmolLM2-360M from HuggingFace
- Verify architecture matches what we built (it uses the same components: RoPE, RMSNorm, SwiGLU, GQA)

### `lora.py` — LoRA from scratch ← **Educational, build it yourself**
```python
class LoRALinear(nn.Module):
    # Freeze original weights W
    # Add low-rank adapters: W + (A @ B) where A is (d, r), B is (r, d)
    # Only train A and B (rank r=32-64, per 2025 research)
    # This trains ~1-2% of total parameters
```
**Why LoRA**: Even with 16GB VRAM on the 3090, full fine-tuning of 360M is tight. LoRA trains only ~3-5M parameters while adapting the full model. Also educational — LoRA is how most real fine-tuning is done in 2025.

### `instruction_dataset.py`
- Format: `<|instruction|>Write a function...<|response|>def func():...<|end_of_text|>`
- Loss masking: only compute loss on response tokens
- **Dataset**: subset of `iamtarun/python_code_instructions_18k_alpaca` (~10K examples)
- Plus hand-curated simple coding problems (~50 for evaluation)

### `finetune.py`
- **Train on RTX 3090** with fp16 mixed precision + Flash Attention 2
- Apply LoRA to attention Q/V projections (standard practice)
- Train for 2000-3000 steps, lr=2e-5
- WSD schedule with short warmup (100 steps)
- Evaluate on held-out coding problems every 500 steps
- **Training time**: ~3-5 hours on 3090 (vs ~12+ hours on M4 MPS)

### `dpo.py` — Direct Preference Optimization (optional but modern)
- Collect pairs: (prompt, chosen_response, rejected_response)
- DPO loss aligns model to prefer better code without a separate reward model
- This is what makes the model "helpful" vs just "code-completing"
- **Simpler than RLHF**: no reward model, no PPO, just a modified cross-entropy loss

### `convert_to_mlx.py` — Port to MLX
- Convert PyTorch weights → MLX format
- Verify identical outputs (within float precision)
- MLX inference will be 2-3x faster on M4

### `quantize.py` — Quantize for speed
- 4-bit quantization (Q4) using MLX's built-in quantization
- Reduces memory ~4x, speeds up inference ~2x
- 360M model: fp32 ~1.4GB → Q4 ~90MB (fits easily in MacBook memory)

### Expected outcomes (SmolLM2-360M fine-tuned)
- Syntactically valid Python: ~80-85% of the time
- Solves simple problems (fizzbuzz, list ops, basic algorithms): ~50-70%
- Follows instructions reasonably well
- Fast inference: ~80+ tokens/sec on M4 via MLX Q4
- Noticeably better than 135M due to 2.5x more parameters

---

## Phase 7: Agentic Coding Assistant — 7-10 days

**Goal**: Build a ReAct + Reflection agent around the fine-tuned model with tool use, code execution, and a CLI interface.

### Tools: `tools/base.py`, `code_executor.py`, `file_ops.py`, `shell.py`

```python
class Tool(ABC):
    name: str
    description: str
    parameters: dict   # JSON schema for structured calling

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult: ...
```

- **CodeExecutor**: Run Python in subprocess with timeout (10s) and memory limit
- **FileOps**: Read/write files in allowed directories
- **Shell**: Execute shell commands (sandboxed, allowlisted)

### `function_calling.py` — Structured tool use ← **Modern approach**
Instead of parsing free-text "ACT: tool_name" (unreliable with small models), use structured format:

```json
{"tool": "execute_python", "args": {"code": "print(2+2)"}}
```

- Constrained decoding: when a tool call is expected, force output to start with `{`
- JSON schema validation before execution
- This is how Claude, GPT-4, and Gemini handle tool use internally

### `agent.py` — ReAct + Reflection Loop

```
1. USER: query
2. THINK: model reasons about the task
3. CALL: model invokes a tool (structured JSON)
4. OBSERVE: tool result fed back
5. REFLECT: model checks if the result is correct ← NEW
6. Repeat 2-5 or output ANSWER
```

**Reflection** (modern addition): After each tool result, the model explicitly checks:
- Did the code execute successfully?
- Does the output match expectations?
- Should I try a different approach?

This catches errors that basic ReAct misses.

### `memory.py` — Context Management
- System prompt + tools description (~200 tokens)
- Conversation history with sliding window
- For a 2048-token context (SmolLM2): ~1800 tokens for conversation
- Summarize older messages when context fills up

### `inference.py` — MLX Inference Engine
- Load quantized MLX model
- KV-cache for fast autoregressive generation
- Streaming output (token by token to terminal)
- Temperature/top-p controls

### `cli.py` — Terminal Interface
```
$ python -m phase7_agent.cli
🤖 Local Coding Assistant (SmolLM2-135M, MLX Q4)
> Write a function to check if a number is prime

THINK: I need to write a Python function...
CALL: execute_python {"code": "def is_prime(n):\n    ..."}
OBSERVE: Function defined successfully
REFLECT: Let me test it with some values...
CALL: execute_python {"code": "print(is_prime(7), is_prime(4))"}
OBSERVE: True False
ANSWER: Here's a function to check if a number is prime: ...
```

### Realistic expectations for Phase 7
- The 360M model will be a limited but functional agent
- Tool use will work ~65-75% of the time with constrained decoding
- Simple coding tasks (1-2 function problems) will often succeed
- Complex multi-step tasks will struggle
- **But you'll understand exactly how agents like Claude Code work internally**

---

## Dependencies & Sequence

```
Phase 1 (Foundations) ─────────┐
                               ├──→ Phase 3 (Transformer) ──→ Phase 4 (Training) ──→ Phase 5 (Generation)
Phase 2 (Tokenizer) ──────────┘                                                           │
                                                                                           ↓
                                                              Phase 6 (Fine-tune SmolLM2) ──→ Phase 7 (Agent)
```

- Phase 1 and 2 can be done **in parallel**
- Phase 3 depends on Phase 1 (understanding) + Phase 2 (tokenizer for testing)
- Phase 4-5 are sequential
- Phase 6 can partially overlap with Phase 5 (downloading/preparing data)
- Phase 7 depends on Phase 6

---

## Timeline

| Phase | Duration | Train on | Key Deliverable |
|---|---|---|---|
| 1: Foundations | 4-5 days | — | 4 notebooks, deep understanding of NN/attention/modern components |
| 2: Tokenizer | 2-3 days | — | Working BPE tokenizer with byte-fallback |
| 3: Transformer | 5-7 days | — | Modern Llama-style 15M param model |
| 4: Training | 5-7 days + ~2h | ROG 3090 | Model generates coherent English text |
| 5: Generation | 2-3 days | — | Interactive REPL with KV-cache, multiple decoding strategies |
| 6: Fine-tune | 7-10 days + ~5h | ROG 3090 | SmolLM2-360M fine-tuned for code, LoRA, DPO, MLX + quantized |
| 7: Agent | 7-10 days | — | Working CLI coding assistant with tool use |
| **Total** | **~5-7 weeks** | | **Local agentic coding assistant running on MacBook** |

---

## Verification Plan

After each phase, verify before moving on:

1. **Phase 1**: Each notebook runs end-to-end. XOR network converges. Attention weights visualize correctly. RoPE/RMSNorm/SwiGLU produce correct outputs vs reference implementations.

2. **Phase 2**: `decode(encode(text)) == text` for 100+ test strings including Unicode/emoji/code. Compression ratio > 2x for English.

3. **Phase 3**: `pytest test_model.py` — shapes correct, causal mask works, runs on MPS, ~15M params.

4. **Phase 4**: Training loss curve decreases. Validation loss tracks training loss. Checkpoints save/load correctly. Model generates recognizable English after training.

5. **Phase 5**: Interactive generation works. KV-cache matches non-cached output. Perplexity measured on validation set.

6. **Phase 6**: LoRA fine-tuning reduces code loss. Model generates syntactically valid Python >70% of the time. MLX conversion produces identical outputs. Quantized model runs at >100 tok/s.

7. **Phase 7**: Agent can: (a) answer a simple question, (b) write and execute a Python function, (c) read a file and answer questions about it, (d) self-correct when code fails.

---

## Key Files to Modify/Create (in order)

1. `pyproject.toml` — project setup
2. `phase1_foundations/01_neural_network.ipynb` — start here
3. `phase2_tokenizer/bpe_tokenizer.py` — can start in parallel with Phase 1
4. `phase3_transformer/rope.py` — the most novel component
5. `phase3_transformer/attention.py` — core of the model
6. `phase3_transformer/model.py` — ties everything together
7. `phase4_training/trainer.py` — where the model actually learns
8. `phase6_finetune/lora.py` — educational LoRA implementation
9. `phase7_agent/agent.py` — the capstone

## References & Resources

- **Sebastian Raschka**: "Build a Large Language Model From Scratch" (2024) + sequel on reasoning models
- **Stanford CS336**: Language Modeling from Scratch (Spring 2026)
- **SmolLM2 paper**: arXiv 2502.02737 — training recipe for small models
- **Karpathy's minbpe**: BPE tokenizer reference implementation
- **MLX documentation**: mlx-framework.org
- **HuggingFace SmolLM**: huggingface.co/HuggingFaceTB
- **WSD schedule**: arXiv 2410.05192 — understanding warmup-stable-decay
