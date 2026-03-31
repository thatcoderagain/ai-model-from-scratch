# AI/ML Concepts Guide — Everything Explained

A reference for every concept, term, and technique used in this project.
Written for someone building their first LLM from scratch.

---

## Table of Contents

1. [How a Language Model Works](#1-how-a-language-model-works)
2. [Neural Network Fundamentals](#2-neural-network-fundamentals)
3. [Transformer Architecture](#3-transformer-architecture)
4. [Modern Components (2025-era)](#4-modern-components-2025-era)
5. [Training a Model](#5-training-a-model)
6. [Tokenization](#6-tokenization)
7. [Text Generation](#7-text-generation)
8. [Model Types and Stages](#8-model-types-and-stages)
9. [Fine-Tuning Techniques](#9-fine-tuning-techniques)
10. [Alignment (DPO/RLHF)](#10-alignment-dporlhf)
11. [Quantization and Deployment](#11-quantization-and-deployment)
12. [HuggingFace Model Zoo Decoded](#12-huggingface-model-zoo-decoded)
13. [Agent Architecture](#13-agent-architecture)
14. [Key Papers](#14-key-papers)

---

## 1. How a Language Model Works

At its core, a language model does one thing: **predict the next token**.

Given: `"The cat sat on the"`
Predict: `"mat"` (with some probability)

That's it. Every capability — writing code, answering questions, reasoning — emerges from training this simple prediction task on trillions of tokens of text.

### The training loop

```
1. Feed text:    "The cat sat on the"
2. Model predicts: probabilities for every word in vocabulary
3. Correct answer: "mat"
4. Compute loss:  how wrong was the prediction? (cross-entropy)
5. Backpropagate: compute gradients for every weight
6. Update weights: nudge them to make "mat" more likely next time
7. Repeat billions of times with different text
```

### Why this produces intelligence

After seeing trillions of examples:
- To predict the next word in a math proof, the model must learn math
- To predict the next token in Python code, the model must learn programming
- To predict the next word in a story, the model must understand narrative, characters, causality

The model doesn't "understand" in the human sense — it builds statistical patterns that happen to capture deep structure of language and knowledge.

---

## 2. Neural Network Fundamentals

### Neuron

The basic unit: `output = activation(weights · inputs + bias)`

- **Weights**: numbers that control how much each input matters. These are LEARNED.
- **Bias**: a number added after the weighted sum. Lets the neuron fire even with zero input.
- **Activation**: a non-linear function applied to the sum. Without it, stacking layers is pointless.

### Activation Functions

| Function | Formula | Used in | Why |
|---|---|---|---|
| **ReLU** | `max(0, x)` | Old hidden layers | Simple, fast, no vanishing gradient for positive values |
| **Sigmoid** | `1/(1+e^(-x))` | Binary output | Squashes to (0,1), represents probability |
| **GELU** | `x · Φ(x)` | GPT-2 FFN | Smooth ReLU, was the default before SwiGLU |
| **Swish/SiLU** | `x · sigmoid(x)` | Modern LLMs (Llama) | Smooth, allows small negatives through |
| **SwiGLU** | `Swish(xW₁) ⊙ (xW₂)` | All 2025 LLMs | Gated activation, 1-2% better than GELU |

### Linear Layer

Many neurons in parallel: `y = x @ W^T + b`

- Input shape: `(batch_size, in_features)`
- Output shape: `(batch_size, out_features)`
- Parameters: `in_features × out_features` weights + `out_features` biases
- This is the fundamental building block — every layer in a transformer uses this

### Weight Initialization

If weights start too large → activations explode. Too small → activations vanish.

| Method | Formula | Used with |
|---|---|---|
| **Xavier/Glorot** | `W ~ N(0, 2/(fan_in + fan_out))` | Sigmoid, Tanh |
| **He/Kaiming** | `W ~ N(0, 2/fan_in)` | ReLU |
| **Normal(0, 0.02)** | `W ~ N(0, 0.02)` | Transformers (standard practice) |

### Backpropagation

Algorithm to compute gradients for ALL parameters in one backward pass.

```
Forward:  x → [Layer1] → [Layer2] → [Loss] → L
Backward: dL/dx ← [Layer1] ← [Layer2] ← [Loss] ← 1
```

Each layer computes: "given the gradient from above, what's my gradient for my parameters and my input?"

**Chain rule**: if `L = f(g(h(x)))`, then `dL/dx = dL/df · df/dg · dg/dh · dh/dx`

This is exactly what PyTorch autograd does automatically with `loss.backward()`.

### Loss Functions

| Loss | Formula | Used for |
|---|---|---|
| **MSE** | `mean((pred - target)²)` | Regression, simple training |
| **Cross-entropy** | `-log(P(correct))` | Classification, language models |

For LLMs, cross-entropy is THE loss: "how surprised is the model by the correct next token?"

### Softmax

Converts raw model outputs (logits) into probabilities:

```
softmax(x_i) = exp(x_i) / Σexp(x_j)
```

Properties: all outputs positive, sum to 1. Always used before cross-entropy.

### Perplexity

`perplexity = exp(cross_entropy_loss)`

Intuitively: "how many tokens is the model choosing between on average?"
- Perplexity 1 = perfect (knows exactly what comes next)
- Perplexity 10 = choosing between ~10 options
- Perplexity 100 = very uncertain

---

## 3. Transformer Architecture

### What is a Transformer?

A neural network architecture built entirely on **attention** — no RNNs, no convolutions. Every modern LLM (GPT, Claude, Llama, Gemma) is a transformer.

Paper: "Attention Is All You Need" — Vaswani et al. (2017, Google)

### Self-Attention

The core mechanism: every token can directly look at every other token.

```
Attention(Q, K, V) = softmax(QK^T / √d_k) · V
```

- **Query (Q)**: "What am I looking for?" (each token generates this)
- **Key (K)**: "What do I contain?" (each token generates this)
- **Value (V)**: "What information do I provide?" (each token generates this)
- **QK^T**: relevance score between every pair of tokens
- **/ √d_k**: scaling to prevent softmax from saturating
- **softmax**: convert scores to probabilities
- **· V**: weighted sum of values based on attention scores

Analogy: Q is a search query, K is the index, V is the content. High Q·K score = this content is relevant.

### Causal Masking

For text generation, token 5 cannot see tokens 6, 7, 8... (they don't exist yet).

Enforced by setting attention scores to -infinity for future positions:
```
1 -∞ -∞ -∞     Token 0 sees: only itself
1  1 -∞ -∞     Token 1 sees: 0, 1
1  1  1 -∞     Token 2 sees: 0, 1, 2
1  1  1  1     Token 3 sees: 0, 1, 2, 3
```

### Multi-Head Attention (MHA)

Run multiple attention operations in parallel, each learning different patterns:
- Head 1 might learn syntax (subject-verb agreement)
- Head 2 might learn semantics (word meaning)
- Head 3 might learn proximity (nearby words)

Split embedding dimension across heads: `d_head = d_model / n_heads`

### Residual Connections

`output = x + sublayer(x)`

Without residuals, gradients must flow through every transformation → they vanish. With residuals, gradients have a direct "highway" back to early layers.

From He et al. (2015), ResNet. This is why transformers can have 100+ layers.

### Pre-Norm vs Post-Norm

```
Pre-norm  (modern): x → Norm → Sublayer → + x
Post-norm (GPT-2):  x → Sublayer → + x → Norm
```

Pre-norm is more stable for training deep networks.

### The Full Transformer Block

```
x → RMSNorm → Attention → + x (residual)
  → RMSNorm → FFN       → + x (residual)
```

A model stacks N of these blocks. Our model uses N=6. Llama 3 70B uses N=80.

### Weight Tying

The token embedding matrix and the output projection (lm_head) share the same weights. Embedding maps token→vector, lm_head maps vector→token — they're inverse operations.

Paper: Press & Wolf (2017). Reduces parameters and improves quality.

### Encoder vs Decoder

| Type | Sees | Used for | Examples |
|---|---|---|---|
| **Encoder-only** | All tokens (bidirectional) | Classification, search, embeddings | BERT, RoBERTa |
| **Decoder-only** | Past tokens only (causal) | Text generation, chat | GPT, Llama, Claude |
| **Encoder-decoder** | Encoder: all, Decoder: past + encoded | Translation, summarization | T5, BART |

We built a **decoder-only** model — this is what 90% of modern LLMs are.

---

## 4. Modern Components (2025-era)

These replaced GPT-2's original components. Every 2024+ model uses them.

### RoPE (Rotary Position Embeddings)

**Problem**: The original transformer had no sense of token position (attention is permutation-invariant). GPT-2 added learned positional embeddings — but these have a fixed max length and don't capture relative position.

**Solution**: Rotate Q and K vectors based on their position. When computing Q·K (attention score), the rotation angles subtract → the score naturally depends on relative position (i-j).

```
θ_i = 10000^(-2i/d)           — frequency for each dimension pair
angle = position × θ_i         — rotation angle
[x_2i, x_2i+1] is rotated by angle in 2D
```

Low-dimension pairs rotate fast (fine position), high-dimension pairs rotate slow (coarse position). Like a clock: seconds (fast), minutes (medium), hours (slow).

Paper: Su et al. (2021), "RoFormer"

**Why it matters**: Models can extrapolate to longer sequences than seen during training, and position information is refreshed at every attention layer (not just added once at input).

### RMSNorm (Root Mean Square Normalization)

```
LayerNorm (old):  (x - mean) / std × γ + β    — two learnable params
RMSNorm (modern): x / RMS(x) × γ              — one learnable param
```

Where `RMS(x) = √(mean(x²) + ε)`

7-64% faster than LayerNorm with equivalent quality. The mean subtraction in LayerNorm turns out to be unnecessary.

Paper: Zhang & Sennrich (2019)

### SwiGLU (Swish-Gated Linear Unit)

```
Old FFN:    GELU(x·W₁ + b₁) · W₂ + b₂           — 2 matrices, 2 biases
SwiGLU FFN: (Swish(x·W_gate) ⊙ (x·W_up)) · W_down  — 3 matrices, no bias
```

The **gate** learns what information to filter. 1-2% better than GELU at same parameter count.

Paper: Shazeer (2020), "GLU Variants Improve Transformer"

### GQA (Grouped Query Attention)

Standard MHA: each Q head has its own K and V → large KV-cache for inference.

GQA: multiple Q heads **share** K/V heads.

```
MHA: 6 Q heads, 6 KV heads   (1:1, no sharing)
GQA: 6 Q heads, 2 KV heads   (3:1, our model)
MQA: 6 Q heads, 1 KV head    (6:1, maximum sharing)
```

GQA saves 3x KV-cache memory with minimal quality loss.

Paper: Ainslie et al. (2023)

---

## 5. Training a Model

### The Training Loop

```python
for step in range(max_steps):
    batch = get_next_batch()           # (input_tokens, target_tokens)
    logits = model(input_tokens)        # forward pass
    loss = cross_entropy(logits, targets)  # how wrong?
    loss.backward()                     # compute gradients
    optimizer.step()                    # update weights
    scheduler.step()                    # adjust learning rate
```

### Key Hyperparameters

| Parameter | What it controls | Typical value |
|---|---|---|
| **Learning rate** | Size of each weight update | 3e-4 (peak) |
| **Batch size** | Samples processed per update | 32-64 (effective) |
| **Max steps** | When to stop training | 10K-100K |
| **Weight decay** | Penalizes large weights (regularization) | 0.1 |
| **Gradient clipping** | Caps gradient magnitude | 1.0 |
| **Warmup steps** | Gradual LR ramp at start | 5% of total |

### Optimizers

| Optimizer | What it does |
|---|---|
| **SGD** | `w -= lr × gradient`. Simplest possible. |
| **Adam** | SGD + momentum + adaptive LR per parameter. The standard. |
| **AdamW** | Adam with decoupled weight decay. Used by all modern LLMs. |
| **Muon** | 2025 breakthrough — 2x more efficient than AdamW for LLMs. |

### WSD Learning Rate Schedule

The modern standard, replacing cosine decay:

```
Phase 1 — Warmup  (5%):   LR ramps linearly from 0 → max_lr
Phase 2 — Stable  (75%):  LR stays constant at max_lr
Phase 3 — Decay   (20%):  LR drops linearly from max_lr → min_lr
```

Why WSD > cosine: cosine wastes most of training at sub-optimal LR. WSD keeps LR high during the stable phase where most learning happens.

Paper: arXiv:2410.05192 (2024)

### Mixed Precision Training (fp16)

- Forward and backward pass: compute in fp16 (2x faster, half memory)
- Optimizer state: kept in fp32 (numerical stability)
- **GradScaler**: prevents fp16 gradients from underflowing to zero

Only works on CUDA GPUs (NVIDIA). Apple MPS doesn't reliably support it.

### Gradient Accumulation

Can't fit batch_size=64 in memory? Process 8 samples at a time, accumulate gradients for 8 steps, then update once. Effective batch = 8 × 8 = 64.

### Checkpointing

Save model + optimizer state periodically. Allows resuming after crash. Save to CPU tensors for cross-device compatibility.

---

## 6. Tokenization

### What is Tokenization?

Converting text into numbers the model can process.

```
"Hello, world!" → [15496, 11, 995, 0]  (token IDs)
```

### Why Not Characters?

Character-level: "Hello" = [H, e, l, l, o] = 5 tokens. Simple but:
- Sequences are too long (slow attention: O(n²))
- Each character carries little meaning

Word-level: "Hello" = [Hello] = 1 token. Compact but:
- Can't handle new words → "unknown" tokens
- Huge vocabulary needed

### BPE (Byte Pair Encoding)

The sweet spot. Start with characters (or bytes), iteratively merge the most frequent pairs.

```
Training:
  Start:  ['h','e','l','l','o'] appears 100 times
  Merge 1: 'l'+'l' → 'll'  (most frequent pair)
  Merge 2: 'he' → 'he'
  Merge 3: 'he'+'ll' → 'hell'
  ...
  Eventually: 'hello' is one token
```

**Byte-level BPE**: Start from 256 byte values instead of characters. Handles ANY text — no unknown tokens possible. Used by GPT-2/3/4, Llama, Claude.

### Special Tokens

Tokens with special meaning added to the vocabulary:

| Token | Purpose |
|---|---|
| `<\|begin_of_text\|>` | Start of document |
| `<\|end_of_text\|>` | End of document — model should stop generating |
| `<\|pad\|>` | Padding for batch alignment |
| `<\|instruction\|>` | Start of user instruction (for instruct models) |
| `<\|response\|>` | Start of model response |

### Vocabulary Size

| Model | Vocab size |
|---|---|
| Our model | 4,096 |
| GPT-2 | 50,257 |
| Llama 3 | 128,256 |
| Gemma 3 | 256,000 |

Larger vocab = more tokens represented as single IDs = shorter sequences = faster. But larger embedding matrix = more parameters.

---

## 7. Text Generation

### Autoregressive Generation

Generate one token at a time, feed output back as input:

```
Prompt:     "The cat"
Step 1:     "The cat" → model → "sat"
Step 2:     "The cat sat" → model → "on"
Step 3:     "The cat sat on" → model → "the"
...
```

### Decoding Strategies

| Strategy | How it works | Use case |
|---|---|---|
| **Greedy** | Always pick the most likely token | Deterministic, often repetitive |
| **Temperature** | Scale logits before softmax. Higher = more random. | Control creativity |
| **Top-k** | Only consider the k most likely tokens | Filter low-quality options |
| **Top-p (nucleus)** | Keep smallest set of tokens with cumulative prob ≥ p | Adaptive filtering |
| **Repetition penalty** | Reduce probability of recently seen tokens | Prevent loops |

**Temperature intuition:**
- `temp=0`: Greedy (always pick top token). Good for code, bad for stories.
- `temp=0.7`: Balanced. Good default for most tasks.
- `temp=1.0`: Raw distribution. More diverse.
- `temp=1.5`: Very random. Creative but may be incoherent.

### KV-Cache

Without cache: generating token N requires recomputing attention for all N-1 previous tokens. O(N²) total.

With cache: store K and V from previous tokens. Only compute new token's Q, K, V. O(N) total.

This is why GQA matters — fewer KV heads = smaller cache = faster generation.

---

## 8. Model Types and Stages

### The Production Pipeline

```
Internet text (trillions of tokens)
    │
    ▼
┌──────────────┐
│  Base Model  │  "I predict the next token"
│              │  Months of training, millions of $$$
└──────┬───────┘
       │  + instruction/response pairs (SFT)
       ▼
┌──────────────┐
│   Instruct   │  "I follow instructions"
│    Model     │  Hours to days of fine-tuning
└──────┬───────┘
       │  + preference pairs (DPO/RLHF)
       ▼
┌──────────────┐
│ Chat/Aligned │  "I'm helpful AND safe"
│    Model     │  Hours to days of alignment
└──────────────┘
```

### Comparison

| Type | Training data | Behavior | Example |
|---|---|---|---|
| **Base** | Raw text (web, books, code) | Completes text. Ask a question, it generates more questions. | `SmolLM2-360M` |
| **Instruct** | Base + (instruction, response) pairs | Follows commands. Ask a question, it answers. | `SmolLM2-360M-Instruct` |
| **Chat/Aligned** | Instruct + (chosen, rejected) pairs | Helpful AND safe. Refuses harmful requests. | Claude, ChatGPT |
| **Specialized** | Any above + domain data | Domain expert. | CodeLlama, BioMistral |

### Can you use base as instruct?

Yes, but you have to trick it with careful prompting:

```python
# Base model — won't work:
prompt = "Write a fibonacci function"
# Output: "in Python. Fibonacci sequences are commonly..."

# Base model — works with format trick:
prompt = "Question: Write a fibonacci function.\nAnswer:\ndef fibonacci(n):"
# Output: continues the code (it's seen Q&A format in training data)

# Instruct model — just works:
prompt = "Write a fibonacci function"
# Output: "def fibonacci(n): ..."
```

Base models CAN work but are inconsistent. Instruct models learned the pattern permanently.

---

## 9. Fine-Tuning Techniques

### Full Fine-Tuning

Update ALL parameters. Best quality but:
- Needs lots of VRAM (full model + gradients + optimizer states)
- Slow (updating 360M+ params per step)
- Risk of catastrophic forgetting (model forgets pre-training knowledge)

### LoRA (Low-Rank Adaptation)

Freeze original weights, add small trainable "adapter" matrices.

```
Original: y = x @ W              (W is frozen)
LoRA:     y = x @ W + x @ A @ B  (only A and B are trained)
```

- A has shape (d_in, rank), B has shape (rank, d_out)
- Rank r=32: trains (d_in + d_out) × 32 params instead of d_in × d_out
- Typically 1-2% of total parameters, recovers 90-95% of full fine-tuning quality

Paper: Hu et al. (2021)

**After training**: merge A@B back into W. Zero inference overhead.

### QLoRA

LoRA on a 4-bit quantized base model. Even less VRAM needed.
The 4-bit model runs forward pass, but LoRA adapters are fp32 for training stability.

### Which Layers to Adapt?

Standard: apply LoRA to attention Q and V projections.
More aggressive: Q, K, V, O, gate, up, down (all linear layers).
Q+V alone captures most of the benefit.

### Other Methods

| Method | How it works |
|---|---|
| **Adapter layers** | Insert small bottleneck modules between layers |
| **Prefix tuning** | Prepend learnable "virtual tokens" to input |
| **Prompt tuning** | Learn soft prompts instead of touching model weights |
| **BitFit** | Only fine-tune the bias terms (~0.1% of params) |

---

## 10. Alignment (DPO/RLHF)

### Why Alignment?

After SFT, the model follows instructions but doesn't know which response is BETTER. It might:
- Write verbose code when concise is better
- Give dangerous instructions if asked
- Hallucinate confidently

### RLHF (Reinforcement Learning from Human Feedback) — Old Approach

```
1. Train a reward model on human preferences
2. Use PPO to optimize the LLM against the reward model
3. KL penalty prevents drifting too far from the SFT model
```

Complex: needs 3 models in memory (LLM + reward model + reference model). Unstable training.

### DPO (Direct Preference Optimization) — Modern Approach

Skip the reward model entirely. Train directly on preference pairs:

```
Prompt:    "Write a function to reverse a string"
Chosen:    def reverse(s): return s[::-1]           ← better
Rejected:  def reverse(s):                          ← worse
               result = ""
               for i in range(len(s)-1, -1, -1):
                   result += s[i]
               return result
```

DPO loss: push model to be more likely to produce "chosen" and less likely to produce "rejected", relative to the reference model.

Paper: Rafailov et al. (2023)

**β (beta)**: controls how much the model can diverge from reference. β=0.1 is typical.

### Newer Variants

| Method | Improvement over DPO |
|---|---|
| **SimPO** | No reference model needed. Fixes DPO's bias toward longer responses. |
| **ORPO** | Combines SFT and preference optimization into one step. |
| **KTO** | Works with binary feedback (thumbs up/down) instead of paired comparisons. |

---

## 11. Quantization and Deployment

### What is Quantization?

Reducing weight precision to make models smaller and faster.

```
fp32:  4 bytes per weight    360M model = 1.4 GB
fp16:  2 bytes per weight    360M model = 720 MB
int4:  0.5 bytes per weight  360M model = ~180 MB
```

### Formats Compared

| Format | Best for | How it works |
|---|---|---|
| **GGUF** | CPU, Mac (llama.cpp) | Portable single file. Variants: Q4_K_M, Q5_K_M, Q8_0. |
| **GPTQ** | NVIDIA GPU | Post-training quantization using calibration data. |
| **AWQ** | NVIDIA GPU | Activation-aware: preserves important weights. Better than GPTQ. |
| **EXL2** | NVIDIA GPU (ExLlama) | Variable bits per layer. Maximum speed. |
| **MLX** | Apple Silicon | Apple's native format. Fastest on M-series chips. |
| **bitsandbytes** | Fine-tuning | Enables QLoRA — training on quantized models. |
| **ONNX** | Cross-platform | Vendor-neutral. For mobile, edge, non-PyTorch deployment. |

### How to Choose

```
Mac inference?        → MLX (fastest) or GGUF (most compatible)
NVIDIA GPU inference? → AWQ > GPTQ > EXL2
CPU inference?        → GGUF
Fine-tuning?          → bitsandbytes (QLoRA)
Mobile/Edge?          → ONNX
```

### Quality vs Size Tradeoff

| Precision | Size | Quality | Speed |
|---|---|---|---|
| fp32 | 100% | 100% | Baseline |
| fp16/bf16 | 50% | ~99.9% | ~2x faster |
| int8 | 25% | ~99% | ~2-3x faster |
| int4 (Q4_K_M) | ~13% | ~95% | ~3-4x faster |
| int2 | ~7% | ~85% | Fastest, lowest quality |

Sweet spot: **4-bit quantization** gives 95% quality at 3.5x compression.

### Serialization Formats

| Format | What it is |
|---|---|
| **SafeTensors** | Secure binary format (can't execute arbitrary code). Default on HuggingFace. |
| **Pickle (.bin)** | Old PyTorch format. Security risk. Being phased out. |
| **GGUF** | Single file with model + metadata. For llama.cpp. |

---

## 12. HuggingFace Model Zoo Decoded

### Reading a Model Name

```
TheBloke/Llama-2-13B-Chat-GPTQ
│         │      │    │     │
│         │      │    │     └── Quantization format (4-bit GPU)
│         │      │    └──────── Training stage (aligned for chat)
│         │      └───────────── Parameter count (13 billion)
│         └──────────────────── Model family
└────────────────────────────── Publisher/converter
```

More examples:

```
HuggingFaceTB/SmolLM2-360M-Instruct
  → SmolLM2, 360M params, instruction-tuned

mlx-community/Llama-3-8B-Instruct-4bit
  → Llama 3, 8B params, instruction-tuned, 4-bit MLX format

microsoft/Phi-4-mini-flash-reasoning
  → Phi-4, small size, fast inference, optimized for reasoning

google/gemma-3-1b-it
  → Gemma 3, 1B params, "it" = instruction-tuned
```

### By Architecture

| Architecture | How it works | Examples |
|---|---|---|
| **Decoder-only** | Predicts next token (causal) | GPT, Llama, SmolLM2, Claude |
| **Encoder-only** | Understands text (bidirectional) | BERT, RoBERTa |
| **Encoder-decoder** | Input→Output transformation | T5, BART |
| **Diffusion** | Iteratively denoises from random noise | Stable Diffusion, FLUX |
| **MoE** (Mixture of Experts) | Routes tokens to specialist sub-networks | Mixtral, Llama 4 |
| **Vision Transformer** | Applies transformer to image patches | CLIP, DINOv2 |
| **State Space (Mamba)** | Linear-time alternative to attention | Mamba, Jamba |

### By Task

| Task | What it does | Example models |
|---|---|---|
| **text-generation** | Generate/complete text | Llama, GPT, SmolLM2 |
| **text-classification** | Categorize text | BERT, RoBERTa |
| **question-answering** | Answer from context | BERT-QA, T5 |
| **text-to-image** | Generate images from text | Stable Diffusion, FLUX |
| **image-to-text** | Caption/describe images | BLIP-2, LLaVA |
| **speech-to-text** | Transcribe audio | Whisper |
| **text-to-speech** | Generate audio | Bark, XTTS |
| **embedding** | Text → vector for search | BGE, E5, Sentence-BERT |
| **translation** | Language A → Language B | NLLB, mBART |
| **object-detection** | Find objects in images | YOLO, DETR |

### By Size

| Class | Params | Hardware needed | Quality |
|---|---|---|---|
| Nano/Micro | 100M-1B | Phone, laptop CPU | Basic |
| Small | 1B-7B | Gaming laptop GPU | Good |
| Medium | 7B-13B | 1 GPU (24GB) | Great |
| Large | 13B-70B | 2-4 GPUs | Excellent |
| XL | 70B+ | GPU cluster | State of the art |

### Fine-Tuning Variants

| Tag | Meaning |
|---|---|
| **Full model** | All weights, ready to use |
| **LoRA adapter (unmerged)** | Just the small adapter (~10-50MB). Needs base model. |
| **LoRA adapter (merged)** | Adapter baked into base. Single model, no overhead. |
| **QLoRA** | LoRA trained on 4-bit quantized base. Least VRAM needed. |

---

## 13. Agent Architecture

### What is an Agent?

A language model + the ability to take actions in the world. The model generates text that is parsed as tool calls, executed, and the results are fed back.

### ReAct Pattern (Reasoning + Acting)

```
1. USER: "What's 347 × 923?"
2. THINK: I should use the calculator tool for this.
3. CALL: {"tool": "execute_python", "args": {"code": "print(347 * 923)"}}
4. OBSERVE: 320281
5. ANSWER: 347 × 923 = 320,281
```

Paper: Yao et al. (2022)

### Reflection Pattern

After each tool result, the model explicitly checks:
- Did it work correctly?
- Does the output match expectations?
- Should I try a different approach?

This catches errors that basic ReAct misses.

Paper: Shinn et al. (2023), "Reflexion"

### Function Calling

The structured format for tool invocation:

```json
{"tool": "execute_python", "args": {"code": "print(2+2)"}}
```

More reliable than free-text parsing. Can be validated against JSON schemas.

### Tools

| Tool | Purpose | Safety |
|---|---|---|
| Code executor | Run Python code | Subprocess with timeout |
| File reader | Read file contents | Directory allowlisting |
| File writer | Write files | Directory allowlisting |
| Shell | Run commands | Command allowlisting |

### Context Window Management

The model can only see `max_tokens` at a time. Everything — system prompt, conversation history, tool results — must fit.

Strategy: keep system prompt (always), keep recent messages, drop oldest when full.

---

## 14. Key Papers

| Year | Paper | What it introduced |
|---|---|---|
| 2015 | He et al. "Deep Residual Learning" | Residual connections (skip connections) |
| 2017 | Vaswani et al. "Attention Is All You Need" | The Transformer architecture |
| 2017 | Press & Wolf "Using the Output Embedding" | Weight tying |
| 2018 | Radford et al. "Improving Language Understanding" | GPT-1 (decoder-only pre-training) |
| 2019 | Radford et al. "Language Models are Unsupervised Multitask Learners" | GPT-2 |
| 2019 | Zhang & Sennrich "Root Mean Square Layer Normalization" | RMSNorm |
| 2020 | Shazeer "GLU Variants Improve Transformer" | SwiGLU activation |
| 2021 | Su et al. "RoFormer" | RoPE (Rotary Position Embeddings) |
| 2021 | Hu et al. "LoRA" | Low-Rank Adaptation for fine-tuning |
| 2022 | Yao et al. "ReAct" | Reasoning + Acting agent pattern |
| 2023 | Ainslie et al. "GQA" | Grouped Query Attention |
| 2023 | Rafailov et al. "Direct Preference Optimization" | DPO (alignment without reward model) |
| 2023 | Shinn et al. "Reflexion" | Self-reflection for agents |
| 2024 | WSD Schedule paper (arXiv:2410.05192) | Warmup-Stable-Decay LR schedule |
| 2025 | SmolLM2 paper (arXiv:2502.02737) | Small model training recipe |
| 2025 | Jordan et al. "Muon is Scalable" | Muon optimizer (2x efficiency) |

---

## What We Built in This Project

| Phase | What | From scratch? |
|---|---|---|
| 1 | Neural networks, backprop, attention in numpy | 100% |
| 2 | BPE tokenizer | 100% |
| 3 | Llama-style transformer (15M params) | 100% |
| 4 | Training pipeline (CUDA/MPS, WSD, grad accum) | 100% |
| 5 | Generation (top-k/p, KV-cache, perplexity) | 100% |
| 6 | LoRA fine-tuning on SmolLM2-360M | LoRA: 100%, base model: HuggingFace |
| 7 | ReAct agent with tool use | 100% |

**167 tests. 102 tokens/sec on MacBook M4. 262MB memory footprint.**

---

## 15. Future Scope: Where to Go From Here

### The Industry Shift

AI is moving from "bigger models in the cloud" to "smaller models everywhere":

- **Privacy**: companies don't want their code/data sent to external APIs
- **Cost**: API calls at scale cost millions per year
- **Latency**: on-device = instant, cloud = network round-trip
- **Offline**: phones, cars, robots can't always reach the internet
- **Regulation**: EU, healthcare, finance require data to stay local

### Path 1: On-Device AI Engineer

Build AI that runs on phones, laptops, IoT devices — no cloud needed.

**What you'd do:**
- Take a 7B model → quantize to 4-bit → run on phone/laptop
- Optimize inference speed (KV-cache, speculative decoding, batching)
- Build custom MLX/GGUF pipelines for Apple devices
- Use knowledge distillation: train a tiny model to mimic a large one

**What to learn next:**
- llama.cpp (C++ inference engine, runs on anything)
- CoreML / ONNX Runtime (mobile deployment)
- Speculative decoding (use a tiny draft model to speed up a big model)
- Knowledge distillation (train SmolLM-135M to mimic Llama-70B)

**Who needs this:** Apple, Google (on-device Gemini Nano), Samsung, any mobile app company

### Path 2: Domain-Specific Model Builder

Fine-tune models for specific industries where generic models fail.

| Domain | Why generic models fail | What you'd build |
|---|---|---|
| Healthcare | Can't use cloud (HIPAA). Generic models hallucinate medical facts. | Fine-tune on medical literature, deploy on-premise |
| Legal | Confidential documents can't leave the firm | Fine-tune on legal precedents, run locally |
| Finance | Trading signals can't have cloud latency | Fine-tune on financial data, edge deployment |
| Internal codebase | Claude doesn't know your internal APIs | Fine-tune on your repo, run as internal copilot |

**The workflow (you already know every step):**
1. Collect domain data (documents, code, Q&A pairs)
2. Fine-tune with LoRA (Phase 6)
3. Align with DPO using domain expert preferences
4. Quantize and deploy on customer's hardware

**What to learn next:**
- Data curation and cleaning (the real bottleneck)
- Evaluation frameworks (how to measure if your medical model is safe)
- RAG (Retrieval Augmented Generation) — combine model with document search

### Path 3: Model Optimization / Compression

Make models smaller and faster without losing quality.

| Technique | What it does | Reduction |
|---|---|---|
| **Quantization** (done) | Reduce weight precision | 4x smaller |
| **Pruning** | Remove unimportant weights/neurons entirely | 2-10x smaller |
| **Knowledge distillation** | Train small model to mimic large model | 10-100x smaller |
| **Architecture search** | Find optimal layer sizes for a given compute budget | Varies |
| **Speculative decoding** | Tiny draft model proposes tokens, big model verifies | 2-4x faster |
| **MoE** (Mixture of Experts) | Only activate relevant expert sub-networks per token | Use 15% of params per token |

**What to learn next:**
- Structured pruning (remove entire attention heads or layers)
- Knowledge distillation training loops
- Neural Architecture Search (NAS)
- Flash Attention, PagedAttention (memory-efficient attention)

### Path 4: AI Product Builder

Ideas that are practical with current skills:

**Local Code Review Bot**
- Reads git diffs, suggests improvements
- Runs entirely on MacBook, no API costs
- Uses: fine-tuned code model + agent framework (Phase 7)

**Private Document Q&A**
- Company uploads documents, asks questions, gets answers
- Add RAG: embed documents → retrieve relevant chunks → feed to model
- Data never leaves their network

**Automated Test Writer**
- Reads a function → writes pytest tests → runs them → iterates until passing
- Uses: code executor + agent loop (Phase 7)

**Teaching Platform**
- This entire project as an interactive course
- Students build their own LLM step by step
- Sell on Udemy or your own platform

### Path 5: Open Source Contributions

| Project | What you could contribute | Relevant skills |
|---|---|---|
| **llama.cpp** | Optimize inference, add new quant formats | Model architecture, C++ |
| **MLX** | Add new model support, improve quantization | Apple Silicon, model conversion |
| **HuggingFace transformers** | Fix bugs, add new architectures | PyTorch, model internals |
| **vLLM** | Optimize serving, PagedAttention | KV-cache, attention optimization |
| **Your own project** | Publish this project as a learning resource | Everything you built |

---

## 16. Concrete Next Projects

### Project A: Better Local Coding Assistant (2-3 weeks)

Take what you built and make it actually useful:

1. Upgrade to SmolLM2-1.7B (bigger model, better code)
2. Add RAG: embed your codebase, retrieve relevant files before generating
3. Add git integration: read diffs, suggest commit messages
4. Fine-tune on YOUR code: collect your own coding patterns as training data
5. Package it: make it installable with `pip install your-local-copilot`

### Project B: Knowledge Distillation Pipeline (2-4 weeks)

Build a tool that creates small specialized models from large ones:

```
Input:  Large model (Llama-70B via API) + task description
Process: Generate training data with the large model, train the small one
Output: Small model (1B) that does that one task well
```

This is how many companies actually build production models.

### Project C: Model Compression Toolkit (3-4 weeks)

Build a toolkit that takes any HuggingFace model and:
1. Prunes unnecessary attention heads (measure which heads matter)
2. Distills knowledge from the full model into a smaller architecture
3. Quantizes to 4-bit
4. Benchmarks quality vs size at each step
5. Outputs a deployment-ready model

---

## 17. Key Optimization Techniques for On-Device AI

### Speculative Decoding

Use a tiny "draft" model (e.g., 135M) to propose several tokens at once.
The large model (e.g., 1.7B) verifies them in a single forward pass.
If the draft was right (often is for common patterns), you got multiple tokens for the cost of one.

**Result:** 2-4x speedup with zero quality loss.

### KV-Cache Optimization

Standard KV-cache grows linearly with sequence length → eventually fills memory.

Solutions:
- **Sliding window attention**: only cache last N tokens (Mistral uses this)
- **PagedAttention**: manage cache like virtual memory pages (vLLM)
- **GQA**: fewer KV heads = smaller cache (what we built)

### Pruning

Not all attention heads are equally important. Research shows you can remove 30-50% of heads with <1% quality loss.

**Structured pruning**: remove entire heads/layers (easy to deploy)
**Unstructured pruning**: zero out individual weights (needs sparse hardware)

### Knowledge Distillation

```
Teacher: Llama-70B (huge, accurate, slow)
Student: Your 1B model (small, fast)

1. Run teacher on thousands of prompts
2. Collect teacher's output probability distributions
3. Train student to match teacher's distributions (softer target than hard labels)
4. Student learns to approximate teacher at 1/70th the size
```

### Flash Attention

Standard attention: O(N²) memory (stores full attention matrix)
Flash Attention: O(N) memory (computes attention in tiles, never materializes full matrix)

**Result:** 2-4x faster, handles much longer sequences. Available on CUDA via PyTorch's `scaled_dot_product_attention`.

---

## 18. Career Relevance

| Role | What they do | How this project helps |
|---|---|---|
| **ML Engineer** | Train and deploy models | You built the full training + deployment pipeline |
| **MLOps Engineer** | Infrastructure for model serving | You understand model formats, quantization, device targeting |
| **AI Product Engineer** | Build products using models | You built an agent with tools, memory, CLI |
| **On-Device AI Engineer** | Optimize models for edge | You did quantization, MLX conversion, understand architecture trade-offs |
| **Research Engineer** | Push state of the art | You understand the papers and can implement from scratch |

**Your competitive advantage:** Most people use `model.generate()` and have no idea what's inside. You built every layer from numpy to a deployed agent. In interviews, when someone asks "how does attention work?" — you don't recite a definition, you say "I implemented it from scratch, here's the repo."
