"""Model configuration — all hyperparameters in one place.

Terminology:
    vocab_size: Number of unique tokens the model can represent.
    n_layer: Number of transformer blocks stacked. More layers = more capacity = more compute.
    n_head: Number of query attention heads. Each head learns different attention patterns.
    n_kv_head: Number of key/value heads (GQA). Fewer than n_head to save memory.
    n_embd: Embedding dimension (d_model). Size of the vector representing each token.
    block_size: Maximum context length. The model can see at most this many tokens.
    ffn_hidden: SwiGLU intermediate dimension. Typically ~2.67x n_embd (to match param count
        of the old 4x GELU FFN, since SwiGLU uses 3 matrices instead of 2).
    dropout: Probability of dropping activations during training (regularization).
    bias: Whether to use bias in linear layers. Modern models skip bias for simplicity.
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    # Vocabulary
    vocab_size: int = 4096

    # Transformer depth and width
    n_layer: int = 6          # number of transformer blocks
    n_head: int = 6           # query heads
    n_kv_head: int = 2        # key/value heads (GQA: 3 query heads per KV head)
    n_embd: int = 384         # embedding dimension

    # Context
    block_size: int = 256     # max sequence length

    # FFN
    ffn_hidden: int = 1024    # SwiGLU intermediate size (≈ 2.67x n_embd)

    # Regularization
    dropout: float = 0.1

    # Architecture choices
    bias: bool = False        # no bias in linears (modern convention)
    rope_base: float = 10000.0  # RoPE frequency base

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0, \
            f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
        assert self.n_head % self.n_kv_head == 0, \
            f"n_head ({self.n_head}) must be divisible by n_kv_head ({self.n_kv_head})"

    @property
    def head_dim(self):
        """Dimension per attention head."""
        return self.n_embd // self.n_head

    def param_count_estimate(self):
        """Rough parameter count (excludes minor components)."""
        d = self.n_embd
        h = self.ffn_hidden
        v = self.vocab_size
        L = self.n_layer
        kv_dim = self.n_kv_head * self.head_dim

        # Per layer: attention (Q + K + V + O) + FFN (gate + up + down) + 2x RMSNorm
        attn_params = d * d + d * kv_dim + d * kv_dim + d * d  # Q, K, V, O
        ffn_params = d * h + d * h + h * d  # gate, up, down
        norm_params = d * 2  # 2 RMSNorm per layer

        per_layer = attn_params + ffn_params + norm_params
        embedding = v * d  # token embedding (shared with lm_head via weight tying)
        final_norm = d

        return L * per_layer + embedding + final_norm


# Preset configurations
TINY_CONFIG = ModelConfig(
    vocab_size=4096, n_layer=2, n_head=2, n_kv_head=1,
    n_embd=64, block_size=128, ffn_hidden=172, dropout=0.1,
)

SMALL_CONFIG = ModelConfig(
    vocab_size=4096, n_layer=6, n_head=6, n_kv_head=2,
    n_embd=384, block_size=256, ffn_hidden=1024, dropout=0.1,
)
