"""Grouped Query Attention (GQA) with RoPE — PyTorch implementation.

GQA is the modern attention variant where multiple query heads share fewer key/value heads.
This saves memory (smaller KV-cache) while maintaining nearly the same quality as full MHA.

Paper: "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"
       Ainslie et al. (2023). https://arxiv.org/abs/2305.13245

Our configuration: 6 Q heads, 2 KV heads → each KV head is shared by 3 Q heads.

Architecture of this module:
    Input x (batch, seq_len, d_model)
    ├── W_q projection → Q (batch, seq_len, n_head * head_dim)
    ├── W_k projection → K (batch, seq_len, n_kv_head * head_dim)  ← smaller
    └── W_v projection → V (batch, seq_len, n_kv_head * head_dim)  ← smaller

    Apply RoPE to Q and K (encode position via rotation)
    Expand K, V to match Q heads (repeat n_head // n_kv_head times)
    Compute attention: softmax(QK^T / √d_head) · V
    Apply causal mask (prevent attending to future tokens)
    Output projection: W_o (d_model → d_model)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from phase3_transformer.rope import apply_rope


class CausalSelfAttention(nn.Module):
    """Grouped Query Attention with RoPE and causal masking.

    Args:
        config: ModelConfig with n_head, n_kv_head, n_embd, etc.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.head_dim
        self.n_groups = config.n_head // config.n_kv_head  # Q heads per KV head

        # Projections
        # Q: full size (n_head * head_dim = n_embd)
        # K, V: smaller (n_kv_head * head_dim)
        self.w_q = nn.Linear(config.n_embd, config.n_head * self.head_dim, bias=config.bias)
        self.w_k = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=config.bias)
        self.w_v = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=config.bias)
        self.w_o = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_embd)
            freqs_cis: (seq_len, head_dim//2) — precomputed RoPE frequencies

        Returns:
            (batch, seq_len, n_embd)
        """
        B, T, C = x.shape

        # Project to Q, K, V
        q = self.w_q(x)  # (B, T, n_head * head_dim)
        k = self.w_k(x)  # (B, T, n_kv_head * head_dim)
        v = self.w_v(x)  # (B, T, n_kv_head * head_dim)

        # Reshape to (B, T, n_heads, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_kv_head, self.head_dim)
        v = v.view(B, T, self.n_kv_head, self.head_dim)

        # Apply RoPE to Q and K (not V — position only affects attention scores)
        q = apply_rope(q, freqs_cis)
        k = apply_rope(k, freqs_cis)

        # Transpose to (B, n_heads, T, head_dim) for attention computation
        q = q.transpose(1, 2)  # (B, n_head, T, head_dim)
        k = k.transpose(1, 2)  # (B, n_kv_head, T, head_dim)
        v = v.transpose(1, 2)  # (B, n_kv_head, T, head_dim)

        # Expand K, V for GQA: repeat each KV head n_groups times
        # (B, n_kv_head, T, head_dim) → (B, n_head, T, head_dim)
        if self.n_groups > 1:
            k = k.repeat_interleave(self.n_groups, dim=1)
            v = v.repeat_interleave(self.n_groups, dim=1)

        # Scaled dot-product attention with causal mask
        # PyTorch's F.scaled_dot_product_attention handles masking and scaling efficiently
        # is_causal=True automatically applies lower-triangular causal mask
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=True,
        )
        # attn_output: (B, n_head, T, head_dim)

        # Concatenate heads: (B, T, n_head * head_dim) = (B, T, n_embd)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection + dropout
        return self.resid_dropout(self.w_o(attn_output))
