"""Transformer Decoder Block — one layer of the transformer.

Each block has two sub-layers with residual connections and pre-normalization:

    x → RMSNorm → Attention → + x (residual)
      → RMSNorm → FFN       → + x (residual)

Key design choices (all from 2024-2025 best practices):

    Pre-norm (normalize BEFORE sublayer, not after):
        GPT-2 used post-norm: x → Attention → LayerNorm → ...
        Modern models use pre-norm: x → RMSNorm → Attention → ...
        Pre-norm is more training-stable because gradients flow cleanly through
        the residual path without being modified by normalization.

    Residual connections (x + sublayer(x)):
        Without these, gradients must flow through every layer's transformations
        to reach early layers — they vanish. With residuals, gradients have a
        direct "highway" from the loss back to early layers.
        From He et al. (2015), "Deep Residual Learning for Image Recognition".

    RMSNorm instead of LayerNorm:
        Simpler (no mean subtraction), faster, same quality. See rmsnorm.py.

A full transformer model stacks N of these blocks (our model uses N=6).
"""

import torch
import torch.nn as nn

from phase3_transformer.rmsnorm import RMSNorm
from phase3_transformer.attention import CausalSelfAttention
from phase3_transformer.feedforward import SwiGLUFFN


class TransformerBlock(nn.Module):
    """One transformer decoder block: attention + FFN with residuals.

    Args:
        config: ModelConfig
    """

    def __init__(self, config):
        super().__init__()
        self.norm1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.norm2 = RMSNorm(config.n_embd)
        self.ffn = SwiGLUFFN(config)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_embd)
            freqs_cis: (seq_len, head_dim//2) — RoPE frequencies
        Returns:
            (batch, seq_len, n_embd)
        """
        # Attention with pre-norm and residual
        x = x + self.attn(self.norm1(x), freqs_cis)

        # FFN with pre-norm and residual
        x = x + self.ffn(self.norm2(x))

        return x
