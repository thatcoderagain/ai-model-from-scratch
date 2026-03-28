"""SwiGLU Feed-Forward Network — PyTorch implementation.

The FFN processes each token independently AFTER attention has mixed information
across the sequence. Think of attention as "gathering relevant context" and FFN as
"processing that context to make decisions."

SwiGLU architecture (Llama/Gemma style):
    gate = Swish(x @ W_gate)     — what information to let through (learned filter)
    up   = x @ W_up              — the information content
    hidden = gate * up            — element-wise gating (filtering)
    output = hidden @ W_down      — project back to model dimension

Paper: "GLU Variants Improve Transformer"
       Shazeer (2020). https://arxiv.org/abs/2002.05202

Comparison with GPT-2 FFN:
    GPT-2:  GELU(x @ W1 + b1) @ W2 + b2    — 2 matrices + 2 biases
    SwiGLU: (Swish(x @ W_gate) * (x @ W_up)) @ W_down  — 3 matrices, no bias

    SwiGLU uses 3 matrices instead of 2, so the intermediate dimension is reduced
    from 4x to ~2.67x to keep total parameters similar. Despite this, SwiGLU
    performs 1-2% better in practice due to the gating mechanism.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network.

    Args:
        config: ModelConfig with n_embd, ffn_hidden, bias, dropout
    """

    def __init__(self, config):
        super().__init__()
        # Three projections (no bias in modern architectures)
        self.w_gate = nn.Linear(config.n_embd, config.ffn_hidden, bias=config.bias)
        self.w_up = nn.Linear(config.n_embd, config.ffn_hidden, bias=config.bias)
        self.w_down = nn.Linear(config.ffn_hidden, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_embd)
        Returns:
            (batch, seq_len, n_embd)
        """
        # Gate: what to let through (Swish activation = SiLU in PyTorch)
        gate = F.silu(self.w_gate(x))

        # Up: the content to be filtered
        up = self.w_up(x)

        # Element-wise gating
        hidden = gate * up

        # Project back to model dimension
        return self.dropout(self.w_down(hidden))
