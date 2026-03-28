"""RMS Normalization — PyTorch implementation.

RMSNorm normalizes by the root mean square of the input, then scales by a learned parameter γ.
Simpler and faster than LayerNorm (no mean subtraction, no bias β).

Paper: "Root Mean Square Layer Normalization"
       Zhang & Sennrich (2019). https://arxiv.org/abs/1910.07467

Formula:
    RMSNorm(x) = x / RMS(x) * γ
    where RMS(x) = sqrt(mean(x²) + ε)

Why RMSNorm over LayerNorm:
    - 7-64% faster (fewer operations: no mean subtraction, no β)
    - Equivalent or better performance in practice
    - Used by Llama, Gemma, Mistral, SmolLM, and all modern LLMs
    - The mean subtraction in LayerNorm turns out to be unnecessary for training stability
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Normalization.

    Args:
        dim: feature dimension to normalize over (last dimension)
        eps: small constant for numerical stability (prevents division by zero)
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # γ (gamma): learned scaling parameter, one per feature dimension
        # Initialized to 1.0 so the layer starts as a no-op (identity)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute RMS along last dimension
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # Normalize and scale
        return (x / rms) * self.weight
