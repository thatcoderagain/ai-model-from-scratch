"""Rotary Position Embeddings (RoPE) — PyTorch implementation.

RoPE encodes position by rotating query and key vectors in 2D subspaces.
The attention score Q·K then naturally depends on RELATIVE position (i - j)
because the rotation angles subtract in the dot product.

Paper: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
       Su et al. (2021). https://arxiv.org/abs/2104.09864

Used by: Llama 1/2/3/4, Gemma, Mistral, Qwen, SmolLM — essentially all 2024+ models.

How it works:
    1. Precompute frequencies: θ_i = base^(-2i/d) for each dimension pair i
    2. For each position pos, compute angles: pos * θ_i
    3. Split vector into pairs of dimensions (2i, 2i+1)
    4. Rotate each pair by its angle using 2D rotation matrix

The key insight: Q_rotated · K_rotated depends on (pos_Q - pos_K),
making attention scores position-relative without any extra parameters.
"""

import torch


def precompute_rope_frequencies(dim: int, max_seq_len: int, base: float = 10000.0,
                                 device: torch.device | None = None) -> torch.Tensor:
    """Precompute complex exponentials for RoPE.

    Args:
        dim: head dimension (must be even)
        max_seq_len: maximum sequence length to precompute for
        base: frequency base (default 10000, from the original paper)
        device: target device

    Returns:
        freqs_cis: (max_seq_len, dim//2) complex tensor of rotation factors
            Each element is exp(i * position * θ) where θ varies by dimension.
    """
    assert dim % 2 == 0, f"RoPE requires even dimension, got {dim}"

    # Frequencies: θ_i = 1 / (base^(2i/dim)) for i = 0, 1, ..., dim/2-1
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))

    # Positions: 0, 1, 2, ..., max_seq_len-1
    positions = torch.arange(max_seq_len, device=device).float()

    # Outer product: (max_seq_len, dim//2)
    angles = torch.outer(positions, freqs)

    # Convert to complex exponential: e^(i*angle) = cos(angle) + i*sin(angle)
    # Using complex numbers makes the rotation elegant and efficient
    freqs_cis = torch.polar(torch.ones_like(angles), angles)

    return freqs_cis


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings to query or key tensor.

    Args:
        x: (batch, seq_len, n_heads, head_dim) — Q or K tensor
        freqs_cis: (seq_len, head_dim//2) — precomputed rotation factors

    Returns:
        Rotated tensor with same shape as x.

    The rotation is applied by:
        1. Viewing x as pairs of dimensions → complex numbers
        2. Multiplying by the complex rotation factor (= rotating in 2D)
        3. Converting back to real pairs
    """
    # Reshape x into complex: (batch, seq_len, n_heads, head_dim//2, 2) -> complex
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))

    # Reshape freqs_cis for broadcasting: (1, seq_len, 1, head_dim//2)
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)

    # Multiply = rotate in 2D complex plane
    x_rotated = x_complex * freqs_cis

    # Convert back to real: complex -> (batch, seq_len, n_heads, head_dim//2, 2) -> flatten
    x_out = torch.view_as_real(x_rotated).flatten(-2)

    return x_out.type_as(x)
