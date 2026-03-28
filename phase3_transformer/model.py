"""Complete GPT-style Language Model — Llama/Gemma architecture.

This is the full model that assembles all components:
    Token Embeddings → N × TransformerBlock → RMSNorm → Language Model Head

Architecture (modern Llama-style, NOT GPT-2):
    - Token embedding (no positional embedding — RoPE handles position)
    - N stacked TransformerBlocks (each: RMSNorm → GQA → residual → RMSNorm → SwiGLU → residual)
    - Final RMSNorm
    - Linear head projecting to vocab_size (weight-tied with token embedding)

Weight tying (Press & Wolf, 2017):
    The token embedding matrix and the output projection (lm_head) share the same weights.
    This makes sense because embedding maps token→vector and lm_head maps vector→token —
    they're inverse operations. Sharing weights reduces parameter count and improves quality.

The model is decoder-only (autoregressive):
    - Input: sequence of token IDs
    - Output: logits (scores) for the next token at each position
    - Training: predict the next token at every position (teacher forcing)
    - Inference: generate one token at a time, feeding output back as input
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from phase3_transformer.config import ModelConfig
from phase3_transformer.rmsnorm import RMSNorm
from phase3_transformer.block import TransformerBlock
from phase3_transformer.rope import precompute_rope_frequencies


class GPT(nn.Module):
    """Decoder-only transformer language model (Llama-style architecture).

    Args:
        config: ModelConfig with all hyperparameters
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Token embedding: token_id → vector of size n_embd
        # No positional embedding — RoPE encodes position via rotation in attention
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)

        # Dropout on embeddings (regularization)
        self.drop = nn.Dropout(config.dropout)

        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layer)
        ])

        # Final normalization before output projection
        self.norm = RMSNorm(config.n_embd)

        # Language model head: project from n_embd → vocab_size to get next-token logits
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: share embedding and output projection weights
        # This means tok_emb.weight and lm_head.weight point to the SAME tensor
        self.lm_head.weight = self.tok_emb.weight

        # Precompute RoPE frequencies (stored as buffer — not a parameter, but saved with model)
        freqs_cis = precompute_rope_frequencies(
            config.head_dim, config.block_size, config.rope_base
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using standard practices for transformers.

        - Linear layers: normal distribution with std=0.02
        - Embedding: normal distribution with std=0.02
        - Output projections (w_o, w_down): scaled by 1/√(2*n_layer) to account for
          residual accumulation. Without this, the residual sum grows with depth.
          From GPT-2 paper, Section 2.3.
        """
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                std = 0.02
                # Scale down output projections of attention and FFN
                if name.endswith("w_o") or name.endswith("w_down"):
                    std *= (2 * self.config.n_layer) ** -0.5
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """Forward pass.

        Args:
            idx: (batch, seq_len) — input token IDs
            targets: (batch, seq_len) — target token IDs (optional, for computing loss)

        Returns:
            logits: (batch, seq_len, vocab_size) — next-token predictions
            loss: scalar tensor if targets provided, else None
        """
        B, T = idx.shape
        assert T <= self.config.block_size, \
            f"Sequence length {T} exceeds block_size {self.config.block_size}"

        # Token embeddings
        x = self.tok_emb(idx)  # (B, T, n_embd)
        x = self.drop(x)

        # Get RoPE frequencies for this sequence length
        freqs_cis = self.freqs_cis[:T]

        # Pass through all transformer blocks
        for block in self.blocks:
            x = block(x, freqs_cis)

        # Final normalization
        x = self.norm(x)

        # Compute logits
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),  # (B*T, vocab_size)
                targets.view(-1),                    # (B*T,)
            )

        return logits, loss

    def count_parameters(self, trainable_only=True):
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: int | None = None) -> torch.Tensor:
        """Generate tokens autoregressively.

        Simple generation loop (no KV-cache — we'll add that in Phase 5).

        Args:
            idx: (batch, seq_len) — conditioning tokens
            max_new_tokens: how many new tokens to generate
            temperature: >1.0 = more random, <1.0 = more deterministic, 0 = greedy
            top_k: if set, only sample from top k most likely tokens
        """
        for _ in range(max_new_tokens):
            # Crop to block_size if needed
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

            # Forward pass
            logits, _ = self(idx_cond)

            # Get logits for the last position only
            logits = logits[:, -1, :]  # (B, vocab_size)

            # Apply temperature
            if temperature > 0:
                logits = logits / temperature
            else:
                # Greedy: pick the most likely token
                idx_next = logits.argmax(dim=-1, keepdim=True)
                idx = torch.cat([idx, idx_next], dim=1)
                continue

            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Sample from the distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append
            idx = torch.cat([idx, idx_next], dim=1)

        return idx
