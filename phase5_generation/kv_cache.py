"""KV-Cache for fast autoregressive generation.

Terminology:
    KV-Cache: During text generation, the model processes tokens one at a time.
        Without caching, generating token N requires recomputing attention for ALL
        previous N-1 tokens (O(N²) total work). With KV-cache, we store the K and V
        tensors from previous tokens and only compute the NEW token's Q, K, V.
        This reduces generation from O(N²) to O(N) total work.

    Prefill: The initial phase where we process the entire prompt at once.
        This populates the cache with K/V for all prompt tokens.

    Decode: After prefill, we generate one token at a time. Each step:
        1. Compute Q, K, V for just the new token
        2. Append new K, V to the cache
        3. Compute attention: new Q attends to ALL cached K/V
        4. This gives us the logits for the next token

    Why GQA helps: With GQA (2 KV heads instead of 6), the cache is 3x smaller.
        For long sequences, the KV-cache is the main memory bottleneck.

Implementation:
    We modify the attention module to accept and return cache tensors.
    The cache is a list of (K, V) tuples, one per layer.
"""

import torch
from dataclasses import dataclass


@dataclass
class KVCache:
    """Stores cached K and V tensors for all layers.

    Shape per layer:
        k_cache: (batch, n_kv_heads, cached_seq_len, head_dim)
        v_cache: (batch, n_kv_heads, cached_seq_len, head_dim)
    """
    k_caches: list[torch.Tensor]  # one per layer
    v_caches: list[torch.Tensor]  # one per layer

    @classmethod
    def empty(cls, n_layers):
        """Create an empty cache (no tokens cached yet)."""
        return cls(k_caches=[None] * n_layers, v_caches=[None] * n_layers)

    def update(self, layer_idx, new_k, new_v):
        """Append new K/V to cache for a specific layer.

        Args:
            layer_idx: which transformer layer
            new_k: (batch, n_kv_heads, new_seq_len, head_dim)
            new_v: (batch, n_kv_heads, new_seq_len, head_dim)

        Returns:
            full_k, full_v: concatenated with cache
        """
        if self.k_caches[layer_idx] is None:
            self.k_caches[layer_idx] = new_k
            self.v_caches[layer_idx] = new_v
        else:
            self.k_caches[layer_idx] = torch.cat([self.k_caches[layer_idx], new_k], dim=2)
            self.v_caches[layer_idx] = torch.cat([self.v_caches[layer_idx], new_v], dim=2)

        return self.k_caches[layer_idx], self.v_caches[layer_idx]

    @property
    def seq_len(self):
        """Current number of cached tokens."""
        if self.k_caches[0] is None:
            return 0
        return self.k_caches[0].shape[2]
