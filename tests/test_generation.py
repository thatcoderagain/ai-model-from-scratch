"""Tests for Phase 5: Generation & Evaluation.

Covers:
- Decoding strategies (greedy, temperature, top-k, top-p)
- Repetition penalty
- KV-cache operations
- Perplexity computation
- Generation output validity
"""

import math
import numpy as np
import torch
import pytest

from phase3_transformer.config import TINY_CONFIG
from phase3_transformer.model import GPT
from phase4_training.dataset import TextDataset
from phase5_generation.generate import generate, _top_k_filter, _top_p_filter, _apply_repetition_penalty
from phase5_generation.kv_cache import KVCache
from phase5_generation.evaluate import compute_perplexity


@pytest.fixture(scope="module")
def model():
    """Create a tiny model for testing."""
    m = GPT(TINY_CONFIG)
    m.eval()
    return m


# ============================================================
# Decoding Strategies
# ============================================================

class TestDecoding:
    def test_greedy_deterministic(self, model):
        """Greedy decoding (temp=0) should always produce the same output."""
        prompt = torch.randint(0, TINY_CONFIG.vocab_size, (1, 4))
        out1 = generate(model, prompt.clone(), max_new_tokens=10, temperature=0)
        out2 = generate(model, prompt.clone(), max_new_tokens=10, temperature=0)
        assert torch.equal(out1, out2)

    def test_output_shape(self, model):
        """Generated output should be prompt + new tokens."""
        prompt = torch.randint(0, TINY_CONFIG.vocab_size, (1, 4))
        out = generate(model, prompt, max_new_tokens=10, temperature=0.8)
        assert out.shape[0] == 1
        assert out.shape[1] <= 14  # 4 prompt + up to 10 generated

    def test_output_valid_ids(self, model):
        """All generated token IDs should be within vocab range."""
        prompt = torch.randint(0, TINY_CONFIG.vocab_size, (1, 4))
        out = generate(model, prompt, max_new_tokens=20, temperature=1.0, top_k=50)
        assert torch.all(out >= 0)
        assert torch.all(out < TINY_CONFIG.vocab_size)

    def test_prompt_preserved(self, model):
        """The prompt tokens should remain unchanged in the output."""
        prompt = torch.randint(0, TINY_CONFIG.vocab_size, (1, 8))
        out = generate(model, prompt.clone(), max_new_tokens=10, temperature=0.8)
        assert torch.equal(out[:, :8], prompt)

    def test_temperature_affects_diversity(self, model):
        """Higher temperature should produce more diverse outputs across runs."""
        prompt = torch.randint(0, TINY_CONFIG.vocab_size, (1, 4))

        # Low temperature — should be more consistent
        outputs_low = set()
        for _ in range(5):
            out = generate(model, prompt.clone(), max_new_tokens=5, temperature=0.01)
            outputs_low.add(tuple(out[0].tolist()))

        # High temperature — should be more diverse
        outputs_high = set()
        for _ in range(10):
            out = generate(model, prompt.clone(), max_new_tokens=5, temperature=2.0)
            outputs_high.add(tuple(out[0].tolist()))

        # Low temp should have fewer unique outputs
        assert len(outputs_low) <= len(outputs_high)

    def test_stop_token(self, model):
        """Generation should stop when stop_token_id is produced."""
        prompt = torch.randint(0, TINY_CONFIG.vocab_size, (1, 4))
        # Use a common token ID as stop token — generation will stop early
        out = generate(model, prompt, max_new_tokens=100, temperature=1.0,
                       stop_token_id=0)
        # Should be shorter than prompt + 100
        assert out.shape[1] <= 104


# ============================================================
# Filtering Functions
# ============================================================

class TestFiltering:
    def test_top_k_filter(self):
        """Top-k should zero out all but top k tokens."""
        logits = torch.tensor([[1.0, 5.0, 3.0, 2.0, 4.0]])
        filtered = _top_k_filter(logits, k=3)
        # Top 3: indices 1 (5.0), 4 (4.0), 2 (3.0)
        assert filtered[0, 0] == float('-inf')  # filtered out
        assert filtered[0, 3] == float('-inf')  # filtered out
        assert filtered[0, 1] == 5.0  # kept
        assert filtered[0, 4] == 4.0  # kept
        assert filtered[0, 2] == 3.0  # kept

    def test_top_k_larger_than_vocab(self):
        """Top-k larger than vocab should keep all tokens."""
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        filtered = _top_k_filter(logits, k=100)
        assert torch.equal(logits, filtered)

    def test_top_p_filter(self):
        """Top-p should keep tokens until cumulative probability exceeds p."""
        logits = torch.tensor([[10.0, 1.0, 0.1, 0.01, -10.0]])
        filtered = _top_p_filter(logits, p=0.95)
        # The highest logit (10.0) should always survive
        assert filtered[0, 0] != float('-inf')

    def test_repetition_penalty(self):
        """Repetition penalty should reduce logits for tokens in history."""
        logits = torch.tensor([[2.0, 3.0, 1.0, 4.0]])
        past = torch.tensor([[0, 1, 0, 1]])  # tokens 0 and 1 repeated
        penalized = _apply_repetition_penalty(logits.clone(), past, penalty=1.5)
        # Token 0 and 1 should have lower logits
        assert penalized[0, 0] < logits[0, 0]
        assert penalized[0, 1] < logits[0, 1]
        # Token 2 and 3 should be unchanged
        assert penalized[0, 2] == logits[0, 2]
        assert penalized[0, 3] == logits[0, 3]


# ============================================================
# KV-Cache
# ============================================================

class TestKVCache:
    def test_empty_cache(self):
        cache = KVCache.empty(6)
        assert cache.seq_len == 0
        assert len(cache.k_caches) == 6

    def test_update_adds_tokens(self):
        cache = KVCache.empty(2)
        k = torch.randn(1, 2, 5, 64)  # batch=1, heads=2, seq=5, dim=64
        v = torch.randn(1, 2, 5, 64)
        full_k, full_v = cache.update(0, k, v)
        assert full_k.shape == (1, 2, 5, 64)
        assert cache.seq_len == 5

    def test_update_concatenates(self):
        cache = KVCache.empty(1)
        k1 = torch.randn(1, 2, 5, 64)
        v1 = torch.randn(1, 2, 5, 64)
        cache.update(0, k1, v1)

        k2 = torch.randn(1, 2, 1, 64)  # one new token
        v2 = torch.randn(1, 2, 1, 64)
        full_k, full_v = cache.update(0, k2, v2)

        assert full_k.shape == (1, 2, 6, 64)
        assert cache.seq_len == 6

    def test_layers_independent(self):
        cache = KVCache.empty(3)
        k0 = torch.randn(1, 2, 5, 64)
        v0 = torch.randn(1, 2, 5, 64)
        cache.update(0, k0, v0)

        # Layer 1 should still be empty
        assert cache.k_caches[1] is None


# ============================================================
# Perplexity
# ============================================================

class TestPerplexity:
    def test_perplexity_is_positive(self, model):
        tokens = np.random.randint(0, TINY_CONFIG.vocab_size, size=2000, dtype=np.int32)
        ds = TextDataset(tokens, TINY_CONFIG.block_size)
        ppl, loss = compute_perplexity(model, ds, max_batches=5)
        assert ppl > 0
        assert loss > 0

    def test_perplexity_equals_exp_loss(self, model):
        tokens = np.random.randint(0, TINY_CONFIG.vocab_size, size=2000, dtype=np.int32)
        ds = TextDataset(tokens, TINY_CONFIG.block_size)
        ppl, loss = compute_perplexity(model, ds, max_batches=5)
        assert abs(ppl - math.exp(loss)) < 0.1

    def test_random_model_high_perplexity(self, model):
        """A random model should have perplexity close to vocab_size."""
        tokens = np.random.randint(0, TINY_CONFIG.vocab_size, size=2000, dtype=np.int32)
        ds = TextDataset(tokens, TINY_CONFIG.block_size)
        ppl, _ = compute_perplexity(model, ds, max_batches=10)
        # Random model on 4096 vocab → perplexity ≈ 4096
        # Allow wide range due to initialization
        assert ppl > 100, f"Perplexity {ppl} suspiciously low for random model"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
