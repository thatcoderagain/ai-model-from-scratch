"""Tests for Phase 3: Modern Transformer Architecture.

Covers:
- Config validation
- RoPE: shape, magnitude preservation, position 0 identity
- RMSNorm: output RMS ≈ 1, shape preservation
- Attention: output shape, causal masking
- FFN: output shape
- Full model: output shape, parameter count, loss computation, causal property, generation
"""

import torch
import pytest

from phase3_transformer.config import ModelConfig, TINY_CONFIG, SMALL_CONFIG
from phase3_transformer.rope import precompute_rope_frequencies, apply_rope
from phase3_transformer.rmsnorm import RMSNorm
from phase3_transformer.attention import CausalSelfAttention
from phase3_transformer.feedforward import SwiGLUFFN
from phase3_transformer.block import TransformerBlock
from phase3_transformer.model import GPT


# ============================================================
# Config
# ============================================================

class TestConfig:
    def test_default_config_valid(self):
        config = ModelConfig()
        assert config.head_dim == 64  # 384 / 6

    def test_param_estimate(self):
        config = SMALL_CONFIG
        est = config.param_count_estimate()
        assert est > 0
        assert est < 100_000_000  # should be ~15M

    def test_invalid_head_divisibility(self):
        with pytest.raises(AssertionError):
            ModelConfig(n_embd=100, n_head=3)  # 100 not divisible by 3

    def test_invalid_kv_head_divisibility(self):
        with pytest.raises(AssertionError):
            ModelConfig(n_head=6, n_kv_head=4)  # 6 not divisible by 4


# ============================================================
# RoPE
# ============================================================

class TestRoPE:
    def test_frequencies_shape(self):
        freqs = precompute_rope_frequencies(64, 128)
        assert freqs.shape == (128, 32)  # (max_seq_len, dim//2)

    def test_preserves_magnitude(self):
        """Rotation should not change vector length."""
        freqs = precompute_rope_frequencies(64, 128)
        x = torch.randn(2, 10, 4, 64)  # (batch, seq, heads, dim)
        x_rot = apply_rope(x, freqs[:10])
        # L2 norm should be preserved per vector
        norm_before = torch.norm(x, dim=-1)
        norm_after = torch.norm(x_rot, dim=-1)
        assert torch.allclose(norm_before, norm_after, atol=1e-5)

    def test_position_zero_identity(self):
        """At position 0, rotation angle is 0 → no change."""
        freqs = precompute_rope_frequencies(64, 128)
        x = torch.randn(1, 1, 4, 64)
        x_rot = apply_rope(x, freqs[:1])
        assert torch.allclose(x, x_rot, atol=1e-5)

    def test_different_positions_differ(self):
        """Same vector at different positions should produce different results."""
        freqs = precompute_rope_frequencies(64, 128)
        x = torch.randn(1, 1, 1, 64)
        x_pos1 = apply_rope(x, freqs[1:2])
        x_pos5 = apply_rope(x, freqs[5:6])
        assert not torch.allclose(x_pos1, x_pos5)


# ============================================================
# RMSNorm
# ============================================================

class TestRMSNorm:
    def test_output_shape(self):
        norm = RMSNorm(64)
        x = torch.randn(2, 10, 64)
        assert norm(x).shape == x.shape

    def test_rms_approximately_one(self):
        """After normalization, RMS of output ≈ 1.0."""
        norm = RMSNorm(64)
        x = torch.randn(4, 16, 64) * 5
        out = norm(x)
        rms = torch.sqrt(torch.mean(out ** 2, dim=-1))
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.1)

    def test_handles_large_values(self):
        norm = RMSNorm(64)
        x = torch.randn(2, 10, 64) * 1000
        out = norm(x)
        assert torch.all(torch.isfinite(out))


# ============================================================
# Attention
# ============================================================

class TestAttention:
    @pytest.fixture
    def attn(self):
        return CausalSelfAttention(TINY_CONFIG)

    def test_output_shape(self, attn):
        x = torch.randn(2, 8, TINY_CONFIG.n_embd)
        freqs = precompute_rope_frequencies(TINY_CONFIG.head_dim, 8)
        out = attn(x, freqs)
        assert out.shape == x.shape

    def test_causal_masking(self):
        """Changing future tokens should NOT affect past outputs."""
        config = TINY_CONFIG
        attn = CausalSelfAttention(config)
        attn.eval()
        freqs = precompute_rope_frequencies(config.head_dim, 8)

        x1 = torch.randn(1, 8, config.n_embd)
        x2 = x1.clone()
        x2[:, 5:, :] = torch.randn(1, 3, config.n_embd)  # change tokens 5,6,7

        with torch.no_grad():
            out1 = attn(x1, freqs)
            out2 = attn(x2, freqs)

        # Outputs for positions 0-4 should be identical
        assert torch.allclose(out1[:, :5, :], out2[:, :5, :], atol=1e-5), \
            "Causal masking violated: future tokens affected past outputs"


# ============================================================
# FFN
# ============================================================

class TestFFN:
    def test_output_shape(self):
        ffn = SwiGLUFFN(TINY_CONFIG)
        x = torch.randn(2, 8, TINY_CONFIG.n_embd)
        assert ffn(x).shape == x.shape


# ============================================================
# TransformerBlock
# ============================================================

class TestBlock:
    def test_output_shape(self):
        block = TransformerBlock(TINY_CONFIG)
        x = torch.randn(2, 8, TINY_CONFIG.n_embd)
        freqs = precompute_rope_frequencies(TINY_CONFIG.head_dim, 8)
        assert block(x, freqs).shape == x.shape


# ============================================================
# Full Model
# ============================================================

class TestGPT:
    @pytest.fixture
    def model(self):
        return GPT(TINY_CONFIG)

    def test_output_shape(self, model):
        idx = torch.randint(0, TINY_CONFIG.vocab_size, (2, 16))
        logits, loss = model(idx)
        assert logits.shape == (2, 16, TINY_CONFIG.vocab_size)
        assert loss is None  # no targets provided

    def test_loss_with_targets(self, model):
        idx = torch.randint(0, TINY_CONFIG.vocab_size, (2, 16))
        targets = torch.randint(0, TINY_CONFIG.vocab_size, (2, 16))
        logits, loss = model(idx, targets)
        assert loss is not None
        assert loss.item() > 0
        # Random predictions on 4096 vocab → loss ≈ ln(4096) ≈ 8.3
        assert loss.item() < 12.0  # sanity check

    def test_parameter_count(self, model):
        count = model.count_parameters()
        assert count > 0
        print(f"\nTiny model: {count:,} parameters")

    def test_small_model_parameter_count(self):
        model = GPT(SMALL_CONFIG)
        count = model.count_parameters()
        # Should be roughly 15M
        assert 10_000_000 < count < 25_000_000, f"Expected ~15M params, got {count:,}"
        print(f"\nSmall model: {count:,} parameters")

    def test_causal_property(self, model):
        """Changing future tokens should not change past logits."""
        model.eval()
        x1 = torch.randint(0, TINY_CONFIG.vocab_size, (1, 16))
        x2 = x1.clone()
        x2[:, 8:] = torch.randint(0, TINY_CONFIG.vocab_size, (1, 8))

        with torch.no_grad():
            logits1, _ = model(x1)
            logits2, _ = model(x2)

        # Positions 0-7 should have identical logits
        assert torch.allclose(logits1[:, :8, :], logits2[:, :8, :], atol=1e-4), \
            "Causal property violated: future tokens affected past logits"

    def test_generate(self, model):
        model.eval()
        prompt = torch.randint(0, TINY_CONFIG.vocab_size, (1, 4))
        output = model.generate(prompt, max_new_tokens=10, temperature=1.0, top_k=10)
        assert output.shape == (1, 14)  # 4 prompt + 10 generated
        assert torch.all(output >= 0)
        assert torch.all(output < TINY_CONFIG.vocab_size)

    def test_generate_greedy(self, model):
        """Greedy generation (temperature=0) should be deterministic."""
        model.eval()
        prompt = torch.randint(0, TINY_CONFIG.vocab_size, (1, 4))
        out1 = model.generate(prompt.clone(), max_new_tokens=5, temperature=0)
        out2 = model.generate(prompt.clone(), max_new_tokens=5, temperature=0)
        assert torch.equal(out1, out2)

    def test_weight_tying(self, model):
        """Token embedding and lm_head should share the same weight tensor."""
        assert model.tok_emb.weight is model.lm_head.weight

    def test_sequence_length_limit(self, model):
        """Should raise error for sequences exceeding block_size."""
        idx = torch.randint(0, TINY_CONFIG.vocab_size, (1, TINY_CONFIG.block_size + 1))
        with pytest.raises(AssertionError):
            model(idx)

    def test_backward_pass(self, model):
        """Loss should be backpropagable."""
        idx = torch.randint(0, TINY_CONFIG.vocab_size, (2, 16))
        targets = torch.randint(0, TINY_CONFIG.vocab_size, (2, 16))
        _, loss = model(idx, targets)
        loss.backward()
        # Check that gradients exist for at least one parameter
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in model.parameters())
        assert has_grad, "No gradients computed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
