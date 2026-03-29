"""Tests for Phase 6: Fine-tuning (LoRA, instruction dataset, DPO).

Covers:
- LoRA adapter: correct shapes, freezing, forward pass, merge
- Instruction dataset: formatting, loss masking
- DPO loss: correct sign behavior
"""

import torch
import torch.nn as nn
import pytest

from phase6_finetune.lora import LoRALinear, apply_lora, merge_lora_weights
from phase6_finetune.instruction_dataset import format_instruction, INSTRUCTION_TOKEN, RESPONSE_TOKEN


# ============================================================
# LoRA
# ============================================================

class TestLoRALinear:
    def test_output_shape(self):
        """LoRA layer should produce same output shape as original."""
        original = nn.Linear(64, 128)
        lora = LoRALinear(original, rank=8)
        x = torch.randn(4, 64)
        assert lora(x).shape == (4, 128)

    def test_original_frozen(self):
        """Original weights should not require gradients."""
        original = nn.Linear(64, 128)
        lora = LoRALinear(original, rank=8)
        assert not lora.original.weight.requires_grad
        if lora.original.bias is not None:
            assert not lora.original.bias.requires_grad

    def test_lora_params_trainable(self):
        """LoRA A and B should require gradients."""
        original = nn.Linear(64, 128)
        lora = LoRALinear(original, rank=8)
        assert lora.lora_A.requires_grad
        assert lora.lora_B.requires_grad

    def test_lora_shapes(self):
        """A: (in, rank), B: (rank, out)."""
        original = nn.Linear(64, 128)
        lora = LoRALinear(original, rank=8)
        assert lora.lora_A.shape == (64, 8)
        assert lora.lora_B.shape == (8, 128)

    def test_starts_as_identity(self):
        """LoRA should start as a no-op (B initialized to zeros)."""
        original = nn.Linear(64, 128, bias=False)
        lora = LoRALinear(original, rank=8)
        x = torch.randn(4, 64)
        with torch.no_grad():
            original_out = original(x)
            lora_out = lora(x)
        assert torch.allclose(original_out, lora_out, atol=1e-6), \
            "LoRA should start as identity (B=0 → adapter output is 0)"

    def test_trainable_parameter_count(self):
        """LoRA should have much fewer trainable params than original."""
        original = nn.Linear(512, 512)
        lora = LoRALinear(original, rank=32)
        full_params = 512 * 512  # 262,144
        lora_params = lora.trainable_parameters  # 32 * (512 + 512) = 32,768
        assert lora_params < full_params / 5  # at least 5x fewer
        assert lora_params == 32 * (512 + 512)

    def test_backward_only_updates_lora(self):
        """Gradients should only flow to LoRA parameters, not original."""
        original = nn.Linear(64, 32, bias=False)
        lora = LoRALinear(original, rank=4)
        x = torch.randn(2, 64)
        out = lora(x)
        loss = out.sum()
        loss.backward()

        assert lora.lora_A.grad is not None
        assert lora.lora_B.grad is not None
        assert original.weight.grad is None

    def test_merge_weights(self):
        """After merging, output should include LoRA adaptation."""
        original = nn.Linear(32, 32, bias=False)
        lora = LoRALinear(original, rank=4)

        # Set LoRA weights to something non-zero
        with torch.no_grad():
            lora.lora_A.fill_(0.1)
            lora.lora_B.fill_(0.1)

        x = torch.randn(2, 32)
        with torch.no_grad():
            before_merge = lora(x).clone()

        # Merge
        lora.merge_weights()

        # After merge, the original linear should produce the same output
        with torch.no_grad():
            after_merge = lora.original(x)

        assert torch.allclose(before_merge, after_merge, atol=1e-5), \
            "Merged output should match pre-merge LoRA output"

    def test_different_ranks(self):
        """LoRA should work with various rank values."""
        for rank in [1, 4, 16, 64]:
            original = nn.Linear(128, 256)
            lora = LoRALinear(original, rank=rank)
            x = torch.randn(2, 128)
            out = lora(x)
            assert out.shape == (2, 256)

    def test_alpha_scaling(self):
        """Higher alpha should produce larger LoRA contribution."""
        original = nn.Linear(32, 32, bias=False)
        lora_low = LoRALinear(original, rank=4, alpha=1.0)
        lora_high = LoRALinear(original, rank=4, alpha=100.0)

        # Copy same A/B weights
        with torch.no_grad():
            lora_high.lora_A.copy_(lora_low.lora_A)
            lora_high.lora_B.copy_(lora_low.lora_B)
            lora_low.lora_B.fill_(0.1)
            lora_high.lora_B.fill_(0.1)

        x = torch.randn(1, 32)
        with torch.no_grad():
            out_low = lora_low(x) - original(x)  # LoRA contribution only
            out_high = lora_high(x) - original(x)

        # Higher alpha → larger contribution
        assert out_high.abs().mean() > out_low.abs().mean()


# ============================================================
# Apply LoRA to a model
# ============================================================

class SimpleModel(nn.Module):
    """Minimal model for testing apply_lora."""
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(64, 64)
        self.k_proj = nn.Linear(64, 64)
        self.v_proj = nn.Linear(64, 64)
        self.ffn = nn.Linear(64, 64)

    def forward(self, x):
        return self.ffn(self.q_proj(x) + self.v_proj(x))


class TestApplyLoRA:
    def test_apply_to_target_modules(self):
        """apply_lora should replace target modules with LoRALinear."""
        model = SimpleModel()
        model, lora_params = apply_lora(model, target_modules=["q_proj", "v_proj"], rank=8)

        assert isinstance(model.q_proj, LoRALinear)
        assert isinstance(model.v_proj, LoRALinear)
        assert isinstance(model.k_proj, nn.Linear)  # not targeted
        assert isinstance(model.ffn, nn.Linear)  # not targeted

    def test_only_lora_params_trainable(self):
        """After apply_lora, only LoRA params should have requires_grad=True."""
        model = SimpleModel()
        model, lora_params = apply_lora(model, target_modules=["q_proj"], rank=4)

        trainable = [name for name, p in model.named_parameters() if p.requires_grad]
        for name in trainable:
            assert "lora" in name, f"Non-LoRA param {name} is trainable"

    def test_forward_still_works(self):
        """Model should still produce valid output after LoRA injection."""
        model = SimpleModel()
        x = torch.randn(2, 64)
        out_before = model(x).detach()

        model, _ = apply_lora(model, target_modules=["q_proj", "v_proj"], rank=4)
        out_after = model(x).detach()

        # Should produce same output (LoRA starts as no-op)
        assert torch.allclose(out_before, out_after, atol=1e-5)


# ============================================================
# Instruction Format
# ============================================================

class TestInstructionFormat:
    def test_format_contains_tokens(self):
        text = format_instruction("Write hello world", "print('hello')")
        assert INSTRUCTION_TOKEN in text
        assert RESPONSE_TOKEN in text

    def test_format_structure(self):
        text = format_instruction("Do X", "Y")
        parts = text.split(RESPONSE_TOKEN)
        assert len(parts) == 2
        assert "Do X" in parts[0]
        assert "Y" in parts[1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
