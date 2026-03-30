"""LoRA (Low-Rank Adaptation) — built from scratch.

Terminology:
    LoRA: A parameter-efficient fine-tuning technique. Instead of updating ALL weights
        in the model, we freeze the original weights and add small "adapter" matrices.
        This trains ~1-2% of the parameters while adapting the full model's behavior.

    Paper: "LoRA: Low-Rank Adaptation of Large Language Models"
           Hu et al. (2021). https://arxiv.org/abs/2106.09685

    How it works:
        Original: y = x @ W           (W is frozen, shape: d_in × d_out)
        LoRA:     y = x @ W + x @ A @ B   (A and B are trainable)

        A has shape (d_in, r) and B has shape (r, d_out), where r << d_in, d_out.
        This means instead of training d_in × d_out parameters, we only train
        d_in × r + r × d_out = r × (d_in + d_out) parameters.

        For d_in = d_out = 512 and r = 32:
            Full:  512 × 512 = 262,144 parameters
            LoRA:  32 × (512 + 512) = 32,768 parameters (8x fewer)

    Rank (r): The bottleneck dimension. Controls the capacity of the adapter.
        Higher rank = more capacity = more parameters. 2025 research suggests r=32-64
        is optimal for most tasks (Optimal Rank Selection, arXiv:2512.15634).

    Alpha (α): A scaling factor applied to the LoRA output: y = x @ W + (α/r) × x @ A @ B.
        This controls the magnitude of the adaptation relative to the original weights.
        Typical: α = 2r (so the scaling factor is 2.0).

    Which layers to adapt: Standard practice is to apply LoRA to the attention Q and V
        projections. Some methods also add it to K, O, gate, up, down — but Q+V alone
        captures most of the benefit with fewer parameters.

    Initialization:
        A: initialized from normal distribution (N(0, σ²))
        B: initialized to zeros
        This means LoRA starts as a no-op (A @ B = 0), and the model begins from
        the pre-trained weights. Training gradually introduces the adaptation.
"""

import torch
import torch.nn as nn
import math


class LoRALinear(nn.Module):
    """A linear layer augmented with a LoRA adapter.

    Computes: y = x @ W + (alpha/r) * x @ A @ B

    W is frozen (original pre-trained weights).
    A and B are trainable (the LoRA adapter).

    Args:
        original_linear: nn.Linear to adapt (weights are frozen)
        rank: LoRA rank (bottleneck dimension)
        alpha: scaling factor (default: 2 * rank)
        dropout: dropout on LoRA path
    """

    def __init__(self, original_linear: nn.Linear, rank: int = 32,
                 alpha: float | None = None, dropout: float = 0.0):
        super().__init__()
        self.original = original_linear
        self.rank = rank
        self.alpha = alpha if alpha is not None else 2.0 * rank
        self.scaling = self.alpha / self.rank

        in_features = original_linear.in_features
        out_features = original_linear.out_features

        # Freeze original weights
        for param in self.original.parameters():
            param.requires_grad = False

        # LoRA adapter matrices — create on the same device as the original weights
        # A: projects DOWN to low rank (in_features → rank)
        # B: projects UP from low rank (rank → out_features)
        device = original_linear.weight.device
        dtype = original_linear.weight.dtype
        self.lora_A = nn.Parameter(torch.empty(in_features, rank, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features, device=device, dtype=dtype))

        # Initialize A with scaled normal distribution
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # Optional dropout on LoRA path
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original frozen path
        original_out = self.original(x)

        # LoRA adapter path
        lora_out = self.lora_dropout(x) @ self.lora_A @ self.lora_B * self.scaling

        return original_out + lora_out

    def merge_weights(self):
        """Merge LoRA weights into the original linear layer.

        After fine-tuning, we can merge A @ B into W so there's no inference overhead.
        W_merged = W + (alpha/r) * A @ B
        """
        with torch.no_grad():
            merged = self.lora_A @ self.lora_B * self.scaling
            self.original.weight.data += merged.T  # transpose because nn.Linear stores W.T

    @property
    def trainable_parameters(self):
        return self.lora_A.numel() + self.lora_B.numel()


def apply_lora(model, target_modules=None, rank=32, alpha=None, dropout=0.0):
    """Apply LoRA adapters to specific layers of a HuggingFace model.

    Args:
        model: HuggingFace model (e.g., SmolLM2)
        target_modules: list of module name patterns to adapt (default: Q and V projections)
        rank: LoRA rank
        alpha: scaling factor
        dropout: dropout rate

    Returns:
        model: modified model with LoRA adapters
        lora_params: list of trainable LoRA parameters
    """
    if target_modules is None:
        # Default: apply to Q and V projections in attention
        # SmolLM2/Llama naming: q_proj, v_proj
        target_modules = ["q_proj", "v_proj"]

    lora_params = []
    replaced = 0

    for name, module in model.named_modules():
        # Check if this module matches any target pattern
        if not any(target in name for target in target_modules):
            continue

        if not isinstance(module, nn.Linear):
            continue

        # Create LoRA wrapper
        lora_layer = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)

        # Replace the module in the parent
        parent_name = ".".join(name.split(".")[:-1])
        child_name = name.split(".")[-1]
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, lora_layer)

        lora_params.extend([lora_layer.lora_A, lora_layer.lora_B])
        replaced += 1

    # Freeze all non-LoRA parameters
    for param in model.parameters():
        param.requires_grad = False
    for param in lora_params:
        param.requires_grad = True

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in lora_params)
    print(f"LoRA applied to {replaced} layers (rank={rank}, alpha={alpha or 2*rank})")
    print(f"  Total params:     {total_params:,}")
    print(f"  Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")

    return model, lora_params


def merge_lora_weights(model):
    """Merge all LoRA adapters back into the base weights.

    After this, the model runs at normal speed with no LoRA overhead.
    Call this after fine-tuning, before saving for deployment.
    """
    merged = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge_weights()
            merged += 1
    print(f"Merged {merged} LoRA adapters into base weights")
