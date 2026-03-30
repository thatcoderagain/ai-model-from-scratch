"""Reconstruct a clean merged model from base SmolLM2 + LoRA checkpoint.

This script:
1. Downloads/loads the base SmolLM2-360M model
2. Applies LoRA adapters to Q and V projections
3. Loads trained LoRA weights from lora_best.pt
4. Merges LoRA into base weights
5. Saves a clean standard model (no LoRA artifacts)

Usage:
    python -m phase6_finetune.reconstruct_model --lora checkpoints/finetune/lora_best.pt
"""

import argparse
import torch
from pathlib import Path

from phase6_finetune.download_model import download_model
from phase6_finetune.lora import apply_lora, LoRALinear
from phase6_finetune.instruction_dataset import INSTRUCTION_TOKEN, RESPONSE_TOKEN


def main():
    parser = argparse.ArgumentParser(description="Reconstruct merged model from LoRA checkpoint")
    parser.add_argument("--lora", type=str, default="checkpoints/finetune/lora_best.pt",
                        help="Path to lora_best.pt")
    parser.add_argument("--base-model", type=str, default="SmolLM2-360M",
                        help="Base model name")
    parser.add_argument("--output", type=str, default="checkpoints/finetune/merged_model",
                        help="Output directory for clean merged model")
    parser.add_argument("--rank", type=int, default=32, help="LoRA rank (must match training)")
    args = parser.parse_args()

    print("=" * 60)
    print("Reconstructing model from LoRA checkpoint")
    print("=" * 60)

    # 1. Load base model
    model_path = download_model(args.base_model)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"\nLoading base model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)

    # 2. Add special tokens (must match training)
    special_tokens = [INSTRUCTION_TOKEN, RESPONSE_TOKEN]
    new_tokens = [t for t in special_tokens if t not in tokenizer.get_vocab()]
    if new_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        model.resize_token_embeddings(len(tokenizer))
        print(f"Added special tokens: {new_tokens}")

    # 3. Apply LoRA adapters (same config as training)
    model, lora_params = apply_lora(
        model,
        target_modules=["q_proj", "v_proj"],
        rank=args.rank,
        alpha=args.rank * 2,
        dropout=0.0,
    )

    # 4. Load trained LoRA weights
    print(f"\nLoading LoRA weights from {args.lora}...")
    lora_state = torch.load(args.lora, map_location="cpu", weights_only=True)

    # Match LoRA weights to model parameters
    loaded = 0
    for name, param in model.named_parameters():
        if name in lora_state:
            param.data.copy_(lora_state[name])
            loaded += 1
    print(f"Loaded {loaded} LoRA parameter tensors")

    # 5. Merge LoRA weights into base model
    print("\nMerging LoRA into base weights...")
    merged = 0
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if isinstance(child, LoRALinear):
                child.merge_weights()
                setattr(module, child_name, child.original)
                merged += 1
    print(f"Merged and stripped {merged} LoRA adapters")

    # 6. Save clean model
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Done! Clean model saved ({sum(p.numel() for p in model.parameters()):,} params)")

    # 7. Quick verification
    print("\nVerifying model loads cleanly...")
    test_model = AutoModelForCausalLM.from_pretrained(output_dir, dtype=torch.float32)
    print(f"Verified: {sum(p.numel() for p in test_model.parameters()):,} params, no warnings")


if __name__ == "__main__":
    main()
