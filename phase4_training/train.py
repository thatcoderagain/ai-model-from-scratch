"""Main training script — CLI entry point.

Usage:
    # Quick smoke test (any hardware, ~30 seconds)
    python -m phase4_training.train --config phase4_training/configs/tiny.yaml

    # Full training (ROG SCAR 17 recommended)
    python -m phase4_training.train --config phase4_training/configs/small.yaml

    # Resume from checkpoint
    python -m phase4_training.train --config phase4_training/configs/small.yaml --resume checkpoints/step_005000.pt
"""

import argparse
import yaml
from pathlib import Path

import torch

from phase3_transformer.config import ModelConfig
from phase3_transformer.model import GPT
from phase2_tokenizer.bpe_tokenizer import BPETokenizer
from phase4_training.dataset import load_and_tokenize, create_datasets
from phase4_training.trainer import Trainer


def load_config(path):
    """Load YAML config file."""
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train a language model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    print("=" * 60)
    print("Local LLM Training")
    print("=" * 60)

    # --- Tokenizer ---
    tokenizer_path = Path("phase2_tokenizer/vocab")
    if tokenizer_path.exists() and (tokenizer_path / "vocab.json").exists():
        print(f"Loading tokenizer from {tokenizer_path}")
        tokenizer = BPETokenizer.load(tokenizer_path)
    else:
        print("Training a fresh tokenizer...")
        from phase2_tokenizer.train_tokenizer import SAMPLE_CORPUS
        tokenizer = BPETokenizer(vocab_size=model_cfg["vocab_size"])
        tokenizer.add_special_tokens([
            "<|begin_of_text|>", "<|end_of_text|>",
            "<|start_code|>", "<|end_code|>", "<|pad|>",
        ])
        tokenizer.train(SAMPLE_CORPUS, verbose=False)
        tokenizer.save(tokenizer_path)

    print(f"Tokenizer: {tokenizer}")

    # --- Dataset ---
    print(f"\nLoading data: {train_cfg['data_source']}")
    tokens = load_and_tokenize(
        train_cfg["data_source"],
        tokenizer,
        max_chars=train_cfg.get("max_chars"),
    )
    print(f"Total tokens: {len(tokens):,}")

    train_ds, val_ds = create_datasets(tokens, model_cfg["block_size"])

    # --- Model ---
    config = ModelConfig(
        vocab_size=model_cfg["vocab_size"],
        n_layer=model_cfg["n_layer"],
        n_head=model_cfg["n_head"],
        n_kv_head=model_cfg["n_kv_head"],
        n_embd=model_cfg["n_embd"],
        block_size=model_cfg["block_size"],
        ffn_hidden=model_cfg["ffn_hidden"],
        dropout=model_cfg["dropout"],
        bias=model_cfg.get("bias", False),
    )

    model = GPT(config)
    print(f"\nModel: {config.n_layer} layers, {model.count_parameters():,} parameters")
    print(f"Estimated: {config.param_count_estimate():,} (from config)")

    # --- Trainer ---
    trainer_config = {
        "max_steps": train_cfg["max_steps"],
        "micro_batch_size": train_cfg["micro_batch_size"],
        "grad_accum_steps": train_cfg["grad_accum_steps"],
        "max_lr": train_cfg["max_lr"],
        "min_lr": train_cfg.get("min_lr", train_cfg["max_lr"] / 10),
        "weight_decay": train_cfg.get("weight_decay", 0.1),
        "grad_clip": train_cfg.get("grad_clip", 1.0),
        "warmup_fraction": train_cfg.get("warmup_fraction", 0.05),
        "decay_fraction": train_cfg.get("decay_fraction", 0.20),
        "log_interval": train_cfg.get("log_interval", 50),
        "eval_interval": train_cfg.get("eval_interval", 500),
        "checkpoint_interval": train_cfg.get("checkpoint_interval", 1000),
        "block_size": model_cfg["block_size"],
    }

    trainer = Trainer(model, train_ds, val_ds, trainer_config)

    # Resume if requested
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # --- Train ---
    print()
    trainer.train()

    # --- Generate samples ---
    print("\n" + "=" * 60)
    print("Sample Generation")
    print("=" * 60)
    model.eval()

    prompts = ["Once upon a time", "The little", "One day"]
    for prompt_text in prompts:
        prompt_ids = tokenizer.encode(prompt_text)
        prompt_tensor = torch.tensor([prompt_ids], device=trainer.device)
        output = model.generate(prompt_tensor, max_new_tokens=50, temperature=0.8, top_k=40)
        generated = tokenizer.decode(output[0].tolist())
        print(f"\n  Prompt: \"{prompt_text}\"")
        print(f"  Generated: \"{generated}\"")


if __name__ == "__main__":
    main()
