"""Fine-tune SmolLM2 with LoRA on code instruction data.

This is the script you run on the ROG SCAR 17 (RTX 3090).
It downloads SmolLM2-360M, applies LoRA adapters, trains on code instructions,
and saves the fine-tuned model.

Usage:
    # Full fine-tuning (ROG SCAR 17, ~3-5 hours)
    python -m phase6_finetune.finetune

    # Quick test (any hardware, ~2 minutes)
    python -m phase6_finetune.finetune --max-examples 100 --max-steps 50

    # Custom model
    python -m phase6_finetune.finetune --model SmolLM2-135M --rank 16

Terminology:
    Fine-tuning: Adapting a pre-trained model for a specific task by continuing
        training on task-specific data. We fine-tune SmolLM2 (general language model)
        to become better at following coding instructions.

    LoRA fine-tuning: Instead of updating all 360M parameters, we freeze them and
        only train ~3-5M LoRA adapter parameters. Same result, fraction of the compute.

    Instruction tuning (SFT): The specific type of fine-tuning where the training data
        is (instruction, response) pairs. The model learns the pattern:
        "given an instruction, produce a helpful code response."
"""

import argparse
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path

from phase6_finetune.download_model import download_model, load_pretrained
from phase6_finetune.lora import apply_lora, merge_lora_weights
from phase6_finetune.instruction_dataset import (
    InstructionDataset, load_code_instructions, collate_fn,
    INSTRUCTION_TOKEN, RESPONSE_TOKEN,
)
from phase4_training.lr_schedule import WSDScheduler


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser(description="Fine-tune SmolLM2 with LoRA")
    parser.add_argument("--model", type=str, default="SmolLM2-360M",
                        help="Model to fine-tune")
    parser.add_argument("--rank", type=int, default=32, help="LoRA rank")
    parser.add_argument("--alpha", type=float, default=None, help="LoRA alpha (default: 2*rank)")
    parser.add_argument("--max-examples", type=int, default=None, help="Limit training examples")
    parser.add_argument("--max-steps", type=int, default=3000, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=4, help="Micro-batch size")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-5, help="Peak learning rate")
    parser.add_argument("--max-length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--output", type=str, default="checkpoints/finetune", help="Output dir")
    parser.add_argument("--eval-interval", type=int, default=500, help="Eval every N steps")
    args = parser.parse_args()

    device = get_device()
    print("=" * 60)
    print("SmolLM2 LoRA Fine-Tuning")
    print("=" * 60)
    print(f"Device: {device}")

    # --- Download and load model ---
    model_path = download_model(args.model)
    model, tokenizer = load_pretrained(model_path, device=device)

    # Add special tokens if needed
    special_tokens = [INSTRUCTION_TOKEN, RESPONSE_TOKEN]
    new_tokens = [t for t in special_tokens if t not in tokenizer.get_vocab()]
    if new_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        model.resize_token_embeddings(len(tokenizer))
        print(f"Added special tokens: {new_tokens}")

    # --- Apply LoRA ---
    model, lora_params = apply_lora(
        model,
        target_modules=["q_proj", "v_proj"],
        rank=args.rank,
        alpha=args.alpha,
        dropout=0.05,
    )

    # --- Load dataset ---
    print("\nLoading instruction dataset...")
    examples = load_code_instructions(max_examples=args.max_examples)
    print(f"Total examples: {len(examples)}")

    # Train/val split
    split = int(len(examples) * 0.95)
    train_examples = examples[:split]
    val_examples = examples[split:]

    train_ds = InstructionDataset(train_examples, tokenizer, max_length=args.max_length)
    val_ds = InstructionDataset(val_examples, tokenizer, max_length=args.max_length)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    print(f"Train: {len(train_ds)} examples, Val: {len(val_ds)} examples")

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        lora_params,
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )

    scheduler = WSDScheduler.from_total_steps(
        optimizer, max_lr=args.lr, min_lr=args.lr / 10,
        total_steps=args.max_steps,
        warmup_fraction=0.05, decay_fraction=0.20,
    )

    # Mixed precision
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # --- Training loop ---
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nTraining: {args.max_steps} steps, batch={args.batch_size}×{args.grad_accum}="
          f"{args.batch_size * args.grad_accum}")
    print("-" * 60)

    model.train()
    step = 0
    best_val_loss = float("inf")
    t0 = time.time()
    train_iter = iter(train_loader)

    while step < args.max_steps:
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for _ in range(args.grad_accum):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss / args.grad_accum

            scaler.scale(loss).backward()
            accum_loss += loss.item()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        scaler.step(optimizer)
        scaler.update()
        lr = scheduler.step()

        step += 1

        if step % 50 == 0:
            dt = time.time() - t0
            print(f"Step {step:5d}/{args.max_steps} | loss {accum_loss:.4f} | lr {lr:.2e} | {dt:.0f}s")

        # --- Evaluation ---
        if step % args.eval_interval == 0:
            model.eval()
            val_loss = 0.0
            val_count = 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(device)
                    labels = batch["labels"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    val_loss += outputs.loss.item()
                    val_count += 1
                    if val_count >= 50:
                        break

            avg_val = val_loss / max(val_count, 1)
            print(f"  >> Val loss: {avg_val:.4f}")

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                # Save LoRA weights only (much smaller than full model)
                lora_state = {name: p.data.clone() for name, p in model.named_parameters() if p.requires_grad}
                torch.save(lora_state, output_dir / "lora_best.pt")
                print(f"  >> New best! Saved lora_best.pt")

            model.train()

    # --- Save final model ---
    total_time = time.time() - t0
    print("-" * 60)
    print(f"Training complete: {step} steps in {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"Best val loss: {best_val_loss:.4f}")

    # Merge LoRA into base weights and save
    merge_lora_weights(model)
    model.save_pretrained(output_dir / "merged_model")
    tokenizer.save_pretrained(output_dir / "merged_model")
    print(f"Merged model saved to {output_dir / 'merged_model'}")


if __name__ == "__main__":
    main()
