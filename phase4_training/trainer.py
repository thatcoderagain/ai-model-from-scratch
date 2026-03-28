"""Training loop — supports CUDA (RTX 3090) and MPS (MacBook M4).

Terminology:
    Mixed precision (fp16/bf16): Use 16-bit floats for forward/backward pass instead
        of 32-bit. Halves memory usage and doubles speed on CUDA GPUs. The optimizer
        still uses fp32 for numerical stability (this is called "mixed" precision).
        Not reliably supported on Apple MPS, so we use fp32 there.

    Gradient accumulation: Simulate larger batch sizes by accumulating gradients over
        multiple micro-batches before doing one optimizer step. If micro_batch=8 and
        grad_accum=8, the effective batch size is 64. Essential for training on limited
        memory — we can't fit batch_size=64 in one forward pass.

    Gradient clipping (max_norm): Cap the total gradient magnitude to prevent
        "exploding gradients" — when gradients become huge and training destabilizes.
        Standard practice: clip to max_norm=1.0.

    Checkpoint: Save model weights + optimizer state + step count to disk.
        Allows resuming training after interruption. We save to CPU tensors
        so checkpoints work across CUDA and MPS devices.

    Tokens per second: Key throughput metric. Higher = faster training.
        RTX 3090 fp16: ~50K-100K tok/s. M4 MPS fp32: ~10K-20K tok/s.

    AdamW: The standard LLM optimizer. Adam with decoupled weight decay.
        β1=0.9 (momentum), β2=0.95 (RMS scaling), weight_decay=0.1.
        From Loshchilov & Hutter (2019), "Decoupled Weight Decay Regularization".
"""

import os
import time
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

from phase4_training.lr_schedule import WSDScheduler


def get_device():
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Trainer:
    """Training loop with multi-device support, gradient accumulation, and logging.

    Args:
        model: GPT model instance
        train_dataset: TextDataset for training
        val_dataset: TextDataset for validation
        config: dict with training hyperparameters
    """

    def __init__(self, model, train_dataset, val_dataset, config):
        self.config = config
        self.device = get_device()

        # Move model to device
        self.model = model.to(self.device)

        # Optimizer: AdamW with standard LLM hyperparameters
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config["max_lr"],
            betas=(0.9, 0.95),
            weight_decay=config.get("weight_decay", 0.1),
            fused=self.device.type == "cuda",  # faster fused kernel on CUDA
        )

        # Learning rate scheduler
        self.scheduler = WSDScheduler.from_total_steps(
            self.optimizer,
            max_lr=config["max_lr"],
            min_lr=config.get("min_lr", config["max_lr"] / 10),
            total_steps=config["max_steps"],
            warmup_fraction=config.get("warmup_fraction", 0.05),
            decay_fraction=config.get("decay_fraction", 0.20),
        )

        # Data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config["micro_batch_size"],
            shuffle=True,
            num_workers=0,  # MPS doesn't support multiprocess workers well
            pin_memory=(self.device.type == "cuda"),
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config["micro_batch_size"],
            shuffle=False,
            num_workers=0,
        )

        # Gradient accumulation
        self.grad_accum_steps = config.get("grad_accum_steps", 1)

        # Mixed precision (CUDA only)
        self.use_amp = self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # Logging
        self.log_interval = config.get("log_interval", 50)
        self.eval_interval = config.get("eval_interval", 500)
        self.checkpoint_interval = config.get("checkpoint_interval", 1000)
        self.checkpoint_dir = Path(config.get("checkpoint_dir", "checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # State
        self.step = 0
        self.tokens_processed = 0
        self.best_val_loss = float("inf")
        self.train_losses = []
        self.val_losses = []

    def _get_batch_iterator(self):
        """Infinite iterator over training batches."""
        while True:
            for batch in self.train_loader:
                yield batch

    @torch.no_grad()
    def estimate_loss(self, num_batches=50):
        """Compute average loss on train and val sets."""
        self.model.eval()
        results = {}

        for name, loader in [("train", self.train_loader), ("val", self.val_loader)]:
            total_loss = 0.0
            count = 0
            for i, (x, y) in enumerate(loader):
                if i >= num_batches:
                    break
                x, y = x.to(self.device), y.to(self.device)
                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    _, loss = self.model(x, y)
                total_loss += loss.item()
                count += 1
            results[name] = total_loss / max(count, 1)

        self.model.train()
        return results

    def save_checkpoint(self, filename=None):
        """Save model + optimizer state to disk."""
        if filename is None:
            filename = f"step_{self.step:06d}.pt"
        path = self.checkpoint_dir / filename

        # Save to CPU to ensure cross-device compatibility
        torch.save({
            "step": self.step,
            "model_state": {k: v.cpu() for k, v in self.model.state_dict().items()},
            "optimizer_state": self.optimizer.state_dict(),
            "config": self.config,
            "best_val_loss": self.best_val_loss,
            "tokens_processed": self.tokens_processed,
        }, path)
        return path

    def load_checkpoint(self, path):
        """Resume training from a checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.step = ckpt["step"]
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        self.tokens_processed = ckpt.get("tokens_processed", 0)
        print(f"Resumed from step {self.step}")

    def train(self):
        """Main training loop."""
        config = self.config
        max_steps = config["max_steps"]
        block_size = config["block_size"]
        grad_clip = config.get("grad_clip", 1.0)

        print(f"Training on {self.device} | {'fp16' if self.use_amp else 'fp32'}")
        print(f"Steps: {max_steps} | Micro-batch: {config['micro_batch_size']} | "
              f"Grad accum: {self.grad_accum_steps} | "
              f"Effective batch: {config['micro_batch_size'] * self.grad_accum_steps}")
        print(f"Block size: {block_size} | Max LR: {config['max_lr']}")
        print(f"Params: {self.model.count_parameters():,}")
        print("-" * 60)

        batch_iter = self._get_batch_iterator()
        self.model.train()
        t0 = time.time()

        while self.step < max_steps:
            # --- Gradient accumulation loop ---
            self.optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0

            for micro_step in range(self.grad_accum_steps):
                x, y = next(batch_iter)
                x, y = x.to(self.device), y.to(self.device)

                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    _, loss = self.model(x, y)
                    # Scale loss by accumulation steps so gradient magnitude is correct
                    scaled_loss = loss / self.grad_accum_steps

                self.scaler.scale(scaled_loss).backward()
                accum_loss += loss.item()
                self.tokens_processed += x.numel()

            # --- Gradient clipping ---
            if grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

            # --- Optimizer step ---
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # --- LR schedule step ---
            lr = self.scheduler.step()

            # --- Logging ---
            avg_loss = accum_loss / self.grad_accum_steps
            self.train_losses.append(avg_loss)
            self.step += 1

            if self.step % self.log_interval == 0:
                dt = time.time() - t0
                tok_per_sec = self.tokens_processed / dt if dt > 0 else 0
                print(f"Step {self.step:6d}/{max_steps} | "
                      f"loss {avg_loss:.4f} | lr {lr:.2e} | "
                      f"{tok_per_sec:.0f} tok/s")

            # --- Validation ---
            if self.step % self.eval_interval == 0:
                losses = self.estimate_loss()
                self.val_losses.append(losses["val"])
                print(f"  >> Eval: train_loss={losses['train']:.4f} val_loss={losses['val']:.4f}")

                if losses["val"] < self.best_val_loss:
                    self.best_val_loss = losses["val"]
                    self.save_checkpoint("best.pt")
                    print(f"  >> New best val_loss! Saved best.pt")

            # --- Checkpoint ---
            if self.step % self.checkpoint_interval == 0:
                path = self.save_checkpoint()
                print(f"  >> Saved checkpoint: {path}")

        # Final save
        total_time = time.time() - t0
        print("-" * 60)
        print(f"Training complete: {max_steps} steps in {total_time:.1f}s "
              f"({total_time/60:.1f}min)")
        print(f"Tokens processed: {self.tokens_processed:,}")
        print(f"Best val loss: {self.best_val_loss:.4f}")
        self.save_checkpoint("final.pt")
