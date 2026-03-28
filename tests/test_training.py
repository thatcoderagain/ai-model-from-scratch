"""Tests for Phase 4: Training Pipeline.

Covers:
- Dataset creation and indexing
- WSD learning rate schedule
- Trainer initialization and basic training steps
- Checkpoint save/load
"""

import tempfile
import numpy as np
import torch
import pytest

from phase4_training.dataset import TextDataset, create_datasets
from phase4_training.lr_schedule import WSDScheduler
from phase4_training.trainer import get_device
from phase3_transformer.config import TINY_CONFIG
from phase3_transformer.model import GPT


# ============================================================
# Dataset
# ============================================================

class TestTextDataset:
    def test_basic_indexing(self):
        """Dataset should return (input, target) offset by 1."""
        tokens = np.arange(100, dtype=np.int32)
        ds = TextDataset(tokens, block_size=10)
        x, y = ds[0]
        assert x.tolist() == list(range(0, 10))
        assert y.tolist() == list(range(1, 11))

    def test_second_chunk(self):
        tokens = np.arange(100, dtype=np.int32)
        ds = TextDataset(tokens, block_size=10)
        x, y = ds[1]
        assert x.tolist() == list(range(10, 20))
        assert y.tolist() == list(range(11, 21))

    def test_length(self):
        tokens = np.arange(100, dtype=np.int32)
        ds = TextDataset(tokens, block_size=10)
        # 99 usable tokens (need 1 extra for target), 99 // 10 = 9 chunks
        assert len(ds) == 9

    def test_short_sequence(self):
        """Sequence shorter than block_size → 0 chunks."""
        tokens = np.arange(5, dtype=np.int32)
        ds = TextDataset(tokens, block_size=10)
        assert len(ds) == 0

    def test_output_dtypes(self):
        tokens = np.arange(50, dtype=np.int32)
        ds = TextDataset(tokens, block_size=10)
        x, y = ds[0]
        assert x.dtype == torch.long
        assert y.dtype == torch.long

    def test_create_datasets_split(self):
        tokens = np.arange(1000, dtype=np.int32)
        train_ds, val_ds = create_datasets(tokens, block_size=10, val_fraction=0.1)
        assert len(train_ds) > 0
        assert len(val_ds) > 0
        assert len(train_ds) > len(val_ds)


# ============================================================
# WSD Learning Rate Schedule
# ============================================================

class TestWSDScheduler:
    def test_warmup_starts_at_zero(self):
        opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=1.0)
        sched = WSDScheduler(opt, max_lr=3e-4, min_lr=3e-5,
                              warmup_steps=100, stable_steps=800, decay_steps=100)
        assert sched.get_lr(0) == 0.0

    def test_warmup_reaches_max(self):
        opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=1.0)
        sched = WSDScheduler(opt, max_lr=3e-4, min_lr=3e-5,
                              warmup_steps=100, stable_steps=800, decay_steps=100)
        lr = sched.get_lr(100)
        assert abs(lr - 3e-4) < 1e-8

    def test_stable_phase(self):
        opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=1.0)
        sched = WSDScheduler(opt, max_lr=3e-4, min_lr=3e-5,
                              warmup_steps=100, stable_steps=800, decay_steps=100)
        # All steps in stable phase should be at max_lr
        for step in [200, 500, 899]:
            assert abs(sched.get_lr(step) - 3e-4) < 1e-8

    def test_decay_reaches_min(self):
        opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=1.0)
        sched = WSDScheduler(opt, max_lr=3e-4, min_lr=3e-5,
                              warmup_steps=100, stable_steps=800, decay_steps=100)
        lr = sched.get_lr(1000)
        assert abs(lr - 3e-5) < 1e-8

    def test_from_total_steps(self):
        opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=1.0)
        sched = WSDScheduler.from_total_steps(opt, max_lr=3e-4, min_lr=3e-5,
                                                total_steps=1000)
        assert sched.warmup_steps == 50   # 5% of 1000
        assert sched.decay_steps == 200   # 20% of 1000
        assert sched.stable_steps == 750  # remainder

    def test_step_updates_optimizer(self):
        param = torch.zeros(1, requires_grad=True)
        opt = torch.optim.SGD([param], lr=1.0)
        sched = WSDScheduler(opt, max_lr=3e-4, min_lr=3e-5,
                              warmup_steps=10, stable_steps=80, decay_steps=10)
        # Step through warmup + 1 into stable phase
        for _ in range(11):
            sched.step()
        # Now in stable phase, lr should be max_lr
        assert abs(opt.param_groups[0]['lr'] - 3e-4) < 1e-8

    def test_monotonic_warmup(self):
        opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=1.0)
        sched = WSDScheduler(opt, max_lr=3e-4, min_lr=3e-5,
                              warmup_steps=100, stable_steps=800, decay_steps=100)
        lrs = [sched.get_lr(s) for s in range(100)]
        for i in range(1, len(lrs)):
            assert lrs[i] >= lrs[i-1], f"LR decreased during warmup at step {i}"

    def test_monotonic_decay(self):
        opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=1.0)
        sched = WSDScheduler(opt, max_lr=3e-4, min_lr=3e-5,
                              warmup_steps=100, stable_steps=800, decay_steps=100)
        lrs = [sched.get_lr(s) for s in range(900, 1001)]
        for i in range(1, len(lrs)):
            assert lrs[i] <= lrs[i-1], f"LR increased during decay at step {900+i}"


# ============================================================
# Trainer (smoke test)
# ============================================================

class TestTrainer:
    def test_device_detection(self):
        device = get_device()
        assert device.type in ("cuda", "mps", "cpu")

    def test_smoke_training(self):
        """Run a few training steps to verify the pipeline works end-to-end."""
        # Create tiny model + random data
        config = TINY_CONFIG
        model = GPT(config)

        tokens = np.random.randint(0, config.vocab_size, size=5000, dtype=np.int32)
        train_ds = TextDataset(tokens[:4000], config.block_size)
        val_ds = TextDataset(tokens[4000:], config.block_size)

        trainer_config = {
            "max_steps": 5,
            "micro_batch_size": 2,
            "grad_accum_steps": 1,
            "max_lr": 1e-3,
            "min_lr": 1e-4,
            "weight_decay": 0.1,
            "grad_clip": 1.0,
            "warmup_fraction": 0.1,
            "decay_fraction": 0.2,
            "log_interval": 2,
            "eval_interval": 5,
            "checkpoint_interval": 100,
            "block_size": config.block_size,
            "checkpoint_dir": tempfile.mkdtemp(),
        }

        from phase4_training.trainer import Trainer
        trainer = Trainer(model, train_ds, val_ds, trainer_config)
        trainer.train()

        # Verify training ran
        assert trainer.step == 5
        assert len(trainer.train_losses) == 5
        assert trainer.tokens_processed > 0

    def test_loss_decreases_on_repeated_data(self):
        """On a tiny dataset, loss should decrease if we overfit."""
        config = TINY_CONFIG
        model = GPT(config)

        # Small repeating dataset — model should be able to memorize this
        tokens = np.tile(np.arange(config.block_size + 1, dtype=np.int32), 50)
        train_ds = TextDataset(tokens, config.block_size)
        val_ds = TextDataset(tokens, config.block_size)

        trainer_config = {
            "max_steps": 50,
            "micro_batch_size": 4,
            "grad_accum_steps": 1,
            "max_lr": 1e-3,
            "min_lr": 1e-4,
            "weight_decay": 0.0,  # no regularization for overfitting test
            "grad_clip": 1.0,
            "warmup_fraction": 0.1,
            "decay_fraction": 0.1,
            "log_interval": 100,
            "eval_interval": 100,
            "checkpoint_interval": 100,
            "block_size": config.block_size,
            "checkpoint_dir": tempfile.mkdtemp(),
        }

        from phase4_training.trainer import Trainer
        trainer = Trainer(model, train_ds, val_ds, trainer_config)
        trainer.train()

        # Loss should decrease
        first_5 = sum(trainer.train_losses[:5]) / 5
        last_5 = sum(trainer.train_losses[-5:]) / 5
        assert last_5 < first_5, f"Loss didn't decrease: first={first_5:.4f}, last={last_5:.4f}"

    def test_checkpoint_save_load(self):
        """Save and load a checkpoint, verify model produces same output."""
        config = TINY_CONFIG
        model = GPT(config)

        tokens = np.random.randint(0, config.vocab_size, size=2000, dtype=np.int32)
        train_ds = TextDataset(tokens[:1500], config.block_size)
        val_ds = TextDataset(tokens[1500:], config.block_size)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer_config = {
                "max_steps": 3,
                "micro_batch_size": 2,
                "grad_accum_steps": 1,
                "max_lr": 1e-3,
                "min_lr": 1e-4,
                "weight_decay": 0.1,
                "grad_clip": 1.0,
                "warmup_fraction": 0.1,
                "decay_fraction": 0.2,
                "log_interval": 100,
                "eval_interval": 100,
                "checkpoint_interval": 100,
                "block_size": config.block_size,
                "checkpoint_dir": tmpdir,
            }

            from phase4_training.trainer import Trainer
            trainer = Trainer(model, train_ds, val_ds, trainer_config)
            trainer.train()

            # Save
            path = trainer.save_checkpoint("test.pt")
            assert path.exists()

            # Load into a new model
            model2 = GPT(config)
            trainer2 = Trainer(model2, train_ds, val_ds, trainer_config)
            trainer2.load_checkpoint(path)

            # Compare outputs
            test_input = torch.randint(0, config.vocab_size, (1, 8))
            model.eval()
            model2.eval()
            with torch.no_grad():
                out1, _ = model(test_input.to(trainer.device))
                out2, _ = model2(test_input.to(trainer2.device))
            assert torch.allclose(out1.cpu(), out2.cpu(), atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
