"""WSD (Warmup-Stable-Decay) Learning Rate Schedule.

The modern standard for LLM training, replacing cosine decay (GPT-2 era).

Three phases:
    1. Warmup (5% of training): Linear ramp from 0 → max_lr
       Why: At the start, random weights produce wild gradients. Small steps
       prevent the optimizer from making catastrophic early updates.

    2. Stable (75% of training): Constant at max_lr
       Why: This is where most learning happens. The high constant LR drives
       rapid exploration of the loss landscape. Research shows the model makes
       progress along a "river valley" direction during this phase.

    3. Decay (20% of training): Linear decay from max_lr → min_lr
       Why: At the end, we want the model to settle into a good minimum.
       Reducing LR lets it converge precisely rather than oscillating.

Paper: "Understanding Warmup-Stable-Decay Learning Rates: A River Valley Loss
        Landscape Perspective" — arXiv:2410.05192 (2024)

Why WSD > Cosine:
    - Cosine decay continuously reduces LR, wasting most of training at sub-optimal rates
    - WSD keeps LR high during the stable phase → faster learning
    - WSD achieves better final loss at the same compute budget
    - Simpler to tune: just set warmup fraction, stable fraction, and max/min LR
"""

import math


class WSDScheduler:
    """Warmup-Stable-Decay learning rate scheduler.

    Args:
        optimizer: PyTorch optimizer
        max_lr: peak learning rate (during stable phase)
        min_lr: minimum learning rate (end of decay phase)
        warmup_steps: number of warmup steps (linear ramp to max_lr)
        stable_steps: number of steps at constant max_lr
        decay_steps: number of steps to decay from max_lr to min_lr
    """

    def __init__(self, optimizer, max_lr, min_lr, warmup_steps, stable_steps, decay_steps):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.stable_steps = stable_steps
        self.decay_steps = decay_steps
        self.total_steps = warmup_steps + stable_steps + decay_steps
        self._step = 0

    def get_lr(self, step=None):
        """Compute learning rate for the given step."""
        if step is None:
            step = self._step

        if step < self.warmup_steps:
            # Linear warmup: 0 → max_lr
            return self.max_lr * (step / max(1, self.warmup_steps))

        elif step < self.warmup_steps + self.stable_steps:
            # Constant at max_lr
            return self.max_lr

        else:
            # Linear decay: max_lr → min_lr
            decay_progress = (step - self.warmup_steps - self.stable_steps) / max(1, self.decay_steps)
            decay_progress = min(1.0, decay_progress)  # clamp
            return self.max_lr - (self.max_lr - self.min_lr) * decay_progress

    def step(self):
        """Update learning rate for current step and advance."""
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self._step += 1
        return lr

    @classmethod
    def from_total_steps(cls, optimizer, max_lr, min_lr, total_steps,
                         warmup_fraction=0.05, decay_fraction=0.20):
        """Create scheduler from total steps and fractions.

        Args:
            total_steps: total training steps
            warmup_fraction: fraction of steps for warmup (default 5%)
            decay_fraction: fraction of steps for decay (default 20%)
        """
        warmup_steps = int(total_steps * warmup_fraction)
        decay_steps = int(total_steps * decay_fraction)
        stable_steps = total_steps - warmup_steps - decay_steps
        return cls(optimizer, max_lr, min_lr, warmup_steps, stable_steps, decay_steps)
