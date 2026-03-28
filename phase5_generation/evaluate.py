"""Model evaluation — perplexity and sample generation.

Terminology:
    Perplexity: The standard metric for language models. Defined as exp(cross_entropy_loss).
        Intuitively: "how many tokens is the model choosing between on average?"
        - Perplexity 1 = perfect prediction (knows exactly what comes next)
        - Perplexity 10 = as confused as choosing randomly from 10 options
        - Perplexity 100 = very uncertain

        For our 15M model on TinyStories, we expect perplexity ~20-50 after training.
        For reference: GPT-2 (1.5B params) achieves perplexity ~20 on general web text.

    Validation loss: Cross-entropy loss on held-out data the model hasn't seen during training.
        If train_loss keeps dropping but val_loss stops or increases → the model is overfitting
        (memorizing training data instead of learning general patterns).
"""

import math
import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def compute_perplexity(model, dataset, device=None, batch_size=8, max_batches=None):
    """Compute perplexity on a dataset.

    Args:
        model: GPT model
        dataset: TextDataset
        device: computation device
        batch_size: evaluation batch size
        max_batches: limit number of batches (None = full dataset)

    Returns:
        perplexity: float
        avg_loss: float (cross-entropy)
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    total_loss = 0.0
    total_batches = 0

    for i, (x, y) in enumerate(loader):
        if max_batches and i >= max_batches:
            break

        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        total_loss += loss.item()
        total_batches += 1

    avg_loss = total_loss / max(total_batches, 1)
    perplexity = math.exp(avg_loss)

    return perplexity, avg_loss


def evaluate_model(model, train_dataset, val_dataset, device=None, max_batches=100):
    """Full evaluation: perplexity on train and val sets.

    Returns:
        dict with train_loss, val_loss, train_ppl, val_ppl
    """
    train_ppl, train_loss = compute_perplexity(model, train_dataset, device, max_batches=max_batches)
    val_ppl, val_loss = compute_perplexity(model, val_dataset, device, max_batches=max_batches)

    return {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_perplexity": train_ppl,
        "val_perplexity": val_ppl,
    }
