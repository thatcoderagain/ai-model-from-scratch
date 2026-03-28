"""Dataset loading and preparation for language model training.

Terminology:
    Token sequence: The model learns by predicting the next token. Given tokens [A, B, C, D],
        input = [A, B, C], target = [B, C, D]. Every position predicts the next one.

    Sliding window: We chop the tokenized corpus into fixed-length chunks of `block_size`
        tokens. Each chunk becomes one training example. No overlap — consecutive chunks
        are adjacent in the original text.

    Memory mapping (memmap): For large datasets, we store tokenized data as a binary file
        on disk and access it without loading everything into RAM. NumPy memmap lets us
        treat a file as if it were an array.

    Train/val split: We hold out ~5% of data to measure generalization. If the model
        memorizes training data but fails on validation data, it's "overfitting".

Supported datasets:
    - TinyStories: ~2.5M short stories, great for small models. Fast to train on.
    - FineWeb-Edu: High-quality educational web text. Better language patterns, larger.
    - Custom text file: Any .txt file you provide.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class TextDataset(Dataset):
    """Dataset that serves fixed-length token sequences for language model training.

    Takes a flat array of token IDs and serves (input, target) pairs where:
        input  = tokens[i : i + block_size]
        target = tokens[i+1 : i + block_size + 1]

    Args:
        tokens: 1D numpy array or torch tensor of token IDs
        block_size: context length (max sequence length the model sees)
    """

    def __init__(self, tokens, block_size):
        if isinstance(tokens, np.ndarray):
            self.tokens = torch.from_numpy(tokens).long()
        else:
            self.tokens = tokens.long()
        self.block_size = block_size

    def __len__(self):
        # Number of complete chunks we can extract
        # -1 because we need one extra token for the target
        return max(0, (len(self.tokens) - 1) // self.block_size)

    def __getitem__(self, idx):
        start = idx * self.block_size
        end = start + self.block_size
        x = self.tokens[start:end]
        y = self.tokens[start + 1:end + 1]
        return x, y


def tokenize_corpus(text, tokenizer):
    """Tokenize a text corpus into a flat array of token IDs."""
    return np.array(tokenizer.encode(text), dtype=np.int32)


def load_and_tokenize(source, tokenizer, max_chars=None, cache_dir="data"):
    """Load text from various sources and tokenize it.

    Args:
        source: One of:
            - "tinystories": download TinyStories from HuggingFace
            - "fineweb-edu": download FineWeb-Edu subset from HuggingFace
            - path to a .txt file
        tokenizer: BPETokenizer instance
        max_chars: limit corpus size (useful for testing)
        cache_dir: directory for cached tokenized data

    Returns:
        tokens: 1D numpy array of token IDs
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    if source == "tinystories":
        tokens = _load_tinystories(tokenizer, max_chars, cache_path)
    elif source == "fineweb-edu":
        tokens = _load_fineweb_edu(tokenizer, max_chars, cache_path)
    elif os.path.isfile(source):
        text = Path(source).read_text(encoding="utf-8")
        if max_chars:
            text = text[:max_chars]
        tokens = tokenize_corpus(text, tokenizer)
    else:
        raise ValueError(f"Unknown source: {source}. Use 'tinystories', 'fineweb-edu', or a file path.")

    return tokens


def _load_tinystories(tokenizer, max_chars, cache_path):
    """Load TinyStories dataset from HuggingFace."""
    cache_file = cache_path / "tinystories_tokens.npy"

    if cache_file.exists():
        print(f"Loading cached tokens from {cache_file}")
        return np.load(cache_file)

    print("Downloading TinyStories from HuggingFace...")
    from datasets import load_dataset
    ds = load_dataset("roneneldan/TinyStories", split="train")

    # Concatenate all stories with end-of-text separator
    texts = []
    total_chars = 0
    for item in ds:
        text = item["text"].strip()
        if text:
            texts.append(text)
            total_chars += len(text)
            if max_chars and total_chars >= max_chars:
                break

    corpus = "\n\n".join(texts)
    if max_chars:
        corpus = corpus[:max_chars]

    print(f"Tokenizing {len(corpus):,} characters...")
    tokens = tokenize_corpus(corpus, tokenizer)

    np.save(cache_file, tokens)
    print(f"Cached {len(tokens):,} tokens to {cache_file}")
    return tokens


def _load_fineweb_edu(tokenizer, max_chars, cache_path):
    """Load FineWeb-Edu subset from HuggingFace."""
    cache_file = cache_path / "fineweb_edu_tokens.npy"

    if cache_file.exists():
        print(f"Loading cached tokens from {cache_file}")
        return np.load(cache_file)

    print("Downloading FineWeb-Edu (sample-10BT) from HuggingFace...")
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)

    texts = []
    total_chars = 0
    target = max_chars or 50_000_000  # default 50MB
    for item in ds:
        text = item["text"].strip()
        if text:
            texts.append(text)
            total_chars += len(text)
            if total_chars >= target:
                break

    corpus = "\n\n".join(texts)
    if max_chars:
        corpus = corpus[:max_chars]

    print(f"Tokenizing {len(corpus):,} characters...")
    tokens = tokenize_corpus(corpus, tokenizer)

    np.save(cache_file, tokens)
    print(f"Cached {len(tokens):,} tokens to {cache_file}")
    return tokens


def create_datasets(tokens, block_size, val_fraction=0.05):
    """Split tokens into train and validation datasets.

    Args:
        tokens: 1D numpy array of token IDs
        block_size: context length
        val_fraction: fraction of data for validation (default 5%)

    Returns:
        train_dataset, val_dataset: TextDataset instances
    """
    n = len(tokens)
    split_idx = int(n * (1 - val_fraction))

    # Ensure split aligns to block_size boundaries
    split_idx = (split_idx // block_size) * block_size

    train_tokens = tokens[:split_idx]
    val_tokens = tokens[split_idx:]

    train_ds = TextDataset(train_tokens, block_size)
    val_ds = TextDataset(val_tokens, block_size)

    print(f"Train: {len(train_tokens):,} tokens ({len(train_ds)} batches of {block_size})")
    print(f"Val:   {len(val_tokens):,} tokens ({len(val_ds)} batches of {block_size})")

    return train_ds, val_ds
