"""Train a BPE tokenizer on a mixed English + Python code corpus.

Usage:
    python -m phase2_tokenizer.train_tokenizer [--vocab-size 4096] [--output phase2_tokenizer/vocab]

This script trains on an embedded sample corpus (English text + Python code)
to produce a tokenizer that handles both natural language and code well.
For production, you'd train on a much larger corpus (GBs of text).
"""

import argparse
from pathlib import Path

from phase2_tokenizer.bpe_tokenizer import BPETokenizer

# Sample training corpus — mix of English and Python code
# In production, this would be loaded from FineWeb-Edu or similar
SAMPLE_CORPUS = """
The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog.
Natural language processing is a subfield of linguistics, computer science, and artificial intelligence.
Machine learning is the study of computer algorithms that can improve automatically through experience.
A neural network is a network or circuit of biological neurons, or a mathematical model of such a network.
The transformer architecture was introduced in the paper "Attention Is All You Need" in 2017.
Large language models like GPT, Claude, and Llama are based on the transformer architecture.
Training a language model involves predicting the next token in a sequence of text.
The model learns to assign probabilities to each possible next token given the context.
Attention mechanisms allow the model to focus on relevant parts of the input sequence.
Modern transformers use techniques like RoPE, RMSNorm, SwiGLU, and grouped query attention.
The loss function for language models is cross-entropy, which measures prediction quality.
Backpropagation computes gradients that tell us how to update the model's parameters.
Optimizers like AdamW use these gradients to adjust weights and minimize the loss.
The learning rate controls the size of each update step during training.
Batch size determines how many examples are processed before each weight update.
Gradient accumulation allows simulating larger batch sizes on limited memory hardware.
Tokenization converts raw text into a sequence of integer tokens that the model can process.
Byte pair encoding is the most common tokenization algorithm used in modern language models.
The vocabulary size determines how many unique tokens the model can represent.
Special tokens like end-of-text markers help the model understand document boundaries.

Once upon a time, there was a small village nestled between two great mountains.
The people of the village were kind and hardworking, and they lived in harmony with nature.
Every morning, the children would run through the fields, laughing and playing together.
The oldest woman in the village was known for her wisdom and her beautiful garden.
She would often sit by the river, watching the water flow and thinking about the world.

The history of computing dates back to the early mechanical calculators of the 17th century.
Charles Babbage designed the Analytical Engine in the 1830s, often considered the first computer.
Ada Lovelace wrote what is recognized as the first algorithm intended for a machine.
The modern electronic computer emerged in the 1940s with machines like ENIAC and Colossus.
The development of transistors in the 1950s led to smaller, faster, and more reliable computers.
The invention of the integrated circuit in 1958 revolutionized the electronics industry.
Personal computers became widely available in the 1980s, transforming how people work and live.

def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

class LinkedList:
    def __init__(self):
        self.head = None

    def append(self, value):
        node = Node(value)
        if not self.head:
            self.head = node
            return
        current = self.head
        while current.next:
            current = current.next
        current.next = node

    def __len__(self):
        count = 0
        current = self.head
        while current:
            count += 1
            current = current.next
        return count

    def __repr__(self):
        values = []
        current = self.head
        while current:
            values.append(str(current.value))
            current = current.next
        return " -> ".join(values)

def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

import numpy as np
import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.head(x)

# Training loop example
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        logits = model(batch["input_ids"])
        loss = criterion(logits.view(-1, vocab_size), batch["labels"].view(-1))
        loss.backward()
        optimizer.step()
        scheduler.step()

print("Training complete!")
print(f"Final loss: {loss.item():.4f}")
""" * 3  # Repeat 3x to give BPE more data to find patterns


def main():
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer")
    parser.add_argument("--vocab-size", type=int, default=4096, help="Target vocabulary size")
    parser.add_argument("--output", type=str, default="phase2_tokenizer/vocab", help="Output directory")
    args = parser.parse_args()

    print(f"Training BPE tokenizer with vocab_size={args.vocab_size}")
    print(f"Corpus size: {len(SAMPLE_CORPUS):,} characters\n")

    # Create and train tokenizer
    tokenizer = BPETokenizer(vocab_size=args.vocab_size)

    # Add special tokens
    special_tokens = [
        "<|begin_of_text|>",
        "<|end_of_text|>",
        "<|start_code|>",
        "<|end_code|>",
        "<|pad|>",
    ]
    tokenizer.add_special_tokens(special_tokens)
    print(f"Special tokens: {special_tokens}\n")

    # Train
    tokenizer.train(SAMPLE_CORPUS, verbose=True)

    # Save
    output_path = Path(args.output)
    tokenizer.save(output_path)
    print(f"\nSaved tokenizer to {output_path}/")

    # Demo
    print("\n" + "=" * 60)
    print("Demo: Encoding & Decoding")
    print("=" * 60)

    test_strings = [
        "Hello, world!",
        "The transformer architecture uses attention mechanisms.",
        "def hello():\n    print('Hello, world!')",
        "x = np.array([1, 2, 3])",
    ]

    for s in test_strings:
        ids = tokenizer.encode(s)
        decoded = tokenizer.decode(ids)
        ratio = len(s.encode("utf-8")) / len(ids)
        print(f"\n  Input:   \"{s}\"")
        print(f"  Tokens:  {ids[:20]}{'...' if len(ids) > 20 else ''} ({len(ids)} tokens)")
        print(f"  Decoded: \"{decoded}\"")
        print(f"  Ratio:   {ratio:.1f}x compression")
        print(f"  Match:   {decoded == s}")


if __name__ == "__main__":
    main()
