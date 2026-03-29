"""Instruction dataset for code fine-tuning.

Terminology:
    Instruction tuning: Training the model to follow instructions by showing it
        (instruction, response) pairs. After this, the model learns the pattern:
        "when given an instruction, produce a helpful response."

    Loss masking: During instruction tuning, we only compute loss on the RESPONSE
        tokens, not the instruction tokens. The model should learn to generate good
        responses, not to parrot instructions back.

    Chat template: A specific format for structuring instruction/response pairs.
        We use a simple format with special tokens:
            <|begin_of_text|>
            <|instruction|>Write a fibonacci function
            <|response|>def fibonacci(n):
                if n <= 1: return n
                return fibonacci(n-1) + fibonacci(n-2)
            <|end_of_text|>

    SFT (Supervised Fine-Tuning): Training on curated (instruction, response) pairs.
        This is the first step of post-training. DPO comes after to refine preferences.

Datasets used:
    - iamtarun/python_code_instructions_18k_alpaca: 18K Python coding instructions
    - Hand-curated evaluation problems (~50 simple coding tasks)
"""

import json
import random
from pathlib import Path
from torch.utils.data import Dataset
import torch


INSTRUCTION_TOKEN = "<|instruction|>"
RESPONSE_TOKEN = "<|response|>"


def format_instruction(instruction, response):
    """Format an instruction-response pair for training."""
    return f"{INSTRUCTION_TOKEN}\n{instruction}\n{RESPONSE_TOKEN}\n{response}"


class InstructionDataset(Dataset):
    """Dataset of (instruction, response) pairs for code fine-tuning.

    Each item returns:
        input_ids: tokenized full sequence
        labels: same as input_ids but with instruction tokens masked (-100)
        attention_mask: all ones (no padding in this simple implementation)

    The loss masking ensures the model only learns to generate responses,
    not to predict instruction tokens.

    Args:
        examples: list of {"instruction": str, "response": str} dicts
        tokenizer: HuggingFace tokenizer
        max_length: maximum sequence length (truncate if longer)
    """

    def __init__(self, examples, tokenizer, max_length=512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Find the response token ID for loss masking
        response_tokens = tokenizer.encode(RESPONSE_TOKEN, add_special_tokens=False)
        self.response_token_ids = response_tokens

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        text = format_instruction(ex["instruction"], ex["response"])

        # Tokenize
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        input_ids = encoded["input_ids"]

        # Create labels with instruction tokens masked
        labels = list(input_ids)
        response_start = self._find_response_start(input_ids)

        # Mask everything before the response with -100 (ignored by cross-entropy)
        for i in range(response_start):
            labels[i] = -100

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def _find_response_start(self, token_ids):
        """Find where the response starts in the token sequence."""
        # Look for the response token pattern
        for i in range(len(token_ids) - len(self.response_token_ids)):
            if token_ids[i:i + len(self.response_token_ids)] == self.response_token_ids:
                return i + len(self.response_token_ids)
        # Fallback: treat everything as response (no masking)
        return 0


def load_code_instructions(max_examples=None, cache_dir="data"):
    """Load Python code instruction dataset from HuggingFace.

    Returns:
        list of {"instruction": str, "response": str} dicts
    """
    cache_path = Path(cache_dir) / "code_instructions.json"

    if cache_path.exists():
        print(f"Loading cached instructions from {cache_path}")
        with open(cache_path) as f:
            examples = json.load(f)
        if max_examples:
            examples = examples[:max_examples]
        return examples

    print("Downloading code instruction dataset...")
    from datasets import load_dataset
    ds = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")

    examples = []
    for item in ds:
        instruction = item.get("instruction", "").strip()
        output = item.get("output", "").strip()
        if instruction and output:
            examples.append({
                "instruction": instruction,
                "response": output,
            })

    # Cache for future use
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(examples, f)
    print(f"Cached {len(examples)} examples to {cache_path}")

    if max_examples:
        examples = examples[:max_examples]

    return examples


# Hand-curated simple evaluation problems
EVAL_PROBLEMS = [
    {"instruction": "Write a Python function that checks if a number is prime.",
     "test": "assert is_prime(7) == True\nassert is_prime(4) == False\nassert is_prime(2) == True"},
    {"instruction": "Write a Python function that returns the factorial of a number.",
     "test": "assert factorial(5) == 120\nassert factorial(0) == 1\nassert factorial(1) == 1"},
    {"instruction": "Write a Python function that reverses a string.",
     "test": "assert reverse_string('hello') == 'olleh'\nassert reverse_string('') == ''"},
    {"instruction": "Write a Python function that finds the maximum element in a list.",
     "test": "assert find_max([1, 3, 2]) == 3\nassert find_max([-1, -5, -2]) == -1"},
    {"instruction": "Write a Python function that checks if a string is a palindrome.",
     "test": "assert is_palindrome('racecar') == True\nassert is_palindrome('hello') == False"},
    {"instruction": "Write a Python function that returns the nth Fibonacci number.",
     "test": "assert fibonacci(0) == 0\nassert fibonacci(1) == 1\nassert fibonacci(10) == 55"},
    {"instruction": "Write a Python function that counts the number of vowels in a string.",
     "test": "assert count_vowels('hello') == 2\nassert count_vowels('xyz') == 0"},
    {"instruction": "Write a Python function that flattens a nested list.",
     "test": "assert flatten([1, [2, 3], [4, [5]]]) == [1, 2, 3, 4, 5]"},
    {"instruction": "Write a Python function that removes duplicates from a list while preserving order.",
     "test": "assert remove_duplicates([1, 2, 2, 3, 1]) == [1, 2, 3]"},
    {"instruction": "Write a Python function that computes the GCD of two numbers.",
     "test": "assert gcd(12, 8) == 4\nassert gcd(7, 3) == 1"},
    {"instruction": "Write a Python function that converts a temperature from Celsius to Fahrenheit.",
     "test": "assert celsius_to_fahrenheit(0) == 32\nassert celsius_to_fahrenheit(100) == 212"},
    {"instruction": "Write a Python function that returns True if a list is sorted in ascending order.",
     "test": "assert is_sorted([1, 2, 3]) == True\nassert is_sorted([3, 1, 2]) == False"},
    {"instruction": "Write a Python function that merges two sorted lists into one sorted list.",
     "test": "assert merge_sorted([1, 3, 5], [2, 4, 6]) == [1, 2, 3, 4, 5, 6]"},
    {"instruction": "Write a Python function that computes the sum of digits of a number.",
     "test": "assert digit_sum(123) == 6\nassert digit_sum(0) == 0"},
    {"instruction": "Write a Python function that generates a list of n random integers between a and b.",
     "test": "result = random_list(5, 1, 10)\nassert len(result) == 5\nassert all(1 <= x <= 10 for x in result)"},
]


def collate_fn(batch):
    """Collate function for DataLoader — pad sequences to same length.

    Uses -100 as padding for labels (ignored by cross-entropy loss).
    """
    max_len = max(item["input_ids"].size(0) for item in batch)

    input_ids = []
    labels = []
    attention_mask = []

    for item in batch:
        seq_len = item["input_ids"].size(0)
        pad_len = max_len - seq_len

        input_ids.append(torch.cat([
            item["input_ids"],
            torch.zeros(pad_len, dtype=torch.long),  # pad with 0
        ]))
        labels.append(torch.cat([
            item["labels"],
            torch.full((pad_len,), -100, dtype=torch.long),  # pad with -100 (ignored)
        ]))
        attention_mask.append(torch.cat([
            torch.ones(seq_len, dtype=torch.long),
            torch.zeros(pad_len, dtype=torch.long),
        ]))

    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attention_mask),
    }
