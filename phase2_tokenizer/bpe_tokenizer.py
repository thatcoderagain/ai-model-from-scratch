"""Byte-level BPE Tokenizer — built from scratch.

This is the same algorithm used by GPT-2/3/4, Llama, and most modern LLMs.
We implement every step: pre-tokenization, BPE training, encoding, decoding.

Key terminology:
    BPE (Byte Pair Encoding): A compression algorithm adapted for tokenization.
        Repeatedly merges the most frequent adjacent pair of tokens into a new token.
        Originally from Sennrich et al. (2016), "Neural Machine Translation of Rare Words
        with Subword Units". Modern LLMs use byte-level BPE (Radford et al., GPT-2, 2019).

    Byte-level: The base vocabulary is the 256 possible byte values (0x00-0xFF).
        This means ANY text (unicode, emoji, binary) can be tokenized — no "unknown" tokens.
        Characters like 'A' map to byte 65, 'é' maps to bytes [195, 169], etc.

    Pre-tokenization: Splitting text into chunks BEFORE applying BPE.
        We use a regex pattern (GPT-4 style) to split on word boundaries, whitespace,
        punctuation, and numbers. BPE merges only happen within these chunks, never across.
        This prevents merging across word boundaries (e.g., "the" + space + "cat").

    Merge: One BPE operation — take the most frequent adjacent pair (e.g., bytes for 't','h')
        and replace all occurrences with a new token (e.g., 'th'). Each merge adds one token
        to the vocabulary. Training stops when vocab reaches target size.

    Vocab size: Total number of unique tokens. 256 (bytes) + num_merges + special_tokens.
        Our model uses 4096 (educational). Production models use 32K-128K.

    Special tokens: Tokens with special meaning not derived from text.
        E.g., <|end_of_text|> signals the end of a document.
        These are assigned IDs above the regular vocabulary.
"""

import re
import json
from pathlib import Path


# GPT-4 style pre-tokenization pattern
# This regex splits text into meaningful chunks before BPE:
#   - Contractions: 's, 't, 're, 've, 'm, 'll, 'd
#   - Words: sequences of letters (with optional leading space)
#   - Numbers: sequences of digits (with optional leading space)
#   - Punctuation: sequences of non-alphanumeric chars (with optional leading space)
#   - Whitespace: sequences of whitespace (trailing, not followed by non-whitespace)
#   - Individual whitespace characters
GPT4_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

# Simpler fallback pattern that works with Python's re module (no \p{} support)
PRETOKENIZE_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w\d]+|\s+(?!\S)|\s+"""


class BPETokenizer:
    """Byte-level BPE tokenizer built from scratch.

    Training:
        1. Start with 256 byte-level tokens as the base vocabulary
        2. Pre-tokenize text into chunks using regex
        3. Convert each chunk to a sequence of byte token IDs
        4. Repeatedly find the most frequent adjacent pair and merge it
        5. Each merge creates a new token and grows the vocabulary by 1
        6. Stop when vocabulary reaches target size

    Encoding (after training):
        1. Pre-tokenize text into chunks
        2. Convert each chunk to bytes
        3. Apply learned merges in priority order (earliest merge first)
        4. Return list of token IDs

    Decoding:
        1. Look up byte sequence for each token ID
        2. Concatenate all bytes
        3. Decode UTF-8
    """

    def __init__(self, vocab_size=4096):
        self.target_vocab_size = vocab_size

        # Base vocabulary: 256 byte values
        # Token ID 0 = byte 0x00, Token ID 255 = byte 0xFF
        self.vocab = {i: bytes([i]) for i in range(256)}

        # Learned merges: (token_a, token_b) -> new_token_id
        # Ordered by when they were learned (priority order)
        self.merges = {}

        # Special tokens: name -> token_id
        self.special_tokens = {}
        self.inverse_special = {}  # token_id -> name

        # Compiled regex for pre-tokenization
        self._pattern = re.compile(PRETOKENIZE_PATTERN)

    def _pre_tokenize(self, text):
        """Split text into chunks using regex before applying BPE.

        Returns list of byte sequences (one per chunk).
        """
        chunks = self._pattern.findall(text)
        return [chunk.encode("utf-8") for chunk in chunks]

    def _get_pair_counts(self, token_sequences):
        """Count all adjacent pairs across all sequences.

        Returns dict: (token_a, token_b) -> count
        """
        counts = {}
        for seq in token_sequences:
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                counts[pair] = counts.get(pair, 0) + 1
        return counts

    def _merge_pair(self, token_sequences, pair, new_id):
        """Replace all occurrences of pair with new_id in all sequences."""
        result = []
        for seq in token_sequences:
            new_seq = []
            i = 0
            while i < len(seq):
                if i < len(seq) - 1 and seq[i] == pair[0] and seq[i + 1] == pair[1]:
                    new_seq.append(new_id)
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1
            result.append(new_seq)
        return result

    def train(self, text, verbose=True):
        """Learn BPE merges from training text.

        Args:
            text: Training corpus as a single string
            verbose: Print progress during training
        """
        num_merges = self.target_vocab_size - 256 - len(self.special_tokens)
        if num_merges <= 0:
            return

        # Step 1: Pre-tokenize and convert to byte token sequences
        byte_chunks = self._pre_tokenize(text)
        token_sequences = [list(chunk) for chunk in byte_chunks]

        if verbose:
            total_tokens = sum(len(s) for s in token_sequences)
            print(f"Training BPE: {len(token_sequences)} chunks, {total_tokens} initial tokens")
            print(f"Target: {num_merges} merges to reach vocab_size={self.target_vocab_size}")

        # Merge IDs start after bytes (256) + special tokens
        merge_id_offset = 256 + len(self.special_tokens)

        # Step 2: Iteratively merge most frequent pair
        for i in range(num_merges):
            # Count all adjacent pairs
            pair_counts = self._get_pair_counts(token_sequences)

            if not pair_counts:
                if verbose:
                    print(f"  No more pairs to merge at step {i}")
                break

            # Find most frequent pair
            best_pair = max(pair_counts, key=pair_counts.get)
            best_count = pair_counts[best_pair]

            if best_count < 2:
                if verbose:
                    print(f"  No pair with count >= 2 at step {i}")
                break

            # Create new token (offset past special tokens to avoid ID collision)
            new_id = merge_id_offset + i
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            # Apply merge to all sequences
            token_sequences = self._merge_pair(token_sequences, best_pair, new_id)

            if verbose and (i + 1) % 500 == 0:
                new_total = sum(len(s) for s in token_sequences)
                merged_bytes = self.vocab[new_id]
                try:
                    display = merged_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    display = str(merged_bytes)
                print(
                    f"  Merge {i+1}/{num_merges}: "
                    f"({best_pair[0]}, {best_pair[1]}) -> {new_id} "
                    f"'{display}' (count={best_count}, tokens={new_total})"
                )

        if verbose:
            final_tokens = sum(len(s) for s in token_sequences)
            ratio = sum(len(c) for c in byte_chunks) / max(final_tokens, 1)
            print(f"Done: vocab_size={len(self.vocab)}, compression={ratio:.2f}x")

    def add_special_tokens(self, tokens):
        """Register special tokens (e.g., <|end_of_text|>).

        Special tokens get IDs 256, 257, ... (right after base byte vocab).
        Merge IDs start AFTER special tokens to avoid collisions.
        Must be called BEFORE train().
        """
        next_id = 256 + len(self.special_tokens)
        for token_name in tokens:
            if token_name not in self.special_tokens:
                self.special_tokens[token_name] = next_id
                self.inverse_special[next_id] = token_name
                next_id += 1

    def _encode_chunk(self, byte_chunk):
        """Encode a single pre-tokenized chunk using learned merges.

        Applies merges in priority order (earliest learned first).
        """
        ids = list(byte_chunk)

        # Apply merges in order they were learned
        for pair, new_id in self.merges.items():
            i = 0
            while i < len(ids) - 1:
                if ids[i] == pair[0] and ids[i + 1] == pair[1]:
                    ids = ids[:i] + [new_id] + ids[i + 2 :]
                    # Don't increment i — check if the new token can merge with previous
                    if i > 0:
                        i -= 1
                else:
                    i += 1

        return ids

    def encode(self, text):
        """Convert text to a list of token IDs.

        Steps:
            1. Check for special tokens (exact string match)
            2. Pre-tokenize remaining text into chunks
            3. Convert each chunk to bytes
            4. Apply learned BPE merges to each chunk
            5. Return concatenated token IDs
        """
        if not text:
            return []

        # Handle special tokens
        special_pattern = (
            "(" + "|".join(re.escape(k) for k in self.special_tokens) + ")"
            if self.special_tokens
            else None
        )

        ids = []
        if special_pattern:
            parts = re.split(special_pattern, text)
            for part in parts:
                if not part:
                    continue
                if part in self.special_tokens:
                    ids.append(self.special_tokens[part])
                else:
                    ids.extend(self._encode_text(part))
        else:
            ids = self._encode_text(text)

        return ids

    def _encode_text(self, text):
        """Encode regular text (no special tokens)."""
        byte_chunks = self._pre_tokenize(text)
        ids = []
        for chunk in byte_chunks:
            ids.extend(self._encode_chunk(chunk))
        return ids

    def decode(self, ids):
        """Convert token IDs back to text.

        Steps:
            1. For each ID, look up its byte sequence (or special token name)
            2. Concatenate all bytes
            3. Decode UTF-8 (with error replacement for invalid sequences)
        """
        byte_pieces = []
        for token_id in ids:
            if token_id in self.inverse_special:
                byte_pieces.append(self.inverse_special[token_id].encode("utf-8"))
            elif token_id in self.vocab:
                byte_pieces.append(self.vocab[token_id])
            else:
                # Unknown token — skip or replace
                byte_pieces.append(b"\xef\xbf\xbd")  # UTF-8 replacement char
        return b"".join(byte_pieces).decode("utf-8", errors="replace")

    def save(self, path):
        """Save tokenizer to directory (vocab.json + merges.txt)."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save vocab: id -> hex-encoded bytes
        vocab_data = {}
        for token_id, byte_val in self.vocab.items():
            vocab_data[str(token_id)] = byte_val.hex()

        with open(path / "vocab.json", "w") as f:
            json.dump(
                {
                    "vocab": vocab_data,
                    "special_tokens": self.special_tokens,
                    "target_vocab_size": self.target_vocab_size,
                },
                f,
                indent=2,
            )

        # Save merges in order (priority matters)
        with open(path / "merges.txt", "w") as f:
            f.write("# BPE merges (pair_a pair_b -> new_id)\n")
            for (a, b), new_id in self.merges.items():
                f.write(f"{a} {b} {new_id}\n")

    @classmethod
    def load(cls, path):
        """Load tokenizer from directory."""
        path = Path(path)

        with open(path / "vocab.json") as f:
            data = json.load(f)

        tok = cls(vocab_size=data["target_vocab_size"])

        # Restore vocab
        tok.vocab = {}
        for token_id_str, hex_bytes in data["vocab"].items():
            tok.vocab[int(token_id_str)] = bytes.fromhex(hex_bytes)

        # Restore special tokens
        tok.special_tokens = data.get("special_tokens", {})
        tok.inverse_special = {v: k for k, v in tok.special_tokens.items()}

        # Restore merges (order matters)
        tok.merges = {}
        with open(path / "merges.txt") as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.strip().split()
                a, b, new_id = int(parts[0]), int(parts[1]), int(parts[2])
                tok.merges[(a, b)] = new_id

        return tok

    def vocab_size(self):
        """Current vocabulary size including special tokens."""
        return len(self.vocab) + len(self.special_tokens)

    def __repr__(self):
        return (
            f"BPETokenizer(vocab={len(self.vocab)}, "
            f"merges={len(self.merges)}, "
            f"special={len(self.special_tokens)})"
        )
