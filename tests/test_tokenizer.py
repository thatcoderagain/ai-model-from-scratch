"""Tests for Phase 2: BPE Tokenizer.

Covers:
- Roundtrip encoding/decoding (most critical property)
- Unicode, emoji, multilingual support
- Code tokenization
- Special tokens
- Edge cases (empty, single char, long text)
- Save/load persistence
- Compression ratio
"""

import tempfile
import numpy as np
import pytest

from phase2_tokenizer.bpe_tokenizer import BPETokenizer


# Shared fixture: a trained tokenizer
@pytest.fixture(scope="module")
def trained_tokenizer():
    """Train a tokenizer once for all tests in this module."""
    tok = BPETokenizer(vocab_size=512)
    tok.add_special_tokens([
        "<|begin_of_text|>",
        "<|end_of_text|>",
        "<|start_code|>",
        "<|end_code|>",
    ])

    corpus = """
    The quick brown fox jumps over the lazy dog. The quick brown fox jumps.
    Natural language processing is a subfield of artificial intelligence.
    Machine learning algorithms improve automatically through experience.
    The transformer architecture uses self-attention mechanisms.
    def fibonacci(n):
        if n <= 1:
            return n
        return fibonacci(n - 1) + fibonacci(n - 2)
    class Model:
        def __init__(self):
            self.layers = []
        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x
    for i in range(10):
        print(f"Step {i}: loss = {loss:.4f}")
    """ * 5
    tok.train(corpus, verbose=False)
    return tok


# ============================================================
# Roundtrip Tests (most important property)
# ============================================================

class TestRoundtrip:
    """decode(encode(text)) must equal text for any valid input."""

    def test_simple_english(self, trained_tokenizer):
        text = "Hello, world!"
        assert trained_tokenizer.decode(trained_tokenizer.encode(text)) == text

    def test_sentence(self, trained_tokenizer):
        text = "The quick brown fox jumps over the lazy dog."
        assert trained_tokenizer.decode(trained_tokenizer.encode(text)) == text

    def test_multiline(self, trained_tokenizer):
        text = "Line one.\nLine two.\nLine three."
        assert trained_tokenizer.decode(trained_tokenizer.encode(text)) == text

    def test_whitespace_variations(self, trained_tokenizer):
        for text in ["  leading", "trailing  ", "  both  ", "multiple   spaces", "\ttab", "\nnewline"]:
            assert trained_tokenizer.decode(trained_tokenizer.encode(text)) == text

    def test_numbers(self, trained_tokenizer):
        text = "There are 42 cats and 3.14 dogs."
        assert trained_tokenizer.decode(trained_tokenizer.encode(text)) == text

    def test_punctuation(self, trained_tokenizer):
        text = "Hello! How are you? I'm fine, thanks. (Really.)"
        assert trained_tokenizer.decode(trained_tokenizer.encode(text)) == text

    @pytest.mark.parametrize("text", [
        "café résumé naïve",
        "Ñoño año",
        "日本語テスト",
        "Привет мир",
        "مرحبا بالعالم",
        "🎉🚀💡",
        "Mixed: Hello 世界 🌍",
    ])
    def test_unicode(self, trained_tokenizer, text):
        """Byte-level BPE must handle any unicode — no 'unknown' tokens."""
        assert trained_tokenizer.decode(trained_tokenizer.encode(text)) == text

    def test_empty_string(self, trained_tokenizer):
        assert trained_tokenizer.encode("") == []
        assert trained_tokenizer.decode([]) == ""


# ============================================================
# Code Tokenization
# ============================================================

class TestCodeTokenization:
    def test_python_function(self, trained_tokenizer):
        code = 'def add(a, b):\n    return a + b'
        assert trained_tokenizer.decode(trained_tokenizer.encode(code)) == code

    def test_python_class(self, trained_tokenizer):
        code = 'class Foo:\n    def __init__(self):\n        self.x = 0'
        assert trained_tokenizer.decode(trained_tokenizer.encode(code)) == code

    def test_indentation_preserved(self, trained_tokenizer):
        code = "if True:\n    if True:\n        pass"
        decoded = trained_tokenizer.decode(trained_tokenizer.encode(code))
        assert decoded == code

    def test_special_chars_in_code(self, trained_tokenizer):
        code = 'x = {"key": [1, 2, 3]}'
        assert trained_tokenizer.decode(trained_tokenizer.encode(code)) == code

    def test_string_literals(self, trained_tokenizer):
        code = """print("Hello, world!")"""
        assert trained_tokenizer.decode(trained_tokenizer.encode(code)) == code


# ============================================================
# Special Tokens
# ============================================================

class TestSpecialTokens:
    def test_special_token_encodes_to_single_id(self, trained_tokenizer):
        ids = trained_tokenizer.encode("<|end_of_text|>")
        assert len(ids) == 1

    def test_special_token_roundtrip(self, trained_tokenizer):
        text = "<|begin_of_text|>Hello<|end_of_text|>"
        assert trained_tokenizer.decode(trained_tokenizer.encode(text)) == text

    def test_special_token_in_middle(self, trained_tokenizer):
        text = "before <|end_of_text|> after"
        assert trained_tokenizer.decode(trained_tokenizer.encode(text)) == text

    def test_multiple_special_tokens(self, trained_tokenizer):
        text = "<|begin_of_text|>code<|start_code|>def f(): pass<|end_code|><|end_of_text|>"
        assert trained_tokenizer.decode(trained_tokenizer.encode(text)) == text


# ============================================================
# Edge Cases
# ============================================================

class TestEdgeCases:
    def test_single_character(self, trained_tokenizer):
        for ch in "abcABC123!@#":
            assert trained_tokenizer.decode(trained_tokenizer.encode(ch)) == ch

    def test_long_text(self, trained_tokenizer):
        text = "The quick brown fox. " * 100
        assert trained_tokenizer.decode(trained_tokenizer.encode(text)) == text

    def test_repeated_characters(self, trained_tokenizer):
        text = "aaaaaaaaaa"
        assert trained_tokenizer.decode(trained_tokenizer.encode(text)) == text

    def test_only_whitespace(self, trained_tokenizer):
        text = "   "
        assert trained_tokenizer.decode(trained_tokenizer.encode(text)) == text

    def test_only_newlines(self, trained_tokenizer):
        text = "\n\n\n"
        assert trained_tokenizer.decode(trained_tokenizer.encode(text)) == text


# ============================================================
# Save/Load
# ============================================================

class TestPersistence:
    def test_save_and_load(self, trained_tokenizer):
        with tempfile.TemporaryDirectory() as tmpdir:
            trained_tokenizer.save(tmpdir)
            loaded = BPETokenizer.load(tmpdir)

            # Same vocab size
            assert len(loaded.vocab) == len(trained_tokenizer.vocab)
            assert len(loaded.merges) == len(trained_tokenizer.merges)
            assert loaded.special_tokens == trained_tokenizer.special_tokens

    def test_loaded_produces_same_encoding(self, trained_tokenizer):
        text = "The transformer uses attention."
        with tempfile.TemporaryDirectory() as tmpdir:
            trained_tokenizer.save(tmpdir)
            loaded = BPETokenizer.load(tmpdir)
            assert loaded.encode(text) == trained_tokenizer.encode(text)

    def test_loaded_roundtrip(self, trained_tokenizer):
        text = "def hello():\n    return 'world'"
        with tempfile.TemporaryDirectory() as tmpdir:
            trained_tokenizer.save(tmpdir)
            loaded = BPETokenizer.load(tmpdir)
            assert loaded.decode(loaded.encode(text)) == text


# ============================================================
# Compression & Properties
# ============================================================

class TestProperties:
    def test_compression_ratio(self, trained_tokenizer):
        """Tokenizer should compress English text (ratio > 1.5x)."""
        text = "The quick brown fox jumps over the lazy dog. " * 10
        ids = trained_tokenizer.encode(text)
        raw_bytes = len(text.encode("utf-8"))
        ratio = raw_bytes / len(ids)
        assert ratio > 1.5, f"Compression ratio {ratio:.2f}x is too low"

    def test_vocab_size_within_target(self, trained_tokenizer):
        """Vocab + special tokens should not exceed target."""
        total = len(trained_tokenizer.vocab) + len(trained_tokenizer.special_tokens)
        assert total <= trained_tokenizer.target_vocab_size + len(trained_tokenizer.special_tokens)

    def test_encode_returns_integers(self, trained_tokenizer):
        ids = trained_tokenizer.encode("Hello")
        assert all(isinstance(i, int) for i in ids)

    def test_all_ids_are_valid(self, trained_tokenizer):
        """Every ID from encode should be decodable."""
        ids = trained_tokenizer.encode("Test string with various tokens 123!")
        all_valid_ids = set(trained_tokenizer.vocab.keys()) | set(trained_tokenizer.special_tokens.values())
        for token_id in ids:
            assert token_id in all_valid_ids, f"Token ID {token_id} not in vocab or special tokens"

    def test_untrained_tokenizer_falls_back_to_bytes(self):
        """An untrained tokenizer should encode as raw bytes."""
        tok = BPETokenizer(vocab_size=256)
        text = "Hi"
        ids = tok.encode(text)
        assert ids == [72, 105]  # ASCII for 'H', 'i'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
