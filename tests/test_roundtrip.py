"""Roundtrip tests: decode(encode(text)) must equal the original text.

This is the most important correctness property of any tokenizer.
A tokenizer that fails roundtrip is lossy and cannot be used in production.
"""

from __future__ import annotations

import pytest

from bpe_tokenizer import BPETokenizer, WordPieceTokenizer

from .conftest import SMALL_CORPUS


# ---------------------------------------------------------------------------
# BPE roundtrip tests
# ---------------------------------------------------------------------------


class TestBPERoundtrip:
    """Roundtrip tests for BPETokenizer: decode(encode(text)) == text."""

    @pytest.mark.parametrize(
        "text",
        [
            "hello",
            "world",
            "the quick brown fox",
            "Hello, World!",
            "machine learning tokenization",
            "the lazy dog",
            "byte pair encoding",
            "natural language processing",
            "a b c d e f g",
            "aaabbbccc",
        ],
    )
    def test_roundtrip_ascii_text(
        self, bpe_tokenizer_small: BPETokenizer, text: str
    ) -> None:
        """BPE decode(encode(text)) should equal original for ASCII text."""
        ids = bpe_tokenizer_small.encode(text)
        decoded = bpe_tokenizer_small.decode(ids)
        assert decoded == text, (
            f"Roundtrip failed for {text!r}: "
            f"got {decoded!r}"
        )

    def test_roundtrip_with_numbers(
        self, bpe_tokenizer_small: BPETokenizer
    ) -> None:
        """Numbers should survive the roundtrip."""
        text = "there are 100 tokens in the vocabulary"
        ids = bpe_tokenizer_small.encode(text)
        decoded = bpe_tokenizer_small.decode(ids)
        assert decoded == text

    def test_roundtrip_punctuation(
        self, bpe_tokenizer_small: BPETokenizer
    ) -> None:
        """Text with punctuation should survive roundtrip."""
        text = "Hello, world! How are you?"
        ids = bpe_tokenizer_small.encode(text)
        decoded = bpe_tokenizer_small.decode(ids)
        assert decoded == text

    def test_roundtrip_single_character(
        self, bpe_tokenizer_small: BPETokenizer
    ) -> None:
        """Single characters should roundtrip correctly."""
        for char in "abcdefghijklmnopqrstuvwxyz":
            ids = bpe_tokenizer_small.encode(char)
            decoded = bpe_tokenizer_small.decode(ids)
            assert decoded == char, f"Roundtrip failed for character {char!r}"

    def test_roundtrip_repeated_word(
        self, bpe_tokenizer_small: BPETokenizer
    ) -> None:
        """Repeated words should roundtrip correctly."""
        text = "the the the fox fox fox"
        ids = bpe_tokenizer_small.encode(text)
        decoded = bpe_tokenizer_small.decode(ids)
        assert decoded == text

    def test_roundtrip_unicode_ascii_subset(
        self, bpe_tokenizer_small: BPETokenizer
    ) -> None:
        """Basic ASCII with common punctuation should roundtrip."""
        text = "Hello World foo bar baz"
        ids = bpe_tokenizer_small.encode(text)
        decoded = bpe_tokenizer_small.decode(ids)
        assert decoded == text

    def test_roundtrip_preserves_whitespace_structure(
        self, bpe_tokenizer_small: BPETokenizer
    ) -> None:
        """Internal whitespace should be preserved in roundtrip."""
        text = "hello world"  # single space
        ids = bpe_tokenizer_small.encode(text)
        decoded = bpe_tokenizer_small.decode(ids)
        assert decoded == text

    @pytest.mark.parametrize(
        "text",
        [
            "tokenization is the foundation",
            "encoding and decoding vocabulary",
            "byte pair merging algorithm",
            "subword units for language models",
            "the dog the cat the fox",
        ],
    )
    def test_roundtrip_corpus_sentences(
        self, bpe_tokenizer_medium: BPETokenizer, text: str
    ) -> None:
        """Sentences from the training corpus should roundtrip perfectly."""
        ids = bpe_tokenizer_medium.encode(text)
        decoded = bpe_tokenizer_medium.decode(ids)
        assert decoded == text


# ---------------------------------------------------------------------------
# WordPiece roundtrip tests
# ---------------------------------------------------------------------------


class TestWordPieceRoundtrip:
    """Roundtrip tests for WordPieceTokenizer.

    Note: WordPiece is lossy for words containing only unknown characters.
    We test roundtrip on words the tokenizer was trained on.
    """

    @pytest.mark.parametrize(
        "text",
        [
            "the",
            "quick",
            "brown",
            "fox",
            "the quick",
            "machine learning",
            "tokenization",
            "the dog barked",
        ],
    )
    def test_roundtrip_training_vocabulary(
        self, wordpiece_tokenizer_medium: WordPieceTokenizer, text: str
    ) -> None:
        """Words seen during training should survive decode(encode(x)) losslessly.

        WordPiece is character-based so unknown characters produce [UNK],
        but known words should reconstruct perfectly.
        """
        ids = wordpiece_tokenizer_medium.encode(text)
        decoded = wordpiece_tokenizer_medium.decode(ids)

        # WordPiece may alter spacing slightly (adds spaces between tokens),
        # so we compare stripped versions for in-vocabulary text
        assert decoded.strip() == text.strip(), (
            f"WordPiece roundtrip failed for {text!r}: got {decoded!r}"
        )

    def test_roundtrip_empty_string(
        self, wordpiece_tokenizer_small: WordPieceTokenizer
    ) -> None:
        """Empty string should roundtrip to empty string."""
        assert wordpiece_tokenizer_small.decode(
            wordpiece_tokenizer_small.encode("")
        ) == ""
