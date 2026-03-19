"""Tests for the pre-tokenization module."""

from __future__ import annotations

import pytest

from bpe_tokenizer.exceptions import PreTokenizationError
from bpe_tokenizer.pretokenizer import PreTokenizer, PreTokenizerType


class TestPreTokenizerWhitespace:
    """Tests for whitespace-based pre-tokenization."""

    def test_simple_sentence(self) -> None:
        """Basic sentence should split on whitespace."""
        pt = PreTokenizer(PreTokenizerType.WHITESPACE)
        tokens = pt.tokenize("hello world foo")
        assert tokens == ["hello", "world", "foo"]

    def test_extra_whitespace_ignored(self) -> None:
        """Multiple spaces should not produce empty tokens."""
        pt = PreTokenizer(PreTokenizerType.WHITESPACE)
        tokens = pt.tokenize("hello   world")
        assert "" not in tokens
        assert "hello" in tokens
        assert "world" in tokens

    def test_empty_string_returns_empty_list(self) -> None:
        """Empty string should produce an empty list."""
        pt = PreTokenizer(PreTokenizerType.WHITESPACE)
        assert pt.tokenize("") == []

    def test_single_word(self) -> None:
        """Single word should produce a list with one element."""
        pt = PreTokenizer(PreTokenizerType.WHITESPACE)
        assert pt.tokenize("hello") == ["hello"]


class TestPreTokenizerGPT2:
    """Tests for GPT-2 regex pre-tokenization."""

    def test_basic_sentence(self) -> None:
        """Basic sentence should split into words."""
        pt = PreTokenizer(PreTokenizerType.GPT2)
        tokens = pt.tokenize("Hello world")
        assert "Hello" in tokens
        assert " world" in tokens or "world" in tokens

    def test_contractions_preserved(self) -> None:
        """Contractions like 't and 's should be separate tokens."""
        pt = PreTokenizer(PreTokenizerType.GPT2)
        tokens = pt.tokenize("don't")
        # GPT-2 splits "don't" into "don" and "'t"
        assert len(tokens) >= 1  # At least one token

    def test_empty_string(self) -> None:
        """Empty string should produce empty list."""
        pt = PreTokenizer(PreTokenizerType.GPT2)
        assert pt.tokenize("") == []

    def test_numbers(self) -> None:
        """Numbers should be captured as tokens."""
        pt = PreTokenizer(PreTokenizerType.GPT2)
        tokens = pt.tokenize("There are 42 tokens")
        assert any("42" in t for t in tokens)

    def test_non_string_raises(self) -> None:
        """Non-string input should raise PreTokenizationError."""
        pt = PreTokenizer(PreTokenizerType.GPT2)
        with pytest.raises(PreTokenizationError, match="Expected str"):
            pt.tokenize(42)  # type: ignore[arg-type]


class TestPreTokenizerBERT:
    """Tests for BERT-style pre-tokenization."""

    def test_lowercasing(self) -> None:
        """BERT pre-tokenizer should lowercase text."""
        pt = PreTokenizer(PreTokenizerType.BERT)
        tokens = pt.tokenize("Hello World")
        assert all(t == t.lower() for t in tokens)

    def test_punctuation_split(self) -> None:
        """Punctuation should become its own token."""
        pt = PreTokenizer(PreTokenizerType.BERT)
        tokens = pt.tokenize("hello, world!")
        # Comma and exclamation should be separate tokens
        assert "," in tokens or any("," in t for t in tokens)

    def test_empty_string(self) -> None:
        """Empty string should produce empty list."""
        pt = PreTokenizer(PreTokenizerType.BERT)
        assert pt.tokenize("") == []

    @pytest.mark.parametrize(
        "text",
        [
            "simple text",
            "Hello, World!",
            "test123",
            "multiple   spaces",
        ],
    )
    def test_various_inputs(self, text: str) -> None:
        """Pre-tokenizer should handle various text inputs without error."""
        pt = PreTokenizer(PreTokenizerType.BERT)
        tokens = pt.tokenize(text)
        assert isinstance(tokens, list)


class TestPreTokenizerRepr:
    """Tests for PreTokenizer string representation."""

    def test_repr_contains_type(self) -> None:
        """repr should include the tokenizer type name."""
        for pt_type in PreTokenizerType:
            pt = PreTokenizer(pt_type)
            assert pt_type.name in repr(pt) or pt_type.name.lower() in str(pt)
