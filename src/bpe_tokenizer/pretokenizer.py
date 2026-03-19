"""Pre-tokenization: split raw text into words before subword tokenization.

Pre-tokenization determines what boundaries the BPE/WordPiece algorithms
are allowed to merge across. GPT-2 style never merges across word boundaries;
BERT-style splits on whitespace and punctuation.

The pre-tokenizer runs BEFORE the main tokenization algorithm and its output
is a list of string "words" that are then split into subwords.
"""

from __future__ import annotations

import logging
import re
from enum import Enum, auto
from typing import Callable

from .exceptions import PreTokenizationError
from .unicode_utils import is_punctuation, is_whitespace

logger = logging.getLogger(__name__)

# GPT-2's original regex pattern for pre-tokenization.
# This splits text into words in a way that never merges:
#  - contractions like 's, 't, 're, 've, 'll, 'd
#  - words with optional leading space
#  - numbers with optional leading space
#  - any non-whitespace character sequence
GPT2_PATTERN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w\d]+|\s+(?!\S)|\s+""",
    re.UNICODE,
)

# Simpler whitespace-based split (used for BERT-style)
WHITESPACE_PATTERN = re.compile(r"\S+", re.UNICODE)


class PreTokenizerType(Enum):
    """Supported pre-tokenizer strategies."""

    WHITESPACE = auto()  # Split on whitespace only (basic)
    GPT2 = auto()  # GPT-2 regex (handles contractions, punctuation)
    BERT = auto()  # BERT-style: whitespace + punctuation split


def _whitespace_split(text: str) -> list[str]:
    """Split text on whitespace boundaries.

    Args:
        text: Input text.

    Returns:
        List of non-whitespace tokens.
    """
    return WHITESPACE_PATTERN.findall(text)


def _gpt2_split(text: str) -> list[str]:
    """Split text using the GPT-2 regex pre-tokenization pattern.

    This preserves leading spaces as part of the word token, which is
    why GPT-2-style tokenizers produce tokens like " hello" (with space).

    Args:
        text: Input text.

    Returns:
        List of word-level tokens.
    """
    return GPT2_PATTERN.findall(text)


def _bert_split(text: str) -> list[str]:
    """Split text using BERT-style pre-tokenization.

    Steps:
    1. Split on whitespace
    2. Within each chunk, split on punctuation boundaries

    Args:
        text: Input text.

    Returns:
        List of word/punctuation tokens (all lowercase for BERT).
    """
    tokens: list[str] = []
    for chunk in text.split():
        # Lower-case (BERT is uncased by default)
        chunk = chunk.lower()
        # Split on punctuation boundaries
        current = ""
        for char in chunk:
            if is_punctuation(char):
                if current:
                    tokens.append(current)
                    current = ""
                tokens.append(char)
            else:
                current += char
        if current:
            tokens.append(current)
    return tokens


# Dispatcher mapping PreTokenizerType to its implementation function
_PRETOKENIZER_DISPATCH: dict[PreTokenizerType, Callable[[str], list[str]]] = {
    PreTokenizerType.WHITESPACE: _whitespace_split,
    PreTokenizerType.GPT2: _gpt2_split,
    PreTokenizerType.BERT: _bert_split,
}


class PreTokenizer:
    """Pre-tokenizer that splits raw text into word-level tokens.

    The pre-tokenizer is the first step in the tokenization pipeline.
    It determines which boundaries the subword algorithm (BPE or WordPiece)
    is NOT allowed to cross — e.g., BPE never merges tokens across different
    "words" as defined by the pre-tokenizer.

    Example:
        >>> pt = PreTokenizer(PreTokenizerType.GPT2)
        >>> pt.tokenize("Hello, world!")
        ['Hello', ',', ' world', '!']
    """

    def __init__(self, tokenizer_type: PreTokenizerType = PreTokenizerType.GPT2) -> None:
        """Initialize the pre-tokenizer.

        Args:
            tokenizer_type: The pre-tokenization strategy to use.
        """
        self.tokenizer_type = tokenizer_type
        self._split_fn = _PRETOKENIZER_DISPATCH[tokenizer_type]
        logger.debug("Initialized PreTokenizer with strategy: %s", tokenizer_type.name)

    def __repr__(self) -> str:
        """Return a human-readable representation of this pre-tokenizer."""
        return f"PreTokenizer(type={self.tokenizer_type.name})"

    def tokenize(self, text: str) -> list[str]:
        """Split text into word-level tokens.

        Args:
            text: Input text to pre-tokenize.

        Returns:
            List of word-level string tokens.

        Raises:
            PreTokenizationError: If the input text cannot be tokenized.
        """
        if not isinstance(text, str):
            raise PreTokenizationError(f"Expected str, got {type(text).__name__}")

        if not text:
            return []

        try:
            tokens = self._split_fn(text)
        except Exception as exc:
            raise PreTokenizationError(f"Pre-tokenization failed: {exc}") from exc

        logger.debug("Pre-tokenized %d chars into %d tokens", len(text), len(tokens))
        return tokens
