"""Shared test fixtures for the bpe-tokenizer test suite."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from bpe_tokenizer import BPETokenizer, WordPieceTokenizer

# ---------------------------------------------------------------------------
# Corpus fixtures
# ---------------------------------------------------------------------------

SMALL_CORPUS = textwrap.dedent("""\
    the quick brown fox jumps over the lazy dog
    the dog barked at the fox in the forest
    machine learning tokenization natural language processing
    byte pair encoding vocabulary subword units
    the cat sat on the mat the rat ate the hat
    learning the alphabet from a to z
    tokenization is the foundation of language models
    understanding tokens helps debug production issues
    the quick fox the lazy dog the brown cat
    encoding decoding vocabulary tokens subwords characters
""")

MEDIUM_CORPUS = SMALL_CORPUS * 20  # Larger for better training

UNICODE_CORPUS = textwrap.dedent("""\
    Hello world café résumé naïve
    the café served crêpes with crème brûlée
    日本語テスト Chinese 中文 Korean 한국어
    emoji test simple words the quick brown fox
    arithmetic 1+1=2 and 2*3=6 and 10/2=5
    hello world foo bar baz qux
""")


# ---------------------------------------------------------------------------
# Trained tokenizer fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def bpe_tokenizer_small() -> BPETokenizer:
    """A BPE tokenizer trained on a small corpus, vocab_size=700."""
    tokenizer = BPETokenizer()
    tokenizer.train(SMALL_CORPUS, vocab_size=700)
    return tokenizer


@pytest.fixture(scope="session")
def bpe_tokenizer_medium() -> BPETokenizer:
    """A BPE tokenizer trained on a medium corpus, vocab_size=800."""
    tokenizer = BPETokenizer()
    tokenizer.train(MEDIUM_CORPUS, vocab_size=800)
    return tokenizer


@pytest.fixture(scope="session")
def wordpiece_tokenizer_small() -> WordPieceTokenizer:
    """A WordPiece tokenizer trained on a small corpus, vocab_size=300."""
    tokenizer = WordPieceTokenizer()
    tokenizer.train(SMALL_CORPUS, vocab_size=300)
    return tokenizer


@pytest.fixture(scope="session")
def wordpiece_tokenizer_medium() -> WordPieceTokenizer:
    """A WordPiece tokenizer trained on a medium corpus, vocab_size=400."""
    tokenizer = WordPieceTokenizer()
    tokenizer.train(MEDIUM_CORPUS, vocab_size=400)
    return tokenizer


# ---------------------------------------------------------------------------
# Temp directory fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_vocab_path(tmp_path: Path) -> Path:
    """Provide a temporary path for saving vocabulary files."""
    return tmp_path / "vocab.json"
