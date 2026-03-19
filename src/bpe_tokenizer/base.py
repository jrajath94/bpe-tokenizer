"""Abstract base tokenizer defining the encode/decode interface.

All concrete tokenizer implementations (BPE, WordPiece) must implement this
interface. This ensures they are interchangeable in downstream code.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from .vocab import TokenizerConfig, TrainingStats, Vocabulary

logger = logging.getLogger(__name__)


class BaseTokenizer(ABC):
    """Abstract base class for all tokenizer implementations.

    Defines the contract that BPE, WordPiece, and any future tokenizer
    must satisfy: train on a corpus, encode text to IDs, decode IDs back
    to text, and persist/load the trained vocabulary.

    Subclasses must implement: train, encode, decode, save, load.
    """

    def __init__(self) -> None:
        """Initialize an untrained tokenizer with an empty vocabulary."""
        self._vocab: Optional[Vocabulary] = None
        self._is_trained: bool = False

    @property
    def is_trained(self) -> bool:
        """Whether the tokenizer has been trained on a corpus."""
        return self._is_trained

    @property
    def vocab_size(self) -> int:
        """Number of tokens in the current vocabulary.

        Returns:
            0 if untrained, otherwise the vocabulary size.
        """
        return len(self._vocab) if self._vocab is not None else 0

    @abstractmethod
    def train(self, corpus: str, vocab_size: int) -> TrainingStats:
        """Train the tokenizer on a text corpus.

        Args:
            corpus: Raw text to train on. Should be representative of the
                    target domain.
            vocab_size: Target number of tokens in the final vocabulary.

        Returns:
            TrainingStats with metrics from the training run.
        """
        ...

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Encode text into a sequence of integer token IDs.

        Args:
            text: Input text to encode.

        Returns:
            List of integer token IDs representing the input.

        Raises:
            EncodingError: If the tokenizer is not trained or encoding fails.
        """
        ...

    @abstractmethod
    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs back to text.

        Args:
            ids: List of integer token IDs to decode.

        Returns:
            Reconstructed text string.

        Raises:
            DecodingError: If any ID is out of range for the current vocabulary.
        """
        ...

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist the trained tokenizer vocabulary to disk.

        Args:
            path: File path where the vocabulary JSON will be written.
        """
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "BaseTokenizer":
        """Load a trained tokenizer from a saved vocabulary file.

        Args:
            path: Path to the vocabulary JSON file.

        Returns:
            A fully initialized tokenizer ready for encode/decode.
        """
        ...

    def token_to_id(self, token: str) -> Optional[int]:
        """Look up the integer ID for a token string.

        Args:
            token: Token string to look up.

        Returns:
            Integer ID if found, None otherwise.
        """
        if self._vocab is None:
            return None
        return self._vocab.token_to_id.get(token)

    def id_to_token(self, token_id: int) -> Optional[str]:
        """Look up the token string for an integer ID.

        Args:
            token_id: Integer ID to look up.

        Returns:
            Token string if found, None otherwise.
        """
        if self._vocab is None:
            return None
        return self._vocab.id_to_token.get(token_id)

    @property
    def config(self) -> Optional[TokenizerConfig]:
        """The tokenizer configuration, or None if untrained."""
        return self._vocab.config if self._vocab else None

    def __repr__(self) -> str:
        """Return a human-readable representation of the tokenizer."""
        status = f"trained, vocab_size={self.vocab_size}" if self._is_trained else "untrained"
        return f"{self.__class__.__name__}({status})"
