"""bpe-tokenizer: BPE and WordPiece tokenization from scratch.

Clean, readable implementations of Byte-Pair Encoding (GPT-2 style) and
WordPiece (BERT-style) tokenization that demonstrate how modern LLM
tokenizers work under the hood.

Quick start:
    >>> from bpe_tokenizer import BPETokenizer
    >>> tokenizer = BPETokenizer()
    >>> tokenizer.train("your corpus text here...", vocab_size=1000)
    >>> ids = tokenizer.encode("Hello, world!")
    >>> tokenizer.decode(ids)
    'Hello, world!'
"""

from .base import BaseTokenizer
from .bpe import BPETokenizer
from .exceptions import (
    DecodingError,
    EncodingError,
    PreTokenizationError,
    TokenizerError,
    TokenizerLoadError,
    VocabBuildError,
)
from .pretokenizer import PreTokenizer, PreTokenizerType
from .vocab import MergeRule, TokenizerConfig, TrainingStats, Vocabulary
from .wordpiece import WordPieceTokenizer

__version__ = "0.1.0"
__author__ = "Rajath John"
__email__ = "jrajath94@gmail.com"

__all__ = [
    # Core tokenizers
    "BPETokenizer",
    "WordPieceTokenizer",
    "BaseTokenizer",
    # Pre-tokenizer
    "PreTokenizer",
    "PreTokenizerType",
    # Data structures
    "Vocabulary",
    "MergeRule",
    "TokenizerConfig",
    "TrainingStats",
    # Exceptions
    "TokenizerError",
    "VocabBuildError",
    "EncodingError",
    "DecodingError",
    "TokenizerLoadError",
    "PreTokenizationError",
]
