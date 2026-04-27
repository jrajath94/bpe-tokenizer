"""Vocabulary management: build, save, load tokenizer vocabularies in JSON format.

A vocabulary maps token strings to integer IDs. This module handles
serialization/deserialization of vocabularies and merge rules so that
a trained tokenizer can be saved and reloaded without retraining.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .exceptions import TokenizerLoadError

logger = logging.getLogger(__name__)

# Version string embedded in saved vocabulary files for forward compatibility
VOCAB_FORMAT_VERSION = "1.0"


@dataclass
class MergeRule:
    """Represents a single BPE merge rule.

    A merge rule encodes the instruction "whenever you see token A followed
    immediately by token B, merge them into a new token AB". The priority
    is determined by the order in the merge list (lower index = applied first).

    Attributes:
        left: The left token of the pair.
        right: The right token of the pair.
        result: The merged token (usually left + right).
        rank: The merge priority (0 = highest priority, applied first).
    """

    left: str
    right: str
    result: str
    rank: int


@dataclass
class TokenizerConfig:
    """Configuration parameters for a trained tokenizer.

    Attributes:
        vocab_size: Total number of tokens in the vocabulary.
        tokenizer_type: One of 'bpe' or 'wordpiece'.
        pretokenizer_type: Pre-tokenization strategy used during training.
        unk_token: The unknown token string (WordPiece only).
        continuation_prefix: Subword continuation prefix (WordPiece only).
        version: The vocabulary format version.
    """

    vocab_size: int
    tokenizer_type: str
    pretokenizer_type: str = "gpt2"
    unk_token: str | None = None
    continuation_prefix: str | None = None
    version: str = VOCAB_FORMAT_VERSION


@dataclass
class TrainingStats:
    """Statistics collected during tokenizer training.

    Attributes:
        corpus_chars: Total characters in the training corpus.
        corpus_tokens: Number of pre-tokenized word tokens.
        initial_vocab_size: Number of unique characters/bytes at start.
        final_vocab_size: Target vocabulary size reached.
        num_merges: Total number of merge operations performed (BPE only).
        training_seconds: Wall-clock time for training.
        tokens_per_second: Approximate encoding throughput after training.
    """

    corpus_chars: int = 0
    corpus_tokens: int = 0
    initial_vocab_size: int = 0
    final_vocab_size: int = 0
    num_merges: int = 0
    training_seconds: float = 0.0
    tokens_per_second: float = 0.0


@dataclass
class Vocabulary:
    """Container for a tokenizer vocabulary and its associated metadata.

    Attributes:
        token_to_id: Mapping from token string to integer ID.
        id_to_token: Mapping from integer ID to token string.
        merge_rules: Ordered list of merge rules (BPE only).
        config: Tokenizer configuration.
        stats: Training statistics.
    """

    token_to_id: dict[str, int] = field(default_factory=dict)
    id_to_token: dict[int, str] = field(default_factory=dict)
    merge_rules: list[MergeRule] = field(default_factory=list)
    config: TokenizerConfig = field(
        default_factory=lambda: TokenizerConfig(vocab_size=0, tokenizer_type="bpe")
    )
    stats: TrainingStats = field(default_factory=TrainingStats)

    def add_token(self, token: str) -> int:
        """Add a token to the vocabulary and return its ID.

        If the token is already present, its existing ID is returned.
        New tokens are assigned IDs sequentially.

        Args:
            token: The token string to add.

        Returns:
            The integer ID assigned to this token.
        """
        if token in self.token_to_id:
            return self.token_to_id[token]

        new_id = len(self.token_to_id)
        self.token_to_id[token] = new_id
        self.id_to_token[new_id] = token
        return new_id

    def __len__(self) -> int:
        """Return the number of tokens in the vocabulary."""
        return len(self.token_to_id)


def save_vocabulary(vocab: Vocabulary, path: Path) -> None:
    """Serialize a vocabulary to a JSON file.

    The saved format includes the vocabulary mapping, merge rules, and
    configuration so that the tokenizer can be fully reconstructed.

    Args:
        vocab: The vocabulary object to save.
        path: Destination file path (will be created or overwritten).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "version": VOCAB_FORMAT_VERSION,
        "config": asdict(vocab.config),
        "stats": asdict(vocab.stats),
        "token_to_id": vocab.token_to_id,
        "merge_rules": [
            {"left": r.left, "right": r.right, "result": r.result, "rank": r.rank}
            for r in vocab.merge_rules
        ],
    }

    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    logger.info("Saved vocabulary (%d tokens) to %s", len(vocab), path)


def load_vocabulary(path: Path) -> Vocabulary:
    """Load a vocabulary from a JSON file.

    Args:
        path: Path to the saved vocabulary JSON file.

    Returns:
        A fully reconstructed Vocabulary object.

    Raises:
        TokenizerLoadError: If the file is missing, malformed, or has an
            incompatible version.
    """
    path = Path(path)

    if not path.exists():
        raise TokenizerLoadError(f"Vocabulary file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        raise TokenizerLoadError(f"Failed to read vocabulary file: {exc}") from exc

    # Validate version compatibility
    saved_version = payload.get("version", "unknown")
    if saved_version != VOCAB_FORMAT_VERSION:
        logger.warning(
            "Version mismatch: saved=%s, current=%s. Attempting to load anyway.",
            saved_version,
            VOCAB_FORMAT_VERSION,
        )

    try:
        config_data = payload["config"]
        config = TokenizerConfig(**config_data)

        stats_data = payload.get("stats", {})
        stats = TrainingStats(**stats_data)

        token_to_id: dict[str, int] = payload["token_to_id"]
        id_to_token: dict[int, str] = {int(v): k for k, v in token_to_id.items()}

        merge_rules = [
            MergeRule(
                left=r["left"],
                right=r["right"],
                result=r["result"],
                rank=r["rank"],
            )
            for r in payload.get("merge_rules", [])
        ]

    except (KeyError, TypeError) as exc:
        raise TokenizerLoadError(f"Malformed vocabulary file: {exc}") from exc

    vocab = Vocabulary(
        token_to_id=token_to_id,
        id_to_token=id_to_token,
        merge_rules=merge_rules,
        config=config,
        stats=stats,
    )

    logger.info("Loaded vocabulary (%d tokens) from %s", len(vocab), path)
    return vocab
