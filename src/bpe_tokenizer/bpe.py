"""Byte-Pair Encoding (BPE) tokenizer implementation.

BPE is a data compression algorithm adapted for text tokenization by Sennrich
et al. (2016). The core insight is to iteratively merge the most frequently
co-occurring pair of symbols in the corpus, growing the vocabulary from
individual characters up to the target size.

Algorithm (Sennrich et al., 2016):
    1. Initialize: split each training word into characters + end-of-word marker
    2. Count all adjacent symbol pairs in the corpus (weighted by word frequency)
    3. Merge the most frequent pair into a new compound symbol
    4. Repeat steps 2-3 until reaching the target vocabulary size
    5. At inference: apply the learned merge rules in priority order

References:
    Sennrich, R., Haddow, B., & Birch, A. (2016).
    Neural Machine Translation of Rare Words with Subword Units.
    ACL 2016. https://arxiv.org/abs/1508.07909
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

from .base import BaseTokenizer
from .exceptions import DecodingError, EncodingError, VocabBuildError
from .pretokenizer import PreTokenizer, PreTokenizerType
from .unicode_utils import bytes_to_text, text_to_bytes
from .vocab import (
    MergeRule,
    TokenizerConfig,
    TrainingStats,
    Vocabulary,
    load_vocabulary,
    save_vocabulary,
)

logger = logging.getLogger(__name__)

# Special end-of-word marker appended to the last character of each word
# during BPE training. This lets the model distinguish "the" as a standalone
# word from "the" as a prefix of "then", "there", etc.
END_OF_WORD = "</w>"


def _build_word_freqs(
    pretokenized: list[str],
    use_byte_level: bool = True,
) -> dict[tuple[str, ...], int]:
    """Convert pre-tokenized words into BPE character sequences with frequencies.

    Each word is split into individual characters (or byte-level characters for
    byte-BPE). The last character gets the END_OF_WORD marker appended.

    Args:
        pretokenized: List of word-level tokens from the pre-tokenizer.
        use_byte_level: If True, convert each word to byte-level Unicode chars
            first, ensuring no unknown characters exist.

    Returns:
        Dictionary mapping character-tuple representations of words to their
        frequency count in the corpus.
    """
    word_freqs: dict[tuple[str, ...], int] = defaultdict(int)

    for word in pretokenized:
        if use_byte_level:
            chars = text_to_bytes(word)
        else:
            chars = list(word)

        if not chars:
            continue

        # Append END_OF_WORD to the last character to mark word boundaries
        char_tuple = tuple(chars[:-1]) + (chars[-1] + END_OF_WORD,)
        word_freqs[char_tuple] += 1

    return word_freqs


def _get_pair_frequencies(
    word_freqs: dict[tuple[str, ...], int],
) -> dict[tuple[str, str], int]:
    """Count the frequency of every adjacent symbol pair across all words.

    This is the core of the BPE inner loop. For a word appearing N times,
    every pair within it contributes N to the pair's total frequency.

    Args:
        word_freqs: Current word representations with their corpus frequencies.

    Returns:
        Dictionary mapping (left_symbol, right_symbol) pairs to total frequency.
    """
    pair_freqs: dict[tuple[str, str], int] = defaultdict(int)

    for word_tuple, freq in word_freqs.items():
        for i in range(len(word_tuple) - 1):
            pair_freqs[(word_tuple[i], word_tuple[i + 1])] += freq

    return pair_freqs


def _merge_pair(
    word_freqs: dict[tuple[str, ...], int],
    pair: tuple[str, str],
    merged: str,
) -> dict[tuple[str, ...], int]:
    """Apply a single merge operation to all words in the corpus.

    For each word that contains the given adjacent pair, replace all
    occurrences of the pair with the merged symbol.

    Args:
        word_freqs: Current word representations.
        pair: The (left, right) symbol pair to merge.
        merged: The new token that replaces the pair.

    Returns:
        Updated word representations with the merge applied.
    """
    new_word_freqs: dict[tuple[str, ...], int] = {}

    left, right = pair

    for word_tuple, freq in word_freqs.items():
        new_symbols: list[str] = []
        i = 0
        # Linear scan, merging all occurrences of the pair in this word
        while i < len(word_tuple):
            if i < len(word_tuple) - 1 and word_tuple[i] == left and word_tuple[i + 1] == right:
                new_symbols.append(merged)
                i += 2  # Skip both symbols — we consumed the pair
            else:
                new_symbols.append(word_tuple[i])
                i += 1

        new_word_freqs[tuple(new_symbols)] = freq

    return new_word_freqs


class BPETokenizer(BaseTokenizer):
    """Byte-Pair Encoding tokenizer.

    Implements the BPE algorithm with byte-level encoding (GPT-2 style)
    to handle any Unicode text without unknown tokens. Byte-level BPE
    is the approach used in GPT-2, RoBERTa, and other OpenAI models.

    The tokenizer operates in two phases:
    - Training: Build a vocabulary and ordered merge rules from a corpus.
    - Inference: Apply merge rules in rank order to encode new text.

    Example:
        >>> tokenizer = BPETokenizer()
        >>> stats = tokenizer.train("Hello, world! " * 100, vocab_size=300)
        >>> ids = tokenizer.encode("Hello")
        >>> text = tokenizer.decode(ids)
        >>> assert text == "Hello"
    """

    def __init__(self) -> None:
        """Initialize an untrained BPE tokenizer."""
        super().__init__()
        # Ordered list of merge rules: (rank, (left, right)) → merged token
        # Lower rank = applied first during encoding
        self._merge_rank: dict[tuple[str, str], int] = {}
        self._pretokenizer: Optional[PreTokenizer] = None

    def train(
        self,
        corpus: str,
        vocab_size: int,
        pretokenizer_type: PreTokenizerType = PreTokenizerType.GPT2,
    ) -> TrainingStats:
        """Train BPE on a text corpus.

        Initializes the vocabulary with byte-level characters, then iteratively
        merges the most frequent pair until reaching the target vocab size.

        Args:
            corpus: Raw text to train on.
            vocab_size: Target vocabulary size (must be > 256 for byte-level BPE).
            pretokenizer_type: Pre-tokenization strategy.

        Returns:
            TrainingStats with training metrics.

        Raises:
            VocabBuildError: If vocab_size is invalid or corpus is empty.
        """
        if vocab_size <= 512:
            raise VocabBuildError(
                f"vocab_size must be > 512 for byte-level BPE (got {vocab_size}). "
                "The first 512 tokens are the byte-level characters and their END_OF_WORD variants."
            )

        if not corpus.strip():
            raise VocabBuildError("Cannot train on an empty corpus.")

        logger.info("Starting BPE training: target vocab_size=%d", vocab_size)
        start_time = time.perf_counter()

        self._pretokenizer = PreTokenizer(pretokenizer_type)
        pretokenized = self._pretokenizer.tokenize(corpus)
        logger.info("Pre-tokenized corpus into %d word tokens", len(pretokenized))

        # Build initial byte-level vocabulary (256 possible bytes)
        vocab = Vocabulary()
        from .unicode_utils import build_byte_to_unicode

        byte_to_unicode = build_byte_to_unicode()

        # Add all 256 raw byte tokens as the initial vocabulary
        for byte_val in range(256):
            vocab.add_token(byte_to_unicode[byte_val])

        # Also add the END_OF_WORD variants of every byte token.
        # A word like "hello" has its last char encoded as 'o</w>' (not 'o').
        # Without 'o</w>' in the initial vocab, encoding fails for unseen
        # word-final characters that never participate in any merge.
        for byte_val in range(256):
            vocab.add_token(byte_to_unicode[byte_val] + END_OF_WORD)

        initial_vocab_size = len(vocab)
        logger.debug("Initial byte vocabulary: %d tokens", initial_vocab_size)

        # Build word frequency table
        word_freqs = _build_word_freqs(pretokenized, use_byte_level=True)
        num_merges_needed = vocab_size - initial_vocab_size

        if num_merges_needed <= 0:
            raise VocabBuildError(
                f"Target vocab_size={vocab_size} is not larger than the initial "
                f"byte vocabulary ({initial_vocab_size}). Increase vocab_size."
            )

        # Core BPE merge loop
        merge_rules: list[MergeRule] = []
        for merge_idx in range(num_merges_needed):
            pair_freqs = _get_pair_frequencies(word_freqs)

            if not pair_freqs:
                logger.warning(
                    "No more pairs to merge at step %d. Stopping early.", merge_idx
                )
                break

            # Select the best pair: highest frequency, ties broken alphabetically
            # for determinism (important for reproducibility)
            best_pair = max(pair_freqs, key=lambda p: (pair_freqs[p], p))
            best_freq = pair_freqs[best_pair]

            if best_freq < 2:
                # Merging a pair that appears only once cannot improve compression
                logger.debug("Best pair frequency is 1, stopping early at step %d", merge_idx)
                break

            merged_token = best_pair[0] + best_pair[1]
            merge_rules.append(
                MergeRule(
                    left=best_pair[0],
                    right=best_pair[1],
                    result=merged_token,
                    rank=merge_idx,
                )
            )
            vocab.add_token(merged_token)
            self._merge_rank[best_pair] = merge_idx

            word_freqs = _merge_pair(word_freqs, best_pair, merged_token)

            if (merge_idx + 1) % 500 == 0:
                logger.debug(
                    "Merge %d/%d: '%s' + '%s' → '%s' (freq=%d)",
                    merge_idx + 1,
                    num_merges_needed,
                    best_pair[0],
                    best_pair[1],
                    merged_token,
                    best_freq,
                )

        elapsed = time.perf_counter() - start_time
        logger.info(
            "BPE training complete: %d merges in %.2fs, final vocab=%d",
            len(merge_rules),
            elapsed,
            len(vocab),
        )

        config = TokenizerConfig(
            vocab_size=len(vocab),
            tokenizer_type="bpe",
            pretokenizer_type=pretokenizer_type.name.lower(),
        )
        stats = TrainingStats(
            corpus_chars=len(corpus),
            corpus_tokens=len(pretokenized),
            initial_vocab_size=initial_vocab_size,
            final_vocab_size=len(vocab),
            num_merges=len(merge_rules),
            training_seconds=elapsed,
        )
        vocab.config = config
        vocab.stats = stats
        vocab.merge_rules = merge_rules

        self._vocab = vocab
        self._is_trained = True
        return stats

    def _apply_merges_to_word(self, word_chars: list[str]) -> list[str]:
        """Apply all learned merge rules to a single word's character list.

        Uses the merge rank dictionary for O(1) rank lookups, applying merges
        in rank order (lowest rank = applied first). This is equivalent to
        finding the highest-priority adjacent pair and merging, then repeating.

        Args:
            word_chars: List of byte-level character strings for one word.

        Returns:
            List of merged token strings for this word.
        """
        # We use a "find minimum rank pair" approach for correctness.
        # At each step: find the pair with the lowest rank, merge it.
        symbols = list(word_chars)

        while len(symbols) > 1:
            # Find the highest-priority (lowest rank) pair
            best_rank = float("inf")
            best_idx = -1

            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                rank = self._merge_rank.get(pair, float("inf"))
                if rank < best_rank:
                    best_rank = rank
                    best_idx = i

            if best_idx == -1 or best_rank == float("inf"):
                # No more applicable merges
                break

            # Merge at best_idx
            merged = symbols[best_idx] + symbols[best_idx + 1]
            symbols = symbols[:best_idx] + [merged] + symbols[best_idx + 2 :]

        return symbols

    def encode(self, text: str) -> list[int]:
        """Encode text into token IDs using learned BPE merge rules.

        Steps:
        1. Pre-tokenize into words
        2. Convert each word to byte-level characters
        3. Apply merge rules in rank order
        4. Look up each resulting token in the vocabulary

        Args:
            text: Input text to encode.

        Returns:
            List of integer token IDs.

        Raises:
            EncodingError: If the tokenizer is not trained or a token is missing
                from the vocabulary.
        """
        if not self._is_trained or self._vocab is None:
            raise EncodingError("Tokenizer is not trained. Call .train() first.")

        if not text:
            return []

        pretokenized = self._pretokenizer.tokenize(text)
        all_ids: list[int] = []

        for word in pretokenized:
            if not word:
                continue

            # Convert to byte-level characters
            byte_chars = text_to_bytes(word)

            # Mark end-of-word on the last character
            byte_chars[-1] = byte_chars[-1] + END_OF_WORD

            # Apply merge rules
            merged_tokens = self._apply_merges_to_word(byte_chars)

            # Look up each merged token in the vocabulary
            for token in merged_tokens:
                token_id = self._vocab.token_to_id.get(token)
                if token_id is None:
                    raise EncodingError(
                        f"Token '{token}' not found in vocabulary. "
                        "This should not happen with byte-level BPE."
                    )
                all_ids.append(token_id)

        return all_ids

    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs back to text.

        Reverses the encode process:
        1. Look up each ID in the vocabulary
        2. Remove END_OF_WORD markers
        3. Reconstruct byte sequences and decode as UTF-8

        Args:
            ids: List of integer token IDs.

        Returns:
            Decoded text string.

        Raises:
            DecodingError: If any ID is out of range.
        """
        if not self._is_trained or self._vocab is None:
            raise DecodingError("Tokenizer is not trained. Call .train() first.")

        if not ids:
            return ""

        tokens: list[str] = []
        for token_id in ids:
            token = self._vocab.id_to_token.get(token_id)
            if token is None:
                raise DecodingError(
                    f"Token ID {token_id} is out of range for vocabulary "
                    f"of size {len(self._vocab)}."
                )
            tokens.append(token)

        # Join all tokens into one string, removing END_OF_WORD markers
        # The END_OF_WORD marker acts as a word boundary: we replace it
        # with nothing (the actual space was part of the next word's
        # byte encoding in GPT-2 style).
        combined = "".join(tokens).replace(END_OF_WORD, "")

        # Decode byte-level characters back to UTF-8 text
        try:
            return bytes_to_text(list(combined))
        except (KeyError, UnicodeDecodeError) as exc:
            raise DecodingError(f"Failed to decode byte sequence: {exc}") from exc

    def save(self, path: Path) -> None:
        """Save the trained vocabulary and merge rules to a JSON file.

        Args:
            path: Destination file path.

        Raises:
            EncodingError: If the tokenizer is not trained.
        """
        if not self._is_trained or self._vocab is None:
            raise EncodingError("Cannot save an untrained tokenizer.")

        save_vocabulary(self._vocab, Path(path))

    @classmethod
    def load(cls, path: Path) -> "BPETokenizer":
        """Load a BPE tokenizer from a saved vocabulary file.

        Args:
            path: Path to the vocabulary JSON file.

        Returns:
            A trained BPETokenizer ready for encoding/decoding.

        Raises:
            TokenizerLoadError: If the file is missing or malformed.
        """
        vocab = load_vocabulary(Path(path))
        tokenizer = cls()
        tokenizer._vocab = vocab
        tokenizer._is_trained = True

        # Rebuild the merge rank lookup from the saved merge rules
        for rule in vocab.merge_rules:
            tokenizer._merge_rank[(rule.left, rule.right)] = rule.rank

        # Reconstruct the pre-tokenizer from the saved config
        pretokenizer_name = vocab.config.pretokenizer_type.upper()
        try:
            pt_type = PreTokenizerType[pretokenizer_name]
        except KeyError:
            logger.warning(
                "Unknown pretokenizer type '%s', defaulting to GPT2", pretokenizer_name
            )
            pt_type = PreTokenizerType.GPT2

        tokenizer._pretokenizer = PreTokenizer(pt_type)
        logger.info("Loaded BPETokenizer with vocab_size=%d", len(vocab))
        return tokenizer
