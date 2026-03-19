"""WordPiece tokenizer implementation (BERT-style).

WordPiece was developed at Google and used in BERT (Devlin et al., 2018).
It is superficially similar to BPE but differs in two important ways:

1. Scoring: WordPiece selects merges that maximize the language model
   likelihood of the training data, not just raw pair frequency.
   The practical heuristic is: score(A,B) = freq(AB) / (freq(A) * freq(B))

2. Inference: WordPiece applies longest-match-first tokenization (greedy
   left-to-right), unlike BPE's merge-rule application.

3. Continuation prefix: Subword tokens that are NOT at the start of a word
   get a '##' prefix (e.g., "running" → ["run", "##ning"]).

References:
    Devlin, J., Chang, M., Lee, K., & Toutanova, K. (2018).
    BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
    NAACL 2019. https://arxiv.org/abs/1810.04805

    Schuster, M., & Nakamura, K. (2012).
    Japanese and Korean Voice Search. ICASSP 2012.
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
from .unicode_utils import BERT_SPECIAL_TOKENS, WORDPIECE_CONTINUATION_PREFIX, normalize_unicode
from .vocab import (
    TokenizerConfig,
    TrainingStats,
    Vocabulary,
    load_vocabulary,
    save_vocabulary,
)

logger = logging.getLogger(__name__)

# The unknown token for out-of-vocabulary input
UNK_TOKEN = "[UNK]"

# Special tokens added before any training tokens
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


def _build_char_vocab(words: list[str]) -> set[str]:
    """Build the initial character vocabulary from all unique characters.

    Unlike byte-level BPE, WordPiece operates on the character level
    directly, so unknown characters become [UNK].

    Args:
        words: List of word strings from pre-tokenization.

    Returns:
        Set of all unique characters appearing in the corpus.
    """
    chars: set[str] = set()
    for word in words:
        for char in word:
            chars.add(char)
            # Add the continuation form of this character
            chars.add(WORDPIECE_CONTINUATION_PREFIX + char)
    return chars


def _word_to_initial_tokens(word: str) -> list[str]:
    """Split a word into its initial WordPiece character tokens.

    The first character has no prefix; subsequent characters get the
    '##' continuation prefix.

    Example:
        "hello" → ["h", "##e", "##l", "##l", "##o"]

    Args:
        word: A single word string.

    Returns:
        List of character-level WordPiece tokens.
    """
    if not word:
        return []

    tokens = [word[0]]
    for char in word[1:]:
        tokens.append(WORDPIECE_CONTINUATION_PREFIX + char)
    return tokens


def _get_wordpiece_pair_scores(
    word_token_seqs: dict[tuple[str, ...], int],
) -> dict[tuple[str, str], float]:
    """Compute WordPiece merge scores for all adjacent pairs.

    Score formula: freq(AB) / (freq(A) * freq(B))
    This favors pairs where the combination is much more common than
    would be expected from independent occurrence of A and B separately.

    Args:
        word_token_seqs: Current tokenized word sequences with frequencies.

    Returns:
        Dictionary mapping (left, right) pairs to their WordPiece scores.
    """
    # Count individual token frequencies
    token_freqs: dict[str, int] = defaultdict(int)
    pair_freqs: dict[tuple[str, str], int] = defaultdict(int)

    for token_seq, freq in word_token_seqs.items():
        for token in token_seq:
            token_freqs[token] += freq
        for i in range(len(token_seq) - 1):
            pair_freqs[(token_seq[i], token_seq[i + 1])] += freq

    # Compute WordPiece score for each pair
    scores: dict[tuple[str, str], float] = {}
    for (left, right), pair_freq in pair_freqs.items():
        # Avoid division by zero (shouldn't happen in practice)
        denominator = token_freqs[left] * token_freqs[right]
        if denominator > 0:
            scores[(left, right)] = pair_freq / denominator

    return scores


def _merge_wordpiece_pair(
    word_token_seqs: dict[tuple[str, ...], int],
    pair: tuple[str, str],
    merged: str,
) -> dict[tuple[str, ...], int]:
    """Merge a pair of adjacent tokens in all word sequences.

    Args:
        word_token_seqs: Current tokenized word sequences with frequencies.
        pair: The (left, right) token pair to merge.
        merged: The new token resulting from the merge.

    Returns:
        Updated word token sequences with the merge applied.
    """
    left, right = pair
    new_seqs: dict[tuple[str, ...], int] = {}

    for token_seq, freq in word_token_seqs.items():
        new_tokens: list[str] = []
        i = 0
        while i < len(token_seq):
            if i < len(token_seq) - 1 and token_seq[i] == left and token_seq[i + 1] == right:
                new_tokens.append(merged)
                i += 2
            else:
                new_tokens.append(token_seq[i])
                i += 1
        new_seqs[tuple(new_tokens)] = freq

    return new_seqs


class WordPieceTokenizer(BaseTokenizer):
    """BERT-style WordPiece tokenizer.

    Trains on a corpus to build a vocabulary of subword tokens, then encodes
    new text using greedy longest-match-first tokenization. Unknown characters
    produce the [UNK] token.

    Example:
        >>> tokenizer = WordPieceTokenizer()
        >>> stats = tokenizer.train("the quick brown fox " * 100, vocab_size=200)
        >>> ids = tokenizer.encode("the quick")
        >>> text = tokenizer.decode(ids)
        >>> assert text == "the quick"
    """

    def __init__(self) -> None:
        """Initialize an untrained WordPiece tokenizer."""
        super().__init__()
        self._pretokenizer: Optional[PreTokenizer] = None
        self._unk_token_id: int = -1

    def train(
        self,
        corpus: str,
        vocab_size: int,
        pretokenizer_type: PreTokenizerType = PreTokenizerType.BERT,
    ) -> TrainingStats:
        """Train WordPiece on a text corpus.

        Builds an initial character vocabulary, then iteratively merges
        pairs with the highest WordPiece score until reaching vocab_size.

        Args:
            corpus: Raw text training corpus.
            vocab_size: Target vocabulary size.
            pretokenizer_type: Pre-tokenization strategy.

        Returns:
            TrainingStats with training metrics.

        Raises:
            VocabBuildError: If the corpus is empty or vocab_size is too small.
        """
        if not corpus.strip():
            raise VocabBuildError("Cannot train on an empty corpus.")

        if vocab_size < len(SPECIAL_TOKENS) + 1:
            raise VocabBuildError(
                f"vocab_size must be at least {len(SPECIAL_TOKENS) + 1} "
                f"to accommodate special tokens."
            )

        logger.info("Starting WordPiece training: target vocab_size=%d", vocab_size)
        start_time = time.perf_counter()

        # BERT pre-tokenizer applies Unicode NFC normalization + lowercasing
        self._pretokenizer = PreTokenizer(pretokenizer_type)
        normalized_corpus = normalize_unicode(corpus)
        pretokenized = self._pretokenizer.tokenize(normalized_corpus)
        logger.info("Pre-tokenized corpus into %d word tokens", len(pretokenized))

        # Initialize vocabulary with special tokens
        vocab = Vocabulary()
        for special_token in SPECIAL_TOKENS:
            vocab.add_token(special_token)

        self._unk_token_id = vocab.token_to_id[UNK_TOKEN]

        # Build initial character vocabulary from all unique characters
        char_vocab = _build_char_vocab(pretokenized)
        for char in sorted(char_vocab):  # Sort for determinism
            vocab.add_token(char)

        initial_vocab_size = len(vocab)
        logger.debug("Initial character vocabulary: %d tokens", initial_vocab_size)

        # Build initial word token sequences
        word_token_seqs: dict[tuple[str, ...], int] = defaultdict(int)
        for word in pretokenized:
            if not word:
                continue
            # Filter out words with characters not in vocabulary
            initial_tokens = _word_to_initial_tokens(word)
            if all(t in vocab.token_to_id for t in initial_tokens):
                word_token_seqs[tuple(initial_tokens)] += 1

        num_merges_needed = vocab_size - initial_vocab_size

        if num_merges_needed <= 0:
            logger.warning(
                "vocab_size=%d is not larger than initial vocab (%d). No merges needed.",
                vocab_size,
                initial_vocab_size,
            )
        else:
            # WordPiece merge loop
            for merge_idx in range(num_merges_needed):
                pair_scores = _get_wordpiece_pair_scores(word_token_seqs)

                if not pair_scores:
                    logger.debug("No more pairs to score at step %d", merge_idx)
                    break

                # Best pair = highest WordPiece score, ties broken alphabetically
                best_pair = max(pair_scores, key=lambda p: (pair_scores[p], p))

                # Create merged token: remove ## prefix from the right token
                left, right = best_pair
                right_stripped = right[len(WORDPIECE_CONTINUATION_PREFIX):]
                merged_token = left + right_stripped

                vocab.add_token(merged_token)
                # Also add its ## form if it would appear in non-initial position
                continuation_form = WORDPIECE_CONTINUATION_PREFIX + merged_token
                if continuation_form not in vocab.token_to_id:
                    vocab.add_token(continuation_form)

                word_token_seqs = _merge_wordpiece_pair(word_token_seqs, best_pair, merged_token)

                if (merge_idx + 1) % 200 == 0:
                    logger.debug(
                        "WordPiece merge %d/%d: '%s' + '%s' → '%s' (score=%.4f)",
                        merge_idx + 1,
                        num_merges_needed,
                        left,
                        right,
                        merged_token,
                        pair_scores[best_pair],
                    )

        elapsed = time.perf_counter() - start_time
        logger.info(
            "WordPiece training complete in %.2fs, final vocab=%d",
            elapsed,
            len(vocab),
        )

        config = TokenizerConfig(
            vocab_size=len(vocab),
            tokenizer_type="wordpiece",
            pretokenizer_type=pretokenizer_type.name.lower(),
            unk_token=UNK_TOKEN,
            continuation_prefix=WORDPIECE_CONTINUATION_PREFIX,
        )
        stats = TrainingStats(
            corpus_chars=len(corpus),
            corpus_tokens=len(pretokenized),
            initial_vocab_size=initial_vocab_size,
            final_vocab_size=len(vocab),
            num_merges=vocab_size - initial_vocab_size,
            training_seconds=elapsed,
        )
        vocab.config = config
        vocab.stats = stats

        self._vocab = vocab
        self._is_trained = True
        return stats

    def _tokenize_word(self, word: str) -> list[str]:
        """Apply greedy longest-match-first tokenization to a single word.

        Tries to match the longest possible token starting at each position.
        If no match is found, the whole word becomes [UNK].

        Args:
            word: A single pre-tokenized word.

        Returns:
            List of WordPiece token strings for this word.
        """
        tokens: list[str] = []
        start = 0

        while start < len(word):
            end = len(word)
            found = False

            # Try progressively shorter substrings until we find a match
            while start < end:
                substr = word[start:end]
                # Add ## prefix for non-initial subwords
                candidate = substr if start == 0 else WORDPIECE_CONTINUATION_PREFIX + substr

                if candidate in self._vocab.token_to_id:
                    tokens.append(candidate)
                    start = end
                    found = True
                    break

                end -= 1

            if not found:
                # No match found for this position — entire word is unknown
                return [UNK_TOKEN]

        return tokens

    def encode(self, text: str) -> list[int]:
        """Encode text into token IDs using WordPiece.

        Args:
            text: Input text to encode.

        Returns:
            List of integer token IDs.

        Raises:
            EncodingError: If the tokenizer is not trained.
        """
        if not self._is_trained or self._vocab is None:
            raise EncodingError("Tokenizer is not trained. Call .train() first.")

        if not text:
            return []

        normalized = normalize_unicode(text)
        pretokenized = self._pretokenizer.tokenize(normalized)

        all_ids: list[int] = []
        for word in pretokenized:
            if not word:
                continue

            word_tokens = self._tokenize_word(word)
            for token in word_tokens:
                token_id = self._vocab.token_to_id.get(token, self._unk_token_id)
                all_ids.append(token_id)

        return all_ids

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back to text, removing WordPiece continuation markers.

        Args:
            ids: List of integer token IDs.

        Returns:
            Decoded text string.

        Raises:
            DecodingError: If any ID is out of range or tokenizer is not trained.
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
                    f"Token ID {token_id} out of range for vocabulary of size {len(self._vocab)}."
                )
            tokens.append(token)

        # Reconstruct text from WordPiece tokens:
        # - Non-## tokens start new words (add space before, unless first token)
        # - ## tokens continue the previous word (strip ## and append directly)
        words: list[str] = []
        current_word = ""

        for token in tokens:
            # Skip special tokens in decoding
            if token in SPECIAL_TOKENS:
                if current_word:
                    words.append(current_word)
                    current_word = ""
                continue

            if token.startswith(WORDPIECE_CONTINUATION_PREFIX):
                current_word += token[len(WORDPIECE_CONTINUATION_PREFIX):]
            else:
                if current_word:
                    words.append(current_word)
                current_word = token

        if current_word:
            words.append(current_word)

        return " ".join(words)

    def save(self, path: Path) -> None:
        """Save the trained WordPiece vocabulary to a JSON file.

        Args:
            path: Destination file path.

        Raises:
            EncodingError: If the tokenizer is not trained.
        """
        if not self._is_trained or self._vocab is None:
            raise EncodingError("Cannot save an untrained tokenizer.")

        save_vocabulary(self._vocab, Path(path))

    @classmethod
    def load(cls, path: Path) -> "WordPieceTokenizer":
        """Load a WordPiece tokenizer from a saved vocabulary file.

        Args:
            path: Path to the vocabulary JSON file.

        Returns:
            A trained WordPieceTokenizer ready for encoding/decoding.
        """
        vocab = load_vocabulary(Path(path))
        tokenizer = cls()
        tokenizer._vocab = vocab
        tokenizer._is_trained = True

        tokenizer._unk_token_id = vocab.token_to_id.get(UNK_TOKEN, -1)

        pretokenizer_name = vocab.config.pretokenizer_type.upper()
        try:
            pt_type = PreTokenizerType[pretokenizer_name]
        except KeyError:
            pt_type = PreTokenizerType.BERT

        tokenizer._pretokenizer = PreTokenizer(pt_type)
        logger.info("Loaded WordPieceTokenizer with vocab_size=%d", len(vocab))
        return tokenizer
