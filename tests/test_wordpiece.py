"""Tests for the WordPiece tokenizer implementation."""

from __future__ import annotations

from pathlib import Path

import pytest

from bpe_tokenizer import WordPieceTokenizer
from bpe_tokenizer.exceptions import DecodingError, EncodingError, VocabBuildError
from bpe_tokenizer.wordpiece import (
    SPECIAL_TOKENS,
    UNK_TOKEN,
    _get_wordpiece_pair_scores,
    _merge_wordpiece_pair,
    _word_to_initial_tokens,
)

from .conftest import MEDIUM_CORPUS, SMALL_CORPUS


# ---------------------------------------------------------------------------
# Internal algorithm tests
# ---------------------------------------------------------------------------


class TestWordToInitialTokens:
    """Tests for the _word_to_initial_tokens helper."""

    def test_single_char_word(self) -> None:
        """Single-char word should produce one token with no ## prefix."""
        tokens = _word_to_initial_tokens("a")
        assert tokens == ["a"]

    def test_multi_char_word_has_continuation_prefix(self) -> None:
        """Non-first characters should have ## prefix."""
        tokens = _word_to_initial_tokens("hello")
        assert tokens == ["h", "##e", "##l", "##l", "##o"]

    def test_empty_word_returns_empty(self) -> None:
        """Empty word should return empty list."""
        tokens = _word_to_initial_tokens("")
        assert tokens == []

    @pytest.mark.parametrize(
        "word, expected_length",
        [
            ("a", 1),
            ("ab", 2),
            ("abc", 3),
            ("hello", 5),
        ],
    )
    def test_token_count_equals_char_count(
        self, word: str, expected_length: int
    ) -> None:
        """Number of tokens should equal number of characters."""
        tokens = _word_to_initial_tokens(word)
        assert len(tokens) == expected_length


class TestWordpiecePairScores:
    """Tests for the _get_wordpiece_pair_scores helper."""

    def test_high_frequency_pair_gets_high_score(self) -> None:
        """A pair that always appears together should have a high score."""
        # "ab" always together: freq(ab)=10, freq(a)=10, freq(b)=10
        # Score = 10 / (10 * 10) = 0.1
        word_seqs = {("a", "##b"): 10}
        scores = _get_wordpiece_pair_scores(word_seqs)
        assert ("a", "##b") in scores
        assert scores[("a", "##b")] > 0

    def test_scores_are_positive(self) -> None:
        """All scores should be positive."""
        word_seqs = {("h", "##e", "##l", "##l", "##o"): 5}
        scores = _get_wordpiece_pair_scores(word_seqs)
        assert all(s > 0 for s in scores.values())

    def test_empty_seqs_returns_empty(self) -> None:
        """Empty input should produce empty scores."""
        scores = _get_wordpiece_pair_scores({})
        assert scores == {}


class TestMergeWordpiecePair:
    """Tests for the _merge_wordpiece_pair helper."""

    def test_basic_merge(self) -> None:
        """Merging adjacent tokens should produce the merged token."""
        seqs = {("a", "##b"): 1}
        result = _merge_wordpiece_pair(seqs, ("a", "##b"), "ab")
        assert ("ab",) in result

    def test_frequency_preserved(self) -> None:
        """Frequency should remain unchanged after merge."""
        seqs = {("a", "##b", "##c"): 7}
        result = _merge_wordpiece_pair(seqs, ("a", "##b"), "ab")
        assert result[("ab", "##c")] == 7


# ---------------------------------------------------------------------------
# WordPieceTokenizer training tests
# ---------------------------------------------------------------------------


class TestWordPieceTraining:
    """Tests for WordPieceTokenizer.train()."""

    def test_train_marks_as_trained(self) -> None:
        """After training, is_trained should be True."""
        tokenizer = WordPieceTokenizer()
        assert not tokenizer.is_trained
        tokenizer.train(SMALL_CORPUS, vocab_size=300)
        assert tokenizer.is_trained

    def test_train_empty_corpus_raises(self) -> None:
        """Empty corpus should raise VocabBuildError."""
        tokenizer = WordPieceTokenizer()
        with pytest.raises(VocabBuildError, match="empty corpus"):
            tokenizer.train("  ", vocab_size=300)

    def test_special_tokens_in_vocab(self) -> None:
        """All special tokens should be in the vocabulary after training."""
        tokenizer = WordPieceTokenizer()
        tokenizer.train(SMALL_CORPUS, vocab_size=300)
        for special in SPECIAL_TOKENS:
            token_id = tokenizer.token_to_id(special)
            assert token_id is not None, f"Special token {special!r} not in vocabulary"

    def test_unk_token_at_index_1(self) -> None:
        """[UNK] token should have ID 1 (BERT convention)."""
        tokenizer = WordPieceTokenizer()
        tokenizer.train(SMALL_CORPUS, vocab_size=300)
        assert tokenizer.token_to_id(UNK_TOKEN) == 1

    @pytest.mark.parametrize("vocab_size", [200, 300, 400])
    def test_train_various_vocab_sizes(self, vocab_size: int) -> None:
        """Training should succeed for different target sizes."""
        tokenizer = WordPieceTokenizer()
        stats = tokenizer.train(SMALL_CORPUS, vocab_size=vocab_size)
        assert tokenizer.is_trained
        assert stats.final_vocab_size > 0

    def test_train_returns_stats(self) -> None:
        """Training should return valid TrainingStats."""
        tokenizer = WordPieceTokenizer()
        stats = tokenizer.train(SMALL_CORPUS, vocab_size=300)
        assert stats.corpus_chars > 0
        assert stats.corpus_tokens > 0
        assert stats.training_seconds >= 0


# ---------------------------------------------------------------------------
# WordPieceTokenizer encode tests
# ---------------------------------------------------------------------------


class TestWordPieceEncode:
    """Tests for WordPieceTokenizer.encode()."""

    def test_encode_returns_list_of_ints(
        self, wordpiece_tokenizer_small: WordPieceTokenizer
    ) -> None:
        """Encode should return a list of non-negative integers."""
        ids = wordpiece_tokenizer_small.encode("the")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) and i >= 0 for i in ids)

    def test_encode_empty_string_returns_empty(
        self, wordpiece_tokenizer_small: WordPieceTokenizer
    ) -> None:
        """Empty string should produce empty list."""
        assert wordpiece_tokenizer_small.encode("") == []

    def test_encode_untrained_raises(self) -> None:
        """Encoding with untrained tokenizer raises EncodingError."""
        tokenizer = WordPieceTokenizer()
        with pytest.raises(EncodingError, match="not trained"):
            tokenizer.encode("hello")

    def test_encode_unknown_word_uses_unk(
        self, wordpiece_tokenizer_small: WordPieceTokenizer
    ) -> None:
        """A word with no matching subwords should produce [UNK]."""
        # Use a word with very unusual characters unlikely to be in the vocab
        unk_id = wordpiece_tokenizer_small.token_to_id(UNK_TOKEN)
        ids = wordpiece_tokenizer_small.encode("xyzqwerty123zyx")
        # At least some tokens should be UNK
        assert unk_id in ids

    @pytest.mark.parametrize(
        "text",
        [
            "the",
            "learning",
            "quick brown fox",
            "tokenization",
        ],
    )
    def test_encode_nonempty_input_produces_nonempty_output(
        self, wordpiece_tokenizer_small: WordPieceTokenizer, text: str
    ) -> None:
        """Any non-empty text should produce at least one token."""
        ids = wordpiece_tokenizer_small.encode(text)
        assert len(ids) > 0

    def test_all_ids_in_vocab_range(
        self, wordpiece_tokenizer_small: WordPieceTokenizer
    ) -> None:
        """All encoded IDs should be within the vocabulary range."""
        ids = wordpiece_tokenizer_small.encode("the quick brown fox")
        vocab_size = wordpiece_tokenizer_small.vocab_size
        assert all(0 <= i < vocab_size for i in ids)


# ---------------------------------------------------------------------------
# WordPieceTokenizer decode tests
# ---------------------------------------------------------------------------


class TestWordPieceDecode:
    """Tests for WordPieceTokenizer.decode()."""

    def test_decode_empty_list_returns_empty(
        self, wordpiece_tokenizer_small: WordPieceTokenizer
    ) -> None:
        """Empty list should decode to empty string."""
        assert wordpiece_tokenizer_small.decode([]) == ""

    def test_decode_untrained_raises(self) -> None:
        """Decoding with untrained tokenizer raises DecodingError."""
        tokenizer = WordPieceTokenizer()
        with pytest.raises(DecodingError, match="not trained"):
            tokenizer.decode([0])

    def test_decode_out_of_range_raises(
        self, wordpiece_tokenizer_small: WordPieceTokenizer
    ) -> None:
        """Out-of-range ID should raise DecodingError."""
        huge_id = wordpiece_tokenizer_small.vocab_size + 99999
        with pytest.raises(DecodingError, match="out of range"):
            wordpiece_tokenizer_small.decode([huge_id])

    def test_continuation_tokens_reconstruct_word(
        self, wordpiece_tokenizer_medium: WordPieceTokenizer
    ) -> None:
        """## continuation tokens should join without spaces."""
        # Encode a word that should be split into subwords, then decode
        word = "tokenization"
        ids = wordpiece_tokenizer_medium.encode(word)
        decoded = wordpiece_tokenizer_medium.decode(ids)
        assert "tokenization" in decoded or "tokenizat" in decoded


# ---------------------------------------------------------------------------
# WordPieceTokenizer save/load tests
# ---------------------------------------------------------------------------


class TestWordPieceSaveLoad:
    """Tests for WordPieceTokenizer persistence."""

    def test_save_creates_file(
        self, wordpiece_tokenizer_small: WordPieceTokenizer, tmp_vocab_path: Path
    ) -> None:
        """save() should create a file at the given path."""
        wordpiece_tokenizer_small.save(tmp_vocab_path)
        assert tmp_vocab_path.exists()

    def test_load_restores_vocab_size(
        self, wordpiece_tokenizer_small: WordPieceTokenizer, tmp_vocab_path: Path
    ) -> None:
        """Loaded tokenizer should have the same vocab_size."""
        wordpiece_tokenizer_small.save(tmp_vocab_path)
        loaded = WordPieceTokenizer.load(tmp_vocab_path)
        assert loaded.vocab_size == wordpiece_tokenizer_small.vocab_size

    def test_load_produces_same_encoding(
        self, wordpiece_tokenizer_small: WordPieceTokenizer, tmp_vocab_path: Path
    ) -> None:
        """Loaded tokenizer should produce identical encodings."""
        wordpiece_tokenizer_small.save(tmp_vocab_path)
        loaded = WordPieceTokenizer.load(tmp_vocab_path)
        text = "the quick brown fox"
        assert wordpiece_tokenizer_small.encode(text) == loaded.encode(text)
