"""Tests for the BPE tokenizer implementation."""

from __future__ import annotations

from pathlib import Path

import pytest

from bpe_tokenizer import BPETokenizer
from bpe_tokenizer.bpe import END_OF_WORD, _build_word_freqs, _get_pair_frequencies, _merge_pair
from bpe_tokenizer.exceptions import DecodingError, EncodingError, VocabBuildError

from .conftest import SMALL_CORPUS

# ---------------------------------------------------------------------------
# Internal algorithm tests
# ---------------------------------------------------------------------------


class TestBuildWordFreqs:
    """Tests for the _build_word_freqs helper."""

    def test_basic_word_creates_end_of_word_marker(self) -> None:
        """Each word's last character should have the END_OF_WORD marker."""
        freqs = _build_word_freqs(["hello"], use_byte_level=False)
        # "hello" → ('h', 'e', 'l', 'l', 'o</w>')
        assert len(freqs) == 1
        word_tuple = list(freqs.keys())[0]
        assert word_tuple[-1].endswith(END_OF_WORD)
        assert freqs[word_tuple] == 1

    def test_repeated_words_accumulate_frequency(self) -> None:
        """The same word repeated N times should have frequency N."""
        words = ["hello"] * 5
        freqs = _build_word_freqs(words, use_byte_level=False)
        word_tuple = list(freqs.keys())[0]
        assert freqs[word_tuple] == 5

    def test_empty_input_returns_empty(self) -> None:
        """Empty word list should produce an empty frequency dict."""
        freqs = _build_word_freqs([], use_byte_level=False)
        assert freqs == {}

    def test_byte_level_encoding(self) -> None:
        """Byte-level mode should encode using byte-to-unicode mapping."""
        freqs = _build_word_freqs(["a"], use_byte_level=True)
        assert len(freqs) == 1
        word_tuple = list(freqs.keys())[0]
        # Single character 'a' should produce a 1-element tuple
        assert len(word_tuple) == 1
        assert word_tuple[0].endswith(END_OF_WORD)


class TestGetPairFrequencies:
    """Tests for the _get_pair_frequencies helper."""

    def test_simple_pair_counting(self) -> None:
        """Pairs in a single word should be counted correctly."""
        # Word "abc" with freq 3 → pairs (a,b)=3 and (b,c)=3
        word_freqs = {("a", "b", "c"): 3}
        pair_freqs = _get_pair_frequencies(word_freqs)
        assert pair_freqs[("a", "b")] == 3
        assert pair_freqs[("b", "c")] == 3

    def test_multiple_words(self) -> None:
        """Pairs across different words should be summed."""
        word_freqs = {("a", "b"): 2, ("a", "b", "c"): 3}
        pair_freqs = _get_pair_frequencies(word_freqs)
        assert pair_freqs[("a", "b")] == 5  # 2 + 3

    def test_single_character_word_no_pairs(self) -> None:
        """A single-character word has no pairs."""
        word_freqs = {("a",): 10}
        pair_freqs = _get_pair_frequencies(word_freqs)
        assert len(pair_freqs) == 0


class TestMergePair:
    """Tests for the _merge_pair helper."""

    def test_basic_merge(self) -> None:
        """Merging ('a', 'b') should produce 'ab' in the word."""
        word_freqs = {("a", "b", "c"): 1}
        result = _merge_pair(word_freqs, ("a", "b"), "ab")
        assert ("ab", "c") in result

    def test_merge_preserves_frequency(self) -> None:
        """Frequency should be preserved after merge."""
        word_freqs = {("a", "b", "c"): 7}
        result = _merge_pair(word_freqs, ("a", "b"), "ab")
        assert result[("ab", "c")] == 7

    def test_merge_all_occurrences_in_word(self) -> None:
        """All occurrences of the pair within a word should be merged."""
        # "abab" → after merging (a,b): "ab" + "ab" → two "ab" tokens
        word_freqs = {("a", "b", "a", "b"): 1}
        result = _merge_pair(word_freqs, ("a", "b"), "ab")
        assert ("ab", "ab") in result

    def test_unaffected_words_unchanged(self) -> None:
        """Words that don't contain the pair should be unchanged."""
        word_freqs = {("c", "d"): 3, ("a", "b"): 2}
        result = _merge_pair(word_freqs, ("a", "b"), "ab")
        assert ("c", "d") in result
        assert result[("c", "d")] == 3


# ---------------------------------------------------------------------------
# BPETokenizer training tests
# ---------------------------------------------------------------------------


class TestBPETraining:
    """Tests for BPETokenizer.train()."""

    def test_train_produces_correct_vocab_size(self) -> None:
        """Final vocab should be close to the requested size."""
        tokenizer = BPETokenizer()
        target = 700
        stats = tokenizer.train(SMALL_CORPUS, vocab_size=target)
        # Allow some slack since we stop early if no more pairs
        assert stats.final_vocab_size <= target
        assert stats.final_vocab_size >= 513  # At least initial 512 + 1 merge

    def test_train_returns_stats(self) -> None:
        """Training should return valid TrainingStats."""
        tokenizer = BPETokenizer()
        stats = tokenizer.train(SMALL_CORPUS, vocab_size=700)
        assert stats.corpus_chars > 0
        assert stats.corpus_tokens > 0
        assert stats.training_seconds > 0
        assert stats.num_merges > 0
        assert stats.initial_vocab_size == 512

    def test_train_marks_tokenizer_as_trained(self) -> None:
        """After training, is_trained should be True."""
        tokenizer = BPETokenizer()
        assert not tokenizer.is_trained
        tokenizer.train(SMALL_CORPUS, vocab_size=700)
        assert tokenizer.is_trained

    def test_train_empty_corpus_raises(self) -> None:
        """Training on empty string should raise VocabBuildError."""
        tokenizer = BPETokenizer()
        with pytest.raises(VocabBuildError, match="empty corpus"):
            tokenizer.train("   ", vocab_size=700)

    def test_train_small_vocab_size_raises(self) -> None:
        """vocab_size <= 512 should raise VocabBuildError."""
        tokenizer = BPETokenizer()
        with pytest.raises(VocabBuildError, match="vocab_size must be > 512"):
            tokenizer.train(SMALL_CORPUS, vocab_size=100)

    @pytest.mark.parametrize("vocab_size", [600, 700, 800])
    def test_train_different_vocab_sizes(self, vocab_size: int) -> None:
        """Training should succeed for various vocab sizes."""
        tokenizer = BPETokenizer()
        stats = tokenizer.train(SMALL_CORPUS, vocab_size=vocab_size)
        assert tokenizer.is_trained
        assert stats.final_vocab_size <= vocab_size


# ---------------------------------------------------------------------------
# BPETokenizer encode tests
# ---------------------------------------------------------------------------


class TestBPEEncode:
    """Tests for BPETokenizer.encode()."""

    def test_encode_returns_list_of_ints(
        self, bpe_tokenizer_small: BPETokenizer
    ) -> None:
        """Encode should return a list of non-negative integers."""
        ids = bpe_tokenizer_small.encode("hello")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) and i >= 0 for i in ids)

    def test_encode_empty_string_returns_empty_list(
        self, bpe_tokenizer_small: BPETokenizer
    ) -> None:
        """Empty string should encode to an empty list."""
        ids = bpe_tokenizer_small.encode("")
        assert ids == []

    def test_encode_before_training_raises(self) -> None:
        """Encoding with an untrained tokenizer should raise EncodingError."""
        tokenizer = BPETokenizer()
        with pytest.raises(EncodingError, match="not trained"):
            tokenizer.encode("hello")

    def test_encode_all_ids_in_vocab_range(
        self, bpe_tokenizer_small: BPETokenizer
    ) -> None:
        """All encoded IDs should be within the vocabulary range."""
        ids = bpe_tokenizer_small.encode("the quick brown fox")
        vocab_size = bpe_tokenizer_small.vocab_size
        assert all(0 <= i < vocab_size for i in ids)

    @pytest.mark.parametrize(
        "text",
        [
            "hello",
            "the quick brown fox",
            "machine learning",
            "12345",
            "hello, world!",
        ],
    )
    def test_encode_produces_nonempty_output_for_nonempty_input(
        self, bpe_tokenizer_small: BPETokenizer, text: str
    ) -> None:
        """Any non-empty text should produce at least one token."""
        ids = bpe_tokenizer_small.encode(text)
        assert len(ids) > 0


# ---------------------------------------------------------------------------
# BPETokenizer decode tests
# ---------------------------------------------------------------------------


class TestBPEDecode:
    """Tests for BPETokenizer.decode()."""

    def test_decode_empty_list_returns_empty_string(
        self, bpe_tokenizer_small: BPETokenizer
    ) -> None:
        """Empty ID list should decode to empty string."""
        text = bpe_tokenizer_small.decode([])
        assert text == ""

    def test_decode_before_training_raises(self) -> None:
        """Decoding with an untrained tokenizer should raise DecodingError."""
        tokenizer = BPETokenizer()
        with pytest.raises(DecodingError, match="not trained"):
            tokenizer.decode([0])

    def test_decode_out_of_range_id_raises(
        self, bpe_tokenizer_small: BPETokenizer
    ) -> None:
        """An ID >= vocab_size should raise DecodingError."""
        huge_id = bpe_tokenizer_small.vocab_size + 99999
        with pytest.raises(DecodingError, match="out of range"):
            bpe_tokenizer_small.decode([huge_id])


# ---------------------------------------------------------------------------
# BPETokenizer save/load tests
# ---------------------------------------------------------------------------


class TestBPESaveLoad:
    """Tests for BPETokenizer.save() and BPETokenizer.load()."""

    def test_save_creates_file(
        self, bpe_tokenizer_small: BPETokenizer, tmp_vocab_path: Path
    ) -> None:
        """save() should create a file at the given path."""
        bpe_tokenizer_small.save(tmp_vocab_path)
        assert tmp_vocab_path.exists()

    def test_load_restores_vocab_size(
        self, bpe_tokenizer_small: BPETokenizer, tmp_vocab_path: Path
    ) -> None:
        """Loaded tokenizer should have the same vocab_size."""
        bpe_tokenizer_small.save(tmp_vocab_path)
        loaded = BPETokenizer.load(tmp_vocab_path)
        assert loaded.vocab_size == bpe_tokenizer_small.vocab_size

    def test_load_produces_same_encoding(
        self, bpe_tokenizer_small: BPETokenizer, tmp_vocab_path: Path
    ) -> None:
        """Loaded tokenizer should produce identical encodings."""
        bpe_tokenizer_small.save(tmp_vocab_path)
        loaded = BPETokenizer.load(tmp_vocab_path)

        test_text = "the quick brown fox"
        ids_original = bpe_tokenizer_small.encode(test_text)
        ids_loaded = loaded.encode(test_text)
        assert ids_original == ids_loaded

    def test_save_untrained_raises(self, tmp_vocab_path: Path) -> None:
        """Saving an untrained tokenizer should raise EncodingError."""
        tokenizer = BPETokenizer()
        with pytest.raises(EncodingError, match="untrained"):
            tokenizer.save(tmp_vocab_path)
