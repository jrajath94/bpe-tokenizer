"""Tests for the vocabulary management module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bpe_tokenizer.exceptions import TokenizerLoadError
from bpe_tokenizer.vocab import (
    MergeRule,
    TokenizerConfig,
    TrainingStats,
    Vocabulary,
    load_vocabulary,
    save_vocabulary,
)


class TestVocabulary:
    """Tests for the Vocabulary data structure."""

    def test_add_token_returns_sequential_ids(self) -> None:
        """Tokens should receive sequential IDs starting at 0."""
        vocab = Vocabulary()
        id_a = vocab.add_token("a")
        id_b = vocab.add_token("b")
        id_c = vocab.add_token("c")
        assert (id_a, id_b, id_c) == (0, 1, 2)

    def test_add_duplicate_token_returns_existing_id(self) -> None:
        """Adding a token that already exists should return its current ID."""
        vocab = Vocabulary()
        id1 = vocab.add_token("hello")
        id2 = vocab.add_token("hello")
        assert id1 == id2
        assert len(vocab) == 1

    def test_len_reflects_token_count(self) -> None:
        """len() should return the number of unique tokens."""
        vocab = Vocabulary()
        assert len(vocab) == 0
        vocab.add_token("a")
        assert len(vocab) == 1
        vocab.add_token("b")
        assert len(vocab) == 2

    def test_id_to_token_is_inverse_of_token_to_id(self) -> None:
        """id_to_token and token_to_id should be mutual inverses."""
        vocab = Vocabulary()
        tokens = ["hello", "world", "foo", "bar"]
        for token in tokens:
            vocab.add_token(token)

        for token in tokens:
            token_id = vocab.token_to_id[token]
            assert vocab.id_to_token[token_id] == token


class TestSaveLoadVocabulary:
    """Tests for save_vocabulary and load_vocabulary."""

    def _make_vocab(self) -> Vocabulary:
        """Create a minimal vocabulary for testing."""
        vocab = Vocabulary()
        vocab.add_token("a")
        vocab.add_token("b")
        vocab.add_token("ab")
        vocab.merge_rules = [
            MergeRule(left="a", right="b", result="ab", rank=0)
        ]
        vocab.config = TokenizerConfig(
            vocab_size=3,
            tokenizer_type="bpe",
            pretokenizer_type="gpt2",
        )
        vocab.stats = TrainingStats(
            corpus_chars=100,
            final_vocab_size=3,
            num_merges=1,
        )
        return vocab

    def test_save_creates_valid_json(self, tmp_path: Path) -> None:
        """Saved vocabulary should be valid JSON."""
        vocab = self._make_vocab()
        path = tmp_path / "vocab.json"
        save_vocabulary(vocab, path)

        with path.open("r") as fh:
            data = json.load(fh)

        assert "token_to_id" in data
        assert "merge_rules" in data
        assert "config" in data
        assert "version" in data

    def test_roundtrip_preserves_vocab_size(self, tmp_path: Path) -> None:
        """Load(save(vocab)).size should equal original vocab size."""
        vocab = self._make_vocab()
        path = tmp_path / "vocab.json"
        save_vocabulary(vocab, path)
        loaded = load_vocabulary(path)
        assert len(loaded) == len(vocab)

    def test_roundtrip_preserves_tokens(self, tmp_path: Path) -> None:
        """All tokens should be present after save/load."""
        vocab = self._make_vocab()
        path = tmp_path / "vocab.json"
        save_vocabulary(vocab, path)
        loaded = load_vocabulary(path)

        for token, token_id in vocab.token_to_id.items():
            assert loaded.token_to_id.get(token) == token_id

    def test_roundtrip_preserves_merge_rules(self, tmp_path: Path) -> None:
        """Merge rules should survive save/load with correct order."""
        vocab = self._make_vocab()
        path = tmp_path / "vocab.json"
        save_vocabulary(vocab, path)
        loaded = load_vocabulary(path)

        assert len(loaded.merge_rules) == len(vocab.merge_rules)
        for original, restored in zip(vocab.merge_rules, loaded.merge_rules):
            assert original.left == restored.left
            assert original.right == restored.right
            assert original.result == restored.result
            assert original.rank == restored.rank

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        """Loading a nonexistent file should raise TokenizerLoadError."""
        with pytest.raises(TokenizerLoadError, match="not found"):
            load_vocabulary(tmp_path / "does_not_exist.json")

    def test_load_invalid_json_raises(self, tmp_path: Path) -> None:
        """Loading a file with invalid JSON should raise TokenizerLoadError."""
        bad_path = tmp_path / "bad.json"
        bad_path.write_text("this is not json {{{", encoding="utf-8")
        with pytest.raises(TokenizerLoadError, match="Failed to read"):
            load_vocabulary(bad_path)

    def test_load_missing_keys_raises(self, tmp_path: Path) -> None:
        """JSON file missing required keys should raise TokenizerLoadError."""
        bad_path = tmp_path / "bad.json"
        bad_path.write_text(json.dumps({"version": "1.0"}), encoding="utf-8")
        with pytest.raises(TokenizerLoadError, match="Malformed"):
            load_vocabulary(bad_path)


class TestMergeRule:
    """Tests for the MergeRule dataclass."""

    def test_fields_accessible(self) -> None:
        """MergeRule fields should be accessible by name."""
        rule = MergeRule(left="a", right="b", result="ab", rank=0)
        assert rule.left == "a"
        assert rule.right == "b"
        assert rule.result == "ab"
        assert rule.rank == 0


class TestTokenizerConfig:
    """Tests for the TokenizerConfig dataclass."""

    def test_default_version_is_set(self) -> None:
        """Config should have a default version string."""
        config = TokenizerConfig(vocab_size=100, tokenizer_type="bpe")
        assert config.version is not None
        assert len(config.version) > 0

    def test_optional_fields_default_to_none(self) -> None:
        """Optional fields should default to None."""
        config = TokenizerConfig(vocab_size=100, tokenizer_type="bpe")
        assert config.unk_token is None
        assert config.continuation_prefix is None
