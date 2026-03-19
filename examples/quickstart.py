"""Quickstart example: train, encode, decode, and compare tokenizers.

Demonstrates:
1. Training a BPE tokenizer on a sample corpus
2. Encoding text and inspecting the token sequence
3. Decoding back to text (roundtrip verification)
4. Training a WordPiece tokenizer for comparison
5. Showing how vocabulary size affects compression
6. Saving and loading a tokenizer

Run:
    python examples/quickstart.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Allow running from the repo root without installation
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bpe_tokenizer import BPETokenizer, WordPieceTokenizer

logging.basicConfig(level=logging.WARNING)


def separator(title: str) -> None:
    """Print a section separator."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def main() -> None:
    """Run the quickstart demonstration."""

    # -----------------------------------------------------------------------
    # Load the sample corpus
    # -----------------------------------------------------------------------
    corpus_path = Path(__file__).parent.parent / "corpora" / "sample_corpus.txt"
    corpus = corpus_path.read_text(encoding="utf-8")
    print(f"Loaded corpus: {len(corpus):,} characters, {len(corpus.split())} words")

    # -----------------------------------------------------------------------
    # 1. Train a BPE tokenizer
    # -----------------------------------------------------------------------
    separator("1. Training BPE Tokenizer (vocab_size=800)")

    bpe = BPETokenizer()
    stats = bpe.train(corpus, vocab_size=800)

    print(f"  Corpus chars     : {stats.corpus_chars:,}")
    print(f"  Pre-tokenized    : {stats.corpus_tokens:,} word tokens")
    print(f"  Initial vocab    : {stats.initial_vocab_size} (byte-level)")
    print(f"  Final vocab      : {stats.final_vocab_size} tokens")
    print(f"  Merge operations : {stats.num_merges}")
    print(f"  Training time    : {stats.training_seconds:.3f}s")

    # -----------------------------------------------------------------------
    # 2. Encode some text and inspect the tokens
    # -----------------------------------------------------------------------
    separator("2. Encoding Text")

    test_texts = [
        "Tokenization is the foundation of language models.",
        "Hello, world!",
        "byte pair encoding",
        "the quick brown fox jumps over the lazy dog",
    ]

    for text in test_texts:
        ids = bpe.encode(text)
        tokens = [bpe.id_to_token(i) for i in ids]
        chars_per_token = len(text) / len(ids) if ids else 0
        print(f"\n  Input   : {text!r}")
        print(f"  Tokens  : {tokens}")
        print(f"  IDs     : {ids}")
        print(f"  Count   : {len(ids)} tokens ({chars_per_token:.1f} chars/token)")

    # -----------------------------------------------------------------------
    # 3. Roundtrip verification
    # -----------------------------------------------------------------------
    separator("3. Roundtrip Verification (decode(encode(text)) == text)")

    roundtrip_texts = [
        "tokenization",
        "the quick brown fox",
        "Hello, world!",
        "byte pair encoding vocabulary",
        "machine learning natural language processing",
    ]

    all_passed = True
    for text in roundtrip_texts:
        ids = bpe.encode(text)
        decoded = bpe.decode(ids)
        passed = decoded == text
        all_passed = all_passed and passed
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {text!r} -> {decoded!r}")

    print(f"\n  All roundtrip tests passed: {all_passed}")

    # -----------------------------------------------------------------------
    # 4. BPE merge step walkthrough
    # -----------------------------------------------------------------------
    separator("4. BPE Algorithm Walkthrough (first 10 merge rules)")

    print("  These are the first 10 merge rules learned from the corpus:")
    print("  (Lower rank = applied first during encoding)\n")
    print(f"  {'Rank':>6}  {'Left Token':<20}  {'Right Token':<20}  {'Result'}")
    print("  " + "-" * 65)
    for rule in bpe._vocab.merge_rules[:10]:
        print(
            f"  {rule.rank:>6}  {rule.left!r:<20}  {rule.right!r:<20}  {rule.result!r}"
        )

    # -----------------------------------------------------------------------
    # 5. Train a WordPiece tokenizer for comparison
    # -----------------------------------------------------------------------
    separator("5. WordPiece Tokenizer (BERT-style, vocab_size=600)")

    wordpiece = WordPieceTokenizer()
    wp_stats = wordpiece.train(corpus, vocab_size=600)

    print(f"  Final vocab      : {wp_stats.final_vocab_size} tokens")
    print(f"  Training time    : {wp_stats.training_seconds:.3f}s")

    print("\n  Encoding 'tokenization is the foundation':")
    wp_text = "tokenization is the foundation"
    wp_ids = wordpiece.encode(wp_text)
    wp_tokens = [wordpiece.id_to_token(i) for i in wp_ids]
    wp_decoded = wordpiece.decode(wp_ids)
    print(f"  Tokens  : {wp_tokens}")
    print(f"  Count   : {len(wp_ids)} tokens")
    print(f"  Decoded : {wp_decoded!r}")

    # -----------------------------------------------------------------------
    # 6. Vocabulary size vs compression ratio
    # -----------------------------------------------------------------------
    separator("6. Vocabulary Size vs Compression Ratio")

    sample_text = corpus[:3000]
    print(f"  {'Vocab Size':>12}  {'Tokens':>10}  {'Chars/Token':>12}")
    print("  " + "-" * 40)
    for target_size in [600, 800, 1000]:
        t = BPETokenizer()
        t.train(corpus, vocab_size=target_size)
        ids = t.encode(sample_text)
        ratio = len(sample_text) / len(ids) if ids else 0
        print(f"  {target_size:>12,}  {len(ids):>10,}  {ratio:>12.2f}")

    # -----------------------------------------------------------------------
    # 7. Save and reload
    # -----------------------------------------------------------------------
    separator("7. Save and Reload")

    save_path = Path("/tmp/bpe_quickstart_vocab.json")
    bpe.save(save_path)
    print(f"  Saved to: {save_path}")

    loaded = BPETokenizer.load(save_path)
    print(f"  Loaded:   vocab_size={loaded.vocab_size}")

    # Verify the loaded tokenizer produces the same output
    original_ids = bpe.encode("tokenization is fundamental")
    loaded_ids = loaded.encode("tokenization is fundamental")
    match = original_ids == loaded_ids
    print(f"  Encoding match after reload: {match}")

    separator("Done")
    print("  bpe-tokenizer quickstart complete.")
    print("  Run 'make test' for the full test suite.")
    print("  Run 'python benchmarks/bench_tokenizer.py' for performance numbers.")


if __name__ == "__main__":
    main()
