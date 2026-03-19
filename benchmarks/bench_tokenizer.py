"""Benchmark suite for the bpe-tokenizer library.

Measures:
- Training throughput: time to build vocabulary at different sizes
- Encoding throughput: tokens/second for large text blobs
- Vocab compression ratio: characters per token
- Memory footprint: vocab dict size at different vocab sizes

Run:
    python benchmarks/bench_tokenizer.py
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bpe_tokenizer import BPETokenizer, WordPieceTokenizer

logging.basicConfig(level=logging.WARNING)

# ---------------------------------------------------------------------------
# Benchmark corpus: use the sample corpus, repeated for scale
# ---------------------------------------------------------------------------

CORPUS_PATH = Path(__file__).parent.parent / "corpora" / "sample_corpus.txt"
BASE_CORPUS = CORPUS_PATH.read_text(encoding="utf-8")

# Create a larger corpus for throughput benchmarks
LARGE_CORPUS = BASE_CORPUS * 50  # ~5x10^5 characters
HUGE_CORPUS = BASE_CORPUS * 200  # ~2x10^6 characters


def bench_bpe_training() -> dict[str, float]:
    """Benchmark BPE training at different vocabulary sizes.

    Returns:
        Dictionary of {label: value} benchmark results.
    """
    print("\n=== BPE Training Benchmarks ===")
    results: dict[str, float] = {}

    for vocab_size in [600, 1000, 2000, 5000]:
        tokenizer = BPETokenizer()
        start = time.perf_counter()
        stats = tokenizer.train(LARGE_CORPUS, vocab_size=vocab_size)
        elapsed = time.perf_counter() - start

        merges_per_sec = stats.num_merges / elapsed if elapsed > 0 else 0
        key = f"bpe_train_vocab{vocab_size}"
        results[key] = elapsed

        print(
            f"  vocab_size={vocab_size:5d}: {elapsed:.2f}s "
            f"({merges_per_sec:,.0f} merges/s, "
            f"{stats.num_merges} merges, "
            f"final_vocab={stats.final_vocab_size})"
        )

    return results


def bench_bpe_encoding() -> dict[str, float]:
    """Benchmark BPE encoding throughput.

    Returns:
        Dictionary of {label: value} benchmark results.
    """
    print("\n=== BPE Encoding Throughput ===")
    results: dict[str, float] = {}

    # Train tokenizer once
    tokenizer = BPETokenizer()
    tokenizer.train(LARGE_CORPUS, vocab_size=1000)

    # Warm up
    tokenizer.encode(BASE_CORPUS[:1000])

    # Benchmark encoding on different text sizes
    test_corpus = HUGE_CORPUS[:100_000]  # 100K characters
    iterations = 3

    total_time = 0.0
    total_tokens = 0
    for _ in range(iterations):
        start = time.perf_counter()
        ids = tokenizer.encode(test_corpus)
        elapsed = time.perf_counter() - start
        total_time += elapsed
        total_tokens = len(ids)

    avg_time = total_time / iterations
    tokens_per_sec = total_tokens / avg_time if avg_time > 0 else 0
    chars_per_token = len(test_corpus) / total_tokens if total_tokens > 0 else 0

    results["bpe_encoding_tokens_per_sec"] = tokens_per_sec
    results["bpe_chars_per_token"] = chars_per_token
    results["bpe_total_tokens"] = float(total_tokens)

    print(f"  Input: {len(test_corpus):,} chars")
    print(f"  Output: {total_tokens:,} tokens")
    print(f"  Throughput: {tokens_per_sec:,.0f} tokens/sec")
    print(f"  Compression: {chars_per_token:.2f} chars/token")
    print(f"  Avg encode time: {avg_time * 1000:.1f}ms")

    return results


def bench_wordpiece_training() -> dict[str, float]:
    """Benchmark WordPiece training at different vocabulary sizes.

    Returns:
        Dictionary of {label: value} benchmark results.
    """
    print("\n=== WordPiece Training Benchmarks ===")
    results: dict[str, float] = {}

    for vocab_size in [300, 500, 1000]:
        tokenizer = WordPieceTokenizer()
        start = time.perf_counter()
        stats = tokenizer.train(LARGE_CORPUS, vocab_size=vocab_size)
        elapsed = time.perf_counter() - start

        key = f"wordpiece_train_vocab{vocab_size}"
        results[key] = elapsed

        print(
            f"  vocab_size={vocab_size:5d}: {elapsed:.2f}s "
            f"(final_vocab={stats.final_vocab_size})"
        )

    return results


def bench_wordpiece_encoding() -> dict[str, float]:
    """Benchmark WordPiece encoding throughput.

    Returns:
        Dictionary of {label: value} benchmark results.
    """
    print("\n=== WordPiece Encoding Throughput ===")
    results: dict[str, float] = {}

    tokenizer = WordPieceTokenizer()
    tokenizer.train(LARGE_CORPUS, vocab_size=600)

    test_corpus = HUGE_CORPUS[:100_000]
    iterations = 3

    total_time = 0.0
    total_tokens = 0
    for _ in range(iterations):
        start = time.perf_counter()
        ids = tokenizer.encode(test_corpus)
        elapsed = time.perf_counter() - start
        total_time += elapsed
        total_tokens = len(ids)

    avg_time = total_time / iterations
    tokens_per_sec = total_tokens / avg_time if avg_time > 0 else 0
    chars_per_token = len(test_corpus) / total_tokens if total_tokens > 0 else 0

    results["wordpiece_encoding_tokens_per_sec"] = tokens_per_sec
    results["wordpiece_chars_per_token"] = chars_per_token

    print(f"  Input: {len(test_corpus):,} chars")
    print(f"  Output: {total_tokens:,} tokens")
    print(f"  Throughput: {tokens_per_sec:,.0f} tokens/sec")
    print(f"  Compression: {chars_per_token:.2f} chars/token")
    print(f"  Avg encode time: {avg_time * 1000:.1f}ms")

    return results


def bench_vocab_compression() -> None:
    """Show how vocab size affects compression ratio.

    Higher vocab size = fewer tokens per character (better compression),
    but requires more memory for the embedding table.
    """
    print("\n=== Vocabulary Size vs Compression Ratio ===")
    print(f"  {'Vocab Size':>12} | {'Chars/Token':>12} | {'Tokens':>10} | {'Ratio vs 256':>14}")
    print("  " + "-" * 56)

    baseline_tokens: int = 0
    test_text = BASE_CORPUS[:5000]

    for vocab_size in [600, 800, 1000, 2000, 5000]:
        tokenizer = BPETokenizer()
        tokenizer.train(LARGE_CORPUS, vocab_size=vocab_size)
        ids = tokenizer.encode(test_text)
        num_tokens = len(ids)
        chars_per_token = len(test_text) / num_tokens if num_tokens > 0 else 0

        if vocab_size == 600:
            baseline_tokens = num_tokens

        ratio_vs_baseline = baseline_tokens / num_tokens if num_tokens > 0 else 1.0

        print(
            f"  {vocab_size:>12,} | {chars_per_token:>12.2f} | "
            f"{num_tokens:>10,} | {ratio_vs_baseline:>14.2f}x"
        )


def bench_memory_footprint() -> None:
    """Estimate memory footprint of vocabulary at different sizes."""
    import sys

    print("\n=== Memory Footprint ===")

    for vocab_size in [600, 1000, 2000, 5000]:
        tokenizer = BPETokenizer()
        tokenizer.train(LARGE_CORPUS, vocab_size=vocab_size)
        vocab = tokenizer._vocab

        # Rough estimate: sum of all token string sizes + overhead per entry
        token_bytes = sum(
            sys.getsizeof(token) + sys.getsizeof(token_id)
            for token, token_id in vocab.token_to_id.items()
        )
        dict_overhead = sys.getsizeof(vocab.token_to_id)
        total_kb = (token_bytes + dict_overhead) / 1024

        print(f"  vocab_size={vocab_size:5d}: ~{total_kb:.1f} KB ({tokenizer.vocab_size} actual tokens)")


def main() -> None:
    """Run all benchmarks and print a summary."""
    print("=" * 60)
    print("bpe-tokenizer benchmark suite")
    print(f"Corpus size: {len(LARGE_CORPUS):,} chars (training), {len(HUGE_CORPUS):,} chars (encoding)")
    print("=" * 60)

    bpe_train_results = bench_bpe_training()
    bpe_encode_results = bench_bpe_encoding()
    wp_train_results = bench_wordpiece_training()
    wp_encode_results = bench_wordpiece_encoding()
    bench_vocab_compression()
    bench_memory_footprint()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"BPE encoding throughput : {bpe_encode_results.get('bpe_encoding_tokens_per_sec', 0):,.0f} tokens/sec")
    print(f"BPE compression ratio   : {bpe_encode_results.get('bpe_chars_per_token', 0):.2f} chars/token")
    print(f"WP encoding throughput  : {wp_encode_results.get('wordpiece_encoding_tokens_per_sec', 0):,.0f} tokens/sec")
    print(f"WP compression ratio    : {wp_encode_results.get('wordpiece_chars_per_token', 0):.2f} chars/token")
    print("=" * 60)


if __name__ == "__main__":
    main()
