"""Command-line interface for the bpe-tokenizer library.

Provides commands for training, encoding, decoding, and comparing tokenizers.

Usage:
    bpe-tokenizer train --corpus corpus.txt --vocab-size 10000 --output vocab.json
    bpe-tokenizer encode --vocab vocab.json --text "Hello, world!"
    bpe-tokenizer decode --vocab vocab.json --ids 15496 11 995
    bpe-tokenizer compare --vocab vocab.json --text "Hello world"
    bpe-tokenizer info --vocab vocab.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .bpe import BPETokenizer
from .wordpiece import WordPieceTokenizer

logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool = False) -> None:
    """Configure logging based on verbosity flag.

    Args:
        verbose: If True, set log level to DEBUG; otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=level,
    )


def _load_tokenizer(vocab_path: Path, tokenizer_type: str = "auto"):
    """Load a tokenizer from a vocab file, inferring type if needed.

    Args:
        vocab_path: Path to the vocabulary JSON file.
        tokenizer_type: 'bpe', 'wordpiece', or 'auto' (infer from file).

    Returns:
        A loaded tokenizer instance.
    """
    if tokenizer_type == "auto":
        # Peek at the file to determine the type
        with vocab_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        tokenizer_type = data.get("config", {}).get("tokenizer_type", "bpe")

    if tokenizer_type == "bpe":
        return BPETokenizer.load(vocab_path)
    elif tokenizer_type == "wordpiece":
        return WordPieceTokenizer.load(vocab_path)
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")


def cmd_train(args: argparse.Namespace) -> int:
    """Execute the train subcommand.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Exit code (0 = success).
    """
    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        logger.error("Corpus file not found: %s", corpus_path)
        return 1

    corpus = corpus_path.read_text(encoding="utf-8")
    output_path = Path(args.output)

    if args.type == "bpe":
        tokenizer = BPETokenizer()
    else:
        tokenizer = WordPieceTokenizer()

    logger.info(
        "Training %s tokenizer: vocab_size=%d on %s",
        args.type.upper(),
        args.vocab_size,
        corpus_path,
    )

    stats = tokenizer.train(corpus, args.vocab_size)
    tokenizer.save(output_path)

    print("Training complete!")
    print(f"  Tokenizer type : {args.type.upper()}")
    print(f"  Final vocab    : {stats.final_vocab_size} tokens")
    print(f"  Merges         : {stats.num_merges}")
    print(f"  Training time  : {stats.training_seconds:.2f}s")
    print(f"  Saved to       : {output_path}")
    return 0


def cmd_encode(args: argparse.Namespace) -> int:
    """Execute the encode subcommand.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Exit code (0 = success).
    """
    vocab_path = Path(args.vocab)
    if not vocab_path.exists():
        logger.error("Vocabulary file not found: %s", vocab_path)
        return 1

    tokenizer = _load_tokenizer(vocab_path)
    text = args.text

    ids = tokenizer.encode(text)
    tokens = [tokenizer.id_to_token(i) for i in ids]

    print(f"Input : {text!r}")
    print(f"Tokens: {tokens}")
    print(f"IDs   : {ids}")
    print(f"Count : {len(ids)} tokens")
    return 0


def cmd_decode(args: argparse.Namespace) -> int:
    """Execute the decode subcommand.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Exit code (0 = success).
    """
    vocab_path = Path(args.vocab)
    if not vocab_path.exists():
        logger.error("Vocabulary file not found: %s", vocab_path)
        return 1

    tokenizer = _load_tokenizer(vocab_path)
    ids = [int(x) for x in args.ids]

    text = tokenizer.decode(ids)
    tokens = [tokenizer.id_to_token(i) for i in ids]

    print(f"IDs   : {ids}")
    print(f"Tokens: {tokens}")
    print(f"Output: {text!r}")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    """Display vocabulary statistics.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Exit code (0 = success).
    """
    vocab_path = Path(args.vocab)
    if not vocab_path.exists():
        logger.error("Vocabulary file not found: %s", vocab_path)
        return 1

    tokenizer = _load_tokenizer(vocab_path)
    config = tokenizer.config

    print(f"Vocabulary: {vocab_path}")
    print(f"  Type        : {config.tokenizer_type.upper()}")
    print(f"  Vocab size  : {tokenizer.vocab_size}")
    print(f"  Version     : {config.version}")
    if config.unk_token:
        print(f"  UNK token   : {config.unk_token!r}")
    if config.continuation_prefix:
        print(f"  Continuation: {config.continuation_prefix!r}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Compare tokenizations between two vocabulary files.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Exit code (0 = success).
    """
    vocab1_path = Path(args.vocab)
    if not vocab1_path.exists():
        logger.error("Vocabulary file not found: %s", vocab1_path)
        return 1

    tokenizer1 = _load_tokenizer(vocab1_path)
    ids1 = tokenizer1.encode(args.text)
    tokens1 = [tokenizer1.id_to_token(i) for i in ids1]

    print(f"Text: {args.text!r}")
    print(f"\n[{vocab1_path.stem}] ({tokenizer1.vocab_size} tokens vocab)")
    print(f"  Tokens: {tokens1}")
    print(f"  IDs   : {ids1}")
    print(f"  Count : {len(ids1)} tokens")

    if args.vocab2:
        vocab2_path = Path(args.vocab2)
        tokenizer2 = _load_tokenizer(vocab2_path)
        ids2 = tokenizer2.encode(args.text)
        tokens2 = [tokenizer2.id_to_token(i) for i in ids2]

        print(f"\n[{vocab2_path.stem}] ({tokenizer2.vocab_size} tokens vocab)")
        print(f"  Tokens: {tokens2}")
        print(f"  IDs   : {ids2}")
        print(f"  Count : {len(ids2)} tokens")

        print(f"\nToken count ratio: {len(ids1)}/{len(ids2)} = {len(ids1)/len(ids2):.2f}x")

    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the CLI.

    Returns:
        Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        prog="bpe-tokenizer",
        description="BPE and WordPiece tokenization from scratch",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # train subcommand
    train_p = subparsers.add_parser("train", help="Train a tokenizer on a corpus")
    train_p.add_argument("--corpus", required=True, help="Path to training corpus text file")
    train_p.add_argument("--vocab-size", type=int, required=True, help="Target vocabulary size")
    train_p.add_argument("--output", required=True, help="Output path for the vocabulary JSON")
    train_p.add_argument(
        "--type",
        choices=["bpe", "wordpiece"],
        default="bpe",
        help="Tokenizer type (default: bpe)",
    )

    # encode subcommand
    encode_p = subparsers.add_parser("encode", help="Encode text to token IDs")
    encode_p.add_argument("--vocab", required=True, help="Path to vocabulary JSON")
    encode_p.add_argument("--text", required=True, help="Text to encode")

    # decode subcommand
    decode_p = subparsers.add_parser("decode", help="Decode token IDs to text")
    decode_p.add_argument("--vocab", required=True, help="Path to vocabulary JSON")
    decode_p.add_argument("--ids", nargs="+", required=True, help="Token IDs to decode")

    # info subcommand
    info_p = subparsers.add_parser("info", help="Show vocabulary statistics")
    info_p.add_argument("--vocab", required=True, help="Path to vocabulary JSON")

    # compare subcommand
    compare_p = subparsers.add_parser("compare", help="Compare tokenizations")
    compare_p.add_argument("--vocab", required=True, help="Path to first vocabulary JSON")
    compare_p.add_argument("--vocab2", help="Path to second vocabulary JSON (optional)")
    compare_p.add_argument("--text", required=True, help="Text to compare")

    return parser


def main() -> None:
    """Entry point for the CLI."""
    parser = build_parser()
    args = parser.parse_args()
    _setup_logging(args.verbose)

    dispatch = {
        "train": cmd_train,
        "encode": cmd_encode,
        "decode": cmd_decode,
        "info": cmd_info,
        "compare": cmd_compare,
    }

    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    exit_code = handler(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
