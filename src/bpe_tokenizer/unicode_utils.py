"""Unicode normalization, byte-level encoding, and special token utilities.

Provides byte-level text encoding (GPT-2 style) where every byte in UTF-8
is a valid vocabulary token, ensuring no unknown tokens for any Unicode text.
"""

from __future__ import annotations

import logging
import unicodedata

logger = logging.getLogger(__name__)

# GPT-2-style byte-to-unicode mapping: maps all 256 byte values to printable
# Unicode characters so that every byte in UTF-8 can be a vocabulary token.
# This avoids collisions with actual text characters.
_BYTE_TO_UNICODE: dict[int, str] | None = None
_UNICODE_TO_BYTE: dict[str, int] | None = None

# Special tokens used by BERT-style tokenizers
BERT_SPECIAL_TOKENS = {
    "[PAD]": 0,
    "[UNK]": 100,
    "[CLS]": 101,
    "[SEP]": 102,
    "[MASK]": 103,
}

# Continuation prefix used by WordPiece for subword tokens
WORDPIECE_CONTINUATION_PREFIX = "##"


def build_byte_to_unicode() -> dict[int, str]:
    """Build the GPT-2 byte-to-unicode mapping.

    Maps every possible byte (0-255) to a unique printable Unicode character.
    Bytes that are already printable ASCII use their natural character;
    the remaining bytes map to code points starting at U+0100.

    Returns:
        A dictionary mapping byte values (0-255) to single Unicode characters.
    """
    global _BYTE_TO_UNICODE
    if _BYTE_TO_UNICODE is not None:
        return _BYTE_TO_UNICODE

    # Start with all printable ASCII characters (33-126 and 161-172, 174-255)
    printable_bytes = list(range(ord("!"), ord("~") + 1))
    printable_bytes += list(range(ord("¡"), ord("¬") + 1))
    printable_bytes += list(range(ord("®"), ord("ÿ") + 1))

    byte_to_unicode: dict[int, str] = {b: chr(b) for b in printable_bytes}

    # Map remaining bytes to code points starting at 256 (U+0100)
    next_codepoint = 256
    for byte_val in range(256):
        if byte_val not in byte_to_unicode:
            byte_to_unicode[byte_val] = chr(next_codepoint)
            next_codepoint += 1

    _BYTE_TO_UNICODE = byte_to_unicode
    logger.debug("Built byte-to-unicode mapping with %d entries", len(byte_to_unicode))
    return byte_to_unicode


def build_unicode_to_byte() -> dict[str, int]:
    """Build the inverse of the byte-to-unicode mapping.

    Returns:
        A dictionary mapping Unicode characters back to byte values (0-255).
    """
    global _UNICODE_TO_BYTE
    if _UNICODE_TO_BYTE is not None:
        return _UNICODE_TO_BYTE

    _UNICODE_TO_BYTE = {v: k for k, v in build_byte_to_unicode().items()}
    return _UNICODE_TO_BYTE


def text_to_bytes(text: str) -> list[str]:
    """Encode text as a sequence of byte-level Unicode characters (GPT-2 style).

    Each character in the output represents one byte of the UTF-8 encoding of
    the input text. This guarantees that any Unicode text can be represented
    in a fixed vocabulary of 256 tokens.

    Args:
        text: Input text to encode.

    Returns:
        List of single Unicode characters representing each byte of the UTF-8
        encoding of the input text.
    """
    byte_to_unicode = build_byte_to_unicode()
    return [byte_to_unicode[b] for b in text.encode("utf-8")]


def bytes_to_text(byte_chars: list[str]) -> str:
    """Decode a sequence of byte-level Unicode characters back to text.

    Args:
        byte_chars: List of single Unicode characters from the byte-level
            encoding scheme (produced by text_to_bytes).

    Returns:
        The original UTF-8 decoded string.

    Raises:
        UnicodeDecodeError: If the byte sequence does not form valid UTF-8.
    """
    unicode_to_byte = build_unicode_to_byte()
    byte_values = bytes([unicode_to_byte[ch] for ch in byte_chars])
    return byte_values.decode("utf-8")


def normalize_unicode(text: str, form: str = "NFC") -> str:
    """Apply Unicode normalization to the input text.

    Args:
        text: Input text to normalize.
        form: Normalization form. One of 'NFC', 'NFKC', 'NFD', 'NFKD'.
            Defaults to 'NFC', which is the most common form for text
            that needs to be compared across sources.

    Returns:
        Normalized text.
    """
    return unicodedata.normalize(form, text)


def is_whitespace(char: str) -> bool:
    """Check if a single character is a Unicode whitespace character.

    Args:
        char: A single character.

    Returns:
        True if the character is whitespace, False otherwise.
    """
    if char in (" ", "\t", "\n", "\r"):
        return True
    cat = unicodedata.category(char)
    return cat == "Zs"


def is_punctuation(char: str) -> bool:
    """Check if a single character is a Unicode punctuation character.

    Args:
        char: A single character.

    Returns:
        True if the character is punctuation, False otherwise.
    """
    cp = ord(char)
    # ASCII punctuation ranges
    if (
        (0x21 <= cp <= 0x2F)
        or (0x3A <= cp <= 0x40)
        or (0x5B <= cp <= 0x60)
        or (0x7B <= cp <= 0x7E)
    ):
        return True
    cat = unicodedata.category(char)
    return cat.startswith("P")
