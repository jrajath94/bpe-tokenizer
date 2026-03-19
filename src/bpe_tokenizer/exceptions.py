"""Custom exceptions for the bpe-tokenizer library."""


class TokenizerError(Exception):
    """Base class for all tokenizer errors."""


class VocabBuildError(TokenizerError):
    """Raised when vocabulary construction fails.

    This can happen when the corpus is too small to reach the target
    vocabulary size, or when merge rules produce invalid states.
    """


class EncodingError(TokenizerError):
    """Raised when encoding fails for a given input text.

    Typically caused by text containing characters outside the known
    byte alphabet, or by attempting to encode with an untrained tokenizer.
    """


class DecodingError(TokenizerError):
    """Raised when decoding fails for given token IDs.

    Usually caused by IDs that are out of range for the current vocabulary.
    """


class TokenizerLoadError(TokenizerError):
    """Raised when loading a saved tokenizer fails.

    Can be caused by a missing file, corrupted JSON, or a version mismatch
    between the saved format and the current library version.
    """


class PreTokenizationError(TokenizerError):
    """Raised when pre-tokenization encounters an unexpected pattern."""
