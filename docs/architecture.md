# Architecture: bpe-tokenizer

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                         bpe-tokenizer                        │
│                                                              │
│  ┌──────────┐    ┌──────────────┐    ┌────────────────────┐ │
│  │Raw Text  │───▶│ PreTokenizer │───▶│  BPETokenizer  or  │ │
│  └──────────┘    └──────────────┘    │  WordPieceTokenizer│ │
│                                      └────────────────────┘ │
│                  ┌──────────────┐           │                │
│                  │  Vocabulary  │◀──────────┤                │
│                  │  (JSON file) │           │                │
│                  └──────────────┘    ┌──────▼─────────────┐ │
│                                      │  Token IDs         │ │
│                                      └────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

### `PreTokenizer` (pretokenizer.py)

Splits raw text into word-level tokens before subword tokenization. The pre-tokenizer determines which boundaries the BPE or WordPiece algorithm is NOT allowed to merge across.

Three strategies:
- `WHITESPACE`: Split on whitespace only
- `GPT2`: Regex-based (handles contractions, preserves leading spaces)
- `BERT`: Lowercase + split on whitespace and punctuation

### `BPETokenizer` (bpe.py)

Implements Sennrich et al. (2016). Uses byte-level encoding (GPT-2 style).

**Training flow:**
1. Pre-tokenize corpus into words
2. Initialize 512-token vocabulary (256 raw bytes + 256 `byte</w>` forms)
3. Count adjacent pair frequencies across all words
4. Merge the most frequent pair into a new token
5. Record the merge as a `MergeRule` with a rank
6. Repeat until reaching target vocab size

**Encoding flow:**
1. Pre-tokenize input into words
2. Convert each word to byte-level characters
3. Apply merge rules in rank order (lowest rank first)
4. Look up each resulting token in the vocabulary

### `WordPieceTokenizer` (wordpiece.py)

Implements Schuster & Nakamura (2012) / Devlin et al. (2018). Character-level with `##` continuation.

**Training flow:**
1. Pre-tokenize with BERT strategy (lowercase + punct split)
2. Initialize vocabulary with special tokens + all characters
3. Score each adjacent pair: `freq(AB) / (freq(A) * freq(B))`
4. Merge the highest-scoring pair
5. Repeat until target vocab size

**Encoding flow:**
1. Greedy longest-match-first from left to right
2. First subword: no prefix. Subsequent: `##` prefix
3. If no match found: entire word becomes `[UNK]`

### `Vocabulary` (vocab.py)

Container for token↔ID mapping and merge rules. Serializes to JSON for persistence.

## Data Flow: Training

```
corpus.txt
    │
    ▼ PreTokenizer.tokenize()
["the", "quick", "brown", ...]   ← word-level tokens
    │
    ▼ text_to_bytes() [BPE] or _word_to_initial_tokens() [WP]
[("t","h","e</w>",freq=42), ...]  ← character-level with END_OF_WORD
    │
    ▼ Merge loop (BPE: max frequency; WP: max score)
merge_rules = [(rank=0, left='Ġ', right='t', result='Ġt'), ...]
vocabulary  = {"Ġt": 512, "Ġth": 513, ...}
    │
    ▼ save_vocabulary()
vocab.json
```

## Data Flow: Encoding

```
"Hello, world!"
    │
    ▼ PreTokenizer.tokenize()
["Hello", ",", " world", "!"]
    │
    ▼ text_to_bytes() → byte-level chars
[["H","e","l","l","o</w>"], [",</w>"], [" ","w","o","r","l","d</w>"], ["!</w>"]]
    │
    ▼ _apply_merges_to_word() — for each word
Apply merge rank 0: ('Ġ','t') → 'Ġt' ... etc.
Result: ["H", "el", "l", "o</w>", ...]
    │
    ▼ Vocabulary.token_to_id lookup
[72, 564, 108, 367, ...]
```

## Key Implementation Details

### Why `char</w>` tokens are in the initial vocabulary

A word ending in character `'e'` has its last token encoded as `'e</w>'`. If only raw `'e'` is in the vocab, encoding fails for any word where `'e'` never participates in a merge with a following character. Adding all 256 `byte</w>` forms to the initial vocab guarantees completeness.

### Merge rank O(1) lookup

During encoding, we repeatedly find the adjacent pair with the lowest rank. A naive implementation would scan the merge rules list O(n) per step. The `_merge_rank` dict maps `(left, right) → rank` for O(1) lookup, making the merge application loop O(k log k) where k is the token count.

### Deterministic tie-breaking

When multiple pairs have the same frequency, we break ties alphabetically (`max(pair_freqs, key=lambda p: (pair_freqs[p], p))`). This ensures identical training runs produce identical vocabularies — critical for reproducibility and for matching saved vocabularies.
