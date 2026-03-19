# X (Twitter) Thread

---

**Tweet 1:**
Why does "gpt-4" cost 3 tokens but "gpt4" costs only 2?

I built BPE tokenization from scratch to find out.
Code: github.com/jrajath94/bpe-tokenizer
🧵

---

**Tweet 2:**
Most engineers treat tokenization as a black box.

That leads to:
- Truncated responses you can't predict
- 3x more tokens than expected for legal/code text
- Bugs that only show up in production

The fix: understand the algorithm from first principles.

---

**Tweet 3:**
BPE (Byte-Pair Encoding) in 4 steps:

1. Split every word into characters: "hello" → ["h","e","l","l","o"]
2. Count all adjacent pairs: ("l","l") = 42 times
3. Merge the most frequent: "ll" is now one token
4. Repeat until vocab is full

[Mermaid diagram: Corpus → PreTokenizer → Merge Loop → Vocabulary → Encoder]

The non-obvious part: the END_OF_WORD marker.
"the</w>" and "the" must be different tokens or "there" tokenizes wrong.

---

**Tweet 4:**
The hardest bug I hit:

Token 'o</w>' not found in vocabulary.

Root cause: I initialized 256 byte tokens. But word-final characters become "byte</w>" at runtime — not in the vocab.

Fix: add all 256 "byte</w>" variants to the initial vocab.

This is why roundtrip tests matter: decode(encode(text)) == text must always hold.

---

**Tweet 5:**
Benchmark results on a 292K char corpus:

BPE encoding:   151,318 tokens/sec
BPE compression: 3.02 chars/token at vocab=1000
WordPiece:      536,218 tokens/sec

For context: tiktoken (Rust) is 10x faster.
This is what the algorithm looks like before optimization.

Vocab compression vs size:
- 600 tokens → 1.72 chars/token
- 1000 tokens → 3.03 chars/token
- 2000 tokens → 5.28 chars/token

---

**Tweet 6:**
The project:

✅ BPE (GPT-2 style, byte-level)
✅ WordPiece (BERT-style, ## continuation)
✅ 144 tests, 92% coverage
✅ CLI: train, encode, decode, compare
✅ No external dependencies

Star it if useful. What should I build next?
github.com/jrajath94/bpe-tokenizer

#AI #MachineLearning #OpenSource #BuildInPublic #NLP #LLM
