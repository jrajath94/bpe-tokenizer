# LinkedIn Post

---

I just open-sourced bpe-tokenizer — here's why tokenization is more important than most ML engineers realize.

At JPMorgan, we spent a week debugging inconsistent LLM response truncation. We eventually traced it to tokenization edge cases: legal terminology with hyphens and special characters was consuming 3x more context window than we expected. We had been treating tiktoken as a black box. We had no framework for predicting this behavior.

I built bpe-tokenizer to fix that knowledge gap permanently. It's from-scratch implementations of BPE (the algorithm behind GPT-2, RoBERTa, LLaMA) and WordPiece (the algorithm behind BERT and DistilBERT) — zero external dependencies, 144 tests at 92% coverage, CLI for training and encoding, and a full architecture writeup. The hardest part was a subtle bug: word-final characters become "char</w>" tokens at runtime but I hadn't included them in the initial vocabulary. This caused encoding failures for rare words — exactly the kind of silent bug that only appears in production with edge-case inputs.

The benchmarks surprised me: pure Python BPE achieves 151K tokens/sec and 3.02 characters per token at vocab_size=1000. WordPiece hits 536K tokens/sec with greedy longest-match encoding. Compression ratio scales predictably with vocabulary size — from 1.72x at 600 tokens to 5.28x at 2000 tokens.

Anyone building LLM infrastructure should understand these algorithms at this level. Tokenization bugs are silent, unpredictable, and show up at the worst times. The best defense is a mental model for what the algorithm actually does.

→ GitHub: github.com/jrajath94/bpe-tokenizer

#AI #MachineLearning #SoftwareEngineering #OpenSource #NLP #LLM #Tokenization
