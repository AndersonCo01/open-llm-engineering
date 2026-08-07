# Day 2 Reflection

## 1. Numerical input

Why must text be converted into integers before entering an LLM?

My answer:

LLMs cannot process raw text directly because neural networks perform mathematical operations on numbers. A tokenizer converts text into integer token IDs. The model then uses those IDs to look up numerical vectors called embeddings, which it can process through matrix multiplication and other calculations.

## 2. Deterministic vocabulary

Why do we sort unique characters before assigning token IDs?

My answer:

We sort the unique characters so token IDs are assigned in a consistent and predictable order. A set removes duplicates but does not guarantee the order we want. Sorting ensures that the same training text always produces the same vocabulary mapping, which prevents token IDs from changing between runs.

## 3. Unknown tokens

What problem does `<UNK>` solve?

My answer:

`<UNK>` represents characters or tokens that are not present in the tokenizer's vocabulary. Instead of crashing when the tokenizer encounters an unseen character, it safely converts that character to the reserved `<UNK>` ID, which is `0` in our implementation.

## 4. Character-level information

What information does character-level tokenization preserve?

My answer:

Character-level tokenization preserves the exact sequence and identity of individual characters, including letters, digits, spaces, and punctuation marks that are present in its vocabulary. It can represent unfamiliar words by splitting them into known characters, although completely unseen characters may still become `<UNK>`.


## 5. Character-level disadvantages

What disadvantages might characters have compared with subword tokens?

My answer:

Compared with subword tokens, character-level tokens produce much longer sequences. This increases computational cost because self-attention scales approximately quadratically with sequence length. Character models may also take longer to learn meaningful semantic patterns and require larger context windows to process the same amount of text. 