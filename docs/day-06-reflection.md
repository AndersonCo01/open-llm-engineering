# Day 6 Reflection

## 1. Attention and feed-forward roles

What different jobs do attention and the feed-forward network perform?

My answer:

Attention allows each token to gather relevant information from other tokens in
the context. The feed-forward network then independently processes and
transforms each token's updated representation.

In short, attention mixes information across tokens, while the feed-forward
network processes information within each token.

## 2. Residual connections

Why do transformer blocks add a sublayer's input back to its output?

My answer:

Residual connections add the original input back to the sublayer's output so
important information is preserved. They also create a more direct path for
gradients, helping deeper transformers train more reliably.

## 3. Layer normalization

What does layer normalization help stabilize?

My answer:

Layer normalization keeps the scale and distribution of each token's hidden
features controlled. This improves optimization stability and helps the model
train more reliably.

## 4. Logit shape

Why does MiniGPT return logits shaped `[batch, sequence, vocabulary]`?

My answer:

MiniGPT returns logits shaped `[batch, sequence, vocabulary]` because it
produces one next-token prediction for every token position in every sequence:

- `batch`: number of sequences;
- `sequence`: positions within each sequence; and
- `vocabulary`: score for every possible next token.

Each `[batch, position, :]` vector contains the vocabulary logits for that
position.

## 5. Training shift

If the input is `[a, b, c]`, what should the targets be, and why?

My answer:

If the original sequence is `[a, b, c, d]`, the training pair should be:

```text
Inputs:  [a, b, c]
Targets: [b, c, d]
```

Each input position predicts the token immediately following it:

- `a` predicts `b`;
- `b` predicts `c`; and
- `c` predicts `d`.

This teaches the model to perform next-token prediction.

## 6. Context cropping

Why does generation retain only the latest `context_length` token IDs?

My answer:

Generation retains only the latest `context_length` token IDs because MiniGPT
cannot process a sequence longer than the context length it was designed or
trained to handle.

The latest tokens are kept because they usually provide the most relevant
context for predicting the next token. Cropping also limits memory use and
computation.

## 7. Bigram versus MiniGPT

What can MiniGPT learn that the Day 3 bigram model cannot?

My answer:

The bigram model predicts the next token using only the current token. MiniGPT
can use multiple earlier tokens through self-attention. Therefore, MiniGPT can
learn:

- longer-range relationships;
- word order and sentence patterns;
- context-dependent meanings;
- relationships between distant tokens; and
- more complex grammatical and semantic patterns.

For example, MiniGPT can use a subject from several positions earlier when
predicting a verb, while a bigram model cannot directly access that distant
context.
