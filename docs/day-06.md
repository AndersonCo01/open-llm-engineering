# Day 6 — Assemble a Miniature GPT

## Goal

Combine the pieces from Days 1–5 into a decoder-only transformer that accepts
token IDs and produces next-token logits.

```text
token IDs
   ↓
token + position embeddings
   ↓
transformer block(s)
   ├── layer norm → causal attention → residual addition
   └── layer norm → feed-forward network → residual addition
   ↓
final layer norm
   ↓
vocabulary logits
```

## New vocabulary

- **Feed-forward network:** transforms each token independently after attention
  has exchanged information between tokens.
- **GELU:** a smooth nonlinear activation commonly used in transformers.
- **Residual connection:** adds a layer's input to its output, creating a direct
  path for information and gradients.
- **Layer normalization:** stabilizes each token vector before a sublayer.
- **Transformer block:** combines causal attention and a feed-forward network.
- **Language-model head:** converts each final token vector into vocabulary
  logits.
- **Decoder-only transformer:** predicts tokens using only preceding context.
- **Context cropping:** keeps only the most recent tokens that fit the model's
  context window during generation.

## Part 1 — Feed-forward network

The feed-forward network expands the feature dimension, applies a nonlinearity,
then projects back:

```text
embedding_dim → 4 × embedding_dim → embedding_dim
```

Unlike attention, it does not mix sequence positions. The same learned network
is applied independently to every token.

Complete `FeedForward.forward()`.

## Part 2 — Transformer block

Use pre-normalization and two residual connections:

```python
x = x + attention(attention_norm(x))
x = x + feed_forward(feed_forward_norm(x))
```

Attention communicates across time. The feed-forward network processes the
result at each position. Residual paths preserve the previous representation.

Complete `TransformerBlock.forward()`.

## Part 3 — Complete MiniGPT and loss

`MiniGPT.forward()` must preserve batch and sequence axes:

```text
token IDs: [batch, sequence]
logits:    [batch, sequence, vocabulary]
```

Every position predicts the token that follows it. Cross-entropy expects:

```text
flat logits:  [batch × sequence, vocabulary]
flat targets: [batch × sequence]
```

Complete `MiniGPT.forward()` and `language_model_loss()`.

## Part 4 — Autoregressive generation

Generation repeats this loop:

1. crop the available IDs to the context window;
2. run MiniGPT;
3. select logits from the final sequence position;
4. convert logits to probabilities;
5. sample and append one token.

Complete `generate_tokens()`.

## Assignment commands

```bash
.venv/bin/python -m pytest -v tests/test_mini_gpt.py -k feed_forward
.venv/bin/python -m pytest -v tests/test_mini_gpt.py -k transformer_block
.venv/bin/python -m pytest -v tests/test_mini_gpt.py -k "mini_gpt or language_model_loss"
.venv/bin/python -m pytest -v tests/test_mini_gpt.py -k generation
```

Then verify the whole project:

```bash
.venv/bin/python -m pytest -v
.venv/bin/python -m ruff check .
```

## Definition of done

- all Day 6 tests pass;
- the complete project tests pass;
- Ruff passes;
- every reflection answer is written in your own words;
- you can trace the tensor shapes from token IDs to vocabulary logits.
