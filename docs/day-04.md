# Day 4 — Embeddings and Context

## Goal

Move beyond a bigram model that sees only one token. Today you will:

1. create fixed-length context windows;
2. turn token IDs into learned vectors;
3. add position vectors so token order is preserved;
4. pool a sequence into one summary vector.

The new pipeline is:

```text
token IDs → context windows → token embeddings ─┐
                                                ├→ contextual input vectors
                    positions → position vectors ┘
```

## New vocabulary

- **Embedding:** a learned dense vector representing a discrete item.
- **Embedding dimension:** the number of features in each embedding vector.
- **Embedding table:** a trainable matrix containing one vector per ID.
- **Context:** the tokens available when making a prediction.
- **Context length:** the number of earlier tokens in one input window.
- **Sliding window:** overlapping fixed-length sections of a sequence.
- **Position embedding:** a learned vector representing a token's position.
- **Sequence length:** the number of token positions in one example.
- **Pooling:** combining multiple vectors into one summary vector.

## Part 1 — Sliding context windows

Given:

```python
token_ids = [1, 2, 3, 4, 5]
context_length = 2
```

create these examples:

```text
context [1, 2] → target 3
context [2, 3] → target 4
context [3, 4] → target 5
```

The resulting shapes are:

```text
contexts: [3, 2]
targets:  [3]
```

Useful syntax:

```python
token_ids[start : start + context_length]
torch.stack(windows)
```

Complete `make_context_windows()`.

## Part 2 — Token embeddings

`nn.Embedding` is a trainable lookup table:

```python
nn.Embedding(vocab_size, embedding_dim)
```

If the input IDs have shape:

```text
[batch, sequence]
```

the output has shape:

```text
[batch, sequence, embedding_dim]
```

The integer IDs select rows; the learned rows contain the useful numerical
representations. A larger token ID does not mean a more important token.

## Part 3 — Position embeddings

Token embeddings identify **what** each token is, but they do not identify
**where** it appears. Create positions with:

```python
positions = torch.arange(sequence_length, device=token_ids.device)
```

For a sequence of length four, this produces:

```text
[0, 1, 2, 3]
```

Look up position vectors and add them to token vectors:

```python
combined = token_vectors + position_vectors
```

PyTorch broadcasting applies the same position vectors across every example in
the batch.

Complete `TokenPositionEmbedding.forward()`.

## Part 4 — Mean pooling

An embedding tensor commonly has shape:

```text
[batch, sequence, embedding_dim]
```

Mean pooling averages dimension `1`, removing the sequence axis:

```text
[batch, embedding_dim]
```

This creates one summary vector per sequence. It is simple, although it treats
every position as equally important.

Complete `mean_pool_sequence()`.

## Assignment

Open `src/open_llm_engineering/embeddings.py` and complete the TODOs in order:

1. `make_context_windows`
2. `TokenPositionEmbedding.forward`
3. `mean_pool_sequence`

Run one group at a time:

```bash
.venv/bin/python -m pytest -v tests/test_embeddings.py -k context
.venv/bin/python -m pytest -v tests/test_embeddings.py -k embedding
.venv/bin/python -m pytest -v tests/test_embeddings.py -k pool
```

Then run everything:

```bash
.venv/bin/python -m pytest -v
.venv/bin/python -m ruff check .
```

## Reflection questions

Answer these in `docs/day-04-reflection.md`:

1. Why does an LLM use embedding vectors instead of token IDs directly?
2. What shapes enter and leave `nn.Embedding` in this lesson?
3. Why do context windows overlap?
4. Why are position embeddings necessary?
5. How does broadcasting help when adding position embeddings?
6. What information is lost by mean pooling the sequence axis?

## Definition of done

- all Day 4 tests pass;
- all project tests pass;
- Ruff passes;
- all reflection questions are answered in your own words;
- you can explain every axis in `[batch, sequence, embedding_dim]`.
