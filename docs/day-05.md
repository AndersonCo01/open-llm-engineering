# Day 5 — Causal Self-Attention

## Goal

Build the mechanism that lets each token gather useful information from earlier
tokens in the same sequence.

Today you will implement:

```text
input vectors
    ├──→ queries ─┐
    ├──→ keys ────┼→ attention scores → causal mask → probabilities
    └──→ values ──┘                                      ↓
                                             weighted value vectors
```

## New vocabulary

- **Self-attention:** tokens exchange information with other tokens in the same
  sequence.
- **Query:** represents what a token is looking for.
- **Key:** represents what a token offers for matching.
- **Value:** contains the information that attention can retrieve.
- **Attention score:** measures query-key compatibility.
- **Attention weight:** a normalized score used to mix value vectors.
- **Causal mask:** prevents a token from looking at future positions.
- **Scaled dot product:** divides query-key scores by the square root of the
  feature dimension.

## Part 1 — Causal mask

For a sequence of length four, create this Boolean matrix:

```text
[[True,  False, False, False],
 [True,  True,  False, False],
 [True,  True,  True,  False],
 [True,  True,  True,  True ]]
```

Rows are the tokens asking questions. Columns are the tokens they could inspect.
A token may inspect itself and earlier positions, but not later positions.

Useful syntax:

```python
torch.ones(length, length, dtype=torch.bool)
torch.tril(matrix)
```

Complete `make_causal_mask()`.

## Part 2 — Scaled dot-product attention

Assume queries, keys, and values have shape:

```text
[batch, sequence, head_dimension]
```

Calculate scores:

```python
scores = queries @ keys.transpose(-2, -1)
scores = scores / math.sqrt(head_dimension)
```

The score shape becomes:

```text
[batch, sequence, sequence]
```

Apply the causal mask before softmax:

```python
scores = scores.masked_fill(~mask, float("-inf"))
```

Softmax turns each score row into weights that sum to one:

```python
weights = torch.softmax(scores, dim=-1)
```

Finally, mix the value vectors:

```python
output = weights @ values
```

Complete `scaled_dot_product_attention()`.

## Part 3 — Learned self-attention

The same input vectors are projected into three different roles:

```python
queries = self.query(x)
keys = self.key(x)
values = self.value(x)
```

The projection matrices are model parameters. Training teaches them which
relationships are useful.

After attention, an output projection transforms the mixed vectors:

```python
return self.output(attended)
```

Complete `CausalSelfAttention.forward()`.

## Why scaling matters

As the query/key dimension grows, raw dot products can become large. Large
values cause softmax to become extremely sharp, which can make optimization
difficult. Dividing by `sqrt(head_dimension)` keeps score magnitudes more stable.

## Assignment

Open `src/open_llm_engineering/attention.py` and complete:

1. `make_causal_mask`
2. `scaled_dot_product_attention`
3. `CausalSelfAttention.forward`

Run each group:

```bash
.venv/bin/python -m pytest -v tests/test_attention.py -k mask
.venv/bin/python -m pytest -v tests/test_attention.py -k scaled
.venv/bin/python -m pytest -v tests/test_attention.py -k module
```

Then run the complete project:

```bash
.venv/bin/python -m pytest -v
.venv/bin/python -m ruff check .
```

## Reflection questions

Answer these in `docs/day-05-reflection.md`:

1. What different roles do queries, keys, and values play?
2. Why are query-key scores divided by `sqrt(head_dimension)`?
3. Why must the causal mask be applied before softmax?
4. What does one row of the attention-weight matrix represent?
5. Why must every row of attention weights sum to one?
6. How is self-attention more powerful than the Day 3 bigram model?

## Definition of done

- all Day 5 tests pass;
- the complete project suite passes;
- Ruff passes;
- all reflection questions are answered in your own words;
- you can explain the shapes of queries, scores, weights, and output.
