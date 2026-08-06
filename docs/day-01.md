# Day 1 — Tensors, Python environments, and Git

## Mental model

An LLM repeatedly transforms numbers. Text first becomes token IDs; token IDs
become vectors; matrices mix those vectors; and the final numbers become token
probabilities. A tensor is the data structure that holds those numbers.

The word **dimension** can be confusing:

- `tensor.ndim` is the number of axes.
- `tensor.shape` is the length of each axis.
- A matrix with shape `[3, 2]` has two dimensions, three rows, and two columns.

In later lessons, a common shape will be `[batch, sequence, features]`:

- `batch`: how many examples are processed together;
- `sequence`: how many tokens are in each example;
- `features`: how many numbers represent each token.

## Guided code, line by line

Open `src/open_llm_engineering/tensors.py`.

`import torch` loads PyTorch, the numerical library used to build the model.
`from torch import Tensor` imports a type used to document function inputs and
outputs. It does not create a tensor by itself.

```python
vector = torch.tensor([1.0, 2.0, 3.0])
```

This creates a rank-1 tensor. The decimal points make its values floating-point
numbers, which can participate in neural-network calculations.

```python
mixed_values = weights @ values
```

`@` performs matrix multiplication. If `weights` has shape `[a, b]`, `values`
must have shape `[b, c]`, and the result has shape `[a, c]`. The two inner sizes
must match because every output value pairs one row from the left matrix with one
column from the right matrix.

## Your assignment

Complete the three functions marked `TODO`. Do not use an AI-generated final
solution on your first attempt.

Acceptance criteria:

1. `matrix_multiply` accepts compatible rank-2 tensors and rejects invalid ones.
2. `cosine_similarity` is implemented from dot product and L2 norms, not a
   ready-made cosine-similarity function.
3. `batch_mean` converts `[batch, sequence, features]` into `[batch, features]`.
4. All tests pass and `ruff check .` reports no errors.

Answer these questions in `docs/day-01-reflection.md`:

1. What is the difference between a tensor's shape and rank?
2. Why can a `[2, 3]` matrix multiply a `[3, 4]` matrix but not `[2, 4]`?
3. What shape does that valid multiplication produce, and why?
4. Why is a zero vector invalid for cosine similarity?
5. In `[batch, sequence, features]`, what does averaging dimension `1` remove?

## Git exercise

After the setup works:

```bash
git switch -c codex/day-01-tensors
git add .
git commit -m "feat: complete Day 1 tensor exercises"
```

Before committing, always inspect `git status` and `git diff --staged`. A commit
should represent one understandable change and its message should explain the
intent.

