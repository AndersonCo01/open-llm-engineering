# Day 3 — Neural Networks and Language Modeling

## Goal

Build and train the smallest useful neural language model: a **bigram model**.
It sees one token and predicts the token that should come next.

By the end of today, you will understand this pipeline:

```text
text → token IDs → input/target pairs → logits → loss → gradients → updated weights
                                                        ↓
                                                 generated tokens
```

## New vocabulary

- **Language model:** predicts probabilities for tokens in a sequence.
- **Bigram:** a pair of adjacent tokens, such as `h → e` in `hello`.
- **Parameter:** a tensor value learned during training.
- **Logit:** a raw score produced by a model before probabilities.
- **Softmax:** converts logits into probabilities that sum to one.
- **Target:** the correct answer the model should predict.
- **Cross-entropy loss:** measures how poorly logits predict the target class.
- **Gradient:** tells how a parameter should change to affect the loss.
- **Backpropagation:** calculates gradients from the loss backward through the model.
- **Optimizer:** updates parameters using their gradients.
- **Learning rate:** controls the size of each parameter update.
- **Epoch:** one complete pass through the training examples.
- **Inference:** using learned parameters without updating them.
- **Sampling:** choosing a token from a probability distribution.

These terms are also available in the LLM Pocket Lab app.

## Part 1 — Create next-token training pairs

Given token IDs:

```python
[2, 5, 5, 7]
```

the model inputs are every token except the last:

```python
[2, 5, 5]
```

and the targets are every token except the first:

```python
[5, 5, 7]
```

The pairs are therefore `2 → 5`, `5 → 5`, and `5 → 7`.

Complete `make_next_token_pairs()` in
`src/open_llm_engineering/language_model.py`.

Important syntax:

```python
token_ids[:-1]  # start through the second-to-last value
token_ids[1:]   # second value through the end
```

## Part 2 — Understand the model

The model uses:

```python
nn.Embedding(vocab_size, vocab_size)
```

For Day 3, think of this as a learnable table:

- each **row** represents the current token ID;
- each **column** is a possible next token;
- each value is a **logit** for that next token.

If the input shape is `[batch]`, the output shape is:

```text
[batch, vocab_size]
```

Complete `BigramLanguageModel.forward()`.

## Part 3 — Calculate loss

Cross-entropy compares the logits with integer target IDs:

```python
loss = torch.nn.functional.cross_entropy(logits, targets)
```

You do **not** call softmax first. Cross-entropy accepts logits directly and
performs the stable normalization internally.

Complete `calculate_loss()`.

## Part 4 — Generate tokens

Generation repeats four steps:

1. give the latest token to the model;
2. convert its logits to probabilities with `softmax`;
3. sample one ID using `torch.multinomial`;
4. append the sampled ID and use it as the next input.

Complete `generate()`.

Use `torch.no_grad()` because generation does not train parameters.

## Part 5 — The training loop

After the four TODOs pass, study this standard sequence:

```python
optimizer.zero_grad()
logits = model(inputs)
loss = calculate_loss(logits, targets)
loss.backward()
optimizer.step()
```

Line by line:

1. `zero_grad()` removes gradients left from the previous step.
2. `model(inputs)` performs the forward pass.
3. `calculate_loss(...)` measures prediction error.
4. `backward()` computes gradients for every model parameter.
5. `step()` updates the parameters using those gradients.

## Assignment

Open `src/open_llm_engineering/language_model.py` in VS Code and complete the
four TODO sections in this order:

1. `make_next_token_pairs`
2. `BigramLanguageModel.forward`
3. `calculate_loss`
4. `generate`

Run one focused test group at a time:

```bash
.venv/bin/python -m pytest -v tests/test_language_model.py -k pairs
.venv/bin/python -m pytest -v tests/test_language_model.py -k forward
.venv/bin/python -m pytest -v tests/test_language_model.py -k loss
.venv/bin/python -m pytest -v tests/test_language_model.py -k generate
```

Then run the complete project checks:

```bash
.venv/bin/python -m pytest -v
.venv/bin/python -m ruff check .
```

## Reflection questions

Write your answers in `docs/day-03-reflection.md`:

1. What is the difference between a logit and a probability?
2. Why are targets shifted one position ahead of inputs?
3. Why should we not apply softmax before `cross_entropy`?
4. What does `loss.backward()` calculate?
5. Why must `optimizer.zero_grad()` run before the next backward pass?
6. What limitation does a bigram model have compared with a transformer?

## Definition of done

- all language-model tests pass;
- the full Python test suite passes;
- Ruff passes;
- you can explain every line in the training loop;
- the reflection answers are written in your own words.
