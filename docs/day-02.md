# Day 2 — Tokenization from scratch

## Why tokenization exists

Language models perform numerical operations, so text must be represented as
integers. A tokenizer performs two core transformations:

```text
text --encode--> token IDs --decode--> text
```

Our first tokenizer treats each character as one token. Production LLMs usually
use subword tokenization, but a character tokenizer exposes the essential ideas
without hiding them behind a library.

## Vocabulary

A vocabulary is a mapping between tokens and integer IDs. Given the training
text `banana`, the unique characters are `a`, `b`, and `n`. We sort them so the
mapping is reproducible:

```python
{"<UNK>": 0, "a": 1, "b": 2, "n": 3}
```

`<UNK>` means "unknown." It handles characters that were not present when the
vocabulary was built. Reserving a special ID prevents the program from crashing
when it encounters new input.

## The four operations you will implement

### Build the vocabulary

Useful Python operations:

```python
unique_characters = sorted(set(training_text))
```

- `set(...)` removes duplicates.
- `sorted(...)` creates a deterministic order.
- `enumerate(items, start=1)` produces an index and value, beginning at 1.

### Reverse the mapping

Encoding needs `token -> ID`; decoding needs `ID -> token`. A dictionary
comprehension can reverse a dictionary:

```python
reverse = {value: key for key, value in original.items()}
```

### Encode

Iterate over input characters and use `dictionary.get(key, fallback)` so unseen
characters become ID `0`.

### Decode

Iterate over IDs, look up each token with a safe fallback, and combine the tokens
using `"".join(...)`.

## Assignment

Complete every `TODO` in `src/open_llm_engineering/tokenizer.py`.

Acceptance criteria:

1. Empty training text raises a helpful `ValueError`.
2. Vocabulary IDs are deterministic and reserve `0` for `<UNK>`.
3. Known text survives an encode/decode round trip.
4. Unknown characters and unknown integer IDs are handled safely.
5. All Day 1 and Day 2 tests pass.

Run only Day 2 tests while working:

```bash
.venv/bin/python -m pytest -v tests/test_tokenizer.py
```

Run the interactive demonstration directly from the source tree:

```bash
PYTHONPATH=src .venv/bin/python -m open_llm_engineering.tokenizer
```

## Reflection

Create `docs/day-02-reflection.md` and answer:

1. Why must text be converted to integers before entering an LLM?
2. Why do we sort the unique characters before assigning IDs?
3. What problem does `<UNK>` solve?
4. What information does character-level tokenization preserve?
5. What disadvantages might characters have compared with subword tokens?
