# Open LLM Engineering

A hands-on journey from tensor fundamentals to a safe, local-first AI agent.
The code is intentionally small, tested, and explained so that each abstraction
can be understood rather than copied.

## Learning outcomes

- Build tokenizers and language models from first principles.
- Implement attention and a miniature GPT with PyTorch.
- Run open-weight models locally.
- Build retrieval-augmented generation and tool-using agents.
- Evaluate models and agents with reproducible tests.

## Day 1: tensors, environments, and Git

Read [`docs/day-01.md`](docs/day-01.md), run the guided example, and then complete
the three functions marked `TODO` in
[`src/open_llm_engineering/tensors.py`](src/open_llm_engineering/tensors.py).

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

### Run the lesson and tests

```bash
python -m open_llm_engineering.tensors
pytest
ruff check .
```

## Progress

- [x] Day 1 — Tensors, environments, and Git
- [x] Day 2 — Tokenization
- [x] Day 3 — Neural networks and language modeling
- [x] Day 4 — Embeddings and context
- [ ] Day 5 — Attention
- [ ] Day 6 — Miniature GPT
- [ ] Day 7 — Evaluation and first release
- [ ] Days 8–14 — Open models, RAG, agents, safety, and capstone

## License

MIT
