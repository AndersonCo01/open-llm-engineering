# Day 7 — Evaluation and First Release

## Goal

Turn MiniGPT from an architecture into a reproducible training artifact. Build
shifted batches, perform gradient updates, evaluate on held-out data, report
perplexity, and save a versioned checkpoint.

```text
token data
   ├── training batches → forward → loss → backward → optimizer step
   └── validation batches → no gradients → mean loss → perplexity
                                                    ↓
                                          versioned checkpoint
```

## New vocabulary

- **Training set:** examples used to update model parameters.
- **Validation set:** separate examples used to measure generalization.
- **Optimizer:** algorithm that changes parameters using their gradients.
- **Training step:** one forward pass, backward pass, and parameter update.
- **Evaluation mode:** disables training-specific layer behavior.
- **Held-out data:** data deliberately excluded from parameter updates.
- **Perplexity:** `exp(cross_entropy)`; lower values indicate better next-token
  predictions on the evaluated data.
- **Checkpoint:** saved model parameters plus metadata needed to identify the
  experiment.
- **Reproducibility:** ability to repeat an experiment with the same setup.
- **Release tag:** a stable Git name such as `v0.1.0` marking a milestone.

## Part 1 — Shifted batches

For the sequence `[0, 1, 2, 3, 4]` and context length three:

```text
input  [0, 1, 2] → target [1, 2, 3]
input  [1, 2, 3] → target [2, 3, 4]
```

Unlike Day 4's single target per window, every input position now has a target.
Complete `make_training_batches()`.

## Part 2 — Training step

One update follows this order:

```python
model.train()
optimizer.zero_grad(set_to_none=True)
logits = model(inputs)
loss = language_model_loss(logits, targets)
loss.backward()
optimizer.step()
```

Old gradients must be cleared because PyTorch accumulates gradients by default.
Complete `train_step()`.

## Part 3 — Honest evaluation

Evaluation must:

- use held-out batches;
- call `model.eval()`;
- run inside `torch.no_grad()`;
- weight batch losses by their number of target positions; and
- restore the model's previous mode afterward.

Convert mean validation loss to perplexity with `math.exp(loss)`. Complete
`evaluate_model()` and `perplexity()`.

## Part 4 — Checkpoints and release metadata

A checkpoint stores a model's `state_dict`, not the whole Python object. This
keeps the artifact easier to inspect and restore into the known architecture.

Store simple metadata such as:

```python
{"epoch": 10, "validation_loss": 1.42, "version": "0.1.0"}
```

Complete `save_checkpoint()` and `load_checkpoint()`.

## Assignment commands

```bash
.venv/bin/python -m pytest -v tests/test_evaluation.py -k training_batches
.venv/bin/python -m pytest -v tests/test_evaluation.py -k train_step
.venv/bin/python -m pytest -v tests/test_evaluation.py -k "evaluation or perplexity"
.venv/bin/python -m pytest -v tests/test_evaluation.py -k checkpoint
```

Then verify the project:

```bash
.venv/bin/python -m pytest -v
.venv/bin/python -m ruff check .
```

## Release checklist

- all tests and Ruff pass;
- Day 7 reflection is complete;
- README marks Day 7 complete;
- GitHub Actions passes on the pull request;
- merge the pull request into `main`;
- create the annotated tag `v0.1.0` after the merge.

## Definition of done

- you can explain training versus evaluation;
- validation loss is calculated without parameter updates;
- perplexity is reported correctly;
- a checkpoint round trip restores identical parameters;
- release `v0.1.0` is reproducible from the repository.
