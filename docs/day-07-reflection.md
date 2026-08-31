# Day 7 Reflection

## 1. Shifted targets

Why is every training target shifted one position ahead of its input?

My answer:

Because a language model learns to predict the next token. For example:

```text
Input:  [I, love, machine, learning]
Target: [love, machine, learning, .]
```

At each position, the model sees the current and previous tokens and tries to
predict what comes next.

## 2. Clearing gradients

Why must gradients be cleared before every training step?

My answer:

In PyTorch, gradients accumulate by default. A typical training step is:

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

- `zero_grad()` clears gradients from the previous step;
- `backward()` calculates new gradients; and
- `step()` updates the parameters.

Without `zero_grad()`, new gradients would be added to old gradients, causing
unintended parameter updates.

## 3. Training versus evaluation

What is the difference between `model.train()` and `model.eval()`?

My answer:

They tell certain layers whether the model is training or evaluating.
`model.train()` activates training behavior, particularly for layers such as
dropout and batch normalization. `model.eval()` switches those layers to
evaluation behavior. Importantly, `model.eval()` does not disable gradient
calculation.

## 4. No gradients during evaluation

Why should evaluation run inside `torch.no_grad()`?

My answer:

During evaluation, we are not training the model, so we do not need gradients.
A common pattern is:

```python
model.eval()

with torch.no_grad():
    predictions = model(inputs)
```

This reduces memory usage, avoids building a computation graph, and makes
evaluation more efficient.

## 5. Held-out validation data

Why should validation data not be used for parameter updates?

My answer:

Validation data is supposed to measure how well the model performs on unseen
data. If validation examples are used to update model parameters, information
from the validation set leaks into training and the validation score becomes
less trustworthy.

```text
Training set   → learn parameters
Validation set → evaluate and tune choices
Test set       → final evaluation
```

## 6. Perplexity

What does perplexity measure, and is a lower or higher value better?

My answer:

Perplexity measures how uncertain or surprised a language model is when
predicting the correct tokens. It is commonly calculated from cross-entropy
loss:

```text
perplexity = exp(loss)

loss 1.0 → perplexity ≈ 2.72
loss 2.0 → perplexity ≈ 7.39
loss 3.0 → perplexity ≈ 20.09
```

Lower perplexity is better. A lower value means the model assigns higher
probability to the correct tokens.

## 7. Checkpoints

What information should a useful model checkpoint contain?

My answer:

At minimum, an inference checkpoint should contain the model parameters. A
checkpoint intended to resume training should also include optimizer state,
the current epoch or step, metrics, and relevant metadata. For example:

```python
checkpoint = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "epoch": epoch,
    "loss": loss,
}
```

## 8. Reproducible release

What makes a model release reproducible rather than merely downloadable?

My answer:

A reproducible release allows another person—or you six months later—to
recreate the same model and understand exactly how it was produced. It should
preserve:

- source-code version;
- model architecture and weights;
- training configuration;
- dataset name and version;
- tokenizer or vocabulary;
- dependency versions;
- random seeds;
- evaluation metrics; and
- instructions for training and inference.

For example, a good release might contain:

```text
project/
├── src/
├── tests/
├── configs/
│   └── model.yaml
├── checkpoints/
│   └── model.pt
├── requirements.txt
├── README.md
└── evaluation.json
```

These concepts describe the complete machine-learning training lifecycle:

```text
data → shifted targets → forward pass → loss → clear gradients
     → backpropagation → parameter update → validation
     → checkpoint → reproducible release
```

Understanding that flow is more important than memorizing individual PyTorch
commands.
