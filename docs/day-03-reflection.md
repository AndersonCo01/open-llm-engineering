# Day 3 Reflection

## 1. Logits versus probabilities

What is the difference between a logit and a probability?

My answer:

Logits are a model's raw, unnormalized output scores. They can be any real number and do not need to add up to 1. Probabilities are normalized values between 0 and 1 that add up to 1. Applying softmax converts logits into probabilities.

## 2. Shifted targets

Why are targets shifted one position ahead of inputs?

My answer:

Targets are shifted one position ahead because the model learns to predict the next token. For each input token at position `t`, the correct target is the token at position `t + 1`.

Inputs: `The cat is`

Targets: `cat is sleeping`

## 3. Cross-entropy

Why should we not apply softmax before `cross_entropy`?

My answer:

We should not apply softmax before `cross_entropy` because PyTorch's `cross_entropy` expects raw logits and internally applies `log_softmax`. Applying softmax first would be redundant, less numerically stable, and would give the function probabilities when it expects logits, changing the loss calculation.

## 4. Backpropagation

What does `loss.backward()` calculate?

My answer:

`loss.backward()` calculates the gradient of the loss with respect to every trainable model parameter. These gradients are stored in each parameter's `.grad` attribute. They describe how changing each parameter would affect the loss, and the optimizer uses them to update the parameters.

## 5. Clearing gradients

Why must `optimizer.zero_grad()` run before the next backward pass?

My answer:

PyTorch accumulates gradients by default. Therefore, `optimizer.zero_grad()` must clear gradients from the previous training step before the next call to `loss.backward()`. Otherwise, old and new gradients would be added together and could produce an incorrect parameter update.

## 6. Bigram limitation

What limitation does a bigram model have compared with a transformer?

My answer:

A bigram model predicts the next token using only the current token, so it cannot understand long-range context, sentence structure, or relationships between distant words. A transformer can consider many previous tokens at once using self-attention, allowing it to model more complex patterns and context.
