"""Day 7 exercises: train, evaluate, and checkpoint MiniGPT."""

from __future__ import annotations

import math
from collections.abc import Sequence
from os import PathLike

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from open_llm_engineering.mini_gpt import language_model_loss

Batch = tuple[Tensor, Tensor]

# TODO 1: Validate rank, context_length, batch_size, and available tokens.
# TODO 2: Build every overlapping context and its one-position shift.
# TODO 3: Stack examples into batches, retaining the final partial batch.
# TODO 4: Return the list of (inputs, targets) batches.


def make_training_batches(
    token_ids: Tensor,
    context_length: int,
    batch_size: int,
) -> list[Batch]:
    """Create shifted input-target batches for next-token training."""
    if token_ids.ndim != 1:
        raise ValueError("token_ids must have rank 1")

    if context_length < 1:
        raise ValueError("context_length must be at least 1")

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    if token_ids.numel() <= context_length:
        raise ValueError("token_ids must contain more values than context_length")

    all_inputs = []
    all_targets = []

    number_of_examples = token_ids.numel() - context_length

    for start in range(number_of_examples):
        end = start + context_length

        inputs = token_ids[start:end]
        targets = token_ids[start + 1 : end + 1]

        all_inputs.append(inputs)
        all_targets.append(targets)

    stacked_inputs = torch.stack(all_inputs)
    stacked_targets = torch.stack(all_targets)

    batches = []

    for start in range(0, number_of_examples, batch_size):
        end = start + batch_size

        batches.append(
            (
                stacked_inputs[start:end],
                stacked_targets[start:end],
            )
        )

    return batches


def train_step(
    model: nn.Module,
    inputs: Tensor,
    targets: Tensor,
    optimizer: Optimizer,
) -> float:
    """Perform one gradient update and return the scalar loss."""
    model.train()
    optimizer.zero_grad(set_to_none=True)

    logits = model(inputs)
    loss = language_model_loss(logits, targets)

    loss.backward()
    optimizer.step()

    return float(loss.item())


def evaluate_model(
    model: nn.Module,
    batches: Sequence[Batch],
) -> float:
    """Return mean loss per target position without tracking gradients."""
    if not batches:
        raise ValueError("batches must contain at least one batch")

    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_positions = 0

    try:
        with torch.no_grad():
            for inputs, targets in batches:
                logits = model(inputs)
                loss = language_model_loss(logits, targets)

                position_count = targets.numel()
                total_loss += float(loss.item()) * position_count
                total_positions += position_count
    finally:
        model.train(was_training)

    return total_loss / total_positions


def perplexity(loss: float) -> float:
    """Convert a finite nonnegative cross-entropy loss to perplexity."""
    if not math.isfinite(loss) or loss < 0:
        raise ValueError("loss must be finite and nonnegative")

    return math.exp(loss)


def save_checkpoint(
    path: str | PathLike[str],
    model: nn.Module,
    *,
    metadata: dict[str, int | float | str] | None = None,
) -> None:
    """Save model parameters and simple release metadata."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "metadata": dict(metadata or {}),
    }

    torch.save(checkpoint, path)


def load_checkpoint(
    path: str | PathLike[str],
    model: nn.Module,
) -> dict[str, int | float | str]:
    """Restore model parameters and return saved metadata."""
    checkpoint = torch.load(path, weights_only=True)

    model.load_state_dict(checkpoint["model_state_dict"])

    metadata = checkpoint.get("metadata", {})
    return dict(metadata)
