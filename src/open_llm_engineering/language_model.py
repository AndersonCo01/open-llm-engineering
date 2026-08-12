"""Day 3 exercises: build the smallest neural language model."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def make_next_token_pairs(token_ids: Tensor) -> tuple[Tensor, Tensor]:
    """Return current-token inputs and one-step-ahead targets.

    The input must be a rank-1 tensor containing at least two token IDs.
    """
    if token_ids.ndim != 1:
        raise ValueError("token_ids must have rank 1")

    if token_ids.numel() < 2:
        raise ValueError("token_ids must contain at least two token IDs")

    inputs = token_ids[:-1]
    targets = token_ids[1:]

    return inputs, targets


class BigramLanguageModel(nn.Module):
    """Predict the next token using only the current token."""

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        if vocab_size < 2:
            raise ValueError("vocab_size must be at least 2")

        # Each row contains next-token logits for one current token ID.
        self.token_logits = nn.Embedding(vocab_size, vocab_size)

    def forward(self, token_ids: Tensor) -> Tensor:
        """Return one next-token logit vector for every input token ID."""
        return self.token_logits(token_ids)


def calculate_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """Return cross-entropy loss for next-token predictions."""
    if logits.ndim != 2:
        raise ValueError("logits must have rank 2")

    if targets.ndim != 1:
        raise ValueError("targets must have rank 1")

    if logits.shape[0] != targets.shape[0]:
        raise ValueError("logits and targets must contain the same number of examples")

    return F.cross_entropy(logits, targets)


def generate(
    model: BigramLanguageModel,
    start_token_id: int,
    max_new_tokens: int,
    *,
    generator: torch.Generator | None = None,
) -> list[int]:
    """Sample token IDs autoregressively from a trained bigram model."""
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens cannot be negative")

    generated_ids = [int(start_token_id)]

    with torch.no_grad():
        for _ in range(max_new_tokens):
            current_token = torch.tensor([generated_ids[-1]], dtype=torch.long)
            logits = model(current_token)
            probabilities = F.softmax(logits, dim=-1)
            sampled = torch.multinomial(probabilities, 1, generator=generator)
            generated_ids.append(int(sampled.item()))

    return generated_ids
