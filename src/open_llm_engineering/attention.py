"""Day 5 exercises: causal scaled dot-product self-attention."""

from __future__ import annotations

import math  # Used after completing scaled attention.

import torch
from torch import Tensor, nn


def make_causal_mask(
    sequence_length: int,
    *,
    device: torch.device | None = None,
) -> Tensor:
    """Return a lower-triangular Boolean attention mask."""
    if sequence_length < 1:
        raise ValueError("Sequence_length must be at least 1")

    full_mask = torch.ones(
        sequence_length,
        sequence_length,
        dtype=torch.bool,
        device=device,
    )

    return torch.tril(full_mask)


def scaled_dot_product_attention(
    queries: Tensor,
    keys: Tensor,
    values: Tensor,
    *,
    mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Return attended values and attention weights.

    Queries, keys, and values must have shape [batch, sequence, features].
    """
    if queries.ndim != 3 or keys.ndim != 3 or values.ndim != 3:
        raise ValueError("queries, keys, and values must have rank 3")

    if queries.shape != keys.shape or queries.shape != values.shape:
        raise ValueError("queries, keys, and values must have identical shapes")

    feature_count = queries.shape[-1]

    scores = queries @ keys.transpose(-2, -1)
    scores = scores / math.sqrt(feature_count)

    if mask is not None:
        sequence_length = queries.shape[1]

        if mask.shape != (sequence_length, sequence_length):
            raise ValueError("mask must have shape [sequence, sequence]")

        if mask.dtype != torch.bool:
            raise ValueError("mask must be Boolean")

        scores = scores.masked_fill(~mask, float("-inf"))

    weights = torch.softmax(scores, dim=-1)
    output = weights @ values

    return output, weights


class CausalSelfAttention(nn.Module):
    """A single-head causal self-attention layer."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be at least 1")

        self.query = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.key = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.value = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.output = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, embeddings: Tensor) -> Tensor:

        if embeddings.ndim != 3:
            raise ValueError("embeddings must have rank 3: [batch, sequence, features]")

        queries = self.query(embeddings)
        keys = self.key(embeddings)
        values = self.value(embeddings)

        sequence_length = embeddings.shape[1]
        mask = make_causal_mask(
            sequence_length,
            device=embeddings.device,
        )

        attended, _ = scaled_dot_product_attention(
            queries,
            keys,
            values,
            mask=mask,
        )

        return self.output(attended)
