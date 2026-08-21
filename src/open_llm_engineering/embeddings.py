"""Day 4 exercises: context windows and learned embeddings."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def make_context_windows(
    token_ids: Tensor,
    context_length: int,
) -> tuple[Tensor, Tensor]:
    """Return sliding context windows and their next-token targets."""
    if token_ids.ndim != 1:
        raise ValueError("token_ids must have rank 1")

    if context_length < 1:
        raise ValueError("context_length must be at least 1")

    if token_ids.numel() <= context_length:
        raise ValueError("sequence must contain more token IDs than the context length")

    contexts = []
    targets = []
    number_of_windows = token_ids.numel() - context_length

    for start in range(number_of_windows):
        end = start + context_length
        context = token_ids[start:end]
        target = token_ids[end]
        contexts.append(context)
        targets.append(target)

    return torch.stack(contexts), torch.stack(targets)


class TokenPositionEmbedding(nn.Module):
    """Combine learned token and position vectors."""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        if vocab_size < 2:
            raise ValueError("vocab_size must be at least 2")
        if context_length < 1:
            raise ValueError("context_length must be at least 1")
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be at least 1")

        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(context_length, embedding_dim)

    def forward(self, token_ids: Tensor) -> Tensor:
        """Return token vectors plus position vectors.

        token_ids must have shape [batch, sequence].
        """
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have rank 2")

        sequence_length = token_ids.shape[1]

        if sequence_length > self.context_length:
            raise ValueError("sequence length cannot exceed the context length")

        token_vectors = self.token_embedding(token_ids)

        position_ids = torch.arange(
            sequence_length,
            device=token_ids.device,
        )
        position_vectors = self.position_embedding(position_ids)

        return token_vectors + position_vectors


def mean_pool_sequence(embeddings: Tensor) -> Tensor:
    """Average [batch, sequence, features] over the sequence axis."""
    if embeddings.ndim != 3:
        raise ValueError("embeddings must have rank 3")

    return embeddings.mean(dim=1)
