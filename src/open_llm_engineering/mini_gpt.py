"""Day 6 exercises: assemble a miniature GPT language model."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from open_llm_engineering.attention import CausalSelfAttention
from open_llm_engineering.embeddings import TokenPositionEmbedding


class FeedForward(nn.Module):
    """A position-wise two-layer neural network."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be at least 1")

        hidden_dim = 4 * embedding_dim
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, embeddings: Tensor) -> Tensor:
        """Transform each token independently while preserving shape."""
        if embeddings.ndim != 3:
            raise ValueError("embeddings must have rank 3: [batch, sequence, features]")

        return self.network(embeddings)


class TransformerBlock(nn.Module):
    """One pre-normalized causal transformer block."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be at least 1")

        self.attention_norm = nn.LayerNorm(embedding_dim)
        self.attention = CausalSelfAttention(embedding_dim)
        self.feed_forward_norm = nn.LayerNorm(embedding_dim)
        self.feed_forward = FeedForward(embedding_dim)

    def forward(self, embeddings: Tensor) -> Tensor:
        """Apply attention and feed-forward residual updates."""
        if embeddings.ndim != 3:
            raise ValueError("embeddings must have rank 3: [batch, sequence, features]")

        embeddings = embeddings + self.attention(self.attention_norm(embeddings))

        embeddings = embeddings + self.feed_forward(self.feed_forward_norm(embeddings))

        return embeddings


class MiniGPT(nn.Module):
    """A small decoder-only transformer that predicts the next token."""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        embedding_dim: int,
        number_of_blocks: int = 1,
    ) -> None:
        super().__init__()
        if number_of_blocks < 1:
            raise ValueError("number_of_blocks must be at least 1")

        self.context_length = context_length
        self.embedding = TokenPositionEmbedding(
            vocab_size,
            context_length,
            embedding_dim,
        )
        self.blocks = nn.ModuleList(
            TransformerBlock(embedding_dim) for _ in range(number_of_blocks)
        )
        self.final_norm = nn.LayerNorm(embedding_dim)
        self.language_model_head = nn.Linear(
            embedding_dim,
            vocab_size,
            bias=False,
        )

    def forward(self, token_ids: Tensor) -> Tensor:
        """Return logits shaped [batch, sequence, vocabulary]."""
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have rank 2: [batch, sequence]")

        embeddings = self.embedding(token_ids)

        for block in self.blocks:
            embeddings = block(embeddings)

        normalized = self.final_norm(embeddings)
        logits = self.language_model_head(normalized)

        return logits


def language_model_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """Return next-token cross-entropy over batch and sequence positions."""
    if logits.ndim != 3:
        raise ValueError("logits must have rank 3:[batch, sequence, vocabulary]")

    if targets.ndim != 2:
        raise ValueError("tagets must have rank 2:[batch, sequence]")

    if logits.shape[:2] != targets.shape:
        raise ValueError(
            "logits and targets must have matchingbatch and sequence dimensions"
        )

    vocab_size = logits.shape[-1]

    flat_logits = logits.reshape(-1, vocab_size)
    flat_targets = targets.reshape(-1)

    return F.cross_entropy(flat_logits, flat_targets)


def generate_tokens(
    model: MiniGPT,
    prompt_ids: list[int],
    max_new_tokens: int,
    *,
    generator: torch.Generator | None = None,
) -> list[int]:
    """Sample new token IDs while respecting the model's context window."""
    if not prompt_ids:
        raise ValueError("prompt_ids cannot be empty")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens cannot be negative")
    generated_ids = [int(token_id) for token_id in prompt_ids]

    with torch.no_grad():
        for _ in range(max_new_tokens):
            context_ids = generated_ids[-model.context_length :]

            context = torch.tensor(
                [context_ids],
                dtype=torch.long,
            )

            logits = model(context)
            final_logits = logits[:, -1, :]
            probabilities = F.softmax(final_logits, dim=-1)

            sampled = torch.multinomial(
                probabilities,
                num_samples=1,
                generator=generator,
            )

            generated_ids.append(int(sampled.item()))

    return generated_ids
