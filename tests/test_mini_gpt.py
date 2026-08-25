import pytest
import torch
from torch import nn
from torch.nn import functional as F

from open_llm_engineering.mini_gpt import (
    FeedForward,
    MiniGPT,
    TransformerBlock,
    generate_tokens,
    language_model_loss,
)


def test_feed_forward_preserves_shape() -> None:
    layer = FeedForward(embedding_dim=4)
    embeddings = torch.randn(2, 3, 4)

    assert layer(embeddings).shape == embeddings.shape


def test_feed_forward_rejects_non_sequence_input() -> None:
    with pytest.raises(ValueError, match="rank 3"):
        FeedForward(4)(torch.ones(2, 4))


def test_transformer_block_preserves_shape() -> None:
    block = TransformerBlock(embedding_dim=4)
    embeddings = torch.randn(2, 3, 4)

    assert block(embeddings).shape == embeddings.shape


def test_transformer_block_residual_path_preserves_input() -> None:
    block = TransformerBlock(embedding_dim=2)
    with torch.no_grad():
        for parameter in block.attention.parameters():
            parameter.zero_()
        for parameter in block.feed_forward.parameters():
            parameter.zero_()

    embeddings = torch.randn(1, 3, 2)

    assert torch.equal(block(embeddings), embeddings)


def test_transformer_block_rejects_non_sequence_input() -> None:
    with pytest.raises(ValueError, match="rank 3"):
        TransformerBlock(3)(torch.ones(2, 3))


def test_mini_gpt_returns_one_logit_vector_per_token() -> None:
    model = MiniGPT(7, context_length=4, embedding_dim=6, number_of_blocks=2)
    token_ids = torch.tensor([[1, 2, 3], [3, 2, 1]])

    assert model(token_ids).shape == (2, 3, 7)


def test_mini_gpt_rejects_non_batch_input() -> None:
    model = MiniGPT(5, context_length=3, embedding_dim=4)

    with pytest.raises(ValueError, match="rank 2"):
        model(torch.tensor([1, 2, 3]))


def test_mini_gpt_respects_context_length() -> None:
    model = MiniGPT(5, context_length=2, embedding_dim=4)

    with pytest.raises(ValueError, match="context length"):
        model(torch.tensor([[1, 2, 3]]))


def test_language_model_loss_matches_pytorch() -> None:
    logits = torch.tensor(
        [[[2.0, 0.0, -1.0], [0.0, 3.0, 1.0]]]
    )
    targets = torch.tensor([[0, 1]])

    expected = F.cross_entropy(logits.reshape(-1, 3), targets.reshape(-1))

    assert torch.equal(language_model_loss(logits, targets), expected)


def test_language_model_loss_rejects_mismatched_positions() -> None:
    with pytest.raises(ValueError, match="batch and sequence"):
        language_model_loss(torch.ones(2, 3, 4), torch.ones(2, 2, dtype=torch.long))


class PredictNextToken(nn.Module):
    context_length = 2

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        batch, sequence = token_ids.shape
        logits = torch.full((batch, sequence, 4), -100.0)
        next_ids = (token_ids + 1) % 4
        logits.scatter_(2, next_ids.unsqueeze(-1), 100.0)
        return logits


def test_generation_samples_from_the_final_position() -> None:
    model = PredictNextToken()

    result = generate_tokens(model, [0], max_new_tokens=3)  # type: ignore[arg-type]

    assert result == [0, 1, 2, 3]


def test_generation_crops_context_without_losing_prompt() -> None:
    model = PredictNextToken()

    result = generate_tokens(model, [2, 3, 0], max_new_tokens=1)  # type: ignore[arg-type]

    assert result == [2, 3, 0, 1]


@pytest.mark.parametrize("prompt_ids,max_new_tokens", [([], 1), ([1], -1)])
def test_generation_rejects_invalid_arguments(
    prompt_ids: list[int],
    max_new_tokens: int,
) -> None:
    with pytest.raises(ValueError):
        generate_tokens(
            PredictNextToken(),  # type: ignore[arg-type]
            prompt_ids,
            max_new_tokens,
        )
