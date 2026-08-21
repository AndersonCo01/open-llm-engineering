import pytest
import torch

from open_llm_engineering.embeddings import (
    TokenPositionEmbedding,
    make_context_windows,
    mean_pool_sequence,
)


def test_context_windows_and_targets_are_shifted() -> None:
    token_ids = torch.tensor([1, 2, 3, 4, 5])

    contexts, targets = make_context_windows(token_ids, context_length=2)

    assert torch.equal(contexts, torch.tensor([[1, 2], [2, 3], [3, 4]]))
    assert torch.equal(targets, torch.tensor([3, 4, 5]))


def test_context_windows_require_rank_one() -> None:
    with pytest.raises(ValueError, match="rank 1"):
        make_context_windows(torch.ones(2, 3), context_length=2)


def test_context_length_must_be_positive() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        make_context_windows(torch.tensor([1, 2, 3]), context_length=0)


def test_context_requires_a_following_target() -> None:
    with pytest.raises(ValueError, match="more token IDs"):
        make_context_windows(torch.tensor([1, 2]), context_length=2)


def test_embedding_output_has_batch_sequence_and_feature_axes() -> None:
    layer = TokenPositionEmbedding(vocab_size=8, context_length=4, embedding_dim=3)
    token_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])

    result = layer(token_ids)

    assert result.shape == (2, 3, 3)


def test_embedding_adds_the_correct_position_to_every_batch_item() -> None:
    layer = TokenPositionEmbedding(vocab_size=4, context_length=3, embedding_dim=2)
    with torch.no_grad():
        layer.token_embedding.weight.zero_()
        layer.position_embedding.weight.copy_(
            torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        )

    result = layer(torch.tensor([[0, 1, 2], [2, 1, 0]]))

    expected_positions = torch.tensor(
        [
            [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
            [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
        ]
    )
    assert torch.equal(result, expected_positions)


def test_embedding_rejects_sequence_longer_than_context() -> None:
    layer = TokenPositionEmbedding(vocab_size=5, context_length=2, embedding_dim=4)

    with pytest.raises(ValueError, match="context length"):
        layer(torch.tensor([[1, 2, 3]]))


def test_mean_pool_sequence_preserves_batch_and_features() -> None:
    embeddings = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[10.0, 20.0], [30.0, 40.0]],
        ]
    )

    result = mean_pool_sequence(embeddings)

    assert torch.equal(result, torch.tensor([[2.0, 3.0], [20.0, 30.0]]))


def test_mean_pool_sequence_requires_rank_three() -> None:
    with pytest.raises(ValueError, match="rank 3"):
        mean_pool_sequence(torch.ones(2, 3))
