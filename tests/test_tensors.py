import pytest
import torch

from open_llm_engineering.tensors import (
    batch_mean,
    cosine_similarity,
    matrix_multiply,
)


def test_matrix_multiply_returns_expected_values() -> None:
    left = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    right = torch.tensor([[2.0], [1.0]])

    result = matrix_multiply(left, right)

    assert torch.equal(result, torch.tensor([[4.0], [10.0]]))


def test_matrix_multiply_rejects_incompatible_shapes() -> None:
    with pytest.raises(ValueError, match="shape"):
        matrix_multiply(torch.ones(2, 3), torch.ones(2, 4))


def test_cosine_similarity_for_same_direction_is_one() -> None:
    result = cosine_similarity(torch.tensor([1.0, 2.0]), torch.tensor([2.0, 4.0]))

    assert torch.isclose(result, torch.tensor(1.0))


def test_cosine_similarity_rejects_zero_vector() -> None:
    with pytest.raises(ValueError, match="zero"):
        cosine_similarity(torch.zeros(2), torch.ones(2))


def test_batch_mean_preserves_batch_and_feature_axes() -> None:
    batch = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[10.0, 20.0], [30.0, 40.0]],
        ]
    )

    result = batch_mean(batch)

    expected = torch.tensor([[2.0, 3.0], [20.0, 30.0]])
    assert torch.equal(result, expected)

