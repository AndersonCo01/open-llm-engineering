import math

import pytest
import torch

from open_llm_engineering.attention import (
    CausalSelfAttention,
    make_causal_mask,
    scaled_dot_product_attention,
)


def test_mask_is_lower_triangular() -> None:
    result = make_causal_mask(4)

    expected = torch.tensor(
        [
            [True, False, False, False],
            [True, True, False, False],
            [True, True, True, False],
            [True, True, True, True],
        ]
    )
    assert torch.equal(result, expected)


def test_mask_requires_positive_sequence_length() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        make_causal_mask(0)


def test_scaled_attention_weights_sum_to_one() -> None:
    queries = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    keys = queries.clone()
    values = torch.tensor([[[10.0, 0.0], [0.0, 20.0]]])

    _, weights = scaled_dot_product_attention(queries, keys, values)

    assert torch.allclose(weights.sum(dim=-1), torch.ones(1, 2))


def test_scaled_attention_matches_known_calculation() -> None:
    queries = torch.tensor([[[1.0, 0.0]]])
    keys = torch.tensor([[[1.0, 0.0]]])
    values = torch.tensor([[[3.0, 7.0]]])

    output, weights = scaled_dot_product_attention(queries, keys, values)

    assert torch.equal(output, values)
    assert torch.equal(weights, torch.ones(1, 1, 1))


def test_scaled_attention_mask_blocks_future_positions() -> None:
    queries = torch.zeros(1, 3, 2)
    keys = torch.zeros(1, 3, 2)
    values = torch.tensor([[[1.0, 0.0], [3.0, 0.0], [8.0, 0.0]]])
    mask = make_causal_mask(3)

    output, weights = scaled_dot_product_attention(
        queries,
        keys,
        values,
        mask=mask,
    )

    assert torch.equal(weights[0, 0], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(output[0, 1], torch.tensor([2.0, 0.0]))
    assert torch.allclose(output[0, 2], torch.tensor([4.0, 0.0]))


def test_scaled_attention_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="identical shapes"):
        scaled_dot_product_attention(
            torch.ones(1, 2, 3),
            torch.ones(1, 2, 4),
            torch.ones(1, 2, 3),
        )


def test_attention_module_preserves_shape() -> None:
    layer = CausalSelfAttention(embedding_dim=4)
    embeddings = torch.randn(2, 3, 4)

    result = layer(embeddings)

    assert result.shape == embeddings.shape


def test_attention_module_first_token_cannot_depend_on_future() -> None:
    layer = CausalSelfAttention(embedding_dim=2)
    identity = torch.eye(2)
    with torch.no_grad():
        layer.query.weight.copy_(identity)
        layer.key.weight.copy_(identity)
        layer.value.weight.copy_(identity)
        layer.output.weight.copy_(identity)

    original = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
    changed_future = original.clone()
    changed_future[0, 1:] = torch.tensor([[100.0, 50.0], [-20.0, 80.0]])

    original_result = layer(original)
    changed_result = layer(changed_future)

    assert torch.allclose(original_result[:, 0], changed_result[:, 0])


def test_attention_module_rejects_non_sequence_input() -> None:
    layer = CausalSelfAttention(embedding_dim=3)

    with pytest.raises(ValueError, match="rank 3"):
        layer(torch.ones(2, 3))


def test_attention_scale_uses_feature_dimension() -> None:
    queries = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]])
    keys = queries.clone()
    values = torch.eye(2).unsqueeze(0)

    _, weights = scaled_dot_product_attention(queries, keys, values)

    expected_first_weight = torch.softmax(
        torch.tensor([4.0, 0.0]) / math.sqrt(2), dim=0
    )[0]
    assert torch.allclose(weights[0, 0, 0], expected_first_weight)
