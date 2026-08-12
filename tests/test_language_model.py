import pytest
import torch

from open_llm_engineering.language_model import (
    BigramLanguageModel,
    calculate_loss,
    generate,
    make_next_token_pairs,
)


def test_make_next_token_pairs_shifts_targets() -> None:
    token_ids = torch.tensor([2, 5, 5, 7])

    inputs, targets = make_next_token_pairs(token_ids)

    assert torch.equal(inputs, torch.tensor([2, 5, 5]))
    assert torch.equal(targets, torch.tensor([5, 5, 7]))


@pytest.mark.parametrize("token_ids", [torch.tensor(4), torch.ones(2, 2)])
def test_make_next_token_pairs_requires_rank_one(token_ids: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="rank 1"):
        make_next_token_pairs(token_ids)


def test_make_next_token_pairs_requires_two_tokens() -> None:
    with pytest.raises(ValueError, match="at least two"):
        make_next_token_pairs(torch.tensor([1]))


def test_forward_returns_one_logit_per_vocabulary_item() -> None:
    model = BigramLanguageModel(vocab_size=4)
    token_ids = torch.tensor([0, 2, 3])

    logits = model(token_ids)

    assert logits.shape == (3, 4)
    assert torch.equal(logits[1], model.token_logits.weight[2])


def test_calculate_loss_matches_pytorch_cross_entropy() -> None:
    logits = torch.tensor([[3.0, 1.0], [0.5, 2.5]])
    targets = torch.tensor([0, 1])

    result = calculate_loss(logits, targets)

    expected = torch.nn.functional.cross_entropy(logits, targets)
    assert torch.allclose(result, expected)


def test_calculate_loss_rejects_mismatched_examples() -> None:
    with pytest.raises(ValueError, match="same number"):
        calculate_loss(torch.ones(3, 4), torch.tensor([1, 2]))


def test_generate_follows_the_only_likely_transition() -> None:
    model = BigramLanguageModel(vocab_size=3)
    with torch.no_grad():
        model.token_logits.weight.fill_(-100.0)
        model.token_logits.weight[0, 1] = 100.0
        model.token_logits.weight[1, 2] = 100.0
        model.token_logits.weight[2, 0] = 100.0

    result = generate(model, start_token_id=0, max_new_tokens=5)

    assert result == [0, 1, 2, 0, 1, 2]


def test_generate_rejects_negative_length() -> None:
    model = BigramLanguageModel(vocab_size=3)

    with pytest.raises(ValueError, match="cannot be negative"):
        generate(model, start_token_id=0, max_new_tokens=-1)
