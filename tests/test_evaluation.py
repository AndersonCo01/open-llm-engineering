import math

import pytest
import torch

from open_llm_engineering.evaluation import (
    evaluate_model,
    load_checkpoint,
    make_training_batches,
    perplexity,
    save_checkpoint,
    train_step,
)
from open_llm_engineering.mini_gpt import MiniGPT, language_model_loss


def test_training_batches_are_shifted_and_shaped_correctly() -> None:
    token_ids = torch.tensor([0, 1, 2, 3, 4])

    batches = make_training_batches(token_ids, context_length=3, batch_size=2)

    inputs, targets = batches[0]
    assert torch.equal(inputs, torch.tensor([[0, 1, 2], [1, 2, 3]]))
    assert torch.equal(targets, torch.tensor([[1, 2, 3], [2, 3, 4]]))


def test_training_batches_keep_final_partial_batch() -> None:
    token_ids = torch.arange(7)

    batches = make_training_batches(token_ids, context_length=2, batch_size=3)

    assert [inputs.shape[0] for inputs, _ in batches] == [3, 2]


@pytest.mark.parametrize(
    "token_ids,context_length,batch_size",
    [
        (torch.ones(2, 2), 1, 1),
        (torch.arange(3), 0, 1),
        (torch.arange(3), 1, 0),
        (torch.arange(3), 3, 1),
    ],
)
def test_training_batches_reject_invalid_arguments(
    token_ids: torch.Tensor,
    context_length: int,
    batch_size: int,
) -> None:
    with pytest.raises(ValueError):
        make_training_batches(token_ids, context_length, batch_size)


def test_train_step_returns_loss_and_changes_parameters() -> None:
    torch.manual_seed(7)
    model = MiniGPT(6, context_length=3, embedding_dim=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    inputs = torch.tensor([[0, 1, 2], [1, 2, 3]])
    targets = torch.tensor([[1, 2, 3], [2, 3, 4]])
    before = model.language_model_head.weight.detach().clone()

    loss = train_step(model, inputs, targets, optimizer)

    assert math.isfinite(loss)
    assert model.training
    assert not torch.equal(model.language_model_head.weight, before)


def test_train_step_clears_gradients_between_updates() -> None:
    model = MiniGPT(5, context_length=2, embedding_dim=4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    inputs = torch.tensor([[0, 1]])
    targets = torch.tensor([[1, 2]])

    first = train_step(model, inputs, targets, optimizer)
    second = train_step(model, inputs, targets, optimizer)

    assert math.isfinite(first)
    assert math.isfinite(second)


def test_evaluation_matches_weighted_batch_losses_and_restores_mode() -> None:
    torch.manual_seed(11)
    model = MiniGPT(5, context_length=2, embedding_dim=4)
    batches = [
        (torch.tensor([[0, 1], [1, 2]]), torch.tensor([[1, 2], [2, 3]])),
        (torch.tensor([[2, 3]]), torch.tensor([[3, 4]])),
    ]
    model.train()
    with torch.no_grad():
        losses = [
            language_model_loss(model(inputs), targets).item()
            for inputs, targets in batches
        ]
    expected = (losses[0] * 4 + losses[1] * 2) / 6

    result = evaluate_model(model, batches)

    assert result == pytest.approx(expected)
    assert model.training
    assert all(parameter.grad is None for parameter in model.parameters())


def test_evaluation_rejects_empty_batches() -> None:
    model = MiniGPT(5, context_length=2, embedding_dim=4)

    with pytest.raises(ValueError, match="at least one"):
        evaluate_model(model, [])


def test_perplexity_is_exponential_loss() -> None:
    assert perplexity(math.log(10.0)) == pytest.approx(10.0)


@pytest.mark.parametrize("loss", [-0.1, float("inf"), float("nan")])
def test_perplexity_rejects_invalid_loss(loss: float) -> None:
    with pytest.raises(ValueError, match="finite and nonnegative"):
        perplexity(loss)


def test_checkpoint_round_trip_restores_parameters_and_metadata(tmp_path) -> None:
    torch.manual_seed(13)
    original = MiniGPT(5, context_length=2, embedding_dim=4)
    restored = MiniGPT(5, context_length=2, embedding_dim=4)
    checkpoint_path = tmp_path / "mini-gpt.pt"

    save_checkpoint(
        checkpoint_path,
        original,
        metadata={"epoch": 3, "validation_loss": 1.25, "version": "0.1.0"},
    )
    metadata = load_checkpoint(checkpoint_path, restored)

    for original_parameter, restored_parameter in zip(
        original.parameters(), restored.parameters(), strict=True
    ):
        assert torch.equal(original_parameter, restored_parameter)
    assert metadata == {
        "epoch": 3,
        "validation_loss": 1.25,
        "version": "0.1.0",
    }
