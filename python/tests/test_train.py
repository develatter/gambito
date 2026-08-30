import torch

from gambito_train.train import loss_fn


def test_loss_rewards_correct_predictions():
    target_move = torch.tensor([7])
    target_z = torch.tensor([1.0])

    good_logits = torch.full((1, 4168), -10.0)
    good_logits[0, 7] = 10.0
    bad_logits = torch.full((1, 4168), -10.0)
    bad_logits[0, 0] = 10.0

    good = loss_fn(good_logits, torch.tensor([0.9]), target_move, target_z)
    bad = loss_fn(bad_logits, torch.tensor([-0.9]), target_move, target_z)
    assert good.ndim == 0  # single scalar, ready for .backward()
    assert good.item() < bad.item()


def test_loss_gradients_reach_both_heads():
    logits = torch.zeros(2, 4168, requires_grad=True)
    value = torch.zeros(2, requires_grad=True)
    loss = loss_fn(logits, value, torch.tensor([3, 4168 - 1]), torch.tensor([0.0, -1.0]))
    loss.backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0
    assert value.grad is not None and value.grad.abs().sum() > 0
