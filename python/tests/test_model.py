import torch

from gambito_train.model import GambitoNet, parameter_count


def test_forward_shapes_and_ranges():
    net = GambitoNet().eval()
    x = torch.zeros(2, 19, 8, 8)
    policy, value = net(x)
    assert policy.shape == (2, 4168)
    assert value.shape == (2,)
    assert (value.abs() <= 1.0).all()


def test_parameter_budget():
    # The .cyw export target is ~1.5 MB of int8 weights.
    n = parameter_count(GambitoNet())
    assert 500_000 < n < 2_500_000, f"{n:,} params"
