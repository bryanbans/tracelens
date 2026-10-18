import pytest
import torch

from tracelens.models import SequenceModel, event_losses


@pytest.mark.parametrize('architecture', ['lstm', 'tcn'])
def test_future_tokens_cannot_change_prefix_predictions(architecture):
    torch.manual_seed(7)
    model = SequenceModel(12, architecture=architecture, embedding=8, hidden=12, dropout=0).eval()
    first = torch.randint(0, 12, (2, 19))
    second = first.clone()
    second[:, 9:] = (second[:, 9:] + 3) % 12
    assert torch.allclose(model(first)[:, :9], model(second)[:, :9], atol=1e-6)


def test_cross_entropy_gradient_matches_finite_difference():
    torch.manual_seed(2)
    model = SequenceModel(5, embedding=4, hidden=6, dropout=0).double().eval()
    tokens = torch.tensor([[1, 2, 3, 1, 4]])
    loss = event_losses(model, tokens).mean()
    loss.backward()
    analytic = model.head.weight.grad[2, 1].item()
    with torch.no_grad():
        original = model.head.weight[2, 1].item()
        epsilon = 1e-5
        model.head.weight[2, 1] = original + epsilon
        plus = event_losses(model, tokens).mean().item()
        model.head.weight[2, 1] = original - epsilon
        minus = event_losses(model, tokens).mean().item()
        model.head.weight[2, 1] = original
    assert analytic == pytest.approx((plus-minus)/(2*epsilon), rel=1e-4, abs=1e-7)


def test_order_probe_preserves_scored_target_counts():
    import numpy as np
    rng = np.random.default_rng(42)
    row = np.array([1, 2, 3, 4, 4, 5])
    changed = row.copy()
    rng.shuffle(changed[1:])
    assert changed[0] == row[0]
    assert sorted(changed[1:]) == sorted(row[1:])
