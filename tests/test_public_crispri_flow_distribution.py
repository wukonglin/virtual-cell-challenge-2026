"""Distribution-flow regressions using tiny synthetic CPU tensors only."""
from __future__ import annotations

import inspect
import math
from pathlib import Path
import sys

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from public_crispri_flow import (  # noqa: E402
    ConditionalPublicFlow, FlowConfig, flow_matching_loss, sample_heun,
)


def _conditions(control):
    return (control.new_zeros((len(control), 2)),
            control.new_zeros((len(control), 2)))


def test_velocity_api_cannot_condition_separately_on_original_control():
    assert tuple(inspect.signature(ConditionalPublicFlow.forward).parameters) == (
        "self", "x_t", "target_features", "context_features", "time", "observed_mask")
    assert "noise_std" not in inspect.signature(sample_heun).parameters
    assert "generator" not in inspect.signature(sample_heun).parameters


def test_masked_extreme_finite_endpoints_do_not_overflow_loss():
    control = torch.tensor([[0., 3e38], [1., -3e38]])
    treated = torch.tensor([[.2, -3e38], [1.2, 3e38]])
    observed = torch.tensor([[True, False], [True, False]])
    model = ConditionalPublicFlow(FlowConfig(2, 2, 2, 8, 1))
    loss = flow_matching_loss(model, control, treated, *_conditions(control), observed,
                              time=torch.full((2, 1), .5))
    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(.04)
    loss.backward()
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters()
               if parameter.grad is not None)


def test_loss_weights_cells_equally_with_unequal_observed_gene_counts():
    control = torch.zeros(2, 3)
    treated = torch.tensor([[1., 77., 77.], [2., 2., 2.]])
    observed = torch.tensor([[True, False, False], [True, True, True]])
    model = ConditionalPublicFlow(FlowConfig(3, 2, 2, 8, 1))
    loss = flow_matching_loss(model, control, treated, *_conditions(control), observed,
                              time=torch.tensor([[.1], [.9]]))
    assert loss.item() == pytest.approx((1. + 4.) / 2.)
    assert loss.item() != pytest.approx((1. + 3. * 4.) / 4.)


def test_nonzero_head_masks_state_values_and_gradients_before_projection():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(918)
        model = ConditionalPublicFlow(FlowConfig(3, 2, 2, 8, 1))
        nn.init.normal_(model.output.weight, std=.2)
    state = torch.tensor([[.2, .7, 1.], [1.2, .4, -.3]], requires_grad=True)
    observed = torch.tensor([[True, False, True], [True, True, False]])
    target, context = _conditions(state)
    time = torch.tensor([[.25], [.75]])
    baseline = model(state, target, context, time, observed)
    changed = state.detach().clone()
    changed[0, 1], changed[1, 2] = 3e38, -3e38
    altered = model(changed, target, context, time, observed)
    assert torch.equal(baseline, altered)
    assert torch.count_nonzero(baseline[observed]) > 0
    assert torch.count_nonzero(baseline[~observed]) == 0
    baseline.square().sum().backward()
    assert torch.count_nonzero(state.grad[~observed]) == 0
    assert torch.count_nonzero(state.grad[observed]) > 0
    changed[0, 0] += 1.
    assert not torch.equal(baseline, model(changed, target, context, time, observed))


class _LinearStateVelocity(nn.Module):
    def forward(self, x_t, target_features, context_features, time, observed_mask):
        return torch.where(observed_mask, x_t, 0.)


def test_heun_linear_ode_has_second_order_convergence_without_clipping():
    control = torch.tensor([[1., -2.], [.5, 10.]], dtype=torch.float64)
    observed = torch.tensor([[True, True], [True, False]])
    source = torch.where(observed, control, 0.)
    exact = math.e * source
    model = _LinearStateVelocity()
    errors = []
    for steps in (4, 8, 16):
        result = sample_heun(model, control, *_conditions(control), observed, steps=steps)
        factor = (1. + 1. / steps + .5 / steps**2)**steps
        torch.testing.assert_close(result, source * factor, rtol=1e-13, atol=1e-13)
        errors.append((result - exact).abs().max().item())
        assert result[0, 1] < 0
        assert result[1, 1] == 0
    assert errors[0] / errors[1] > 3.4
    assert errors[1] / errors[2] > 3.6
    assert model.training


def test_heun_evaluates_both_time_endpoints_and_restores_eval_mode():
    class TimeVelocity(nn.Module):
        def __init__(self):
            super().__init__()
            self.times = []

        def forward(self, x_t, target_features, context_features, time, observed_mask):
            assert not self.training
            self.times.append(time.clone())
            return 2. * time.expand_as(x_t) * observed_mask

    control = torch.tensor([[-2., 9.], [.5, 1.]], dtype=torch.float64)
    observed = torch.tensor([[True, False], [True, True]])
    model = TimeVelocity().eval()
    result = sample_heun(model, control, *_conditions(control), observed, steps=2)
    torch.testing.assert_close(result, torch.where(observed, control + 1., 0.))
    assert [time[0, 0].item() for time in model.times] == [0., .5, .5, 1.]
    assert not model.training


@pytest.mark.parametrize("was_training", [False, True])
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_velocity_raises_and_restores_model_mode(was_training, value):
    class BadVelocity(nn.Module):
        def forward(self, x_t, target_features, context_features, time, observed_mask):
            assert not self.training
            return torch.full_like(x_t, value)

    control = torch.ones(2, 2)
    model = BadVelocity().train(was_training)
    with pytest.raises(ValueError, match="(?i)finite"):
        sample_heun(model, control, *_conditions(control),
                    torch.ones_like(control, dtype=torch.bool), steps=2)
    assert model.training is was_training


def test_nonfinite_observed_objective_is_rejected():
    control = torch.tensor([[0.]])
    treated = torch.tensor([[3e38]])
    model = ConditionalPublicFlow(FlowConfig(1, 2, 2, 8, 1))
    with pytest.raises(ValueError, match="(?i)finite"):
        flow_matching_loss(model, control, treated, *_conditions(control),
                           torch.ones_like(control, dtype=torch.bool),
                           time=torch.zeros(1, 1))
