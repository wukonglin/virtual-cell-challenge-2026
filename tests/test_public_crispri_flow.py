"""Synthetic CPU-only tests. No public expression, pretrained weights or cloud."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from public_crispri_flow import ConditionalPublicFlow, FlowConfig, flow_matching_loss, sample_heun


@pytest.fixture
def batch():
    gen = torch.Generator().manual_seed(41)
    control = torch.rand(8, 5, generator=gen)
    features = torch.randn(8, 3, generator=gen)
    context = torch.randn(8, 4, generator=gen)
    mask = torch.ones_like(control, dtype=torch.bool)
    return control, features, context, mask


def test_initialization_is_control_baseline_and_preserves_cell_variance(batch):
    model = ConditionalPublicFlow(FlowConfig(5, 3, 4, 16, 1))
    control, target, context, mask = batch
    result = sample_heun(model, control, target, context, mask, steps=3)
    assert torch.equal(result, control)
    assert torch.equal(result.var(0), control.var(0))
    assert model.training


def test_loss_backpropagates_and_step_updates_head(batch):
    model = ConditionalPublicFlow(FlowConfig(5, 3, 4, 16, 1))
    control, target, context, mask = batch
    before = model.output.weight.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    loss = flow_matching_loss(model, control, control + 0.3, target, context, mask,
                              time=torch.full((8, 1), .5))
    assert loss.item() == pytest.approx(.09)
    loss.backward()
    optimizer.step()
    assert not torch.equal(before, model.output.weight)


def test_target_context_and_evolving_state_have_real_computational_paths(batch):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model = ConditionalPublicFlow(FlowConfig(5, 3, 4, 16, 1))
        torch.nn.init.normal_(model.output.weight, std=.1)
    control, target, context, mask = batch
    control, target, context = [v.clone().requires_grad_() for v in (control, target, context)]
    value = model(control, target, context, torch.full((8, 1), .5), mask)
    value.square().sum().backward()
    for variable in (control, target, context):
        assert variable.grad is not None and variable.grad.abs().sum() > 0


def test_missing_genes_do_not_affect_loss_or_rollout(batch):
    control, target, context, mask = batch
    mask[:, -1] = False
    model = ConditionalPublicFlow(FlowConfig(5, 3, 4, 16, 1))
    treated = control + .2
    treated[:, -1] = 1e6
    loss = flow_matching_loss(model, control, treated, target, context, mask,
                              time=torch.full((8, 1), .5))
    assert loss.item() == pytest.approx(.04)
    result = sample_heun(model, control, target, context, mask, steps=2)
    assert (result[:, -1] == 0).all()
    assert torch.equal(result[:, :-1], control[:, :-1])


def test_inference_only_noise_is_not_supported(batch):
    model = ConditionalPublicFlow(FlowConfig(5, 3, 4, 16, 1))
    with pytest.raises(TypeError, match="noise_std"):
        sample_heun(model, *batch, steps=1, noise_std=.1)


@pytest.mark.parametrize("steps", [0, -1, True, 2.5])
def test_invalid_integration_steps_rejected(batch, steps):
    with pytest.raises(ValueError, match="steps"):
        sample_heun(ConditionalPublicFlow(FlowConfig(5, 3, 4)), *batch, steps=steps)


def test_bad_mask_and_time_rejected(batch):
    control, target, context, mask = batch
    model = ConditionalPublicFlow(FlowConfig(5, 3, 4, 16, 1))
    with pytest.raises(ValueError, match="observed"):
        sample_heun(model, control, target, context, torch.zeros_like(mask))
    with pytest.raises(ValueError, match="time"):
        model(control, target, context, torch.full((8, 1), 1.01), mask)
    with pytest.raises(ValueError, match="Boolean"):
        sample_heun(model, control, target, context, mask.float())


@pytest.mark.parametrize("dimension", [0, -1, True, 1.5])
def test_invalid_dimensions_rejected(dimension):
    with pytest.raises(ValueError, match="positive integer"):
        FlowConfig(dimension, 3, 4)
