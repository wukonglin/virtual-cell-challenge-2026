"""Small continuous-condition flow prototype, not a trained/count-emitting model.

Inputs are fold-preprocessed continuous expression. A future public-data loader
must authenticate sources, mask reserved targets BEFORE expression reads, draw
controls/treated cells from the same matching stratum, and balance conditions.
Unpaired population samples are not measured before/after pairs. This module
does not download data, choose splits, fit feature encoders or submit jobs.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn


@dataclass(frozen=True)
class FlowConfig:
    gene_dim: int
    target_dim: int
    context_dim: int
    hidden_dim: int = 128
    layers: int = 2

    def __post_init__(self):
        for name, value in vars(self).items():
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")


def _matrix(value, shape, name, *, reference=None):
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape):
        raise ValueError(f"Invalid {name} shape")
    if not value.is_floating_point() or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite floating point")
    if reference is not None and (value.device != reference.device or value.dtype != reference.dtype):
        raise ValueError(f"{name} device/dtype differs")


def _mask(mask, control):
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
        raise ValueError("observed_mask must be Boolean")
    if mask.shape != control.shape or mask.device != control.device:
        raise ValueError("observed_mask shape/device differs")
    if not mask.any(dim=1).all():
        raise ValueError("Every cell must have observed genes")


def _time(time, control):
    _matrix(time, (len(control), 1), "time", reference=control)
    if not ((time >= 0) & (time <= 1)).all():
        raise ValueError("time must lie in [0, 1]")


class ConditionalPublicFlow(nn.Module):
    """MLP velocity field with continuous target and population-context branches.

    GenePT/GO missing-modality flags should be included in target_features by
    the conditioning module. No discrete recipient-context embedding is used.
    A zero-initialized head starts as the no-change control baseline; this is
    an initialization property, NOT evidence that trained samples avoid collapse.
    The exact starting cell is NOT an extra condition: doing so would turn the
    conditional source distribution into a point mass under unpaired training.
    """

    def __init__(self, config: FlowConfig):
        super().__init__()
        self.config = config
        width = config.hidden_dim
        self.state_projection = nn.Linear(config.gene_dim * 2, width)
        self.target_projection = nn.Linear(config.target_dim, width)
        self.context_projection = nn.Linear(config.context_dim, width)
        self.time_projection = nn.Linear(3, width)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 2),
                          nn.SiLU(), nn.Linear(width * 2, width))
            for _ in range(config.layers)
        ])
        self.output = nn.Linear(width, config.gene_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x_t, target_features, context_features, time, observed_mask):
        cfg = self.config
        if not isinstance(x_t, torch.Tensor) or x_t.ndim != 2 or len(x_t) == 0:
            raise ValueError("x_t must be a nonempty cell matrix")
        n = len(x_t)
        _matrix(x_t, (n, cfg.gene_dim), "x_t")
        _matrix(target_features, (n, cfg.target_dim), "target_features", reference=x_t)
        _matrix(context_features, (n, cfg.context_dim), "context_features", reference=x_t)
        _mask(observed_mask, x_t)
        _time(time, x_t)
        observed = observed_mask.to(x_t.dtype)
        state = torch.cat((torch.where(observed_mask, x_t, 0.0), observed), dim=-1)
        clock = torch.cat((time, torch.sin(math.pi * time), torch.cos(math.pi * time)), dim=-1)
        hidden = (self.state_projection(state) + self.target_projection(target_features)
                  + self.context_projection(context_features) + self.time_projection(clock))
        for block in self.blocks:
            hidden = hidden + block(hidden)
        return torch.where(observed_mask, self.output(torch.nn.functional.silu(hidden)), 0.0)


def flow_matching_loss(model, control, treated, target_features, context_features,
                       observed_mask, *, time):
    """Linear-path conditional flow loss on caller-matched unpaired populations.

    Time is supplied explicitly for auditable RNG. Equal per-cell weighting
    handles differing observed-gene counts. The caller must balance datasets,
    targets and contexts. Missing genes are not converted to biological zeros.
    """
    _matrix(control, control.shape, "control")
    _matrix(treated, control.shape, "treated", reference=control)
    _mask(observed_mask, control)
    _time(time, control)
    observed = observed_mask.to(control.dtype)
    source = torch.where(observed_mask, control, 0.0)
    destination = torch.where(observed_mask, treated, 0.0)
    x_t = (1 - time) * source + time * destination
    target_velocity = destination - source
    predicted = model(x_t, target_features, context_features, time, observed_mask)
    residual = torch.where(observed_mask, predicted, 0.0).float() - target_velocity.float()
    loss = (residual.square().sum(dim=1) / observed.sum(dim=1)).mean()
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite flow objective")
    return loss


@torch.inference_mode()
def sample_heun(model, control, target_features, context_features, observed_mask,
                *, steps=32):
    """Integrate a control-population continuous flow without rounding/clipping.

    Cell diversity starts from sampled controls, the same base as training. The
    output remains continuous expression, not UMI counts. Unobserved coordinates
    are zero placeholders and MUST travel with the original observed_mask.
    """
    if type(steps) is not int or steps < 1:
        raise ValueError("steps must be a positive integer")
    _matrix(control, control.shape, "control")
    _mask(observed_mask, control)
    x = torch.where(observed_mask, control, 0.0)
    was_training = model.training
    model.eval()
    try:
        dt = 1.0 / steps
        for step in range(steps):
            time = control.new_full((len(control), 1), step / steps)
            first = model(x, target_features, context_features, time, observed_mask)
            later = control.new_full((len(control), 1), (step + 1) / steps)
            second = model(x + dt * first, target_features, context_features, later, observed_mask)
            x = torch.where(observed_mask, x + dt * 0.5 * (first + second), 0.0)
            if not torch.isfinite(x).all():
                raise ValueError("Nonfinite rollout; no clipping fallback is permitted")
    finally:
        model.train(was_training)
    return x
