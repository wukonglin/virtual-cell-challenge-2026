"""Actual source-local public CRISPRi flow fitting; no I/O or cloud operations.

The caller authenticates cache/GO provenance and registers roles before calling.
This module independently checks the in-memory row/target boundaries. It never
uses panel responses, mixes treated sources, or emits competition count files.

The flow coordinate has an explicit zero atom at -0.25 and sqrt(x) for x > 0.
Its decoder is ReLU(z)^2, used identically for evaluation and inference. This is
a model parameterization, not an inference-only clipping repair. Data are the
registered 512-gene log1p(CP10000) proxy, NOT full-gene UMI counts. Unpaired
control/treated draws are population samples, not measured before/after pairs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import time
from typing import Callable

import numpy as np
import torch
from torch import nn

SCHEMA = "public-auxiliary-flow-v10"
SOURCES = ("replogle_k562", "nadig_jurkat")
ZERO_ATOM = -0.25


def require(value, message):
    if not value:
        raise ValueError(message)


def _ids(values, name, *, empty=False):
    require(not isinstance(values, (str, bytes)), name + " must be a sequence")
    result = tuple(values)
    require((empty or len(result) > 0) and all(isinstance(v, str) and v and v == v.strip()
            for v in result) and len(set(result)) == len(result), "Invalid " + name)
    return result


def _indices(value, size, name, *, ndim=1):
    a = np.asarray(value)
    require(a.ndim == ndim and a.dtype.kind in "iu" and a.dtype.kind != "b"
            and a.size > 0 and np.all(a >= 0) and np.all(a < size), "Invalid " + name)
    return a.astype(np.int64, copy=False)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                      allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class SourceData:
    source: str
    expression: np.ndarray
    original_rows: np.ndarray
    target_ids: tuple[str, ...]
    group_ids: tuple[str, ...]
    group_target: np.ndarray
    group_pool: np.ndarray
    treated_indices: tuple[np.ndarray, ...]
    fit_reference_indices: np.ndarray
    context_indices: np.ndarray
    context: np.ndarray
    fit_targets: tuple[str, ...]
    tune_targets: tuple[str, ...]
    excluded_targets: tuple[str, ...]


def prepare_source(arrays, *, fit_targets, tune_targets, excluded_targets):
    """Check the deduplicated cache interface, including exact role partitions.

    ``excluded_targets`` must include the caller-authenticated protected/panel
    unions. This pure function cannot discover omitted exclusions on its own.
    Data ownership is not transferred; callers must not mutate arrays afterwards.
    """
    source = str(np.asarray(arrays["source"]).item())
    require(source in SOURCES, "Exactly one registered public source is required")
    x = np.asarray(arrays["expression"])
    require(x.ndim == 2 and min(x.shape) > 0 and x.dtype == np.float32
            and np.isfinite(x).all() and np.all(x >= 0), "Invalid nonnegative expression")
    rows = np.asarray(arrays["original_rows"])
    require(rows.shape == (len(x),) and rows.dtype.kind in "iu" and np.all(rows >= 0)
            and np.all(rows[1:] > rows[:-1]), "Original rows must be sorted unique identities")
    row_role = np.asarray(arrays["row_role"])
    require(row_role.shape == rows.shape and row_role.dtype == np.uint8
            and np.all(np.isin(row_role, (1, 2, 3))), "Invalid row role")
    genes = _ids(np.asarray(arrays["genes"]).tolist(), "genes")
    require(len(genes) == x.shape[1], "Expression gene axis differs")
    groups = _ids(np.asarray(arrays["group_ids"]).tolist(), "group IDs")
    pool_ids = _ids(np.asarray(arrays["pool_ids"]).tolist(), "pool IDs")
    raw_targets = np.asarray(arrays["targets"])
    require(raw_targets.ndim == 1 and len(raw_targets) == len(groups)
            and all(isinstance(v, str) and v and v == v.strip() for v in raw_targets.tolist()),
            "Invalid group target labels")
    target_ids = tuple(sorted(set(raw_targets.tolist())))
    fit, tune = _ids(fit_targets, "fit targets"), _ids(tune_targets, "tuning targets")
    denied = _ids(excluded_targets, "excluded targets")
    require(len(fit) >= 2 and len(tune) >= 2 and not set(fit) & set(tune)
            and set(fit) | set(tune) == set(target_ids), "Fit/tuning targets must partition the source")
    require(not set(target_ids) & set(denied), "Protected/panel target admitted")
    lookup = {target: i for i, target in enumerate(target_ids)}
    group_target = np.asarray([lookup[t] for t in raw_targets], dtype=np.int64)
    pool = _indices(arrays["group_pool"], len(pool_ids), "group pool")
    require(pool.shape == (len(groups),) and set(pool.tolist()) == set(range(len(pool_ids))),
            "Every registered control pool must be used")
    group_batches = np.asarray(arrays["batches"])
    pool_batches = np.asarray(arrays["pool_batches"])
    require(group_batches.shape == (len(groups),) and pool_batches.shape == (len(pool_ids),)
            and np.array_equal(group_batches, pool_batches[pool]), "Group/control batch mismatch")
    offsets = np.asarray(arrays["treated_offsets"])
    require(offsets.shape == (len(groups) + 1,) and offsets.dtype.kind in "iu"
            and offsets[0] == 0 and np.all(np.diff(offsets.astype(np.int64)) >= 8)
            and np.all(np.diff(offsets.astype(np.int64)) <= 32), "Invalid treated group offsets")
    flat = _indices(arrays["treated_indices"], len(x), "treated indices")
    require(offsets[-1] == len(flat) and len(np.unique(flat)) == len(flat),
            "Repeated/missing treated row positions")
    reference = _indices(arrays["fit_reference_indices"], len(x), "fit reference indices", ndim=2)
    context_rows = _indices(arrays["context_indices"], len(x), "context indices", ndim=2)
    require(reference.shape == (len(pool_ids), 16) and context_rows.shape == (len(pool_ids), 32),
            "Expected disjoint fit16/context32 pools")
    for index, label, role in ((flat, "treated", 1), (reference, "fit reference", 2),
                               (context_rows, "context", 3)):
        require(len(np.unique(index)) == index.size and np.all(row_role[index] == role),
                "Repeated or mixed " + label + " rows")
        require(set(np.flatnonzero(row_role == role).tolist()) == set(index.reshape(-1).tolist()),
                "Unassigned " + label + " role rows")
    context = np.asarray(arrays["context"])
    require(context.shape == (len(pool_ids), 2 * x.shape[1]) and context.dtype == np.float32
            and np.isfinite(context).all(), "Invalid matched-control context")
    observed = x[context_rows].astype(np.float64)
    reconstructed = np.concatenate((observed.mean(1), observed.std(1)), axis=1).astype(np.float32)
    require(np.array_equal(context, reconstructed), "Context differs from context32 mean/std")
    treated = tuple(flat[int(offsets[i]):int(offsets[i + 1])] for i in range(len(groups)))
    for target in target_ids:
        ids = np.flatnonzero(raw_targets == target)
        require(len(ids) >= 2 and len(set(group_batches[ids].tolist())) == len(ids),
                "Each auxiliary target needs distinct replicated batches")
    return SourceData(source, x, rows, target_ids, groups, group_target, pool, treated,
                      reference, context_rows, context, fit, tune, denied)


@dataclass(frozen=True)
class PilotConfig:
    gene_dim: int
    target_dim: int
    hidden_dim: int = 128
    layers: int = 2
    steps: int = 600
    batch_size: int = 128
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    heun_steps: int = 16
    seed: int = 20260926
    max_seconds: float = 7200.0

    def __post_init__(self):
        for name in ("gene_dim", "target_dim", "hidden_dim", "layers", "steps", "batch_size", "heun_steps"):
            require(type(getattr(self, name)) is int and getattr(self, name) >= 1,
                    name + " must be a positive integer")
        require(type(self.seed) is int and 0 <= self.seed < 2**32, "Invalid seed")
        for name in ("learning_rate", "grad_clip", "max_seconds"):
            value = getattr(self, name)
            require(type(value) in (int, float) and math.isfinite(value) and value > 0,
                    "Invalid " + name)
        require(type(self.weight_decay) in (int, float) and math.isfinite(self.weight_decay)
                and self.weight_decay >= 0, "Invalid weight decay")


def _tensor(value, name, *, columns=None):
    require(isinstance(value, torch.Tensor) and value.ndim == 2 and len(value) > 0
            and value.is_floating_point() and torch.isfinite(value).all().item(), "Invalid " + name)
    if columns is not None:
        require(value.shape[1] == columns, name + " feature dimension differs")


def encode_expression(expression):
    _tensor(expression, "expression")
    require(torch.all(expression >= 0).item(), "Expression must be nonnegative")
    return torch.where(expression > 0, torch.sqrt(expression), ZERO_ATOM)


def decode_expression(coordinate):
    _tensor(coordinate, "flow coordinate")
    decoded = torch.relu(coordinate).square()
    require(torch.isfinite(decoded).all().item(), "Nonfinite decoded expression")
    return decoded


class AuxiliaryConditionalFlow(nn.Module):
    """Continuous target/context-conditioned velocity; no target/context IDs."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        h = config.hidden_dim
        self.state_projection = nn.Linear(config.gene_dim, h)
        self.target_projection = nn.Linear(config.target_dim, h)
        self.context_projection = nn.Linear(config.gene_dim * 2, h)
        self.time_projection = nn.Linear(3, h)
        self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 2*h),
            nn.SiLU(), nn.Linear(2*h, h)) for _ in range(config.layers)])
        self.output = nn.Linear(h, config.gene_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, coordinate, features, context, clock):
        _tensor(coordinate, "coordinate", columns=self.config.gene_dim)
        for value, name, width in ((features, "target features", self.config.target_dim),
                (context, "control context", 2*self.config.gene_dim), (clock, "time", 1)):
            _tensor(value, name, columns=width)
            require(len(value) == len(coordinate) and value.dtype == coordinate.dtype
                    and value.device == coordinate.device, name + " batch/device/dtype differs")
        require(torch.all((clock >= 0) & (clock <= 1)).item(), "Time must be in [0,1]")
        t = torch.cat((clock, torch.sin(math.pi*clock), torch.cos(math.pi*clock)), dim=-1)
        h = (self.state_projection(coordinate) + self.target_projection(features)
             + self.context_projection(context) + self.time_projection(t))
        for block in self.blocks:
            h = h + block(h)
        return self.output(torch.nn.functional.silu(h))


def matching_loss(model, control, treated, features, context, clock):
    """Equal-gene squared velocity loss along the registered zero-atom path."""
    require(control.shape == treated.shape and control.dtype == treated.dtype
            and control.device == treated.device, "Control/treated tensor mismatch")
    start, end = encode_expression(control), encode_expression(treated)
    position = (1-clock)*start + clock*end
    estimate = model(position, features, context, clock)
    loss = (estimate.float() - (end-start).float()).square().mean()
    require(torch.isfinite(loss).item(), "Nonfinite flow matching loss")
    return loss


@torch.inference_mode()
def sample_heun(model, control, features, context, *, steps=16):
    require(type(steps) is int and steps > 0, "Positive integer Heun steps required")
    coordinate = encode_expression(control)
    was_training = model.training
    model.eval()
    try:
        dt = 1.0/steps
        for i in range(steps):
            clock = control.new_full((len(control), 1), i/steps)
            first = model(coordinate, features, context, clock)
            later = control.new_full((len(control), 1), (i+1)/steps)
            second = model(coordinate + dt*first, features, context, later)
            coordinate = coordinate + 0.5*dt*(first+second)
            require(torch.isfinite(coordinate).all().item(), "Nonfinite flow rollout")
        return decode_expression(coordinate)
    finally:
        model.train(was_training)


def arm_features(data, features, *, arm, seed):
    """Separate fit/tune derangements; no learned transforms or held-feature fit."""
    values = np.asarray(features)
    require(values.ndim == 2 and values.shape[0] == len(data.target_ids) and values.shape[1] > 0
            and values.dtype.kind == "f" and np.isfinite(values).all(), "Invalid target feature matrix")
    require(arm in ("true", "shuffled", "constant"), "Unknown target arm")
    values = values.astype(np.float32, copy=True)
    require(np.isfinite(values).all(), "Target features overflow float32")
    lookup = {target: i for i, target in enumerate(data.target_ids)}
    mapping = {target: target for target in data.target_ids}
    if arm == "constant":
        fit = [lookup[t] for t in data.fit_targets]
        values[:] = values[fit].astype(np.float64).mean(axis=0).astype(np.float32)
        mapping = {target: "fit_target_feature_mean" for target in data.target_ids}
    elif arm == "shuffled":
        rng = np.random.default_rng(seed)
        original = values.copy()
        for targets in (data.fit_targets, data.tune_targets):
            order = rng.permutation(np.asarray([lookup[t] for t in sorted(targets)]))
            shifted = np.roll(order, 1)
            values[order] = original[shifted]
            mapping.update({data.target_ids[int(i)]: data.target_ids[int(j)] for i, j in zip(order, shifted)})
    return values, mapping


def sample_training_batch(data, features, rng, batch_size):
    """Uniform target, then its group, then independent treated/fit16 control."""
    require(type(batch_size) is int and batch_size > 0, "Invalid batch size")
    lookup = {target: i for i, target in enumerate(data.target_ids)}
    targets = np.asarray([lookup[t] for t in sorted(data.fit_targets)], dtype=np.int64)
    selected = rng.choice(targets, size=batch_size, replace=True)
    group_lookup = {int(t): np.flatnonzero(data.group_target == t) for t in targets}
    groups = np.asarray([rng.choice(group_lookup[int(t)]) for t in selected], dtype=np.int64)
    pools = data.group_pool[groups]
    treated = np.asarray([rng.choice(data.treated_indices[int(g)]) for g in groups])
    control = data.fit_reference_indices[pools, rng.integers(0, 16, size=batch_size)]
    clock = rng.random((batch_size, 1), dtype=np.float32)
    return {"control": data.expression[control], "treated": data.expression[treated],
            "features": features[selected], "context": data.context[pools], "clock": clock,
            "control_indices": control, "treated_indices": treated, "target_indices": selected}


def _distance(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    # Arithmetic roundoff can make a true squared distance marginally negative.
    return np.maximum(np.sum(a*a, axis=1)[:, None] + np.sum(b*b, axis=1)[None, :] - 2*a@b.T, 0.)


def _mmd(a, b, bandwidth):
    return float(np.exp(-_distance(a, a)/bandwidth).mean()
                 + np.exp(-_distance(b, b)/bandwidth).mean()
                 - 2*np.exp(-_distance(a, b)/bandwidth).mean())


def population_metrics(predicted, control, treated):
    """Unpaired diagnostics; RBF scale uses only matched context32 controls."""
    p, c, t = [np.asarray(a, dtype=np.float64) for a in (predicted, control, treated)]
    require(p.ndim == c.ndim == t.ndim == 2 and p.shape == c.shape and p.shape[1] == t.shape[1]
            and min(len(p), len(t)) >= 2 and all(np.isfinite(a).all() for a in (p, c, t))
            and all(np.all(a >= 0) for a in (p, c, t)), "Invalid population metric inputs")
    d = _distance(c, c)
    distances = d[np.triu_indices(len(c), 1)]
    positive = distances[distances > 0]
    bandwidth = max(float(np.median(positive)) if len(positive) else 1.0, 1e-12)
    pc, cc, tc = [a.mean(0) for a in (p, c, t)]
    pv, cv, tv = [a.var(0).mean() for a in (p, c, t)]
    pz, cz, tz = [float(np.mean(a == 0)) for a in (p, c, t)]
    p_mmd, c_mmd = _mmd(p, t, bandwidth), _mmd(c, t, bandwidth)
    require(min(p_mmd, c_mmd) >= -1e-12, "MMD outside numeric tolerance")
    return {"model_centroid_mse": float(np.mean((pc-tc)**2)),
            "control_centroid_mse": float(np.mean((cc-tc)**2)),
            "model_mmd2": max(p_mmd, 0.), "control_mmd2": max(c_mmd, 0.),
            "model_variance": float(pv), "control_variance": float(cv), "treated_variance": float(tv),
            "variance_ratio_to_control": float(pv/(cv+1e-12)),
            "variance_ratio_to_treated": float(pv/(tv+1e-12)),
            "model_zero_fraction": pz, "control_zero_fraction": cz, "treated_zero_fraction": tz,
            "model_zero_fraction_error": abs(pz-tz), "control_zero_fraction_error": abs(cz-tz),
            "negative_fraction": float(np.mean(p < 0)),
            "changed_zero_fraction": float(np.mean((c == 0) & (p > 0))),
            "new_zero_fraction": float(np.mean((c > 0) & (p == 0))),
            "max_prediction": float(p.max()), "control_to_prediction_mse": float(np.mean((p-c)**2))}


def evaluate_source(model, data, features, *, steps=16, batch_size=256, device=None):
    """Auxiliary-target held-out diagnostic, NOT the cross-source/panel gate.

    context32, never fit_reference16, supplies the evaluation control population.
    Same controls are the model base and comparator. Metrics average groups
    within target, then targets equally; repeated donors are not independent.
    ``model=None`` computes the exact control baseline without model integration.
    """
    require(type(batch_size) is int and batch_size > 0, "Invalid evaluation batch size")
    require(np.asarray(features).shape[0] == len(data.target_ids), "Feature target axis differs")
    if device is None:
        device = next(model.parameters()).device if model is not None else "cpu"
    wanted = {data.target_ids.index(t) for t in data.tune_targets}
    groups = [i for i, target in enumerate(data.group_target) if int(target) in wanted]
    results = []
    # Work a bounded number of groups at once; pool donors are not copied globally.
    groups_per_chunk = max(1, batch_size//32)
    for start in range(0, len(groups), groups_per_chunk):
        active = groups[start:start+groups_per_chunk]
        controls = np.concatenate([data.expression[data.context_indices[data.group_pool[g]]] for g in active])
        feature_rows = np.repeat(features[data.group_target[active]], 32, axis=0)
        context_rows = np.repeat(data.context[data.group_pool[active]], 32, axis=0)
        if model is None:
            predictions = controls.copy()
        else:
            tensors = [torch.as_tensor(a, dtype=torch.float32, device=device)
                       for a in (controls, feature_rows, context_rows)]
            predictions = sample_heun(model, *tensors, steps=steps).cpu().numpy()
        for j, group in enumerate(active):
            sliced = slice(j*32, (j+1)*32)
            metrics = population_metrics(predictions[sliced], controls[sliced],
                                         data.expression[data.treated_indices[group]])
            results.append({"group_id": data.group_ids[group],
                            "target": data.target_ids[int(data.group_target[group])], **metrics})
    require(len(results) > 0, "No tuning groups")
    keys = tuple(k for k in results[0] if k not in ("group_id", "target"))
    by_target = []
    for target in sorted(data.tune_targets):
        rows = [row for row in results if row["target"] == target]
        by_target.append({"target": target, "groups": len(rows),
            **{key: float(np.mean([row[key] for row in rows])) for key in keys}})
    metrics = {key: float(np.mean([row[key] for row in by_target])) for key in keys}
    return {"schema": SCHEMA, "source": data.source, "scope": "source_local_auxiliary_target_holdout",
        "evaluation_base": "context32_disjoint_from_fit_reference16",
        "aggregation": "equal_group_within_target_then_equal_target",
        "targets": len(by_target), "groups": len(results), "metrics": metrics,
        "per_target": by_target, "per_group": results, "submission_ready": False,
        "independent_cross_source_evaluation": False}


@dataclass
class TrainingResult:
    model: AuxiliaryConditionalFlow
    features: np.ndarray
    history: list[dict]
    evaluation: dict
    metadata: dict
    optimizer_state: dict


def train_source(data, features, config, *, arm="true", device="cpu",
                 progress: Callable[[int, dict], None] | None = None):
    """Fixed-step AdamW pretraining with every optimizer step reported to caller.

    All arms share model initialization and independent identical sampling/time
    RNG streams. The shuffle RNG never changes expression draws. Tune responses
    are read only for initial/final diagnostics; they never select a checkpoint,
    train transforms, or terminate optimization. No automatic next-stage launch.
    """
    require(isinstance(data, SourceData) and isinstance(config, PilotConfig), "Prepared source/config required")
    require(np.asarray(features).ndim == 2 and config.gene_dim == data.expression.shape[1]
            and config.target_dim == np.asarray(features).shape[1],
            "Configured dimensions differ")
    conditioned, mapping = arm_features(data, features, arm=arm, seed=config.seed)
    device = torch.device(device)
    require(device.type in ("cpu", "cuda"), "Only explicit CPU or CUDA devices supported")
    if device.type == "cuda":
        require(device.index is not None and torch.cuda.is_available(), "Explicit available CUDA index required")
    devices = [] if device.type == "cpu" else [device.index]
    started = time.monotonic()
    baseline = evaluate_source(None, data, conditioned, batch_size=config.batch_size)
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(config.seed)
        model = AuxiliaryConditionalFlow(config).to(device=device, dtype=torch.float32)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        rng = np.random.default_rng(config.seed)
        history, row_fingerprint = [], hashlib.sha256()
        model.train()
        for step in range(1, config.steps+1):
            require(time.monotonic()-started <= config.max_seconds, "Flow pilot wall-clock limit exceeded")
            batch = sample_training_batch(data, conditioned, rng, config.batch_size)
            for name in ("control_indices", "treated_indices", "target_indices", "clock"):
                row_fingerprint.update(np.ascontiguousarray(batch[name]).tobytes())
            tensors = [torch.as_tensor(batch[name], device=device, dtype=torch.float32)
                       for name in ("control", "treated", "features", "context", "clock")]
            optimizer.zero_grad(set_to_none=True)
            loss = matching_loss(model, *tensors)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            require(all(torch.isfinite(p).all().item() for p in model.parameters()), "Nonfinite model parameter")
            values = {"flow_matching_loss": float(loss.detach().cpu()), "gradient_norm": float(norm.detach().cpu()),
                "learning_rate": float(optimizer.param_groups[0]["lr"]), "examples_seen": float(step*config.batch_size)}
            history.append({"step": step, **values})
            if progress is not None:
                progress(step, values.copy())
        require(time.monotonic()-started <= config.max_seconds, "Flow pilot wall-clock limit exceeded")
        final = evaluate_source(model, data, conditioned, steps=config.heun_steps,
                                batch_size=config.batch_size, device=device)
        require(time.monotonic()-started <= config.max_seconds, "Flow pilot wall-clock limit exceeded")
    metadata = {"schema": SCHEMA, "source": data.source, "arm": arm, "config": asdict(config),
        "fit_targets": list(data.fit_targets), "tune_targets": list(data.tune_targets),
        "fit_targets_sha256": _digest(list(data.fit_targets)), "tune_targets_sha256": _digest(list(data.tune_targets)),
        "excluded_targets_sha256": _digest(list(data.excluded_targets)),
        "feature_mapping": mapping, "feature_matrix_sha256": hashlib.sha256(conditioned.tobytes()).hexdigest(),
        "sampling_sha256": row_fingerprint.hexdigest(), "training_base": "fit_reference16_only",
        "evaluation_base": "context32_disjoint_from_fit_reference16", "target_balanced_sampling": True,
        "coordinate": "zero_atom=-0.25; positive=sqrt(log1p_CP10000)", "decoder": "ReLU(z)^2_at_all_rollouts",
        "exact_initial_cell_as_extra_condition": False, "checkpoint_selection": "fixed_final_step_only",
        "optimizer_steps": config.steps, "parameter_count": sum(p.numel() for p in model.parameters()),
        "elapsed_seconds": time.monotonic()-started, "initial_control_baseline": baseline,
        "held_source_treated_used": False, "cross_source_gate_passed": False, "submission_ready": False}
    return TrainingResult(model, conditioned, history, final, metadata, optimizer.state_dict())
