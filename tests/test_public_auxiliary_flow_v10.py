"""Synthetic CPU-only actual optimizer tests; no public files, GPU, or network."""
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import public_auxiliary_flow_v10 as flow


@pytest.fixture
def arrays():
    rng = np.random.default_rng(51)
    # Eight targets, two source batches each; only four cells per gene dimension.
    n_targets, pools, genes, treated = 8, 2, 5, 8
    n_groups = n_targets*pools
    n_treated = n_groups*treated
    n = n_treated + pools*48
    expression = rng.uniform(0, 2, (n, genes)).astype(np.float32)
    expression[rng.random(expression.shape) < .4] = 0
    reference = np.arange(n_treated, n_treated+32).reshape(2, 16)
    context_rows = np.arange(n_treated+32, n).reshape(2, 32)
    observed = expression[context_rows].astype(np.float64)
    context = np.concatenate((observed.mean(1), observed.std(1)), axis=1).astype(np.float32)
    return {"source": np.asarray("replogle_k562"), "expression": expression,
        "original_rows": np.arange(n, dtype=np.int64)+1000,
        "row_role": np.concatenate((np.ones(n_treated, dtype=np.uint8),
            np.full(32, 2, dtype=np.uint8), np.full(64, 3, dtype=np.uint8))),
        "genes": np.asarray([f"g{i}" for i in range(genes)]),
        "group_ids": np.asarray([f"group{i}" for i in range(n_groups)]),
        "targets": np.asarray([f"t{i}" for i in range(n_targets) for _ in range(pools)]),
        "batches": np.asarray([f"b{i}" for _ in range(n_targets) for i in range(pools)]),
        "pool_ids": np.asarray(["pool0", "pool1"]), "pool_batches": np.asarray(["b0", "b1"]),
        "group_pool": np.tile(np.arange(2), n_targets),
        "treated_offsets": np.arange(n_groups+1)*treated,
        "treated_indices": np.arange(n_treated), "fit_reference_indices": reference,
        "context_indices": context_rows, "context": context}


def prepare(arrays, **overrides):
    roles = {"fit_targets": [f"t{i}" for i in range(6)], "tune_targets": ["t6", "t7"],
             "excluded_targets": ["protected", "panel"]}
    return flow.prepare_source(arrays, **(roles | overrides))


@pytest.fixture
def data(arrays):
    return prepare(arrays)


@pytest.fixture
def features():
    # GO features + availability flags; absent GenePT stays explicitly zero.
    a = np.random.default_rng(3).random((8, 3)).astype(np.float32)
    return np.column_stack((a, np.ones(8, dtype=np.float32), np.zeros(8, dtype=np.float32)))


def tiny_config(**kwargs):
    return flow.PilotConfig(5, 5, hidden_dim=12, layers=1, steps=4,
        batch_size=16, heun_steps=2, learning_rate=.01, **kwargs)


def test_real_cache_interface_and_role_partition(data):
    assert data.source == "replogle_k562"
    assert len(data.target_ids) == 8
    assert data.expression.shape == (224, 5)
    assert set(np.concatenate(data.treated_indices)).isdisjoint(data.context_indices.reshape(-1))
    assert set(data.fit_reference_indices.reshape(-1)).isdisjoint(data.context_indices.reshape(-1))


@pytest.mark.parametrize("mutation,match", [
    (lambda a: a.update(source=np.asarray("nadig_jurkat+replogle_k562")), "source"),
    (lambda a: a["expression"].__setitem__((0, 0), -1), "expression"),
    (lambda a: a["expression"].__setitem__((0, 0), np.nan), "expression"),
    (lambda a: a.update(expression=a["expression"].astype(np.float64)), "expression"),
    (lambda a: a["original_rows"].__setitem__(1, a["original_rows"][0]), "rows"),
    (lambda a: a["row_role"].__setitem__(0, 3), "treated"),
    (lambda a: a["row_role"].__setitem__(0, 4), "role"),
    (lambda a: a["genes"].__setitem__(0, a["genes"][1]), "genes"),
    (lambda a: a["group_ids"].__setitem__(0, a["group_ids"][1]), "group IDs"),
    (lambda a: a["group_pool"].__setitem__(0, 9), "pool"),
    (lambda a: a["batches"].__setitem__(0, "bad"), "batch"),
    (lambda a: a["treated_offsets"].__setitem__(1, 7), "offsets"),
    (lambda a: a["treated_indices"].__setitem__(1, a["treated_indices"][0]), "Repeated"),
    (lambda a: a["fit_reference_indices"].__setitem__((0, 0), a["context_indices"][0, 0]), "fit reference"),
    (lambda a: a["context_indices"].__setitem__((0, 0), a["context_indices"][0, 1]), "context"),
    (lambda a: a["context"].__setitem__((0, 0), 9), "Context differs"),
])
def test_corrupt_arrays_rejected(arrays, mutation, match):
    mutation(arrays)
    with pytest.raises(ValueError, match=match):
        prepare(arrays)


@pytest.mark.parametrize("roles,match", [
    ({"excluded_targets": ["t0"]}, "Protected"),
    ({"excluded_targets": []}, "excluded"),
    ({"fit_targets": ["t0"]}, "partition"),
    ({"tune_targets": ["t0", "t7"]}, "partition"),
    ({"fit_targets": [f"t{i}" for i in range(5)]}, "partition"),
    ({"fit_targets": ["t0", "t0"]}, "fit targets"),
])
def test_target_boundary_rejected(arrays, roles, match):
    with pytest.raises(ValueError, match=match):
        prepare(arrays, **roles)


def test_context_only_and_dedup_reconstruction_proves_controls(data, arrays):
    arrays["expression"][:128] += 100
    again = prepare(arrays)
    assert np.array_equal(again.context, data.context)
    arrays["expression"][arrays["context_indices"][0, 0], 0] += 1
    with pytest.raises(ValueError, match="Context differs"):
        prepare(arrays)


def test_encoding_and_shared_decoder_preserve_zeros_and_nonnegative():
    x = torch.tensor([[0., .01, .25, 1., 4.]])
    coordinate = flow.encode_expression(x)
    assert torch.equal(coordinate, torch.tensor([[-.25, .1, .5, 1., 2.]]))
    torch.testing.assert_close(flow.decode_expression(coordinate), x)
    assert torch.equal(flow.decode_expression(torch.tensor([[-100., -.25, 0., 1.]])),
                       torch.tensor([[0., 0., 0., 1.]]))


class ConstantVelocity(torch.nn.Module):
    def __init__(self, amount):
        super().__init__()
        self.amount = amount

    def forward(self, coordinate, features, context, clock):
        return torch.full_like(coordinate, self.amount)


def test_zero_atom_resists_small_motion_and_allows_learned_activation():
    x = torch.tensor([[0., .01, 1.]])
    f, c = torch.zeros(1, 1), torch.zeros(1, 6)
    slow = flow.sample_heun(ConstantVelocity(.1), x, f, c, steps=4)
    assert slow[0, 0].item() == 0
    active = flow.sample_heun(ConstantVelocity(.5), x, f, c, steps=4)
    assert active[0, 0].item() == pytest.approx(.25**2)
    off = flow.sample_heun(ConstantVelocity(-.5), x, f, c, steps=4)
    assert off[0, 1].item() == 0
    assert (off >= 0).all()


def test_zero_head_control_identity_and_mode_restore(data, features):
    model = flow.AuxiliaryConditionalFlow(tiny_config())
    control = torch.as_tensor(data.expression[data.context_indices[0]])
    f = torch.as_tensor(np.repeat(features[0:1], 32, axis=0))
    c = torch.as_tensor(np.repeat(data.context[0:1], 32, axis=0))
    predicted = flow.sample_heun(model, control, f, c, steps=2)
    torch.testing.assert_close(predicted, control, rtol=1e-6, atol=1e-7)
    assert torch.equal(predicted == 0, control == 0)
    assert model.training
    model.eval()
    flow.sample_heun(model, control, f, c, steps=1)
    assert not model.training


def test_target_context_and_state_have_computational_gradient_paths():
    torch.manual_seed(9)
    model = flow.AuxiliaryConditionalFlow(tiny_config())
    torch.nn.init.normal_(model.output.weight, std=.1)
    x, f, c = [torch.rand(4, n, requires_grad=True) for n in (5, 5, 10)]
    model(x, f, c, torch.ones(4, 1)*.5).square().sum().backward()
    for a in (x, f, c):
        assert torch.isfinite(a.grad).all() and a.grad.abs().sum() > 0


def test_flow_loss_exact_zero_atom_path_and_real_optimizer_gradient():
    model = flow.AuxiliaryConditionalFlow(tiny_config())
    x0 = torch.zeros(4, 5)
    x1 = torch.ones(4, 5)
    loss = flow.matching_loss(model, x0, x1, torch.ones(4, 5), torch.ones(4, 10), torch.ones(4, 1)*.5)
    assert loss.item() == pytest.approx(1.25**2)
    loss.backward()
    assert model.output.weight.grad.abs().sum() > 0


def test_sampling_balances_only_fit_targets_and_training_reference_rows(data, features):
    batch = flow.sample_training_batch(data, features, np.random.default_rng(62), 12000)
    assert set(batch["target_indices"]) == set(range(6))
    assert set(batch["control_indices"]).issubset(data.fit_reference_indices.reshape(-1))
    assert set(batch["control_indices"]).isdisjoint(data.context_indices.reshape(-1))
    assert set(batch["treated_indices"]).issubset(set(np.concatenate(data.treated_indices[:12])))
    counts = np.bincount(batch["target_indices"])
    assert counts.max()-counts.min() < 300
    for i in range(20):
        p = next(p for p in range(2) if batch["control_indices"][i] in data.fit_reference_indices[p])
        np.testing.assert_array_equal(batch["context"][i], data.context[p])


def test_feature_arms_deterministic_no_cross_partition_and_mean_fit_only(data, features):
    original = features.copy()
    true, true_map = flow.arm_features(data, features, arm="true", seed=5)
    assert np.array_equal(true, original) and all(k == v for k, v in true_map.items())
    shuffled, mapping = flow.arm_features(data, features, arm="shuffled", seed=5)
    assert all(k != v for k, v in mapping.items())
    for target, donor in mapping.items():
        assert (target in data.fit_targets) == (donor in data.fit_targets)
    again, other_map = flow.arm_features(data, features, arm="shuffled", seed=5)
    assert np.array_equal(again, shuffled) and other_map == mapping
    constant, _ = flow.arm_features(data, features, arm="constant", seed=5)
    np.testing.assert_allclose(constant, np.broadcast_to(features[:6].astype(np.float64).mean(0), constant.shape))
    features[6:] = 1000
    held_changed, _ = flow.arm_features(data, features, arm="constant", seed=5)
    assert np.array_equal(held_changed, constant)


def test_metric_baseline_identity_and_collapse_detection():
    rng = np.random.default_rng(99)
    c = rng.uniform(0, 3, (32, 5))
    c[rng.random(c.shape) < .2] = 0
    t = c[:16].copy()
    baseline = flow.population_metrics(c, c, t)
    assert baseline["model_mmd2"] == baseline["control_mmd2"]
    assert baseline["model_centroid_mse"] == baseline["control_centroid_mse"]
    assert baseline["control_to_prediction_mse"] == 0
    collapsed = np.broadcast_to(c.mean(0), c.shape)
    metrics = flow.population_metrics(collapsed, c, t)
    assert metrics["variance_ratio_to_control"] < 1e-20
    assert metrics["model_zero_fraction"] == 0


def test_evaluator_uses_context32_not_training_reference(data, features, monkeypatch):
    seen = []
    def record(model, control, feature, context, *, steps):
        seen.extend(control.cpu().numpy())
        return control
    monkeypatch.setattr(flow, "sample_heun", record)
    model = flow.AuxiliaryConditionalFlow(tiny_config())
    result = flow.evaluate_source(model, data, features, steps=1, batch_size=64)
    expected = np.concatenate([data.expression[data.context_indices[p]] for p in (0, 1, 0, 1)])
    np.testing.assert_array_equal(seen, expected)
    assert result["targets"] == 2 and result["groups"] == 4
    assert len(result["per_target"]) == 2 and len(result["per_group"]) == 4
    assert not result["submission_ready"] and not result["independent_cross_source_evaluation"]
    assert result["metrics"]["model_centroid_mse"] == result["metrics"]["control_centroid_mse"]


def test_actual_optimizer_steps_loss_callbacks_and_optimizer_lineage(data, features):
    events = []
    result = flow.train_source(data, features, tiny_config(), progress=lambda step, values: events.append((step, values)))
    assert [v[0] for v in events] == [1, 2, 3, 4]
    assert len(result.history) == 4 and result.metadata["optimizer_steps"] == 4
    assert result.metadata["checkpoint_selection"] == "fixed_final_step_only"
    assert result.model.output.weight.abs().sum() > 0
    assert result.optimizer_state["state"]
    assert all(float(v["step"]) == 4 for v in result.optimizer_state["state"].values())
    assert all(torch.isfinite(value).all() for state in result.optimizer_state["state"].values()
               for value in state.values() if isinstance(value, torch.Tensor))
    assert all(np.isfinite(v[1]["flow_matching_loss"]) for v in events)
    assert result.evaluation["metrics"]["negative_fraction"] == 0


def test_same_rng_across_arms_and_weights_change_with_target_conditioning(data, features):
    results = [flow.train_source(data, features, tiny_config(), arm=arm) for arm in ("true", "shuffled", "constant")]
    assert len({r.metadata["sampling_sha256"] for r in results}) == 1
    for other in results[1:]:
        assert not torch.equal(results[0].model.output.weight, other.model.output.weight)


def test_tuning_responses_cannot_change_training_weights(arrays, features):
    first = flow.train_source(prepare(arrays), features, tiny_config())
    arrays["expression"][96:128] += 20
    second = flow.train_source(prepare(arrays), features, tiny_config())
    for name, value in first.model.state_dict().items():
        assert torch.equal(value, second.model.state_dict()[name])
    assert first.history == second.history
    assert first.evaluation["metrics"]["model_centroid_mse"] != second.evaluation["metrics"]["model_centroid_mse"]


def test_saved_numeric_model_can_replay_without_refitting(data, features):
    trained = flow.train_source(data, features, tiny_config())
    restored = flow.AuxiliaryConditionalFlow(tiny_config())
    numeric = {k: v.detach().cpu().numpy().copy() for k, v in trained.model.state_dict().items()}
    restored.load_state_dict({k: torch.from_numpy(v) for k, v in numeric.items()}, strict=True)
    replay = flow.evaluate_source(restored, data, trained.features, steps=2, batch_size=16)
    assert replay == trained.evaluation


def test_training_restores_global_torch_rng_state(data, features):
    torch.manual_seed(329)
    before = torch.random.get_rng_state().clone()
    flow.train_source(data, features, tiny_config())
    assert torch.equal(torch.random.get_rng_state(), before)


def test_callback_failure_stops_training(data, features):
    seen = []
    def fail(step, values):
        seen.append(step)
        raise RuntimeError("durable journal failed")
    with pytest.raises(RuntimeError, match="journal failed"):
        flow.train_source(data, features, tiny_config(), progress=fail)
    assert seen == [1]


def test_wall_limit_checked(data, features):
    with pytest.raises(ValueError, match="wall-clock"):
        flow.train_source(data, features, tiny_config(max_seconds=1e-12))


@pytest.mark.parametrize("field,value", [("steps", 0), ("batch_size", True), ("gene_dim", -1),
    ("hidden_dim", 2.5), ("learning_rate", float("nan")), ("weight_decay", -1), ("grad_clip", 0),
    ("seed", -1), ("seed", True), ("max_seconds", float("inf"))])
def test_bad_configuration_rejected(field, value):
    with pytest.raises(ValueError):
        replace(tiny_config(), **{field: value})


@pytest.mark.parametrize("features_kind", ["nan", "integer", "wrong_axis"])
def test_bad_features_rejected(data, features, features_kind):
    if features_kind == "nan":
        features[0, 0] = np.nan
    elif features_kind == "integer":
        features = features.astype(np.int64)
    else:
        features = features[:7]
    with pytest.raises(ValueError, match="feature"):
        flow.arm_features(data, features, arm="true", seed=3)


def test_no_pickle_io_network_or_automatic_submission_api():
    source = Path(flow.__file__).read_text()
    for token in ("torch.load(", "torch.save(", "requests.", "subprocess.", "wandb.init(", "open("):
        assert token not in source
