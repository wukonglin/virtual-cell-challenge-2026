"""Synthetic-only tests: real pilot cells are never opened here."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import public_mean_effect_baseline as b
from public_flow_conditioning import FeatureTable


def table():
    return FeatureTable("go", ("A", "B", "C"), ("g1", "g2", "g3"),
        np.array([[1., 0, 0], [0, 1, 1], [1, 1, 0]]), "synthetic", "a" * 64)


def split(targets=("A", "B", "A", "B"), sources=("K", "K", "J", "J")):
    controls, treated, labels, context = [], [], [], []
    for i, target in enumerate(targets):
        control = np.array([[1 + j / 20, 2 + j / 30] for j in range(8)], dtype=np.float32)
        control += i / 10
        delta = np.array([.2, -.1]) if target == "A" else np.array([-.05, .3])
        controls.append(control)
        treated.append((control + delta).astype(np.float32))
        labels.extend([i] * 8)
        context.append(np.r_[control.astype(np.float64).mean(0), control.astype(np.float64).std(0)])
    return SimpleNamespace(targets=targets, sources=sources, control=np.concatenate(controls),
        control_group=np.array(labels), treated=np.concatenate(treated), group=np.array(labels),
        context=np.array(context, dtype=np.float32))


def cache():
    return SimpleNamespace(train=split(), dev=split(("A", "B"), ("H", "H")),
        genes=("g1", "g2"), contract_sha256="b" * 64, source_manifest_sha256="c" * 64)


def test_balancing_not_cell_or_group_frequency():
    weights = b.balanced_weights(("A", "A", "B", "A"), ("K", "K", "K", "J"))
    np.testing.assert_allclose(weights, [.125, .125, .25, .5])


@pytest.mark.parametrize("targets,sources", [((), ()), (("A",), ()), (("A", "B"), ("K",))])
def test_balancing_refuses_misalignment(targets, sources):
    with pytest.raises(ValueError):
        b.balanced_weights(targets, sources)


def test_group_delta_uses_exact_matched_controls():
    means = b.group_means(split())
    np.testing.assert_allclose(means["delta"], [[.2, -.1], [-.05, .3], [.2, -.1], [-.05, .3]], atol=2e-7)


def test_transfer_sourcebalanced():
    train = {"targets": ("A", "A", "A", "B"), "sources": ("K", "K", "J", "J"),
             "delta": np.array([[1., 2], [3, 4], [10, 20], [99, 98]])}
    np.testing.assert_allclose(b.transfer_delta(train, ("A", "B")), [[6, 11.5], [99, 98]])
    with pytest.raises(ValueError, match="absent"):
        b.transfer_delta(train, ("missing",))


def test_ridge_penalizes_intercept_zero_prior():
    beta = b.ridge_fit(np.ones((4, 1)), np.ones((4, 2)), np.ones(4) / 4)
    np.testing.assert_allclose(beta, [[1 / 11, 1 / 11]])


def test_ridge_dual_matches_primal():
    rng = np.random.default_rng(7)
    x, y = rng.normal(size=(7, 10)), rng.normal(size=(7, 3))
    w = np.arange(1., 8.) / 28
    expected = np.linalg.solve(x.T @ (w[:, None] * x) + 10 * np.eye(10), x.T @ (w[:, None] * y))
    np.testing.assert_allclose(b.ridge_fit(x, y, w), expected, atol=1e-12)


@pytest.mark.parametrize("weights", [np.array([1., 0]), np.array([.5, np.nan]), np.array([.3, .3]), np.ones(3) / 3])
def test_reject_invalid_ridge_weights(weights):
    with pytest.raises(ValueError):
        b.ridge_fit(np.ones((2, 1)), np.ones((2, 2)), weights)


@pytest.mark.parametrize("penalty", [0, -1, np.nan, np.inf])
def test_reject_invalid_penalty(penalty):
    with pytest.raises(ValueError):
        b.ridge_fit(np.ones((2, 1)), np.ones((2, 2)), np.array([.5, .5]), penalty)


def test_context_floor_no_recipient_fitting():
    mean, scale = b.weighted_standardize(np.array([[1., 3], [1, 5]]), np.array([.5, .5]))
    np.testing.assert_allclose(mean, [1, 4])
    np.testing.assert_allclose(scale, [.1, 1])


@pytest.mark.parametrize("arm", b.ARMS)
def test_fit_receives_no_development_treated_and_is_deterministic(arm):
    data = cache()
    train = b.group_means(data.train)
    pred, meta, arrays = b.fit_predict(train, data.dev.targets, data.dev.context, table(), arm, "fold")
    data.dev.treated[:] = np.nan
    again, _, second = b.fit_predict(train, data.dev.targets, data.dev.context, table(), arm, "fold")
    np.testing.assert_array_equal(pred, again)
    assert np.isfinite(pred).all()
    for key in arrays:
        np.testing.assert_array_equal(arrays[key], second[key])


def test_target_transform_fold_local_excludes_unseen_annotation():
    a, _, transform, aligned = b.target_design(table(), ("A", "B"), ("A",), "true", "fold")
    modified = FeatureTable("go", ("A", "B", "C"), ("g1", "g2", "g3"),
        np.array([[1., 0, 0], [0, 1, 1], [1e6, 1e6, 1e6]]), "synthetic", "a" * 64)
    aa, _, changed, _ = b.target_design(modified, ("A", "B"), ("A",), "true", "fold")
    np.testing.assert_array_equal(a, aa)
    np.testing.assert_array_equal(transform.mean, changed.mean)
    assert aligned.target_ids == ("A", "B")


@pytest.mark.parametrize("arm", ["true", "constant", "shuffled"])
def test_masks_preserved(arm):
    _, _, _, aligned = b.target_design(table(), ("A", "B", "MISSING"), ("A",), arm, "fold")
    np.testing.assert_array_equal(aligned.present[:, 0], [True, True, False])
    np.testing.assert_array_equal(aligned.values[2], [0, 0, 0])


@pytest.mark.parametrize("arm", b.ARMS)
def test_unknown_recipient_targets_refused(arm):
    with pytest.raises(ValueError, match="admission"):
        b.fit_predict(b.group_means(split()), ("NEW",), np.ones((1, 4)), table(), arm, "fold")


def test_fold_held_context_not_in_fit_and_admission_explicit():
    data = split(("A", "B", "A", "C"), ("K", "K", "J", "J"))
    folds = b.inner_validation(b.group_means(data), table(), "true", "fold")
    assert [f["held_source"] for f in folds] == ["J", "K"]
    for fold in folds:
        assert set(fold["fit_groups"]).isdisjoint(fold["admitted_groups"])
        assert len(fold["admitted_groups"]) == len(fold["excluded_missing_training_target_groups"]) == 1
        assert fold["used_for_hyperparameter_or_arm_selection"] is False
        fitted_targets = fold["fit"]["target_ids"]
        assert "A" in fitted_targets
        assert ("C" not in fitted_targets) if fold["held_source"] == "J" else ("B" not in fitted_targets)


def test_fold_no_overlap_is_reported_not_imputed():
    data = split(("A", "B"), ("K", "J"))
    folds = b.inner_validation(b.group_means(data), table(), "true", "fold")
    assert all(f["status"] == "no_overlapping_targets" and f["metrics"] is None for f in folds)


def test_evaluation_unclipped_negative_outputs_and_same_samples():
    data = split(("A",), ("H",))
    _, baseline, arrays = b.evaluate(data, np.zeros((1, 2)))
    _, shifted, other = b.evaluate(data, np.full((1, 2), -5.))
    assert baseline["negative_fraction"] == 0
    assert shifted["negative_fraction"] == 1
    assert (other["prediction"] < 0).all()
    np.testing.assert_array_equal(arrays["control_cache_rows"], other["control_cache_rows"])
    np.testing.assert_array_equal(arrays["treated_cache_rows"], other["treated_cache_rows"])
    assert baseline["model_centroid_mse"] == baseline["control_centroid_mse"]
    assert baseline["model_mmd2"] == baseline["control_mmd2"]


def test_mean_shift_retains_control_variance():
    data = split(("A",), ("H",))
    _, _, arrays = b.evaluate(data, np.array([[.3, -.1]]))
    np.testing.assert_allclose(arrays["prediction"].var(0), data.control.astype(np.float64).var(0), atol=1e-14)


def test_null_shift_cosine_is_not_invented():
    assert b.delta_diagnostics(np.zeros(2), np.ones(2))["shift_cosine"] is None


def test_source_balanced_aggregation():
    groups = [{"group": 0, "target": "A", "source": "K", "m": 1},
              {"group": 1, "target": "A", "source": "K", "m": 3},
              {"group": 2, "target": "B", "source": "K", "m": 6},
              {"group": 3, "target": "A", "source": "J", "m": 10}]
    assert b.aggregate_groups(groups)["m"] == 7


@pytest.mark.parametrize("arm", b.ARMS)
def test_end_to_end_synthetic_new_outputs_and_safe_arrays(tmp_path, arm):
    out = tmp_path / arm
    summary = b.run_baseline(cache(), table(), arm, out)
    assert summary["development_used_for_fit_or_selection"] is False
    assert summary["full_axis_submission_ready"] is False
    assert summary["posttraining_performed"] is False
    for filename in ("weights.npz", "predictions.npz"):
        with np.load(out / filename, allow_pickle=False) as values:
            assert all(not values[key].dtype.hasobject for key in values)
    assert json.loads((out / "summary.json").read_text())["arm"] == arm
    with pytest.raises(FileExistsError):
        b.run_baseline(cache(), table(), arm, out)


def contract():
    names = ("public_mean_effect_baseline.py", "train_public_flow_pilot.py", "public_flow_conditioning.py", "build_go_target_features.py", "public_crispri_flow.py")
    return {"schema": b.SCHEMA, "policy": b.policy(), "inputs": {"a": "b"},
        "prior_development_exposure_acknowledged": True, "challenge_2026_treated_allowed": False,
        "code_sha256": {name: b.sha256_file(Path(b.__file__).with_name(name)) for name in names}}


def test_registration_is_required_and_authenticated(tmp_path):
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract()))
    b.validate_diagnostic_contract(path, b.sha256_file(path), {"a": "b"})
    with pytest.raises(ValueError, match="SHA"):
        b.validate_diagnostic_contract(path, "0" * 64, {"a": "b"})


@pytest.mark.parametrize("field,value", [("schema", "bad"), ("policy", {}), ("inputs", {}),
    ("prior_development_exposure_acknowledged", False), ("challenge_2026_treated_allowed", True), ("code_sha256", {})])
def test_registration_rejects_changed_scope(tmp_path, field, value):
    value_contract = contract()
    value_contract[field] = value
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(value_contract))
    with pytest.raises(ValueError):
        b.validate_diagnostic_contract(path, b.sha256_file(path), {"a": "b"})


def test_no_overwrite_safe_npz(tmp_path):
    path = tmp_path / "test.npz"
    b.save_arrays(path, {"a": np.ones(2)})
    with pytest.raises(FileExistsError):
        b.save_arrays(path, {"a": np.zeros(2)})
    with pytest.raises(ValueError, match="Object"):
        b.save_arrays(tmp_path / "bad.npz", {"a": np.array([{}], dtype=object)})


@pytest.mark.parametrize("hostname", ["login02.anvil.rcac.purdue.edu", "aida2", "unknown"])
def test_host_guard_refuses_fake_slurm_environment(monkeypatch, hostname):
    monkeypatch.setattr(b.socket, "gethostname", lambda: hostname)
    monkeypatch.setenv("SLURM_JOB_ID", "20539599")
    args = ["--arm", "true", "--output-dir", "/tmp/not-created"]
    for name in ("cache", "target-features", "feature-receipt", "contract", "source-manifest", "diagnostic-contract"):
        args += ["--" + name, "/tmp/not-read", "--" + name + "-sha256", "a" * 64]
    with pytest.raises(RuntimeError, match="compute host"):
        b.main(args)
