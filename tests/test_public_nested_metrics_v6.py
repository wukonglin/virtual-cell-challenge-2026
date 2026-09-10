"""Synthetic-only roundoff, evidence preservation and no-refit replay tests."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import public_nested_metrics_v6 as metrics
from train_public_flow_pilot import rbf_mmd2
from test_public_sparse_hurdle_v4 import fixture
from test_public_target_context_v5 import fit_first


def fold_report(values=(-1e-13, .2, .4)):
    groups = [{"source": "J", "target": target, "batch": str(i), "group": i,
               "model_mmd2": value, "control_mmd2": -1e-13,
               "model_centroid_mse": .5}
              for i, (target, value) in enumerate(zip(("A", "A", "B"), values))]
    targets = [{"source": "J", "target": target, "batches": n,
                "target_pooled_mse": .3, "control_target_pooled_mse": .4}
               for target, n in (("A", 2), ("B", 1))]
    return {"fold_id": "synthetic", "groups": groups, "targets_report": targets,
            "metrics": metrics.v3._aggregate(groups, targets), "exact_ntc_identity_verified": True}


def test_policy_fixed_tolerance_and_json_stability():
    policy = metrics.policy()
    assert policy == json.loads(json.dumps(policy, allow_nan=False))
    assert policy["negative_roundoff_tolerance"] == 1e-12
    assert policy["scientific_comparison_inequalities_changed"] is False


@pytest.mark.parametrize("value", [-1e-12, -1e-13, -float(np.finfo(float).eps), -np.nextafter(0., 1.)])
def test_negative_roundoff_only_becomes_exact_zero(value):
    assert metrics.canonical_mmd2(float(value)) == 0.0


@pytest.mark.parametrize("value", [0., -0., 1e-15, .1, 1., 2., 0, 2])
def test_nonnegative_values_preserved_exactly(value):
    assert metrics.canonical_mmd2(value) == value


@pytest.mark.parametrize("value", [float(np.nextafter(-1e-12, -np.inf)), -1e-11, -1.,
                                  float("nan"), float("inf"), -float("inf"),
                                  None, True, False, "0", [], {}, 10**1000])
def test_numerically_or_semantically_invalid_values_fail_closed(value):
    with pytest.raises(ValueError, match="roundoff tolerance"):
        metrics.canonical_mmd2(value)


def test_permuted_identical_empirical_distributions_roundoff_is_not_failure():
    rng = np.random.default_rng(20260927)
    observed = []
    for _ in range(100):
        cells = rng.uniform(0, 3, (8, 1))
        permutation = rng.permutation(8)
        raw = rbf_mmd2(cells, cells[permutation])
        observed.append(raw)
        assert abs(raw) <= metrics.MMD_ROUNDOFF_TOLERANCE
        assert metrics.canonical_mmd2(raw) >= 0
    assert min(observed) < 0  # The frozen biased estimator's actual roundoff corner.


def test_report_retains_raw_values_and_macro_recomputes_after_group_correction():
    original = fold_report(); before = copy.deepcopy(original)
    result = metrics.canonicalize_report(original)
    assert original == before
    assert result["raw_metrics"] == before["metrics"]
    assert result["metrics"] == metrics.v3._aggregate(result["groups"], result["targets_report"])
    assert result["metrics"]["model_mmd2"] == .25  # (mean A .1 + mean B .4)/2, not cell/batch micro.
    assert result["metrics"]["control_mmd2"] == 0
    assert result["metrics"]["raw_control_mmd2"] == -1e-13
    assert result["metrics"]["control_mmd2_roundoff_correction"] == 1e-13
    assert result["metrics"]["raw_model_mmd2"] == before["metrics"]["model_mmd2"]
    assert result["targets_report"] == before["targets_report"]
    assert result["exact_ntc_identity_verified"] is True
    assert result["mmd_roundoff_policy"] == metrics.policy()
    for row, old in zip(result["groups"], before["groups"]):
        for key in metrics.MMD_KEYS:
            assert row["raw_" + key] == old[key]
            assert row[key + "_roundoff_correction"] == row[key] - old[key]
    assert result == json.loads(json.dumps(result, allow_nan=False))


@pytest.mark.parametrize("key", metrics.MMD_KEYS)
@pytest.mark.parametrize("value", [-1.1e-12, float("nan"), None])
def test_bad_group_rejected_before_new_aggregation(monkeypatch, key, value):
    original = fold_report(); original["groups"][-1][key] = value
    def forbidden(*args, **kwargs):
        raise AssertionError("Aggregation must not run before validating every MMD")
    monkeypatch.setattr(metrics.v3, "_aggregate", forbidden)
    with pytest.raises(ValueError, match="roundoff tolerance"):
        metrics.canonicalize_report(original)


@pytest.mark.parametrize("field", ["raw_model_mmd2", "raw_control_mmd2",
                                 "model_mmd2_roundoff_correction", "control_mmd2_roundoff_correction"])
def test_existing_raw_evidence_cannot_be_overwritten(field):
    original = fold_report(); original["groups"][0][field] = 1.
    with pytest.raises(ValueError, match="overwrite"):
        metrics.canonicalize_report(original)


def test_second_wrapping_is_refused():
    result = metrics.canonicalize_report(fold_report())
    with pytest.raises(ValueError, match="already wrapped"):
        metrics.canonicalize_report(result)


@pytest.mark.parametrize("mutation", ["metrics", "groups", "targets_report", "row"])
def test_missing_schema_is_refused(mutation):
    original = fold_report()
    if mutation == "row": original["groups"][0] = None
    else: original[mutation] = []
    with pytest.raises(ValueError): metrics.canonicalize_report(original)


def test_wrapper_passes_frozen_outputs_through_without_alteration(monkeypatch):
    original, outputs = fold_report(), {"nonnegative": np.array([[0., .2]])}
    seen = []
    monkeypatch.setattr(metrics.sparse, "evaluate_fold", lambda *args: (seen.append(args) or original, outputs))
    args = tuple(object() for _ in range(5))
    report, returned = metrics.evaluate_fold(*args)
    assert returned is outputs and seen == [args]
    assert report == metrics.canonicalize_report(original)


def test_real_evaluator_synthetic_data_replays_without_fitting(monkeypatch):
    arrays, registration, tables = fixture()
    fold, aligned, prior, _, decoder, _ = fit_first(arrays, registration, tables)
    raw, original_outputs = metrics.sparse.evaluate_fold(arrays, fold, aligned, decoder, prior)
    def no_fit(*args, **kwargs): raise AssertionError("Frozen replay must not fit")
    monkeypatch.setattr(np.linalg, "solve", no_fit)
    before, outputs = metrics.evaluate_fold(arrays, fold, aligned, decoder, prior)
    after, replay = metrics.evaluate_fold(arrays, fold, aligned, decoder, prior)
    assert before == after == metrics.canonicalize_report(raw)
    assert before["raw_metrics"] == raw["metrics"]
    for key in outputs:
        np.testing.assert_array_equal(outputs[key], original_outputs[key])
        np.testing.assert_array_equal(outputs[key], replay[key])
