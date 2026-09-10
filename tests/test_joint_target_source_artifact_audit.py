"""Full-schema synthetic artifact replay and rehashed-tamper checks.

Only generated 56-target, 398-group, two-gene arrays are fitted. No real cache,
public GO files, W&B SDK, cloud connection or scheduler is used.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_public_joint_target_source_v3 as prep
import public_joint_target_source_v3 as model
import run_public_joint_target_source_v3 as runner
from public_flow_conditioning import FeatureTable


def synthetic_cohort():
    """31 four-batch and25 three-batch targets per source:199+199 groups."""
    target_ids = tuple(f"SYNTHETIC_{i:02}" for i in range(56))
    roster, controls, treated, control_group, treated_group, contexts = [], [], [], [], [], []
    for source_index, source in enumerate(prep.v2.SOURCE_IDS):
        for target_index, target in enumerate(target_ids):
            for batch in range(4 if target_index < 31 else 3):
                group = len(roster)
                roster.append({"group_id": group, "source": source, "target": target,
                               "batch": f"synthetic_batch_{batch}"})
                reference = np.array([[0.0, 0.4], [0.2, 0.5], [0.3, 0.6]], dtype=np.float32)
                reference[:, 1] += 0.2 * source_index + 0.02 * batch
                effect = np.array([0.002 * (target_index % 7), 0.003 * (target_index % 5)], dtype=np.float32)
                observed = reference[:2] + effect
                # Separate synthetic context values, never the response-reference
                # arithmetic; only feature shapes matter for this artifact audit.
                contexts.append(np.r_[reference.mean(0) + 0.01, reference.std(0) + 0.02])
                controls.append(reference)
                treated.append(observed)
                control_group.extend([group] * len(reference))
                treated_group.extend([group] * len(observed))
    arrays = {"targets": np.asarray([g["target"] for g in roster]),
        "sources": np.asarray([g["source"] for g in roster]),
        "batches": np.asarray([g["batch"] for g in roster]),
        "genes": np.asarray(["SYNTHETIC_GENE_A", "SYNTHETIC_GENE_B"]),
        "control": np.concatenate(controls), "treated": np.concatenate(treated),
        "control_group": np.asarray(control_group, dtype=np.int64),
        "group": np.asarray(treated_group, dtype=np.int64), "context": np.asarray(contexts)}
    registration = {"targets": list(target_ids), "group_roster": roster,
        "excluded_targets": [f"DENIED_{i:03}" for i in range(772)],
        "folds": prep.make_folds(target_ids, roster)}
    features = np.array([[float(i % 2 == 0), float(i % 3 == 0), 1.0] for i in range(56)])
    # Keep one target without supplied features to exercise missingness replay.
    table = FeatureTable("go", target_ids[:-1], ("SYNTHETIC_TERM_A", "SYNTHETIC_TERM_B", "SYNTHETIC_TERM_C"),
                         features[:-1], "synthetic-only", "a" * 64)
    tables = {fold["fold_id"]: table for fold in registration["folds"]}
    assert len(roster) == 398 and len(target_ids) == 56
    return arrays, registration, tables


def independent_control_baselines(arrays):
    """Direct matched differences; deliberately no model metric helper calls."""
    result = {}
    for source in sorted(set(arrays["sources"])):
        target_errors = []
        for target in sorted(set(arrays["targets"])):
            groups = np.flatnonzero((arrays["sources"] == source) & (arrays["targets"] == target))
            differences = []
            for group in groups:
                anchor = arrays["control"][arrays["control_group"] == group].astype(np.float64)
                observed = arrays["treated"][arrays["group"] == group].astype(np.float64)
                differences.append(anchor.mean(axis=0) - observed.mean(axis=0))
            pooled = np.mean(differences, axis=0)
            target_errors.append(float(np.dot(pooled, pooled) / len(pooled)))
        result[str(source)] = float(np.mean(target_errors))
    return result


@pytest.fixture(scope="module")
def saved_artifacts(tmp_path_factory):
    arrays, registration, tables = synthetic_cohort()
    root = tmp_path_factory.mktemp("synthetic_joint_artifacts")
    artifacts = {}
    for arm in ("control", "true"):
        path = root / arm
        summary = model.run_arm(arrays, registration, tables, arm, path,
                                {"diagnostic_contract_sha256": "b" * 64})
        artifacts[arm] = {"path": path, "summary": summary}
    return arrays, registration, artifacts


@pytest.fixture
def case(request, tmp_path, monkeypatch, saved_artifacts):
    arrays, registration, artifacts = saved_artifacts
    arm = getattr(request, "param", "true")
    saved = artifacts[arm]
    output = tmp_path / arm
    shutil.copytree(saved["path"], output)
    monkeypatch.setattr(runner, "CONTROL_POOLED", independent_control_baselines(arrays))
    return output, copy.deepcopy(saved["summary"]), arrays, registration


def replace_npz(case, filename, mutate):
    output, summary, _, _ = case
    path = output / filename
    before = runner.sha256_file(path)
    with np.load(path, allow_pickle=False) as source:
        values = {key: source[key].copy() for key in source.files}
    mutate(values)
    np.savez_compressed(path, **values)
    digest = runner.sha256_file(path)
    assert digest != before
    summary[filename.removesuffix(".npz") + "_sha256"] = digest
    assert runner.sha256_file(path) == summary[filename.removesuffix(".npz") + "_sha256"]


@pytest.mark.parametrize("case", ["control", "true"], indirect=True)
def test_exact_saved_prediction_replay_full_schema(case):
    output, summary, arrays, registration = case
    runner.verify_summary(summary, summary["arm"])
    result = runner.verify_saved_predictions(output, summary, arrays, registration)
    assert result == {"exact_projection_replay": True, "cache_row_mapping_verified": True,
        "saved_weights_and_transforms_replayed": True, "secondary_mmd_replayed": True,
        "fold_source_global_core_metrics_replayed": True,
        "target_pooled_mse_replayed": True, "out_of_fold_groups": 398, "out_of_fold_source_targets": 112}
    assert len(summary["folds"]) == 8
    assert all(len(f["fit_targets"]) == 42 and len(f["held_targets"]) == 14 for f in summary["folds"])


@pytest.mark.parametrize("case", ["control", "true"], indirect=True)
@pytest.mark.parametrize("key", ["raw", "nonnegative"])
def test_rehashed_changed_cell_prediction_is_rejected(case, key):
    def tamper(values):
        values["fold_0_" + key][0, 0] += 0.125
    replace_npz(case, "predictions.npz", tamper)
    with pytest.raises(ValueError, match="projection replay"):
        runner.verify_saved_predictions(*case)


@pytest.mark.parametrize("key", ["control_cache_rows", "treated_cache_rows", "group", "treated_group"])
def test_rehashed_changed_row_mapping_is_rejected(case, key):
    def tamper(values):
        values["fold_0_" + key][0] += 1
    replace_npz(case, "predictions.npz", tamper)
    with pytest.raises(ValueError, match="row mapping"):
        runner.verify_saved_predictions(*case)


@pytest.mark.parametrize("key", ["held_source", "held_groups"])
def test_rehashed_changed_evaluation_role_is_rejected(case, key):
    def tamper(values):
        if key == "held_source":
            values["fold_0_" + key] = np.asarray("synthetic_other_source")
        else:
            values["fold_0_" + key] = values["fold_0_" + key][::-1].copy()
    replace_npz(case, "predictions.npz", tamper)
    with pytest.raises(ValueError, match="evaluation group roles"):
        runner.verify_saved_predictions(*case)


@pytest.mark.parametrize("key", ["fit_source", "held_source", "fit_targets", "held_targets"])
def test_rehashed_changed_weight_role_is_rejected(case, key):
    def tamper(values):
        if key.endswith("source"):
            values["fold_0_" + key] = np.asarray("synthetic_other_source")
        else:
            values["fold_0_" + key] = values["fold_0_" + key][::-1].copy()
    replace_npz(case, "weights.npz", tamper)
    with pytest.raises(ValueError, match="fit/held roles"):
        runner.verify_saved_predictions(*case)


@pytest.mark.parametrize("case", ["control", "true"], indirect=True)
def test_rehashed_changed_weight_coefficient_is_rejected(case):
    def tamper(values):
        values["fold_0_coefficients"][0, 0] += 0.125
    replace_npz(case, "weights.npz", tamper)
    with pytest.raises(ValueError, match="weights/transforms"):
        runner.verify_saved_predictions(*case)


@pytest.mark.parametrize("key", ["group_control_mean", "group_treated_mean", "group_raw_mean", "group_predicted_mean"])
def test_rehashed_changed_saved_group_mean_is_rejected(case, key):
    def tamper(values):
        values["fold_0_" + key][0, 0] += 0.125
    replace_npz(case, "predictions.npz", tamper)
    with pytest.raises(ValueError, match="group mean"):
        runner.verify_saved_predictions(*case)


@pytest.mark.parametrize("filename", ["predictions.npz", "weights.npz"])
def test_rehashed_changed_gene_order_is_rejected(case, filename):
    def tamper(values):
        values["genes"] = values["genes"][::-1].copy()
    replace_npz(case, filename, tamper)
    with pytest.raises(ValueError, match="gene axis"):
        runner.verify_saved_predictions(*case)


@pytest.mark.parametrize("location", ["source", "fold", "aggregate"])
@pytest.mark.parametrize("metric", ["target_pooled_mse", "raw_target_pooled_mse", "control_target_pooled_mse",
                                    "model_mmd2", "control_mmd2"])
def test_changed_reported_core_metric_is_rejected(case, location, metric):
    output, summary, arrays, registration = case
    values = (summary["source_metrics"][0]["metrics"] if location == "source" else
              summary["folds"][0]["metrics"] if location == "fold" else summary["aggregate"])
    values[metric] += 0.125
    # This helper receives an already authenticated parsed summary. Rewriting and
    # reading it simulates a fresh authenticated artifact, not stale-hash failure.
    path = output / "summary.json"
    path.write_text(json.dumps(summary, allow_nan=False))
    parsed = json.loads(path.read_text())
    with pytest.raises(ValueError, match="metric replay"):
        runner.verify_saved_predictions(output, parsed, arrays, registration)


def test_changed_control_reference_identity_is_rejected(case):
    case[1]["source_metrics"][0]["metrics"]["control_target_pooled_mse"] += 0.125
    with pytest.raises(ValueError, match="metric replay|control baseline"):
        runner.verify_saved_predictions(*case)
