"""Synthetic v5 core integration, no-refit replay and source/target isolation."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import public_target_context_v5 as model
from public_target_context_tracking import METRICS, TRAINING_METRICS
from test_public_sparse_hurdle_v4 import fixture


def fit_first(arrays, registration, tables, arm="true"):
    fold = model.v3.joint_folds(tuple(arrays["targets"]), tuple(arrays["sources"]), registration)[0]
    table = tables[fold.target_fold]
    aligned, receipt = model.v3.role_features(table, fold.fit_targets, fold.held_targets,
        arm=model.feature_arm(arm), fold_id=fold.target_fold)
    receipt = {**receipt, "consumer_arm": arm}
    prior = model.sparse.training_positive_prior(arrays, fold)
    response = model.sparse.training_responses(arrays, fold, prior)
    args = model.fit_arguments(arrays, fold)
    decoder = model.kernel.fit_kernel_decoder(args["training_targets"], args["training_sources"], args["context"],
        response, aligned=aligned, arm=arm, fold=fold, representation=model.sparse.REPRESENTATION)
    state = model.model_arrays(decoder, prior, response, fold, aligned, receipt, table)
    return fold, aligned, prior, response, decoder, state


def save_mutation(output, summary, filename, key, change=None):
    """Only temporary synthetic artifacts are replaced to test reauthorization."""
    path = output / filename
    with np.load(path, allow_pickle=False) as saved:
        payload = {name: saved[name].copy() for name in saved}
    if change is None:
        payload[key].flat[0] += 1
    else:
        payload[key] = change(payload[key])
    path.unlink()
    summary["weights_sha256" if filename == "weights.npz" else "predictions_sha256"] = model.save_arrays(path, payload)


def test_policy_is_json_stable_and_retains_sparse_decoder_and_fixed_arms():
    assert model.policy() == json.loads(json.dumps(model.policy(), allow_nan=False))
    assert model.policy()["decoder"] == model.sparse.policy()
    assert model.ARMS == model.kernel.ARMS
    assert model.policy()["hyperparameter_selection"] is False
    assert model.policy()["count_emitter"] is False
    assert model.feature_arm("additive") == "true"


@pytest.mark.parametrize("arm", model.ARMS)
def test_all_seven_arms_complete_and_replay_without_any_ridge_solve(tmp_path, monkeypatch, arm):
    arrays, registration, tables = fixture()
    events = []
    output = tmp_path / arm
    summary = model.run_arm(arrays, registration, tables, arm, output,
        {"diagnostic_contract_sha256": "a" * 64}, progress=events.append)
    assert summary["schema"] == "public-target-context-summary-v5"
    assert summary["status"] == "completed_unpromoted_diagnostic"
    assert len(events) == len(summary["folds"]) == 4  # Synthetic two target folds × two sources.
    assert summary["exact_ntc_identity_verified"] is True
    assert summary["exact_zero_effect_identity_verified"] is True
    assert summary["count_emitter"] is summary["promoted"] is summary["submission_performed"] is False
    assert set(METRICS) <= set(summary["aggregate"])
    for event, fold in zip(events, summary["folds"]):
        assert event["held_source"] == fold["held_source"]
        assert event["training_diagnostics"] == fold["training_diagnostics"]
        assert set(TRAINING_METRICS) <= set(event["training_diagnostics"])
        assert all(np.isfinite(v) and v >= 0 for v in event["training_diagnostics"].values())
        assert 0 <= event["training_diagnostics"]["training_unchanged_count_fraction"] <= 1
        assert fold["feature_receipt"]["consumer_arm"] == arm
    assert summary["aggregate"]["target_pooled_mse"] == pytest.approx(
        np.mean([s["metrics"]["target_pooled_mse"] for s in summary["source_metrics"]]))
    def forbidden(*args, **kwargs):
        pytest.fail("Artifact replay attempted a new ridge fit or solve")
    monkeypatch.setattr(model.kernel, "fit_kernel_decoder", forbidden)
    monkeypatch.setattr(np.linalg, "solve", forbidden)
    audit = model.verify_artifacts(output, summary, arrays, registration, tables)
    assert audit["passed"] is True and audit["ridge_refits"] == 0
    assert audit["kernel_system_verified"] is True and audit["training_diagnostics_replayed"] is True
    persisted = json.loads((output / "summary.json").read_text())
    assert model.verify_artifacts(output, persisted, arrays, registration, tables)["passed"]
    with np.load(output / "predictions.npz", allow_pickle=False) as saved:
        assert len(saved["fold_0_group"]) != len(saved["fold_0_treated_group"])
        np.testing.assert_array_equal(arrays["control_rows"][saved["fold_0_control_cache_rows"]], saved["fold_0_control_original_rows"])
        np.testing.assert_array_equal(arrays["treated_rows"][saved["fold_0_treated_cache_rows"]], saved["fold_0_treated_original_rows"])
    if arm == "control":
        assert summary["aggregate"]["target_pooled_mse"] == summary["aggregate"]["control_target_pooled_mse"]
        assert summary["aggregate"]["zero_fraction"] == summary["aggregate"]["control_zero_fraction"]
        assert all(f["training_diagnostics"]["effective_df"] == 0 for f in summary["folds"])
    with pytest.raises(FileExistsError):
        model.run_arm(arrays, registration, tables, arm, output, {"diagnostic_contract_sha256": "a" * 64})


@pytest.mark.parametrize("mutation", ["prior", "prior_count", "prior_row", "prior_source", "dual", "kernel_context",
    "training_response", "training_groups", "go_receipt", "kernel_diagnostics", "output", "donor", "row", "head",
    "fold_training_diagnostics", "source_metric", "aggregate", "fold_go_receipt"])
def test_rehashed_saved_state_or_report_mutations_fail_replay(tmp_path, mutation):
    arrays, registration, tables = fixture()
    output = tmp_path / mutation
    summary = model.run_arm(arrays, registration, tables, "true", output, {"diagnostic_contract_sha256": "a" * 64})
    weight_keys = {"prior": "positive_prior", "prior_count": "prior_positive_counts", "prior_row": "prior_original_rows",
        "prior_source": "prior_source_ids", "dual": "kernel_dual_coefficients", "kernel_context": "kernel_training_context_features",
        "training_response": "training_response", "training_groups": "training_group_ids",
        "go_receipt": "feature_receipt_json", "kernel_diagnostics": "kernel_training_diagnostics_json"}
    prediction_keys = {"output": "nonnegative", "donor": "donor_original_rows", "row": "control_original_rows", "head": "head_delta"}
    if mutation in weight_keys:
        change = None
        if mutation == "prior_source":
            change = lambda value: np.full(value.shape, "forbidden_source")
        elif mutation in {"go_receipt", "kernel_diagnostics"}:
            def change(value):
                record = json.loads(value.item())
                if mutation == "go_receipt": record["roles_crossed"] = True
                else: record["effective_df"] += 1
                return np.asarray(json.dumps(record, sort_keys=True))
        save_mutation(output, summary, "weights.npz", "fold_0_" + weight_keys[mutation], change)
    elif mutation in prediction_keys:
        save_mutation(output, summary, "predictions.npz", "fold_0_" + prediction_keys[mutation])
    elif mutation == "fold_training_diagnostics":
        summary["folds"][0]["training_diagnostics"]["training_unchanged_count_fraction"] -= .1
    elif mutation == "source_metric": summary["source_metrics"][0]["metrics"]["target_pooled_occupancy_mse"] += .1
    elif mutation == "aggregate": summary["aggregate"]["target_pooled_mse"] += .1
    else: summary["folds"][0]["feature_receipt"]["consumer_arm"] = "context"
    with pytest.raises(ValueError):
        model.verify_artifacts(output, summary, arrays, registration, tables)


def test_jointly_rehashed_dual_and_heads_still_fail_registered_system(tmp_path):
    arrays, registration, tables = fixture()
    output = tmp_path / "both"
    summary = model.run_arm(arrays, registration, tables, "true", output, {"diagnostic_contract_sha256": "a" * 64})
    save_mutation(output, summary, "weights.npz", "fold_0_kernel_dual_coefficients")
    save_mutation(output, summary, "predictions.npz", "fold_0_head_delta")
    with pytest.raises(ValueError, match="system residual"):
        model.verify_artifacts(output, summary, arrays, registration, tables)


@pytest.mark.parametrize("arm", ["context", "additive", "true", "shuffled_20260921"])
def test_held_source_and_target_responses_cannot_change_fold_training_state(arm):
    arrays, registration, tables = fixture()
    fold, aligned, prior, response, decoder, before = fit_first(arrays, registration, tables, arm)
    forbidden_groups = set(fold.held_target_indices_all_sources) | {
        i for i, source in enumerate(arrays["sources"]) if source == fold.held_source}
    arrays["treated"][np.isin(arrays["group"], list(forbidden_groups))] += 100
    held_source = [i for i, source in enumerate(arrays["sources"]) if source == fold.held_source]
    arrays["control"][np.isin(arrays["control_group"], held_source)] += 200
    arrays["context"][held_source] += 300
    arrays["context_control"][np.isin(arrays["context_control_group"], held_source)] += 400
    _, _, after_prior, after_response, after_decoder, after = fit_first(arrays, registration, tables, arm)
    assert set(before) == set(after)
    for key in before:
        np.testing.assert_array_equal(before[key], after[key], err_msg=key)
    np.testing.assert_array_equal(prior.values, after_prior.values)
    np.testing.assert_array_equal(response, after_response)
    np.testing.assert_array_equal(decoder.dual_coefficients, after_decoder.dual_coefficients)


def test_training_quantization_uses_only_fitting_reference_and_fitting_context():
    arrays, registration, tables = fixture()
    fold, aligned, _, response, decoder, _ = fit_first(arrays, registration, tables)
    before = model.training_diagnostics(decoder, response, arrays, fold, aligned)
    # Poison all data pools irrelevant to this diagnostic, after the synthetic
    # decoder and its authenticated training response are already constructed.
    arrays["control"][~np.isin(arrays["control_group"], fold.fit_indices)] = np.nan
    arrays["context"][[i for i in range(len(arrays["context"])) if i not in fold.fit_indices]] = np.nan
    arrays["treated"][:] = np.nan
    arrays["context_control"][:] = np.nan
    after = model.training_diagnostics(decoder, response, arrays, fold, aligned)
    assert before == after


def test_training_quantization_matches_frozen_sparse_count_rule():
    arrays, registration, tables = fixture()
    fold, aligned, prior, response, decoder, _ = fit_first(arrays, registration, tables)
    expected = []
    for group in fold.fit_indices:
        selected = np.flatnonzero(arrays["control_group"] == group)
        control = arrays["control"][selected]
        packet = model.v3._subset(aligned, (str(arrays["targets"][group]),))
        head = decoder.predict_delta(packet, arrays["context"][group:group+1], is_ntc=np.array([False]))[0]
        width = control.shape[1]
        prediction = model.sparse.sparse_anchor(control, np.zeros_like(control), prior.values,
            head[:width], head[width:], control_rows=arrays["control_rows"][selected],
            context_rows=np.arange(len(control)) + 100000, group_id=group, genes=tuple(arrays["genes"]))
        expected.append(prediction.metrics["unchanged_count_fraction"])
    targets = tuple(str(arrays["targets"][i]) for i in fold.fit_indices)
    sources = tuple(str(arrays["sources"][i]) for i in fold.fit_indices)
    weighted = float(model.balanced_weights(targets, sources) @ expected)
    report = model.training_diagnostics(decoder, response, arrays, fold, aligned)
    assert report["training_unchanged_count_fraction"] == weighted


def test_training_diagnostics_rejects_a_different_response():
    arrays, registration, tables = fixture()
    fold, aligned, _, response, decoder, _ = fit_first(arrays, registration, tables)
    changed = response.copy()
    changed.flat[0] += 1
    with pytest.raises(ValueError, match="admitted training response"):
        model.training_diagnostics(decoder, changed, arrays, fold, aligned)


def test_invalid_role_manifest_fails_before_fitting(tmp_path, monkeypatch):
    arrays, registration, tables = fixture()
    registration = copy.deepcopy(registration)
    first = registration["folds"][0]["directions"][0]
    first["fit_group_ids"][0] = first["joint_eval_group_ids"][0]
    monkeypatch.setattr(model.kernel, "fit_kernel_decoder", lambda *a, **k: pytest.fail("Fit occurred before role admission"))
    with pytest.raises(ValueError, match="fit_group_ids"):
        model.run_arm(arrays, registration, tables, "true", tmp_path / "bad", {"diagnostic_contract_sha256": "a" * 64})
    assert not (tmp_path / "bad").exists()
