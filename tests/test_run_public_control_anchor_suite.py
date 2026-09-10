"""Synthetic orchestration checks; no expression, network, GPU or real W&B."""
import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_public_control_anchor_suite as suite


def test_register_only_hashes_opaque_inputs(tmp_path):
    pilot = tmp_path / "pilot"
    for relative in suite.FILES.values():
        path = pilot / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"opaque, intentionally not a numpy archive")
    target = tmp_path / "registered.json"
    digest = suite.register(pilot, target)
    contract = suite.validate_diagnostic_contract(target, digest, suite.bindings(pilot))
    assert contract["registration_npz_decoded"] is False
    assert contract["screen"] == suite.screening_policy()
    assert contract["stage"] == "validation"
    with pytest.raises(FileExistsError):
        suite.register(pilot, target)
    contract["policy"]["gain"] = 0.5
    target.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="policy"):
        suite.validate_diagnostic_contract(target, suite.sha256_file(target), suite.bindings(pilot))


def test_symlinks_are_not_inputs(tmp_path):
    target = tmp_path / "regular"
    target.write_bytes(b"x")
    (tmp_path / "cache.npz").symlink_to(target)
    with pytest.raises(ValueError, match="Regular"):
        suite.bindings(tmp_path)


def test_fake_slurm_does_not_run_on_login_host(tmp_path, monkeypatch):
    monkeypatch.setattr(suite.socket, "gethostname", lambda: "login03.anvil")
    monkeypatch.setenv("SLURM_JOB_ID", "20539599")
    monkeypatch.setattr(suite.subprocess, "call", lambda *a, **k: pytest.fail("Must not launch"))
    with pytest.raises(RuntimeError, match="BioHPC"):
        suite.run(tmp_path, tmp_path / "missing", "a" * 64, tmp_path / "output")


def metrics(mse=0.9, mmd=0.8):
    return {"model_centroid_mse": mse, "control_centroid_mse": 1.0,
            "model_mmd2": mmd, "control_mmd2": 1.0,
            "negative_fraction": 0.0, "projection_fraction": 0.1}


def folds(mse=0.9, mmd=0.8):
    return [{"held_source": source, "train_sources": sorted(suite.SOURCES - {source}),
             "aggregate": metrics(mse, mmd)} for source in sorted(suite.SOURCES)]


def completion(tmp_path):
    run = tmp_path / "runs" / "true"
    tracking = tmp_path / "tracking" / "true"
    run.mkdir(parents=True)
    tracking.mkdir(parents=True)
    for name in ("weights", "predictions"):
        (run / (name + ".npz")).write_bytes(name.encode())
    summary = {"schema": "public-control-anchor-validation-summary-v2",
        "status": "completed_unpromoted_diagnostic", "arm": "true", "stage": "validation",
        "diagnostic_contract_sha256": "a" * 64, "policy": suite.baseline.policy(),
        "posttraining_performed": False, "development_used_for_fit_or_selection": False,
        "count_emitter": False, "exact_ntc_identity_verified": True,
        "exact_zero_effect_identity_verified": True, "outer_folds": folds(),
        "weights_sha256": suite.sha256_file(run / "weights.npz"),
        "predictions_sha256": suite.sha256_file(run / "predictions.npz"),
        "deployment_rollout_diagnostics": {"kind_metrics": {"context": metrics()}}}
    receipt = {"execution_status": "completed", "training_exit_code": 0, "tracking_errors": 0,
        "event_count": 3, "mode": "offline", "stage": "validation", "cloud_synced": False,
        "run_id": "aabbccddeeff"}
    (run / "summary.json").write_text(json.dumps(summary))
    (tracking / "tracking.json").write_text(json.dumps(receipt))
    return run, tracking, summary, receipt


def test_exit_zero_does_not_substitute_for_artifact_hashes(tmp_path):
    run, _, _, _ = completion(tmp_path)
    assert suite.verify_arm(tmp_path, "true", "a" * 64)["metrics"]["model_centroid_mse"] == 0.9
    (run / "predictions.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash"):
        suite.verify_arm(tmp_path, "true", "a" * 64)


@pytest.mark.parametrize("key,value", [("tracking_errors", 1), ("event_count", 2),
    ("cloud_synced", True), ("training_exit_code", 2), ("mode", "online"), ("stage", "posttrain")])
def test_incomplete_tracking_stops_suite(tmp_path, key, value):
    _, tracking, _, receipt = completion(tmp_path)
    receipt[key] = value
    (tracking / "tracking.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="W&B"):
        suite.verify_arm(tmp_path, "true", "a" * 64)


@pytest.mark.parametrize("mutation", ["leak", "nan", "null_identity", "negative_error"])
def test_invalid_scientific_evidence_rejected(tmp_path, mutation):
    run, _, summary, _ = completion(tmp_path)
    if mutation == "leak":
        summary["outer_folds"][0]["train_sources"] = list(suite.SOURCES)
    elif mutation == "nan":
        summary["outer_folds"][0]["aggregate"]["model_centroid_mse"] = float("nan")
    elif mutation == "null_identity":
        summary["exact_ntc_identity_verified"] = False
    else:
        summary["outer_folds"][0]["aggregate"]["model_centroid_mse"] = -1.0
    (run / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        suite.verify_arm(tmp_path, "true", "a" * 64)


def passing_arms():
    return [{"arm": arm, "outer_folds": folds(mse, mse)} for arm, mse in
            (("control", 1.0), ("context", 0.99), ("true", 0.9), ("shuffled", 0.98))]


def test_fixed_effect_size_gate_never_promotes():
    result = suite.screen(passing_arms())
    assert result["passed"] is True
    assert result["promoted"] is False
    assert result["posttraining_performed"] is False


@pytest.mark.parametrize("mutation", ["tiny_gain", "one_source_bad", "worse_mmd", "negative", "reference_mismatch"])
def test_average_gain_does_not_hide_failed_source(mutation):
    arms = copy.deepcopy(passing_arms())
    true = next(a for a in arms if a["arm"] == "true")
    if mutation == "tiny_gain":
        for fold in true["outer_folds"]:
            fold["aggregate"]["model_centroid_mse"] = 0.979
    elif mutation == "one_source_bad":
        true["outer_folds"][0]["aggregate"]["model_centroid_mse"] = 1.1
    elif mutation == "worse_mmd":
        true["outer_folds"][0]["aggregate"]["model_mmd2"] = 1.1
    elif mutation == "negative":
        true["outer_folds"][0]["aggregate"]["negative_fraction"] = 0.01
    else:
        true["outer_folds"][0]["aggregate"]["control_centroid_mse"] = 2.0
        with pytest.raises(ValueError, match="reference"):
            suite.screen(arms)
        return
    assert suite.screen(arms)["passed"] is False


def test_missing_arm_is_not_a_gate_pass():
    with pytest.raises(ValueError, match="Every fixed arm"):
        suite.screen(passing_arms()[:-1])
