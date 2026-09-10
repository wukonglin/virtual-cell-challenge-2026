"""Synthetic orchestration tests; never launch real training or tracking."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_public_mean_effect_suite as suite


def test_register_only_hashes_opaque_inputs(tmp_path):
    pilot = tmp_path / "pilot"
    for relative in suite.FILES.values():
        path = pilot / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"opaque bytes; deliberately not a numpy file")
    target = tmp_path / "registered.json"
    digest = suite.register(pilot, target)
    contract = json.loads(target.read_text())
    assert digest == suite.sha256_file(target)
    assert contract["registration_npz_decoded"] is False
    assert contract["policy"] == suite.baseline.policy()
    assert len(contract["code_sha256"]) == 5
    assert set(contract["tracking_code_sha256"]) == {"wandb_training.py", "run_training_with_wandb.py"}
    with pytest.raises(FileExistsError):
        suite.register(pilot, target)


def test_source_alias_is_rejected(tmp_path):
    target = tmp_path / "regular"
    target.write_bytes(b"x")
    (tmp_path / "cache.npz").symlink_to(target)
    with pytest.raises(ValueError, match="Regular"):
        suite.bindings(tmp_path)


def test_fake_slurm_does_not_start_wrapper_on_login(tmp_path, monkeypatch):
    monkeypatch.setattr(suite.socket, "gethostname", lambda: "login03.anvil")
    monkeypatch.setenv("SLURM_JOB_ID", "1234")
    monkeypatch.setattr(suite.subprocess, "call", lambda *a, **k: pytest.fail("Must not launch"))
    with pytest.raises(RuntimeError, match="BioHPC"):
        suite.run(tmp_path, tmp_path / "missing", "a" * 64, tmp_path / "output")


def completion(tmp_path):
    run = tmp_path / "runs" / "true"
    tracking = tmp_path / "tracking" / "true"
    run.mkdir(parents=True)
    tracking.mkdir(parents=True)
    for name in ("weights", "predictions"):
        (run / (name + ".npz")).write_bytes(name.encode())
    summary = {"status": "completed_unpromoted_diagnostic", "arm": "true",
        "provenance": {"diagnostic_contract_sha256": "a" * 64},
        "policy": suite.baseline.policy(), "posttraining_performed": False,
        "development_used_for_fit_or_selection": False,
        "weights_sha256": suite.sha256_file(run / "weights.npz"),
        "predictions_sha256": suite.sha256_file(run / "predictions.npz"),
        "deployment_rollout_diagnostics": {"kind_metrics": {"context": {"model_centroid_mse": 1.2}}}}
    receipt = {"execution_status": "completed", "training_exit_code": 0, "tracking_errors": 0,
        "event_count": 3, "mode": "offline", "cloud_synced": False, "run_id": "aabbccddeeff"}
    (run / "summary.json").write_text(json.dumps(summary))
    (tracking / "tracking.json").write_text(json.dumps(receipt))
    return run, tracking, summary, receipt


def test_artifacts_and_receipts_not_exit_zero_alone(tmp_path):
    completion(tmp_path)
    result = suite.verify_arm(tmp_path, "true", "a" * 64)
    assert result["metrics"]["model_centroid_mse"] == 1.2
    (tmp_path / "runs" / "true" / "weights.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash"):
        suite.verify_arm(tmp_path, "true", "a" * 64)


@pytest.mark.parametrize("key,value", [("tracking_errors", 1), ("event_count", 2),
    ("cloud_synced", True), ("training_exit_code", 2), ("mode", "online")])
def test_incomplete_tracking_stops_suite(tmp_path, key, value):
    _, tracking, _, receipt = completion(tmp_path)
    receipt[key] = value
    (tracking / "tracking.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="W&B"):
        suite.verify_arm(tmp_path, "true", "a" * 64)
