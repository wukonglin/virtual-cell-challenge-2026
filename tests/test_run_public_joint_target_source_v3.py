"""Synthetic-only execution admission, fixed screen and offline orchestration."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_public_joint_target_source_v3 as runner
from public_joint_target_source_tracking import JointTargetSourceTracker
from test_wandb_training import FakeSDK


def summary(arm, primary=1.0, mmd=0.1):
    metrics = dict.fromkeys(runner.CORE_METRICS, 0.1)
    metrics.update(target_pooled_mse=primary, model_mmd2=mmd, negative_fraction=0.0)
    return {"schema": "public-joint-target-source-summary-v3", "arm": arm,
        "status": "completed_unpromoted_diagnostic", "prior_public_development_exposure_acknowledged": True,
        "policy": runner.model.policy(), "stage": "validation", "aggregate": dict(metrics),
        "source_metrics": [{"source": s, "metrics": dict(metrics)} for s in sorted(runner.SOURCES)],
        "folds": [{"fold_id": str(i), "fit_targets": [str(t) for t in range(42)],
            "held_targets": [str(t) for t in range(42, 56)], "metrics": dict(metrics)} for i in range(8)],
        "exact_ntc_identity_verified": True, "exact_zero_effect_identity_verified": True,
        **dict.fromkeys(("count_emitter", "posttraining_performed", "promoted", "submission_performed",
            "hyperparameter_selection", "H1_treated_read", "RPE1_treated_read", "challenge_treated_used",
            "held_roles_used_for_fitting"), False)}


def candidates():
    return {arm: summary(arm, 0.98 if arm == "true" else 1.0, 0.09 if arm == "true" else 0.1)
            for arm in runner.model.ARMS}


def test_fixed_all_comparator_screen_can_pass_without_promotion():
    decision = runner.screen(candidates())
    assert decision["passed"] and not decision["automatic_promotion"]
    assert not decision["leaderboard_improvement_claimed"]
    assert all(len(r["checks"]) == 10 for r in decision["sources"])


@pytest.mark.parametrize("arm", ["control", "context", *runner.model.ARMS[3:]])
def test_one_stronger_comparator_in_one_source_fails(arm):
    values = candidates()
    values[arm]["source_metrics"][0]["metrics"]["target_pooled_mse"] = 0.97
    assert not runner.screen(values)["passed"]


@pytest.mark.parametrize("arm", ["control", "context", *runner.model.ARMS[3:]])
def test_mmd_noninferiority_required_against_each_comparator(arm):
    values = candidates()
    values[arm]["source_metrics"][1]["metrics"]["model_mmd2"] = 0.08
    assert not runner.screen(values)["passed"]


def test_tiny_gain_is_not_quality_pass():
    values = candidates()
    values["true"]["source_metrics"][0]["metrics"]["target_pooled_mse"] = 0.999
    assert not runner.screen(values)["passed"]


def test_all_shuffles_required():
    values = candidates(); values.pop(runner.model.ARMS[-1])
    with pytest.raises(ValueError, match="six"):
        runner.screen(values)


@pytest.mark.parametrize("change", ["nan", "negative", "fraction", "identity", "promotion", "folds", "sources"])
def test_invalid_summary_fails_closed(change):
    value = summary("true")
    if change == "nan": value["aggregate"]["target_pooled_mse"] = float("nan")
    elif change == "negative": value["aggregate"]["negative_fraction"] = 0.01
    elif change == "fraction": value["aggregate"]["zero_fraction"] = 1.1
    elif change == "identity": value["exact_ntc_identity_verified"] = False
    elif change == "promotion": value["promoted"] = True
    elif change == "folds": value["folds"].pop()
    else: value["source_metrics"].pop()
    with pytest.raises(ValueError): runner.verify_summary(value, "true")


@pytest.fixture
def registered(tmp_path, monkeypatch):
    role = tmp_path / "role.json"
    role.write_text("not an expression archive")
    role_sha = runner.sha256_file(role)
    record = {"inputs": {"cache_sha256": "a" * 64, "contract_sha256": "b" * 64,
              "source_manifest_sha256": "c" * 64},
              "input_paths": {k: str(tmp_path / k) for k in ("cache", "contract", "source_manifest")},
              "excluded_targets": [str(i) for i in range(772)]}
    def validate(path, digest):
        runner.authenticated_bytes(path, digest, 1 << 20)
        return copy.deepcopy(record)
    monkeypatch.setattr(runner.preparation, "load_registration", validate)
    output = tmp_path / "registration"
    digest = runner.register(role, role_sha, output)
    return output / "contract.json", digest, role, record


def test_expression_free_execution_registration(registered):
    path, digest, _, _ = registered
    value, record = runner.validate_contract(path, digest)
    assert value["expression_decoded"] is False
    assert value["cpu_only"] and value["wandb_mode"] == "offline"
    assert set(value["code_sha256"]) == set(runner.MODULES)


@pytest.mark.parametrize("key,value", [("automatic_promotion", True), ("new_raw_expression_allowed", True),
    ("expression_decoded", True), ("cpu_only", False), ("stage", "pretrain"), ("events_per_arm", 8)])
def test_execution_policy_cannot_be_relaxed(registered, key, value):
    path, _, _, _ = registered
    record = json.loads(path.read_text()); record[key] = value; path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="policy"):
        runner.validate_contract(path, runner.sha256_file(path))


def test_changed_metadata_rejected(registered):
    path, digest, role, _ = registered
    role.write_text("changed roles")
    with pytest.raises(ValueError, match="SHA"):
        runner.validate_contract(path, digest)


def test_changed_code_pin_rejected(registered):
    path, _, _, _ = registered
    record = json.loads(path.read_text()); record["code_sha256"][runner.MODULES[0]] = "f" * 64
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Implementation"):
        runner.validate_contract(path, runner.sha256_file(path))


def test_fake_slurm_id_cannot_authorize_login_node(tmp_path, monkeypatch):
    monkeypatch.setattr(runner.socket, "gethostname", lambda: "login02.anvil")
    monkeypatch.setenv("SLURM_JOB_ID", "20539599")
    monkeypatch.setattr(runner, "load_cache", lambda *a, **k: pytest.fail("No decoding"))
    with pytest.raises(RuntimeError, match="compute host"):
        runner.run(tmp_path / "none", "a" * 64, tmp_path / "output")


@pytest.mark.parametrize("log_failure", [False, True])
def test_mock_six_arm_suite_records_fifty_four_events(registered, tmp_path, monkeypatch, log_failure):
    path, digest, _, record = registered
    monkeypatch.setattr(runner.socket, "gethostname", lambda: "cbsuvlaminck3.biohpc.cornell.edu")
    monkeypatch.setattr(runner.preparation, "load_feature_tables", lambda *a: {"synthetic": True})
    monkeypatch.setattr(runner, "load_cache", lambda *a: ({"synthetic": np.zeros(2)}, record))
    monkeypatch.setattr(runner, "verify_saved_predictions", lambda *a: {"synthetic_audit": True})
    sdks = []
    def tracker(*args, **kwargs):
        sdk = FakeSDK(log_error=log_failure); sdks.append(sdk)
        return JointTargetSourceTracker(*args, sdk=sdk, **kwargs)
    monkeypatch.setattr(runner, "JointTargetSourceTracker", tracker)
    def fit(arrays, registration, tables, arm, output_dir, provenance, progress=None):
        assert not arrays["synthetic"].flags.writeable
        output_dir.mkdir()
        value = candidates()[arm]
        value["diagnostic_contract_sha256"] = provenance["diagnostic_contract_sha256"]
        value["provenance"] = provenance
        for i in range(8):
            progress({"fold_index": i, "held_source": sorted(runner.SOURCES)[i % 2],
                      "target_fold": str(i // 2), "metrics": value["aggregate"]})
        runner.dump_new(output_dir / "summary.json", value)
        return value
    monkeypatch.setattr(runner.model, "run_arm", fit)
    output = tmp_path / "run"
    if log_failure:
        with pytest.raises(ValueError, match="W&B"):
            runner.run(path, digest, output)
        assert not (output / "complete.json").exists()
        assert len(sdks) == 1
    else:
        assert runner.run(path, digest, output) == 0
        complete = json.loads((output / "complete.json").read_text())
        assert complete["screen_passed"] and not complete["promoted"]
        assert complete["tracking_events"] == 54 and complete["tracking_errors"] == 0
        assert len(sdks) == 6
    for sdk in sdks:
        assert len(sdk.run.logged) == 9
        assert sdk.initialized[0]["mode"] == "offline"
        assert [data["analysis/event_index"] for data, _ in sdk.run.logged] == list(range(1, 10))
