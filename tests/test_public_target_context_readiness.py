"""One-shot v5 readiness: synthetic authenticated reports, no model or data reads."""
from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import public_target_context_readiness as check

NOW = datetime(2026, 9, 10, 12, tzinfo=timezone.utc)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False))
    return check.evidence.digest(path)


class SyntheticRunner:
    """Exercise checker orchestration; the real runner has independent tests."""
    def __init__(self):
        self.contract_calls = 0
        self.summary_calls = []
        self.screen_calls = 0

    def validate_contract(self, path, sha):
        self.contract_calls += 1
        return check.evidence.read_json(path, sha), {"synthetic": True}

    def verify_summary(self, summary, arm, contract_sha=None):
        self.summary_calls.append(arm)
        if summary.get("schema") != check.SUMMARY_SCHEMA or summary.get("arm") != arm:
            raise ValueError("synthetic summary shape invalid")
        if contract_sha is not None and summary.get("diagnostic_contract_sha256") != contract_sha:
            raise ValueError("synthetic contract invalid")

    def screen(self, summaries):
        self.screen_calls += 1
        assert set(summaries) == set(check.ARMS)
        rows = []
        for source in sorted(check.SOURCES):
            values = {arm: next(row["metrics"] for row in report["source_metrics"] if row["source"] == source)
                      for arm, report in summaries.items()}
            true = values["true"]
            condition = true["target_pooled_mse"] <= .99 * values["control"]["target_pooled_mse"]
            distribution = true["model_mmd2"] <= values["control"]["model_mmd2"]
            rows.append({"source": source, "passed": condition and distribution,
                "conditioning_passed": condition, "distribution_passed": distribution, "safety_passed": True,
                "checks": {"primary": condition, "distribution": distribution, "safety": True}})
        return {"policy": {"synthetic_threshold": .01}, "sources": rows,
            **{key: all(row[key] for row in rows) for key in ("passed", "conditioning_passed", "distribution_passed", "safety_passed")},
            "automatic_promotion": False, "leaderboard_improvement_claimed": False}


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    runner = SyntheticRunner()
    monkeypatch.setattr(check, "get_runner", lambda: runner)
    policy = {"synthetic_fixed_model": True}
    contract = {"schema": check.SUITE_SCHEMA, "stage": "validation", "cpu_only": True,
        "wandb_mode": "offline", "policy": policy, "screen": {"synthetic_threshold": .01}}
    cp = tmp_path / "contract.json"; cs = write(cp, contract)
    runtime = {"hostname": "cbsuvlaminck3.example", "device": "cpu"}
    suite = {"schema": check.SUITE_SCHEMA, "completed": True, "completed_at": NOW.isoformat(),
        "diagnostic_contract_path": str(cp), "diagnostic_contract_sha256": cs,
        "runtime": runtime, "tracking_events": 63, "tracking_errors": 0, "runs": [], "source_metrics": {},
        **dict.fromkeys(check.FALSE_SUITE_FLAGS, False)}
    summaries = {}
    for index, arm in enumerate(check.ARMS):
        parent = tmp_path / arm; parent.mkdir()
        pins = {}
        for name in ("weights", "predictions"):
            path = parent / (name + ".npz"); path.write_bytes(("synthetic opaque " + name).encode())
            pins[name + "_sha256"] = check.evidence.digest(path)
        summary = {"schema": check.SUMMARY_SCHEMA, "arm": arm, "stage": "validation",
            "status": "completed_unpromoted_diagnostic", "diagnostic_contract_sha256": cs,
            "provenance": {"diagnostic_contract_sha256": cs}, "policy": policy, "runtime": runtime.copy(),
            "exact_ntc_identity_verified": True, "exact_zero_effect_identity_verified": True,
            "prior_public_development_exposure_acknowledged": True,
            **dict.fromkeys(check.FALSE_SUMMARY_FLAGS, False), **pins,
            "source_metrics": [{"source": source, "metrics": {"target_pooled_mse": .8 if arm == "true" else 1.0,
                "model_mmd2": .4 if arm == "true" else .5, "negative_fraction": 0.0}} for source in sorted(check.SOURCES)]}
        sp = parent / "summary.json"; ss = write(sp, summary)
        tracking = {"schema": "vcc-training-tracking-v1", "run_id": f"synthetic_{index}", "stage": "validation",
            "mode": "offline", "cloud_synced": False, "execution_status": "completed", "training_exit_code": 0,
            "event_count": 9, "tracking_errors": 0, "scientific_quality_inferred_from_exit": False,
            "config": {"diagnostic_contract_sha256": cs}}
        tp = parent / "tracking.json"; ts = write(tp, tracking)
        suite["runs"].append({"arm": arm, "summary_path": str(sp), "summary_sha256": ss,
            "tracking_path": str(tp), "tracking_sha256": ts, "run_id": tracking["run_id"],
            "tracking_events": 9, "tracking_errors": 0, "cloud_synced": False,
            "artifact_audit": {**dict.fromkeys(check.AUDIT_TRUE, True), "ridge_refits": 0,
                "out_of_fold_groups": 398, "out_of_fold_source_targets": 112}})
        suite["source_metrics"][arm] = copy.deepcopy(summary["source_metrics"])
        summaries[arm] = summary
    suite["decision"] = runner.screen(summaries)
    data = {"root": tmp_path, "suite": suite, "summaries": summaries, "runner": runner}
    seal(data)
    return data


def seal(data):
    suite = data["suite"]; path = data["root"] / "suite.json"
    data["path"], data["sha"] = path, write(path, suite)
    write(data["root"] / "complete.json", {"schema": check.SUITE_SCHEMA, "completed": True,
        "completed_at": suite["completed_at"], "suite_sha256": data["sha"],
        "diagnostic_contract_sha256": suite["diagnostic_contract_sha256"], "screen_passed": suite["decision"]["passed"],
        "tracking_events": 63, "tracking_errors": 0, "cloud_synced": False, "promoted": False, "submission_performed": False})


def evaluate(data):
    return check.evaluate(data["path"], data["sha"], now=NOW)


def reseal_summary(data, arm="true", *, copy_metrics=False):
    row = next(row for row in data["suite"]["runs"] if row["arm"] == arm)
    row["summary_sha256"] = write(Path(row["summary_path"]), data["summaries"][arm])
    if copy_metrics:
        data["suite"]["source_metrics"][arm] = copy.deepcopy(data["summaries"][arm]["source_metrics"])
    seal(data)


def test_passing_science_only_enables_promoter_not_feng_or_submission(fixture):
    report = evaluate(fixture)
    assert report["suite_evidence_valid"] and report["scientific_gate_passed"] and report["promoter_ablation_ready"]
    assert report["artifact_bytes_verified"] and not report["feng_posttraining_ready"] and not report["submission_ready"]
    assert "flow_parent" in report["reasons"]["feng"] and "raw_count_emitter" in report["reasons"]["submission"]
    assert fixture["runner"].contract_calls == 1 and set(fixture["runner"].summary_calls) == set(check.ARMS)
    assert not any(report[key] for key in ("credentials_read", "network_used", "raw_arrays_decoded",
        "automatic_training", "automatic_submission", "online_notifications", "v4_watcher_modified"))


def test_failed_recomputed_science_is_valid_evidence_but_not_readiness(fixture, monkeypatch):
    fixture["summaries"]["true"]["source_metrics"][0]["metrics"]["target_pooled_mse"] = 1.1
    fixture["suite"]["decision"] = fixture["runner"].screen(fixture["summaries"])
    reseal_summary(fixture, copy_metrics=True)
    original = check.evidence.digest
    def digest(path):
        assert Path(path).suffix != ".npz", "Failed gate should not trigger artifact-byte reads"
        return original(path)
    monkeypatch.setattr(check.evidence, "digest", digest)
    report = evaluate(fixture)
    assert report["suite_evidence_valid"] and not report["scientific_gate_passed"]
    assert not report["promoter_ablation_ready"] and not report["artifact_bytes_verified"]


@pytest.mark.parametrize("mutation", ["schema", "device", "host", "bool_decision", "source_bool", "audit",
    "missing_arm", "duplicate_run", "event_count", "pretraining", "copied_metric", "copied_metric_type", "decision"])
def test_rehashed_suite_mutations_fail_closed(fixture, mutation):
    suite = fixture["suite"]
    if mutation == "schema": suite["schema"] = "public-sparse-hurdle-execution-v4"
    elif mutation == "device": suite["runtime"]["device"] = "cuda"
    elif mutation == "host": suite["runtime"]["hostname"] = "aida-login"
    elif mutation == "bool_decision": suite["decision"]["passed"] = 1
    elif mutation == "source_bool": suite["decision"]["sources"][0]["checks"]["safety"] = 1
    elif mutation == "audit": suite["runs"][0]["artifact_audit"]["passed"] = False
    elif mutation == "missing_arm": suite["runs"].pop()
    elif mutation == "duplicate_run": suite["runs"][1]["run_id"] = suite["runs"][0]["run_id"]
    elif mutation == "event_count": suite["tracking_events"] = 54
    elif mutation == "pretraining": suite["pretraining_performed"] = True
    elif mutation == "copied_metric": suite["source_metrics"]["true"][0]["metrics"]["target_pooled_mse"] = .1
    elif mutation == "copied_metric_type": suite["source_metrics"]["true"][0]["metrics"]["negative_fraction"] = False
    elif mutation == "decision": suite["decision"]["leaderboard_improvement_claimed"] = True
    seal(fixture)
    assert not evaluate(fixture)["suite_evidence_valid"]


@pytest.mark.parametrize("mutation", ["device", "host", "stage", "provenance", "held_fit", "emitter"])
def test_rehashed_summary_runtime_and_provenance_mutations_fail(fixture, mutation):
    summary = fixture["summaries"]["true"]
    if mutation == "device": summary["runtime"]["device"] = "cuda"
    elif mutation == "host": summary["runtime"]["hostname"] = "anvil-login"
    elif mutation == "stage": summary["stage"] = "pretrain"
    elif mutation == "provenance": summary["provenance"]["diagnostic_contract_sha256"] = "c" * 64
    elif mutation == "held_fit": summary["held_roles_used_for_fitting"] = True
    elif mutation == "emitter": summary["count_emitter"] = True
    reseal_summary(fixture)
    assert not evaluate(fixture)["suite_evidence_valid"]


@pytest.mark.parametrize("copy_metrics", [False, True])
def test_rehashed_changed_summary_metrics_cannot_keep_old_pass(fixture, copy_metrics):
    fixture["summaries"]["true"]["source_metrics"][0]["metrics"]["target_pooled_mse"] = 2.0
    reseal_summary(fixture, copy_metrics=copy_metrics)
    report = evaluate(fixture)
    assert not report["suite_evidence_valid"]
    assert report["reasons"]["promoter"] == ("recomputed_decision_mismatch" if copy_metrics else "copied_source_metrics_mismatch")


@pytest.mark.parametrize("key,value", [("training_exit_code", False), ("mode", "online"),
    ("event_count", 8), ("tracking_errors", 1), ("scientific_quality_inferred_from_exit", True)])
def test_rehashed_tracking_mutations_fail(fixture, key, value):
    row = fixture["suite"]["runs"][0]
    path = Path(row["tracking_path"]); receipt = check.evidence.read_json(path)
    receipt[key] = value; row["tracking_sha256"] = write(path, receipt)
    seal(fixture)
    assert not evaluate(fixture)["suite_evidence_valid"]


@pytest.mark.parametrize("name", ["weights", "predictions"])
def test_changed_artifact_bytes_block_passing_gate(fixture, name):
    (fixture["root"] / "true" / (name + ".npz")).write_bytes(b"tampered synthetic bytes")
    report = evaluate(fixture)
    assert not report["promoter_ablation_ready"] and report["reasons"]["promoter"] == "evidence_hash_mismatch"


@pytest.mark.parametrize("key", check.AUDIT_TRUE)
def test_each_critical_replay_check_is_required(fixture, key):
    del fixture["suite"]["runs"][0]["artifact_audit"][key]
    seal(fixture)
    assert evaluate(fixture)["reasons"]["promoter"] == "artifact_replay_not_passed"


@pytest.mark.parametrize("key,value", [("ridge_refits", False), ("ridge_refits", 1),
    ("out_of_fold_groups", 397), ("out_of_fold_source_targets", 111)])
def test_replay_counts_require_exact_typed_values(fixture, key, value):
    fixture["suite"]["runs"][0]["artifact_audit"][key] = value
    seal(fixture)
    assert evaluate(fixture)["reasons"]["promoter"] == "artifact_replay_not_passed"


@pytest.mark.parametrize("hours", [-169, 1])
def test_stale_or_future_completion_rejected(fixture, hours):
    fixture["suite"]["completed_at"] = (NOW + timedelta(hours=hours)).isoformat()
    seal(fixture)
    assert not evaluate(fixture)["suite_evidence_valid"]


def test_missing_or_retimestamped_completion_rejected(fixture):
    path = fixture["root"] / "complete.json"
    value = check.evidence.read_json(path); value["completed_at"] = (NOW + timedelta(minutes=1)).isoformat()
    write(path, value)
    assert not evaluate(fixture)["suite_evidence_valid"]
    path.unlink()
    assert not evaluate(fixture)["suite_evidence_valid"]


def test_cli_writes_fresh_record_only_and_has_no_watch_mode(fixture, monkeypatch, capsys):
    monkeypatch.setattr(check.socket, "gethostname", lambda: "cbsuvlaminck3")
    monkeypatch.setattr(check, "evaluate", lambda *a, **kw: {"schema": check.SCHEMA, "submission_ready": False})
    path = fixture["root"] / "readiness.json"
    args = ["--suite", str(fixture["path"]), "--suite-sha256", fixture["sha"], "--output", str(path)]
    assert check.main(args) == 0 and json.loads(path.read_text())["submission_ready"] is False
    before = path.read_bytes()
    with pytest.raises(SystemExit) as error: check.main(args)
    assert error.value.code == 2 and path.read_bytes() == before
    assert "submission_ready" in capsys.readouterr().out


def test_cli_refuses_login_host_before_read_or_write(fixture, monkeypatch):
    monkeypatch.setattr(check.socket, "gethostname", lambda: "login")
    monkeypatch.setattr(check, "evaluate", lambda *a, **kw: pytest.fail("Should not read evidence on login host"))
    with pytest.raises(SystemExit):
        check.main(["--suite", str(fixture["path"]), "--suite-sha256", fixture["sha"], "--output", str(fixture["root"] / "new.json")])
    assert not (fixture["root"] / "new.json").exists()
