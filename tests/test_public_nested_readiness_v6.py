"""Authenticated synthetic metadata only: no real data, model or service calls."""
from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import public_nested_readiness_v6 as check

NOW = datetime(2026, 9, 10, 12, tzinfo=timezone.utc)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False))
    return check.evidence.digest(path)


class SyntheticRunner:
    """Exact argument forwarding is tested; real runner validates full rosters."""
    def __init__(self):
        self.roles = {"role": "authenticated_outer", "targets": list(range(56))}
        self.nested = {"role": "authenticated_inner", "fit": 28, "held": 14}
        self.contract_calls = 0
        self.summary_calls = []
        self.audit_calls = []
        self.equal_calls = []

    def equal(self, left, right, label):
        self.equal_calls.append(label)
        if json.dumps(left, sort_keys=True, allow_nan=False) != json.dumps(right, sort_keys=True, allow_nan=False):
            raise ValueError(label)

    def validate_contract(self, path, sha):
        self.contract_calls += 1
        contract = check.evidence.read_json(path, sha)
        if contract["outer_roles"] != self.roles or contract["nested_roles"] != self.nested:
            raise ValueError("Synthetic authenticated role metadata mismatch")
        return contract, self.roles, self.nested

    def verify_summary(self, summary, arm, contract_sha=None, *, contract=None, roles=None, nested=None):
        self.summary_calls.append(arm)
        assert roles is self.roles and nested is self.nested
        assert contract is not None
        if summary["diagnostic_contract_sha256"] != contract_sha:
            raise ValueError("Contract mismatch")
        if summary["outer_roster"] != roles or summary["inner_roster"] != nested:
            raise ValueError("Synthetic exact outer/inner role roster mismatch")

    def verify_audit(self, audit, arm):
        self.audit_calls.append(arm)
        if not isinstance(audit, dict) or not all(audit.get(key) is True for key in check.AUDIT_TRUE):
            raise ValueError("Audit failed")
        expected = {"ridge_refits": 0, "inner_candidates_replayed": 0 if arm == "control" else 24,
            "out_of_fold_groups": 398, "out_of_fold_source_targets": 112}
        if any(type(audit.get(key)) is not int or audit[key] != value for key, value in expected.items()):
            raise ValueError("Audit count failed")

    def screen(self, summaries):
        assert set(summaries) == set(check.ARMS)
        rows = []
        for source in sorted(check.SOURCES):
            values = {arm: next(row["metrics"] for row in report["source_metrics"] if row["source"] == source)
                for arm, report in summaries.items()}
            conditioning = values["true"]["target_pooled_mse"] <= .99 * values["control"]["target_pooled_mse"]
            distribution = values["true"]["model_mmd2"] <= values["control"]["model_mmd2"]
            rows.append({"source": source, "passed": conditioning and distribution,
                "conditioning_passed": conditioning, "distribution_passed": distribution, "safety_passed": True,
                "checks": {"primary": conditioning, "distribution": distribution, "safety": True}})
        return {"policy": {"synthetic_threshold": .01}, "sources": rows,
            **{key: all(row[key] for row in rows) for key in check.DECISION_FLAGS},
            "automatic_promotion": False, "leaderboard_improvement_claimed": False}


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    runner = SyntheticRunner()
    monkeypatch.setattr(check, "get_runner", lambda: runner)
    policy = {"synthetic_nested_model": True}
    contract = {"schema": check.SUITE_SCHEMA, "stage": "validation", "cpu_only": True,
        "wandb_mode": "offline", "policy": policy, "screen": {"synthetic_threshold": .01},
        "outer_roles": runner.roles, "nested_roles": runner.nested}
    cp = tmp_path / "contract.json"; cs = write(cp, contract)
    runtime = {"hostname": "cbsuvlaminck3.example", "device": "cpu"}
    suite = {"schema": check.SUITE_SCHEMA, "completed": True, "completed_at": NOW.isoformat(),
        "diagnostic_contract_path": str(cp), "diagnostic_contract_sha256": cs,
        "runtime": runtime, "tracking_events": 207, "tracking_errors": 0, "runs": [], "source_metrics": {},
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
            "outer_roster": copy.deepcopy(runner.roles), "inner_roster": copy.deepcopy(runner.nested),
            "hyperparameter_selection": arm != "control", "inner_selection_performed": arm != "control",
            "selection_scope": "inner_only_disjoint_reference_rows",
            "exact_ntc_identity_verified": True, "exact_zero_effect_identity_verified": True,
            "prior_public_development_exposure_acknowledged": True,
            **dict.fromkeys(check.FALSE_SUMMARY_FLAGS, False), **pins,
            "source_metrics": [{"source": source, "metrics": {"target_pooled_mse": .8 if arm == "true" else 1.0,
                "model_mmd2": .4 if arm == "true" else .5, "negative_fraction": 0.0}} for source in sorted(check.SOURCES)]}
        sp = parent / "summary.json"; ss = write(sp, summary)
        events = check.EVENTS_BY_ARM[arm]
        tracking = {"schema": "vcc-training-tracking-v1", "run_id": f"synthetic_{index}", "stage": "validation",
            "mode": "offline", "cloud_synced": False, "execution_status": "completed", "training_exit_code": 0,
            "event_count": events, "tracking_errors": 0, "scientific_quality_inferred_from_exit": False,
            "config": {"diagnostic_contract_sha256": cs}}
        tp = parent / "tracking.json"; ts = write(tp, tracking)
        suite["runs"].append({"arm": arm, "summary_path": str(sp), "summary_sha256": ss,
            "tracking_path": str(tp), "tracking_sha256": ts, "run_id": tracking["run_id"],
            "tracking_events": events, "tracking_errors": 0, "cloud_synced": False,
            "artifact_audit": {**dict.fromkeys(check.AUDIT_TRUE, True), "ridge_refits": 0,
                "inner_candidates_replayed": 0 if arm == "control" else 24,
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
        "tracking_events": 207, "tracking_errors": 0, "cloud_synced": False, "promoted": False, "submission_performed": False})


def evaluate(data):
    return check.evaluate(data["path"], data["sha"], now=NOW)


def reseal_summary(data, arm="true", *, copy_metrics=False):
    row = next(row for row in data["suite"]["runs"] if row["arm"] == arm)
    row["summary_sha256"] = write(Path(row["summary_path"]), data["summaries"][arm])
    if copy_metrics:
        data["suite"]["source_metrics"][arm] = copy.deepcopy(data["summaries"][arm]["source_metrics"])
    seal(data)


def test_complete_seven_arm_nested_evidence_promoter_only(fixture):
    report = evaluate(fixture)
    assert report["suite_evidence_valid"] and report["scientific_gate_passed"] and report["promoter_ablation_ready"]
    assert report["artifact_bytes_verified"] and not report["feng_posttraining_ready"] and not report["submission_ready"]
    runner = fixture["runner"]
    assert runner.contract_calls == 1 and set(runner.summary_calls) == set(check.ARMS)
    assert set(runner.audit_calls) == set(check.ARMS) and len(runner.audit_calls) == 7
    assert "recomputed_decision_mismatch" in runner.equal_calls
    assert not any(report[key] for key in ("credentials_read", "network_used", "raw_arrays_decoded", "automatic_training",
        "automatic_submission", "online_notifications", "prior_watchers_modified", "external_eligibility_bundle_supported"))
    assert sum(check.EVENTS_BY_ARM.values()) == 207


def test_failed_science_keeps_metadata_valid_and_does_not_hash_arrays(fixture, monkeypatch):
    fixture["summaries"]["true"]["source_metrics"][0]["metrics"]["target_pooled_mse"] = 1.1
    fixture["suite"]["decision"] = fixture["runner"].screen(fixture["summaries"])
    reseal_summary(fixture, copy_metrics=True)
    original = check.evidence.digest
    def digest(path):
        assert Path(path).suffix != ".npz", "Failed gate must not read model or output bytes"
        return original(path)
    monkeypatch.setattr(check.evidence, "digest", digest)
    report = evaluate(fixture)
    assert report["suite_evidence_valid"] and not report["scientific_gate_passed"]
    assert not report["promoter_ablation_ready"] and not report["artifact_bytes_verified"]


@pytest.mark.parametrize("mutation", ["schema", "device", "host", "bool_decision", "source_bool", "missing_source",
    "duplicate_source", "missing_arm", "duplicate_run", "event_count", "pretraining", "copied_metric", "copied_metric_type",
    "promotion", "decision_aggregate", "decision_category"])
def test_rehashed_suite_mutations_fail_closed(fixture, mutation):
    suite = fixture["suite"]
    if mutation == "schema": suite["schema"] = "public-target-context-execution-v5"
    elif mutation == "device": suite["runtime"]["device"] = "cuda"
    elif mutation == "host": suite["runtime"]["hostname"] = "aida-login"
    elif mutation == "bool_decision": suite["decision"]["passed"] = 1
    elif mutation == "source_bool": suite["decision"]["sources"][0]["checks"]["safety"] = 1
    elif mutation == "missing_source": suite["decision"]["sources"].pop()
    elif mutation == "duplicate_source": suite["decision"]["sources"][1] = copy.deepcopy(suite["decision"]["sources"][0])
    elif mutation == "missing_arm": suite["runs"].pop()
    elif mutation == "duplicate_run": suite["runs"][1]["run_id"] = suite["runs"][0]["run_id"]
    elif mutation == "event_count": suite["tracking_events"] = 63
    elif mutation == "pretraining": suite["pretraining_performed"] = True
    elif mutation == "copied_metric": suite["source_metrics"]["true"][0]["metrics"]["target_pooled_mse"] = .1
    elif mutation == "copied_metric_type": suite["source_metrics"]["true"][0]["metrics"]["negative_fraction"] = False
    elif mutation == "promotion": suite["decision"]["leaderboard_improvement_claimed"] = True
    elif mutation == "decision_aggregate": suite["decision"]["passed"] = False
    elif mutation == "decision_category": suite["decision"]["sources"][0]["conditioning_passed"] = False
    seal(fixture)
    assert not evaluate(fixture)["suite_evidence_valid"]


@pytest.mark.parametrize("mutation", ["outer_roster", "inner_roster", "device", "host", "stage", "provenance",
    "outer_fit", "outer_selection", "emitter", "selection_flag", "selection_type", "inner_flag", "scope"])
def test_rehashed_summary_roles_and_selection_mutations_fail(fixture, mutation):
    summary = fixture["summaries"]["true"]
    if mutation == "outer_roster": summary["outer_roster"]["targets"].pop()
    elif mutation == "inner_roster": summary["inner_roster"]["fit"] = 42
    elif mutation == "device": summary["runtime"]["device"] = "cuda"
    elif mutation == "host": summary["runtime"]["hostname"] = "anvil-login"
    elif mutation == "stage": summary["stage"] = "pretrain"
    elif mutation == "provenance": summary["provenance"]["diagnostic_contract_sha256"] = "c" * 64
    elif mutation == "outer_fit": summary["outer_evaluation_roles_used_for_fitting"] = True
    elif mutation == "outer_selection": summary["outer_response_selection"] = True
    elif mutation == "emitter": summary["count_emitter"] = True
    elif mutation == "selection_flag": summary["hyperparameter_selection"] = False
    elif mutation == "selection_type": summary["hyperparameter_selection"] = 1
    elif mutation == "inner_flag": summary["inner_selection_performed"] = False
    elif mutation == "scope": summary["selection_scope"] = "outer_response"
    reseal_summary(fixture)
    report = evaluate(fixture)
    assert not report["suite_evidence_valid"] and not report["submission_ready"]


@pytest.mark.parametrize("key", ["hyperparameter_selection", "inner_selection_performed"])
def test_control_cannot_claim_selection(fixture, key):
    fixture["summaries"]["control"][key] = True
    reseal_summary(fixture, "control")
    assert not evaluate(fixture)["suite_evidence_valid"]


@pytest.mark.parametrize("copy_metrics", [False, True])
def test_rehashed_changed_metrics_cannot_keep_old_pass(fixture, copy_metrics):
    fixture["summaries"]["true"]["source_metrics"][0]["metrics"]["target_pooled_mse"] = 2.0
    reseal_summary(fixture, copy_metrics=copy_metrics)
    report = evaluate(fixture)
    assert not report["suite_evidence_valid"]
    assert report["reasons"]["promoter"] == ("recomputed_decision_mismatch" if copy_metrics else "copied_source_metrics_mismatch")


@pytest.mark.parametrize("arm", ["control", "true"])
@pytest.mark.parametrize("key,value", [("training_exit_code", False), ("mode", "online"), ("event_count", 8),
    ("tracking_errors", 1), ("scientific_quality_inferred_from_exit", True), ("run_id", "different")])
def test_rehashed_tracking_mutations_fail(fixture, arm, key, value):
    row = next(row for row in fixture["suite"]["runs"] if row["arm"] == arm)
    path = Path(row["tracking_path"]); receipt = check.evidence.read_json(path)
    receipt[key] = value; row["tracking_sha256"] = write(path, receipt)
    seal(fixture)
    assert not evaluate(fixture)["suite_evidence_valid"]


@pytest.mark.parametrize("arm,events", [("control", 33), ("true", 9), ("true", 33.0)])
def test_swapped_or_untyped_event_counts_fail_even_when_copied(fixture, arm, events):
    row = next(row for row in fixture["suite"]["runs"] if row["arm"] == arm)
    path = Path(row["tracking_path"]); receipt = check.evidence.read_json(path)
    receipt["event_count"] = events; row["tracking_events"] = events
    row["tracking_sha256"] = write(path, receipt); seal(fixture)
    assert not evaluate(fixture)["suite_evidence_valid"]


@pytest.mark.parametrize("key", check.AUDIT_TRUE)
def test_each_replay_assertion_required(fixture, key):
    del fixture["suite"]["runs"][1]["artifact_audit"][key]
    seal(fixture)
    assert evaluate(fixture)["reasons"]["promoter"] == "artifact_replay_not_passed"


@pytest.mark.parametrize("arm,key,value", [("control", "inner_candidates_replayed", 24),
    ("true", "inner_candidates_replayed", 0), ("true", "inner_candidates_replayed", 24.0),
    ("true", "ridge_refits", False), ("true", "ridge_refits", 1),
    ("true", "out_of_fold_groups", 397), ("true", "out_of_fold_source_targets", 111)])
def test_exact_typed_replay_counts(fixture, arm, key, value):
    row = next(row for row in fixture["suite"]["runs"] if row["arm"] == arm)
    row["artifact_audit"][key] = value; seal(fixture)
    assert evaluate(fixture)["reasons"]["promoter"] == "artifact_replay_not_passed"


@pytest.mark.parametrize("name", ["weights", "predictions"])
def test_changed_opaque_artifacts_block_passed_gate(fixture, name):
    (fixture["root"] / "true" / (name + ".npz")).write_bytes(b"tampered opaque bytes")
    report = evaluate(fixture)
    assert not report["promoter_ablation_ready"] and report["reasons"]["promoter"] == "evidence_hash_mismatch"


@pytest.mark.parametrize("hours", [-169, 1])
def test_stale_or_future_completion_rejected(fixture, hours):
    fixture["suite"]["completed_at"] = (NOW + timedelta(hours=hours)).isoformat(); seal(fixture)
    assert not evaluate(fixture)["suite_evidence_valid"]


@pytest.mark.parametrize("key,value", [("completed", 1), ("tracking_events", 63), ("tracking_errors", False),
    ("screen_passed", 1), ("suite_sha256", "e" * 64), ("diagnostic_contract_sha256", "e" * 64),
    ("completed_at", (NOW + timedelta(minutes=1)).isoformat())])
def test_completion_binding_mutations(fixture, key, value):
    path = fixture["root"] / "complete.json"; record = check.evidence.read_json(path)
    record[key] = value; write(path, record)
    assert not evaluate(fixture)["suite_evidence_valid"]


@pytest.mark.parametrize("kind", ["symlink", "fifo", "directory", "duplicate_key", "nonfinite", "oversized"])
def test_nonregular_or_malformed_completion_metadata_fails_without_blocking(fixture, kind):
    path = fixture["root"] / "complete.json"
    original = path.read_bytes(); path.unlink()
    if kind == "symlink":
        target = fixture["root"] / "other.json"; target.write_bytes(original); path.symlink_to(target)
    elif kind == "fifo": os.mkfifo(path)
    elif kind == "directory": path.mkdir()
    elif kind == "duplicate_key": path.write_bytes(b'{"completed":true,"completed":true}')
    elif kind == "nonfinite": path.write_bytes(b'{"x":1e999}')
    elif kind == "oversized": path.write_bytes(b" " * (check.evidence.MAX_JSON_BYTES + 1))
    assert not evaluate(fixture)["suite_evidence_valid"]


def test_cli_fresh_output_only_no_daemon(fixture, monkeypatch, capsys):
    monkeypatch.setattr(check.socket, "gethostname", lambda: "cbsuvlaminck3")
    monkeypatch.setattr(check, "evaluate", lambda *a, **kw: {"schema": check.SCHEMA, "submission_ready": False})
    path = fixture["root"] / "readiness.json"
    args = ["--suite", str(fixture["path"]), "--suite-sha256", fixture["sha"], "--output", str(path)]
    assert check.main(args) == 0 and json.loads(path.read_text())["submission_ready"] is False
    assert path.stat().st_mode & 0o777 == 0o600
    before = path.read_bytes()
    with pytest.raises(SystemExit) as error: check.main(args)
    assert error.value.code == 2 and path.read_bytes() == before
    assert "submission_ready" in capsys.readouterr().out


def test_cli_refuses_login_host_before_reads_or_writes(fixture, monkeypatch):
    monkeypatch.setattr(check.socket, "gethostname", lambda: "login")
    monkeypatch.setattr(check, "evaluate", lambda *a, **kw: pytest.fail("Evidence read attempted on login host"))
    with pytest.raises(SystemExit):
        check.main(["--suite", str(fixture["path"]), "--suite-sha256", fixture["sha"], "--output", str(fixture["root"] / "new.json")])
    assert not (fixture["root"] / "new.json").exists()
