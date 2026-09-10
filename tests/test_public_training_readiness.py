"""Synthetic readiness receipts only; no credentials, network, model or real data."""
from __future__ import annotations

from argparse import Namespace
import copy
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import public_training_readiness as ready

NOW = datetime(2026, 9, 10, 12, tzinfo=timezone.utc)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False))
    return {"path": str(path), "sha256": ready.digest(path)}


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    code = tmp_path / "code"
    code.mkdir(); (code / "synthetic.py").write_text("# synthetic frozen code\n")
    monkeypatch.setattr(ready, "SCRIPT_DIR", code)
    protocol = tmp_path / "protocol.md"; protocol.write_text("Synthetic fixed protocol")
    prior = write(tmp_path / "prior.json", {"schema": "synthetic-prior"})
    policy = {"mode": "synthetic-fixed"}
    screen = {"minimum_relative_improvement": 0.01}
    contract = write(tmp_path / "contract.json", {"schema": ready.SUITE_SCHEMA, "stage": "validation",
        "policy": policy, "screen": screen, "protocol": {"path": str(protocol), "sha256": ready.digest(protocol)},
        "prior_v3_contract": prior, "code_sha256": {"synthetic.py": ready.digest(code / "synthetic.py")}})
    sources = [{"source": source, "passed": True, "conditioning_passed": True,
        "distribution_passed": True, "safety_passed": True,
        "checks": {"conditioning": True, "distribution": True, "safety": True}} for source in sorted(ready.SOURCES)]
    suite = {"schema": ready.SUITE_SCHEMA, "completed": True, "completed_at": NOW.isoformat(),
        "diagnostic_contract_path": contract["path"], "diagnostic_contract_sha256": contract["sha256"],
        "decision": {"passed": True, "conditioning_passed": True, "distribution_passed": True,
            "safety_passed": True, "sources": sources, "policy": screen},
        "runs": [], "tracking_events": 54, "tracking_errors": 0, "cloud_synced": False,
        "promoted": False, "submission_performed": False}
    for index, arm in enumerate(ready.ARMS):
        (tmp_path / arm).mkdir()
        artifact_pins = {}
        for name in ("weights", "predictions"):
            artifact = tmp_path / arm / (name + ".npz")
            artifact.write_bytes(("Synthetic opaque " + name).encode())
            artifact_pins[name + "_sha256"] = ready.digest(artifact)
        summary = {"schema": ready.SUMMARY_SCHEMA, "status": "completed_unpromoted_diagnostic",
            "stage": "validation", "arm": arm, "policy": policy,
            "diagnostic_contract_sha256": contract["sha256"], "exact_ntc_identity_verified": True,
            "exact_zero_effect_identity_verified": True, "prior_public_development_exposure_acknowledged": True,
            **dict.fromkeys(("count_emitter", "posttraining_performed", "promoted", "submission_performed",
                "hyperparameter_selection", "H1_treated_read", "RPE1_treated_read", "challenge_treated_used",
            "held_roles_used_for_fitting"), False), **artifact_pins}
        report = write(tmp_path / arm / "summary.json", summary)
        tracking = write(tmp_path / arm / "tracking.json", {"schema": "vcc-training-tracking-v1",
            "stage": "validation", "mode": "offline", "cloud_synced": False, "execution_status": "completed",
            "training_exit_code": 0, "event_count": 9, "tracking_errors": 0, "run_id": f"synthetic_{index}",
            "scientific_quality_inferred_from_exit": False})
        suite["runs"].append({"arm": arm, "summary_path": report["path"], "summary_sha256": report["sha256"],
            "tracking_path": tracking["path"], "tracking_sha256": tracking["sha256"], "run_id": f"synthetic_{index}",
            "tracking_events": 9, "tracking_errors": 0, "artifact_audit": {"passed": True}})
    data = {"root": tmp_path, "suite": suite}
    seal(data)
    return data


def seal(data):
    suite = data["suite"]
    record = write(data["root"] / "suite.json", suite)
    data["path"], data["sha"] = Path(record["path"]), record["sha256"]
    write(data["root"] / "complete.json", {"schema": ready.SUITE_SCHEMA, "completed": True,
        "suite_sha256": data["sha"], "diagnostic_contract_sha256": suite["diagnostic_contract_sha256"],
        "completed_at": suite["completed_at"], "screen_passed": suite["decision"]["passed"]})


def evaluate(data, **kwargs):
    return ready.evaluate(data["path"], data["sha"], now=NOW, **kwargs)


def test_completed_passing_suite_only_enables_promoter(evidence):
    state = evaluate(evidence)
    assert state["suite_evidence_valid"] and state["scientific_gate_passed"] and state["promoter_ablation_ready"]
    assert not state["feng_posttraining_ready"] and not state["submission_ready"]
    assert state["reasons"] == {"feng": "optional_evidence_not_registered", "submission": "optional_evidence_not_registered"}
    assert not state["credentials_read"] and not state["external_actions_performed"]


@pytest.mark.parametrize("category", ["conditioning", "distribution", "safety"])
def test_completed_optimizer_or_execution_does_not_override_failed_science(evidence, category):
    decision = evidence["suite"]["decision"]
    decision["passed"] = decision[category + "_passed"] = False
    decision["sources"][0]["passed"] = decision["sources"][0][category + "_passed"] = False
    decision["sources"][0]["checks"][category] = False
    seal(evidence)
    state = evaluate(evidence)
    assert state["suite_evidence_valid"] and not state["scientific_gate_passed"]
    assert not any(state[key] for key in ("promoter_ablation_ready", "feng_posttraining_ready", "submission_ready"))


@pytest.mark.parametrize("mutation", ["bool_number", "source_bool_number", "empty_checks", "false_check",
    "source_category", "missing_arm", "duplicate_arm", "audit_fail", "tracking_fail", "external_action"])
def test_resealed_malformed_decisions_fail_closed(evidence, mutation):
    suite = evidence["suite"]
    if mutation == "bool_number": suite["decision"]["passed"] = 1
    elif mutation == "source_bool_number": suite["decision"]["sources"][0]["passed"] = 1
    elif mutation == "empty_checks": suite["decision"]["sources"][0]["checks"] = {}
    elif mutation == "false_check": suite["decision"]["sources"][0]["checks"]["safety"] = False
    elif mutation == "source_category": suite["decision"]["sources"][0]["distribution_passed"] = False
    elif mutation == "missing_arm": suite["runs"].pop()
    elif mutation == "duplicate_arm": suite["runs"][1] = copy.deepcopy(suite["runs"][0])
    elif mutation == "audit_fail": suite["runs"][0]["artifact_audit"]["passed"] = False
    elif mutation == "tracking_fail": suite["tracking_errors"] = 1
    elif mutation == "external_action": suite["submission_performed"] = True
    seal(evidence)
    state = evaluate(evidence)
    assert not state["suite_evidence_valid"] and not state["promoter_ablation_ready"]


@pytest.mark.parametrize("kind", ["suite", "summary", "tracking", "protocol", "code", "prior", "weights", "predictions"])
def test_changed_pinned_bytes_fail_closed(evidence, kind):
    paths = {"suite": evidence["path"], "summary": Path(evidence["suite"]["runs"][0]["summary_path"]),
        "tracking": Path(evidence["suite"]["runs"][0]["tracking_path"]), "protocol": evidence["root"] / "protocol.md",
        "code": evidence["root"] / "code/synthetic.py", "prior": evidence["root"] / "prior.json",
        "weights": evidence["root"] / "control/weights.npz", "predictions": evidence["root"] / "control/predictions.npz"}
    with paths[kind].open("a") as stream: stream.write(" ")
    state = evaluate(evidence)
    assert not state["suite_evidence_valid"] and "evidence_hash_mismatch" in state["reasons"].values()


def test_missing_completion_and_retimestamped_receipt_fail_closed(evidence):
    path = evidence["root"] / "complete.json"
    receipt = ready.read_json(path)
    receipt["completed_at"] = (NOW + timedelta(hours=1)).isoformat()
    write(path, receipt)
    assert not evaluate(evidence)["suite_evidence_valid"]
    path.unlink()
    assert not evaluate(evidence)["suite_evidence_valid"]


@pytest.mark.parametrize("hours", [-169, 1])
def test_stale_or_future_scientific_evidence_rejected(evidence, hours):
    evidence["suite"]["completed_at"] = (NOW + timedelta(hours=hours)).isoformat()
    seal(evidence)
    assert "stale_or_future_evidence" in evaluate(evidence)["reasons"].values()


def external_evidence(data):
    root, sha = data["root"] / "optional", data["sha"]
    root.mkdir()
    checkpoint = root / "checkpoint.bin"; checkpoint.write_bytes(b"synthetic checkpoint")
    parent = write(root / "parent.json", {"schema": "public-trainable-parent-readiness-v1", "suite_sha256": sha,
        "completed": True, "trainable": True, "compatible": True, "public_only": True,
        "compatibility": dict.fromkeys(("input_gene_axis_verified", "output_gene_axis_verified",
            "conditioning_verified", "representation_verified"), True),
        "checkpoint": {"path": str(checkpoint), "sha256": ready.digest(checkpoint)}})
    source = write(root / "source.json", {"schema": "synthetic-source"})
    split = write(root / "split.json", {"schema": "synthetic-split"})
    feng = write(root / "feng.json", {"schema": "public-feng-posttraining-eligibility-v1", "suite_sha256": sha,
        **dict.fromkeys(("completed", "public_only", "donor_eligible", "batch_eligible", "target_exclusions_verified",
            "matched_controls_verified", "representation_compatible"), True), "source_manifest": source, "split_manifest": split})
    prediction = root / "prediction.h5ad"; prediction.write_bytes(b"synthetic format-validated receipt fixture, not real AnnData")
    package = root / "prediction.vcc"; package.write_bytes(b"synthetic package receipt fixture")
    emitter = write(root / "emitter.json", {"schema": "vcc-submission-emitter-readiness-v1", "suite_sha256": sha,
        "completed": True, "full_output_verified": True, "public_only_provenance": True,
        "count_space_calibration_validated": True, "representation": "raw_counts", "shape": [360000, 18533],
        "contexts": ["A", "B", "C"], "checks": dict.fromkeys(("finite", "nonnegative", "integer_valued",
            "gene_order_verified", "target_panel_verified", "context_labels_verified", "cell_counts_verified", "no_control_cells"), True),
        "prediction": {"path": str(prediction), "sha256": ready.digest(prediction)},
        "package": {"path": str(package), "sha256": ready.digest(package)}})
    dry = write(root / "dryrun.json", {"dry_run": True, "verified_targets": True, "normalization": "counts-preserved",
        "n_cells": 360000, "n_genes": 18533, "cells_per_context": {key: 120000 for key in "ABC"},
        "input": str(prediction), "nnz": 100})
    status = write(root / "status.json", {"schema": "vcc-sanitized-readiness-status-v1", "checked_at": NOW.isoformat(),
        "credential_present": True, "can_submit": True, "quota_available": True, "inflight_submission": False})
    manifest = {"schema": "public-training-readiness-evidence-v1", "suite_sha256": sha, "parent": parent,
        "feng": feng, "emitter": emitter, "official_dryrun": dry, "status": status}
    return root, manifest


def evaluate_external(data, manifest):
    pin = write(data["root"] / "optional_manifest.json", manifest)
    return evaluate(data, evidence_path=pin["path"], evidence_sha=pin["sha256"])


def test_all_typed_pinned_eligibility_receipts_can_enable_future_branches(evidence):
    _, manifest = external_evidence(evidence)
    state = evaluate_external(evidence, manifest)
    assert state["feng_posttraining_ready"] and state["submission_ready"]
    assert state["reasons"] == {} and not state["external_actions_performed"]


@pytest.mark.parametrize("kind,key,value", [("parent", "trainable", False), ("feng", "donor_eligible", False),
    ("feng", "batch_eligible", False), ("emitter", "representation", "log1p_cp10k"),
    ("emitter", "shape", [360000, 512]), ("official_dryrun", "verified_targets", False),
    ("status", "credential_present", False), ("status", "quota_available", False),
    ("status", "checked_at", (NOW - timedelta(hours=2)).isoformat()), ("status", "token", "SYNTHETIC_SECRET")])
def test_ineligible_or_stale_optional_evidence_blocks_its_branch(evidence, kind, key, value):
    _, manifest = external_evidence(evidence)
    pin = manifest[kind]
    receipt = ready.read_json(pin["path"]); receipt[key] = value
    manifest[kind] = write(Path(pin["path"]), receipt)
    state = evaluate_external(evidence, manifest)
    assert state["promoter_ablation_ready"]
    assert not state["feng_posttraining_ready"] if kind in ("parent", "feng") else not state["submission_ready"]
    assert "SYNTHETIC_SECRET" not in json.dumps(state)


def monitor_args(data, command="watch"):
    return Namespace(command=command, suite=data["path"], suite_sha256=data["sha"], evidence_manifest=None,
        evidence_sha256=None, output_dir=data["root"] / "watch", interval_seconds=5,
        max_hours=10/3600, max_evidence_age_hours=168)


def test_watch_bounded_and_emits_only_transitions(evidence, monkeypatch, capsys):
    monkeypatch.setattr(ready.socket, "gethostname", lambda: "cbsuvlaminck3.example")
    monkeypatch.setattr(ready, "evaluate", lambda *a, **kw: {"checked_at": str(clock[0]), "submission_ready": False})
    clock = [0.0]
    def sleep(seconds): clock[0] += seconds
    args = monitor_args(evidence)
    receipt = ready.monitor(args, monotonic=lambda: clock[0], sleep=sleep)
    assert receipt["stop_reason"] == "time_bound" and receipt["checks"] == 3 and receipt["state_changes"] == 1
    assert len((args.output_dir / "states.jsonl").read_text().splitlines()) == 1
    assert len(capsys.readouterr().out.splitlines()) == 1
    assert not receipt["automatic_submission"] and not receipt["online_notifications"]
    assert receipt["new_evidence_requires_restart"]
    run = ready.read_json(args.output_dir / "run.json")
    heartbeat = ready.read_json(args.output_dir / "heartbeat.json")
    assert run["pid"] == os.getpid() and run["hostname"] == "cbsuvlaminck3.example"
    assert heartbeat["checks"] == 3 and heartbeat["state_changes"] == 1
    assert heartbeat["last_checked_at"] == "10.0" and heartbeat["running"] is False
    assert heartbeat["watcher_sha256"] == run["watcher_sha256"] == receipt["watcher_sha256"]


def test_sigterm_stop_writes_final_receipt(evidence, monkeypatch):
    monkeypatch.setattr(ready.socket, "gethostname", lambda: "cbsuvlaminck6")
    monkeypatch.setattr(ready, "evaluate", lambda *a, **kw: {"submission_ready": False})
    receipt = ready.monitor(monitor_args(evidence), stop=lambda: True)
    assert receipt["stop_reason"] == "signal" and receipt["checks"] == 1
    assert (evidence["root"] / "watch/receipt.json").is_file()


@pytest.mark.parametrize("field,value", [("interval_seconds", 4), ("interval_seconds", float("nan")),
    ("max_hours", 25), ("max_hours", 0), ("max_evidence_age_hours", float("inf"))])
def test_unsafe_watch_bounds_rejected_before_writes(evidence, monkeypatch, field, value):
    monkeypatch.setattr(ready.socket, "gethostname", lambda: "cbsuvlaminck3")
    args = monitor_args(evidence); setattr(args, field, value)
    with pytest.raises(ready.EvidenceError): ready.monitor(args)
    assert not args.output_dir.exists()


def test_headnode_and_existing_output_are_refused(evidence, monkeypatch):
    args = monitor_args(evidence)
    monkeypatch.setattr(ready.socket, "gethostname", lambda: "aida-login")
    with pytest.raises(ready.EvidenceError, match="compute_host"): ready.monitor(args)
    monkeypatch.setattr(ready.socket, "gethostname", lambda: "cbsuvlaminck3")
    args.output_dir.mkdir()
    with pytest.raises(ready.EvidenceError, match="fresh_output"): ready.monitor(args)


@pytest.mark.parametrize("payload", ['{"a":1,"a":2}', '{"a":NaN}', '{"a":1e999}', '[]'])
def test_bad_json_rejected(tmp_path, payload):
    path = tmp_path / "bad.json"; path.write_text(payload)
    with pytest.raises(ready.EvidenceError): ready.read_json(path, ready.digest(path))


@pytest.mark.parametrize("kind", ["symlink", "fifo", "directory"])
def test_special_evidence_files_rejected_without_blocking(tmp_path, kind):
    target = tmp_path / "special"
    if kind == "symlink":
        original = tmp_path / "original.json"; original.write_text("{}")
        target.symlink_to(original)
    elif kind == "fifo": os.mkfifo(target)
    else: target.mkdir()
    with pytest.raises(ready.EvidenceError): ready.read_json(target)
    with pytest.raises(ready.EvidenceError): ready.digest(target)


def test_failed_scientific_gate_does_not_read_large_arrays(evidence, monkeypatch):
    suite = evidence["suite"]
    suite["decision"]["passed"] = suite["decision"]["conditioning_passed"] = False
    row = suite["decision"]["sources"][0]
    row["passed"] = row["conditioning_passed"] = row["checks"]["conditioning"] = False
    seal(evidence)
    original = ready.digest
    def digest(path):
        assert Path(path).suffix != ".npz", "Failed gate should not trigger bulk array reads"
        return original(path)
    monkeypatch.setattr(ready, "digest", digest)
    state = evaluate(evidence)
    assert state["suite_evidence_valid"] and not state["promoter_ablation_ready"]
