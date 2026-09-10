"""One-shot v5 evidence audit; never a daemon, model trainer or submitter.

The frozen v4 watcher keeps its own evidence pins. This separate checker imports
the v5 runner's contract/summary validators and recomputes its decision from
authenticated summaries. No external eligibility bundle is implemented here:
Feng post-training and VCC submission are ALWAYS not ready.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import socket

import public_training_readiness as evidence

SCHEMA = "public-target-context-readiness-v5"
SUITE_SCHEMA = "public-target-context-execution-v5"
SUMMARY_SCHEMA = "public-target-context-summary-v5"
HELPER_SHA256 = "02922513329a61f5c8672d6f0de439ff6debae8ad219dfc8c287d9a2cbd7c73f"
ARMS = ("control", "context", "additive", "true", "shuffled_20260921", "shuffled_20260922", "shuffled_20260923")
FALSE_SUMMARY_FLAGS = ("count_emitter", "posttraining_performed", "promoted", "submission_performed",
    "hyperparameter_selection", "H1_treated_read", "RPE1_treated_read", "challenge_treated_used", "held_roles_used_for_fitting")
FALSE_SUITE_FLAGS = ("pretraining_performed", "posttraining_performed", "promoted", "submission_performed",
    "new_raw_expression_read", "gpu_used", "download_performed", "cloud_synced")
SOURCES = {"nadig_jurkat", "replogle_k562"}
AUDIT_TRUE = ("passed", "saved_head_replay", "training_prior_replay", "kernel_system_verified",
    "training_only_transforms_verified", "deterministic_donor_replay", "original_row_mapping_verified",
    "pooled_and_batch_metrics_replayed", "training_diagnostics_replayed")


def get_runner():
    """Lazy import permits isolated synthetic tests before model code is staged."""
    return importlib.import_module("run_public_target_context_v5")


def require(condition, code):
    evidence.require(condition, code)


def typed_integer(record, key, expected):
    return type(record.get(key)) is int and record[key] == expected


def same_json(left, right):
    """Preserve typed JSON equality, including false versus0 and true versus1."""
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def runtime_verified(value):
    return (isinstance(value, dict) and value.get("device") == "cpu"
        and isinstance(value.get("hostname"), str)
        and value["hostname"].split(".")[0] in {"cbsuvlaminck3", "cbsuvlaminck6"})


def verify(path, sha, *, now):
    require(evidence.digest(Path(evidence.__file__)) == HELPER_SHA256, "frozen_evidence_helper_changed")
    suite = evidence.read_json(path, sha)
    require(suite.get("schema") == SUITE_SCHEMA and suite.get("completed") is True, "v5_suite_not_complete")
    evidence.fresh(suite.get("completed_at"), now, 168)
    require(runtime_verified(suite.get("runtime")), "suite_runtime_not_approved_cpu")
    require(all(suite.get(key) is False for key in FALSE_SUITE_FLAGS), "forbidden_suite_activity")
    require(typed_integer(suite, "tracking_events", 63) and typed_integer(suite, "tracking_errors", 0), "suite_tracking_invalid")
    complete = evidence.read_json(Path(path).with_name("complete.json"))
    require(complete.get("schema") == SUITE_SCHEMA and complete.get("completed") is True
        and complete.get("suite_sha256") == sha and complete.get("completed_at") == suite["completed_at"]
        and complete.get("diagnostic_contract_sha256") == suite.get("diagnostic_contract_sha256")
        and typed_integer(complete, "tracking_events", 63) and typed_integer(complete, "tracking_errors", 0)
        and all(complete.get(key) is False for key in ("cloud_synced", "promoted", "submission_performed")), "completion_binding_invalid")
    contract_pin = {"path": suite.get("diagnostic_contract_path"), "sha256": suite.get("diagnostic_contract_sha256")}
    sealed_contract = evidence.binding(contract_pin)
    runner = get_runner()
    contract, _ = runner.validate_contract(Path(contract_pin["path"]), contract_pin["sha256"])
    require(contract == sealed_contract and contract.get("schema") == SUITE_SCHEMA
        and contract.get("stage") == "validation" and contract.get("cpu_only") is True
        and contract.get("wandb_mode") == "offline", "validated_contract_mismatch")
    runs = suite.get("runs")
    require(isinstance(runs, list) and len(runs) == 7 and all(isinstance(run, dict) for run in runs)
        and {run.get("arm") for run in runs} == set(ARMS), "seven_fixed_arms_required")
    require(all(isinstance(run.get("run_id"), str) and run["run_id"] for run in runs)
        and len({run["run_id"] for run in runs}) == 7, "distinct_tracking_runs_required")
    require(isinstance(suite.get("source_metrics"), dict) and set(suite["source_metrics"]) == set(ARMS), "copied_source_metrics_incomplete")
    summaries = {}
    for run in runs:
        arm = run["arm"]
        require(typed_integer(run, "tracking_events", 9) and typed_integer(run, "tracking_errors", 0)
            and run.get("cloud_synced") is False, "run_tracking_invalid")
        audit = run.get("artifact_audit")
        require(isinstance(audit, dict) and all(audit.get(key) is True for key in AUDIT_TRUE)
            and typed_integer(audit, "ridge_refits", 0) and typed_integer(audit, "out_of_fold_groups", 398)
            and typed_integer(audit, "out_of_fold_source_targets", 112), "artifact_replay_not_passed")
        summary = evidence.binding({"path": run.get("summary_path"), "sha256": run.get("summary_sha256")})
        runner.verify_summary(summary, arm, contract_pin["sha256"])
        require(summary.get("schema") == SUMMARY_SCHEMA and summary.get("arm") == arm
            and summary.get("stage") == "validation" and summary.get("status") == "completed_unpromoted_diagnostic"
            and summary.get("policy") == contract.get("policy")
            and summary.get("diagnostic_contract_sha256") == contract_pin["sha256"]
            and summary.get("provenance", {}).get("diagnostic_contract_sha256") == contract_pin["sha256"], "summary_provenance_invalid")
        require(runtime_verified(summary.get("runtime")), "summary_runtime_not_approved_cpu")
        require(all(summary.get(key) is False for key in FALSE_SUMMARY_FLAGS), "forbidden_summary_activity")
        evidence.all_true(summary, ("exact_ntc_identity_verified", "exact_zero_effect_identity_verified",
            "prior_public_development_exposure_acknowledged"), "summary_safety_invalid")
        require(same_json(suite["source_metrics"][arm], summary.get("source_metrics")), "copied_source_metrics_mismatch")
        tracking = evidence.binding({"path": run.get("tracking_path"), "sha256": run.get("tracking_sha256")})
        require(tracking.get("schema") == "vcc-training-tracking-v1" and tracking.get("stage") == "validation"
            and tracking.get("mode") == "offline" and tracking.get("cloud_synced") is False
            and tracking.get("execution_status") == "completed" and typed_integer(tracking, "training_exit_code", 0)
            and typed_integer(tracking, "event_count", 9) and typed_integer(tracking, "tracking_errors", 0)
            and tracking.get("run_id") == run["run_id"] and tracking.get("scientific_quality_inferred_from_exit") is False
            and tracking.get("config", {}).get("diagnostic_contract_sha256") == contract_pin["sha256"], "tracking_receipt_invalid")
        summaries[arm] = summary
    decision = runner.screen(summaries)
    require(isinstance(decision, dict) and type(decision.get("passed")) is bool
        and all(type(decision.get(key)) is bool for key in ("conditioning_passed", "distribution_passed", "safety_passed")), "untyped_recomputed_decision")
    # Equality alone treats1 == True; reject untyped stored decisions explicitly.
    stored = suite.get("decision")
    require(isinstance(stored, dict) and all(type(stored.get(key)) is bool for key in
        ("passed", "conditioning_passed", "distribution_passed", "safety_passed")), "untyped_stored_decision")
    for row in stored.get("sources", []):
        require(isinstance(row, dict) and type(row.get("passed")) is bool
            and all(type(row.get(key)) is bool for key in ("conditioning_passed", "distribution_passed", "safety_passed"))
            and isinstance(row.get("checks"), dict) and bool(row["checks"])
            and all(type(value) is bool for value in row["checks"].values()), "untyped_stored_source_decision")
    require(same_json(stored, decision) and same_json(decision.get("policy"), contract.get("screen")), "recomputed_decision_mismatch")
    require(type(complete.get("screen_passed")) is bool and complete["screen_passed"] == decision["passed"], "completion_decision_mismatch")
    if decision["passed"]:
        for run in runs:
            summary = summaries[run["arm"]]
            for name in ("weights", "predictions"):
                evidence.binding({"path": str(Path(run["summary_path"]).with_name(name + ".npz")),
                    "sha256": summary.get(name + "_sha256")}, json_document=False)
    return decision


def evaluate(suite_path, suite_sha, *, now=None):
    now = now or datetime.now(timezone.utc)
    report = {"schema": SCHEMA, "checked_at": now.isoformat(), "suite_sha256": suite_sha,
        "checker_sha256": evidence.digest(Path(__file__)), "frozen_helper_sha256": HELPER_SHA256,
        "suite_evidence_valid": False, "scientific_gate_passed": False, "promoter_ablation_ready": False,
        "feng_posttraining_ready": False, "submission_ready": False, "artifact_bytes_verified": False,
        "reasons": {"feng": "no_authenticated_compatible_trainable_public_flow_parent_or_feng_posttraining_contract",
            "submission": "no_validated_full_gene_raw_count_emitter_or_official_prep_receipt"},
        "credentials_read": False, "network_used": False, "raw_arrays_decoded": False,
        "automatic_training": False, "automatic_submission": False, "online_notifications": False,
        "v4_watcher_modified": False, "external_eligibility_bundle_supported": False}
    try:
        decision = verify(suite_path, suite_sha, now=now)
        report.update(suite_evidence_valid=True, scientific_gate_passed=decision["passed"],
            promoter_ablation_ready=decision["passed"], artifact_bytes_verified=decision["passed"],
            decision=decision)
        if not decision["passed"]:
            report["reasons"]["promoter"] = "scientific_or_distribution_or_safety_gate_failed"
    except Exception as error:
        # Imported validators can include file paths in error strings. Keep the
        # outward reason typed/non-sensitive rather than serializing exceptions.
        report["reasons"]["promoter"] = str(error) if isinstance(error, evidence.EvidenceError) else "v5_validation_failed"
    return report


def write_new(path, report):
    payload = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--suite-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if socket.gethostname().split(".")[0] not in {"cbsuvlaminck3", "cbsuvlaminck6"}:
        parser.exit(2, "Readiness artifact checks require an approved BioHPC compute host.\n")
    report = evaluate(args.suite, args.suite_sha256)
    try:
        write_new(args.output, report)
    except OSError:
        parser.exit(2, "A writable fresh output path is required; nothing is overwritten.\n")
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    # Successful status recording is not scientific success or authorization.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
