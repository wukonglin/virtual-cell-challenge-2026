"""One-shot, metadata-only v6 readiness audit; never training or submission.

Authenticated nested roles and summaries are validated by the pinned v6 runner.
All seven replay receipts and all 207 offline tracking events are required.
Opaque model/output bytes are hashed only after the recomputed scientific gate
passes. No array is decoded, and no downstream eligibility bundle is supported:
Feng post-training and challenge submission always remain not ready.
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

SCHEMA = "public-nested-readiness-v6"
SUITE_SCHEMA = "public-nested-regularization-execution-v6"
SUMMARY_SCHEMA = "public-nested-regularization-summary-v6"
HELPER_SHA256 = "02922513329a61f5c8672d6f0de439ff6debae8ad219dfc8c287d9a2cbd7c73f"
ARMS = ("control", "context", "additive", "true", "shuffled_20260921", "shuffled_20260922", "shuffled_20260923")
EVENTS_BY_ARM = {arm: 9 if arm == "control" else 33 for arm in ARMS}
TOTAL_EVENTS = 207
FALSE_SUMMARY_FLAGS = ("count_emitter", "posttraining_performed", "promoted", "submission_performed",
    "H1_treated_read", "RPE1_treated_read", "challenge_treated_used",
    "outer_evaluation_roles_used_for_fitting", "outer_response_selection")
FALSE_SUITE_FLAGS = ("pretraining_performed", "posttraining_performed", "promoted", "submission_performed",
    "new_raw_expression_read", "gpu_used", "download_performed", "cloud_synced")
SOURCES = {"nadig_jurkat", "replogle_k562"}
AUDIT_TRUE = ("passed", "saved_head_replay", "training_prior_replay", "kernel_system_verified",
    "training_only_transforms_verified", "deterministic_donor_replay", "original_row_mapping_verified",
    "pooled_and_batch_metrics_replayed", "training_diagnostics_replayed", "inner_selection_replayed",
    "disjoint_inner_references_verified")
DECISION_FLAGS = ("passed", "conditioning_passed", "distribution_passed", "safety_passed")


def get_runner():
    """Lazy import keeps synthetic metadata tests independent of model fitting."""
    return importlib.import_module("run_public_nested_v6")


def require(condition, code):
    evidence.require(condition, code)


def typed_integer(record, key, expected):
    return type(record.get(key)) is int and record[key] == expected


def typed_equal(runner, actual, expected, code):
    """Use the runner's canonical equality; never Python's bool/int equality."""
    try:
        runner.equal(actual, expected, code)
    except (ValueError, TypeError, OverflowError):
        raise evidence.EvidenceError(code) from None


def runtime_verified(value):
    return (isinstance(value, dict) and value.get("device") == "cpu"
        and isinstance(value.get("hostname"), str)
        and value["hostname"].split(".")[0] in {"cbsuvlaminck3", "cbsuvlaminck6"})


def decision_shape(decision):
    require(isinstance(decision, dict) and all(type(decision.get(key)) is bool for key in DECISION_FLAGS),
        "untyped_scientific_decision")
    rows = decision.get("sources")
    require(isinstance(rows, list) and len(rows) == 2 and all(isinstance(row, dict) for row in rows)
        and {row.get("source") for row in rows} == SOURCES, "missing_source_decisions")
    for row in rows:
        require(all(type(row.get(key)) is bool for key in DECISION_FLAGS)
            and isinstance(row.get("checks"), dict) and bool(row["checks"])
            and all(type(value) is bool for value in row["checks"].values()), "untyped_source_decision")
        require(row["passed"] == all(row["checks"].values())
            and row["passed"] == all(row[key] for key in DECISION_FLAGS[1:]), "inconsistent_source_decision")
    require(all(decision[key] == all(row[key] for row in rows) for key in DECISION_FLAGS)
        and decision["passed"] == all(decision[key] for key in DECISION_FLAGS[1:]), "inconsistent_scientific_decision")
    require(decision.get("automatic_promotion") is False and decision.get("leaderboard_improvement_claimed") is False,
        "unexpected_promotion_claim")


def verify(path, sha, *, now):
    require(evidence.digest(Path(evidence.__file__)) == HELPER_SHA256, "frozen_evidence_helper_changed")
    suite = evidence.read_json(path, sha)
    require(suite.get("schema") == SUITE_SCHEMA and suite.get("completed") is True, "v6_suite_not_complete")
    evidence.fresh(suite.get("completed_at"), now, 168)
    require(runtime_verified(suite.get("runtime")), "suite_runtime_not_approved_cpu")
    require(all(suite.get(key) is False for key in FALSE_SUITE_FLAGS), "forbidden_suite_activity")
    require(typed_integer(suite, "tracking_events", TOTAL_EVENTS) and typed_integer(suite, "tracking_errors", 0),
        "suite_tracking_invalid")
    complete = evidence.read_json(Path(path).with_name("complete.json"))
    require(complete.get("schema") == SUITE_SCHEMA and complete.get("completed") is True
        and complete.get("suite_sha256") == sha and complete.get("completed_at") == suite["completed_at"]
        and complete.get("diagnostic_contract_sha256") == suite.get("diagnostic_contract_sha256")
        and typed_integer(complete, "tracking_events", TOTAL_EVENTS) and typed_integer(complete, "tracking_errors", 0)
        and all(complete.get(key) is False for key in ("cloud_synced", "promoted", "submission_performed")),
        "completion_binding_invalid")
    contract_pin = {"path": suite.get("diagnostic_contract_path"), "sha256": suite.get("diagnostic_contract_sha256")}
    sealed_contract = evidence.binding(contract_pin)
    runner = get_runner()
    contract, roles, nested = runner.validate_contract(Path(contract_pin["path"]), contract_pin["sha256"])
    typed_equal(runner, contract, sealed_contract, "validated_contract_mismatch")
    require(contract.get("schema") == SUITE_SCHEMA and contract.get("stage") == "validation"
        and contract.get("cpu_only") is True and contract.get("wandb_mode") == "offline", "validated_contract_mismatch")
    runs = suite.get("runs")
    require(isinstance(runs, list) and len(runs) == 7 and all(isinstance(run, dict) for run in runs)
        and {run.get("arm") for run in runs} == set(ARMS), "seven_fixed_arms_required")
    require(all(isinstance(run.get("run_id"), str) and run["run_id"] for run in runs)
        and len({run["run_id"] for run in runs}) == 7, "distinct_tracking_runs_required")
    require(isinstance(suite.get("source_metrics"), dict) and set(suite["source_metrics"]) == set(ARMS),
        "copied_source_metrics_incomplete")
    summaries = {}
    for run in runs:
        arm = run["arm"]
        events = EVENTS_BY_ARM[arm]
        require(typed_integer(run, "tracking_events", events) and typed_integer(run, "tracking_errors", 0)
            and run.get("cloud_synced") is False, "run_tracking_invalid")
        try:
            runner.verify_audit(run.get("artifact_audit"), arm)
        except (ValueError, TypeError, AttributeError, KeyError):
            raise evidence.EvidenceError("artifact_replay_not_passed") from None
        summary = evidence.binding({"path": run.get("summary_path"), "sha256": run.get("summary_sha256")})
        # Exact outer and nested role rosters must not be omitted here.
        runner.verify_summary(summary, arm, contract_pin["sha256"], contract=contract, roles=roles, nested=nested)
        require(summary.get("schema") == SUMMARY_SCHEMA and summary.get("arm") == arm
            and summary.get("stage") == "validation" and summary.get("status") == "completed_unpromoted_diagnostic"
            and summary.get("diagnostic_contract_sha256") == contract_pin["sha256"]
            and summary.get("provenance", {}).get("diagnostic_contract_sha256") == contract_pin["sha256"],
            "summary_provenance_invalid")
        typed_equal(runner, summary.get("policy"), contract.get("policy"), "summary_policy_invalid")
        require(runtime_verified(summary.get("runtime")), "summary_runtime_not_approved_cpu")
        require(all(summary.get(key) is False for key in FALSE_SUMMARY_FLAGS), "forbidden_summary_activity")
        require(summary.get("hyperparameter_selection") is (arm != "control")
            and summary.get("inner_selection_performed") is (arm != "control")
            and summary.get("selection_scope") == "inner_only_disjoint_reference_rows", "nested_selection_flags_invalid")
        evidence.all_true(summary, ("exact_ntc_identity_verified", "exact_zero_effect_identity_verified",
            "prior_public_development_exposure_acknowledged"), "summary_safety_invalid")
        typed_equal(runner, suite["source_metrics"][arm], summary.get("source_metrics"), "copied_source_metrics_mismatch")
        tracking = evidence.binding({"path": run.get("tracking_path"), "sha256": run.get("tracking_sha256")})
        require(tracking.get("schema") == "vcc-training-tracking-v1" and tracking.get("stage") == "validation"
            and tracking.get("mode") == "offline" and tracking.get("cloud_synced") is False
            and tracking.get("execution_status") == "completed" and typed_integer(tracking, "training_exit_code", 0)
            and typed_integer(tracking, "event_count", events) and typed_integer(tracking, "tracking_errors", 0)
            and tracking.get("run_id") == run["run_id"] and tracking.get("scientific_quality_inferred_from_exit") is False
            and tracking.get("config", {}).get("diagnostic_contract_sha256") == contract_pin["sha256"],
            "tracking_receipt_invalid")
        summaries[arm] = summary
    decision = runner.screen(summaries)
    decision_shape(decision)
    decision_shape(suite.get("decision"))
    typed_equal(runner, suite["decision"], decision, "recomputed_decision_mismatch")
    typed_equal(runner, decision.get("policy"), contract.get("screen"), "recomputed_decision_policy_mismatch")
    require(type(complete.get("screen_passed")) is bool and complete["screen_passed"] == decision["passed"],
        "completion_decision_mismatch")
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
        "prior_watchers_modified": False, "external_eligibility_bundle_supported": False}
    try:
        decision = verify(suite_path, suite_sha, now=now)
        report.update(suite_evidence_valid=True, scientific_gate_passed=decision["passed"],
            promoter_ablation_ready=decision["passed"], artifact_bytes_verified=decision["passed"], decision=decision)
        if not decision["passed"]:
            report["reasons"]["promoter"] = "scientific_or_distribution_or_safety_gate_failed"
    except Exception as error:
        # Imported validators may mention paths; report only safe typed codes.
        report["reasons"]["promoter"] = str(error) if isinstance(error, evidence.EvidenceError) else "v6_validation_failed"
    return report


def write_new(path, report):
    payload = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


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
    return 0  # Successful recording is not scientific success or authorization.


if __name__ == "__main__":
    raise SystemExit(main())
