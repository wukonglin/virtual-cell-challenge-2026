"""Read-only, fixed-evidence readiness checks; never training, upload or notification.

Only this watcher's fresh output directory is written. Inputs are rehashed on
every check. A new suite/evidence version requires a restarted watcher with new
pins. Credential evidence is sanitized externally; this module never resolves
credentials or contacts a service. All readiness branches fail closed.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import stat
import time
import uuid

SCHEMA = "public-training-readiness-v1"
SUITE_SCHEMA = "public-sparse-hurdle-execution-v4"
SUMMARY_SCHEMA = "public-sparse-hurdle-summary-v4"
ARMS = ("control", "context", "true", "shuffled_20260921", "shuffled_20260922", "shuffled_20260923")
SOURCES = {"nadig_jurkat", "replogle_k562"}
SCRIPT_DIR = Path(__file__).resolve().parent
HASH = re.compile(r"[a-f0-9]{64}")
MAX_JSON_BYTES = 16 << 20


class EvidenceError(ValueError):
    """An intentionally non-sensitive evidence failure code."""


def require(value, reason):
    if not value:
        raise EvidenceError(reason)


@contextmanager
def regular_reader(path):
    """No symlink following or FIFO blocking, including a stat/open race."""
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = None
    try:
        require(not Path(path).is_symlink(), "regular_evidence_file_required")
        descriptor = os.open(path, flags)
        require(stat.S_ISREG(os.fstat(descriptor).st_mode), "regular_evidence_file_required")
        stream = os.fdopen(descriptor, "rb")
        descriptor = None
        with stream:
            yield stream
    except OSError:
        raise EvidenceError("missing_unreadable_or_nonregular_evidence") from None
    finally:
        if descriptor is not None:
            os.close(descriptor)


def digest(path):
    value = hashlib.sha256()
    with regular_reader(path) as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path, expected=None):
    with regular_reader(path) as handle:
        payload = handle.read(MAX_JSON_BYTES + 1)
    require(len(payload) <= MAX_JSON_BYTES, "oversized_evidence")
    if expected is not None:
        require(isinstance(expected, str) and HASH.fullmatch(expected), "invalid_sha256")
        require(hashlib.sha256(payload).hexdigest() == expected, "evidence_hash_mismatch")
    def reject(value):
        raise EvidenceError("nonfinite_json")
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate_json_key")
            result[key] = value
        return result
    def finite_float(value):
        parsed = float(value)
        require(math.isfinite(parsed), "nonfinite_json")
        return parsed
    try:
        value = json.loads(payload, parse_constant=reject, parse_float=finite_float, object_pairs_hook=pairs)
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise EvidenceError("invalid_json") from None
    require(isinstance(value, dict), "object_evidence_required")
    return value


def binding(record, *, json_document=True):
    require(isinstance(record, dict) and set(record) == {"path", "sha256"}, "invalid_evidence_binding")
    require(isinstance(record["path"], str) and Path(record["path"]).is_absolute(), "absolute_evidence_path_required")
    require(isinstance(record["sha256"], str) and HASH.fullmatch(record["sha256"]), "invalid_sha256")
    if json_document:
        return read_json(record["path"], record["sha256"])
    require(digest(record["path"]) == record["sha256"], "evidence_hash_mismatch")
    return record


def timestamp(value):
    require(isinstance(value, str), "missing_evidence_timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise EvidenceError("invalid_evidence_timestamp") from None
    require(parsed.tzinfo is not None and parsed.utcoffset() is not None, "timezone_required")
    return parsed.astimezone(timezone.utc)


def fresh(value, now, max_hours):
    age = (now - timestamp(value)).total_seconds()
    require(-300 <= age <= max_hours * 3600, "stale_or_future_evidence")


def all_true(record, keys, reason):
    require(isinstance(record, dict) and all(record.get(key) is True for key in keys), reason)


def verify_suite(path, sha, *, now, max_age_hours):
    suite = read_json(path, sha)
    require(suite.get("schema") == SUITE_SCHEMA and suite.get("completed") is True, "suite_not_complete")
    fresh(suite.get("completed_at"), now, max_age_hours)
    complete = read_json(Path(path).with_name("complete.json"))
    require(complete.get("schema") == SUITE_SCHEMA and complete.get("completed") is True
        and complete.get("suite_sha256") == sha
        and complete.get("diagnostic_contract_sha256") == suite.get("diagnostic_contract_sha256")
        and complete.get("completed_at") == suite.get("completed_at"), "completion_binding_mismatch")
    contract = binding({"path": suite.get("diagnostic_contract_path"), "sha256": suite.get("diagnostic_contract_sha256")})
    require(contract.get("schema") == SUITE_SCHEMA and contract.get("stage") == "validation", "wrong_contract")
    binding(contract.get("protocol"), json_document=False)
    binding(contract.get("prior_v3_contract"))
    codes = contract.get("code_sha256")
    require(isinstance(codes, dict) and bool(codes), "missing_code_pins")
    for name, expected in codes.items():
        require(isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9_]+\.py", name), "invalid_code_pin")
        binding({"path": str(SCRIPT_DIR / name), "sha256": expected}, json_document=False)
    decision = suite.get("decision")
    require(isinstance(decision, dict) and decision.get("policy") == contract.get("screen"), "decision_policy_mismatch")
    parts = ("passed", "conditioning_passed", "distribution_passed", "safety_passed")
    require(all(type(decision.get(key)) is bool for key in parts), "untyped_scientific_decision")
    require(decision["passed"] == all(decision[key] for key in parts[1:]), "inconsistent_scientific_decision")
    require(type(complete.get("screen_passed")) is bool and complete["screen_passed"] == decision["passed"], "completion_decision_mismatch")
    sources = decision.get("sources")
    require(isinstance(sources, list) and len(sources) == 2
        and {row.get("source") for row in sources if isinstance(row, dict)} == SOURCES, "missing_source_decisions")
    for row in sources:
        checks = row.get("checks")
        require(type(row.get("passed")) is bool and isinstance(checks, dict) and bool(checks)
            and all(type(value) is bool for value in checks.values()), "untyped_source_checks")
        require(row["passed"] == all(checks.values()), "inconsistent_source_checks")
        require(all(type(row.get(key)) is bool for key in parts), "untyped_source_decision")
        require(row["passed"] == all(row[key] for key in parts[1:]), "inconsistent_source_decision")
    require(decision["passed"] == all(row["passed"] for row in sources), "inconsistent_source_decision")
    require(all(decision[key] == all(row[key] for row in sources) for key in parts[1:]), "inconsistent_source_category")
    require(suite.get("promoted") is False and suite.get("submission_performed") is False, "unexpected_external_action")
    require(type(suite.get("tracking_events")) is int and suite["tracking_events"] == 54
        and type(suite.get("tracking_errors")) is int and suite["tracking_errors"] == 0
        and suite.get("cloud_synced") is False, "unverified_tracking")
    runs = suite.get("runs")
    require(isinstance(runs, list) and len(runs) == 6
        and {row.get("arm") for row in runs if isinstance(row, dict)} == set(ARMS), "missing_fixed_arms")
    for run in runs:
        require(run.get("tracking_events") == 9 and type(run.get("tracking_events")) is int
            and run.get("tracking_errors") == 0 and type(run.get("tracking_errors")) is int, "unverified_run_tracking")
        require(isinstance(run.get("artifact_audit"), dict) and run["artifact_audit"].get("passed") is True, "artifact_audit_failed")
        summary = binding({"path": run.get("summary_path"), "sha256": run.get("summary_sha256")})
        require(summary.get("schema") == SUMMARY_SCHEMA and summary.get("arm") == run["arm"]
            and summary.get("stage") == "validation" and summary.get("status") == "completed_unpromoted_diagnostic"
            and summary.get("diagnostic_contract_sha256") == suite["diagnostic_contract_sha256"]
            and summary.get("policy") == contract.get("policy"), "summary_binding_mismatch")
        all_true(summary, ("exact_ntc_identity_verified", "exact_zero_effect_identity_verified",
                          "prior_public_development_exposure_acknowledged"), "summary_safety_failed")
        require(all(summary.get(key) is False for key in ("count_emitter", "posttraining_performed", "promoted",
            "submission_performed", "hyperparameter_selection", "H1_treated_read", "RPE1_treated_read",
            "challenge_treated_used", "held_roles_used_for_fitting")), "forbidden_data_or_action")
        # A failed screen cannot enable a downstream branch. Avoid rereading its
        # large arrays each poll, but authenticate BOTH files before any readiness
        # can become true. No arrays are decoded by this watcher.
        if decision["passed"]:
            for name in ("weights", "predictions"):
                binding({"path": str(Path(run["summary_path"]).with_name(name + ".npz")),
                         "sha256": summary.get(name + "_sha256")}, json_document=False)
        tracking = binding({"path": run.get("tracking_path"), "sha256": run.get("tracking_sha256")})
        require(tracking.get("schema") == "vcc-training-tracking-v1" and tracking.get("stage") == "validation"
            and tracking.get("mode") == "offline" and tracking.get("cloud_synced") is False
            and tracking.get("execution_status") == "completed" and type(tracking.get("training_exit_code")) is int
            and tracking["training_exit_code"] == 0
            and type(tracking.get("event_count")) is int and tracking["event_count"] == 9
            and type(tracking.get("tracking_errors")) is int and tracking["tracking_errors"] == 0
            and tracking.get("run_id") == run.get("run_id") and isinstance(run.get("run_id"), str)
            and tracking.get("scientific_quality_inferred_from_exit") is False, "tracking_receipt_mismatch")
    return suite, decision["passed"]


def verify_parent(manifest, suite_sha):
    parent = binding(manifest.get("parent"))
    require(parent.get("schema") == "public-trainable-parent-readiness-v1" and parent.get("suite_sha256") == suite_sha, "wrong_parent_evidence")
    all_true(parent, ("completed", "trainable", "compatible", "public_only"), "parent_not_trainable_compatible")
    all_true(parent.get("compatibility"), ("input_gene_axis_verified", "output_gene_axis_verified",
        "conditioning_verified", "representation_verified"), "parent_compatibility_unverified")
    binding(parent.get("checkpoint"), json_document=False)


def verify_feng(manifest, suite_sha):
    verify_parent(manifest, suite_sha)
    feng = binding(manifest.get("feng"))
    require(feng.get("schema") == "public-feng-posttraining-eligibility-v1" and feng.get("suite_sha256") == suite_sha, "wrong_feng_evidence")
    all_true(feng, ("completed", "public_only", "donor_eligible", "batch_eligible", "target_exclusions_verified",
                   "matched_controls_verified", "representation_compatible"), "feng_not_eligible")
    binding(feng.get("source_manifest"))
    binding(feng.get("split_manifest"))


def verify_submission(manifest, suite_sha, now):
    emitter = binding(manifest.get("emitter"))
    require(emitter.get("schema") == "vcc-submission-emitter-readiness-v1" and emitter.get("suite_sha256") == suite_sha, "wrong_emitter_evidence")
    all_true(emitter, ("completed", "full_output_verified", "public_only_provenance", "count_space_calibration_validated"), "emitter_not_validated")
    require(emitter.get("representation") == "raw_counts" and emitter.get("shape") == [360000, 18533]
        and emitter.get("contexts") == ["A", "B", "C"], "wrong_submission_shape_or_space")
    all_true(emitter.get("checks"), ("finite", "nonnegative", "integer_valued", "gene_order_verified",
        "target_panel_verified", "context_labels_verified", "cell_counts_verified", "no_control_cells"), "submission_format_unverified")
    prediction = binding(emitter.get("prediction"), json_document=False)
    binding(emitter.get("package"), json_document=False)
    dry = binding(manifest.get("official_dryrun"))
    require(dry.get("dry_run") is True and dry.get("verified_targets") is True
        and dry.get("normalization") == "counts-preserved" and dry.get("n_cells") == 360000
        and dry.get("n_genes") == 18533 and dry.get("cells_per_context") == {key: 120000 for key in "ABC"}
        and isinstance(dry.get("input"), str) and Path(dry["input"]).resolve() == Path(prediction["path"]).resolve(), "official_dryrun_failed")
    require(type(dry.get("nnz")) is int and 0 < dry["nnz"] <= 4750000000, "submission_density_invalid")
    status = binding(manifest.get("status"))
    require(set(status) == {"schema", "checked_at", "credential_present", "can_submit", "quota_available", "inflight_submission"}
        and status["schema"] == "vcc-sanitized-readiness-status-v1", "unsafe_status_schema")
    fresh(status["checked_at"], now, 1)
    all_true(status, ("credential_present", "can_submit", "quota_available"), "credential_or_quota_unavailable")
    require(status.get("inflight_submission") is False, "submission_inflight")


def evaluate(suite_path, suite_sha, *, evidence_path=None, evidence_sha=None, now=None, max_age_hours=168):
    now = now or datetime.now(timezone.utc)
    require(type(max_age_hours) in (int, float) and math.isfinite(max_age_hours) and 0 < max_age_hours <= 720, "invalid_evidence_age_bound")
    state = {"schema": SCHEMA, "checked_at": now.isoformat(), "suite_sha256": suite_sha,
        "suite_evidence_valid": False, "scientific_gate_passed": False,
        "promoter_ablation_ready": False, "feng_posttraining_ready": False, "submission_ready": False,
        "reasons": {}, "external_actions_performed": False, "credentials_read": False,
        "new_evidence_requires_restart": True, "online_notifications": False}
    try:
        _, passed = verify_suite(suite_path, suite_sha, now=now, max_age_hours=max_age_hours)
        state["suite_evidence_valid"] = True
        state["scientific_gate_passed"] = passed
        require(passed, "scientific_or_distribution_or_safety_gate_failed")
        state["promoter_ablation_ready"] = True
    except (EvidenceError, KeyError, TypeError, ValueError) as error:
        reason = str(error) if isinstance(error, EvidenceError) else "malformed_suite_evidence"
        state["reasons"] = dict.fromkeys(("promoter", "feng", "submission"), reason)
        return state
    try:
        require(evidence_path is not None and evidence_sha is not None, "optional_evidence_not_registered")
        manifest = read_json(evidence_path, evidence_sha)
        require(manifest.get("schema") == "public-training-readiness-evidence-v1"
            and manifest.get("suite_sha256") == suite_sha, "evidence_manifest_binding_mismatch")
        require(set(manifest) <= {"schema", "suite_sha256", "parent", "feng", "emitter", "official_dryrun", "status"}, "unexpected_evidence_kind")
    except (EvidenceError, TypeError, ValueError) as error:
        state["reasons"].update(dict.fromkeys(("feng", "submission"), str(error) if isinstance(error, EvidenceError) else "malformed_optional_evidence"))
        return state
    for branch, validator in (("feng", lambda: verify_feng(manifest, suite_sha)),
                              ("submission", lambda: verify_submission(manifest, suite_sha, now))):
        try:
            validator()
            state[branch + ("_posttraining_ready" if branch == "feng" else "_ready")] = True
        except (EvidenceError, KeyError, TypeError, ValueError) as error:
            state["reasons"][branch] = str(error) if isinstance(error, EvidenceError) else "malformed_optional_evidence"
    return state


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp." + uuid.uuid4().hex)
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def monitor(args, *, stop=None, monotonic=time.monotonic, sleep=time.sleep):
    hostname = socket.gethostname()
    require(hostname.split(".")[0] in {"cbsuvlaminck3", "cbsuvlaminck6"}, "compute_host_required")
    require(type(args.interval_seconds) in (int, float) and math.isfinite(args.interval_seconds) and args.interval_seconds >= 5, "interval_must_be_at_least_five_seconds")
    require(type(args.max_hours) in (int, float) and math.isfinite(args.max_hours) and 0 < args.max_hours <= 24, "watch_bound_must_be_at_most_24_hours")
    require(type(args.max_evidence_age_hours) in (int, float) and math.isfinite(args.max_evidence_age_hours)
        and 0 < args.max_evidence_age_hours <= 720, "invalid_evidence_age_bound")
    output = Path(args.output_dir)
    require(not os.path.lexists(output), "fresh_output_directory_required")
    output.mkdir(parents=True, mode=0o700)
    started, checks, changes, previous = monotonic(), 0, 0, None
    deadline = started + args.max_hours * 3600
    started_at = datetime.now(timezone.utc)
    watcher_sha = digest(Path(__file__))
    runtime = {"schema": SCHEMA, "pid": os.getpid(), "hostname": hostname,
        "started_at": started_at.isoformat(), "deadline_at": (started_at + timedelta(hours=args.max_hours)).isoformat(),
        "watcher_sha256": watcher_sha, "suite_sha256": args.suite_sha256, "evidence_sha256": args.evidence_sha256,
        "interval_seconds": args.interval_seconds, "max_hours": args.max_hours,
        "new_evidence_requires_restart": True, "online_notifications": False,
        "automatic_training": False, "automatic_submission": False}
    atomic_json(output / "run.json", runtime)
    stopped = stop or (lambda: False)
    reason = "single_check"
    with (output / "states.jsonl").open("x", buffering=1) as journal:
        while True:
            state = evaluate(args.suite, args.suite_sha256, evidence_path=args.evidence_manifest,
                evidence_sha=args.evidence_sha256, max_age_hours=args.max_evidence_age_hours)
            checks += 1
            compared = {key: value for key, value in state.items() if key != "checked_at"}
            if compared != previous:
                changes += 1
                journal.write(json.dumps(state, sort_keys=True, allow_nan=False) + "\n")
                journal.flush(); os.fsync(journal.fileno())
                atomic_json(output / "state.json", state)
                print(json.dumps(state, sort_keys=True, allow_nan=False), flush=True)
                previous = compared
            atomic_json(output / "heartbeat.json", {**runtime, "checks": checks, "state_changes": changes,
                "last_checked_at": state.get("checked_at", datetime.now(timezone.utc).isoformat()),
                "elapsed_seconds": monotonic() - started, "running": args.command == "watch" and not stopped()})
            if args.command == "check":
                break
            if stopped():
                reason = "signal"; break
            if monotonic() >= deadline:
                reason = "time_bound"; break
            next_check = min(deadline, monotonic() + args.interval_seconds)
            while not stopped() and monotonic() < next_check:
                sleep(max(0, min(60, next_check - monotonic())))
            if stopped():
                reason = "signal"; break
    receipt = {"schema": SCHEMA, "finished_at": datetime.now(timezone.utc).isoformat(), "stop_reason": reason,
        "checks": checks, "state_changes": changes, "suite_sha256": args.suite_sha256,
        "evidence_sha256": args.evidence_sha256, "watcher_sha256": watcher_sha,
        "interval_seconds": args.interval_seconds, "max_hours": args.max_hours,
        "elapsed_seconds": monotonic() - started, "last_state": state,
        "read_only_inputs": True, "external_actions_performed": False, "credentials_read": False,
        "online_notifications": False, "automatic_training": False, "automatic_submission": False,
        "new_evidence_requires_restart": True}
    atomic_json(output / "receipt.json", receipt)
    atomic_json(output / "heartbeat.json", {**runtime, "checks": checks, "state_changes": changes,
        "last_checked_at": state.get("checked_at", datetime.now(timezone.utc).isoformat()),
        "elapsed_seconds": monotonic() - started, "running": False, "stop_reason": reason})
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("check", "watch"):
        item = sub.add_parser(command)
        item.add_argument("--suite", type=Path, required=True)
        item.add_argument("--suite-sha256", required=True)
        item.add_argument("--evidence-manifest", type=Path)
        item.add_argument("--evidence-sha256")
        item.add_argument("--output-dir", type=Path, required=True)
        item.add_argument("--interval-seconds", type=float, default=300)
        item.add_argument("--max-hours", type=float, default=24)
        item.add_argument("--max-evidence-age-hours", type=float, default=168)
    args = parser.parse_args(argv)
    stopped = {"value": False}
    def stop_handler(signum, frame):
        stopped["value"] = True
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, stop_handler)
    try:
        monitor(args, stop=lambda: stopped["value"])
    except EvidenceError as error:
        parser.exit(2, f"readiness watcher: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
