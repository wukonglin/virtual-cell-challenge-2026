"""Bounded read-only v8.1 completion monitoring and saved-fit verification.

Never starts or stops training, initializes W&B, reads raw expression, uploads,
or submits. Only this monitor's fresh directory may receive status/log files.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import stat
import subprocess
import sys
import time

import run_public_expanded_v81 as runner
import public_training_readiness as safe

SCHEMA = "public-expanded-completion-watch-v8.1"
MAX_WAIT_HOURS = 4
MAX_VERIFY_SECONDS = 4 * 3600
MAX_VERIFY_AS = 16 << 30
MAX_VERIFY_LOG_BYTES = 8 << 20
STOP_GRACE_SECONDS = 10
FLAGS = {"read_only_inputs": True, "model_fitting_performed": False,
    "training_process_modified": False, "wandb_initialized": False,
    "raw_expression_read": False, "credentials_read": False, "gpu_used": False,
    "upload_performed": False, "submission_performed": False, "online_notifications": False}


def require(value, reason):
    if not value:
        raise safe.EvidenceError(reason)


def equal(actual, expected, reason):
    require(json.dumps(actual, sort_keys=True, allow_nan=False) ==
            json.dumps(expected, sort_keys=True, allow_nan=False), reason)


def now():
    return datetime.now(timezone.utc).isoformat()


def _signature(stream):
    info = os.fstat(stream.fileno())
    return tuple(getattr(info, key) for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns"))


def _parse(payload):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate_json_key")
            result[key] = value
        return result
    def reject(value):
        raise safe.EvidenceError("nonfinite_json")
    def finite(value):
        value = float(value)
        require(math.isfinite(value), "nonfinite_json")
        return value
    try:
        value = json.loads(payload, parse_constant=reject, parse_float=finite, object_pairs_hook=pairs)
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise safe.EvidenceError("invalid_json") from None
    require(isinstance(value, dict), "object_json_required")
    return value


def _read(path, maximum=safe.MAX_JSON_BYTES):
    with safe.regular_reader(path) as stream:
        before = _signature(stream)
        require(before[2] <= maximum, "oversized_evidence")
        payload = stream.read(maximum + 1)
        require(_signature(stream) == before, "evidence_changed_during_read")
    require(len(payload) <= maximum, "oversized_evidence")
    return payload


def _optional(path):
    if not os.path.lexists(path):
        return None
    payload = _read(path)
    if not payload.strip():
        return None
    return _parse(payload), hashlib.sha256(payload).hexdigest()


def _directory(path, *, optional=False):
    if optional and not os.path.lexists(path):
        return False
    try:
        require(stat.S_ISDIR(Path(path).lstat().st_mode), "regular_directory_required")
    except OSError:
        raise safe.EvidenceError("missing_directory") from None
    return True


def contract_path(path):
    path = Path(path).absolute()
    require(path.name == "contract.json" and not path.is_symlink(), "canonical_nonsymlink_contract_required")
    # Workspace mount aliases may resolve to the same physical directory. The
    # canonical leaf itself must not be a symlink; validation paths are derived.
    return path.resolve()


def poll_state(contract, contract_sha):
    """Read only completion/tracking metadata after fully revalidating the seal."""
    contract = contract_path(contract)
    require(isinstance(contract_sha, str) and safe.HASH.fullmatch(contract_sha), "invalid_contract_sha")
    runner.validate_contract(contract, contract_sha)
    root = contract.parent / "validation"
    base = {"schema": SCHEMA, "checked_at": now(), "contract_path": str(contract),
        "contract_sha256": contract_sha, "validation_path": str(root), "suite_verified": False, **FLAGS}
    if not _directory(root, optional=True):
        return {**base, "status": "pending", "reason": "validation_directory_not_present", "tracking": []}
    tracking = []
    if _directory(root / "tracking", optional=True):
        for arm in runner.EVENTS:
            directory = root / "tracking" / arm
            if not _directory(directory, optional=True):
                continue
            result = _optional(directory / "tracking.json")
            if result is None:
                continue
            receipt, digest = result
            for key, expected in {"schema": "vcc-training-tracking-v1", "stage": "validation",
                    "mode": "offline", "group": runner.GROUP, "cloud_synced": False,
                    "raw_artifacts_uploaded": False}.items():
                equal(receipt.get(key), expected, "tracking_binding_mismatch")
            equal(receipt.get("config", {}).get("diagnostic_contract_sha256"), contract_sha,
                  "tracking_contract_mismatch")
            status = receipt.get("execution_status")
            require(status in {"initializing", "running", "completed", "failed", "interrupted",
                               "tracking_initialization_failed"}, "unknown_tracking_lifecycle")
            row = {"arm": arm, "status": status, "sha256": digest}
            for field in ("event_count", "tracking_errors"):
                if status in {"completed", "failed", "interrupted"}:
                    require(field in receipt, "missing_terminal_tracking_count")
                value = receipt.get(field, 0)
                require(type(value) is int and value >= 0, "invalid_tracking_count")
                row[field] = value
            if status in {"completed", "failed", "interrupted"}:
                require(type(receipt.get("training_exit_code")) is int, "invalid_tracking_exit")
                row["exit_code"] = receipt["training_exit_code"]
            if status == "completed":
                require(row["exit_code"] == 0, "inconsistent_tracking_completion")
                require(row["event_count"] == runner.EVENTS[arm], "incomplete_terminal_tracking_events")
            tracking.append(row)
    failures = [row for row in tracking if row["status"] in {"failed", "interrupted", "tracking_initialization_failed"}
                or row["tracking_errors"] > 0]
    if failures:
        return {**base, "status": "recorded_failure", "reason": "tracking_recorded_failure",
                "tracking": tracking, "failed_arms": [row["arm"] for row in failures]}
    completion = _optional(root / "complete.json")
    if completion is None:
        return {**base, "status": "pending", "reason": "completion_marker_missing_or_empty", "tracking": tracking}
    marker, marker_sha = completion
    for key, expected in {"schema": runner.SCHEMA, "completed": True,
            "diagnostic_contract_sha256": contract_sha, "tracking_events": 207, "tracking_errors": 0,
            "cloud_synced": False, "promoted": False, "submission_performed": False}.items():
        equal(marker.get(key), expected, "completion_binding_mismatch")
    require(type(marker.get("screen_passed")) is bool, "typed_screen_flag_required")
    suite_sha = marker.get("suite_sha256")
    require(isinstance(suite_sha, str) and safe.HASH.fullmatch(suite_sha), "invalid_suite_sha")
    payload = _read(root / "suite.json")
    require(hashlib.sha256(payload).hexdigest() == suite_sha, "suite_hash_mismatch")
    suite = _parse(payload)
    for key, expected in {"schema": runner.SCHEMA, "completed": True,
            "diagnostic_contract_path": str(contract), "diagnostic_contract_sha256": contract_sha,
            "completed_at": marker.get("completed_at")}.items():
        equal(suite.get(key), expected, "suite_completion_binding_mismatch")
    safe.timestamp(marker.get("completed_at"))
    equal(suite.get("decision", {}).get("passed"), marker["screen_passed"], "completion_screen_binding_mismatch")
    return {**base, "status": "suite_complete_unverified", "reason": "ready_for_saved_fit_replay",
        "tracking": tracking, "suite_sha256": suite_sha, "completion_sha256": marker_sha}


def _result(value, suite_sha):
    expected = {"schema": "public-expanded-verification-v8", "verified": True, "suite_sha256": suite_sha,
        "groups": 1767, "saved_fits_replayed": 192, "control_references_replayed": 8, "ridge_refits": 0,
        "tracking_events_replayed": 207, "feng_ready": False, "submission_ready": False}
    require(isinstance(value, dict) and set(value) == set(expected) |
            {"conditioning_passed", "distribution_passed", "safety_passed"}, "verification_result_fields")
    for key, wanted in expected.items():
        equal(value.get(key), wanted, "verification_result_binding")
    for key in ("conditioning_passed", "distribution_passed", "safety_passed"):
        require(type(value[key]) is bool, "typed_verified_decision_required")
    return value


def _child_limits():
    # A stronger address-space bound also caps RSS. This applies only to the
    # verifier child, never the separate training PID or the polling process.
    resource.setrlimit(resource.RLIMIT_AS, (MAX_VERIFY_AS, MAX_VERIFY_AS))
    cpu_seconds = MAX_VERIFY_SECONDS * runner.THREADS
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_VERIFY_LOG_BYTES, MAX_VERIFY_LOG_BYTES))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def verify_child(contract, contract_sha, suite_sha, watcher_sha):
    require(safe.digest(Path(__file__).resolve()) == watcher_sha, "watcher_code_changed")
    _child_limits()
    state = poll_state(contract, contract_sha)
    require(state["status"] == "suite_complete_unverified" and state["suite_sha256"] == suite_sha,
            "verification_input_changed")
    result = runner.verify_suite(Path(state["validation_path"]) / "suite.json", suite_sha)
    print(json.dumps(_result(result, suite_sha), sort_keys=True), flush=True)
    return result


def _terminate(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=STOP_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=STOP_GRACE_SECONDS)


def bounded_verify(contract, contract_sha, suite_sha, output_dir, watcher_sha, *,
                   stop=lambda: False, heartbeat=lambda phase: None,
                   monotonic=time.monotonic, sleep=time.sleep):
    """Run only the sealed verifier child, bounded by wall time, memory and log."""
    output_dir = Path(output_dir)
    logfile = output_dir / "verifier.log"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    env = dict(os.environ)
    env.update({name: str(runner.THREADS) for name in runner.THREAD_VARIABLES})
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "--verify-child", "--contract", str(contract),
        "--contract-sha256", contract_sha, "--expected-suite-sha256", suite_sha,
        "--watcher-sha256", watcher_sha]
    with os.fdopen(os.open(logfile, flags, 0o600), "wb") as log:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                   env=env, start_new_session=True, close_fds=True)
        started = monotonic()
        try:
            while process.poll() is None:
                require(not stop(), "verification_interrupted")
                require(monotonic() - started < MAX_VERIFY_SECONDS, "verification_time_bound")
                require(logfile.lstat().st_size <= MAX_VERIFY_LOG_BYTES, "verification_log_bound")
                heartbeat("verifying")
                sleep(max(0, min(5, MAX_VERIFY_SECONDS - (monotonic() - started))))
            require(process.returncode == 0, "verification_process_failed")
        finally:
            _terminate(process)
    payload = _read(logfile, MAX_VERIFY_LOG_BYTES)
    lines = [line for line in payload.splitlines() if line.strip()]
    require(bool(lines), "missing_verification_result")
    result = _result(_parse(lines[-1]), suite_sha)
    return result, {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload),
                    "elapsed_seconds": monotonic() - started, "child_pid": process.pid}


def monitor(args, *, stop=lambda: False, monotonic=time.monotonic, sleep=time.sleep, verifier=bounded_verify):
    require(runner.socket.gethostname().split(".")[0] in {"cbsuvlaminck3", "cbsuvlaminck6"}, "compute_host_required")
    require(type(args.interval_seconds) in (int, float) and math.isfinite(args.interval_seconds)
            and 5 <= args.interval_seconds <= 60, "interval_must_be_5_to_60_seconds")
    require(type(args.max_hours) in (int, float) and math.isfinite(args.max_hours)
            and 0 < args.max_hours <= MAX_WAIT_HOURS, "wait_bound_must_be_at_most_four_hours")
    contract = contract_path(args.contract)
    initial = poll_state(contract, args.contract_sha256)
    output = Path(args.output_dir).absolute()
    require(not os.path.lexists(output), "fresh_monitor_directory_required")
    output = output.resolve()
    forbidden = contract.parent / "validation"
    require(output != contract.parent and not output.is_relative_to(forbidden)
            and not forbidden.is_relative_to(output), "monitor_must_not_overlap_training_outputs")
    output.mkdir(parents=True, mode=0o700)
    started, started_at = monotonic(), datetime.now(timezone.utc)
    deadline = started + args.max_hours * 3600
    watcher_sha = safe.digest(Path(__file__).resolve())
    run = {"schema": SCHEMA, "pid": os.getpid(), "started_at": started_at.isoformat(),
        "wait_deadline_at": (started_at + timedelta(hours=args.max_hours)).isoformat(),
        "contract_path": str(contract), "contract_sha256": args.contract_sha256,
        "validation_path": str(forbidden), "watcher_sha256": watcher_sha,
        "interval_seconds": args.interval_seconds, "maximum_wait_hours": args.max_hours,
        "maximum_verification_seconds": MAX_VERIFY_SECONDS, "maximum_total_seconds": args.max_hours * 3600 + MAX_VERIFY_SECONDS + 2 * STOP_GRACE_SECONDS,
        "verification_address_space_bytes": MAX_VERIFY_AS, "verification_log_bytes": MAX_VERIFY_LOG_BYTES,
        "verification_cpu_threads": runner.THREADS, **FLAGS}
    safe.atomic_json(output / "run.json", run)
    checks, state, verification = 0, initial, None
    def heartbeat(phase):
        safe.atomic_json(output / "heartbeat.json", {**run, "last_checked_at": now(),
            "checks": checks, "phase": phase, "running": True, "elapsed_seconds": monotonic() - started})
    reason = "unknown"
    try:
        while True:
            require(safe.digest(Path(__file__).resolve()) == watcher_sha, "watcher_code_changed")
            if checks:
                state = poll_state(contract, args.contract_sha256)
            checks += 1
            safe.atomic_json(output / "state.json", state)
            heartbeat(state["status"])
            if stop():
                reason = "signal"; break
            if state["status"] == "recorded_failure":
                reason = "recorded_failure"; break
            if state["status"] == "suite_complete_unverified":
                verification, log = verifier(contract, args.contract_sha256, state["suite_sha256"], output,
                    watcher_sha, stop=stop, heartbeat=heartbeat, monotonic=monotonic, sleep=sleep)
                _result(verification, state["suite_sha256"])
                after = poll_state(contract, args.contract_sha256)
                require(after["status"] == "suite_complete_unverified"
                        and after["suite_sha256"] == state["suite_sha256"]
                        and after["completion_sha256"] == state["completion_sha256"], "completion_changed_during_verification")
                require(safe.digest(Path(__file__).resolve()) == watcher_sha, "watcher_code_changed")
                safe.atomic_json(output / "verification.json", {"schema": SCHEMA, "verified_at": now(),
                    "contract_sha256": args.contract_sha256, "suite_sha256": state["suite_sha256"],
                    "completion_sha256": state["completion_sha256"], "watcher_sha256": watcher_sha,
                    "verifier_log": log, "result": verification, **FLAGS})
                state = {**after, "status": "verified_complete", "suite_verified": True,
                    "reason": "saved_fit_and_tracking_replay_passed", "verification": verification,
                    "scientific_gate_passed": all(verification[key] for key in
                        ("conditioning_passed", "distribution_passed", "safety_passed")),
                    "automatic_next_stage": False}
                safe.atomic_json(output / "state.json", state)
                reason = "verified_complete"; break
            if monotonic() >= deadline:
                reason = "wait_time_bound"; break
            sleep(min(args.interval_seconds, max(0, deadline - monotonic())))
    except Exception as error:
        reason = str(error) if isinstance(error, safe.EvidenceError) else "malformed_or_failed_evidence"
        state = {**state, "status": "verification_failed" if state.get("status") == "suite_complete_unverified" else "evidence_error",
                 "suite_verified": False, "reason": reason}
        safe.atomic_json(output / "state.json", state)
    receipt = {**run, "finished_at": now(), "stop_reason": reason, "checks": checks,
        "elapsed_seconds": monotonic() - started, "suite_verified": reason == "verified_complete",
        "verification_written": reason == "verified_complete", "last_state": state}
    safe.atomic_json(output / "receipt.json", receipt)
    safe.atomic_json(output / "heartbeat.json", {**run, "last_checked_at": now(), "checks": checks,
        "phase": state["status"], "running": False, "stop_reason": reason, "elapsed_seconds": monotonic() - started})
    print(json.dumps({"schema": SCHEMA, "stop_reason": reason, "suite_verified": receipt["suite_verified"],
                      "receipt_path": str(output / "receipt.json")}, sort_keys=True), flush=True)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--interval-seconds", type=float, default=60)
    parser.add_argument("--max-hours", type=float, default=4)
    parser.add_argument("--verify-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--expected-suite-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--watcher-sha256", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.verify_child:
        verify_child(args.contract, args.contract_sha256, args.expected_suite_sha256, args.watcher_sha256)
        return 0
    if args.output_dir is None:
        parser.error("Monitoring requires its own fresh output directory")
    stopped = {"value": False}
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: stopped.update(value=True))
    receipt = monitor(args, stop=lambda: stopped["value"])
    return 0 if receipt["suite_verified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
