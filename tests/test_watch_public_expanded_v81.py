"""Synthetic clocks and metadata only; no real models, W&B init or raw inputs."""
from argparse import Namespace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import watch_public_expanded_v81 as v


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
    return v.safe.digest(path)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    contract = tmp_path / "execution" / "contract.json"
    sha = write(contract, {"schema": v.runner.SCHEMA, "sealed": True})
    calls = []
    def validate(path, expected):
        calls.append((Path(path), expected))
        assert Path(path) == contract
        value = v.safe.read_json(path, expected)
        v.require(value == {"schema": v.runner.SCHEMA, "sealed": True}, "synthetic_contract_changed")
        return value, {}, {}, {}
    monkeypatch.setattr(v.runner, "validate_contract", validate)
    monkeypatch.setattr(v.runner.socket, "gethostname", lambda: "cbsuvlaminck3.test")
    monkeypatch.setattr(v.runner.model.previous.kernel, "fit_kernel_decoder", lambda *a, **k: pytest.fail("model fit"))
    monkeypatch.setattr(v.runner.ExpandedValidationTracker, "__init__", lambda *a, **k: pytest.fail("W&B init"))
    return {"contract": contract, "sha": sha, "validation": contract.parent / "validation",
            "calls": calls, "tmp": tmp_path}


def tracking(data, status="running", arm="control", **extra):
    value = {"schema": "vcc-training-tracking-v1", "stage": "validation", "mode": "offline",
        "group": v.runner.GROUP, "cloud_synced": False, "raw_artifacts_uploaded": False,
        "config": {"diagnostic_contract_sha256": data["sha"]}, "execution_status": status}
    if status in {"completed", "failed", "interrupted"}:
        value.update(training_exit_code=0 if status == "completed" else 1,
                     event_count=v.runner.EVENTS[arm] if status == "completed" else 0, tracking_errors=0)
    value.update(extra)
    path = data["validation"] / "tracking" / arm / "tracking.json"
    write(path, value)
    return path, value


def complete(data):
    suite = {"schema": v.runner.SCHEMA, "completed": True,
        "diagnostic_contract_path": str(data["contract"]), "diagnostic_contract_sha256": data["sha"],
        "completed_at": "2026-09-10T17:00:00+00:00", "decision": {"passed": False}}
    suite_sha = write(data["validation"] / "suite.json", suite)
    marker = {"schema": v.runner.SCHEMA, "completed": True, "suite_sha256": suite_sha,
        "diagnostic_contract_sha256": data["sha"], "completed_at": suite["completed_at"],
        "screen_passed": False, "tracking_events": 207, "tracking_errors": 0,
        "cloud_synced": False, "promoted": False, "submission_performed": False}
    write(data["validation"] / "complete.json", marker)
    return suite_sha, suite, marker


def result(sha):
    return {"schema": "public-expanded-verification-v8", "verified": True, "suite_sha256": sha,
        "groups": 1767, "saved_fits_replayed": 192, "control_references_replayed": 8,
        "ridge_refits": 0, "tracking_events_replayed": 207, "conditioning_passed": False,
        "distribution_passed": False, "safety_passed": True, "feng_ready": False, "submission_ready": False}


class Clock:
    def __init__(self, callback=lambda: None):
        self.seconds, self.sleeps, self.callback = 0., [], callback
    def monotonic(self): return self.seconds
    def sleep(self, seconds):
        assert 0 <= seconds <= 60
        self.seconds += seconds
        self.sleeps.append(seconds)
        self.callback()


def args(data, **changes):
    return Namespace(**{"contract": data["contract"], "contract_sha256": data["sha"],
        "output_dir": data["tmp"] / "monitor", "interval_seconds": 60, "max_hours": .05, **changes})


def fake_verifier(contract, contract_sha, suite_sha, output_dir, watcher_sha, **kwargs):
    kwargs["heartbeat"]("verifying")
    return result(suite_sha), {"sha256": "e" * 64, "bytes": 300, "elapsed_seconds": 1., "child_pid": 1234}


@pytest.mark.parametrize("kind", ["missing_directory", "missing_marker", "empty_marker", "whitespace_marker", "suite_without_marker", "empty_tracking"])
def test_missing_or_empty_evidence_is_pending_not_failed(setup, kind):
    data = setup
    if kind != "missing_directory": data["validation"].mkdir()
    if kind == "empty_marker": (data["validation"] / "complete.json").write_bytes(b"")
    if kind == "whitespace_marker": (data["validation"] / "complete.json").write_bytes(b" \n")
    if kind == "suite_without_marker": write(data["validation"] / "suite.json", {})
    if kind == "empty_tracking":
        path, _ = tracking(data)
        path.write_bytes(b"")
    state = v.poll_state(data["contract"], data["sha"])
    assert state["status"] == "pending" and state["suite_verified"] is False


@pytest.mark.parametrize("status", ["failed", "interrupted", "tracking_initialization_failed"])
def test_recorded_failures_exit_without_verifier(setup, status):
    data = setup
    tracking(data, status)
    receipt = v.monitor(args(data), verifier=lambda *a, **k: pytest.fail("verifier called for failed run"))
    assert receipt["stop_reason"] == "recorded_failure" and not receipt["suite_verified"]
    assert receipt["last_state"]["failed_arms"] == ["control"]
    assert not (data["tmp"] / "monitor/verification.json").exists()


def test_nonzero_tracking_errors_fail_even_if_process_completed(setup):
    tracking(setup, "completed", tracking_errors=1)
    assert v.poll_state(setup["contract"], setup["sha"])["status"] == "recorded_failure"


@pytest.mark.parametrize("field", ["event_count", "tracking_errors", "training_exit_code"])
def test_terminal_tracking_requires_present_typed_counts(setup, field):
    path, receipt = tracking(setup, "completed")
    del receipt[field]
    write(path, receipt)
    with pytest.raises(ValueError): v.poll_state(setup["contract"], setup["sha"])


def test_completed_tracking_requires_full_expected_event_roster(setup):
    tracking(setup, "completed", event_count=0)
    with pytest.raises(ValueError): v.poll_state(setup["contract"], setup["sha"])


def test_all_false_science_still_verifies_integrity_without_enabling_next_stage(setup):
    complete(setup)
    def verifier(*a, **k):
        value, log = fake_verifier(*a, **k)
        for key in ("conditioning_passed", "distribution_passed", "safety_passed"):
            value[key] = False
        return value, log
    receipt = v.monitor(args(setup), verifier=verifier)
    assert receipt["suite_verified"] and receipt["stop_reason"] == "verified_complete"
    assert receipt["last_state"]["scientific_gate_passed"] is False
    assert receipt["last_state"]["automatic_next_stage"] is False
    assert receipt["last_state"]["verification"]["feng_ready"] is False
    assert receipt["last_state"]["verification"]["submission_ready"] is False


def test_pending_then_complete_full_verification_and_receipt(setup):
    data = setup
    tracking(data)
    clock = Clock(lambda: complete(data))
    receipt = v.monitor(args(data), monotonic=clock.monotonic, sleep=clock.sleep, verifier=fake_verifier)
    assert receipt["stop_reason"] == "verified_complete" and receipt["suite_verified"]
    assert receipt["checks"] == 2 and clock.sleeps == [60]
    assert len(data["calls"]) == 3  # initial, completion poll, post-verifier seal check
    assert receipt["last_state"]["verification"]["conditioning_passed"] is False
    assert receipt["last_state"]["verification"]["submission_ready"] is False
    output = data["tmp"] / "monitor"
    assert set(p.name for p in output.iterdir()) == {"run.json", "state.json", "heartbeat.json", "receipt.json", "verification.json"}
    assert v.safe.read_json(output / "heartbeat.json")["running"] is False
    assert all(receipt[key] is value for key, value in v.FLAGS.items())


def test_pending_then_recorded_failure(setup):
    clock = Clock(lambda: tracking(setup, "failed"))
    receipt = v.monitor(args(setup), monotonic=clock.monotonic, sleep=clock.sleep,
                        verifier=lambda *a, **k: pytest.fail("verification called"))
    assert receipt["checks"] == 2 and receipt["stop_reason"] == "recorded_failure"


def test_wait_timeout_is_not_training_failure_or_completion(setup):
    clock = Clock()
    receipt = v.monitor(args(setup, max_hours=90/3600), monotonic=clock.monotonic, sleep=clock.sleep)
    assert clock.sleeps == [60, 30]
    assert receipt["stop_reason"] == "wait_time_bound" and not receipt["suite_verified"]
    assert receipt["last_state"]["status"] == "pending"
    assert receipt["training_process_modified"] is False


def test_signal_stops_only_monitor(setup):
    receipt = v.monitor(args(setup), stop=lambda: True)
    assert receipt["stop_reason"] == "signal" and not receipt["training_process_modified"]


@pytest.mark.parametrize("kind", ["marker_symlink", "marker_fifo", "tracking_symlink", "tracking_fifo", "validation_symlink", "contract_symlink"])
def test_unsafe_evidence_paths_rejected(setup, kind):
    data = setup
    contract = data["contract"]
    if kind.startswith("tracking"):
        path, _ = tracking(data)
    elif kind == "contract_symlink": path = contract
    else:
        complete(data)
        path = data["validation"] if kind == "validation_symlink" else data["validation"] / "complete.json"
    copy = data["tmp"] / "copy"
    path.rename(copy)
    if "fifo" in kind: os.mkfifo(path)
    else: path.symlink_to(copy)
    with pytest.raises(ValueError): v.poll_state(contract, data["sha"])


@pytest.mark.parametrize("kind", ["contract", "suite_sha", "marker_contract", "suite_contract_path", "suite_contract_sha", "schema", "screen_bool", "completed_bool", "tracking_contract", "partial_marker"])
def test_mismatched_or_malformed_evidence_never_triggers_verification(setup, kind):
    data = setup
    _, suite, marker = complete(data)
    if kind == "contract": data["contract"].write_text("{}")
    elif kind == "suite_sha": marker["suite_sha256"] = "f" * 64
    elif kind == "marker_contract": marker["diagnostic_contract_sha256"] = "f" * 64
    elif kind == "suite_contract_path": suite["diagnostic_contract_path"] = "/tmp/elsewhere/contract.json"
    elif kind == "suite_contract_sha": suite["diagnostic_contract_sha256"] = "f" * 64
    elif kind == "schema": marker["schema"] = "other"
    elif kind == "screen_bool": marker["screen_passed"] = 0
    elif kind == "completed_bool": marker["completed"] = 1
    elif kind == "tracking_contract": tracking(data, config={"diagnostic_contract_sha256": "f" * 64})
    elif kind == "partial_marker":
        (data["validation"] / "complete.json").write_text('{"schema":')
        with pytest.raises(ValueError): v.poll_state(data["contract"], data["sha"])
        return
    if kind.startswith("suite_contract"):
        marker["suite_sha256"] = write(data["validation"] / "suite.json", suite)
    write(data["validation"] / "complete.json", marker)
    with pytest.raises(ValueError): v.poll_state(data["contract"], data["sha"])


def test_contract_revalidated_each_poll_failure_is_recorded(setup):
    clock = Clock(lambda: setup["contract"].write_text("{}"))
    receipt = v.monitor(args(setup), monotonic=clock.monotonic, sleep=clock.sleep)
    assert receipt["stop_reason"] == "evidence_hash_mismatch"
    assert receipt["last_state"]["status"] == "evidence_error"


@pytest.mark.parametrize("kind", ["failure", "timeout", "refit", "wrong_sha", "changed_marker"])
def test_verification_failures_never_write_success(setup, kind):
    data = setup
    complete(data)
    def verifier(*a, **k):
        if kind == "failure": raise RuntimeError("synthetic verifier failed")
        if kind == "timeout": raise v.safe.EvidenceError("verification_time_bound")
        value, log = fake_verifier(*a, **k)
        if kind == "refit": value["ridge_refits"] = 1
        elif kind == "wrong_sha": value["suite_sha256"] = "f" * 64
        elif kind == "changed_marker":
            path = data["validation"] / "complete.json"
            path.write_text(path.read_text() + " ")
        return value, log
    receipt = v.monitor(args(data), verifier=verifier)
    assert receipt["last_state"]["status"] == "verification_failed" and not receipt["suite_verified"]
    assert not (data["tmp"] / "monitor/verification.json").exists()


def test_child_calls_only_frozen_verifier_after_bindings(setup, monkeypatch):
    data = setup
    sha, _, _ = complete(data)
    calls = []
    monkeypatch.setattr(v, "_child_limits", lambda: calls.append("limits"))
    def verify(path, expected):
        calls.append((Path(path), expected))
        return result(expected)
    monkeypatch.setattr(v.runner, "verify_suite", verify)
    found = v.verify_child(data["contract"], data["sha"], sha, v.safe.digest(Path(v.__file__)))
    assert found["verified"] and calls == ["limits", (data["validation"] / "suite.json", sha)]


def test_child_limits_are_local_and_bounded(monkeypatch):
    seen = []
    monkeypatch.setattr(v.resource, "setrlimit", lambda resource, pair: seen.append((resource, pair)))
    v._child_limits()
    assert seen == [(v.resource.RLIMIT_AS, (16 << 30, 16 << 30)),
        (v.resource.RLIMIT_CPU, (4*3600*8, 4*3600*8)),
        (v.resource.RLIMIT_FSIZE, (8 << 20, 8 << 20)), (v.resource.RLIMIT_CORE, (0, 0))]


def test_watcher_code_change_during_verification_prevents_success(setup, monkeypatch):
    complete(setup)
    original, changed = v.safe.digest, {"value": False}
    def digest(path):
        if Path(path).resolve() == Path(v.__file__).resolve() and changed["value"]:
            return "f" * 64
        return original(path)
    monkeypatch.setattr(v.safe, "digest", digest)
    def verifier(*a, **k):
        value = fake_verifier(*a, **k)
        changed["value"] = True
        return value
    receipt = v.monitor(args(setup), verifier=verifier)
    assert receipt["stop_reason"] == "watcher_code_changed" and not receipt["suite_verified"]
    assert not (setup["tmp"] / "monitor/verification.json").exists()


@pytest.mark.parametrize("mode", ["success", "timeout", "interrupted", "exit_failed", "stubborn", "oversize_log"])
def test_bounded_subprocess_uses_fixed_verify_command_and_stops_only_own_child(setup, monkeypatch, mode):
    data = setup
    sha, _, _ = complete(data)
    output = data["tmp"] / "child_output"
    output.mkdir()
    clock, captured = Clock(), {}
    class Process:
        pid = 9876
        def __init__(self, command, **kwargs):
            captured.update(command=command, kwargs=kwargs, terminated=0, killed=0)
            kwargs["stdout"].write((json.dumps(result(sha)) + "\n").encode())
            kwargs["stdout"].flush()
            self.returncode = 0 if mode in {"success", "oversize_log"} else (1 if mode == "exit_failed" else None)
        def poll(self): return self.returncode
        def terminate(self): captured["terminated"] += 1; self.returncode = None if mode == "stubborn" else -15
        def kill(self): captured["killed"] += 1; self.returncode = -9
        def wait(self, timeout):
            if self.returncode is None: raise subprocess.TimeoutExpired("owned-verifier", timeout)
            return self.returncode
    monkeypatch.setattr(v.subprocess, "Popen", Process)
    monkeypatch.setattr(v, "MAX_VERIFY_SECONDS", 10)
    if mode == "oversize_log": monkeypatch.setattr(v, "MAX_VERIFY_LOG_BYTES", 10)
    kwargs = dict(stop=lambda: mode == "interrupted", monotonic=clock.monotonic, sleep=clock.sleep)
    if mode == "success":
        found, log = v.bounded_verify(data["contract"], data["sha"], sha, output, "e" * 64, **kwargs)
        assert found["verified"] and log["child_pid"] == 9876
    else:
        with pytest.raises(ValueError): v.bounded_verify(data["contract"], data["sha"], sha, output, "e" * 64, **kwargs)
    assert "--verify-child" in captured["command"] and "--expected-suite-sha256" in captured["command"]
    assert captured["command"][0] == sys.executable
    assert captured["kwargs"]["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert all(captured["kwargs"]["env"][key] == "8" for key in v.runner.THREAD_VARIABLES)
    assert captured["kwargs"]["stdin"] == subprocess.DEVNULL and captured["kwargs"]["start_new_session"] is True
    assert captured["terminated"] == (1 if mode in {"timeout", "interrupted", "stubborn"} else 0)
    assert captured["killed"] == (1 if mode == "stubborn" else 0)


@pytest.mark.parametrize("changes", [{"max_hours": 5}, {"max_hours": True}, {"max_hours": float("nan")},
    {"interval_seconds": 1}, {"interval_seconds": 61}, {"interval_seconds": True}])
def test_unbounded_or_untyped_options_rejected(setup, changes):
    with pytest.raises(ValueError): v.monitor(args(setup, **changes))


def test_monitor_cannot_write_into_training_outputs(setup):
    complete(setup)
    with pytest.raises(ValueError): v.monitor(args(setup, output_dir=setup["validation"] / "watch"))


def test_existing_monitor_never_overwritten(setup):
    v.monitor(args(setup), stop=lambda: True)
    path = setup["tmp"] / "monitor/receipt.json"
    before = path.read_bytes()
    with pytest.raises(ValueError): v.monitor(args(setup), stop=lambda: True)
    assert path.read_bytes() == before
