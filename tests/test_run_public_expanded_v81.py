"""Narrow no-follow W&B log-link recovery; scientific v8 behavior stays frozen."""
import copy
import ast
import inspect
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_public_expanded_v81 as runner
import run_public_expanded_v8 as frozen
from test_run_public_expanded_v8 import roles_fixture


@pytest.fixture
def debug_link(tmp_path, monkeypatch):
    monkeypatch.setattr(runner.resource, "getrusage", lambda _: SimpleNamespace(ru_maxrss=1024))
    monkeypatch.setattr(runner.time, "monotonic", lambda: 10.)
    arm, run_id = "control", "abcdef123456"
    receipt = {"run_id": run_id, "mode": "offline", "group": runner.GROUP, "stage": "validation",
               "cloud_synced": False, "execution_status": "running"}
    receipt_path = tmp_path / "tracking" / arm / "tracking.json"
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_text(json.dumps(receipt))
    path = receipt_path.parent / "sdk/wandb" / ("offline-run-20260910_120000-" + run_id) / "logs/debug-core.log"
    path.parent.mkdir(parents=True)
    target = str(Path.home() / ".cache/wandb/logs/core-debug-20260910_114702.log")
    path.symlink_to(target)
    return tmp_path, path, target, receipt_path, receipt


def test_exact_debug_link_counted_without_resolve_open_stat_or_walk(debug_link, monkeypatch):
    root, link, target, receipt, _ = debug_link
    original_stat, original_open = Path.stat, os.open
    def guarded_stat(path, *args, **kwargs):
        if str(path) == target or (path == link and kwargs.get("follow_symlinks", True)):
            pytest.fail("External debug target was statted")
        return original_stat(path, *args, **kwargs)
    def guarded_open(path, *args, **kwargs):
        if str(path) in (str(link), target): pytest.fail("External debug log was opened")
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "stat", guarded_stat)
    monkeypatch.setattr(os, "open", guarded_open)
    monkeypatch.setattr(Path, "resolve", lambda *a, **k: pytest.fail("Debug path was resolved"))
    monkeypatch.setattr(os, "walk", lambda *a, **k: pytest.fail("os.walk may stat symlink targets during classification"))
    assert runner.sdk_debug_link_bytes(link, root) == len(target.encode())
    value = runner.resources(root, 0.)
    assert value["output_bytes"] == receipt.stat().st_size + len(target.encode())


@pytest.mark.parametrize("status", ["running", "completed", "failed"])
def test_expected_lifecycles_and_different_core_timestamp(debug_link, status):
    root, path, target, receipt_path, receipt = debug_link
    receipt["execution_status"] = status
    receipt_path.write_text(json.dumps(receipt))
    assert runner.sdk_debug_link_bytes(path, root) == len(target.encode())


@pytest.mark.parametrize("part", ["run_id", "mode", "group", "stage", "cloud", "bool_cloud", "lifecycle", "missing"])
def test_debug_link_tracking_binding_fail_closed(debug_link, part):
    root, path, _, receipt_path, receipt = debug_link
    if part == "run_id": receipt["run_id"] = "fedcba654321"
    elif part == "mode": receipt["mode"] = "online"
    elif part == "group": receipt["group"] = "public-nested-regularization-v6"
    elif part == "stage": receipt["stage"] = "pretrain"
    elif part == "cloud": receipt["cloud_synced"] = True
    elif part == "bool_cloud": receipt["cloud_synced"] = 0
    elif part == "lifecycle": receipt["execution_status"] = "tracking_initialization_failed"
    else: receipt.pop("run_id")
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError): runner.sdk_debug_link_bytes(path, root)


@pytest.mark.parametrize("part", ["wrong_home", "relative", "traversal", "wrong_cache", "wrong_name", "suffix", "bad_stamp", "model_target"])
def test_debug_link_wrong_or_external_target_rejected(debug_link, part):
    root, path, target, _, _ = debug_link
    if part == "wrong_home": target = "/home/someone_else/.cache/wandb/logs/core-debug-20260910_114702.log"
    elif part == "relative": target = "core-debug-20260910_114702.log"
    elif part == "traversal": target = str(Path.home() / ".cache/wandb/logs/../logs/core-debug-20260910_114702.log")
    elif part == "wrong_cache": target = str(Path.home() / ".cache/elsewhere/logs/core-debug-20260910_114702.log")
    elif part == "wrong_name": target = str(Path.home() / ".cache/wandb/logs/debug-core.log")
    elif part == "suffix": target += ".other"
    elif part == "bad_stamp": target = str(Path.home() / ".cache/wandb/logs/core-debug-20260910_11470.log")
    else: target = str(root / "model.npz")
    path.unlink(); path.symlink_to(target)
    with pytest.raises(ValueError): runner.resources(root, 0.)


@pytest.mark.parametrize("relative", [
    "model.npz", "runs/control/debug-core.log", "tracking/control/debug-core.log",
    "tracking/unknown/sdk/wandb/offline-run-20260910_120000-abcdef123456/logs/debug-core.log",
    "tracking/control/sdk/wandb/run-20260910_120000-abcdef123456/logs/debug-core.log",
    "tracking/control/sdk/wandb/offline-run-20260910_120000-abcdef12345/logs/debug-core.log",
    "tracking/control/sdk/wandb/offline-run-20260910_120000-abcdef123456/logs/debug-coreXlog",
    "tracking/control/sdk/wandb/offline-run-20260910_120000-abcdef123456/logs/debug.log",
    "tracking/control/sdk/wandb/offline-run-20260910_120000-abcdef123456/logs/debug-core.log/child",
])
def test_only_exact_sdk_relative_link_path_permitted(tmp_path, relative):
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(Path.home() / ".cache/wandb/logs/core-debug-20260910_114702.log")
    with pytest.raises(ValueError): runner.sdk_debug_link_bytes(path, tmp_path)


@pytest.mark.parametrize("kind", ["directory_symlink", "fifo", "receipt_symlink", "receipt_fifo", "regular_debug_name"])
def test_nonregular_or_changed_type_still_rejected(debug_link, kind):
    root, path, _, receipt_path, _ = debug_link
    if kind == "directory_symlink":
        (root / "directory").mkdir(); (root / "linked_directory").symlink_to(root / "directory", target_is_directory=True)
    elif kind == "fifo": os.mkfifo(root / "pipe")
    elif kind == "receipt_symlink":
        moved = root / "receipt.json"; receipt_path.rename(moved); receipt_path.symlink_to(moved)
    elif kind == "receipt_fifo": receipt_path.unlink(); os.mkfifo(receipt_path)
    else:
        path.unlink(); path.write_text("ordinary log")
        with pytest.raises((ValueError, OSError)): runner.sdk_debug_link_bytes(path, root)
        return
    with pytest.raises((ValueError, OSError)): runner.resources(root, 0.)


def test_scientific_model_screen_budget_and_tracking_remain_frozen():
    assert runner.SCHEMA != frozen.SCHEMA and runner.SUMMARY_SCHEMA == frozen.SUMMARY_SCHEMA
    assert runner.model is frozen.model and runner.preparation is frozen.preparation
    for key in ("model", "screen", "cpu_threads", "maximum_peak_rss_bytes", "maximum_output_bytes",
                "maximum_elapsed_seconds", "gpu_used", "events_by_arm", "total_events", "fitted_two_head_models"):
        assert runner.policy()[key] == frozen.policy()[key]
    assert runner.ExpandedValidationTracker is frozen.ExpandedValidationTracker
    assert set(frozen.MODULES) <= set(runner.MODULES)


def test_scientific_and_execution_function_asts_unchanged_except_entrypoint_name():
    # Recovery must not silently change role checks, fitting order, metric
    # projection, gates, replay, or the exact tracking event reconstruction.
    names = ("require_roles", "provenance", "screen_projection", "verify_summary", "screen",
             "verify_audit", "runtime_preflight", "expected_journal", "verify_tracking", "run")
    for name in names:
        source = inspect.getsource(getattr(runner, name)).replace("run_public_expanded_v81.py", "run_public_expanded_v8.py")
        expected = inspect.getsource(getattr(frozen, name))
        assert ast.dump(ast.parse(source)) == ast.dump(ast.parse(expected)), name


@pytest.fixture
def registration(tmp_path, monkeypatch):
    roles, nested = roles_fixture()
    protocol = tmp_path / "protocol.md"; protocol.write_text("New operational recovery protocol, same science")
    role_path = tmp_path / "roles.json"; role_path.write_text("{}")
    record = {"protocol": {"path": str(protocol), "sha256": runner.safe.digest(protocol)}, "inputs": {"cache_sha256": "a" * 64}}
    parent = {"contract": {"path": str(tmp_path / "failed-contract.json"), "sha256": "b" * 64},
              "tracking_failure": {"path": str(tmp_path / "failed-tracking.json"), "sha256": "c" * 64}}
    monkeypatch.setattr(runner.preparation, "validate_registration", lambda *a, **k: (copy.deepcopy(record), copy.deepcopy(roles), copy.deepcopy(nested)))
    monkeypatch.setattr(runner, "operational_parent", lambda: copy.deepcopy(parent))
    monkeypatch.setattr(runner, "codes", lambda: {"run_public_expanded_v81.py": "d" * 64})
    output = tmp_path / "execution"
    sha = runner.register(role_path, runner.safe.digest(role_path), protocol, output)
    path = output / "contract.json"
    value, _, _, _ = runner.validate_contract(path, sha)
    return path, value, parent


def test_recovery_registration_records_failed_parent(registration):
    _, value, parent = registration
    assert value["schema"] == runner.SCHEMA
    assert value["operational_parent"] == parent and value["expression_decoded"] is False


@pytest.mark.parametrize("part", ["extra", "parent_sha", "parent_path", "schema", "policy", "code"])
def test_rehashed_recovery_contract_changes_rejected(registration, part):
    path, value, _ = registration
    if part == "extra": value["operational_parent"]["extra"] = True
    elif part == "parent_sha": value["operational_parent"]["tracking_failure"]["sha256"] = "e" * 64
    elif part == "parent_path": value["operational_parent"]["contract"]["path"] += "x"
    elif part == "schema": value["schema"] = frozen.SCHEMA
    elif part == "policy": value["policy"] = frozen.policy()
    else: value["code_sha256"]["run_public_expanded_v81.py"] = "e" * 64
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError): runner.validate_contract(path, runner.safe.digest(path))


@pytest.mark.parametrize("part", ["valid", "event", "success", "online", "completion", "dangling_completion", "tampered_bytes"])
def test_operational_parent_pins_failed_zero_event_receipt(tmp_path, monkeypatch, part):
    monkeypatch.setattr(runner, "FAILED_ROOT", tmp_path)
    calls = []
    monkeypatch.setattr(runner.operational_prior, "validate_contract", lambda p, s: calls.append((p, s)))
    path = tmp_path / "validation/tracking/control/tracking.json"
    path.parent.mkdir(parents=True)
    value = {"execution_status": "failed", "training_exit_code": 1, "event_count": 0,
             "cloud_synced": False, "mode": "offline", "stage": "validation"}
    if part == "event": value["event_count"] = 1
    elif part == "success": value["execution_status"] = "completed"
    elif part == "online": value["mode"] = "online"
    path.write_text(json.dumps(value))
    monkeypatch.setattr(runner, "FAILED_TRACKING_SHA", runner.safe.digest(path))
    completion = tmp_path / "validation/complete.json"
    if part == "completion": completion.write_text("{}")
    elif part == "dangling_completion": completion.symlink_to(tmp_path / "missing.json")
    elif part == "tampered_bytes": path.write_text(json.dumps({**value, "ignored": True}))
    if part == "valid":
        result = runner.operational_parent()
        assert result["tracking_failure"]["sha256"] == runner.FAILED_TRACKING_SHA
        assert calls == [(tmp_path / "contract.json", runner.FAILED_CONTRACT_SHA)]
    else:
        with pytest.raises(ValueError): runner.operational_parent()
