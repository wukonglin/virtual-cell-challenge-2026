"""Synthetic tracker checks: fake SDK only, no cells, credentials, or cloud."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import public_sparse_hurdle_tracking as tracking
from test_wandb_training import FakeSDK


@pytest.fixture(autouse=True)
def isolated_tracking_environment(monkeypatch):
    for key in tuple(os.environ):
        if key.startswith("WANDB_") or key in ("RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK"):
            monkeypatch.delenv(key, raising=False)


@pytest.mark.parametrize("key", tracking.METRICS)
def test_exact_metric_allowlist_and_privacy(key):
    assert tracking.scalar_sparse("all", {key: 0.2, "cell_ids": ["private"],
        "data_path": "/private/cells", "quality_gate_passed": True, "secret": 2.0,
        "activated_cells": [1, 2], "target": "private", "shift_cosine": .8}) == {
        f"{tracking.PREFIX}/all/{key}": 0.2}


@pytest.mark.parametrize("value", [True, False, None, "0.1", [], {}, float("nan"),
                                    float("inf"), -float("inf"), 10 ** 1000])
def test_nonnumeric_nonfinite_and_huge_values_are_rejected(value):
    assert tracking.scalar_sparse("all", dict.fromkeys(tracking.METRICS, value)) == {}


@pytest.mark.parametrize("scope", ["/private/path", "jurkat", "k562", "", "ALL", None, [], {}])
def test_scopes_are_exact(scope):
    assert tracking.scalar_sparse(scope, {"target_pooled_mse": 0.1}) == {}


@pytest.mark.parametrize("values", [None, [], 0.1, "private"])
def test_payload_requires_dictionary(values):
    assert tracking.scalar_sparse("all", values) == {}


@pytest.mark.parametrize("key", tracking.METRICS)
def test_all_metrics_nonnegative_without_silent_clipping(key):
    assert tracking.scalar_sparse("all", {key: -0.000001}) == {}
    assert tracking.scalar_sparse("all", {key: 0.0}) == {f"{tracking.PREFIX}/all/{key}": 0.0}


@pytest.mark.parametrize("key", sorted(tracking.FRACTIONS))
def test_fraction_upper_bound(key):
    assert tracking.scalar_sparse("all", {key: 1.000001}) == {}
    assert tracking.scalar_sparse("all", {key: 1}) == {f"{tracking.PREFIX}/all/{key}": 1}


@pytest.mark.parametrize("key", sorted(tracking.NONNEGATIVE - tracking.FRACTIONS))
def test_squared_metrics_have_no_arbitrary_upper_bound(key):
    assert tracking.scalar_sparse("replogle_k562", {key: 3.5}) == {
        f"{tracking.PREFIX}/k562/{key}": 3.5}


@pytest.mark.parametrize("arm", tracking.ARMS)
def test_each_fixed_arm_has_nine_offline_events_and_no_sensitive_config(tmp_path, arm):
    sdk = FakeSDK()
    output = tmp_path / "tracking"
    tracker = tracking.SparseHurdleTracker(output, arm=arm, sdk=sdk, config={
        "diagnostic_contract_sha256": "a" * 64, "seed": 20260921,
        "private_path": "/private/data", "cell_ids": ["private"], "api_key": "SECRET",
        "entrypoint_sha256": "/private/script", "feature_mode": "SECRET"})
    for _ in range(4):
        tracker.log_sparse("nadig_jurkat", dict.fromkeys(tracking.METRICS, 0.1))
        tracker.log_sparse("replogle_k562", dict.fromkeys(tracking.METRICS, 0.2))
    tracker.log_sparse("all", dict.fromkeys(tracking.METRICS, 0.15))
    tracker.finish(0)

    receipt = json.loads((output / "tracking.json").read_text())
    journal = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    assert receipt["stage"] == "validation" and receipt["mode"] == "offline"
    assert receipt["project"] == "virtual-cell-challenge-2026"
    assert receipt["group"] == "public-sparse-hurdle-v4"
    assert receipt["event_count"] == 9 and receipt["tracking_errors"] == 0
    assert receipt["cloud_synced"] is False
    assert receipt["scientific_quality_inferred_from_exit"] is False
    assert receipt["raw_artifacts_uploaded"] is False
    assert receipt["config"] == {"diagnostic_contract_sha256": "a" * 64, "seed": 20260921}
    assert len(sdk.run.logged) == len(journal) == 9
    assert [row["event_index"] for row in journal] == list(range(1, 10))
    assert [row["analysis/event_index"] for row in journal] == list(range(1, 10))
    assert [step for _, step in sdk.run.logged] == list(range(1, 10))
    for row in journal:
        assert len(row) == len(tracking.METRICS) + 2
        assert all(k.startswith(tracking.PREFIX + "/") or k in {"event_index", "analysis/event_index"} for k in row)
    assert sdk.initialized[0]["name"] == arm
    assert sdk.initialized[0]["mode"] == "offline"
    assert sdk.initialized[0]["config"] == {"stage": "validation", **receipt["config"]}
    assert ((f"{tracking.PREFIX}/*",), {"step_metric": "analysis/event_index"}) in sdk.run.defined
    settings = sdk.initialized[0]["settings"]
    assert settings["console"] == "off"
    for key in ("disable_git", "disable_code", "x_disable_stats", "x_disable_meta",
                "x_disable_machine_info", "disable_job_creation"):
        assert settings[key] is True
    assert settings["save_code"] is False and settings["x_save_requirements"] is False
    assert "private" not in (output / "tracking.json").read_text()
    assert "SECRET" not in (output / "tracking.json").read_text()


@pytest.mark.parametrize("arm", ["shuffled", "shuffled_20260924", "constant", "true/path", None, [], {}])
def test_invalid_arm_rejected_before_side_effects(tmp_path, arm):
    sdk = FakeSDK()
    output = tmp_path / "tracking"
    with pytest.raises(ValueError, match="six registered"):
        tracking.SparseHurdleTracker(output, arm=arm, sdk=sdk)
    assert not output.exists() and not sdk.initialized


def test_invalid_config_rejected_before_side_effects(tmp_path):
    with pytest.raises(ValueError, match="dictionary"):
        tracking.SparseHurdleTracker(tmp_path / "tracking", arm="true", config=["private"], sdk=FakeSDK())
    assert not (tmp_path / "tracking").exists()


def test_invalid_payload_does_not_increment_event_count(tmp_path):
    sdk = FakeSDK()
    tracker = tracking.SparseHurdleTracker(tmp_path / "tracking", arm="control", sdk=sdk)
    tracker.log_sparse("all", {"target_pooled_mse": -1.0, "zero_fraction": 2.0})
    tracker.log_sparse("/private", {"target_pooled_mse": 0.1})
    tracker.log_sparse("all", {})
    tracker.finish(0)
    assert tracker.events == 0 and sdk.run.logged == []
    assert (tmp_path / "tracking/metrics.jsonl").read_text() == ""


@pytest.mark.parametrize("log_error,finish_error", [(True, False), (False, True), (True, True)])
def test_sdk_failures_preserve_journal_and_actual_partial_run_status(tmp_path, log_error, finish_error):
    sdk = FakeSDK(log_error=log_error, finish_error=finish_error)
    tracker = tracking.SparseHurdleTracker(tmp_path / "tracking", arm="true", sdk=sdk)
    tracker.log_sparse("all", {"unresolved_activation_fraction": 0.1})
    tracker.finish(17)
    receipt = json.loads((tmp_path / "tracking/tracking.json").read_text())
    assert receipt["event_count"] == 1
    assert receipt["tracking_errors"] == int(log_error) + int(finish_error)
    assert receipt["execution_status"] == "failed" and receipt["training_exit_code"] == 17
    assert receipt["cloud_synced"] is False
    assert len((tmp_path / "tracking/metrics.jsonl").read_text().splitlines()) == 1


def test_partial_interruption_not_misreported_as_completed_arm(tmp_path):
    tracker = tracking.SparseHurdleTracker(tmp_path / "tracking", arm="context", sdk=FakeSDK())
    tracker.log_sparse("nadig_jurkat", {"zero_fraction": .6})
    tracker.finish(130, interrupted=True)
    receipt = json.loads((tmp_path / "tracking/tracking.json").read_text())
    assert receipt["execution_status"] == "interrupted"
    assert receipt["event_count"] == 1 and receipt["scientific_quality_inferred_from_exit"] is False


def test_sdk_initialization_failure_is_recorded_without_secret(tmp_path):
    class FailedSDK(FakeSDK):
        def init(self, **kwargs):
            raise RuntimeError("SECRET initialization failure")
    with pytest.raises(RuntimeError):
        tracking.SparseHurdleTracker(tmp_path / "tracking", arm="true", sdk=FailedSDK())
    text = (tmp_path / "tracking/tracking.json").read_text()
    assert json.loads(text)["execution_status"] == "tracking_initialization_failed"
    assert "SECRET" not in text
