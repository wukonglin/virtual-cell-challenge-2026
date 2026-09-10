"""Synthetic-only tracker checks: no real cells, GPU, cloud calls or credentials."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import public_joint_target_source_tracking as tracking
from test_wandb_training import FakeSDK


@pytest.mark.parametrize("key", tracking.METRICS)
def test_exact_metric_allowlist(key):
    assert tracking.scalar_joint("all", {key: 0.2, "cell_ids": ["private"],
        "data_path": "/private/cells", "quality_gate_passed": True, "secret": 2.0}) == {
        f"{tracking.PREFIX}/all/{key}": 0.2}


@pytest.mark.parametrize("value", [True, False, None, "0.1", [], {}, float("nan"),
                                    float("inf"), -float("inf"), 10 ** 1000])
def test_nonnumeric_nonfinite_and_huge_values_are_rejected(value):
    assert tracking.scalar_joint("all", dict.fromkeys(tracking.METRICS, value)) == {}


@pytest.mark.parametrize("scope", ["/private/path", "jurkat", "", "ALL", None, [], {}])
def test_scopes_are_exact(scope):
    assert tracking.scalar_joint(scope, {"target_pooled_mse": 0.1}) == {}


@pytest.mark.parametrize("values", [None, [], 0.1, "private"])
def test_payload_requires_dictionary(values):
    assert tracking.scalar_joint("all", values) == {}


@pytest.mark.parametrize("key", sorted(tracking.NONNEGATIVE))
def test_nonnegative_errors_and_corrections(key):
    assert tracking.scalar_joint("all", {key: -0.000001}) == {}
    assert tracking.scalar_joint("all", {key: 0.0}) == {f"{tracking.PREFIX}/all/{key}": 0.0}


@pytest.mark.parametrize("key", sorted(tracking.FRACTIONS))
def test_fraction_bounds(key):
    assert tracking.scalar_joint("all", {key: 1.000001}) == {}
    assert tracking.scalar_joint("all", {key: 1}) == {f"{tracking.PREFIX}/all/{key}": 1}


@pytest.mark.parametrize("value", [-1.000001, 1.000001])
def test_cosine_out_of_bounds_rejected(value):
    assert tracking.scalar_joint("all", {"shift_cosine": value}) == {}


@pytest.mark.parametrize("value", [-1, -0.2, 0, 1])
def test_signed_cosine_allowed(value):
    assert tracking.scalar_joint("nadig_jurkat", {"shift_cosine": value}) == {
        f"{tracking.PREFIX}/jurkat/shift_cosine": value}


def test_squared_metrics_have_no_arbitrary_upper_bound():
    assert tracking.scalar_joint("replogle_k562", {"target_pooled_mse": 3.5}) == {
        f"{tracking.PREFIX}/k562/target_pooled_mse": 3.5}


@pytest.mark.parametrize("arm", tracking.ARMS)
def test_each_fixed_arm_has_nine_offline_events_and_no_sensitive_config(tmp_path, arm):
    sdk = FakeSDK()
    output = tmp_path / "tracking"
    tracker = tracking.JointTargetSourceTracker(output, arm=arm, sdk=sdk, config={
        "diagnostic_contract_sha256": "a" * 64, "seed": 20260921,
        "private_path": "/private/data", "cell_ids": ["private"], "api_key": "SECRET",
        "entrypoint_sha256": "/private/script", "feature_mode": "SECRET"})
    for _ in range(4):
        tracker.log_joint("nadig_jurkat", {"target_pooled_mse": 0.1})
        tracker.log_joint("replogle_k562", {"target_pooled_mse": 0.2})
    tracker.log_joint("all", {"target_pooled_mse": 0.15})
    tracker.finish(0)

    receipt = json.loads((output / "tracking.json").read_text())
    journal = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    assert receipt["stage"] == "validation" and receipt["mode"] == "offline"
    assert receipt["project"] == "virtual-cell-challenge-2026"
    assert receipt["group"] == "public-joint-target-source-v3"
    assert receipt["event_count"] == 9 and receipt["tracking_errors"] == 0
    assert receipt["cloud_synced"] is False
    assert receipt["scientific_quality_inferred_from_exit"] is False
    assert receipt["raw_artifacts_uploaded"] is False
    assert receipt["config"] == {"diagnostic_contract_sha256": "a" * 64, "seed": 20260921}
    assert len(sdk.run.logged) == len(journal) == 9
    assert [row["event_index"] for row in journal] == list(range(1, 10))
    assert [row["analysis/event_index"] for row in journal] == list(range(1, 10))
    assert [step for _, step in sdk.run.logged] == list(range(1, 10))
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
        tracking.JointTargetSourceTracker(output, arm=arm, sdk=sdk)
    assert not output.exists() and not sdk.initialized


def test_invalid_config_rejected_before_side_effects(tmp_path):
    with pytest.raises(ValueError, match="dictionary"):
        tracking.JointTargetSourceTracker(tmp_path / "tracking", arm="true", config=["private"], sdk=FakeSDK())
    assert not (tmp_path / "tracking").exists()


def test_invalid_payload_does_not_increment_event_count(tmp_path):
    sdk = FakeSDK()
    tracker = tracking.JointTargetSourceTracker(tmp_path / "tracking", arm="control", sdk=sdk)
    tracker.log_joint("all", {"target_pooled_mse": -1.0, "zero_fraction": 2.0})
    tracker.log_joint("/private", {"target_pooled_mse": 0.1})
    tracker.log_joint("all", {})
    tracker.finish(0)
    assert tracker.events == 0 and sdk.run.logged == []
    assert (tmp_path / "tracking/metrics.jsonl").read_text() == ""


def test_logging_failure_preserves_journal_and_partial_run_receipt(tmp_path):
    sdk = FakeSDK(log_error=True)
    tracker = tracking.JointTargetSourceTracker(tmp_path / "tracking", arm="true", sdk=sdk)
    tracker.log_joint("all", {"shift_cosine": -0.1})
    tracker.finish(1)
    receipt = json.loads((tmp_path / "tracking/tracking.json").read_text())
    assert receipt["event_count"] == 1 and receipt["tracking_errors"] == 1
    assert receipt["execution_status"] == "failed" and receipt["cloud_synced"] is False
    assert len((tmp_path / "tracking/metrics.jsonl").read_text().splitlines()) == 1
