"""Synthetic scalar/privacy/failure tests, using a fake SDK and no real data."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import public_target_context_tracking as tracking
import public_sparse_hurdle_tracking as frozen
from test_wandb_training import FakeSDK


@pytest.fixture(autouse=True)
def isolated_tracking_environment(monkeypatch):
    for key in tuple(os.environ):
        if key.startswith("WANDB_") or key in ("RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK"):
            monkeypatch.delenv(key, raising=False)


def test_validation_allowlist_is_exact_frozen_v4_list():
    assert tracking.METRICS == frozen.METRICS
    assert tracking.FRACTIONS == frozen.FRACTIONS
    assert tracking.ARMS == ("control", "context", "additive", "true", *frozen.ARMS[3:])


@pytest.mark.parametrize("key", tracking.METRICS)
def test_validation_allowlist_drops_sensitive_and_training_fields(key):
    values = {key: .2, "cell_ids": ["private"], "data_path": "/private", "quality_gate_passed": True,
              "coefficient_matrix": [[1.]], "effective_df": 9., "secret": 1.}
    assert tracking.scalar_kernel("all", values) == {f"{tracking.PREFIX}/all/{key}": .2}


@pytest.mark.parametrize("key", tracking.TRAINING_METRICS)
def test_training_allowlist_drops_sensitive_and_validation_fields(key):
    training = {key: .2, "targets": ["private"], "data_path": "/private", "fitted_head": [[1.]],
                "target_pooled_mse": .2, "quality_gate_passed": True, "secret": 1.}
    assert tracking.scalar_kernel("nadig_jurkat", {}, training) == {
        f"{tracking.TRAINING_PREFIX}/jurkat/training/{key}": .2}


@pytest.mark.parametrize("value", [True, False, None, "0.1", [], {}, float("nan"), float("inf"), -float("inf"), 10 ** 1000])
def test_nonnumeric_nonfinite_and_huge_values_are_omitted_not_zeroed(value):
    assert tracking.scalar_kernel("all", dict.fromkeys(tracking.METRICS, value),
                                  dict.fromkeys(tracking.TRAINING_METRICS, value)) == {}


@pytest.mark.parametrize("scope", ["/private/path", "jurkat", "k562", "", "ALL", None, [], {}])
def test_scopes_are_exact_for_both_namespaces(scope):
    assert tracking.scalar_kernel(scope, {"target_pooled_mse": .1}, {"effective_df": 1.}) == {}


@pytest.mark.parametrize("payload", [None, [], 0.1, "private"])
def test_invalid_optional_payload_does_not_discard_other_valid_block(payload):
    assert tracking.scalar_kernel("all", {"target_pooled_mse": .1}, payload) == {
        f"{tracking.PREFIX}/all/target_pooled_mse": .1}
    assert tracking.scalar_kernel("all", payload, {"effective_df": 1.}) == {
        f"{tracking.TRAINING_PREFIX}/all/training/effective_df": 1.}


@pytest.mark.parametrize("key", tracking.METRICS)
def test_validation_metrics_nonnegative_and_fraction_bounds(key):
    assert tracking.scalar_kernel("all", {key: -.00001}) == {}
    assert tracking.scalar_kernel("all", {key: 0.}) == {f"{tracking.PREFIX}/all/{key}": 0.}
    expected = {} if key in tracking.FRACTIONS else {f"{tracking.PREFIX}/all/{key}": 1.01}
    assert tracking.scalar_kernel("all", {key: 1.01}) == expected


@pytest.mark.parametrize("key", tracking.TRAINING_METRICS)
def test_training_metrics_nonnegative_and_fraction_bounds(key):
    assert tracking.scalar_kernel("all", {}, {key: -.00001}) == {}
    assert tracking.scalar_kernel("all", {}, {key: 0.}) == {f"{tracking.TRAINING_PREFIX}/all/training/{key}": 0.}
    expected = {} if key in tracking.TRAINING_FRACTIONS else {f"{tracking.TRAINING_PREFIX}/all/training/{key}": 1.01}
    assert tracking.scalar_kernel("all", {}, {key: 1.01}) == expected


@pytest.mark.parametrize("arm", tracking.ARMS)
def test_seven_fixed_arms_nine_offline_events_with_training_in_same_step(tmp_path, arm):
    sdk = FakeSDK()
    output = tmp_path / "tracking"
    tracker = tracking.TargetContextTracker(output, arm=arm, sdk=sdk, config={
        "diagnostic_contract_sha256": "a" * 64, "seed": 20260921,
        "data_path": "/private/data", "cell_ids": ["private"], "api_key": "SECRET",
        "entrypoint_sha256": "/private/script", "feature_mode": "SECRET"})
    for _ in range(4):
        tracker.log_kernel("nadig_jurkat", dict.fromkeys(tracking.METRICS, .1), dict.fromkeys(tracking.TRAINING_METRICS, .2))
        tracker.log_kernel("replogle_k562", dict.fromkeys(tracking.METRICS, .2), dict.fromkeys(tracking.TRAINING_METRICS, .3))
    tracker.log_kernel("all", dict.fromkeys(tracking.METRICS, .15))
    tracker.finish(0)
    receipt = json.loads((output / "tracking.json").read_text())
    journal = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    assert receipt["stage"] == "validation" and receipt["mode"] == "offline"
    assert receipt["project"] == "virtual-cell-challenge-2026"
    assert receipt["group"] == "public-target-context-v5"
    assert receipt["event_count"] == 9 and receipt["tracking_errors"] == 0
    assert receipt["cloud_synced"] is False and receipt["raw_artifacts_uploaded"] is False
    assert receipt["scientific_quality_inferred_from_exit"] is False
    assert receipt["config"] == {"diagnostic_contract_sha256": "a" * 64, "seed": 20260921}
    assert len(sdk.run.logged) == len(journal) == 9
    assert [row["event_index"] for row in journal] == list(range(1, 10))
    assert [row["analysis/event_index"] for row in journal] == list(range(1, 10))
    assert [step for _, step in sdk.run.logged] == list(range(1, 10))
    for row in journal[:8]:
        assert len(row) == 2 + len(tracking.METRICS) + len(tracking.TRAINING_METRICS)
    assert len(journal[-1]) == 2 + len(tracking.METRICS)
    assert not any(key.startswith(tracking.TRAINING_PREFIX) for key in journal[-1])
    assert f"{tracking.TRAINING_PREFIX}/jurkat/training/effective_df" in journal[0]
    assert sdk.initialized[0]["name"] == arm and sdk.initialized[0]["mode"] == "offline"
    assert sdk.initialized[0]["config"] == {"stage": "validation", **receipt["config"]}
    for prefix in (tracking.PREFIX, tracking.TRAINING_PREFIX):
        assert ((f"{prefix}/*",), {"step_metric": "analysis/event_index"}) in sdk.run.defined
    settings = sdk.initialized[0]["settings"]
    assert settings["console"] == "off"
    for key in ("disable_git", "disable_code", "x_disable_stats", "x_disable_meta", "x_disable_machine_info", "disable_job_creation"):
        assert settings[key] is True
    assert settings["save_code"] is False and settings["x_save_requirements"] is False
    assert "private" not in (output / "tracking.json").read_text()
    assert "SECRET" not in (output / "tracking.json").read_text()


@pytest.mark.parametrize("arm", ["shuffled", "shuffled_20260924", "constant", "true/path", None, [], {}])
def test_invalid_arm_fails_before_any_side_effect(tmp_path, arm):
    sdk = FakeSDK()
    with pytest.raises(ValueError, match="seven registered"):
        tracking.TargetContextTracker(tmp_path / "tracking", arm=arm, sdk=sdk)
    assert not (tmp_path / "tracking").exists() and not sdk.initialized


def test_invalid_config_fails_before_any_side_effect(tmp_path):
    with pytest.raises(ValueError, match="dictionary"):
        tracking.TargetContextTracker(tmp_path / "tracking", arm="true", config=["private"], sdk=FakeSDK())
    assert not (tmp_path / "tracking").exists()


def test_invalid_metrics_never_create_fake_zero_validation_events(tmp_path):
    sdk = FakeSDK()
    tracker = tracking.TargetContextTracker(tmp_path / "tracking", arm="control", sdk=sdk)
    tracker.log_kernel("all", {"target_pooled_mse": -1.}, {"kernel_trace": float("nan")})
    tracker.log_kernel("/private", {"target_pooled_mse": .1}, {"effective_df": 1.})
    tracker.log_kernel("all", {}, {})
    tracker.finish(0)
    assert tracker.events == 0 and sdk.run.logged == []
    assert (tmp_path / "tracking/metrics.jsonl").read_text() == ""


@pytest.mark.parametrize("log_error,finish_error", [(True, False), (False, True), (True, True)])
def test_sdk_failures_preserve_journal_and_actual_partial_status(tmp_path, log_error, finish_error):
    sdk = FakeSDK(log_error=log_error, finish_error=finish_error)
    tracker = tracking.TargetContextTracker(tmp_path / "tracking", arm="true", sdk=sdk)
    tracker.log_kernel("nadig_jurkat", {"unchanged_count_fraction": 1.}, {"effective_df": .2})
    tracker.finish(17)
    receipt = json.loads((tmp_path / "tracking/tracking.json").read_text())
    assert receipt["event_count"] == 1 and receipt["tracking_errors"] == int(log_error) + int(finish_error)
    assert receipt["execution_status"] == "failed" and receipt["training_exit_code"] == 17
    assert receipt["cloud_synced"] is False
    assert len((tmp_path / "tracking/metrics.jsonl").read_text().splitlines()) == 1


def test_interrupted_run_retains_actual_event_count(tmp_path):
    tracker = tracking.TargetContextTracker(tmp_path / "tracking", arm="context", sdk=FakeSDK())
    tracker.log_kernel("nadig_jurkat", {"zero_fraction": .6})
    tracker.finish(130, interrupted=True)
    receipt = json.loads((tmp_path / "tracking/tracking.json").read_text())
    assert receipt["execution_status"] == "interrupted" and receipt["event_count"] == 1
    assert receipt["scientific_quality_inferred_from_exit"] is False
