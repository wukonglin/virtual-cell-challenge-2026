"""Offline-only expanded-cohort tracking glue; no real SDK or public data."""
import json
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import public_expanded_tracking_v8 as tracking
import public_nested_tracking_v6 as prior
from test_wandb_training import FakeSDK


@pytest.fixture(autouse=True)
def clean_tracking_environment(monkeypatch):
    for key in tuple(os.environ):
        if key.startswith("WANDB_") or key in ("RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK"):
            monkeypatch.delenv(key, raising=False)


@pytest.mark.parametrize("arm", tracking.ARMS)
def test_separate_offline_group_exact_schedule_and_private_fields(tmp_path, arm):
    sdk = FakeSDK()
    output = tmp_path / "tracking"
    tracker = tracking.ExpandedValidationTracker(output, arm=arm, sdk=sdk, config={
        "diagnostic_contract_sha256": "a" * 64, "data_path": "/private", "secret": "PRIVATE"})
    for _ in range(4):
        for source in ("nadig_jurkat", "replogle_k562"):
            if arm != "control":
                for lam in prior.REGULARIZATION_GRID:
                    tracker.log_inner(source, lam, {"target_pooled_mse": .2}, {"effective_df": .3})
            tracker.log_outer(source, {"target_pooled_mse": .1},
                regularization=None if arm == "control" else 10.)
    tracker.log_outer("all", {"target_pooled_mse": .1})
    tracker.finish(0)
    receipt = json.loads((output / "tracking.json").read_text())
    journal = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    count = 9 if arm == "control" else 33
    assert receipt["event_count"] == count and receipt["tracking_errors"] == 0
    assert receipt["mode"] == "offline" and receipt["stage"] == "validation"
    assert receipt["group"] == tracking.GROUP != prior.GROUP
    assert receipt["cloud_synced"] is receipt["raw_artifacts_uploaded"] is False
    assert receipt["config"] == {"diagnostic_contract_sha256": "a" * 64}
    assert [row["analysis/event_index"] for row in journal] == list(range(1, count + 1))
    assert [step for _, step in sdk.run.logged] == list(range(1, count + 1))
    assert len([row for row in journal if any(k.startswith("validation/inner_regularization/") for k in row)]) == (0 if arm == "control" else 24)
    assert journal[-1] == {"event_index": count, "analysis/event_index": count,
                          "validation/nested_outer/all/target_pooled_mse": .1}
    assert all("PRIVATE" not in path.read_text() and "/private" not in path.read_text()
               for path in (output / "tracking.json", output / "metrics.jsonl"))
    settings = sdk.initialized[0]["settings"]
    assert settings["console"] == "off" and settings["save_code"] is False
    assert all(settings[k] is True for k in ("disable_git", "disable_code", "x_disable_stats", "x_disable_meta", "x_disable_machine_info"))
    for phase in tracking.PHASES:
        for namespace in ("validation", "diagnostics"):
            assert ((f"{namespace}/{phase}/*",), {"step_metric": "analysis/event_index"}) in sdk.run.defined


def test_total_events_match_frozen_budget():
    assert sum(9 if arm == "control" else 33 for arm in tracking.ARMS) == 207


@pytest.mark.parametrize("arm", [None, [], {}, "true/path", "shuffled_20260924", ""])
def test_bad_arm_has_no_side_effect(tmp_path, arm):
    sdk = FakeSDK()
    with pytest.raises(ValueError):
        tracking.ExpandedValidationTracker(tmp_path / "tracking", arm=arm, sdk=sdk)
    assert not (tmp_path / "tracking").exists() and not sdk.initialized


@pytest.mark.parametrize("config", [[], "private", 7])
def test_bad_config_has_no_side_effect(tmp_path, config):
    with pytest.raises(ValueError):
        tracking.ExpandedValidationTracker(tmp_path / "tracking", arm="true", config=config, sdk=FakeSDK())
    assert not (tmp_path / "tracking").exists()


def test_inherited_no_fake_events_and_control_no_selection(tmp_path):
    tracker = tracking.ExpandedValidationTracker(tmp_path / "tracking", arm="control", sdk=FakeSDK())
    with pytest.raises(ValueError):
        tracker.log_inner("nadig_jurkat", .1, {"target_pooled_mse": .1})
    with pytest.raises(ValueError):
        tracker.log_outer("nadig_jurkat", {"target_pooled_mse": .1}, regularization=10.)
    tracker.log_outer("all", {"target_pooled_mse": float("nan"), "private": 1.})
    tracker.finish(1)
    assert tracker.events == 0


@pytest.mark.parametrize("failure", ["log_error", "finish_error"])
def test_sdk_failure_not_reported_as_success(tmp_path, failure):
    tracker = tracking.ExpandedValidationTracker(tmp_path / "tracking", arm="true", sdk=FakeSDK(**{failure: True}))
    tracker.log_inner("nadig_jurkat", .1, {"target_pooled_mse": .1})
    tracker.finish(0)
    receipt = json.loads((tmp_path / "tracking/tracking.json").read_text())
    assert receipt["tracking_errors"] == 1
    assert receipt["cloud_synced"] is False
