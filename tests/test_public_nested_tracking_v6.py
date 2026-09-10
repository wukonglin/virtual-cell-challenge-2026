"""Synthetic nested-stage accounting, scalar privacy and fake-SDK failures."""
import json
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import public_nested_tracking_v6 as tracking
import public_target_context_tracking as frozen
from test_wandb_training import FakeSDK


@pytest.fixture(autouse=True)
def isolated_tracking_environment(monkeypatch):
    for key in tuple(os.environ):
        if key.startswith("WANDB_") or key in ("RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK"):
            monkeypatch.delenv(key, raising=False)


def test_registered_grid_arms_and_allowlists_are_exact():
    assert tracking.REGULARIZATION_GRID == (.1, 1., 10.)
    for key in ("ARMS", "METRICS", "FRACTIONS", "TRAINING_METRICS", "TRAINING_FRACTIONS"):
        assert getattr(tracking, key) == getattr(frozen, key)


@pytest.mark.parametrize("value", [True, False, None, "0.1", [], {}, float("nan"), float("inf"),
                                  -float("inf"), 10**1000, 0., -1., .01, .10000001, 100.])
def test_invalid_regularization_raises_without_events(tmp_path, value):
    tracker = tracking.NestedValidationTracker(tmp_path / "tracking", arm="true", sdk=FakeSDK())
    with pytest.raises(ValueError):
        tracker.log_inner("nadig_jurkat", value, {"model_mmd2": 0.})
    # None is the intentional no-selected-lambda outer aggregate default.
    if value is not None:
        with pytest.raises(ValueError):
            tracker.log_outer("nadig_jurkat", {"model_mmd2": 0.}, regularization=value)
    tracker.finish(1)
    assert tracker.events == 0


@pytest.mark.parametrize("value", [.1, 1., 10., 1, 10])
def test_grid_values_are_recorded_as_numeric_lambda(value):
    assert tracking.checked_regularization(value) == float(value)


@pytest.mark.parametrize("phase", tracking.PHASES)
@pytest.mark.parametrize("key", tracking.METRICS)
def test_validation_allowlist_rejects_metadata_and_training_keys(phase, key):
    values = {key: .2, "cell_ids": ["private"], "data_path": "/private", "quality_gate_passed": True,
              "coefficients": [[1.]], "effective_df": 3., "secret": 1.}
    assert tracking.scalar_nested(phase, "nadig_jurkat", values) == {
        f"validation/{phase}/jurkat/{key}": .2}


@pytest.mark.parametrize("key", tracking.TRAINING_METRICS)
def test_training_namespace_contains_only_allowlisted_scalars(key):
    values = {key: .2, "targets": ["private"], "data_path": "/private", "weights": [[1.]],
              "target_pooled_mse": .2, "quality_gate_passed": True}
    assert tracking.scalar_nested("nested_outer", "replogle_k562", {}, values) == {
        f"diagnostics/nested_outer/k562/training/{key}": .2}


@pytest.mark.parametrize("value", [True, False, None, "0.1", [], {}, float("nan"), float("inf"),
                                  -float("inf"), 10**1000])
def test_bad_scalars_are_omitted_never_zero_filled(value):
    assert tracking.scalar_nested("nested_outer", "all", dict.fromkeys((*tracking.METRICS, *tracking.RAW_METRICS), value),
                                  dict.fromkeys(tracking.TRAINING_METRICS, value)) == {}


@pytest.mark.parametrize("key", (*tracking.METRICS, *tracking.TRAINING_METRICS))
def test_nonnegative_metrics_and_fraction_bounds(key):
    is_training = key in tracking.TRAINING_METRICS
    def scalar(value):
        return tracking.scalar_nested("nested_outer", "all", {} if is_training else {key: value},
                                      {key: value} if is_training else None)
    assert scalar(-1e-15) == {}
    assert list(scalar(0.).values()) == [0.]
    fractions = tracking.FRACTIONS | tracking.TRAINING_FRACTIONS
    if key in fractions:
        assert scalar(1.01) == {}
    else:
        assert list(scalar(1.01).values()) == [1.01]


@pytest.mark.parametrize("payload", [None, [], "private", .1])
def test_invalid_optional_block_does_not_discard_valid_scalars(payload):
    assert tracking.scalar_nested("nested_outer", "all", {"model_mmd2": 0.}, payload) == {
        "validation/nested_outer/all/model_mmd2": 0.}
    assert tracking.scalar_nested("nested_outer", "all", payload, {"effective_df": .1}) == {
        "diagnostics/nested_outer/all/training/effective_df": .1}


@pytest.mark.parametrize("phase", tracking.PHASES)
def test_signed_raw_mmd_separate_from_nonnegative_validation_namespace(phase):
    values = {"model_mmd2": 0., "control_mmd2": -1e-13, "raw_model_mmd2": -1e-12,
              "raw_control_mmd2": -1e-13, "model_mmd2_roundoff_correction": 1e-12}
    assert tracking.scalar_nested(phase, "nadig_jurkat", values) == {
        f"validation/{phase}/jurkat/model_mmd2": 0.,
        f"diagnostics/{phase}/jurkat/raw_mmd2/model": -1e-12,
        f"diagnostics/{phase}/jurkat/raw_mmd2/control": -1e-13}
    assert tracking.scalar_nested(phase, "nadig_jurkat", {"raw_model_mmd2": -1.000001e-12}) == {}


@pytest.mark.parametrize("scope", ["jurkat", "k562", "/private", "", None, [], {}])
def test_scope_labels_are_exact(scope):
    assert tracking.scalar_nested("nested_outer", scope, {"model_mmd2": .1}) == {}


def test_inner_all_scope_and_unknown_phase_are_not_admitted():
    assert tracking.scalar_nested("inner_regularization", "all", {"model_mmd2": .1}) == {}
    assert tracking.scalar_nested("private", "all", {"model_mmd2": .1}) == {}


@pytest.mark.parametrize("arm", tracking.ARMS)
def test_offline_nested_event_schedule_and_privacy(tmp_path, arm):
    sdk = FakeSDK(); output = tmp_path / "tracking"
    tracker = tracking.NestedValidationTracker(output, arm=arm, sdk=sdk, config={
        "diagnostic_contract_sha256": "a"*64, "seed": 20260920,
        "data_path": "/private/data", "cell_ids": ["private"], "api_key": "SECRET"})
    for _ in range(4):
        for source in ("nadig_jurkat", "replogle_k562"):
            if arm != "control":
                for regularization in tracking.REGULARIZATION_GRID:
                    tracker.log_inner(source, regularization, dict.fromkeys(tracking.METRICS, .1),
                                      dict.fromkeys(tracking.TRAINING_METRICS, .2))
            tracker.log_outer(source, dict.fromkeys(tracking.METRICS, .15),
                              None if arm == "control" else dict.fromkeys(tracking.TRAINING_METRICS, .3),
                              regularization=None if arm == "control" else .1)
    tracker.log_outer("all", dict.fromkeys(tracking.METRICS, .15))
    tracker.finish(0)
    receipt = json.loads((output / "tracking.json").read_text())
    journal = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    expected = 9 if arm == "control" else 33
    assert receipt["event_count"] == expected and receipt["tracking_errors"] == 0
    assert receipt["stage"] == "validation" and receipt["mode"] == "offline"
    assert receipt["group"] == tracking.GROUP and receipt["cloud_synced"] is False
    assert receipt["scientific_quality_inferred_from_exit"] is receipt["raw_artifacts_uploaded"] is False
    assert receipt["config"] == {"diagnostic_contract_sha256": "a"*64, "seed": 20260920}
    assert [row["analysis/event_index"] for row in journal] == list(range(1, expected+1))
    assert [step for _, step in sdk.run.logged] == list(range(1, expected+1))
    inner = [r for r in journal if any(k.startswith("validation/inner_regularization") for k in r)]
    assert len(inner) == (0 if arm == "control" else 24)
    assert len(journal) - len(inner) == 9
    assert not any("regularization" in key or "diagnostics" in key for key in journal[-1])
    assert not any("private" in text or "SECRET" in text for text in ((output / "tracking.json").read_text(),
                                                                      (output / "metrics.jsonl").read_text()))
    for phase in tracking.PHASES:
        for namespace in ("validation", "diagnostics"):
            assert ((f"{namespace}/{phase}/*",), {"step_metric": "analysis/event_index"}) in sdk.run.defined
    settings = sdk.initialized[0]["settings"]
    assert settings["console"] == "off" and settings["save_code"] is False
    for key in ("disable_git", "disable_code", "x_disable_stats", "x_disable_meta", "x_disable_machine_info"):
        assert settings[key] is True
    assert settings["x_save_requirements"] is False


def test_total_registered_event_budget_is_207():
    assert 9 + (len(tracking.ARMS)-1)*33 == 207


def test_control_has_no_inner_event_or_selected_regularization(tmp_path):
    tracker = tracking.NestedValidationTracker(tmp_path / "tracking", arm="control", sdk=FakeSDK())
    with pytest.raises(ValueError, match="no inner"):
        tracker.log_inner("nadig_jurkat", .1, {"model_mmd2": .1})
    with pytest.raises(ValueError, match="no selected"):
        tracker.log_outer("nadig_jurkat", {"model_mmd2": .1}, regularization=1.)
    tracker.finish(0)
    assert tracker.events == 0


@pytest.mark.parametrize("arm", [None, [], {}, "true/path", "constant", "shuffled_20260924"])
def test_invalid_arm_rejected_before_side_effect(tmp_path, arm):
    sdk = FakeSDK()
    with pytest.raises(ValueError, match="seven registered"):
        tracking.NestedValidationTracker(tmp_path / "tracking", arm=arm, sdk=sdk)
    assert not (tmp_path / "tracking").exists() and not sdk.initialized


def test_invalid_config_rejected_before_side_effect(tmp_path):
    with pytest.raises(ValueError, match="dictionary"):
        tracking.NestedValidationTracker(tmp_path / "tracking", arm="true", config=["private"], sdk=FakeSDK())
    assert not (tmp_path / "tracking").exists()


def test_invalid_metric_blocks_do_not_emit_lambda_only_or_fake_zero_events(tmp_path):
    sdk = FakeSDK(); tracker = tracking.NestedValidationTracker(tmp_path / "tracking", arm="true", sdk=sdk)
    tracker.log_inner("nadig_jurkat", .1, {"model_mmd2": -1}, {"effective_df": float("nan")})
    tracker.log_inner("private", 1., {"model_mmd2": .1})
    tracker.log_outer("all", {}, regularization=10.)
    tracker.finish(0)
    assert tracker.events == 0 and sdk.run.logged == []


@pytest.mark.parametrize("log_error,finish_error", [(True, False), (False, True), (True, True)])
def test_sdk_failures_retain_journal_and_actual_failed_partial_status(tmp_path, log_error, finish_error):
    sdk = FakeSDK(log_error=log_error, finish_error=finish_error)
    tracker = tracking.NestedValidationTracker(tmp_path / "tracking", arm="true", sdk=sdk)
    tracker.log_inner("nadig_jurkat", .1, {"model_mmd2": 0.}, {"effective_df": .2})
    tracker.finish(17)
    receipt = json.loads((tmp_path / "tracking/tracking.json").read_text())
    assert receipt["event_count"] == 1 and receipt["tracking_errors"] == int(log_error)+int(finish_error)
    assert receipt["execution_status"] == "failed" and receipt["training_exit_code"] == 17
    assert receipt["cloud_synced"] is False
    assert len((tmp_path / "tracking/metrics.jsonl").read_text().splitlines()) == 1


def test_interrupted_partial_run_is_not_completed(tmp_path):
    tracker = tracking.NestedValidationTracker(tmp_path / "tracking", arm="context", sdk=FakeSDK())
    tracker.log_inner("replogle_k562", 1., {"model_mmd2": .1})
    tracker.finish(130, interrupted=True)
    receipt = json.loads((tmp_path / "tracking/tracking.json").read_text())
    assert receipt["execution_status"] == "interrupted" and receipt["event_count"] == 1
