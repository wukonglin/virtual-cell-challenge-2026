"""Offline allowlist regression tests for the public GenePT/GO flow route."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import wandb_training as tracking  # noqa: E402


NEW_HASH_KEYS = ("split_sha256", "contract_sha256", "source_manifest_sha256",
                 "target_features_sha256", "context_transform_sha256")
NEW_FAMILIES = ("genept", "go", "genept_go")
NEW_DIAGNOSTICS = {"variance_ratio": 0.85, "covariance_effective_rank": 8.4,
                   "library_size_error": 0.12, "zero_fraction": 0.65,
                   "duplicate_fraction": 0.0}


@pytest.mark.parametrize("value,keep", [(0.0, True), (0.13, True), (1, True), (-0.1, False), (1.01, False), (True, False)])
def test_negative_expression_fraction_is_explicitly_recorded_and_bounded(value, keep):
    result = tracking.scalar_metrics({"kind_metrics": {"context": {"negative_fraction": value}}})
    assert result == ({"validation/context/negative_fraction": value} if keep else {})


@pytest.mark.parametrize("key,value", [("weight_decay", 0.0001), ("integration_steps", 16), ("shuffle_seed", 20260910)])
def test_frozen_pilot_hyperparameters_are_recorded_without_paths(key, value):
    assert tracking.safe_config({key: value}) == {key: value}
    assert tracking.config_from_command(["train.py", "--" + key.replace("_", "-"), str(value)]) == {key: value}
    assert tracking.safe_config({key: "/private/path"}) == {}


@pytest.mark.parametrize("family", NEW_FAMILIES)
def test_named_feature_arms_and_seeded_shuffles_are_recorded(family):
    shuffle = f"{family}_shuffled_20260909"
    result = tracking.scalar_metrics({"mse": {family: 0.12, shuffle: 0.2},
        "decisions": {family: {"conditioning_screen_passed": False, "private": "omit"}}})
    assert result == {f"cv/mse/{family}": 0.12, f"cv/mse/{shuffle}": 0.2,
                      f"quality/{family}_conditioning_screen_passed": 0}


@pytest.mark.parametrize("arm", ["genept_shuffled_42", "go_shuffled_202609090",
    "genept_go_shuffled_20260909/private", "GenePT", "unknown", "genept_seed_20260909"])
def test_unregistered_arms_and_malformed_shuffle_names_are_dropped(arm):
    assert not tracking.allowed_cv_arm(arm)
    assert tracking.scalar_metrics({"mse": {arm: 0.1},
        "decisions": {arm: {"conditioning_screen_passed": True}}}) == {}


@pytest.mark.parametrize("key", NEW_HASH_KEYS)
def test_exact_hash_keys_accept_config_and_both_command_flag_forms(key):
    digest = "a1" * 32
    flag = "--" + key.replace("_", "-")
    assert tracking.safe_config({key: digest}) == {key: digest}
    assert tracking.config_from_command(["python", "train.py", flag, digest]) == {key: digest}
    assert tracking.config_from_command(["python", "train.py", f"{flag}={digest}"]) == {key: digest}


@pytest.mark.parametrize("value", ["a" * 63, "a" * 65, "A" * 64, "/private/source.json",
                                  "https://private/source", "SECRET", None, ["a" * 64], 3])
def test_hash_validation_cannot_upload_paths_or_untyped_payloads(value):
    assert tracking.safe_config({key: value for key in NEW_HASH_KEYS}) == {}
    if isinstance(value, str):
        command = ["python", "train.py", "--source-manifest-sha256", value]
        assert tracking.config_from_command(command) == {}


def test_hash_support_does_not_expand_to_arbitrary_config_or_argv():
    config = {"unregistered_sha256": "a" * 64, "manifest": "/private/data.json",
              "raw_argv": ["secret"], "api_key": "secret", "context_transform": [[1, 2]]}
    assert tracking.safe_config(config) == {}
    assert tracking.config_from_command(["python", "/private/train.py", "--manifest", "/private/data.json",
        "--api-key=secret", "--unregistered-sha256", "a" * 64,
        "--split-sha256", "b" * 64]) == {"split_sha256": "b" * 64}


@pytest.mark.parametrize("kind", ["context", "target"])
@pytest.mark.parametrize("nested", [False, True])
def test_aggregate_diagnostics_require_registered_kind_metrics_scope(kind, nested):
    payload = {"kind_metrics": {kind: {**NEW_DIAGNOSTICS,
        "raw_counts": [[1, 2]], "gene": "SYNTHETIC_PRIVATE_GENE", "arbitrary_metric": 0.1}}}
    if nested:
        payload = {"deployment_rollout_diagnostics": payload}
    assert tracking.scalar_metrics(payload) == {
        f"validation/{kind}/{key}": value for key, value in NEW_DIAGNOSTICS.items()}
    assert tracking.scalar_metrics(NEW_DIAGNOSTICS) == {}
    assert tracking.scalar_metrics({"kind_metrics": {"private-donor": NEW_DIAGNOSTICS}}) == {}


@pytest.mark.parametrize("value", [True, "0.1", None, [0.1], {"value": 0.1},
                                  float("nan"), float("inf"), -0.1])
def test_new_diagnostics_reject_nonscalar_nonfinite_and_negative_values(value):
    payload = {"kind_metrics": {"target": {key: value for key in NEW_DIAGNOSTICS}}}
    assert tracking.scalar_metrics(payload) == {}


def test_fraction_bounds_do_not_cap_rank_or_variance_ratio():
    payload = {"kind_metrics": {"target": {"variance_ratio": 2.0,
        "covariance_effective_rank": 25.0, "library_size_error": 100.0,
        "zero_fraction": 1.01, "duplicate_fraction": 1.01}}}
    assert tracking.scalar_metrics(payload) == {"validation/target/variance_ratio": 2.0,
        "validation/target/covariance_effective_rank": 25.0,
        "validation/target/library_size_error": 100.0}
    assert tracking.scalar_metrics({"kind_metrics": {"context": {
        "zero_fraction": 1, "duplicate_fraction": 0, "covariance_effective_rank": 0}}}) == {
        "validation/context/zero_fraction": 1, "validation/context/duplicate_fraction": 0,
        "validation/context/covariance_effective_rank": 0}


def test_generic_status_and_nonnumeric_approval_do_not_become_quality_pass():
    assert tracking.scalar_metrics({"status": "pass", "passed": True}) == {}
    assert tracking.scalar_metrics({"schema": "private-generic-status", "status": "pass"}) == {}
    assert tracking.scalar_metrics({"decisions": {family: {"conditioning_screen_passed": "true"}
        for family in NEW_FAMILIES}}) == {}


def test_legacy_arms_config_and_signed_mmd_are_unchanged():
    legacy = ("true", "constant", "zero", "esm2", "lingshu", "fusion", "shuffled_20260909",
              "esm2_shuffled_20260909", "lingshu_shuffled_20260909", "fusion_shuffled_lingshu_20260909")
    result = tracking.scalar_metrics({"step": 1, "loss": 0.5, "mmd": -0.002,
        "mse": {arm: 0.1 for arm in legacy},
        "kind_metrics": {"context": {"model_mmd2": -0.001, "control_mmd2": -0.003}}})
    for arm in legacy:
        assert result[f"cv/mse/{arm}"] == 0.1
    assert result["train/global_step"] == 1
    assert result["train/mmd"] == -0.002
    assert result["validation/context/model_mmd2"] == -0.001
    assert result["validation/context/control_mmd2"] == -0.003
    assert tracking.safe_config({"steps": 20, "learning_rate": 0.001,
        "feature_mode": "true", "parent_checkpoint_sha256": "d" * 64}) == {
        "steps": 20, "learning_rate": 0.001, "feature_mode": "true",
        "parent_checkpoint_sha256": "d" * 64}
    assert tracking.scalar_metrics({"schema": "vcc-lingshu-scdfm-public-quality-preflight-v1",
                                    "status": "fail"}) == {"quality/public_gate_passed": 0}


def test_tracker_journal_rechecks_metrics_and_safely_records_summary_hashes(tmp_path):
    recorded = []

    class Config(dict):
        def update(self, values, *, allow_val_change=False):
            super().update(values)

    run = SimpleNamespace(define_metric=lambda *args, **kwargs: None,
        log=lambda data, step: recorded.append((dict(data), step)),
        summary={}, config=Config(), finish=lambda **kwargs: None)
    sdk = SimpleNamespace(run=None, Settings=lambda **kwargs: kwargs, init=lambda **kwargs: run)
    tracker = tracking.TrainingTracker(tmp_path / "tracking", stage="posttrain", sdk=sdk)
    tracker.log({"cv/mse/genept": 0.1, "cv/mse/genept_shuffled_20260909": 0.2,
        "validation/target/variance_ratio": -1, "validation/target/duplicate_fraction": 1.1,
        "validation/target/zero_fraction": 0.5, "validation/target/model_mmd2": -0.001,
        "validation/private/variance_ratio": 1, "private": "DO_NOT_SAVE"})
    tracker.summary({"source_manifest_sha256": "b" * 64, "split_sha256": "c" * 64,
        "target_features_sha256": "/private/table.npz", "raw_counts": [[1, 2]],
        "decisions": {"genept_go": {"conditioning_screen_passed": False}}})
    tracker.finish(0)
    assert recorded[0] == ({"cv/mse/genept": 0.1, "cv/mse/genept_shuffled_20260909": 0.2,
        "validation/target/zero_fraction": 0.5, "validation/target/model_mmd2": -0.001}, 1)
    assert recorded[1] == ({"quality/genept_go_conditioning_screen_passed": 0}, 2)
    assert run.config == {"source_manifest_sha256": "b" * 64, "split_sha256": "c" * 64}
    receipt = json.loads((tmp_path / "tracking" / "tracking.json").read_text())
    assert receipt["tracking_errors"] == 0
    assert receipt["cloud_synced"] is False
    assert receipt["config"] == run.config
    for filename in ("metrics.jsonl", "tracking.json"):
        content = (tmp_path / "tracking" / filename).read_text()
        assert "DO_NOT_SAVE" not in content
        assert "/private" not in content
