"""No real W&B, raw cells or credentials in these synthetic tests."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import public_reliability_tracking as tracking
from test_wandb_training import FakeSDK


@pytest.mark.parametrize("key", tracking.METRICS)
def test_allow_only_explicit_scalars(key):
    assert tracking.scalar_reliability("all", {key: 0.2, "cell_ids": ["private"], "secret": 0.4}) == {
        f"validation/reliability/all/{key}": 0.2}


@pytest.mark.parametrize("value", [True, None, "0.1", [], {}, float("nan"), float("inf")])
def test_nonfinite_or_nonscalar_are_not_logged(value):
    assert tracking.scalar_reliability("all", dict.fromkeys(tracking.METRICS, value)) == {}


def test_signed_statistics_and_bounded_agreement():
    result = tracking.scalar_reliability("replogle_k562", {
        "real_group_energy": -0.1, "null_half_disagreement_mse": -0.1,
        "real_half_crossdot": -0.1, "null_between_batch_crossdot": -0.2,
        "real_half_agreement": -1.0, "null_half_agreement": 1.001})
    assert result == {"validation/reliability/k562/real_half_crossdot": -0.1,
                      "validation/reliability/k562/null_between_batch_crossdot": -0.2,
                      "validation/reliability/k562/real_half_agreement": -1.0}
    assert tracking.scalar_reliability("/private/path", {"real_group_energy": 1.0}) == {}


def test_offline_allowlisted_journal_and_receipt(tmp_path):
    sdk = FakeSDK()
    tracker = tracking.ReliabilityTracker(tmp_path / "tracking", sdk=sdk,
        config={"diagnostic_contract_sha256": "a" * 64, "private_path": "/private"})
    tracker.log_reliability("nadig_jurkat", {"real_group_energy": 0.1, "cells": [[1.0]]})
    tracker.log_reliability("replogle_k562", {"real_group_energy": 0.2})
    tracker.log_reliability("all", {"real_group_energy": 0.15})
    tracker.finish(0)
    receipt = json.loads((tmp_path / "tracking/tracking.json").read_text())
    assert receipt["stage"] == "validation" and receipt["mode"] == "offline"
    assert receipt["event_count"] == 3 and receipt["tracking_errors"] == 0
    assert receipt["cloud_synced"] is False
    assert receipt["config"] == {"diagnostic_contract_sha256": "a" * 64}
    assert len(sdk.run.logged) == 3
    assert sdk.initialized[0]["settings"]["x_disable_stats"] is True
    assert sdk.initialized[0]["settings"]["disable_git"] is True


def test_logging_failure_is_visible_in_receipt(tmp_path):
    sdk = FakeSDK(log_error=True)
    tracker = tracking.ReliabilityTracker(tmp_path / "tracking", sdk=sdk)
    tracker.log_reliability("all", {"real_half_crossdot": -0.1})
    tracker.finish(0)
    receipt = json.loads((tmp_path / "tracking/tracking.json").read_text())
    assert receipt["tracking_errors"] == 1 and receipt["event_count"] == 1
    assert receipt["cloud_synced"] is False
