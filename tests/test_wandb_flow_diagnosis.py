"""Bounded scalar logging for conservative public mean-effect diagnostics."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import wandb_training as tracking


@pytest.mark.parametrize("key,value,keep", [
    ("predicted_shift_rms", 0.0, True), ("predicted_shift_rms", 2.5, True),
    ("observed_shift_rms", 0.2, True), ("observed_shift_rms", -1.0, False),
    ("shift_cosine", -1.0, True), ("shift_cosine", 1.0, True),
    ("shift_cosine", -1.001, False), ("shift_cosine", 1.001, False),
    ("shift_dot_mean", -0.12, True), ("shift_dot_mean", 0.0, True),
])
def test_shift_values_respect_units_and_signed_direction(key, value, keep):
    payload = {"kind_metrics": {"context": {key: value}}}
    assert tracking.scalar_metrics(payload) == (
        {f"validation/context/{key}": value} if keep else {})


@pytest.mark.parametrize("value", [True, None, "0.1", [], {}, float("nan"), float("inf")])
def test_shift_values_must_be_finite_scalars(value):
    assert tracking.scalar_metrics({"kind_metrics": {"context": {
        key: value for key in tracking.SHIFT_DIAGNOSTICS}}}) == {}


def test_only_registered_aggregate_keys_and_hashes_leave_summary():
    digest = "a" * 64
    assert tracking.safe_config({"diagnostic_contract_sha256": digest,
        "diagnostic_contract": "/private/contract.json"}) == {"diagnostic_contract_sha256": digest}
    assert tracking.config_from_command(["--diagnostic-contract-sha256", digest]) == {
        "diagnostic_contract_sha256": digest}
    assert tracking.safe_config({"diagnostic_contract_sha256": "/private"}) == {}
    assert tracking.scalar_metrics({"shift_cosine": 0.4, "groups": [{"shift_cosine": 0.8}]}) == {}
    assert tracking.scalar_metrics({"deployment_rollout_diagnostics": {"kind_metrics": {
        "context": {"shift_cosine": 0.4, "cell_ids": ["private"], "new_unregistered": 0.9}}}}) == {
            "validation/context/shift_cosine": 0.4}
