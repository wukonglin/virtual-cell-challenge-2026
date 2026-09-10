"""Only bounded aggregate diagnostics, never cell arrays, enter tracking."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import wandb_training as tracking


@pytest.mark.parametrize("key", tracking.ANCHOR_DIAGNOSTICS)
def test_anchor_metrics_allow_finite_aggregate(key):
    expected = {f"validation/context/{key}": 0.1}
    assert tracking.scalar_metrics({"kind_metrics": {"context": {key: 0.1}}}) == expected


@pytest.mark.parametrize("value", [True, None, "0.1", [], {}, float("inf"), float("nan")])
def test_anchor_metrics_reject_invalid(value):
    assert not tracking.scalar_metrics({"kind_metrics": {"context": {
        key: value for key in tracking.ANCHOR_DIAGNOSTICS}}})


@pytest.mark.parametrize("key", ["projection_fraction", "raw_negative_fraction"])
@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_fractions_are_bounded(key, value):
    assert not tracking.valid_diagnostic(key, value)


def test_nonnegative_magnitudes_signed_mmd_and_private_fields():
    for key in set(tracking.ANCHOR_DIAGNOSTICS) - {"raw_model_mmd2", "ntc_sham_mmd2"}:
        assert not tracking.valid_diagnostic(key, -0.1)
    assert tracking.valid_diagnostic("raw_model_mmd2", -1e-12)
    assert tracking.scalar_metrics({"kind_metrics": {"context": {
        "predictions": [[1.0]], "cell_ids": ["private"], "projection_fraction": 0.1}}}) == {
            "validation/context/projection_fraction": 0.1}
