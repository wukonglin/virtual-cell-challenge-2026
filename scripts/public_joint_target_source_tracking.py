"""Offline, scalar-only tracking for the opt-in joint target/source v3 gate.

The caller supplies eight fixed fold records and one final aggregate per arm.
Partial/failed runs retain their actual event count; successful execution or
tracking does not imply that the scientific quality gate passed. Frozen tracking
and model code are deliberately unchanged.
"""
from __future__ import annotations

import json

from wandb_training import TrainingTracker, finite_scalar

ARMS = ("control", "context", "true", "shuffled_20260921",
        "shuffled_20260922", "shuffled_20260923")
SCOPES = {"nadig_jurkat": "jurkat", "replogle_k562": "k562", "all": "all"}
METRICS = ("target_pooled_mse", "control_target_pooled_mse", "raw_target_pooled_mse",
           "model_centroid_mse", "control_centroid_mse", "model_mmd2", "control_mmd2",
           "negative_fraction", "zero_fraction", "control_zero_fraction",
           "projection_fraction", "projection_mean_absolute_correction",
           "projection_centroid_distortion_rms", "shift_cosine")
FRACTIONS = frozenset(("negative_fraction", "zero_fraction", "control_zero_fraction",
                       "projection_fraction"))
NONNEGATIVE = frozenset(METRICS) - {"shift_cosine"}
PREFIX = "validation/joint_target_source"


def scalar_joint(scope, values):
    """Drop unregistered, nonscalar, nonfinite or out-of-range values.

MMD here is the registered biased, diagonal-included squared discrepancy; this
new tracker requires nonnegative values rather than silently clipping negatives.
No arbitrary labels, paths, cell identifiers or scientific pass claims survive.
"""
    if not isinstance(scope, str) or scope not in SCOPES or not isinstance(values, dict):
        return {}
    result = {}
    for key in METRICS:
        value = values.get(key)
        if not finite_scalar(value):
            continue
        if key in NONNEGATIVE and value < 0:
            continue
        if key in FRACTIONS and value > 1:
            continue
        if key == "shift_cosine" and not -1 <= value <= 1:
            continue
        result[f"{PREFIX}/{SCOPES[scope]}/{key}"] = value
    return result


class JointTargetSourceTracker(TrainingTracker):
    """Six fixed arms; offline validation with monotonic analysis event steps."""

    def __init__(self, output_dir, *, arm, config=None, sdk=None):
        if not isinstance(arm, str) or arm not in ARMS:
            raise ValueError("Expected one of the six registered joint target/source arms")
        if config is not None and not isinstance(config, dict):
            raise ValueError("Tracking config must be a dictionary of allowlisted values")
        super().__init__(output_dir, stage="validation", mode="offline",
                         project="virtual-cell-challenge-2026",
                         group="public-joint-target-source-v3", name=arm,
                         config=config, sdk=sdk)
        self.run.define_metric("analysis/event_index")
        self.run.define_metric(f"{PREFIX}/*", step_metric="analysis/event_index")

    def log_joint(self, scope, values):
        metrics = scalar_joint(scope, values)
        if not metrics:
            return
        self.events += 1
        metrics["analysis/event_index"] = self.events
        self.journal.write(json.dumps({"event_index": self.events, **metrics}, allow_nan=False) + "\n")
        try:
            self.run.log(metrics, step=self.events)
        except Exception:
            self.failures += 1
