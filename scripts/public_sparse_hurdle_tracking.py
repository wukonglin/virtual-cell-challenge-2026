"""Offline, scalar-only tracking for the registered sparse hurdle v4 audit.

The caller supplies eight fixed fold records and one aggregate per completed
arm. Failed/partial runs retain the real event count. Execution completion is
not evidence that the scientific gate passed. Existing trackers stay frozen.
"""
from __future__ import annotations

import json

from wandb_training import TrainingTracker, finite_scalar

ARMS = ("control", "context", "true", "shuffled_20260921",
        "shuffled_20260922", "shuffled_20260923")
SCOPES = {"nadig_jurkat": "jurkat", "replogle_k562": "k562", "all": "all"}
METRICS = ("target_pooled_mse", "control_target_pooled_mse",
           "model_centroid_mse", "control_centroid_mse", "model_mmd2", "control_mmd2",
           "negative_fraction", "zero_fraction", "control_zero_fraction", "treated_zero_fraction",
           "target_pooled_occupancy_mse", "control_target_pooled_occupancy_mse",
           "unchanged_count_fraction", "occupancy_clip_fraction", "amplitude_clip_fraction",
           "unresolved_activation_fraction", "requested_realized_occupancy_mse")
FRACTIONS = frozenset(("negative_fraction", "zero_fraction", "control_zero_fraction",
                       "treated_zero_fraction", "unchanged_count_fraction", "occupancy_clip_fraction",
                       "amplitude_clip_fraction", "unresolved_activation_fraction"))
NONNEGATIVE = frozenset(METRICS)
PREFIX = "validation/sparse_hurdle"


def scalar_sparse(scope, values):
    """Keep only registered finite scalars; never clip invalid measurements.

MMD is the fixed biased, diagonal-included squared discrepancy, not a signed
unbiased estimator. No arbitrary labels, paths, identifiers, expression arrays,
count vectors, or scientific pass claims are admitted to the journal or SDK.
"""
    if not isinstance(scope, str) or scope not in SCOPES or not isinstance(values, dict):
        return {}
    result = {}
    for key in METRICS:
        value = values.get(key)
        if not finite_scalar(value) or value < 0:
            continue
        if key in FRACTIONS and value > 1:
            continue
        result[f"{PREFIX}/{SCOPES[scope]}/{key}"] = value
    return result


class SparseHurdleTracker(TrainingTracker):
    """Six fixed arms; offline validation with monotonic analysis event steps."""

    def __init__(self, output_dir, *, arm, config=None, sdk=None):
        if not isinstance(arm, str) or arm not in ARMS:
            raise ValueError("Expected one of the six registered sparse hurdle arms")
        if config is not None and not isinstance(config, dict):
            raise ValueError("Tracking config must be a dictionary of allowlisted values")
        super().__init__(output_dir, stage="validation", mode="offline",
                         project="virtual-cell-challenge-2026",
                         group="public-sparse-hurdle-v4", name=arm,
                         config=config, sdk=sdk)
        self.run.define_metric("analysis/event_index")
        self.run.define_metric(f"{PREFIX}/*", step_metric="analysis/event_index")

    def log_sparse(self, scope, values):
        metrics = scalar_sparse(scope, values)
        if not metrics:
            return
        self.events += 1
        metrics["analysis/event_index"] = self.events
        self.journal.write(json.dumps({"event_index": self.events, **metrics}, allow_nan=False) + "\n")
        try:
            self.run.log(metrics, step=self.events)
        except Exception:
            self.failures += 1
