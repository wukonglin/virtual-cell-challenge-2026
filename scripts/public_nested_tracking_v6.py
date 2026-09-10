"""Private-by-default scalar tracking for nested public regularization v6.

Every scope names the OUTER held source, including inner events whose fitting
and tuning cells belong exclusively to the opposite source. Optional training
diagnostics share their evaluation event and never imply held-source fitting.
The caller supplies 33 events per learned arm and nine for the control; this
tracker records actual events, including partial failures, without inventing
completion or scientific success.
"""
from __future__ import annotations

import json

from public_nested_metrics_v6 import MMD_ROUNDOFF_TOLERANCE
from public_target_context_tracking import (ARMS, SCOPES, METRICS, FRACTIONS,
                                           TRAINING_METRICS, TRAINING_FRACTIONS)
from wandb_training import TrainingTracker, finite_scalar

REGULARIZATION_GRID = (0.1, 1.0, 10.0)
PHASES = ("inner_regularization", "nested_outer")
GROUP = "public-nested-regularization-v6"
RAW_METRICS = ("raw_model_mmd2", "raw_control_mmd2")


def checked_regularization(value):
    if not finite_scalar(value) or value not in REGULARIZATION_GRID:
        raise ValueError("Regularization must be one of the three registered finite numeric values")
    return float(value)


def scalar_nested(phase, scope, values, training=None):
    """Keep only finite allowlisted scalars; invalid values are not zero-filled."""
    if not isinstance(phase, str) or phase not in PHASES:
        return {}
    if not isinstance(scope, str) or scope not in SCOPES:
        return {}
    if phase == "inner_regularization" and scope == "all":
        return {}
    result = {}
    blocks = ((values, METRICS, FRACTIONS, f"validation/{phase}/{SCOPES[scope]}"),
              (training, TRAINING_METRICS, TRAINING_FRACTIONS,
               f"diagnostics/{phase}/{SCOPES[scope]}/training"))
    for payload, keys, fractions, prefix in blocks:
        if not isinstance(payload, dict):
            continue
        for key in keys:
            value = payload.get(key)
            if finite_scalar(value) and value >= 0 and (key not in fractions or value <= 1):
                result[f"{prefix}/{key}"] = value
    if isinstance(values, dict):
        for key in RAW_METRICS:
            value = values.get(key)
            if finite_scalar(value) and value >= -MMD_ROUNDOFF_TOLERANCE:
                suffix = "model" if key == "raw_model_mmd2" else "control"
                result[f"diagnostics/{phase}/{SCOPES[scope]}/raw_mmd2/{suffix}"] = value
    return result


class NestedValidationTracker(TrainingTracker):
    """Seven registered arms, separate inner/outer namespaces, offline only."""

    def __init__(self, output_dir, *, arm, config=None, sdk=None):
        if not isinstance(arm, str) or arm not in ARMS:
            raise ValueError("Expected one of the seven registered nested-validation arms")
        if config is not None and not isinstance(config, dict):
            raise ValueError("Tracking config must be a dictionary of allowlisted values")
        self.arm = arm
        super().__init__(output_dir, stage="validation", mode="offline",
                         project="virtual-cell-challenge-2026", group=GROUP,
                         name=arm, config=config, sdk=sdk)
        self.run.define_metric("analysis/event_index")
        for phase in PHASES:
            for namespace in ("validation", "diagnostics"):
                self.run.define_metric(f"{namespace}/{phase}/*", step_metric="analysis/event_index")

    def _log_nested(self, phase, scope, metrics, training, regularization):
        values = scalar_nested(phase, scope, metrics, training)
        if not values:
            return
        if regularization is not None:
            values[f"validation/{phase}/{SCOPES[scope]}/regularization"] = regularization
        self.events += 1
        values["analysis/event_index"] = self.events
        self.journal.write(json.dumps({"event_index": self.events, **values}, allow_nan=False) + "\n")
        try:
            self.run.log(values, step=self.events)
        except Exception:
            self.failures += 1

    def log_inner(self, held_outer_source, regularization, metrics, training=None):
        if self.arm == "control":
            raise ValueError("Control has no inner fitting or inner evaluation events")
        regularization = checked_regularization(regularization)
        self._log_nested("inner_regularization", held_outer_source, metrics, training, regularization)

    def log_outer(self, scope, metrics, training=None, regularization=None):
        if self.arm == "control" and regularization is not None:
            raise ValueError("Control has no selected regularization")
        if regularization is not None:
            regularization = checked_regularization(regularization)
        self._log_nested("nested_outer", scope, metrics, training, regularization)
