"""Offline scalar tracking for the registered target/context kernel v5 audit.

Each completed arm supplies eight fold events plus one validation aggregate.
The source scope names the HELD-source evaluation fold. Optional training
diagnostics refer to that fold's opposite-source fit; their scope must not be
interpreted as training on the held source. They share the same event/step and
never create a separate event. Missing or invalid scalars are omitted, not
zero-filled. This tracker does not infer scientific success or upload arrays.
"""
from __future__ import annotations

import json

from wandb_training import TrainingTracker, finite_scalar

ARMS = ("control", "context", "additive", "true", "shuffled_20260921",
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
TRAINING_METRICS = ("effective_df", "kernel_trace", "occupancy_response_rms", "occupancy_fitted_rms",
                    "amplitude_response_rms", "amplitude_fitted_rms", "training_unchanged_count_fraction")
TRAINING_FRACTIONS = frozenset(("training_unchanged_count_fraction",))
PREFIX = "validation/target_context"
TRAINING_PREFIX = "diagnostics/target_context"


def scalar_kernel(scope, values, training=None):
    """Admit only registered finite nonnegative scalars and bounded fractions.

MMD is the registered biased squared discrepancy. Arbitrary labels, raw
expressions, identifiers, coefficient arrays, paths and quality claims never
reach either the local journal or SDK. Training metrics have a distinct
namespace, indexed by the held-source fold rather than their fitting source.
"""
    if not isinstance(scope, str) or scope not in SCOPES:
        return {}
    result = {}
    blocks = ((values, METRICS, FRACTIONS, f"{PREFIX}/{SCOPES[scope]}"),
              (training, TRAINING_METRICS, TRAINING_FRACTIONS,
               f"{TRAINING_PREFIX}/{SCOPES[scope]}/training"))
    for payload, keys, fractions, prefix in blocks:
        if not isinstance(payload, dict):
            continue
        for key in keys:
            value = payload.get(key)
            if not finite_scalar(value) or value < 0 or (key in fractions and value > 1):
                continue
            result[f"{prefix}/{key}"] = value
    return result


class TargetContextTracker(TrainingTracker):
    """Seven fixed arms, offline validation, nine caller-supplied final events."""

    def __init__(self, output_dir, *, arm, config=None, sdk=None):
        if not isinstance(arm, str) or arm not in ARMS:
            raise ValueError("Expected one of the seven registered target/context arms")
        if config is not None and not isinstance(config, dict):
            raise ValueError("Tracking config must be a dictionary of allowlisted values")
        super().__init__(output_dir, stage="validation", mode="offline",
                         project="virtual-cell-challenge-2026", group="public-target-context-v5",
                         name=arm, config=config, sdk=sdk)
        self.run.define_metric("analysis/event_index")
        self.run.define_metric(f"{PREFIX}/*", step_metric="analysis/event_index")
        self.run.define_metric(f"{TRAINING_PREFIX}/*", step_metric="analysis/event_index")

    def log_kernel(self, scope, metrics, training=None):
        values = scalar_kernel(scope, metrics, training)
        if not values:
            return
        self.events += 1
        values["analysis/event_index"] = self.events
        self.journal.write(json.dumps({"event_index": self.events, **values}, allow_nan=False) + "\n")
        try:
            self.run.log(values, step=self.events)
        except Exception:
            self.failures += 1
