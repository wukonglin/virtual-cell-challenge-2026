"""Opt-in scalar-only reliability tracking; the frozen v2 tracker is unchanged."""
from __future__ import annotations
import json

from wandb_training import TrainingTracker, finite_scalar

SCOPES = {"nadig_jurkat": "jurkat", "replogle_k562": "k562", "all": "all"}
METRICS = tuple(f"{kind}_{metric}" for kind in ("real", "null") for metric in
                ("group_energy", "target_energy", "half_crossdot", "half_agreement",
                 "between_batch_crossdot", "half_disagreement_mse"))


def scalar_reliability(scope, values):
    if scope not in SCOPES or not isinstance(values, dict):
        return {}
    result = {}
    for key in METRICS:
        value = values.get(key)
        if not finite_scalar(value):
            continue
        if (key.endswith("energy") or key.endswith("disagreement_mse")) and value < 0:
            continue
        if key.endswith("agreement") and not -1 <= value <= 1:
            continue
        result[f"validation/reliability/{SCOPES[scope]}/{key}"] = value
    return result


class ReliabilityTracker(TrainingTracker):
    def __init__(self, output_dir, *, config=None, sdk=None):
        super().__init__(output_dir, stage="validation", mode="offline",
                         project="virtual-cell-challenge-2026", group="public-effect-reliability-v1",
                         name="matched-null-and-batch-reliability", config=config, sdk=sdk)
        self.run.define_metric("analysis/event_index")
        self.run.define_metric("validation/reliability/*", step_metric="analysis/event_index")

    def log_reliability(self, scope, values):
        metrics = scalar_reliability(scope, values)
        if not metrics:
            return
        self.events += 1
        metrics["analysis/event_index"] = self.events
        self.journal.write(json.dumps({"event_index": self.events, **metrics}, allow_nan=False) + "\n")
        try:
            self.run.log(metrics, step=self.events)
        except Exception:
            self.failures += 1
