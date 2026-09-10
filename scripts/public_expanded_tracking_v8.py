"""Separate offline W&B group for expanded-cohort nested validation."""
from public_nested_tracking_v6 import NestedValidationTracker, ARMS, PHASES
from wandb_training import TrainingTracker

GROUP = "public-expanded-regularization-v8"


class ExpandedValidationTracker(NestedValidationTracker):
    """Reuse frozen scalar filtering and event methods, never its run identity."""

    def __init__(self, output_dir, *, arm, config=None, sdk=None):
        if not isinstance(arm, str) or arm not in ARMS:
            raise ValueError("Expected a registered expanded-validation arm")
        if config is not None and not isinstance(config, dict):
            raise ValueError("Tracking config must be a dictionary")
        self.arm = arm
        TrainingTracker.__init__(self, output_dir, stage="validation", mode="offline",
            project="virtual-cell-challenge-2026", group=GROUP, name=arm, config=config, sdk=sdk)
        self.run.define_metric("analysis/event_index")
        for phase in PHASES:
            for namespace in ("validation", "diagnostics"):
                self.run.define_metric(f"{namespace}/{phase}/*", step_metric="analysis/event_index")
