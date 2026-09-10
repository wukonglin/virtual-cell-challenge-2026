#!/usr/bin/env python3
"""Validate task-correlated Anvil training-to-inference lineage."""

from __future__ import annotations

import re
import math
from collections.abc import Mapping
from numbers import Integral, Real
from pathlib import Path
from collections.abc import Sequence
from typing import Any


TRAINING_RUNTIME_SCHEMA = "vcc-anvil-state-training-runtime-v2"
TRAINING_PARALLELISM = "independent_slurm_array_task_single_gpu_no_ddp"
ARCHIVE_CALLBACK_SCHEMA = "vcc-state-validation-step-archive-v1"


class LineageError(RuntimeError):
    """A required Slurm or upstream-training lineage assertion failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise LineageError(message)


def _canonical_unsigned_integer(value: Any, label: str) -> str:
    require(isinstance(value, str), f"{label} must be a string")
    require(re.fullmatch(r"0|[1-9][0-9]*", value) is not None, f"{label} is invalid")
    return value


def _canonical_positive_integer(value: Any, label: str) -> str:
    text = _canonical_unsigned_integer(value, label)
    require(text != "0", f"{label} must be positive")
    return text


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be a mapping")
    return value


def _integer(value: Any, label: str) -> int:
    require(
        isinstance(value, Integral) and not isinstance(value, bool),
        f"{label} must be an integer",
    )
    return int(value)


def _finite_float(value: Any, label: str) -> float:
    require(
        isinstance(value, Real) and not isinstance(value, bool),
        f"{label} must be numeric",
    )
    result = float(value)
    require(math.isfinite(result), f"{label} must be finite")
    return result


def validate_selected_earliest_argmin(
    validation_results: Any,
    archived_candidates: Any,
    selected: Any,
    *,
    expected_steps: Sequence[int],
) -> dict[str, Any]:
    """Recompute and bind the earliest finite validation-loss argmin.

    The selected row must equal the recomputed result and the unique archive
    row for that step must carry the same loss and checkpoint path.
    """

    require(isinstance(validation_results, list), "validation results must be a list")
    require(isinstance(archived_candidates, list), "archived candidates must be a list")
    require(isinstance(selected, Mapping), "selected checkpoint must be a mapping")

    validations: list[dict[str, Any]] = []
    for index, item in enumerate(validation_results):
        row = _mapping(item, f"validation result {index}")
        validations.append(
            {
                "global_step": _integer(
                    row.get("global_step"), f"validation result {index} global_step"
                ),
                "val_loss": _finite_float(
                    row.get("val_loss"), f"validation result {index} val_loss"
                ),
            }
        )
    expected = [int(step) for step in expected_steps]
    require(len(expected) == len(set(expected)), "expected validation steps are duplicated")
    require(
        sorted(row["global_step"] for row in validations) == sorted(expected),
        "validation cadence differs from the registered run",
    )

    archives: list[dict[str, Any]] = []
    for index, item in enumerate(archived_candidates):
        row = _mapping(item, f"archived candidate {index}")
        checkpoint = row.get("checkpoint")
        require(
            isinstance(checkpoint, str) and checkpoint != "",
            f"archived candidate {index} checkpoint is invalid",
        )
        archives.append(
            {
                "global_step": _integer(
                    row.get("global_step"), f"archived candidate {index} global_step"
                ),
                "val_loss": _finite_float(
                    row.get("val_loss"), f"archived candidate {index} val_loss"
                ),
                "checkpoint": checkpoint,
            }
        )
    require(
        sorted(row["global_step"] for row in archives) == sorted(expected),
        "archive cadence differs from the registered run",
    )

    validation_by_step = {row["global_step"]: row["val_loss"] for row in validations}
    archive_by_step = {row["global_step"]: row for row in archives}
    require(
        len(validation_by_step) == len(validations),
        "validation results contain duplicate steps",
    )
    require(len(archive_by_step) == len(archives), "archives contain duplicate steps")
    for step, validation_loss in validation_by_step.items():
        require(
            archive_by_step[step]["val_loss"] == validation_loss,
            f"archive loss differs from validation result at step {step}",
        )

    argmin = min(validations, key=lambda row: (row["val_loss"], row["global_step"]))
    selected_step = _integer(selected.get("global_step"), "selected global_step")
    selected_loss = _finite_float(selected.get("val_loss"), "selected val_loss")
    selected_checkpoint = selected.get("checkpoint")
    require(
        isinstance(selected_checkpoint, str) and selected_checkpoint != "",
        "selected checkpoint path is invalid",
    )
    require(
        selected_step == argmin["global_step"] and selected_loss == argmin["val_loss"],
        "selected checkpoint is not the earliest finite validation-loss argmin",
    )
    matching_archive = archive_by_step[selected_step]
    require(
        matching_archive["val_loss"] == selected_loss,
        "selected loss differs from its archived candidate row",
    )
    require(
        Path(matching_archive["checkpoint"]) == Path(selected_checkpoint),
        "selected checkpoint differs from its archived candidate row",
    )
    return {
        "global_step": selected_step,
        "val_loss": selected_loss,
        "checkpoint": selected_checkpoint,
    }


def validate_correlated_training_runtime(
    runtime: Mapping[str, Any],
    environment: Mapping[str, str],
    *,
    expected_seed: int,
    expected_run_name: str,
    registered_task_count: int = 4,
) -> dict[str, str]:
    """Authenticate one v2 training runtime against the running inference task.

    The dependency is intentionally exact rather than merely containing an
    ``aftercorr`` clause. This prevents an OR dependency or a second upstream
    job from weakening the one-training-task/one-inference-task relationship.
    """

    require(registered_task_count > 0, "registered task count must be positive")
    training_array_job_id = _canonical_positive_integer(
        environment.get("VCC_TRAINING_ARRAY_JOB_ID"),
        "VCC_TRAINING_ARRAY_JOB_ID",
    )
    inference_job_id = _canonical_positive_integer(
        environment.get("SLURM_JOB_ID"), "SLURM_JOB_ID"
    )
    inference_array_job_id = _canonical_positive_integer(
        environment.get("SLURM_ARRAY_JOB_ID"), "SLURM_ARRAY_JOB_ID"
    )
    inference_task_id = _canonical_unsigned_integer(
        environment.get("SLURM_ARRAY_TASK_ID"), "SLURM_ARRAY_TASK_ID"
    )
    inference_task_count = _canonical_positive_integer(
        environment.get("SLURM_ARRAY_TASK_COUNT"), "SLURM_ARRAY_TASK_COUNT"
    )
    require(
        int(inference_task_id) < registered_task_count,
        "inference array task is outside the registered task map",
    )
    require(
        int(inference_task_count) == registered_task_count,
        "inference array task count differs from the registered task map",
    )
    require(
        inference_array_job_id != training_array_job_id,
        "training and inference array job IDs must differ",
    )

    expected_dependency = f"aftercorr:{training_array_job_id}"
    dependency = environment.get("SLURM_JOB_DEPENDENCY")
    require(
        dependency == expected_dependency,
        f"SLURM_JOB_DEPENDENCY must be exactly {expected_dependency!r}",
    )

    require(
        runtime.get("schema") == TRAINING_RUNTIME_SCHEMA,
        "upstream training runtime schema mismatch",
    )
    require(
        runtime.get("run_name") == expected_run_name,
        "upstream training run name differs from the array registry",
    )
    registered_seed = runtime.get("registered_seed")
    require(
        isinstance(registered_seed, int)
        and not isinstance(registered_seed, bool)
        and registered_seed == expected_seed,
        "upstream training seed differs from the array registry",
    )
    training_devices = runtime.get("training_devices")
    require(
        isinstance(training_devices, int)
        and not isinstance(training_devices, bool)
        and training_devices == 1,
        "upstream training was not registered as a one-device fit",
    )
    require(
        runtime.get("parallelism") == TRAINING_PARALLELISM,
        "upstream training parallelism contract mismatch",
    )
    require(
        runtime.get("wandb_enabled") is False,
        "upstream training runtime does not record W&B disabled",
    )

    upstream_slurm = _mapping(runtime.get("slurm"), "upstream training slurm")
    upstream_array_job_id = _canonical_positive_integer(
        upstream_slurm.get("array_job_id"), "upstream training array_job_id"
    )
    upstream_task_id = _canonical_unsigned_integer(
        upstream_slurm.get("array_task_id"), "upstream training array_task_id"
    )
    upstream_task_count = _canonical_positive_integer(
        upstream_slurm.get("array_task_count"), "upstream training array_task_count"
    )
    upstream_job_id = _canonical_positive_integer(
        upstream_slurm.get("job_id"), "upstream training job_id"
    )
    require(
        upstream_array_job_id == training_array_job_id,
        "declared training array job ID differs from the upstream runtime",
    )
    require(
        upstream_task_id == inference_task_id,
        "upstream and inference array task IDs are not correlated",
    )
    require(
        int(upstream_task_count) == registered_task_count,
        "upstream array task count differs from the registered task map",
    )
    require(
        upstream_job_id != inference_job_id,
        "upstream training and inference task job IDs must differ",
    )

    source = _mapping(runtime.get("source"), "upstream training source")
    shim = _mapping(
        source.get("checkpoint_archive_shim"),
        "upstream checkpoint archive shim",
    )
    require(
        shim.get("callback_state_schema") == ARCHIVE_CALLBACK_SCHEMA,
        "upstream checkpoint archive callback schema mismatch",
    )

    return {
        "training_array_job_id": training_array_job_id,
        "training_array_task_id": upstream_task_id,
        "training_task_job_id": upstream_job_id,
        "inference_array_job_id": inference_array_job_id,
        "inference_array_task_id": inference_task_id,
        "inference_task_job_id": inference_job_id,
        "array_task_count": str(registered_task_count),
        "slurm_job_dependency": expected_dependency,
    }
