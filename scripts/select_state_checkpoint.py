#!/usr/bin/env python3
"""Select and authenticate the minimum-validation-loss STATE checkpoint.

The preferred layout remains validation-aligned ``stepNNNNN.ckpt`` archives.
Every archive is restricted-loaded and authenticated against both metrics.csv
and the route-owned callback state embedded in the checkpoint.  For the pinned
STATE runtime, which writes only ``best.ckpt`` plus ``last.ckpt``, the selector
supports that exact two-file layout as a fail-closed fallback.  A fallback
source is eligible only when its root ``global_step`` is the earliest step
attaining the minimum finite ``val_loss`` and its serialized callback state
independently corroborates that loss.

Both modes publish the existing v1 JSON contract and a relative
``selected.ckpt`` symlink without overwriting either output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import stat
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


SCHEMA = "vcc-state-checkpoint-selection-v1"
STEP_ARCHIVE_LAYOUT = "validation_step_archives"
BEST_LAST_LAYOUT = "pinned_state_best_last_only"
CALLBACK_SCORE_RTOL = 1e-6
CALLBACK_SCORE_ATOL = 1e-8
ARCHIVE_CALLBACK_STATE_KEY = "VCCValidationStepArchive"
ARCHIVE_CALLBACK_SCHEMA = "vcc-state-validation-step-archive-v1"
ARCHIVE_CALLBACK_HOOK = "on_validation_end"
ARCHIVE_PUBLISH_METHOD = "same_directory_temporary_then_atomic_hard_link_no_overwrite"
ARCHIVE_LOSS_RTOL = 1e-7
ARCHIVE_LOSS_ATOL = 1e-9


class CheckpointSelectionError(RuntimeError):
    """A checkpoint-selection precondition failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckpointSelectionError(message)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--metrics",
        type=Path,
        default=None,
        help="Defaults to RUN_DIR/version_0/metrics.csv.",
    )
    parser.add_argument(
        "--output-link",
        type=Path,
        default=None,
        help="Defaults to RUN_DIR/checkpoints/selected.ckpt.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Defaults to RUN_DIR/checkpoints/selected_checkpoint.json.",
    )
    return parser.parse_args(argv)


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _refuse_existing(path: Path, label: str) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {label}: {path}")


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish JSON atomically without replacing an existing path or symlink."""

    path = Path(os.path.abspath(os.fspath(path.expanduser())))
    _refuse_existing(path, "selection JSON")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(12)}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(temporary, flags, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as error:
            raise FileExistsError(f"Refusing to overwrite selection JSON: {path}") from error
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def atomic_symlink_no_overwrite(path: Path, target: str) -> None:
    """Publish a relative symlink atomically while preserving no-overwrite."""

    path = Path(os.path.abspath(os.fspath(path.expanduser())))
    require(not os.path.isabs(target), "selected checkpoint symlink target must be relative")
    _refuse_existing(path, "selection link")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(12)}.tmp")
    try:
        os.symlink(target, temporary)
        try:
            # With follow_symlinks=False, link(2) hard-links the symlink inode;
            # unlike os.replace, it can never overwrite a racing destination.
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as error:
            raise FileExistsError(f"Refusing to overwrite selection link: {path}") from error
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def atomic_hardlink_no_overwrite(source: Path, destination: Path) -> None:
    """Materialize an archive with one atomic link(2), never a checkpoint copy."""

    source = Path(os.path.abspath(os.fspath(source.expanduser())))
    destination = Path(os.path.abspath(os.fspath(destination.expanduser())))
    _refuse_existing(destination, "validation-step archive")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination, follow_symlinks=False)
    except FileExistsError as error:
        raise FileExistsError(
            f"Refusing to overwrite validation-step archive: {destination}"
        ) from error
    source_stat = source.stat(follow_symlinks=False)
    destination_stat = destination.stat(follow_symlinks=False)
    require(
        stat.S_ISREG(destination_stat.st_mode),
        "Materialized validation-step archive is not regular",
    )
    require(
        (source_stat.st_dev, source_stat.st_ino)
        == (destination_stat.st_dev, destination_stat.st_ino),
        "Materialized validation-step archive is not a hard link to its source",
    )


def _validation_results(
    metrics_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    """Return unique finite validation rows under the established step+1 mapping."""

    require(metrics_path.is_file(), f"Metrics file is missing: {metrics_path}")
    before_hash = sha256_file(metrics_path)
    try:
        metrics = pd.read_csv(metrics_path)
    except Exception as error:
        raise CheckpointSelectionError(f"Cannot read metrics CSV: {error}") from error
    required = {"step", "val_loss"}
    require(required.issubset(metrics.columns), f"Metrics file lacks columns: {sorted(required)}")

    reported = metrics.loc[metrics["val_loss"].notna(), ["step", "val_loss"]].copy()
    require(not reported.empty, "No validation results are available")
    try:
        steps = pd.to_numeric(reported["step"], errors="raise").to_numpy(dtype=np.float64)
        losses = pd.to_numeric(reported["val_loss"], errors="raise").to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise CheckpointSelectionError("Validation step/val_loss is not numeric") from error
    require(np.isfinite(steps).all(), "Validation steps must be finite")
    require(np.equal(steps, np.floor(steps)).all(), "Validation steps must be integers")
    require((steps >= 0).all(), "Validation steps must be nonnegative")
    global_steps = steps.astype(np.int64) + 1

    finite_mask = np.isfinite(losses)
    validation = [
        {"global_step": int(global_step), "val_loss": float(loss)}
        for global_step, loss in zip(global_steps[finite_mask], losses[finite_mask], strict=True)
    ]
    ignored = [
        {"global_step": int(global_step), "val_loss": str(loss)}
        for global_step, loss in zip(global_steps[~finite_mask], losses[~finite_mask], strict=True)
    ]
    require(validation, "No finite validation losses are available")
    finite_steps = [row["global_step"] for row in validation]
    require(
        len(finite_steps) == len(set(finite_steps)),
        "Finite validation results contain duplicate global steps",
    )
    after_hash = sha256_file(metrics_path)
    require(before_hash == after_hash, "Metrics file changed while it was being read")
    return validation, ignored, before_hash


def _scalar_float(value: Any, label: str) -> float:
    """Convert a Python/NumPy/Torch scalar without importing torch here."""

    require(not isinstance(value, (bool, np.bool_)), f"{label} is boolean")
    if hasattr(value, "detach") and hasattr(value, "numel"):
        require(int(value.numel()) == 1, f"{label} is not scalar")
        value = value.detach().cpu().item()
        require(not isinstance(value, (bool, np.bool_)), f"{label} is boolean")
    array = np.asarray(value)
    require(array.size == 1, f"{label} is not scalar")
    try:
        return float(array.reshape(-1)[0])
    except (TypeError, ValueError) as error:
        raise CheckpointSelectionError(f"{label} is not numeric") from error


def _callback_best_evidence(checkpoint: Mapping[str, Any]) -> list[dict[str, Any]]:
    callbacks = checkpoint.get("callbacks")
    if callbacks is None:
        return []
    require(isinstance(callbacks, Mapping), "Lightning checkpoint callbacks field is invalid")
    evidence: list[dict[str, Any]] = []
    for callback_key, state in callbacks.items():
        if not isinstance(state, Mapping) or "best_model_score" not in state:
            continue
        key = str(callback_key)
        monitor = state.get("monitor")
        if monitor != "val_loss" and "val_loss" not in key:
            continue
        raw_score = state.get("best_model_score")
        if raw_score is None:
            continue
        score = _scalar_float(raw_score, "ModelCheckpoint best_model_score")
        require(np.isfinite(score), "ModelCheckpoint best_model_score is not finite")
        best_path = str(state.get("best_model_path", ""))
        if best_path:
            require(
                Path(best_path).name == "best.ckpt",
                "val_loss callback best_model_path does not name best.ckpt",
            )
        evidence.append(
            {
                "callback_key": key,
                "best_model_score": score,
                "best_model_path": best_path or None,
            }
        )
    return evidence


def _archive_callback_evidence(checkpoint: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Extract only the route-owned validation-step archive callback state."""

    callbacks = checkpoint.get("callbacks")
    if callbacks is None:
        return []
    require(isinstance(callbacks, Mapping), "Lightning checkpoint callbacks field is invalid")
    evidence: list[dict[str, Any]] = []
    for callback_key, state in callbacks.items():
        if str(callback_key) != ARCHIVE_CALLBACK_STATE_KEY:
            continue
        require(isinstance(state, Mapping), "Validation archive callback state is invalid")
        require(
            state.get("schema") == ARCHIVE_CALLBACK_SCHEMA,
            "Validation archive callback schema mismatch",
        )
        raw_global_step = state.get("global_step")
        raw_archive_count = state.get("archive_count")
        require(
            isinstance(raw_global_step, (int, np.integer))
            and not isinstance(raw_global_step, (bool, np.bool_)),
            "Validation archive callback global_step is not an integer",
        )
        require(
            isinstance(raw_archive_count, (int, np.integer))
            and not isinstance(raw_archive_count, (bool, np.bool_)),
            "Validation archive callback archive_count is not an integer",
        )
        global_step = _scalar_float(raw_global_step, "archive callback global_step")
        archive_count = _scalar_float(raw_archive_count, "archive callback archive_count")
        val_loss = _scalar_float(state.get("val_loss"), "archive callback val_loss")
        require(
            np.isfinite(global_step) and global_step == np.floor(global_step) and global_step > 0,
            "Validation archive callback global_step is invalid",
        )
        require(
            np.isfinite(archive_count)
            and archive_count == np.floor(archive_count)
            and archive_count > 0,
            "Validation archive callback archive_count is invalid",
        )
        require(np.isfinite(val_loss), "Validation archive callback val_loss is not finite")
        filename_value = state.get("filename")
        require(
            isinstance(filename_value, str) and filename_value != "",
            "Validation archive callback filename is missing or invalid",
        )
        filename = filename_value
        require(state.get("hook") == ARCHIVE_CALLBACK_HOOK, "Validation archive hook mismatch")
        require(
            state.get("publish_method") == ARCHIVE_PUBLISH_METHOD,
            "Validation archive publish method mismatch",
        )
        evidence.append(
            {
                "schema": ARCHIVE_CALLBACK_SCHEMA,
                "global_step": int(global_step),
                "val_loss": float(val_loss),
                "filename": filename,
                "archive_count": int(archive_count),
                "hook": ARCHIVE_CALLBACK_HOOK,
                "publish_method": ARCHIVE_PUBLISH_METHOD,
            }
        )
    return evidence


def inspect_lightning_checkpoint(path: Path, *, label: str) -> dict[str, Any]:
    """Hash and restricted-load one immutable, regular Lightning checkpoint."""

    resolved = Path(os.path.abspath(os.fspath(path.expanduser())))
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(resolved, flags)
    except OSError as error:
        raise CheckpointSelectionError(f"Cannot open {label}: {resolved}: {error}") from error
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), f"{label} is not a regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            for block in iter(lambda: handle.read(8 << 20), b""):
                digest.update(block)
            handle.seek(0)
            try:
                import torch  # noqa: PLC0415 - step archives need no torch

                checkpoint = torch.load(handle, map_location="cpu", weights_only=True)
            except Exception as error:
                raise CheckpointSelectionError(
                    f"Cannot restricted-load {label} as a Lightning checkpoint: {error}"
                ) from error
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    require(
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        f"{label} changed while it was authenticated",
    )
    require(isinstance(checkpoint, Mapping), f"{label} root is not a mapping")
    require("global_step" in checkpoint, f"{label} has no global_step")
    global_step_value = _scalar_float(checkpoint["global_step"], f"{label} global_step")
    require(np.isfinite(global_step_value), f"{label} global_step is not finite")
    require(global_step_value == np.floor(global_step_value), f"{label} global_step is not integral")
    global_step = int(global_step_value)
    require(global_step > 0, f"{label} global_step must be positive")
    callback_evidence = _callback_best_evidence(checkpoint)
    archive_evidence = _archive_callback_evidence(checkpoint)
    del checkpoint
    return {
        "checkpoint": str(resolved),
        "checkpoint_size_bytes": int(after.st_size),
        "checkpoint_sha256": digest.hexdigest(),
        "global_step": global_step,
        "restricted_load": "torch.load(weights_only=True,map_location=cpu)",
        "callback_best_evidence": callback_evidence,
        "validation_archive_evidence": archive_evidence,
    }


def _step_archive_candidates(
    validation: Sequence[Mapping[str, Any]], checkpoint_dir: Path
) -> tuple[list[dict[str, Any]], list[int]]:
    candidates: list[dict[str, Any]] = []
    missing_steps: list[int] = []
    ordered_validation = sorted(validation, key=lambda row: int(row["global_step"]))
    for expected_archive_count, row in enumerate(ordered_validation, start=1):
        global_step = int(row["global_step"])
        val_loss = float(row["val_loss"])
        checkpoint = checkpoint_dir / f"step{global_step:05d}.ckpt"
        if not checkpoint.is_file():
            missing_steps.append(global_step)
            continue
        metadata = inspect_lightning_checkpoint(
            checkpoint,
            label=f"validation-step archive {checkpoint.name}",
        )
        require(
            metadata["global_step"] == global_step,
            f"{checkpoint.name} root global_step does not match its validation row",
        )
        archive_evidence = metadata["validation_archive_evidence"]
        require(
            len(archive_evidence) == 1,
            f"{checkpoint.name} must contain exactly one {ARCHIVE_CALLBACK_STATE_KEY} state",
        )
        archive_state = archive_evidence[0]
        require(
            archive_state["global_step"] == global_step,
            f"{checkpoint.name} archive callback global_step does not match its validation row",
        )
        require(
            archive_state["filename"] == checkpoint.name,
            f"{checkpoint.name} archive callback filename mismatch",
        )
        require(
            np.isclose(
                float(archive_state["val_loss"]),
                val_loss,
                rtol=ARCHIVE_LOSS_RTOL,
                atol=ARCHIVE_LOSS_ATOL,
            ),
            f"{checkpoint.name} archive callback val_loss does not match metrics.csv",
        )
        require(
            archive_state["archive_count"] == expected_archive_count,
            f"{checkpoint.name} archive_count does not match the complete validation sequence",
        )
        candidates.append(
            {
                "global_step": global_step,
                "val_loss": val_loss,
                "checkpoint": Path(metadata["checkpoint"]),
                "checkpoint_metadata": metadata,
                "archive_callback_state": archive_state,
            }
        )
    return candidates, missing_steps


def _fallback_source_check(
    checkpoint_label: str,
    metadata: Mapping[str, Any],
    *,
    expected_step: int,
    expected_loss: float,
) -> dict[str, Any]:
    """Return an auditable eligibility decision for one best/last source."""

    step_matches = metadata["global_step"] == expected_step
    callback_checks: list[dict[str, Any]] = []
    for item in metadata["callback_best_evidence"]:
        score = float(item["best_model_score"])
        matches = bool(
            np.isclose(
                score,
                expected_loss,
                rtol=CALLBACK_SCORE_RTOL,
                atol=CALLBACK_SCORE_ATOL,
            )
        )
        callback_checks.append(
            {
                **item,
                "matches_minimum_finite_val_loss": matches,
            }
        )
    callback_crosscheck = bool(callback_checks) and all(
        item["matches_minimum_finite_val_loss"] for item in callback_checks
    )

    archive_evidence = metadata["validation_archive_evidence"]
    require(
        len(archive_evidence) <= 1,
        f"{checkpoint_label} contains ambiguous validation archive callback state",
    )
    archive_check: dict[str, Any] | None = None
    archive_crosscheck = False
    if archive_evidence:
        state = archive_evidence[0]
        expected_filename = f"step{expected_step:05d}.ckpt"
        archive_check = {
            **state,
            "global_step_matches": state["global_step"] == expected_step,
            "filename_matches": state["filename"] == expected_filename,
            "val_loss_matches": bool(
                np.isclose(
                    float(state["val_loss"]),
                    expected_loss,
                    rtol=ARCHIVE_LOSS_RTOL,
                    atol=ARCHIVE_LOSS_ATOL,
                )
            ),
        }
        archive_crosscheck = all(
            archive_check[key]
            for key in ("global_step_matches", "filename_matches", "val_loss_matches")
        )

    corroborated = callback_crosscheck or archive_crosscheck
    return {
        "checkpoint": checkpoint_label,
        "root_global_step": int(metadata["global_step"]),
        "root_global_step_matches_minimum": step_matches,
        "callback_best_score_checks": callback_checks,
        "callback_best_score_crosscheck": callback_crosscheck,
        "validation_archive_callback_check": archive_check,
        "validation_archive_callback_crosscheck": archive_crosscheck,
        "has_independent_loss_corroboration": corroborated,
        "eligible_source": step_matches and corroborated,
    }


def _best_last_selection(
    validation: Sequence[Mapping[str, Any]], checkpoint_dir: Path
) -> tuple[dict[str, Any], dict[str, Any], list[int]]:
    names = sorted(path.name for path in checkpoint_dir.iterdir() if path.suffix == ".ckpt")
    require(
        names == ["best.ckpt", "last.ckpt"],
        "Without validation-step archives, checkpoint layout must be exactly "
        "best.ckpt plus last.ckpt",
    )
    minimum = min(validation, key=lambda row: (float(row["val_loss"]), int(row["global_step"])))
    best = inspect_lightning_checkpoint(checkpoint_dir / "best.ckpt", label="best.ckpt")
    last = inspect_lightning_checkpoint(checkpoint_dir / "last.ckpt", label="last.ckpt")
    expected_step = int(minimum["global_step"])
    expected_loss = float(minimum["val_loss"])
    require(
        last["global_step"] >= best["global_step"],
        "last.ckpt predates best.ckpt",
    )

    source_checks = [
        _fallback_source_check(
            checkpoint_label,
            metadata,
            expected_step=expected_step,
            expected_loss=expected_loss,
        )
        for checkpoint_label, metadata in (("best.ckpt", best), ("last.ckpt", last))
    ]
    eligible = [item for item in source_checks if item["eligible_source"]]
    require(
        bool(eligible),
        "Neither best.ckpt nor last.ckpt has an exact defensible cross-check for "
        "the minimum finite val_loss",
    )
    # Prefer best.ckpt when both independently authenticate.  This preserves
    # the pinned runtime's intended artifact while still rejecting the observed
    # stale-best metadata case and allowing a defensible last.ckpt recovery.
    selected_source_label = str(eligible[0]["checkpoint"])
    selected_metadata = best if selected_source_label == "best.ckpt" else last
    selected_source_check = eligible[0]

    selected = {
        "global_step": expected_step,
        "val_loss": expected_loss,
        "checkpoint": Path(selected_metadata["checkpoint"]),
    }
    authentication = {
        "exact_checkpoint_filenames": names,
        "minimum_finite_validation": dict(minimum),
        "best_checkpoint": best,
        "last_checkpoint": last,
        "best_global_step_matches_minimum": best["global_step"] == expected_step,
        "last_global_step_not_before_best": True,
        "source_eligibility_checks": source_checks,
        "selected_source": selected_source_label,
        "selected_source_checkpoint": selected_metadata,
        "selected_source_check": selected_source_check,
        "callback_best_score_checks": selected_source_check["callback_best_score_checks"],
        "callback_best_score_available": bool(
            selected_source_check["callback_best_score_checks"]
        ),
        "fallback_authentication_rule": (
            "root global_step equals earliest minimum finite metrics row and either "
            "ModelCheckpoint best_model_score or VCCValidationStepArchive state "
            "exactly corroborates val_loss"
        ),
        "callback_score_tolerance": {
            "relative": CALLBACK_SCORE_RTOL,
            "absolute": CALLBACK_SCORE_ATOL,
        },
        "archive_loss_tolerance": {
            "relative": ARCHIVE_LOSS_RTOL,
            "absolute": ARCHIVE_LOSS_ATOL,
        },
    }
    unarchived = [
        int(row["global_step"])
        for row in validation
        if int(row["global_step"]) != expected_step
    ]
    return selected, authentication, unarchived


def build_selection(
    run_dir: Path,
    metrics_path: Path,
    output_link: Path,
) -> dict[str, Any]:
    """Build the v1 receipt without publishing either output."""

    checkpoint_dir = run_dir / "checkpoints"
    require(checkpoint_dir.is_dir(), f"Checkpoint directory is missing: {checkpoint_dir}")
    validation, ignored_nonfinite, metrics_sha256 = _validation_results(metrics_path)
    candidates, missing_steps = _step_archive_candidates(validation, checkpoint_dir)

    if candidates:
        require(
            not missing_steps,
            "Validation-step archive layout is incomplete; refusing partial-archive selection",
        )
        layout = STEP_ARCHIVE_LAYOUT
        selected = min(candidates, key=lambda item: (item["val_loss"], item["global_step"]))
        selected_path = Path(selected["checkpoint"])
        selected_metadata = selected["checkpoint_metadata"]
        layout_authentication: dict[str, Any] = {
            "selection_source": "authenticated validation-aligned stepNNNNN.ckpt archives",
            "best_last_fallback_used": False,
            "archive_callback_state_key": ARCHIVE_CALLBACK_STATE_KEY,
            "archive_callback_schema": ARCHIVE_CALLBACK_SCHEMA,
            "all_finite_validation_steps_archived": True,
            "authenticated_archives": [
                {
                    "checkpoint_metadata": item["checkpoint_metadata"],
                    "archive_callback_state": item["archive_callback_state"],
                }
                for item in candidates
            ],
            "archive_loss_tolerance": {
                "relative": ARCHIVE_LOSS_RTOL,
                "absolute": ARCHIVE_LOSS_ATOL,
            },
        }
        archived_candidates = candidates
    else:
        layout = BEST_LAST_LAYOUT
        selected, layout_authentication, missing_steps = _best_last_selection(
            validation, checkpoint_dir
        )
        source_path = Path(selected["checkpoint"])
        selected_metadata = layout_authentication["selected_source_checkpoint"]
        selected_path = checkpoint_dir / f"step{int(selected['global_step']):05d}.ckpt"
        layout_authentication["materialization"] = {
            "method": "atomic_regular_hard_link_no_copy",
            "source_checkpoint": str(source_path),
            "step_archive": str(selected_path),
            "materialized": False,
        }
        archived_candidates = [
            {
                "global_step": selected["global_step"],
                "val_loss": selected["val_loss"],
                "checkpoint": selected_path,
            }
        ]

    return {
        "schema": SCHEMA,
        "selection_rule": "minimum validation loss; earliest step breaks ties",
        "checkpoint_layout": layout,
        "metrics": str(metrics_path),
        "metrics_sha256": metrics_sha256,
        "validation_results": [dict(row) for row in validation],
        "nonfinite_validation_results_ignored": ignored_nonfinite,
        "archived_candidates": [
            {
                "global_step": int(item["global_step"]),
                "val_loss": float(item["val_loss"]),
                "checkpoint": str(item["checkpoint"]),
            }
            for item in archived_candidates
        ],
        "unarchived_validation_steps": missing_steps,
        "layout_authentication": layout_authentication,
        "selected": {
            "global_step": int(selected["global_step"]),
            "val_loss": float(selected["val_loss"]),
            "checkpoint": str(selected_path),
            "checkpoint_size_bytes": int(selected_metadata["checkpoint_size_bytes"]),
            "checkpoint_sha256": str(selected_metadata["checkpoint_sha256"]),
            "link": str(output_link),
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_dir.resolve()
    checkpoint_dir = run_dir / "checkpoints"
    metrics_path = (args.metrics or run_dir / "version_0" / "metrics.csv").resolve()
    output_link = Path(
        os.path.abspath(os.fspath(args.output_link or checkpoint_dir / "selected.ckpt"))
    )
    output_json = Path(
        os.path.abspath(
            os.fspath(args.output_json or checkpoint_dir / "selected_checkpoint.json")
        )
    )
    require(output_link != output_json, "Selection link and JSON paths must differ")
    _refuse_existing(output_link, "selection link")
    _refuse_existing(output_json, "selection JSON")

    payload = build_selection(run_dir, metrics_path, output_link)
    selected_path = Path(payload["selected"]["checkpoint"])
    if payload["checkpoint_layout"] == BEST_LAST_LAYOUT:
        materialization = payload["layout_authentication"]["materialization"]
        source_path = Path(materialization["source_checkpoint"])
        atomic_hardlink_no_overwrite(source_path, selected_path)
        require(
            selected_path.stat().st_size == payload["selected"]["checkpoint_size_bytes"],
            "Materialized validation-step archive size differs from verified source checkpoint",
        )
        require(
            sha256_file(selected_path) == payload["selected"]["checkpoint_sha256"],
            "Materialized validation-step archive hash differs from verified source checkpoint",
        )
        materialization["materialized"] = True
        materialization["same_device_and_inode"] = True
    relative_target = os.path.relpath(selected_path, output_link.parent)
    atomic_symlink_no_overwrite(output_link, relative_target)
    atomic_json(output_json, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    payload = run(parse_args(argv))
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
