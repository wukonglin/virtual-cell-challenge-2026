#!/usr/bin/env python3
"""Select an exact-step STATE checkpoint from validation-aligned archives."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    checkpoint_dir = run_dir / "checkpoints"
    metrics_path = (args.metrics or run_dir / "version_0" / "metrics.csv").resolve()
    output_link = (args.output_link or checkpoint_dir / "selected.ckpt").absolute()
    output_json = (
        args.output_json or checkpoint_dir / "selected_checkpoint.json"
    ).absolute()
    if output_link.exists() or output_link.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {output_link}")
    if output_json.exists():
        raise FileExistsError(f"Refusing to overwrite {output_json}")

    metrics = pd.read_csv(metrics_path)
    required = {"step", "val_loss"}
    if not required.issubset(metrics.columns):
        raise ValueError(f"Metrics file lacks columns: {sorted(required)}")
    validation = metrics.loc[metrics["val_loss"].notna(), ["step", "val_loss"]].copy()
    if validation.empty:
        raise ValueError("No validation results are available")
    validation["global_step"] = validation["step"].astype(int) + 1
    validation["val_loss"] = validation["val_loss"].astype(float)

    candidates: list[dict[str, Any]] = []
    missing_steps: list[int] = []
    for row in validation.itertuples(index=False):
        global_step = int(row.global_step)
        checkpoint = checkpoint_dir / f"step{global_step:05d}.ckpt"
        if not checkpoint.is_file():
            missing_steps.append(global_step)
            continue
        candidates.append(
            {
                "global_step": global_step,
                "val_loss": float(row.val_loss),
                "checkpoint": checkpoint.resolve(),
            }
        )
    if not candidates:
        raise FileNotFoundError("No validation-aligned checkpoint archive is available")

    selected = min(candidates, key=lambda item: (item["val_loss"], item["global_step"]))
    selected_path = Path(selected["checkpoint"])
    relative_target = os.path.relpath(selected_path, output_link.parent)
    output_link.parent.mkdir(parents=True, exist_ok=True)
    temporary_link = output_link.with_name(f".{output_link.name}.{os.getpid()}.tmp")
    try:
        os.symlink(relative_target, temporary_link)
        os.replace(temporary_link, output_link)
    finally:
        if temporary_link.is_symlink():
            temporary_link.unlink()

    payload = {
        "schema": "vcc-state-checkpoint-selection-v1",
        "selection_rule": "minimum validation loss; earliest step breaks ties",
        "metrics": str(metrics_path),
        "metrics_sha256": sha256_file(metrics_path),
        "validation_results": [
            {
                "global_step": int(row.global_step),
                "val_loss": float(row.val_loss),
            }
            for row in validation.itertuples(index=False)
        ],
        "archived_candidates": [
            {
                "global_step": item["global_step"],
                "val_loss": item["val_loss"],
                "checkpoint": str(item["checkpoint"]),
            }
            for item in candidates
        ],
        "unarchived_validation_steps": missing_steps,
        "selected": {
            "global_step": selected["global_step"],
            "val_loss": selected["val_loss"],
            "checkpoint": str(selected_path),
            "checkpoint_size_bytes": selected_path.stat().st_size,
            "checkpoint_sha256": sha256_file(selected_path),
            "link": str(output_link),
        },
    }
    atomic_json(output_json, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
