#!/usr/bin/env python3
"""Audit target coverage in an official STATE perturbation embedding map."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import torch


SCHEMA = "vcc-state-checkpoint-coverage-v1"
TARGET_COLUMN = "target_gene"
ALLOWED_CHECKPOINT_GLOBALS = frozenset(
    {
        "numpy._core.multiarray.scalar",
        "numpy.core.multiarray.scalar",
        "numpy.dtype",
    }
)


class UnsafeCheckpointError(RuntimeError):
    """Raised when a checkpoint requests a non-allowlisted pickle global."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Safely compare STATE perturbation-map keys with challenge target genes."
        )
    )
    parser.add_argument("--pert-map", type=Path, required=True)
    parser.add_argument("--pert-counts", type=Path, required=True)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional JSON output. Existing files are never overwritten.",
    )
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def sha256_strings(values: Iterable[str]) -> str:
    encoded = json.dumps(
        list(values), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def describe_file(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    require(resolved.is_file(), f"Missing input file: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _numpy_scalar_constructor() -> Any:
    core = getattr(np, "_core", None)
    if core is None:
        core = np.core
    return core.multiarray.scalar


def numpy_safe_globals() -> list[Any]:
    """Return only the NumPy globals required by the official string-key map."""
    scalar = _numpy_scalar_constructor()
    globals_: list[Any] = [
        (scalar, "numpy._core.multiarray.scalar"),
        (scalar, "numpy.core.multiarray.scalar"),
        (np.dtype, "numpy.dtype"),
    ]
    unicode_dtype_type = type(np.dtype("U"))
    if unicode_dtype_type is not np.dtype:
        globals_.append(unicode_dtype_type)
    return globals_


@contextmanager
def isolated_user_safe_globals() -> Iterator[None]:
    """Hide ambient user allowlists while inspecting and loading the checkpoint."""
    get_globals = getattr(torch.serialization, "get_safe_globals", None)
    clear_globals = getattr(torch.serialization, "clear_safe_globals", None)
    add_globals = getattr(torch.serialization, "add_safe_globals", None)
    if get_globals is None or clear_globals is None or add_globals is None:
        raise UnsafeCheckpointError(
            "This PyTorch version cannot isolate checkpoint safe globals"
        )
    previous = list(get_globals())
    clear_globals()
    try:
        yield
    finally:
        clear_globals()
        add_globals(previous)


def checkpoint_globals(path: Path) -> list[str]:
    scanner = getattr(torch.serialization, "get_unsafe_globals_in_checkpoint", None)
    if scanner is None:
        raise UnsafeCheckpointError(
            "This PyTorch version cannot preflight checkpoint globals safely"
        )
    try:
        declared = sorted(set(str(name) for name in scanner(path)))
    except Exception as error:
        raise UnsafeCheckpointError(
            f"Could not inspect checkpoint globals safely: {error}"
        ) from error
    unexpected = sorted(set(declared) - ALLOWED_CHECKPOINT_GLOBALS)
    if unexpected:
        raise UnsafeCheckpointError(
            "Checkpoint references unsupported globals: " + ", ".join(unexpected)
        )
    return declared


def load_perturbation_map(path: Path) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Load a tensor-only perturbation map without enabling arbitrary pickle code."""
    resolved = path.resolve()
    require(resolved.is_file(), f"Missing perturbation map: {resolved}")
    try:
        with isolated_user_safe_globals():
            declared = checkpoint_globals(resolved)
            with torch.serialization.safe_globals(numpy_safe_globals()):
                payload = torch.load(resolved, map_location="cpu", weights_only=True)
    except Exception as error:
        raise UnsafeCheckpointError(
            f"Restricted weights-only load failed for {resolved}: {error}"
        ) from error

    require(type(payload) is dict, "Perturbation map must be a plain dictionary")
    require(bool(payload), "Perturbation map is empty")
    normalized: dict[str, torch.Tensor] = {}
    for raw_key, value in payload.items():
        require(
            isinstance(raw_key, (str, np.str_)),
            f"Perturbation map contains a non-string key: {type(raw_key)!r}",
        )
        key = str(raw_key)
        require(bool(key), "Perturbation map contains an empty key")
        require(key not in normalized, f"Duplicate normalized perturbation key: {key}")
        require(
            type(value) is torch.Tensor,
            f"Perturbation map value for {key} is not a plain tensor",
        )
        require(
            value.ndim == 1,
            f"Perturbation embedding for {key} is not one-dimensional",
        )
        require(value.numel() > 0, f"Perturbation embedding for {key} is empty")
        normalized[key] = value

    dimensions = {int(value.numel()) for value in normalized.values()}
    require(len(dimensions) == 1, "Perturbation embeddings have inconsistent dimensions")
    return normalized, declared


def read_targets(path: Path) -> list[str]:
    resolved = path.resolve()
    require(resolved.is_file(), f"Missing target CSV: {resolved}")
    with resolved.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None, f"Target CSV has no header: {resolved}")
        require(
            TARGET_COLUMN in reader.fieldnames,
            f"Target CSV lacks required column {TARGET_COLUMN!r}",
        )
        targets: list[str] = []
        for row_number, row in enumerate(reader, start=2):
            value = row.get(TARGET_COLUMN)
            require(value is not None, f"Missing target value at CSV row {row_number}")
            require(
                value == value.strip(),
                f"Target has surrounding whitespace at row {row_number}",
            )
            require(bool(value), f"Empty target value at CSV row {row_number}")
            targets.append(value)
    require(bool(targets), "Target CSV has no target rows")
    return targets


def build_report(pert_map_path: Path, pert_counts_path: Path) -> dict[str, Any]:
    embedding_map, declared_globals = load_perturbation_map(pert_map_path)
    target_values = read_targets(pert_counts_path)
    target_counts = Counter(target_values)
    duplicate_targets = sorted(
        target for target, count in target_counts.items() if count > 1
    )
    unique_targets = sorted(target_counts)
    embedding_keys = sorted(embedding_map)
    target_set = set(unique_targets)
    embedding_set = set(embedding_keys)
    overlap = sorted(target_set & embedding_set)
    missing = sorted(target_set - embedding_set)
    extra = sorted(embedding_set - target_set)
    dimensions = sorted({int(value.numel()) for value in embedding_map.values()})
    dtypes = sorted({str(value.dtype) for value in embedding_map.values()})

    return {
        "schema": SCHEMA,
        "inputs": {
            "perturbation_map": describe_file(pert_map_path),
            "perturbation_counts": describe_file(pert_counts_path),
        },
        "safety": {
            "torch_load_weights_only": True,
            "checkpoint_declared_nondefault_globals": declared_globals,
            "allowed_checkpoint_globals": sorted(ALLOWED_CHECKPOINT_GLOBALS),
            "unknown_checkpoint_globals": [],
        },
        "embedding_map": {
            "entries": len(embedding_map),
            "unique_keys": len(embedding_keys),
            "keys_sha256": sha256_strings(embedding_keys),
            "dimensions": dimensions,
            "dtypes": dtypes,
            "non_targeting_key_present": "non-targeting" in embedding_map,
        },
        "challenge_targets": {
            "rows": len(target_values),
            "unique_genes": len(unique_targets),
            "duplicate_rows": len(target_values) - len(unique_targets),
            "duplicate_genes": duplicate_targets,
            "ordered_values_sha256": sha256_strings(target_values),
            "unique_genes_sha256": sha256_strings(unique_targets),
        },
        "coverage": {
            "overlap_genes": len(overlap),
            "overlap_fraction": len(overlap) / len(unique_targets),
            "overlap_genes_sha256": sha256_strings(overlap),
            "missing_genes": missing,
            "missing_gene_count": len(missing),
            "missing_genes_sha256": sha256_strings(missing),
            "extra_embedding_keys": extra,
            "extra_embedding_key_count": len(extra),
            "extra_embedding_keys_sha256": sha256_strings(extra),
            "full_target_coverage": not missing,
        },
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    output = path.absolute()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    args = parse_args()
    report = build_report(args.pert_map, args.pert_counts)
    if args.output_json is not None:
        atomic_json(args.output_json, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
