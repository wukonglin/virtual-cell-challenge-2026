#!/usr/bin/env python3
"""Safely audit an official scDFM checkpoint without executing model code.

The auditor opens the checkpoint read-only, authenticates its bytes, and loads
it on CPU with PyTorch's restricted ``weights_only`` loader.  It never imports
the upstream scDFM package, instantiates a model, restores an optimizer, or
modifies the checkpoint.  A JSON receipt is written only when explicitly
requested and an existing receipt is never replaced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, BinaryIO

import torch


SCHEMA = "vcc-scdfm-checkpoint-audit-v1"
REQUIRED_TOP_LEVEL_KEYS = (
    "eval_score",
    "iteration",
    "model_state_dict",
    "optimizer_state_dict",
    "scheduler_state_dict",
)
PERTURBATION_EMBEDDING_KEY = "perturbation_embedder.embedding.weight"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
FINITE_CHECK_CHUNK_ELEMENTS = 1 << 20

METADATA_KEYS = {
    "config": frozenset(
        {
            "args",
            "cfg",
            "config",
            "configuration",
            "hparams",
            "hyperparameters",
        }
    ),
    "vocab": frozenset(
        {
            "gene_ids",
            "gene_names",
            "gene_vocab",
            "genes",
            "tokenizer",
            "vocab",
            "vocabulary",
        }
    ),
    "graph": frozenset(
        {
            "adjacency",
            "edge_index",
            "edges",
            "gene_graph",
            "graph",
            "graph_dict",
            "network",
        }
    ),
}


class CheckpointAuditError(RuntimeError):
    """Raised when a checkpoint violates the immutable audit contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckpointAuditError(message)


def _normalise_expected_sha256(expected_sha256: str | None) -> str | None:
    if expected_sha256 is None:
        return None
    normalised = expected_sha256.strip().lower()
    _require(
        SHA256_PATTERN.fullmatch(normalised) is not None,
        "Expected SHA-256 must contain exactly 64 hexadecimal characters",
    )
    return normalised


def _sha256_stream(handle: BinaryIO, chunk_size: int = 8 << 20) -> str:
    """Hash an already-open file and rewind it for the restricted loader."""

    digest = hashlib.sha256()
    while block := handle.read(chunk_size):
        digest.update(block)
    handle.seek(0)
    return digest.hexdigest()


def _open_regular_read_only(path: Path) -> tuple[BinaryIO, os.stat_result]:
    """Open one non-symlink regular file without following its final link."""

    display_path = Path(os.path.abspath(os.fspath(path.expanduser())))
    try:
        initial = display_path.lstat()
    except OSError as error:
        raise CheckpointAuditError(
            f"Unable to inspect scDFM checkpoint {display_path}: {error}"
        ) from error
    _require(
        not stat.S_ISLNK(initial.st_mode),
        f"scDFM checkpoint must not be a symbolic link: {display_path}",
    )
    _require(
        stat.S_ISREG(initial.st_mode),
        f"scDFM checkpoint must be a regular file: {display_path}",
    )

    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(display_path, flags)
    except OSError as error:
        raise CheckpointAuditError(
            f"Unable to open scDFM checkpoint read-only: {display_path}: {error}"
        ) from error

    try:
        opened = os.fstat(descriptor)
        _require(
            stat.S_ISREG(opened.st_mode),
            f"Opened scDFM checkpoint is not a regular file: {display_path}",
        )
        _require(
            (opened.st_dev, opened.st_ino)
            == (initial.st_dev, initial.st_ino),
            f"scDFM checkpoint changed while it was being opened: {display_path}",
        )
        return os.fdopen(descriptor, "rb"), opened
    except BaseException:
        os.close(descriptor)
        raise


def _is_finite_tensor(tensor: torch.Tensor) -> bool:
    """Check finite floating-point values with bounded temporary allocations."""

    if not (tensor.is_floating_point() or tensor.is_complex()):
        return True
    flattened = tensor.detach()
    if not flattened.is_contiguous():
        flattened = flattened.contiguous()
    flattened = flattened.view(-1)
    for start in range(0, flattened.numel(), FINITE_CHECK_CHUNK_ELEMENTS):
        block = flattened[start : start + FINITE_CHECK_CHUNK_ELEMENTS]
        if not bool(torch.isfinite(block).all().item()):
            return False
    return True


class _TensorInventory:
    """Validate tensors and collect deterministic aggregate statistics."""

    def __init__(self) -> None:
        self.tensor_count = 0
        self.element_count = 0
        self.dtype_counts: Counter[str] = Counter()

    def add(self, tensor: torch.Tensor, path: str) -> None:
        _require(
            tensor.layout == torch.strided,
            f"Unsupported non-strided tensor at {path}: {tensor.layout}",
        )
        _require(
            tensor.device.type == "cpu",
            f"Tensor was not mapped to CPU at {path}: {tensor.device}",
        )
        shape = tuple(int(dimension) for dimension in tensor.shape)
        _require(
            all(dimension >= 0 for dimension in shape),
            f"Invalid tensor shape at {path}: {shape}",
        )
        expected_elements = math.prod(shape) if shape else 1
        _require(
            tensor.numel() == expected_elements,
            f"Tensor element count does not match its shape at {path}",
        )
        _require(
            _is_finite_tensor(tensor),
            f"Non-finite tensor value detected at {path}",
        )

        self.tensor_count += 1
        self.element_count += tensor.numel()
        self.dtype_counts[str(tensor.dtype).removeprefix("torch.")] += 1

    def receipt(self) -> dict[str, Any]:
        return {
            "tensor_count": self.tensor_count,
            "element_count": self.element_count,
            "dtype_tensor_counts": dict(sorted(self.dtype_counts.items())),
        }

    def merge(self, other: "_TensorInventory") -> None:
        """Add already-validated aggregate statistics without rescanning tensors."""

        self.tensor_count += other.tensor_count
        self.element_count += other.element_count
        self.dtype_counts.update(other.dtype_counts)


def _mapping_key_path(parent: str, key: object) -> str:
    if isinstance(key, str):
        return f"{parent}.{key}"
    return f"{parent}[{type(key).__name__}:{key!r}]"


def _walk_checkpoint(
    value: object,
    *,
    path: str,
    inventory: _TensorInventory,
    active_containers: set[int],
) -> None:
    """Validate the restricted checkpoint object graph and inventory tensors."""

    if isinstance(value, torch.Tensor):
        inventory.add(value, path)
        return
    if value is None or isinstance(value, (bool, int, str, bytes)):
        return
    if isinstance(value, float):
        _require(math.isfinite(value), f"Non-finite scalar value detected at {path}")
        return

    if isinstance(value, Mapping):
        identity = id(value)
        _require(identity not in active_containers, f"Container cycle detected at {path}")
        active_containers.add(identity)
        try:
            for key, child in value.items():
                _require(
                    key is None
                    or isinstance(key, (bool, int, float, str, bytes)),
                    f"Unsupported mapping key type at {path}: {type(key).__name__}",
                )
                if isinstance(key, float):
                    _require(
                        math.isfinite(key),
                        f"Non-finite mapping key detected at {path}",
                    )
                _walk_checkpoint(
                    child,
                    path=_mapping_key_path(path, key),
                    inventory=inventory,
                    active_containers=active_containers,
                )
        finally:
            active_containers.remove(identity)
        return

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        identity = id(value)
        _require(identity not in active_containers, f"Container cycle detected at {path}")
        active_containers.add(identity)
        try:
            for index, child in enumerate(value):
                _walk_checkpoint(
                    child,
                    path=f"{path}[{index}]",
                    inventory=inventory,
                    active_containers=active_containers,
                )
        finally:
            active_containers.remove(identity)
        return

    raise CheckpointAuditError(
        f"Unsupported checkpoint value type at {path}: {type(value).__name__}"
    )


def _validate_model_state_dict(model_state_dict: object) -> _TensorInventory:
    _require(
        isinstance(model_state_dict, Mapping),
        "model_state_dict must be a mapping",
    )
    _require(bool(model_state_dict), "model_state_dict must not be empty")
    inventory = _TensorInventory()
    for name, tensor in model_state_dict.items():
        _require(
            isinstance(name, str) and bool(name),
            "Every model_state_dict key must be a non-empty string",
        )
        _require(
            isinstance(tensor, torch.Tensor),
            f"model_state_dict value must be a tensor: {name}",
        )
        inventory.add(tensor, f"checkpoint.model_state_dict.{name}")
    return inventory


def _metadata_receipt(checkpoint: Mapping[str, object]) -> dict[str, Any]:
    """Find explicit metadata keys outside the three state dictionaries."""

    matches: dict[str, set[str]] = {category: set() for category in METADATA_KEYS}
    skipped_roots = {
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
    }

    def visit(value: object, path: str, active: set[int]) -> None:
        if isinstance(value, Mapping):
            identity = id(value)
            if identity in active:
                return
            active.add(identity)
            try:
                for key, child in value.items():
                    child_path = _mapping_key_path(path, key)
                    if isinstance(key, str):
                        normalised = key.strip().lower().replace("-", "_")
                        for category, names in METADATA_KEYS.items():
                            if normalised in names:
                                matches[category].add(child_path)
                    visit(child, child_path, active)
            finally:
                active.remove(identity)
        elif isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            identity = id(value)
            if identity in active:
                return
            active.add(identity)
            try:
                for index, child in enumerate(value):
                    visit(child, f"{path}[{index}]", active)
            finally:
                active.remove(identity)

    for key, value in checkpoint.items():
        if key not in skipped_roots:
            visit({key: value}, "checkpoint", set())

    return {
        category: {
            "embedded": bool(matches[category]),
            "paths": sorted(matches[category]),
        }
        for category in sorted(matches)
    }


def _perturbation_embedding_receipt(
    model_state_dict: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    matches = sorted(
        name
        for name in model_state_dict
        if name == PERTURBATION_EMBEDDING_KEY
        or name.endswith(f".{PERTURBATION_EMBEDDING_KEY}")
    )
    _require(
        len(matches) <= 1,
        "Multiple perturbation embedding tensors were found: " + ", ".join(matches),
    )
    if not matches:
        return {
            "present": False,
            "state_dict_key": None,
            "vocabulary_size": None,
            "hidden_dimension": None,
        }

    name = matches[0]
    tensor = model_state_dict[name]
    _require(
        tensor.ndim == 2,
        f"Perturbation embedding must be rank two: {name} has shape {tuple(tensor.shape)}",
    )
    return {
        "present": True,
        "state_dict_key": name,
        "vocabulary_size": int(tensor.shape[0]),
        "hidden_dimension": int(tensor.shape[1]),
    }


def _validate_iteration(value: object) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool),
        "iteration must be an integer",
    )
    _require(value >= 0, "iteration must be non-negative")
    return value


def _validate_eval_score(value: object) -> int | float | None:
    if value is None:
        return None
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        "eval_score must be a finite number or null",
    )
    _require(math.isfinite(float(value)), "eval_score must be finite when present")
    return value


def audit_checkpoint(
    checkpoint_path: Path,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Authenticate and structurally audit one checkpoint without model code."""

    expected_sha256 = _normalise_expected_sha256(expected_sha256)
    display_path = Path(os.path.abspath(os.fspath(checkpoint_path.expanduser())))

    handle, opened_stat = _open_regular_read_only(display_path)
    with handle:
        digest = _sha256_stream(handle)
        if expected_sha256 is not None:
            _require(
                digest == expected_sha256,
                f"Checkpoint SHA-256 mismatch: {digest}; expected {expected_sha256}",
            )
        try:
            checkpoint = torch.load(
                handle,
                map_location="cpu",
                weights_only=True,
            )
        except Exception as error:
            raise CheckpointAuditError(
                "Restricted weights-only checkpoint loading failed: "
                f"{type(error).__name__}: {error}"
            ) from error
        after_load = os.fstat(handle.fileno())
        _require(
            (
                after_load.st_dev,
                after_load.st_ino,
                after_load.st_size,
                after_load.st_mtime_ns,
            )
            == (
                opened_stat.st_dev,
                opened_stat.st_ino,
                opened_stat.st_size,
                opened_stat.st_mtime_ns,
            ),
            "scDFM checkpoint changed during restricted loading",
        )

    _require(isinstance(checkpoint, Mapping), "Checkpoint root must be a mapping")
    _require(
        all(isinstance(key, str) for key in checkpoint),
        "Every checkpoint top-level key must be a string",
    )
    missing = [key for key in REQUIRED_TOP_LEVEL_KEYS if key not in checkpoint]
    _require(
        not missing,
        "Checkpoint is missing required top-level keys: " + ", ".join(missing),
    )
    _require(
        isinstance(checkpoint["optimizer_state_dict"], Mapping),
        "optimizer_state_dict must be a mapping",
    )
    _require(
        isinstance(checkpoint["scheduler_state_dict"], Mapping),
        "scheduler_state_dict must be a mapping",
    )

    model_tensors = _validate_model_state_dict(checkpoint["model_state_dict"])
    all_tensors = _TensorInventory()
    all_tensors.merge(model_tensors)
    for key, value in checkpoint.items():
        if key == "model_state_dict":
            continue
        _walk_checkpoint(
            value,
            path=f"checkpoint.{key}",
            inventory=all_tensors,
            active_containers=set(),
        )
    iteration = _validate_iteration(checkpoint["iteration"])
    eval_score = _validate_eval_score(checkpoint["eval_score"])
    perturbation_embedding = _perturbation_embedding_receipt(
        checkpoint["model_state_dict"]
    )

    return {
        "schema": SCHEMA,
        "status": "passed",
        "checks": {
            "all_floating_and_complex_tensor_values_are_finite": True,
            "all_tensor_shapes_are_valid": True,
            "all_tensors_are_cpu_strided": True,
            "checkpoint_is_non_symlink_regular_file": True,
            "checkpoint_unchanged_during_audit": True,
            "checkpoint_loaded_with_restricted_weights_only_mode": True,
            "required_top_level_keys_are_present": True,
            "sha256_matches_expected_when_provided": True,
            "state_dictionaries_are_mappings": True,
        },
        "checkpoint": {
            "path": str(display_path),
            "size_bytes": opened_stat.st_size,
            "sha256": digest,
            "expected_sha256": expected_sha256,
        },
        "structure": {
            "top_level_keys": sorted(checkpoint),
            "required_top_level_keys": list(REQUIRED_TOP_LEVEL_KEYS),
            "iteration": iteration,
            "eval_score": eval_score,
            "metadata": _metadata_receipt(checkpoint),
            "perturbation_embedding": perturbation_embedding,
            "tensors": {
                "checkpoint": all_tensors.receipt(),
                "model_state_dict": {
                    **model_tensors.receipt(),
                    "parameter_count_from_state_dict": model_tensors.element_count,
                },
            },
        },
    }


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically publish a new deterministic receipt without overwriting."""

    path = Path(os.path.abspath(os.fspath(path.expanduser())))
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(f"Refusing to overwrite receipt: {path}") from error
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Safely authenticate and audit one official scDFM checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Official scDFM checkpoint.pt file.",
    )
    parser.add_argument(
        "--expected-sha256",
        default=None,
        help="Optional expected checkpoint SHA-256 digest.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional new JSON receipt. No file is written when omitted.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    receipt = audit_checkpoint(
        args.checkpoint,
        expected_sha256=args.expected_sha256,
    )
    if args.output is not None:
        write_json_atomic(args.output, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return receipt


if __name__ == "__main__":
    main()
