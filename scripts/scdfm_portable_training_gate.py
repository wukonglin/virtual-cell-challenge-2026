#!/usr/bin/env python3
"""Portable, fail-closed consumer for the sealed V7 training data gate.

This module intentionally does not import the offline split builder. In
particular, importing it must never import scikit-learn or reconstruct the
registered target clusters. The only permitted operation is to consume the
repository-tracked manifest, preflight, and hash lock, then derive the exact
allowed source-row sequence from authenticated label metadata.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, BinaryIO, Mapping, Sequence

import h5py
import numpy as np


LOCK_SCHEMA = "vcc-scdfm-portable-training-hash-lock-v1"
MANIFEST_SCHEMA = "vcc-scdfm-jurkat-non-harm-manifest-v1"
PREFLIGHT_SCHEMA = "vcc-scdfm-jurkat-training-preflight-v1"
CANONICAL_ENCODING = "python-json-indent2-sort-keys-utf8-lf-final-newline"
CONTROL_LABEL = "non-targeting"
SHA256_HEX = frozenset("0123456789abcdef")
PROTECTED_STAGES = (
    "model_fit",
    "checkpoint_selection",
    "normalization_fit",
    "feature_selection",
    "graph_construction",
    "decoder_fit",
    "hyperparameter_selection",
)
REPOSITORY_PATHS = MappingProxyType(
    {
        "split_configuration": "configs/scdfm/vcc2026_v7_gamma1.toml",
        "manifest": "artifacts/scdfm/v7/splits/jurkat_non_harm.json",
        "preflight": (
            "artifacts/scdfm/v7/training/jurkat_training_preflight.json"
        ),
        "lock": "artifacts/scdfm/v7/training/portable_training_gate_lock.json",
    }
)
DATASET_ROLES = MappingProxyType(
    {
        "nadig_jurkat": "training",
        "competition_train.h5": "training",
        # This source is enumerated by the row audit, but the registered
        # whole-HepG2 holdout prevents it from becoming fit supervision.
        "hepg2.h5": "evaluation_only",
        "jurkat.h5": "training",
        "k562.h5": "training",
        "k562_gwps.h5": "training",
        "rpe1.h5": "training",
    }
)


class PortableTrainingGateError(RuntimeError):
    """Raised when an input violates the portable V7 training contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PortableTrainingGateError(message)


def is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value).issubset(SHA256_HEX)
    )


def canonical_json_bytes(value: object) -> bytes:
    """Return the canonical semantic encoding used by row-index hashes."""

    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def formatted_json_bytes(value: object) -> bytes:
    """Return the exact repository encoding for sealed JSON artifacts."""

    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def strict_json_loads(raw: bytes, label: str) -> Any:
    """Parse strict UTF-8 JSON and reject duplicate keys and non-finite values."""

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PortableTrainingGateError(
                    f"Duplicate JSON key in {label}: {key}"
                )
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise PortableTrainingGateError(
            f"Non-finite JSON number in {label}: {value}"
        )

    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except PortableTrainingGateError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PortableTrainingGateError(f"Invalid {label} JSON: {error}") from error


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _same_identity(before: os.stat_result, after: os.stat_result) -> bool:
    return (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) == (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )


def open_regular_no_follow(path: Path, label: str) -> tuple[BinaryIO, os.stat_result]:
    """Open one regular file while rejecting symlinks in every component."""

    absolute = _absolute(path)
    parts = absolute.parts
    require(len(parts) >= 2 and parts[0] == os.sep, f"Invalid {label} path")
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    file_flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    file_flags |= getattr(os, "O_NOFOLLOW", 0)
    directory_descriptor: int | None = None
    descriptor: int | None = None
    try:
        directory_descriptor = os.open(os.sep, directory_flags)
        for component in parts[1:-1]:
            next_descriptor = os.open(
                component,
                directory_flags,
                dir_fd=directory_descriptor,
            )
            os.close(directory_descriptor)
            directory_descriptor = next_descriptor
        descriptor = os.open(
            parts[-1],
            file_flags,
            dir_fd=directory_descriptor,
        )
    except OSError as error:
        if descriptor is not None:
            os.close(descriptor)
        raise PortableTrainingGateError(
            f"Unable to open {label}: {type(error).__name__}: {error}"
        ) from error
    finally:
        if directory_descriptor is not None:
            os.close(directory_descriptor)
    require(descriptor is not None, f"Unable to open {label}")
    handle = os.fdopen(descriptor, "rb")
    opened = os.fstat(handle.fileno())
    if not stat.S_ISREG(opened.st_mode):
        handle.close()
        raise PortableTrainingGateError(f"{label} is not a regular file")
    return handle, opened


def _hash_stream(handle: BinaryIO, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    while block := handle.read(chunk_size):
        digest.update(block)
    return digest.hexdigest()


def read_regular_bytes(path: Path, label: str) -> tuple[bytes, dict[str, Any]]:
    handle, opened = open_regular_no_follow(path, label)
    with handle:
        raw = handle.read()
        after = os.fstat(handle.fileno())
        require(_same_identity(opened, after), f"{label} changed while being read")
    return raw, {
        "filename": Path(path).name,
        "size_bytes": opened.st_size,
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _safe_repository_path(value: object, label: str) -> str:
    require(isinstance(value, str) and value, f"Missing {label} repository path")
    require("\\" not in value, f"Backslash is forbidden in {label} repository path")
    path = PurePosixPath(value)
    require(not path.is_absolute(), f"Absolute {label} repository path is forbidden")
    require(".." not in path.parts and "." not in path.parts, f"Unsafe {label} repository path")
    normalized = path.as_posix()
    require(normalized == value, f"Non-canonical {label} repository path")
    return normalized


def resolve_repository_path(root: Path, value: object, label: str) -> Path:
    relative = _safe_repository_path(value, label)
    root = _absolute(root)
    resolved = _absolute(root / PurePosixPath(relative))
    require(
        os.path.commonpath((os.fspath(root), os.fspath(resolved))) == os.fspath(root),
        f"{label} escapes the repository root",
    )
    return resolved


def _artifact_descriptor(
    repository_path: str,
    descriptor: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "repository_path": repository_path,
        "filename": descriptor["filename"],
        "schema": payload["schema"],
        "status": payload["status"],
        "size_bytes": descriptor["size_bytes"],
        "sha256": descriptor["sha256"],
        "canonical_payload_sha256": canonical_sha256(payload),
    }


def _validate_source_descriptor(value: object, dataset: str) -> Mapping[str, Any]:
    require(isinstance(value, dict), f"Missing source descriptor: {dataset}")
    require(
        isinstance(value.get("filename"), str) and value["filename"],
        f"Missing source filename: {dataset}",
    )
    require(
        isinstance(value.get("size_bytes"), int)
        and not isinstance(value["size_bytes"], bool)
        and value["size_bytes"] > 0,
        f"Invalid source size: {dataset}",
    )
    require(is_sha256(value.get("sha256")), f"Invalid source SHA-256: {dataset}")
    return value


def build_portable_hash_lock(
    manifest_path: Path,
    preflight_path: Path,
) -> dict[str, Any]:
    """Derive the deterministic portable lock from two sealed artifacts."""

    manifest_raw, manifest_descriptor = read_regular_bytes(
        manifest_path, "sealed manifest"
    )
    preflight_raw, preflight_descriptor = read_regular_bytes(
        preflight_path, "sealed preflight"
    )
    manifest = strict_json_loads(manifest_raw, "sealed manifest")
    preflight = strict_json_loads(preflight_raw, "sealed preflight")
    require(isinstance(manifest, dict), "Manifest root must be an object")
    require(isinstance(preflight, dict), "Preflight root must be an object")
    require(manifest.get("schema") == MANIFEST_SCHEMA, "Bad manifest schema")
    require(manifest.get("status") == "sealed", "Manifest is not production sealed")
    require(preflight.get("schema") == PREFLIGHT_SCHEMA, "Bad preflight schema")
    require(preflight.get("status") == "passed", "Preflight did not pass")
    require(
        manifest_raw == formatted_json_bytes(manifest),
        "Manifest bytes are not in the registered canonical encoding",
    )
    require(
        preflight_raw == formatted_json_bytes(preflight),
        "Preflight bytes are not in the registered canonical encoding",
    )

    manifest_lock = _artifact_descriptor(
        REPOSITORY_PATHS["manifest"], manifest_descriptor, manifest
    )
    preflight_lock = _artifact_descriptor(
        REPOSITORY_PATHS["preflight"], preflight_descriptor, preflight
    )
    expected_manifest_cross_link = {
        "filename": manifest_descriptor["filename"],
        "size_bytes": manifest_descriptor["size_bytes"],
        "sha256": manifest_descriptor["sha256"],
    }
    require(
        preflight.get("sealed_manifest") == expected_manifest_cross_link,
        "Preflight does not bind the exact manifest bytes",
    )
    manifest_config = manifest.get("registration", {}).get("configuration")
    preflight_config = preflight.get("configuration")
    require(
        isinstance(manifest_config, dict) and manifest_config == preflight_config,
        "Manifest and preflight split-configuration descriptors differ",
    )
    require(
        manifest_config.get("filename") == Path(REPOSITORY_PATHS["split_configuration"]).name,
        "Split-configuration filename differs from registration",
    )
    require(
        isinstance(manifest_config.get("size_bytes"), int)
        and manifest_config["size_bytes"] > 0
        and is_sha256(manifest_config.get("sha256")),
        "Invalid split-configuration descriptor",
    )

    partition = manifest.get("partition")
    require(isinstance(partition, dict), "Manifest lacks partition")
    excluded_targets = partition.get("global_exclusion_targets")
    require(
        isinstance(excluded_targets, list)
        and excluded_targets
        and all(isinstance(value, str) and value for value in excluded_targets),
        "Manifest has invalid global exclusion targets",
    )
    require(
        excluded_targets == sorted(set(excluded_targets)),
        "Global exclusion targets are not sorted and unique",
    )
    excluded_count = len(excluded_targets)
    excluded_hash = canonical_sha256(excluded_targets)
    require(
        partition.get("global_exclusion_target_count") == excluded_count,
        "Manifest exclusion-target count changed",
    )
    require(
        partition.get("global_exclusion_target_list_sha256") == excluded_hash,
        "Manifest exclusion-target hash changed",
    )
    require(
        preflight.get("global_exclusion_target_count") == excluded_count
        and preflight.get("global_exclusion_target_list_sha256") == excluded_hash,
        "Manifest and preflight exclusion targets differ",
    )
    scopes = preflight.get("global_exclusion_scopes")
    require(scopes == list(PROTECTED_STAGES), "Protected-stage registration changed")
    require(
        manifest.get("data_firewall", {}).get("global_exclusion_scopes") == scopes,
        "Manifest and preflight protected stages differ",
    )
    source_contract = preflight.get("consumer_contract")
    require(isinstance(source_contract, dict), "Preflight lacks consumer contract")
    for key in (
        "only_registered_allowed_source_row_indices_may_be_loaded",
        "complete_allowed_index_sequence_must_be_authorized_before_any_stage",
        "receipt_must_be_authenticated_before_model_fit",
        "receipt_must_be_authenticated_before_checkpoint_selection",
        "receipt_must_be_authenticated_before_normalization_or_feature_selection",
        "receipt_must_be_authenticated_before_graph_or_decoder_construction",
        "held_target_labels_in_any_allowed_batch_are_fatal",
    ):
        require(source_contract.get(key) is True, f"Preflight consumer contract failed: {key}")
    require(
        source_contract.get("expression_matrix_values_accessed_by_preflight") is False,
        "Preflight accessed expression values",
    )
    require(
        source_contract.get("production_training_ready") is False
        and source_contract.get("portable_trainer_integration_implemented") is False,
        "Sealed audit artifacts must predate portable-trainer activation",
    )

    dataset_values = preflight.get("datasets")
    require(isinstance(dataset_values, list) and dataset_values, "Preflight has no datasets")
    locked_datasets: list[dict[str, Any]] = []
    observed_names: set[str] = set()
    for value in dataset_values:
        require(isinstance(value, dict), "Preflight dataset entry is not an object")
        dataset = value.get("dataset")
        require(isinstance(dataset, str) and dataset, "Preflight dataset has no identity")
        require(dataset not in observed_names, f"Duplicate preflight dataset: {dataset}")
        observed_names.add(dataset)
        role = DATASET_ROLES.get(dataset)
        require(role is not None, f"No portable role registered for dataset: {dataset}")
        source = _validate_source_descriptor(value.get("source"), dataset)
        row_count = value.get("row_count")
        allowed_count = value.get("allowed_row_count")
        excluded_row_count = value.get("excluded_row_count")
        require(
            all(
                isinstance(number, int)
                and not isinstance(number, bool)
                and number >= 0
                for number in (row_count, allowed_count, excluded_row_count)
            ),
            f"Invalid row counts: {dataset}",
        )
        require(
            row_count == allowed_count + excluded_row_count,
            f"Allowed and excluded row counts do not cover source: {dataset}",
        )
        require(source.get("row_count", source.get("cell_count")) == row_count, f"Source row count differs: {dataset}")
        for key in (
            "allowed_source_row_indices_sha256",
            "excluded_source_row_indices_sha256",
            "allowed_target_list_sha256",
        ):
            require(is_sha256(value.get(key)), f"Invalid {key}: {dataset}")
        require(
            value.get("held_targets_absent_from_allowed_rows") is True,
            f"Held targets remain in allowed rows: {dataset}",
        )
        locked_datasets.append(
            {
                "dataset": dataset,
                "role": role,
                "source": {
                    "filename": source["filename"],
                    "size_bytes": source["size_bytes"],
                    "sha256": source["sha256"],
                    "row_count": row_count,
                },
                "allowed_row_count": allowed_count,
                "excluded_row_count": excluded_row_count,
                "allowed_source_row_indices_sha256": value[
                    "allowed_source_row_indices_sha256"
                ],
                "excluded_source_row_indices_sha256": value[
                    "excluded_source_row_indices_sha256"
                ],
                "allowed_target_list_sha256": value[
                    "allowed_target_list_sha256"
                ],
            }
        )
    locked_datasets.sort(key=lambda item: item["dataset"])

    return {
        "schema": LOCK_SCHEMA,
        "status": "locked",
        "hash_algorithm": "sha256",
        "canonical_json_encoding": CANONICAL_ENCODING,
        "repository_artifacts": {
            "split_configuration": {
                "repository_path": REPOSITORY_PATHS["split_configuration"],
                "filename": manifest_config["filename"],
                "size_bytes": manifest_config["size_bytes"],
                "sha256": manifest_config["sha256"],
            },
            "manifest": manifest_lock,
            "preflight": preflight_lock,
        },
        "manifest_preflight_cross_link": expected_manifest_cross_link,
        "firewall": {
            "global_exclusion_target_count": excluded_count,
            "global_exclusion_target_list_sha256": excluded_hash,
            "protected_stages": list(PROTECTED_STAGES),
        },
        "datasets": locked_datasets,
        "consumer_contract": {
            "activation_scope": "phase1_portable_trainer_adapter_smoke",
            "compact_model_training_ready": False,
            "official_submission_allowed": False,
            "manifest_and_preflight_are_repository_tracked": True,
            "trainer_must_verify_expected_lock_sha256": True,
            "trainer_must_not_import_or_run_kmeans": True,
            "trainer_derives_rows_from_authenticated_labels_only": True,
            "complete_allowed_row_hash_required_before_any_protected_stage": True,
            "every_batch_must_bind_source_indices_and_labels": True,
            "evaluation_only_sources_cannot_enter_fit_stages": True,
            "absolute_paths_host_metadata_and_timestamps_embedded": False,
        },
    }


@dataclass(frozen=True)
class DatasetRule:
    dataset: str
    role: str
    source: Mapping[str, Any]
    allowed_row_count: int
    excluded_row_count: int
    allowed_source_row_indices_sha256: str
    excluded_source_row_indices_sha256: str
    allowed_target_list_sha256: str


@dataclass(frozen=True)
class PortableTrainingGate:
    """Authenticated lock state used to open registered HDF5 sources."""

    lock_descriptor: Mapping[str, Any]
    manifest_descriptor: Mapping[str, Any]
    preflight_descriptor: Mapping[str, Any]
    split_configuration_descriptor: Mapping[str, Any]
    excluded_targets: frozenset[str]
    protected_stages: frozenset[str]
    datasets: Mapping[str, DatasetRule]

    def open_h5_source(self, dataset: str, path: Path) -> "AuthorizedH5Source":
        rule = self.datasets.get(dataset)
        require(rule is not None, f"Dataset is absent from portable lock: {dataset}")
        return AuthorizedH5Source(self, rule, path)


def load_portable_training_gate(
    *,
    repository_root: Path,
    lock_path: Path,
    manifest_path: Path,
    preflight_path: Path,
    split_configuration_path: Path,
    expected_lock_sha256: str,
) -> PortableTrainingGate:
    """Authenticate tracked gate artifacts without importing the split builder."""

    require(is_sha256(expected_lock_sha256), "A lowercase expected lock SHA-256 is required")
    lock_raw, lock_descriptor = read_regular_bytes(lock_path, "portable hash lock")
    require(
        lock_descriptor["sha256"] == expected_lock_sha256,
        "Portable hash lock differs from the independently expected SHA-256",
    )
    lock = strict_json_loads(lock_raw, "portable hash lock")
    require(isinstance(lock, dict), "Portable hash lock root must be an object")
    require(
        lock_raw == formatted_json_bytes(lock),
        "Portable hash lock is not in the registered canonical encoding",
    )
    expected = build_portable_hash_lock(manifest_path, preflight_path)
    require(lock == expected, "Portable hash lock does not match sealed artifacts")

    artifacts = lock["repository_artifacts"]
    supplied_paths = {
        "split_configuration": split_configuration_path,
        "manifest": manifest_path,
        "preflight": preflight_path,
        "lock": lock_path,
    }
    for label, supplied in supplied_paths.items():
        registered_path = REPOSITORY_PATHS[label]
        expected_path = resolve_repository_path(repository_root, registered_path, label)
        require(
            _absolute(supplied) == expected_path,
            f"{label} path differs from the repository-tracked location",
        )
    config_raw, config_descriptor = read_regular_bytes(
        split_configuration_path, "split-audit configuration"
    )
    del config_raw
    require(
        config_descriptor
        == {
            "filename": artifacts["split_configuration"]["filename"],
            "size_bytes": artifacts["split_configuration"]["size_bytes"],
            "sha256": artifacts["split_configuration"]["sha256"],
        },
        "Split-audit configuration differs from the sealed descriptor",
    )
    manifest_raw, manifest_descriptor = read_regular_bytes(manifest_path, "sealed manifest")
    preflight_raw, preflight_descriptor = read_regular_bytes(preflight_path, "sealed preflight")
    require(
        manifest_descriptor
        == {
            "filename": artifacts["manifest"]["filename"],
            "size_bytes": artifacts["manifest"]["size_bytes"],
            "sha256": artifacts["manifest"]["sha256"],
        },
        "Manifest identity changed after portable-lock authentication",
    )
    require(
        preflight_descriptor
        == {
            "filename": artifacts["preflight"]["filename"],
            "size_bytes": artifacts["preflight"]["size_bytes"],
            "sha256": artifacts["preflight"]["sha256"],
        },
        "Preflight identity changed after portable-lock authentication",
    )
    manifest = strict_json_loads(manifest_raw, "sealed manifest")
    preflight = strict_json_loads(preflight_raw, "sealed preflight")
    excluded_values = manifest["partition"]["global_exclusion_targets"]
    require(
        canonical_sha256(excluded_values)
        == lock["firewall"]["global_exclusion_target_list_sha256"],
        "Portable gate exclusion-target hash changed",
    )
    require(
        preflight["sealed_manifest"] == lock["manifest_preflight_cross_link"],
        "Portable gate lost its manifest/preflight cross-link",
    )
    rules: dict[str, DatasetRule] = {}
    for value in lock["datasets"]:
        rules[value["dataset"]] = DatasetRule(
            dataset=value["dataset"],
            role=value["role"],
            source=MappingProxyType(dict(value["source"])),
            allowed_row_count=value["allowed_row_count"],
            excluded_row_count=value["excluded_row_count"],
            allowed_source_row_indices_sha256=value[
                "allowed_source_row_indices_sha256"
            ],
            excluded_source_row_indices_sha256=value[
                "excluded_source_row_indices_sha256"
            ],
            allowed_target_list_sha256=value["allowed_target_list_sha256"],
        )
    return PortableTrainingGate(
        lock_descriptor=MappingProxyType(dict(lock_descriptor)),
        manifest_descriptor=MappingProxyType(dict(manifest_descriptor)),
        preflight_descriptor=MappingProxyType(dict(preflight_descriptor)),
        split_configuration_descriptor=MappingProxyType(dict(config_descriptor)),
        excluded_targets=frozenset(excluded_values),
        protected_stages=frozenset(lock["firewall"]["protected_stages"]),
        datasets=MappingProxyType(rules),
    )


def _require_hard_path(handle: h5py.File, path: str) -> None:
    parts: list[str] = []
    for component in PurePosixPath(path).parts:
        parts.append(component)
        joined = "/".join(parts)
        require(
            isinstance(handle.get(joined, getlink=True), h5py.HardLink),
            f"HDF5 metadata path is not a hard link: {joined}",
        )


def _require_internal_dataset(dataset: h5py.Dataset, label: str) -> None:
    require(not dataset.is_virtual, f"HDF5 dataset is virtual: {label}")
    require(not dataset.external, f"HDF5 dataset uses external storage: {label}")


def _decode_strings(values: np.ndarray) -> list[str]:
    return [
        item.decode("utf-8") if isinstance(item, (bytes, np.bytes_)) else str(item)
        for item in np.asarray(values).reshape(-1)
    ]


def _read_h5ad_column(handle: h5py.File, path: str) -> list[str]:
    _require_hard_path(handle, path)
    require(path in handle, f"Missing HDF5 metadata column: {path}")
    node = handle[path]
    if isinstance(node, h5py.Group):
        _require_hard_path(handle, f"{path}/codes")
        _require_hard_path(handle, f"{path}/categories")
        codes_node = node.get("codes")
        categories_node = node.get("categories")
        require(
            isinstance(codes_node, h5py.Dataset)
            and isinstance(categories_node, h5py.Dataset),
            f"Malformed categorical metadata: {path}",
        )
        _require_internal_dataset(codes_node, f"{path}/codes")
        _require_internal_dataset(categories_node, f"{path}/categories")
        categories = _decode_strings(categories_node[:])
        codes = np.asarray(codes_node[:], dtype=np.int64)
        require(len(categories) == len(set(categories)), f"Duplicate categories: {path}")
        require(np.all((codes >= 0) & (codes < len(categories))), f"Bad category codes: {path}")
        return [categories[int(code)] for code in codes]
    require(isinstance(node, h5py.Dataset), f"Unsupported metadata node: {path}")
    _require_internal_dataset(node, path)
    if "categories" in node.attrs:
        reference = node.attrs["categories"]
        if isinstance(reference, bytes):
            try:
                reference = reference.decode("utf-8")
            except UnicodeDecodeError as error:
                raise PortableTrainingGateError(
                    f"Invalid category path: {path}"
                ) from error
        if isinstance(reference, str):
            category_path = reference.lstrip("/")
            require(category_path, f"Empty category path: {path}")
            _require_hard_path(handle, category_path)
            categories_node = handle[category_path]
        else:
            require(
                isinstance(reference, h5py.Reference) and bool(reference),
                f"Unsupported category reference: {path}",
            )
            categories_node = handle[reference]
            require(
                isinstance(categories_node.name, str)
                and categories_node.name not in {"", "/"},
                f"Unlinked category dataset: {path}",
            )
            _require_hard_path(handle, categories_node.name.lstrip("/"))
        require(isinstance(categories_node, h5py.Dataset), f"Bad category target: {path}")
        _require_internal_dataset(categories_node, f"{path} categories")
        categories = _decode_strings(categories_node[:])
        codes = np.asarray(node[:], dtype=np.int64)
        require(len(categories) == len(set(categories)), f"Duplicate categories: {path}")
        require(np.all((codes >= 0) & (codes < len(categories))), f"Bad category codes: {path}")
        return [categories[int(code)] for code in codes]
    return _decode_strings(node[:])


@dataclass
class AuthorizedH5Source:
    """Same-descriptor HDF5 view restricted to the locked allowed rows."""

    gate: PortableTrainingGate
    rule: DatasetRule
    path: Path
    labels: tuple[str, ...] = field(init=False, default=())
    allowed_source_row_indices: tuple[int, ...] = field(init=False, default=())
    excluded_source_row_indices: tuple[int, ...] = field(init=False, default=())
    _handle: BinaryIO | None = field(init=False, default=None, repr=False)
    _opened: os.stat_result | None = field(init=False, default=None, repr=False)
    _h5: h5py.File | None = field(init=False, default=None, repr=False)
    _authorized_stages: set[str] = field(init=False, default_factory=set, repr=False)

    def __enter__(self) -> "AuthorizedH5Source":
        require(self._handle is None, "Authorized source is already open")
        handle, opened = open_regular_no_follow(self.path, f"source {self.rule.dataset}")
        try:
            digest = _hash_stream(handle)
            require(Path(self.path).name == self.rule.source["filename"], f"Source filename changed: {self.rule.dataset}")
            require(opened.st_size == self.rule.source["size_bytes"], f"Source size changed: {self.rule.dataset}")
            require(digest == self.rule.source["sha256"], f"Source SHA-256 changed: {self.rule.dataset}")
            handle.seek(0)
            h5 = h5py.File(handle, mode="r")
            label_path = "obs/gene" if self.rule.dataset == "nadig_jurkat" else (
                "obs/target_gene" if h5.get("obs/target_gene", getlink=True) is not None else "obs/gene"
            )
            labels = _read_h5ad_column(h5, label_path)
            require(len(labels) == self.rule.source["row_count"], f"Source row count changed: {self.rule.dataset}")
            require(all(labels), f"Empty source label: {self.rule.dataset}")
            excluded_targets = self.gate.excluded_targets
            allowed = tuple(index for index, value in enumerate(labels) if value not in excluded_targets)
            excluded = tuple(index for index, value in enumerate(labels) if value in excluded_targets)
            require(len(allowed) == self.rule.allowed_row_count, f"Allowed row count changed: {self.rule.dataset}")
            require(len(excluded) == self.rule.excluded_row_count, f"Excluded row count changed: {self.rule.dataset}")
            require(canonical_sha256(list(allowed)) == self.rule.allowed_source_row_indices_sha256, f"Allowed row sequence changed: {self.rule.dataset}")
            require(canonical_sha256(list(excluded)) == self.rule.excluded_source_row_indices_sha256, f"Excluded row sequence changed: {self.rule.dataset}")
            require(canonical_sha256(sorted(set(labels) - set(excluded_targets))) == self.rule.allowed_target_list_sha256, f"Allowed target set changed: {self.rule.dataset}")
            self._handle = handle
            self._opened = opened
            self._h5 = h5
            self.labels = tuple(labels)
            self.allowed_source_row_indices = allowed
            self.excluded_source_row_indices = excluded
            return self
        except Exception:
            handle.close()
            raise

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        try:
            if self._h5 is not None:
                self._h5.close()
            if self._handle is not None and not self._handle.closed:
                after = os.fstat(self._handle.fileno())
                require(self._opened is not None and _same_identity(self._opened, after), f"Source changed while open: {self.rule.dataset}")
        finally:
            if self._handle is not None and not self._handle.closed:
                self._handle.close()
            self._h5 = None
            self._handle = None

    def authorize_stage(self, stage: str) -> None:
        require(stage in self.gate.protected_stages, f"Unregistered protected stage: {stage}")
        require(self._h5 is not None, "Authorized source is not open")
        require(self.rule.role == "training", f"Evaluation-only source cannot enter {stage}: {self.rule.dataset}")
        # The complete sequence was reconstructed and hash-checked in __enter__.
        require(len(self.allowed_source_row_indices) == self.rule.allowed_row_count, f"Incomplete allowed row sequence: {self.rule.dataset}")
        self._authorized_stages.add(stage)

    @property
    def authorized_stages(self) -> tuple[str, ...]:
        """Return the sorted stages opened after complete row authorization."""

        return tuple(sorted(self._authorized_stages))

    def validate_batch(
        self,
        source_row_indices: Sequence[int],
        labels: Sequence[str],
        stage: str,
    ) -> None:
        require(stage in self._authorized_stages, f"Stage was not authorized before batch access: {stage}")
        observed = tuple(source_row_indices)
        require(len(observed) == len(labels) and observed, "Batch indices and labels do not align")
        require(all(isinstance(index, int) and not isinstance(index, bool) for index in observed), "Batch has non-integer source indices")
        require(len(set(observed)) == len(observed), "Batch repeats a source row")
        allowed = set(self.allowed_source_row_indices)
        require(all(index in allowed for index in observed), f"Batch contains unauthorized source rows: {self.rule.dataset}")
        require(all(0 <= index < len(self.labels) for index in observed), "Batch source index is out of bounds")
        expected_labels = tuple(self.labels[index] for index in observed)
        require(tuple(labels) == expected_labels, f"Batch labels do not match authenticated source rows: {self.rule.dataset}")
        contamination = sorted(set(labels) & set(self.gate.excluded_targets))
        require(not contamination, f"Batch contains sealed target labels: {contamination[:8]}")

    def read_dense_expression(
        self,
        source_row_indices: Sequence[int],
        labels: Sequence[str],
        gene_indices: Sequence[int],
        stage: str,
    ) -> np.ndarray:
        """Read one authorized dense batch after complete-selection approval."""

        self.validate_batch(source_row_indices, labels, stage)
        require(self._h5 is not None, "Authorized source is not open")
        _require_hard_path(self._h5, "X")
        matrix = self._h5["X"]
        require(isinstance(matrix, h5py.Dataset), "Portable smoke supports a dense X dataset only")
        _require_internal_dataset(matrix, "X")
        rows = tuple(source_row_indices)
        genes = tuple(gene_indices)
        require(rows == tuple(sorted(rows)), "Batch source indices must be increasing")
        require(genes and genes == tuple(sorted(set(genes))), "Gene indices must be increasing and unique")
        require(genes[0] >= 0, "Gene index is negative")
        require(genes[-1] < matrix.shape[1], "Gene index is out of bounds")
        values = np.asarray(matrix[list(rows), :], dtype=np.float32)[:, list(genes)]
        require(values.shape == (len(rows), len(genes)), "Dense batch shape changed")
        require(np.all(np.isfinite(values)) and np.all(values >= 0), "Expression batch is not finite nonnegative")
        return values
