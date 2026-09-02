#!/usr/bin/env python3
"""Build and authenticate the sealed V7 Jurkat non-harm split.

This module reads Jurkat labels, batch/design metadata, cell identifiers, and
matrix shape metadata.  It never reads values from the treated expression
matrix.  Production split construction accepts only the exact Arc-supplied
opaque target-feature artifact registered by Gate 1c.  Gate 1c authenticates
the released bytes and their structure; it does not claim the unavailable
upstream model, sequence mapping, or pooling lineage required by Gate 1b.
Training remains closed until a portable trainer consumes the immutable
manifest and preflight without recomputing the split.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import secrets
import stat
import tomllib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, BinaryIO, Mapping, Sequence

import h5py
import numpy as np
import sklearn
import torch
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits


MANIFEST_SCHEMA = "vcc-scdfm-jurkat-non-harm-manifest-v1"
REGISTRATION_SCHEMA = "vcc-scdfm-jurkat-non-harm-registration-v1"
TRAINING_PREFLIGHT_SCHEMA = "vcc-scdfm-jurkat-training-preflight-v1"
COMPLETE_ESM2_RECEIPT_SCHEMA = "vcc-esm2-target-conditioning-provenance-v1"
OPAQUE_ESM2_RECEIPT_SCHEMA = (
    "vcc-arc-esm2-target-features-authentication-gate1c-v2"
)
LEGACY_OPAQUE_ESM2_RECEIPT_SCHEMA = (
    "vcc-arc-esm2-target-features-authentication-v1"
)
CONTROL_LABEL = "non-targeting"
UNKNOWN_VALUES = {"", "UNKNOWN", "UNKNOWN_NOT_PROVIDED", "UNRESOLVED_REQUIRED"}
SHA256_HEX = set("0123456789abcdef")
_TEST_ONLY_SYNTHETIC_LINEAGE_BYPASS = object()
OFFICIAL_GATE_1C = {
    "source_url": (
        "https://storage.googleapis.com/vcc_data_prod/datasets/state/"
        "competition_support_set.zip"
    ),
    "receipt_path": (
        "dataset/state_support/receipts/"
        "esm2_target_features_gate1c_v2.json"
    ),
    "receipt_size_bytes": 4_864,
    "receipt_sha256": "a858816ce4ac4dc6581d0dda698c9b146bb6e33c70f132078317c3fd0b99be62",
    "source_object": {
        "provider": "Google Cloud Storage",
        "generation": "1763329752965344",
        "size_bytes": 8_716_992_349,
        "crc32c_base64": "MBP7uA==",
        "crc32c_hex": "3013fbb8",
        "component_count": 32,
        "last_modified": "Sun, 16 Nov 2025 21:49:12 GMT",
        "metadata_scope": "observed_http_object_metadata",
        "source_url_claimed_permanently_immutable": False,
    },
    "archive_filename": "competition_support_set.zip",
    "archive_size_bytes": 8_716_992_349,
    "archive_sha256": "f1d5fd56a2eee240e5585fac52bb75b0b830a7be6a5f6de1bf773bff4f507f7c",
    "member_name": "competition_support_set/ESM2_pert_features.pt",
    "member_size_bytes": 410_886_729,
    "member_sha256": "a210e1cc7901513999b2bca3836ba9e2f203cd008be4e9a9d6412a2267de9748",
    "member_crc32": "85d91f50",
    "feature_filename": "ESM2_pert_features.pt",
    "feature_count": 19_790,
    "embedding_dimension": 5_120,
    "tensor_dtype": "torch.float32",
    "tensor_layout": "torch.strided",
    "canonical_key_list_sha256": "47f457d410373f0ef4e2743b6652c20b813660c9230acbe8fd5c80c7d1e94ef1",
    "canonical_map_sha256": "e41e97901c2e208e27c631f67c533f01bab2314ef1f9367190f832256866b399",
    "target_manifest_filename": "pert_counts.csv",
    "target_manifest_size_bytes": 1_906,
    "target_manifest_sha256": "f57edd7b912ebd718efc7ee9d0f334772513e7cc418d133ce525470e373b3276",
    "target_count": 300,
    "ordered_target_list_sha256": "f170abd94f4735ab7a16efc8a3263076d50e0e470f989d38033d17e2edf5fd13",
    "ordered_target_feature_subset_sha256": "2149553a0783bb3959403e62913b1ff9983e2abc2a1e99025608c0660d12d2d8",
}


class JurkatManifestError(RuntimeError):
    """Raised when an input violates the sealed-split contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise JurkatManifestError(message)


def canonical_json_bytes(value: object) -> bytes:
    """Return the registered canonical JSON encoding."""

    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def formatted_json_bytes(value: object) -> bytes:
    """Return the exact on-disk encoding used for sealed JSON artifacts."""

    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value).issubset(SHA256_HEX)
    )


def strict_json_loads(raw: bytes, label: str) -> Any:
    """Decode strict UTF-8 JSON while rejecting duplicate object keys."""

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise JurkatManifestError(f"Duplicate JSON key in {label}: {key}")
            result[key] = value
        return result

    try:
        text = raw.decode("utf-8")
        return json.loads(text, object_pairs_hook=unique_object)
    except JurkatManifestError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise JurkatManifestError(f"Invalid {label} JSON: {error}") from error


def absolute_path(path: Path) -> Path:
    """Normalize a path lexically without following a symlink."""

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
    """Open a regular file while rejecting symlinks in every path component."""

    absolute = absolute_path(path)
    parts = absolute.parts
    require(len(parts) >= 2 and parts[0] == os.sep, f"Invalid {label} path")
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    file_flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    file_flags |= getattr(os, "O_NOFOLLOW", 0)
    directory_descriptor: int | None = None
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
        descriptor = os.open(parts[-1], file_flags, dir_fd=directory_descriptor)
    except OSError as error:
        raise JurkatManifestError(
            f"Unable to open {label}: {type(error).__name__}: {error}"
        ) from error
    finally:
        if directory_descriptor is not None:
            os.close(directory_descriptor)
    handle = os.fdopen(descriptor, "rb")
    opened = os.fstat(handle.fileno())
    if not stat.S_ISREG(opened.st_mode):
        handle.close()
        raise JurkatManifestError(f"{label} is not a regular file")
    return handle, opened


def sha256_stream(handle: BinaryIO, chunk_size: int = 8 << 20) -> str:
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


def resolve_registered_path(repository: Path, value: object, label: str) -> Path:
    require(isinstance(value, str) and value not in UNKNOWN_VALUES, f"Unresolved {label}")
    path = Path(value)
    return absolute_path(path if path.is_absolute() else repository / path)


def _decode_strings(values: np.ndarray) -> list[str]:
    return [
        item.decode("utf-8") if isinstance(item, (bytes, np.bytes_)) else str(item)
        for item in np.asarray(values).reshape(-1)
    ]


def _require_hard_path(handle: h5py.File, path: str) -> None:
    """Reject link indirection in every component of a consumed HDF5 path."""

    prefix: list[str] = []
    for component in Path(path).parts:
        prefix.append(component)
        joined = "/".join(prefix)
        link = handle.get(joined, getlink=True)
        require(isinstance(link, h5py.HardLink), f"HDF5 metadata path is not a hard link: {joined}")


def _require_internal_dataset(dataset: h5py.Dataset, label: str) -> None:
    require(not dataset.is_virtual, f"HDF5 metadata dataset is virtual: {label}")
    require(not dataset.external, f"HDF5 metadata dataset uses external storage: {label}")


def _require_internal_category_dataset(
    handle: h5py.File,
    reference: object,
    label: str,
) -> h5py.Dataset:
    """Resolve an in-file categorical target without following path links."""

    if isinstance(reference, bytes):
        try:
            reference = reference.decode("utf-8")
        except UnicodeDecodeError as error:
            raise JurkatManifestError(f"Invalid category path in {label}") from error
    if isinstance(reference, str):
        category_path = reference.lstrip("/")
        require(category_path, f"Empty category path in {label}")
        _require_hard_path(handle, category_path)
        category_node = handle[category_path]
    else:
        require(
            isinstance(reference, h5py.Reference) and bool(reference),
            f"Bad category reference in {label}",
        )
        category_node = handle[reference]
        require(
            isinstance(category_node.name, str)
            and category_node.name not in {"", "/"},
            f"Unlinked category dataset in {label}",
        )
        _require_hard_path(handle, category_node.name.lstrip("/"))
    require(isinstance(category_node, h5py.Dataset), f"Bad category target in {label}")
    _require_internal_dataset(category_node, f"{label} category reference")
    return category_node


def read_h5ad_column(handle: h5py.File, path: str) -> list[str]:
    """Read one AnnData dataframe column without touching ``X`` values."""

    _require_hard_path(handle, path)
    require(path in handle, f"Missing H5AD metadata column: {path}")
    node = handle[path]
    if isinstance(node, h5py.Group):
        require("codes" in node and "categories" in node, f"Malformed categorical {path}")
        _require_hard_path(handle, f"{path}/codes")
        _require_hard_path(handle, f"{path}/categories")
        codes_node = node["codes"]
        categories_node = node["categories"]
        require(
            isinstance(codes_node, h5py.Dataset)
            and isinstance(categories_node, h5py.Dataset),
            f"Malformed categorical datasets in {path}",
        )
        _require_internal_dataset(codes_node, f"{path}/codes")
        _require_internal_dataset(categories_node, f"{path}/categories")
        codes = np.asarray(codes_node[:], dtype=np.int64)
        categories = _decode_strings(categories_node[:])
        require(len(categories) == len(set(categories)), f"Duplicate categories in {path}")
        require(np.all((codes >= 0) & (codes < len(categories))), f"Bad codes in {path}")
        return [categories[int(code)] for code in codes]
    require(isinstance(node, h5py.Dataset), f"Unsupported H5AD node: {path}")
    _require_internal_dataset(node, path)
    if "categories" in node.attrs:
        reference = node.attrs["categories"]
        category_node = _require_internal_category_dataset(handle, reference, path)
        categories = _decode_strings(category_node[:])
        require(len(categories) == len(set(categories)), f"Duplicate categories in {path}")
        codes = np.asarray(node[:], dtype=np.int64)
        require(np.all((codes >= 0) & (codes < len(categories))), f"Bad codes in {path}")
        return [categories[int(code)] for code in codes]
    return _decode_strings(node[:])


def read_jurkat_metadata(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Authenticate the raw file and return label-only split inputs."""

    handle, opened = open_regular_no_follow(path, "raw Jurkat H5AD")
    with handle:
        raw_sha256 = sha256_stream(handle)
        handle.seek(0)
        try:
            with h5py.File(handle, mode="r") as data:
                require("X" in data, "Raw Jurkat H5AD lacks X")
                _require_hard_path(data, "X")
                matrix = data["X"]
                require(hasattr(matrix, "shape") and len(matrix.shape) == 2, "Bad X shape")
                if isinstance(matrix, h5py.Dataset):
                    _require_internal_dataset(matrix, "X")
                shape = (int(matrix.shape[0]), int(matrix.shape[1]))
                matrix_dtype = str(matrix.dtype)
                cell_barcodes = read_h5ad_column(data, "obs/cell_barcode")
                labels = read_h5ad_column(data, "obs/gene")
                batches = read_h5ad_column(data, "obs/gem_group")
                guide_designs = read_h5ad_column(data, "obs/sgID_AB")
                gene_names = read_h5ad_column(data, "var/gene_name")
                require(
                    len(cell_barcodes) == len(labels) == len(batches) == len(guide_designs) == shape[0],
                    "Jurkat obs metadata length differs from X rows",
                )
                require(len(gene_names) == shape[1], "Jurkat var metadata differs from X columns")
                require(all(cell_barcodes), "Empty cell barcode")
                require(all(labels), "Empty Jurkat target label")
                require(len(set(cell_barcodes)) == len(cell_barcodes), "Duplicate cell barcodes")
                require(CONTROL_LABEL in labels, "Jurkat controls are absent")
        except JurkatManifestError:
            raise
        except Exception as error:
            raise JurkatManifestError(
                f"Unable to read raw Jurkat metadata: {type(error).__name__}: {error}"
            ) from error
        after = os.fstat(handle.fileno())
        require(_same_identity(opened, after), "Raw Jurkat H5AD changed during authentication")

    counts = Counter(labels)
    treated_targets = sorted(set(labels) - {CONTROL_LABEL})
    metadata = {
        "path": Path(path),
        "shape": shape,
        "matrix_dtype": matrix_dtype,
        "cell_barcodes": cell_barcodes,
        "labels": labels,
        "batches": batches,
        "guide_designs": guide_designs,
        "gene_names": sorted(set(gene_names)),
        "target_counts": counts,
        "treated_targets": treated_targets,
    }
    descriptor = {
        "filename": Path(path).name,
        "size_bytes": opened.st_size,
        "sha256": raw_sha256,
        "shape": list(shape),
        "matrix_dtype": matrix_dtype,
        "obs_columns_used": ["cell_barcode", "gene", "gem_group", "sgID_AB"],
        "var_columns_used": ["gene_name"],
        "expression_matrix_values_accessed": False,
        "opaque_file_bytes_hashed_without_expression_deserialization": True,
        "cell_count": shape[0],
        "gene_columns": shape[1],
        "treated_target_count": len(treated_targets),
        "control_cells": int(counts[CONTROL_LABEL]),
        "batch_count": len(set(batches)),
        "guide_design_count": len(set(guide_designs)),
    }
    return metadata, descriptor


def read_target_csv(path: Path, expected_sha256: str) -> tuple[list[str], dict[str, Any]]:
    raw, descriptor = read_regular_bytes(path, "historical public panel")
    require(descriptor["sha256"] == expected_sha256, "Historical panel SHA-256 changed")
    try:
        rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8"), newline="")))
    except (UnicodeDecodeError, csv.Error) as error:
        raise JurkatManifestError(f"Invalid historical panel CSV: {error}") from error
    require(rows and "target_gene" in rows[0], "Historical panel lacks target_gene")
    targets = [row["target_gene"] for row in rows]
    require(all(targets) and len(set(targets)) == len(targets), "Bad historical targets")
    descriptor["target_count"] = len(targets)
    descriptor["target_list_sha256"] = canonical_sha256(sorted(targets))
    return targets, descriptor


def read_state_support_source(
    path: Path,
    registered: Mapping[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    """Authenticate one STATE H5 and return its per-row target labels."""

    handle, opened = open_regular_no_follow(path, f"STATE support {Path(path).name}")
    with handle:
        digest = sha256_stream(handle)
        require(opened.st_size == registered.get("size_bytes"), f"STATE support size changed: {path}")
        require(digest == registered.get("sha256"), f"STATE support SHA-256 changed: {path}")
        handle.seek(0)
        try:
            with h5py.File(handle, mode="r") as data:
                target_link = data.get("obs/target_gene", getlink=True)
                column = "obs/target_gene" if target_link is not None else "obs/gene"
                labels = read_h5ad_column(data, column)
        except JurkatManifestError:
            raise
        except Exception as error:
            raise JurkatManifestError(
                f"Unable to read STATE support target metadata: {error}"
            ) from error
        after = os.fstat(handle.fileno())
        require(_same_identity(opened, after), f"STATE support changed: {path}")
    targets = set(labels) - {CONTROL_LABEL}
    require(targets, f"STATE support target set is empty: {path}")
    target_digest = canonical_sha256(sorted(targets))
    require(len(targets) == registered.get("target_count"), f"STATE support target count changed: {path}")
    require(
        target_digest == registered.get("target_list_sha256"),
        f"STATE support target-list SHA-256 changed: {path}",
    )
    return labels, {
        "filename": Path(path).name,
        "size_bytes": opened.st_size,
        "sha256": digest,
        "row_count": len(labels),
        "target_count": len(targets),
        "target_list_sha256": target_digest,
        "expression_matrix_values_accessed": False,
    }


def read_state_support_targets(
    sources: Sequence[tuple[Path, Mapping[str, Any]]],
) -> tuple[set[str], list[dict[str, Any]]]:
    union: set[str] = set()
    descriptors: list[dict[str, Any]] = []
    for path, registered in sources:
        labels, descriptor = read_state_support_source(path, registered)
        union.update(set(labels) - {CONTROL_LABEL})
        descriptors.append(descriptor)
    descriptors.sort(key=lambda item: str(item["filename"]))
    return union, descriptors


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    array = tensor.detach().cpu().contiguous().numpy()
    return array.astype("<f4", copy=False).tobytes(order="C")


def canonical_feature_key_sha256(keys: Sequence[str]) -> str:
    return hashlib.sha256(
        json.dumps(sorted(keys), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def canonical_feature_map_sha256(features: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(features):
        tensor = features[key]
        key_bytes = key.encode("utf-8")
        metadata = json.dumps(
            {"shape": list(tensor.shape), "dtype": str(tensor.dtype)},
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(key_bytes).to_bytes(8, "little"))
        digest.update(key_bytes)
        digest.update(len(metadata).to_bytes(8, "little"))
        digest.update(metadata)
        digest.update(_tensor_bytes(tensor))
    return digest.hexdigest()


def validate_complete_esm2_receipt(
    receipt: Mapping[str, Any],
    receipt_descriptor: Mapping[str, Any],
    config_provenance: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    require(
        receipt.get("schema")
        not in {OPAQUE_ESM2_RECEIPT_SCHEMA, LEGACY_OPAQUE_ESM2_RECEIPT_SCHEMA},
        "ESM2 upstream provenance is incomplete: an opaque feature receipt cannot open Gate 1b",
    )
    require(receipt.get("schema") == COMPLETE_ESM2_RECEIPT_SCHEMA, "Bad ESM2 receipt schema")
    require(receipt.get("status") == "passed", "ESM2 receipt did not pass")
    contract = receipt.get("contract")
    require(isinstance(contract, dict), "ESM2 receipt lacks contract")
    require(contract.get("provenance_complete") is True, "ESM2 upstream provenance is incomplete")
    require(contract.get("gate_1b_allowed") is True, "ESM2 receipt keeps Gate 1b closed")
    require(config_provenance.get("provenance_complete") is True, "Config ESM2 provenance is incomplete")
    require(config_provenance.get("gate_1b_allowed") is True, "Config keeps Gate 1b closed")
    require(
        config_provenance.get("embedding_receipt_sha256") == receipt_descriptor["sha256"],
        "Config does not bind the exact ESM2 receipt",
    )
    upstream = receipt.get("upstream_provenance")
    require(isinstance(upstream, dict), "ESM2 receipt lacks upstream provenance")
    for key in (
        "model_id",
        "model_revision",
        "pooling_rule",
    ):
        require(
            isinstance(upstream.get(key), str) and upstream[key] not in UNKNOWN_VALUES,
            f"ESM2 receipt has unresolved upstream field: {key}",
        )
    config_bindings = {
        "model_id": "model_id",
        "model_revision": "model_revision",
        "pooling_rule": "pooling_rule",
    }
    for receipt_key, config_key in config_bindings.items():
        require(
            upstream.get(receipt_key) == config_provenance.get(config_key),
            f"Config does not bind ESM2 upstream field: {config_key}",
        )
    artifact_fields = {
        "weights_artifact": ("weights_artifact", "weights_sha256"),
        "protein_sequence_artifact": (
            "protein_sequence_source",
            "protein_sequence_source_sha256",
        ),
        "gene_to_protein_mapping_artifact": (
            "gene_to_protein_mapping",
            "gene_to_protein_mapping_sha256",
        ),
    }
    for receipt_key, (config_path_key, config_hash_key) in artifact_fields.items():
        descriptor = upstream.get(receipt_key)
        require(isinstance(descriptor, dict), f"ESM2 receipt lacks {receipt_key}")
        require(
            isinstance(descriptor.get("filename"), str) and descriptor["filename"],
            f"ESM2 receipt has no filename for {receipt_key}",
        )
        require(
            isinstance(descriptor.get("size_bytes"), int)
            and not isinstance(descriptor["size_bytes"], bool)
            and descriptor["size_bytes"] > 0,
            f"ESM2 receipt has invalid size for {receipt_key}",
        )
        require(is_sha256(descriptor.get("sha256")), f"ESM2 receipt has invalid SHA-256 for {receipt_key}")
        require(
            descriptor["sha256"] == config_provenance.get(config_hash_key),
            f"Config does not bind exact {receipt_key} SHA-256",
        )
        require(
            isinstance(config_provenance.get(config_path_key), str)
            and config_provenance[config_path_key] not in UNKNOWN_VALUES,
            f"Config does not register a path for {receipt_key}",
        )
    artifact = receipt.get("feature_artifact")
    require(isinstance(artifact, dict), "ESM2 receipt lacks feature_artifact")
    structure = artifact.get("structure")
    require(isinstance(structure, dict), "ESM2 receipt lacks feature structure")
    return artifact, upstream


def validate_official_gate_1c_receipt(
    receipt: Mapping[str, Any],
    registration: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Authenticate Arc's official opaque target features without lineage claims."""

    require(receipt.get("schema") == OPAQUE_ESM2_RECEIPT_SCHEMA, "Bad Gate 1c receipt schema")
    require(receipt.get("status") == "passed", "Gate 1c artifact receipt did not pass")
    contract = receipt.get("contract")
    require(isinstance(contract, dict), "Gate 1c receipt lacks contract")
    expected_contract = {
        "artifact_role": "opaque_arc_supplied_target_features",
        "authentication_scope": "artifact_identity_and_structure_only",
        "canonical_tensor_byte_order": "little-endian-float32",
        "does_not_establish_upstream_model_provenance": True,
        "provenance_complete": False,
        "gate_1b_allowed": False,
        "opaque_artifact_training_mode": "official_opaque_artifact",
        "official_opaque_artifact_training_allowed": True,
        "gate_1c_allowed": True,
        "gate_1c_scope": (
            "authenticated_arc_supplied_artifact_identity_and_structure_only"
        ),
    }
    require(contract == expected_contract, "Gate 1c contract differs from registration")
    require(registration.get("official_opaque_artifact") is True, "Gate 1c mode is not official")
    require(registration.get("gate_1c_allowed") is True, "Gate 1c is not enabled")
    require(registration.get("gate_1b_allowed") is False, "Gate 1b must remain closed")
    expected_registration_contract = {
        "status": "authenticated_opaque_artifact",
        "role": "artifact_identity_and_structure_only",
        "opaque_artifact_training_mode": "official_opaque_artifact",
        "official_opaque_artifact_training_allowed": True,
        "gate_1c_scope": (
            "authenticated_arc_supplied_artifact_identity_and_structure_only"
        ),
        "provenance_complete": False,
        "does_not_satisfy_target_conditioning_provenance": True,
        "vcc_target_coverage": OFFICIAL_GATE_1C["target_count"],
    }
    for key, expected_value in expected_registration_contract.items():
        require(
            registration.get(key) == expected_value,
            f"Gate 1c configuration contract drift: {key}",
        )
    require(
        registration.get("upstream_model_metadata_provided") is False,
        "Gate 1c cannot claim upstream model metadata",
    )
    require(
        registration.get("raw_receipt_schema") == OPAQUE_ESM2_RECEIPT_SCHEMA,
        "Gate 1c configuration does not require the v2 receipt schema",
    )
    require(
        registration.get("raw_receipt") == OFFICIAL_GATE_1C["receipt_path"],
        "Gate 1c configuration does not require the registered v2 receipt path",
    )
    require(
        registration.get("raw_receipt_size_bytes")
        == OFFICIAL_GATE_1C["receipt_size_bytes"],
        "Gate 1c configuration does not bind the v2 receipt size",
    )
    require(
        registration.get("raw_receipt_sha256")
        == OFFICIAL_GATE_1C["receipt_sha256"],
        "Gate 1c configuration does not bind the v2 receipt SHA-256",
    )
    config_bindings = {
        "source_url": "source_url",
        "source_object_provider": ("source_object", "provider"),
        "source_object_generation": ("source_object", "generation"),
        "source_object_last_modified": ("source_object", "last_modified"),
        "source_object_size_bytes": ("source_object", "size_bytes"),
        "source_object_crc32c_base64": ("source_object", "crc32c_base64"),
        "source_object_crc32c_hex": ("source_object", "crc32c_hex"),
        "source_object_component_count": ("source_object", "component_count"),
        "source_object_metadata_scope": ("source_object", "metadata_scope"),
        "source_url_claimed_permanently_immutable": (
            "source_object",
            "source_url_claimed_permanently_immutable",
        ),
        "source_archive_size_bytes": "archive_size_bytes",
        "source_archive_sha256": "archive_sha256",
        "archive_member": "member_name",
        "archive_member_size_bytes": "member_size_bytes",
        "archive_member_sha256": "member_sha256",
        "archive_member_crc32": "member_crc32",
        "artifact_size_bytes": "member_size_bytes",
        "artifact_sha256": "member_sha256",
        "feature_count": "feature_count",
        "embedding_dimension": "embedding_dimension",
        "tensor_dtype": "tensor_dtype",
        "tensor_layout": "tensor_layout",
        "canonical_key_list_sha256": "canonical_key_list_sha256",
        "canonical_map_sha256": "canonical_map_sha256",
        "target_manifest_size_bytes": "target_manifest_size_bytes",
        "target_manifest_sha256": "target_manifest_sha256",
        "target_count": "target_count",
        "ordered_target_list_sha256": "ordered_target_list_sha256",
        "ordered_target_feature_subset_sha256": (
            "ordered_target_feature_subset_sha256"
        ),
    }
    for config_key, official_key in config_bindings.items():
        if isinstance(official_key, tuple):
            expected_value = OFFICIAL_GATE_1C[official_key[0]][official_key[1]]
        else:
            expected_value = OFFICIAL_GATE_1C[official_key]
        require(
            registration.get(config_key) == expected_value,
            f"Gate 1c configuration drift: {config_key}",
        )
    require(
        Path(str(registration.get("source_archive"))).name
        == OFFICIAL_GATE_1C["archive_filename"],
        "Gate 1c source-archive filename drift",
    )
    require(
        Path(str(registration.get("artifact"))).name
        == OFFICIAL_GATE_1C["feature_filename"],
        "Gate 1c feature-artifact filename drift",
    )

    archive = receipt.get("archive")
    require(isinstance(archive, dict), "Gate 1c receipt lacks archive")
    require(archive.get("source_url") == OFFICIAL_GATE_1C["source_url"], "Official source URL drift")
    require(
        archive.get("source_object") == OFFICIAL_GATE_1C["source_object"],
        "Official GCS source-object identity drift",
    )
    require(
        archive.get("local_filename") == OFFICIAL_GATE_1C["archive_filename"],
        "Official archive filename drift",
    )
    require(
        archive.get("size_bytes") == OFFICIAL_GATE_1C["archive_size_bytes"],
        "Official archive size drift",
    )
    require(
        archive.get("sha256") == OFFICIAL_GATE_1C["archive_sha256"],
        "Official archive SHA-256 drift",
    )
    member = archive.get("member")
    require(isinstance(member, dict), "Gate 1c receipt lacks archive member")
    require(
        member
        == {
            "name": OFFICIAL_GATE_1C["member_name"],
            "size_bytes": OFFICIAL_GATE_1C["member_size_bytes"],
            "sha256": OFFICIAL_GATE_1C["member_sha256"],
            "zip_crc32": OFFICIAL_GATE_1C["member_crc32"],
            "encrypted": False,
            "symbolic_link": False,
        },
        "Official archive-member identity drift",
    )

    artifact = receipt.get("feature_artifact")
    require(isinstance(artifact, dict), "Gate 1c receipt lacks feature artifact")
    for key in ("sha256", "expected_sha256", "post_load_sha256"):
        require(
            artifact.get(key) == OFFICIAL_GATE_1C["member_sha256"],
            f"Official feature {key} drift",
        )
    require(
        artifact.get("local_filename") == OFFICIAL_GATE_1C["feature_filename"],
        "Official feature filename drift",
    )
    require(
        artifact.get("size_bytes") == OFFICIAL_GATE_1C["member_size_bytes"],
        "Official feature size drift",
    )
    for key in (
        "matches_registered_archive_member",
        "post_load_sha256_matches_preload",
        "restricted_weights_only_load",
        "sha256_matches_expected_when_provided",
    ):
        require(artifact.get(key) is True, f"Gate 1c artifact check is not true: {key}")
    require(artifact.get("map_location") == "cpu", "Gate 1c map_location drift")
    require(artifact.get("explicit_safe_globals") == [], "Gate 1c safe-global drift")
    structure = artifact.get("structure")
    require(isinstance(structure, dict), "Gate 1c receipt lacks feature structure")
    structure_bindings = {
        "feature_count": "feature_count",
        "embedding_dimension": "embedding_dimension",
        "tensor_dtype": "tensor_dtype",
        "tensor_layout": "tensor_layout",
        "canonical_key_list_sha256": "canonical_key_list_sha256",
        "canonical_map_sha256": "canonical_map_sha256",
    }
    for receipt_key, registered_key in structure_bindings.items():
        require(
            structure.get(receipt_key) == OFFICIAL_GATE_1C[registered_key],
            f"Official feature-structure drift: {receipt_key}",
        )
    require(structure.get("tensor_device") == "cpu", "Gate 1c tensor-device drift")
    require(structure.get("all_values_finite") is True, "Gate 1c features are not finite")
    require(structure.get("all_vectors_nonzero") is True, "Gate 1c features include zero vectors")

    target_manifest = receipt.get("target_manifest")
    require(isinstance(target_manifest, dict), "Gate 1c receipt lacks target manifest")
    require(
        target_manifest
        == {
            "local_filename": OFFICIAL_GATE_1C["target_manifest_filename"],
            "size_bytes": OFFICIAL_GATE_1C["target_manifest_size_bytes"],
            "sha256": OFFICIAL_GATE_1C["target_manifest_sha256"],
            "target_count": OFFICIAL_GATE_1C["target_count"],
            "target_column": "target_gene",
            "targets_are_unique": True,
            "ordered_target_list_sha256": OFFICIAL_GATE_1C[
                "ordered_target_list_sha256"
            ],
        },
        "Official ordered target-manifest identity drift",
    )
    coverage = receipt.get("vcc_target_coverage")
    require(isinstance(coverage, dict), "Gate 1c receipt lacks VCC coverage")
    require(
        coverage
        == {
            "covered_targets": OFFICIAL_GATE_1C["target_count"],
            "requested_targets": OFFICIAL_GATE_1C["target_count"],
            "missing_targets": 0,
            "ordered_target_feature_subset_sha256": OFFICIAL_GATE_1C[
                "ordered_target_feature_subset_sha256"
            ],
        },
        "Official ordered VCC target-feature coverage drift",
    )
    upstream = receipt.get("upstream_provenance")
    require(isinstance(upstream, dict), "Gate 1c receipt lacks upstream disclaimer")
    for key in (
        "model_id",
        "model_revision",
        "weights_sha256",
        "protein_sequence_source",
        "gene_to_protein_mapping",
        "pooling_rule",
    ):
        require(upstream.get(key) == "UNKNOWN_NOT_PROVIDED", f"Gate 1b field must remain unknown: {key}")
    require(upstream.get("inferred_from_embedding_dimension") is False, "Gate 1b metadata was inferred")
    checks = receipt.get("checks")
    require(isinstance(checks, dict) and checks, "Gate 1c receipt lacks checks")
    require(all(value is True for value in checks.values()), "A Gate 1c receipt check did not pass")
    return artifact


def authenticate_upstream_provenance_artifacts(
    repository: Path,
    upstream: Mapping[str, Any],
    config_provenance: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Hash the actual weights, sequence, and mapping files from safe descriptors."""

    specifications = {
        "weights": ("weights_artifact", "weights_artifact"),
        "protein_sequences": ("protein_sequence_artifact", "protein_sequence_source"),
        "gene_to_protein_mapping": (
            "gene_to_protein_mapping_artifact",
            "gene_to_protein_mapping",
        ),
    }
    authenticated: dict[str, dict[str, Any]] = {}
    for result_key, (receipt_key, config_key) in specifications.items():
        registered = upstream[receipt_key]
        path = resolve_registered_path(repository, config_provenance.get(config_key), result_key)
        handle, opened = open_regular_no_follow(path, f"ESM2 {result_key}")
        with handle:
            digest = sha256_stream(handle)
            after = os.fstat(handle.fileno())
            require(_same_identity(opened, after), f"ESM2 {result_key} changed while hashing")
        require(Path(path).name == registered["filename"], f"ESM2 {result_key} filename changed")
        require(opened.st_size == registered["size_bytes"], f"ESM2 {result_key} size changed")
        require(digest == registered["sha256"], f"ESM2 {result_key} SHA-256 changed")
        authenticated[result_key] = {
            "filename": Path(path).name,
            "size_bytes": opened.st_size,
            "sha256": digest,
        }
    require(
        authenticated["weights"]["sha256"] == config_provenance.get("weights_sha256"),
        "Actual ESM2 weights do not match config",
    )
    require(
        authenticated["protein_sequences"]["sha256"]
        == config_provenance.get("protein_sequence_source_sha256"),
        "Actual protein sequences do not match config",
    )
    require(
        authenticated["gene_to_protein_mapping"]["sha256"]
        == config_provenance.get("gene_to_protein_mapping_sha256"),
        "Actual gene-to-protein mapping does not match config",
    )
    return authenticated


def load_authenticated_features(
    path: Path,
    receipt_artifact: Mapping[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    handle, opened = open_regular_no_follow(path, "ESM2 feature artifact")
    with handle:
        digest = sha256_stream(handle)
        require(opened.st_size == receipt_artifact.get("size_bytes"), "ESM2 artifact size changed")
        require(digest == receipt_artifact.get("sha256"), "ESM2 artifact SHA-256 changed")
        handle.seek(0)
        try:
            payload = torch.load(handle, map_location="cpu", weights_only=True)
        except Exception as error:
            raise JurkatManifestError(f"Restricted ESM2 loading failed: {error}") from error
        require(type(payload) is dict, "ESM2 payload root must be a plain dict")
        features = payload
        expected_dimension = int(receipt_artifact["structure"]["embedding_dimension"])
        for key, value in features.items():
            require(isinstance(key, str) and key, "ESM2 keys must be non-empty strings")
            require(type(value) is torch.Tensor, f"ESM2 value is not a plain tensor: {key}")
            require(value.device.type == "cpu" and value.dtype == torch.float32, f"Bad ESM2 tensor: {key}")
            require(tuple(value.shape) == (expected_dimension,), f"Bad ESM2 shape: {key}")
            require(bool(torch.isfinite(value).all()) and bool(torch.count_nonzero(value)), f"Bad ESM2 values: {key}")
        key_digest = canonical_feature_key_sha256(list(features))
        map_digest = canonical_feature_map_sha256(features)
        structure = receipt_artifact["structure"]
        require(len(features) == structure.get("feature_count"), "ESM2 feature-count drift")
        require(key_digest == structure.get("canonical_key_list_sha256"), "ESM2 key digest changed")
        require(map_digest == structure.get("canonical_map_sha256"), "ESM2 map digest changed")
        handle.seek(0)
        require(sha256_stream(handle) == digest, "ESM2 artifact changed during loading")
        after = os.fstat(handle.fileno())
        require(_same_identity(opened, after), "ESM2 artifact identity changed")
    return features, {
        "filename": Path(path).name,
        "size_bytes": opened.st_size,
        "sha256": digest,
        "feature_count": len(features),
        "embedding_dimension": expected_dimension,
        "canonical_key_list_sha256": key_digest,
        "canonical_map_sha256": map_digest,
    }


def choose_cluster_subset(cluster_counts: Mapping[int, int], desired: int) -> tuple[int, ...]:
    """Choose complete clusters nearest the requested evaluable-target count."""

    require(desired > 0 and cluster_counts, "Invalid held-cluster request")
    possibilities: dict[int, tuple[int, ...]] = {0: ()}
    for cluster, count in sorted(cluster_counts.items()):
        require(count >= 0, "Negative cluster count")
        additions = {
            total + count: selected + (int(cluster),)
            for total, selected in possibilities.items()
            if count > 0
        }
        for total, selected in additions.items():
            prior = possibilities.get(total)
            if prior is None or selected < prior:
                possibilities[total] = selected
    candidates = [(total, selected) for total, selected in possibilities.items() if total > 0]
    require(candidates, "No cluster contains an eligible non-harm target")
    return min(
        candidates,
        key=lambda item: (
            abs(item[0] - desired),
            item[0] > desired,
            len(item[1]),
            item[1],
        ),
    )[1]


def _descriptor_matches_registration(descriptor: Mapping[str, Any], registration: Mapping[str, Any]) -> None:
    require(descriptor["size_bytes"] == registration.get("raw_size_bytes"), "Raw Jurkat size drift")
    require(descriptor["sha256"] == registration.get("raw_sha256"), "Raw Jurkat hash drift")
    require(descriptor["shape"] == registration.get("raw_shape"), "Raw Jurkat shape drift")
    require(
        descriptor["treated_target_count"] == registration.get("raw_treated_target_count"),
        "Raw Jurkat target-count drift",
    )
    require(descriptor["control_cells"] == registration.get("raw_control_cells"), "Raw Jurkat control-count drift")


def load_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    raw, descriptor = read_regular_bytes(path, "V7 configuration")
    try:
        config = tomllib.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise JurkatManifestError(f"Invalid V7 TOML: {error}") from error
    require(config.get("experiment", {}).get("schema") == "vcc-public-v7-scdfm-experiment-v1", "Bad V7 schema")
    registration = config.get("search", {}).get("jurkat_non_harm_registration")
    require(isinstance(registration, dict), "V7 config lacks Jurkat split registration")
    require(registration.get("schema") == REGISTRATION_SCHEMA, "Bad Jurkat registration schema")
    return config, descriptor


def registered_manifest_path(config_path: Path) -> Path:
    """Return the only manifest path authorized by the V7 configuration."""

    config_path = absolute_path(config_path)
    config, _ = load_config(config_path)
    return resolve_registered_path(
        config_path.parents[2],
        config.get("search", {}).get("secondary_non_harm_manifest"),
        "registered Jurkat manifest",
    )


def registered_training_preflight_path(config_path: Path) -> Path:
    """Return the only training-preflight path authorized by the config."""

    config_path = absolute_path(config_path)
    config, _ = load_config(config_path)
    registration = config["search"]["jurkat_non_harm_registration"]
    return resolve_registered_path(
        config_path.parents[2],
        registration.get("training_preflight_receipt"),
        "registered training preflight",
    )


def construct_manifest(
    config_path: Path,
    *,
    _test_only_lineage_bypass: object | None = None,
) -> dict[str, Any]:
    """Construct a deterministic final manifest or fail before publication."""

    test_only_lineage_bypass = (
        _test_only_lineage_bypass is _TEST_ONLY_SYNTHETIC_LINEAGE_BYPASS
    )
    config_path = absolute_path(config_path)
    repository = config_path.parents[2]
    config, config_descriptor = load_config(config_path)
    registration = config["search"]["jurkat_non_harm_registration"]
    provenance = config["target_conditioning_provenance"]

    if test_only_lineage_bypass:
        receipt_value = provenance.get("embedding_receipt")
        receipt_hash = provenance.get("embedding_receipt_sha256")
        require(
            isinstance(receipt_value, str) and receipt_value not in UNKNOWN_VALUES,
            "Synthetic test receipt is unresolved",
        )
        opaque: Mapping[str, Any] = {}
    else:
        opaque_value = config.get("target_conditioning_artifact_authentication")
        require(
            isinstance(opaque_value, dict),
            "No authenticated official opaque ESM2 artifact is registered",
        )
        opaque = opaque_value
        receipt_value = opaque.get("raw_receipt")
        receipt_hash = opaque.get("raw_receipt_sha256")
        require(config.get("experiment", {}).get("training_ready") is False, "V7 training must remain closed")
        require(provenance.get("provenance_complete") is False, "Gate 1b provenance must remain incomplete")
        require(provenance.get("gate_1b_allowed") is False, "Gate 1b must remain closed")
        require(
            provenance.get("feature_derivation_replay_implemented") is False,
            "Gate 1b derivation replay must remain unavailable",
        )
    receipt_path = resolve_registered_path(repository, receipt_value, "ESM2 receipt")
    receipt_raw, receipt_descriptor = read_regular_bytes(receipt_path, "ESM2 provenance receipt")
    require(
        receipt_descriptor["sha256"] == receipt_hash,
        "Registered ESM2 receipt SHA-256 changed",
    )
    if not test_only_lineage_bypass:
        require(
            receipt_descriptor["size_bytes"] == opaque.get("raw_receipt_size_bytes"),
            "Registered Gate 1c receipt size changed",
        )
    receipt = strict_json_loads(receipt_raw, "ESM2 receipt")
    if test_only_lineage_bypass:
        receipt_artifact, upstream_provenance = validate_complete_esm2_receipt(
            receipt,
            receipt_descriptor,
            provenance,
        )
        upstream_artifacts = authenticate_upstream_provenance_artifacts(
            repository,
            upstream_provenance,
            provenance,
        )
        conditioning_gate = {
            "gate": "private_synthetic_test_seam",
            "gate_1b_allowed": False,
            "gate_1c_allowed": False,
            "production_training_ready": False,
        }
    else:
        receipt_artifact = validate_official_gate_1c_receipt(receipt, opaque)
        upstream_artifacts = {}
        conditioning_gate = {
            "gate": "gate_1c_official_opaque_artifact",
            "official_opaque_artifact": True,
            "gate_1b_allowed": False,
            "gate_1c_allowed": True,
            "upstream_model_provenance_complete": False,
            "production_training_ready": False,
        }
    require(
        int(receipt_artifact["structure"]["embedding_dimension"])
        == int(OFFICIAL_GATE_1C["embedding_dimension"] if not test_only_lineage_bypass else provenance.get("embedding_dimension", -1)),
        "Config does not bind the ESM2 embedding dimension",
    )

    feature_path = resolve_registered_path(repository, registration.get("esm2_artifact"), "ESM2 artifact")
    features, feature_descriptor = load_authenticated_features(feature_path, receipt_artifact)

    raw_path = resolve_registered_path(repository, registration.get("raw_jurkat"), "raw Jurkat")
    raw, raw_descriptor = read_jurkat_metadata(raw_path)
    _descriptor_matches_registration(raw_descriptor, registration)

    historical_path = resolve_registered_path(repository, registration.get("historical_panel"), "historical panel")
    historical_targets, historical_descriptor = read_target_csv(
        historical_path, str(registration.get("historical_panel_sha256"))
    )
    support_values = registration.get("state_support_sources")
    require(isinstance(support_values, list) and support_values, "No STATE support sources registered")
    support_sources: list[tuple[Path, Mapping[str, Any]]] = []
    for value in support_values:
        require(isinstance(value, dict), "STATE support registration must be a table")
        support_sources.append(
            (
                resolve_registered_path(repository, value.get("path"), "STATE support source"),
                value,
            )
        )
    support_targets, support_descriptors = read_state_support_targets(support_sources)
    require(
        len(support_targets) == registration.get("expected_state_support_target_union"),
        "STATE support target-union drift",
    )
    require(
        canonical_sha256(sorted(support_targets))
        == registration.get("expected_state_support_target_union_sha256"),
        "STATE support target-union SHA-256 drift",
    )

    historical_set = set(historical_targets)
    forbidden_evaluation = historical_set | support_targets
    raw_targets = raw["treated_targets"]
    universe = sorted(set(raw_targets) & set(features))
    require(universe, "No Jurkat targets have authenticated embeddings")
    dimension = feature_descriptor["embedding_dimension"]
    matrix = np.stack(
        [features[target].detach().cpu().numpy() for target in universe]
    ).astype(np.float32)
    require(matrix.shape == (len(universe), dimension), "Bad clustering matrix")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    require(np.all(np.isfinite(norms)) and np.all(norms > 0), "Bad ESM2 norms")
    matrix /= norms

    cluster_count = int(registration["cluster_count"])
    seed = int(registration["seed"])
    n_init = int(registration["kmeans_n_init"])
    max_iter = int(registration["kmeans_max_iter"])
    tolerance = float(registration["kmeans_tol"])
    require(1 < cluster_count < len(universe), "Invalid cluster_count")
    require(n_init > 0, "Invalid kmeans_n_init")
    require(max_iter > 0 and tolerance > 0, "Invalid KMeans convergence registration")
    require(registration.get("kmeans_algorithm") == "lloyd", "Only lloyd is registered")
    require(registration.get("kmeans_init") == "k-means++", "Only k-means++ is registered")
    require(registration.get("thread_policy") == "threadpoolctl-limit-1", "Bad thread policy")
    require(registration.get("numpy_version") == np.__version__, "NumPy version drift")
    require(
        registration.get("scikit_learn_version") == sklearn.__version__,
        "scikit-learn version drift",
    )

    def cluster_once() -> np.ndarray:
        with threadpool_limits(limits=1):
            return KMeans(
                n_clusters=cluster_count,
                init="k-means++",
                random_state=seed,
                n_init=n_init,
                max_iter=max_iter,
                tol=tolerance,
                algorithm="lloyd",
                copy_x=True,
            ).fit_predict(matrix.copy()).astype(np.int64)

    labels = cluster_once()
    repeated_labels = cluster_once()
    require(
        np.array_equal(labels, repeated_labels),
        "Repeated single-thread KMeans assignments are not deterministic",
    )
    require(
        len(np.unique(labels)) == cluster_count,
        "An official target-feature cluster is empty",
    )
    cluster_by_target = dict(zip(universe, labels.tolist(), strict=True))

    minimum_cells = int(registration["minimum_treated_cells"])
    desired_targets = int(registration["desired_held_targets"])
    gene_axis = set(raw["gene_names"])
    eligible = {
        target
        for target in universe
        if raw["target_counts"][target] >= minimum_cells
        and target in gene_axis
        and target not in forbidden_evaluation
    }
    cluster_eligible_counts = {
        cluster: sum(cluster_by_target[target] == cluster for target in eligible)
        for cluster in range(cluster_count)
    }
    held_clusters = choose_cluster_subset(cluster_eligible_counts, desired_targets)
    held_cluster_set = set(held_clusters)
    global_exclusion_targets = sorted(
        target for target in universe if cluster_by_target[target] in held_cluster_set
    )
    scored_targets = sorted(eligible & set(global_exclusion_targets))
    require(len(scored_targets) == desired_targets, "Registered split does not hit held-target goal")
    require(not (set(scored_targets) & historical_set), "Historical target entered non-harm scoring")
    require(not (set(scored_targets) & support_targets), "STATE support target entered non-harm scoring")

    target_rows = [
        {
            "target_gene": target,
            "esm2_cluster": int(cluster_by_target[target]),
            "treated_cells": int(raw["target_counts"][target]),
            "embedding_float32_sha256": hashlib.sha256(_tensor_bytes(features[target])).hexdigest(),
        }
        for target in scored_targets
    ]
    assignment_rows = [
        {"target_gene": target, "esm2_cluster": int(cluster_by_target[target])}
        for target in universe
    ]
    scored_target_set = set(scored_targets)
    global_exclusion_target_set = set(global_exclusion_targets)
    scored_cells = [
        barcode
        for barcode, target in zip(raw["cell_barcodes"], raw["labels"], strict=True)
        if target in scored_target_set
    ]
    excluded_cells = [
        barcode
        for barcode, target in zip(raw["cell_barcodes"], raw["labels"], strict=True)
        if target in global_exclusion_target_set
    ]
    control_cells = [
        barcode
        for barcode, target in zip(raw["cell_barcodes"], raw["labels"], strict=True)
        if target == CONTROL_LABEL
    ]

    global_scopes = registration.get("global_exclusion_scopes")
    expected_scopes = [
        "model_fit",
        "checkpoint_selection",
        "normalization_fit",
        "feature_selection",
        "graph_construction",
        "decoder_fit",
        "hyperparameter_selection",
    ]
    require(global_scopes == expected_scopes, "Global exclusion scopes changed")
    if test_only_lineage_bypass:
        manifest_status = "sealed_test_only"
        lineage_firewall = {
            "gate": "private_synthetic_test_seam",
            "gate_1b_upstream_provenance_complete": False,
            "gate_1b_allowed": False,
            "gate_1c_allowed": False,
            "test_only_synthetic_lineage_bypass_used": True,
            "production_training_ready": False,
            "production_manifest_publication_allowed": False,
        }
    else:
        manifest_status = "sealed"
        lineage_firewall = {
            "gate": "gate_1c_official_opaque_artifact",
            "gate_1b_upstream_provenance_complete": False,
            "gate_1b_allowed": False,
            "gate_1c_official_opaque_artifact": True,
            "gate_1c_allowed": True,
            "test_only_synthetic_lineage_bypass_used": False,
            "production_training_ready": False,
            "production_manifest_publication_allowed": True,
        }
    return {
        "schema": MANIFEST_SCHEMA,
        "status": manifest_status,
        "lineage_firewall": lineage_firewall,
        "registration": {
            "configuration": config_descriptor,
            "seed": seed,
            "cluster_algorithm": "sklearn.cluster.KMeans/lloyd",
            "cluster_count": cluster_count,
            "kmeans_n_init": n_init,
            "kmeans_init": "k-means++",
            "kmeans_max_iter": max_iter,
            "kmeans_tol": tolerance,
            "kmeans_copy_x": True,
            "thread_policy": "threadpoolctl-limit-1",
            "repeated_build_assignment_identity_required": True,
            "embedding_normalization": "row-l2-float32",
            "minimum_treated_cells": minimum_cells,
            "desired_held_targets": desired_targets,
            "scikit_learn_version": sklearn.__version__,
            "numpy_version": np.__version__,
            "torch_version": torch.__version__,
        },
        "data_firewall": {
            "effect_values_accessed_for_selection": False,
            "expression_matrix_values_accessed": False,
            "allowed_selection_inputs": [
                "raw target labels and treated-cell counts",
                "raw cell barcodes, batch/design labels, and matrix shape metadata",
                "raw gene-axis membership",
                "historical target identifiers",
                "ARC/STATE support target identifiers",
                "authenticated target embeddings",
            ],
            "historical_targets_forbidden_from_scoring": True,
            "state_support_targets_forbidden_from_scoring": True,
            "held_cluster_members_globally_excluded": True,
            "global_exclusion_scopes": global_scopes,
            "controls_may_condition_the_in_distribution_model": True,
            "controls_are_not_treated_supervision": True,
            "downstream_training_receipt_must_bind_manifest_sha256": True,
        },
        "inputs": {
            "raw_jurkat": raw_descriptor,
            "target_conditioning_receipt": receipt_descriptor,
            "target_embeddings": feature_descriptor,
            "target_conditioning_gate": conditioning_gate,
            "target_conditioning_upstream_artifacts": upstream_artifacts,
            "historical_panel": historical_descriptor,
            "state_support": {
                "files": support_descriptors,
                "union_target_count": len(support_targets),
                "union_target_list_sha256": canonical_sha256(sorted(support_targets)),
            },
        },
        "partition": {
            "clustering_universe_target_count": len(universe),
            "clustering_universe_target_list_sha256": canonical_sha256(universe),
            "clustering_universe_embedding_sha256": canonical_sha256(
                [
                    [target, hashlib.sha256(_tensor_bytes(features[target])).hexdigest()]
                    for target in universe
                ]
            ),
            "cluster_assignment_sha256": canonical_sha256(assignment_rows),
            "repeated_cluster_assignment_sha256": canonical_sha256(
                [
                    {"target_gene": target, "esm2_cluster": int(label)}
                    for target, label in zip(universe, repeated_labels.tolist(), strict=True)
                ]
            ),
            "cluster_assignments": assignment_rows,
            "held_cluster_ids": list(held_clusters),
            "held_cluster_ids_sha256": canonical_sha256(list(held_clusters)),
            "scored_held_targets": target_rows,
            "scored_held_target_count": len(scored_targets),
            "scored_held_target_list_sha256": canonical_sha256(scored_targets),
            "scored_held_cell_count": len(scored_cells),
            "scored_held_cell_barcode_sha256": canonical_sha256(scored_cells),
            "global_exclusion_targets": global_exclusion_targets,
            "global_exclusion_target_count": len(global_exclusion_targets),
            "global_exclusion_target_list_sha256": canonical_sha256(global_exclusion_targets),
            "global_exclusion_cell_count": len(excluded_cells),
            "global_exclusion_cell_barcode_sha256": canonical_sha256(excluded_cells),
            "historical_forbidden_target_count": len(historical_set),
            "historical_forbidden_target_list_sha256": canonical_sha256(sorted(historical_set)),
            "state_support_forbidden_target_count": len(support_targets),
            "state_support_forbidden_target_list_sha256": canonical_sha256(sorted(support_targets)),
            "control_cell_count": len(control_cells),
            "control_cell_barcode_sha256": canonical_sha256(control_cells),
        },
        "assertions": {
            "all_scored_targets_are_complete_cluster_members": True,
            "all_members_of_held_clusters_are_in_global_exclusion_targets": True,
            "every_held_target_excludes_all_guides_batches_and_cells": True,
            "historical_and_state_support_targets_are_absent_from_scoring": True,
            "no_held_target_or_cluster_member_may_enter_any_fit_transform_or_graph": True,
            "gate_1b_upstream_provenance_remains_incomplete": True,
            "gate_1b_remains_closed": True,
            "official_gate_1c_identity_and_structure_are_sufficient_for_split": (
                not test_only_lineage_bypass
            ),
            "self_asserted_upstream_assets_cannot_open_production_gate_1b": True,
            "training_remains_closed_until_portable_preflight_consumer_exists": True,
            "repeated_single_thread_cluster_builds_are_identical": True,
        },
    }


def _open_or_create_directory_no_follow(path: Path) -> int:
    """Return an open directory descriptor after a symlink-safe traversal."""

    absolute = Path(os.path.abspath(os.fspath(Path(path).expanduser())))
    parts = absolute.parts
    require(len(parts) >= 1 and parts[0] == os.sep, "Invalid output directory")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(os.sep, flags)
    try:
        for component in parts[1:]:
            try:
                next_descriptor = os.open(component, flags, dir_fd=descriptor)
            except FileNotFoundError:
                os.mkdir(component, mode=0o755, dir_fd=descriptor)
                next_descriptor = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
    except Exception:
        os.close(descriptor)
        raise
    return descriptor


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish a deterministic manifest without replacing any existing path."""

    path = Path(os.path.abspath(os.fspath(Path(path).expanduser())))
    require(path.name not in {"", ".", ".."}, "Invalid sealed-artifact filename")
    parent_descriptor = _open_or_create_directory_no_follow(path.parent)
    temporary_name = f".{path.name}.{os.getpid()}.{secrets.token_hex(12)}.tmp"
    temporary_created = False
    try:
        try:
            os.stat(path.name, dir_fd=parent_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(f"Refusing to overwrite sealed artifact: {path}")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        file_descriptor = os.open(
            temporary_name,
            flags,
            0o600,
            dir_fd=parent_descriptor,
        )
        temporary_created = True
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(
                temporary_name,
                path.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileExistsError as error:
            raise FileExistsError(f"Refusing to overwrite sealed artifact: {path}") from error
        os.fsync(parent_descriptor)
    finally:
        if temporary_created:
            try:
                os.unlink(temporary_name, dir_fd=parent_descriptor)
            except FileNotFoundError:
                pass
        os.close(parent_descriptor)


def authenticate_manifest(
    config_path: Path,
    manifest_path: Path,
    *,
    _test_only_lineage_bypass: object | None = None,
) -> dict[str, Any]:
    """Reconstruct and byte-compare a sealed manifest from authenticated inputs."""

    config_path = absolute_path(config_path)
    manifest_path = absolute_path(manifest_path)
    require(
        manifest_path == registered_manifest_path(config_path),
        "Manifest path is not the path registered by the V7 configuration",
    )
    raw, descriptor = read_regular_bytes(manifest_path, "sealed Jurkat manifest")
    observed = strict_json_loads(raw, "sealed manifest")
    expected = construct_manifest(
        config_path,
        _test_only_lineage_bypass=_test_only_lineage_bypass,
    )
    require(observed == expected, "Sealed Jurkat manifest does not reconstruct exactly")
    require(
        raw == formatted_json_bytes(expected),
        "Sealed Jurkat manifest is not the exact canonical on-disk encoding",
    )
    return {
        "schema": "vcc-scdfm-jurkat-non-harm-authentication-v1",
        "status": (
            "passed_test_only"
            if _test_only_lineage_bypass is _TEST_ONLY_SYNTHETIC_LINEAGE_BYPASS
            else "passed"
        ),
        "manifest": descriptor,
        "checks": {
            "manifest_reconstructs_exactly": True,
            "manifest_bytes_match_canonical_reconstruction": True,
            "metadata_only_selection": True,
            "whole_esm2_clusters_held": True,
            "historical_and_state_support_targets_excluded_from_scoring": True,
            "global_training_preprocessing_and_graph_exclusion_registered": True,
            "gate_1b_upstream_provenance_complete": False,
            "gate_1b_allowed": False,
            "gate_1c_official_opaque_artifact_authenticated": (
                _test_only_lineage_bypass is not _TEST_ONLY_SYNTHETIC_LINEAGE_BYPASS
            ),
            "production_training_ready": False,
        },
    }


def enforce_training_labels_exclude_manifest(
    labels: Sequence[str],
    excluded_targets: set[str],
    label: str,
) -> tuple[list[int], list[int]]:
    """Return allowed/excluded row indices or reject a malformed label axis."""

    require(all(isinstance(value, str) and value for value in labels), f"Bad labels: {label}")
    allowed = [index for index, value in enumerate(labels) if value not in excluded_targets]
    excluded = [index for index, value in enumerate(labels) if value in excluded_targets]
    require(
        not ({labels[index] for index in allowed} & excluded_targets),
        f"Held targets remain in allowed training rows: {label}",
    )
    return allowed, excluded


def assert_loaded_training_batch_is_clean(
    labels: Sequence[str],
    excluded_targets: set[str],
    label: str,
) -> None:
    """Fail if a downstream loaded batch contains any sealed target label."""

    contamination = sorted(set(labels) & excluded_targets)
    require(
        not contamination,
        f"Loaded training batch contains sealed target labels ({label}): {contamination[:8]}",
    )


def construct_training_preflight(
    config_path: Path,
    manifest_path: Path,
    *,
    _runtime_allowed_rows: dict[str, tuple[int, ...]] | None = None,
    _test_only_lineage_bypass: object | None = None,
) -> dict[str, Any]:
    """Create the mandatory row-filter contract consumed before any V7 fitting."""

    test_only_lineage_bypass = (
        _test_only_lineage_bypass is _TEST_ONLY_SYNTHETIC_LINEAGE_BYPASS
    )
    config_path = absolute_path(config_path)
    manifest_path = absolute_path(manifest_path)
    authentication = authenticate_manifest(
        config_path,
        manifest_path,
        _test_only_lineage_bypass=_test_only_lineage_bypass,
    )
    manifest_raw, manifest_descriptor = read_regular_bytes(manifest_path, "sealed Jurkat manifest")
    manifest = strict_json_loads(manifest_raw, "sealed manifest")
    require(
        manifest_descriptor == authentication["manifest"],
        "Manifest identity changed after authentication",
    )
    config, config_descriptor = load_config(config_path)
    registration = config["search"]["jurkat_non_harm_registration"]
    require(registration.get("training_preflight_required") is True, "Training preflight is not required")
    require(
        registration.get("training_preflight_schema") == TRAINING_PREFLIGHT_SCHEMA,
        "Bad training-preflight schema registration",
    )
    excluded_targets = set(manifest["partition"]["global_exclusion_targets"])
    require(excluded_targets, "Sealed manifest has no globally excluded targets")
    repository = config_path.parents[2]

    raw_path = resolve_registered_path(repository, registration.get("raw_jurkat"), "raw Jurkat")
    raw, raw_descriptor = read_jurkat_metadata(raw_path)
    _descriptor_matches_registration(raw_descriptor, registration)
    raw_allowed, raw_excluded = enforce_training_labels_exclude_manifest(
        raw["labels"], excluded_targets, "nadig_jurkat"
    )
    if _runtime_allowed_rows is not None:
        _runtime_allowed_rows["nadig_jurkat"] = tuple(raw_allowed)
    datasets: list[dict[str, Any]] = [
        {
            "dataset": "nadig_jurkat",
            "source": raw_descriptor,
            "row_count": len(raw["labels"]),
            "allowed_row_count": len(raw_allowed),
            "excluded_row_count": len(raw_excluded),
            "allowed_source_row_indices_sha256": canonical_sha256(raw_allowed),
            "excluded_source_row_indices_sha256": canonical_sha256(raw_excluded),
            "allowed_target_list_sha256": canonical_sha256(
                sorted(set(raw["labels"]) - excluded_targets)
            ),
            "held_targets_absent_from_allowed_rows": True,
        }
    ]

    support_values = registration["state_support_sources"]
    for value in support_values:
        require(isinstance(value, dict), "STATE support registration must be a table")
        path = resolve_registered_path(repository, value.get("path"), "STATE support source")
        labels, descriptor = read_state_support_source(path, value)
        allowed, excluded = enforce_training_labels_exclude_manifest(
            labels, excluded_targets, Path(path).name
        )
        dataset_name = Path(path).name
        require(
            dataset_name != "nadig_jurkat"
            and dataset_name not in {item["dataset"] for item in datasets},
            f"Duplicate training dataset identity: {dataset_name}",
        )
        if _runtime_allowed_rows is not None:
            _runtime_allowed_rows[dataset_name] = tuple(allowed)
        datasets.append(
            {
                "dataset": dataset_name,
                "source": descriptor,
                "row_count": len(labels),
                "allowed_row_count": len(allowed),
                "excluded_row_count": len(excluded),
                "allowed_source_row_indices_sha256": canonical_sha256(allowed),
                "excluded_source_row_indices_sha256": canonical_sha256(excluded),
                "allowed_target_list_sha256": canonical_sha256(
                    sorted(set(labels) - excluded_targets)
                ),
                "held_targets_absent_from_allowed_rows": True,
            }
        )
    require(
        manifest["data_firewall"]["global_exclusion_scopes"]
        == registration["global_exclusion_scopes"],
        "Manifest and training preflight exclusion scopes differ",
    )
    return {
        "schema": TRAINING_PREFLIGHT_SCHEMA,
        "status": "passed_test_only" if test_only_lineage_bypass else "passed",
        "configuration": config_descriptor,
        "sealed_manifest": manifest_descriptor,
        "global_exclusion_target_count": len(excluded_targets),
        "global_exclusion_target_list_sha256": canonical_sha256(sorted(excluded_targets)),
        "global_exclusion_scopes": registration["global_exclusion_scopes"],
        "datasets": datasets,
        "consumer_contract": {
            "only_registered_allowed_source_row_indices_may_be_loaded": True,
            "complete_allowed_index_sequence_must_be_authorized_before_any_stage": True,
            "receipt_must_be_authenticated_before_model_fit": True,
            "receipt_must_be_authenticated_before_checkpoint_selection": True,
            "receipt_must_be_authenticated_before_normalization_or_feature_selection": True,
            "receipt_must_be_authenticated_before_graph_or_decoder_construction": True,
            "held_target_labels_in_any_allowed_batch_are_fatal": True,
            "expression_matrix_values_accessed_by_preflight": False,
            "production_training_ready": False,
            "portable_trainer_integration_implemented": False,
        },
    }


def _authenticate_training_preflight(
    config_path: Path,
    manifest_path: Path,
    receipt_path: Path,
    runtime_allowed_rows: dict[str, tuple[int, ...]] | None = None,
    test_only_lineage_bypass: object | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config_path = absolute_path(config_path)
    manifest_path = absolute_path(manifest_path)
    receipt_path = absolute_path(receipt_path)
    require(
        receipt_path == registered_training_preflight_path(config_path),
        "Training-preflight path is not registered by the V7 configuration",
    )
    raw, descriptor = read_regular_bytes(receipt_path, "sealed training preflight")
    observed = strict_json_loads(raw, "sealed training preflight")
    expected = construct_training_preflight(
        config_path,
        manifest_path,
        _runtime_allowed_rows=runtime_allowed_rows,
        _test_only_lineage_bypass=test_only_lineage_bypass,
    )
    require(observed == expected, "Training preflight does not reconstruct exactly")
    require(
        raw == formatted_json_bytes(expected),
        "Training preflight is not the exact canonical on-disk encoding",
    )
    authentication = {
        "schema": "vcc-scdfm-jurkat-training-preflight-authentication-v1",
        "status": (
            "passed_test_only"
            if test_only_lineage_bypass is _TEST_ONLY_SYNTHETIC_LINEAGE_BYPASS
            else "passed"
        ),
        "receipt": descriptor,
        "sealed_manifest": expected["sealed_manifest"],
        "checks": {
            "receipt_reconstructs_exactly": True,
            "receipt_bytes_match_canonical_reconstruction": True,
            "receipt_path_matches_configuration": True,
            "manifest_sha256_is_bound": True,
            "all_allowed_source_row_index_hashes_reconstruct": True,
            "held_targets_are_absent_from_all_allowed_rows": True,
            "production_training_ready": False,
        },
    }
    return authentication, expected


def authenticate_training_preflight(
    config_path: Path,
    manifest_path: Path,
    receipt_path: Path,
    *,
    _test_only_lineage_bypass: object | None = None,
) -> dict[str, Any]:
    """Reconstruct exact preflight bytes in the pinned offline audit env."""

    authentication, _ = _authenticate_training_preflight(
        config_path,
        manifest_path,
        receipt_path,
        test_only_lineage_bypass=_test_only_lineage_bypass,
    )
    return authentication


@dataclass(frozen=True)
class TrainingPreflightGate:
    """Test-only row authorization; this is not a portable trainer adapter."""

    manifest_sha256: str
    preflight_sha256: str
    excluded_targets: frozenset[str]
    allowed_source_row_indices: Mapping[str, tuple[int, ...]]
    protected_stages: frozenset[str]
    production_training_ready: bool

    def authorize_complete_selection(
        self,
        dataset: str,
        source_row_indices: Sequence[int],
        selected_labels: Sequence[str],
        stage: str,
    ) -> None:
        """Authorize one exact, complete row selection before a protected stage."""

        require(stage in self.protected_stages, f"Unregistered protected stage: {stage}")
        expected = self.allowed_source_row_indices.get(dataset)
        require(expected is not None, f"Dataset is absent from training preflight: {dataset}")
        observed = tuple(source_row_indices)
        require(
            all(isinstance(index, int) and not isinstance(index, bool) for index in observed),
            f"Non-integer source row index: {dataset}",
        )
        require(observed == expected, f"Training row selection differs from preflight: {dataset}")
        require(
            len(selected_labels) == len(expected),
            f"Training labels do not align to authorized rows: {dataset}",
        )
        assert_loaded_training_batch_is_clean(
            selected_labels,
            set(self.excluded_targets),
            f"{dataset}/{stage}",
        )

    def assert_batch(self, labels: Sequence[str], stage: str) -> None:
        """Reject sealed labels in every later mini-batch or graph payload."""

        require(stage in self.protected_stages, f"Unregistered protected stage: {stage}")
        assert_loaded_training_batch_is_clean(
            labels,
            set(self.excluded_targets),
            stage,
        )


def load_training_preflight_gate(
    config_path: Path,
    manifest_path: Path,
    receipt_path: Path,
    *,
    _test_only_lineage_bypass: object | None = None,
) -> TrainingPreflightGate:
    """Exercise the row gate in offline synthetic tests; never start training."""

    require(
        _test_only_lineage_bypass is _TEST_ONLY_SYNTHETIC_LINEAGE_BYPASS,
        "Portable trainer integration is not implemented; training_ready=false",
    )
    runtime_allowed_rows: dict[str, tuple[int, ...]] = {}
    authentication, expected = _authenticate_training_preflight(
        config_path,
        manifest_path,
        receipt_path,
        runtime_allowed_rows,
        test_only_lineage_bypass=_test_only_lineage_bypass,
    )
    manifest_raw, manifest_descriptor = read_regular_bytes(
        absolute_path(manifest_path),
        "sealed Jurkat manifest",
    )
    require(
        manifest_descriptor == expected["sealed_manifest"],
        "Manifest identity changed while loading the training gate",
    )
    manifest = strict_json_loads(manifest_raw, "sealed Jurkat manifest")
    excluded_targets = frozenset(manifest["partition"]["global_exclusion_targets"])
    require(excluded_targets, "Training gate has no excluded targets")
    require(
        canonical_sha256(sorted(excluded_targets))
        == expected["global_exclusion_target_list_sha256"],
        "Training gate exclusion-target hash changed",
    )
    return TrainingPreflightGate(
        manifest_sha256=manifest_descriptor["sha256"],
        preflight_sha256=authentication["receipt"]["sha256"],
        excluded_targets=excluded_targets,
        allowed_source_row_indices=MappingProxyType(dict(runtime_allowed_rows)),
        protected_stages=frozenset(expected["global_exclusion_scopes"]),
        production_training_ready=False,
    )
