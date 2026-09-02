#!/usr/bin/env python3
"""Authenticate the opaque Arc-supplied VCC ESM2 target-feature artifact.

This auditor binds the exact competition support archive, the registered ZIP
member, the extracted PyTorch artifact, and the ordered VCC target manifest.
The PyTorch payload is loaded from the same already-authenticated descriptor
with ``weights_only=True`` and is accepted only as a plain mapping from unique
gene-symbol strings to finite, nonzero float32 vectors of the registered
dimension.

Passing this audit establishes the identity and structural integrity of the
derived feature artifact only.  Arc's release does not identify the upstream
ESM2 model, immutable model revision, weights, protein-sequence source,
gene-to-protein/isoform mapping, or pooling rule.  The receipt therefore keeps
model provenance and V7 Gate 1b explicitly closed while permitting the
separate, narrowly scoped Gate 1c official-opaque-artifact training mode.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import os
import re
import secrets
import stat
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Mapping, Sequence

import torch


SCHEMA = "vcc-arc-esm2-target-features-authentication-gate1c-v2"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
CRC32_PATTERN = re.compile(r"[0-9a-f]{8}")


class TargetFeatureAuthenticationError(RuntimeError):
    """Raised when an input violates the registered artifact contract."""


@dataclass(frozen=True)
class TargetFeatureSpec:
    """Immutable identity and structure fields for one feature release."""

    source_url: str
    source_object_provider: str
    source_object_generation: str
    source_object_last_modified: str
    source_object_size_bytes: int
    source_object_crc32c_base64: str
    source_object_crc32c_hex: str
    source_object_component_count: int
    archive_filename: str
    archive_size_bytes: int
    archive_sha256: str
    archive_member: str
    member_size_bytes: int
    member_sha256: str
    member_crc32: str
    extracted_filename: str
    feature_count: int
    embedding_dimension: int
    key_list_sha256: str
    canonical_map_sha256: str
    target_manifest_filename: str
    target_manifest_size_bytes: int
    target_manifest_sha256: str
    target_count: int
    ordered_target_list_sha256: str
    target_feature_subset_sha256: str


OFFICIAL_SPEC = TargetFeatureSpec(
    source_url=(
        "https://storage.googleapis.com/vcc_data_prod/datasets/state/"
        "competition_support_set.zip"
    ),
    source_object_provider="Google Cloud Storage",
    source_object_generation="1763329752965344",
    source_object_last_modified="Sun, 16 Nov 2025 21:49:12 GMT",
    source_object_size_bytes=8_716_992_349,
    source_object_crc32c_base64="MBP7uA==",
    source_object_crc32c_hex="3013fbb8",
    source_object_component_count=32,
    archive_filename="competition_support_set.zip",
    archive_size_bytes=8_716_992_349,
    archive_sha256=(
        "f1d5fd56a2eee240e5585fac52bb75b0b830a7be6a5f6de1bf773bff4f507f7c"
    ),
    archive_member="competition_support_set/ESM2_pert_features.pt",
    member_size_bytes=410_886_729,
    member_sha256=(
        "a210e1cc7901513999b2bca3836ba9e2f203cd008be4e9a9d6412a2267de9748"
    ),
    member_crc32="85d91f50",
    extracted_filename="ESM2_pert_features.pt",
    feature_count=19_790,
    embedding_dimension=5_120,
    key_list_sha256=(
        "47f457d410373f0ef4e2743b6652c20b813660c9230acbe8fd5c80c7d1e94ef1"
    ),
    canonical_map_sha256=(
        "e41e97901c2e208e27c631f67c533f01bab2314ef1f9367190f832256866b399"
    ),
    target_manifest_filename="pert_counts.csv",
    target_manifest_size_bytes=1_906,
    target_manifest_sha256=(
        "f57edd7b912ebd718efc7ee9d0f334772513e7cc418d133ce525470e373b3276"
    ),
    target_count=300,
    ordered_target_list_sha256=(
        "f170abd94f4735ab7a16efc8a3263076d50e0e470f989d38033d17e2edf5fd13"
    ),
    target_feature_subset_sha256=(
        "2149553a0783bb3959403e62913b1ff9983e2abc2a1e99025608c0660d12d2d8"
    ),
)

DEFAULT_ARCHIVE = Path("dataset/state_support/competition_support_set.zip")
DEFAULT_FEATURES = Path("dataset/state_support/extracted/ESM2_pert_features.pt")
DEFAULT_TARGETS = Path("dataset/controls/pert_counts.csv")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TargetFeatureAuthenticationError(message)


def _validate_spec(spec: TargetFeatureSpec) -> None:
    for field_name in (
        "archive_sha256",
        "member_sha256",
        "key_list_sha256",
        "canonical_map_sha256",
        "target_manifest_sha256",
        "ordered_target_list_sha256",
        "target_feature_subset_sha256",
    ):
        value = getattr(spec, field_name)
        _require(
            isinstance(value, str) and SHA256_PATTERN.fullmatch(value) is not None,
            f"Invalid registered SHA-256 field: {field_name}",
        )
    _require(
        CRC32_PATTERN.fullmatch(spec.member_crc32) is not None,
        "Invalid registered member CRC32",
    )
    _require(
        isinstance(spec.source_object_generation, str)
        and spec.source_object_generation.isdecimal()
        and int(spec.source_object_generation) > 0,
        "Invalid registered source-object generation",
    )
    _require(
        isinstance(spec.source_object_provider, str)
        and bool(spec.source_object_provider),
        "Invalid registered source-object provider",
    )
    _require(
        isinstance(spec.source_object_last_modified, str)
        and bool(spec.source_object_last_modified),
        "Invalid registered source-object Last-Modified value",
    )
    _require(
        CRC32_PATTERN.fullmatch(spec.source_object_crc32c_hex) is not None,
        "Invalid registered source-object CRC32C hex value",
    )
    try:
        decoded_crc32c = base64.b64decode(
            spec.source_object_crc32c_base64,
            validate=True,
        )
    except (ValueError, TypeError) as error:
        raise TargetFeatureAuthenticationError(
            "Invalid registered source-object CRC32C base64 value"
        ) from error
    _require(
        len(decoded_crc32c) == 4
        and decoded_crc32c.hex() == spec.source_object_crc32c_hex,
        "Registered source-object CRC32C base64 and hex values disagree",
    )
    _require(
        spec.source_object_size_bytes == spec.archive_size_bytes,
        "Registered source-object size differs from the archive size",
    )
    for field_name in (
        "archive_size_bytes",
        "member_size_bytes",
        "feature_count",
        "embedding_dimension",
        "target_manifest_size_bytes",
        "target_count",
        "source_object_size_bytes",
        "source_object_component_count",
    ):
        value = getattr(spec, field_name)
        _require(
            isinstance(value, int) and not isinstance(value, bool) and value > 0,
            f"Invalid positive integer field: {field_name}",
        )


def _sha256_stream(handle: BinaryIO, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    while block := handle.read(chunk_size):
        digest.update(block)
    return digest.hexdigest()


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


def _open_regular_read_only(path: Path, label: str) -> tuple[BinaryIO, os.stat_result]:
    """Open a file through non-symlink directory descriptors on POSIX."""

    path = Path(os.path.abspath(os.fspath(Path(path).expanduser())))
    parts = path.parts
    _require(len(parts) >= 2 and parts[0] == os.sep, f"Invalid {label} path")
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
        descriptor = os.open(
            parts[-1],
            file_flags,
            dir_fd=directory_descriptor,
        )
    except OSError as error:
        raise TargetFeatureAuthenticationError(
            f"Unable to open {label}: {type(error).__name__}: {error}"
        ) from error
    finally:
        if directory_descriptor is not None:
            os.close(directory_descriptor)
    # Unbuffered reads ensure the post-load digest observes same-inode writes
    # instead of replaying bytes cached by a Python buffered reader.
    handle = os.fdopen(descriptor, "rb", buffering=0)
    opened = os.fstat(handle.fileno())
    if not stat.S_ISREG(opened.st_mode):
        handle.close()
        raise TargetFeatureAuthenticationError(f"{label} is not a regular file")
    return handle, opened


def _authenticate_archive(
    path: Path,
    spec: TargetFeatureSpec,
) -> dict[str, Any]:
    """Bind the registered ZIP and hash its feature member without extraction."""

    handle, opened = _open_regular_read_only(path, "support archive")
    with handle:
        _require(
            Path(path).name == spec.archive_filename,
            f"Unexpected support archive filename: {Path(path).name}",
        )
        _require(
            opened.st_size == spec.archive_size_bytes,
            f"Unexpected support archive size: {opened.st_size}; "
            f"expected {spec.archive_size_bytes}",
        )
        archive_digest = _sha256_stream(handle)
        _require(
            archive_digest == spec.archive_sha256,
            f"Unexpected support archive SHA-256: {archive_digest}; "
            f"expected {spec.archive_sha256}",
        )

        handle.seek(0)
        try:
            with zipfile.ZipFile(handle, mode="r") as archive:
                matches = [
                    info
                    for info in archive.infolist()
                    if info.filename == spec.archive_member
                ]
                _require(
                    len(matches) == 1,
                    "Support archive must contain exactly one registered feature member",
                )
                info = matches[0]
                _require(not info.is_dir(), "Registered feature member is a directory")
                _require(
                    not (info.flag_bits & 0x1),
                    "Registered feature member must not be encrypted",
                )
                unix_mode = (info.external_attr >> 16) & 0xFFFF
                _require(
                    not stat.S_ISLNK(unix_mode),
                    "Registered feature member must not be a symbolic link",
                )
                _require(
                    info.file_size == spec.member_size_bytes,
                    f"Unexpected feature member size: {info.file_size}; "
                    f"expected {spec.member_size_bytes}",
                )
                observed_crc32 = f"{info.CRC:08x}"
                _require(
                    observed_crc32 == spec.member_crc32,
                    f"Unexpected feature member CRC32: {observed_crc32}; "
                    f"expected {spec.member_crc32}",
                )
                with archive.open(info, mode="r") as member_handle:
                    member_digest = _sha256_stream(member_handle)
                _require(
                    member_digest == spec.member_sha256,
                    f"Unexpected feature member SHA-256: {member_digest}; "
                    f"expected {spec.member_sha256}",
                )
        except TargetFeatureAuthenticationError:
            raise
        except (
            OSError,
            RuntimeError,
            NotImplementedError,
            zipfile.BadZipFile,
            zlib.error,
        ) as error:
            raise TargetFeatureAuthenticationError(
                "Support archive is not a readable registered ZIP: "
                f"{type(error).__name__}: {error}"
            ) from error

        after = os.fstat(handle.fileno())
        _require(
            _same_identity(opened, after),
            "Support archive changed during authentication",
        )

    return {
        "source_url": spec.source_url,
        "source_object": {
            "provider": spec.source_object_provider,
            "generation": spec.source_object_generation,
            "last_modified": spec.source_object_last_modified,
            "size_bytes": spec.source_object_size_bytes,
            "crc32c_base64": spec.source_object_crc32c_base64,
            "crc32c_hex": spec.source_object_crc32c_hex,
            "component_count": spec.source_object_component_count,
            "metadata_scope": "observed_http_object_metadata",
            "source_url_claimed_permanently_immutable": False,
        },
        "local_filename": Path(path).name,
        "size_bytes": opened.st_size,
        "sha256": archive_digest,
        "member": {
            "name": spec.archive_member,
            "size_bytes": info.file_size,
            "sha256": member_digest,
            "zip_crc32": observed_crc32,
            "encrypted": False,
            "symbolic_link": False,
        },
    }


def _canonical_key_list_sha256(keys: Sequence[str]) -> str:
    encoded = json.dumps(
        sorted(keys),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    """Return registered little-endian float32 bytes for a CPU tensor."""

    array = tensor.detach().contiguous().numpy()
    return array.astype("<f4", copy=False).tobytes(order="C")


def load_restricted_tensor_mapping(
    path: Path,
    *,
    label: str,
    expected_sha256: str | None,
    expected_size_bytes: int | None = None,
    safe_globals: Sequence[object] = (),
    include_local_path_identity: bool = False,
) -> tuple[object, dict[str, Any]]:
    """Load a tensor mapping through a hashed, unchanged open descriptor.

    ``safe_globals`` exists only for legacy tensor dictionaries that encode
    NumPy scalar string keys.  Callers must pass a short explicit allowlist and
    must still validate the complete object graph after loading.
    """

    if expected_sha256 is not None:
        _require(
            SHA256_PATTERN.fullmatch(expected_sha256) is not None,
            f"Invalid expected SHA-256 for {label}",
        )
    if expected_size_bytes is not None:
        _require(
            isinstance(expected_size_bytes, int)
            and not isinstance(expected_size_bytes, bool)
            and expected_size_bytes > 0,
            f"Invalid expected byte size for {label}",
        )

    handle, opened = _open_regular_read_only(path, label)
    with handle:
        if expected_size_bytes is not None:
            _require(
                opened.st_size == expected_size_bytes,
                f"Unexpected {label} size: {opened.st_size}; "
                f"expected {expected_size_bytes}",
            )
        preload_digest = _sha256_stream(handle)
        if expected_sha256 is not None:
            _require(
                preload_digest == expected_sha256,
                f"Unexpected {label} SHA-256: {preload_digest}; "
                f"expected {expected_sha256}",
            )

        handle.seek(0)
        try:
            with torch.serialization.safe_globals(list(safe_globals)):
                payload = torch.load(
                    handle,
                    map_location="cpu",
                    weights_only=True,
                )
        except Exception as error:
            raise TargetFeatureAuthenticationError(
                f"Restricted weights-only {label} loading failed: "
                f"{type(error).__name__}: {error}"
            ) from error

        handle.seek(0)
        post_load_digest = _sha256_stream(handle)
        _require(
            post_load_digest == preload_digest,
            f"{label} bytes changed during restricted loading",
        )
        after = os.fstat(handle.fileno())
        _require(
            _same_identity(opened, after),
            f"{label} changed during restricted loading",
        )

    descriptor: dict[str, Any] = {
        "local_filename": Path(path).name,
        "size_bytes": opened.st_size,
        "sha256": preload_digest,
        "expected_sha256": expected_sha256,
        "sha256_matches_expected_when_provided": (
            expected_sha256 is None or preload_digest == expected_sha256
        ),
        "post_load_sha256": post_load_digest,
        "post_load_sha256_matches_preload": True,
        "restricted_weights_only_load": True,
        "map_location": "cpu",
        "explicit_safe_globals": [
            f"{value.__module__}.{value.__qualname__}"
            for value in safe_globals
            if hasattr(value, "__module__") and hasattr(value, "__qualname__")
        ],
    }
    if include_local_path_identity:
        # This is derived from the exact pathname opened through the
        # ancestor/final O_NOFOLLOW boundary above; consumers use it to retain
        # the historical descriptor contract without reopening the file.
        descriptor["path"] = os.path.abspath(
            os.fspath(Path(path).expanduser())
        )
        descriptor["mtime_ns"] = opened.st_mtime_ns
    return payload, descriptor


def _canonical_map_sha256(features: Mapping[str, torch.Tensor]) -> str:
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


def _target_subset_sha256(
    features: Mapping[str, torch.Tensor],
    targets: Sequence[str],
) -> str:
    digest = hashlib.sha256()
    for target in targets:
        key_bytes = target.encode("utf-8")
        digest.update(len(key_bytes).to_bytes(8, "little"))
        digest.update(key_bytes)
        digest.update(_tensor_bytes(features[target]))
    return digest.hexdigest()


def _validate_feature_payload(
    payload: object,
    spec: TargetFeatureSpec,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    _require(type(payload) is dict, "Feature payload root must be a plain dict")
    features = payload
    _require(
        len(features) == spec.feature_count,
        f"Unexpected feature count: {len(features)}; expected {spec.feature_count}",
    )
    _require(
        all(isinstance(key, str) and bool(key) for key in features),
        "Every feature key must be a non-empty string",
    )
    _require(
        len(set(features)) == len(features),
        "Feature keys must be unique",
    )

    for key, tensor in features.items():
        _require(
            type(tensor) is torch.Tensor,
            f"Feature value must be a plain tensor: {key}",
        )
        _require(tensor.device.type == "cpu", f"Feature tensor must be on CPU: {key}")
        _require(
            tensor.layout == torch.strided,
            f"Feature tensor must use strided layout: {key}",
        )
        _require(
            tensor.is_contiguous(),
            f"Feature tensor must be contiguous: {key}",
        )
        _require(
            tensor.dtype == torch.float32,
            f"Feature tensor must be float32: {key}",
        )
        _require(
            tuple(tensor.shape) == (spec.embedding_dimension,),
            f"Unexpected feature shape for {key}: {tuple(tensor.shape)}; "
            f"expected ({spec.embedding_dimension},)",
        )
        _require(bool(torch.isfinite(tensor).all()), f"Non-finite feature: {key}")
        _require(bool(torch.count_nonzero(tensor)), f"All-zero feature: {key}")

    keys = list(features)
    key_list_digest = _canonical_key_list_sha256(keys)
    _require(
        key_list_digest == spec.key_list_sha256,
        f"Unexpected canonical feature key-list SHA-256: {key_list_digest}; "
        f"expected {spec.key_list_sha256}",
    )
    canonical_digest = _canonical_map_sha256(features)
    _require(
        canonical_digest == spec.canonical_map_sha256,
        f"Unexpected canonical feature-map SHA-256: {canonical_digest}; "
        f"expected {spec.canonical_map_sha256}",
    )
    return features, {
        "feature_count": len(features),
        "embedding_dimension": spec.embedding_dimension,
        "tensor_dtype": "torch.float32",
        "tensor_layout": "torch.strided",
        "tensor_device": "cpu",
        "all_values_finite": True,
        "all_vectors_nonzero": True,
        "canonical_key_list_sha256": key_list_digest,
        "canonical_map_sha256": canonical_digest,
        "canonical_map_encoding": (
            "sorted UTF-8 keys; uint64-le lengths; compact JSON shape/dtype; "
            "little-endian float32 C-order tensor bytes"
        ),
    }


def _authenticate_extracted_features(
    path: Path,
    spec: TargetFeatureSpec,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    _require(
        Path(path).name == spec.extracted_filename,
        f"Unexpected extracted feature filename: {Path(path).name}",
    )
    payload, load_descriptor = load_restricted_tensor_mapping(
        path,
        label="extracted feature artifact",
        expected_sha256=spec.member_sha256,
        expected_size_bytes=spec.member_size_bytes,
    )
    features, structure = _validate_feature_payload(payload, spec)

    return features, {
        **load_descriptor,
        "matches_registered_archive_member": True,
        "structure": structure,
    }


def load_authenticated_arc_feature_map(
    path: Path,
    *,
    spec: TargetFeatureSpec = OFFICIAL_SPEC,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Public safe-loader boundary for consumers of the registered Arc map."""

    _validate_spec(spec)
    return _authenticate_extracted_features(Path(path), spec)


def _authenticate_targets(
    path: Path,
    spec: TargetFeatureSpec,
) -> tuple[list[str], dict[str, Any]]:
    handle, opened = _open_regular_read_only(path, "VCC target manifest")
    with handle:
        _require(
            Path(path).name == spec.target_manifest_filename,
            f"Unexpected target manifest filename: {Path(path).name}",
        )
        _require(
            opened.st_size == spec.target_manifest_size_bytes,
            f"Unexpected target manifest size: {opened.st_size}; "
            f"expected {spec.target_manifest_size_bytes}",
        )
        raw = handle.read()
        manifest_digest = hashlib.sha256(raw).hexdigest()
        _require(
            manifest_digest == spec.target_manifest_sha256,
            f"Unexpected target manifest SHA-256: {manifest_digest}; "
            f"expected {spec.target_manifest_sha256}",
        )
        after = os.fstat(handle.fileno())
        _require(
            _same_identity(opened, after),
            "VCC target manifest changed during authentication",
        )

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise TargetFeatureAuthenticationError(
            f"Target manifest is not strict UTF-8: {error}"
        ) from error
    rows = list(csv.reader(io.StringIO(text, newline="")))
    _require(bool(rows), "Target manifest is empty")
    _require(rows[0] == ["target_gene"], "Unexpected target manifest header")
    _require(
        all(len(row) == 1 and bool(row[0]) for row in rows[1:]),
        "Every target manifest row must contain one non-empty target_gene",
    )
    targets = [row[0] for row in rows[1:]]
    _require(
        len(targets) == spec.target_count,
        f"Unexpected target count: {len(targets)}; expected {spec.target_count}",
    )
    _require(len(set(targets)) == len(targets), "VCC target names must be unique")
    ordered_digest = hashlib.sha256(
        json.dumps(
            targets,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    _require(
        ordered_digest == spec.ordered_target_list_sha256,
        f"Unexpected ordered target-list SHA-256: {ordered_digest}; "
        f"expected {spec.ordered_target_list_sha256}",
    )
    return targets, {
        "local_filename": Path(path).name,
        "size_bytes": opened.st_size,
        "sha256": manifest_digest,
        "target_count": len(targets),
        "target_column": "target_gene",
        "targets_are_unique": True,
        "ordered_target_list_sha256": ordered_digest,
    }


def authenticate_target_features(
    archive_path: Path,
    feature_path: Path,
    target_manifest_path: Path,
    *,
    spec: TargetFeatureSpec = OFFICIAL_SPEC,
) -> dict[str, Any]:
    """Authenticate all registered inputs and return a deterministic receipt."""

    _validate_spec(spec)
    archive = _authenticate_archive(Path(archive_path), spec)
    features, feature_descriptor = _authenticate_extracted_features(
        Path(feature_path), spec
    )
    targets, target_descriptor = _authenticate_targets(
        Path(target_manifest_path), spec
    )

    missing = [target for target in targets if target not in features]
    _require(
        not missing,
        f"Feature artifact is missing requested VCC targets: {missing[:8]}",
    )
    subset_digest = _target_subset_sha256(features, targets)
    _require(
        subset_digest == spec.target_feature_subset_sha256,
        f"Unexpected ordered VCC target-feature subset SHA-256: {subset_digest}; "
        f"expected {spec.target_feature_subset_sha256}",
    )

    return {
        "schema": SCHEMA,
        "status": "passed",
        "contract": {
            "artifact_role": "opaque_arc_supplied_target_features",
            "authentication_scope": "artifact_identity_and_structure_only",
            "canonical_tensor_byte_order": "little-endian-float32",
            "does_not_establish_upstream_model_provenance": True,
            "opaque_artifact_training_mode": "official_opaque_artifact",
            "official_opaque_artifact_training_allowed": True,
            "provenance_complete": False,
            "gate_1b_allowed": False,
            "gate_1c_allowed": True,
            "gate_1c_scope": (
                "authenticated_arc_supplied_artifact_identity_and_structure_only"
            ),
        },
        "checks": {
            "archive_is_non_symlink_regular_file": True,
            "archive_size_and_sha256_match_registration": True,
            "archive_unchanged_during_authentication": True,
            "source_object_metadata_is_registered_observation": True,
            "source_url_is_not_claimed_permanently_immutable": True,
            "registered_member_is_unique_unencrypted_and_not_symlink": True,
            "registered_member_size_crc32_and_sha256_match": True,
            "extracted_artifact_matches_registered_member_bytes": True,
            "artifact_loaded_with_restricted_weights_only_mode": True,
            "artifact_unchanged_during_restricted_loading": True,
            "all_feature_keys_are_unique_nonempty_strings": True,
            "all_features_are_finite_nonzero_float32_registered_shape": True,
            "canonical_feature_hashes_match_registration": True,
            "target_manifest_identity_and_order_match_registration": True,
            "all_ordered_vcc_targets_have_exact_features": True,
            "receipt_omits_absolute_local_paths": True,
            "upstream_model_provenance_remains_fail_closed": True,
        },
        "archive": archive,
        "feature_artifact": feature_descriptor,
        "target_manifest": target_descriptor,
        "vcc_target_coverage": {
            "covered_targets": len(targets),
            "requested_targets": len(targets),
            "missing_targets": 0,
            "ordered_target_feature_subset_sha256": subset_digest,
        },
        "upstream_provenance": {
            "model_id": "UNKNOWN_NOT_PROVIDED",
            "model_revision": "UNKNOWN_NOT_PROVIDED",
            "weights_sha256": "UNKNOWN_NOT_PROVIDED",
            "protein_sequence_source": "UNKNOWN_NOT_PROVIDED",
            "gene_to_protein_mapping": "UNKNOWN_NOT_PROVIDED",
            "pooling_rule": "UNKNOWN_NOT_PROVIDED",
            "inferred_from_embedding_dimension": False,
        },
    }


def _open_or_create_directory_no_follow(path: Path) -> int:
    """Open/create a directory through dirfds while rejecting all symlinks."""

    absolute = Path(os.path.abspath(os.fspath(Path(path).expanduser())))
    parts = absolute.parts
    _require(len(parts) >= 1 and parts[0] == os.sep, "Invalid receipt directory")
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
    """Atomically publish through a symlink-safe dirfd without overwriting."""

    path = Path(os.path.abspath(os.fspath(Path(path).expanduser())))
    _require(path.name not in {"", ".", ".."}, "Invalid receipt filename")
    parent_descriptor = _open_or_create_directory_no_follow(path.parent)
    temporary_name = f".{path.name}.{os.getpid()}.{secrets.token_hex(12)}.tmp"
    temporary_created = False
    try:
        try:
            os.stat(path.name, dir_fd=parent_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(f"Refusing to overwrite receipt: {path}")
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
            raise FileExistsError(f"Refusing to overwrite receipt: {path}") from error
        os.fsync(parent_descriptor)
    finally:
        if temporary_created:
            try:
                os.unlink(temporary_name, dir_fd=parent_descriptor)
            except FileNotFoundError:
                pass
        os.close(parent_descriptor)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Authenticate Arc's opaque VCC ESM2 target-feature artifact "
            "without asserting upstream model provenance."
        )
    )
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional new JSON receipt; an existing path is never replaced.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    receipt = authenticate_target_features(
        args.archive,
        args.features,
        args.targets,
    )
    if args.output is not None:
        write_json_atomic(args.output, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return receipt


if __name__ == "__main__":
    main()
