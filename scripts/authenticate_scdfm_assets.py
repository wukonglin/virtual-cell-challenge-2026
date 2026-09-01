#!/usr/bin/env python3
"""Authenticate the official scDFM archives and optional extracted inputs.

The official scDFM release does not publish cryptographic checksums for its
Google Drive artifacts.  This tool therefore binds each locally downloaded
archive to its exact official byte length, computes a SHA-256 digest for later
receipts, validates every ZIP member and CRC, and rejects paths that would be
unsafe to extract.  It never extracts archive contents.  If an extracted root
is supplied, the required Norman and ComboSciPlex files are authenticated
against sizes and streamed SHA-256 digests read from the already-validated ZIP
members.

Receipt payloads deliberately contain only local base names, never absolute
input paths.  No receipt is written unless ``--output`` is supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import ntpath
import os
import re
import stat
import tempfile
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Mapping, Sequence


SCHEMA = "vcc-scdfm-assets-authentication-v2"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
CRC32_PATTERN = re.compile(r"[0-9a-f]{8}")
DEFAULT_DOWNLOAD_DIRECTORY = Path("dataset/scdfm/downloads")


@dataclass(frozen=True)
class AssetSpec:
    """Immutable identity fields for one official archive."""

    key: str
    official_filename: str
    google_drive_file_id: str
    expected_size_bytes: int
    expected_sha256: str | None = None


ASSET_SPECS = (
    AssetSpec(
        key="checkpoints",
        official_filename="scDFM_ckpts.zip",
        google_drive_file_id="1ObRTXCt5_H3TIC54A6nOCKycltq88BwX",
        expected_size_bytes=5_931_951_086,
        expected_sha256="373e69c80b88e1735cd65afe3039385cdc91fe0ef33e64b324228e029df4e937",
    ),
    AssetSpec(
        key="norman",
        official_filename="norman.zip",
        google_drive_file_id="1FuG8y2pdS1XTLko7h4STKppH1PM4Oyzf",
        expected_size_bytes=614_006_395,
        expected_sha256="794564cef2c0ee8f8d7fbe2ee69e87cd4eba3f14a995bd28639b6d330d76cf24",
    ),
    AssetSpec(
        key="combosciplex",
        official_filename="combosciplex.zip",
        google_drive_file_id="1kqev8qqA9W_QH59wFgoFMqiBKzUAFjAG",
        expected_size_bytes=702_210_085,
        expected_sha256="60095672660ca3d983fe85cafb26541df3139790d7f87e1deb0e62283f0235f4",
    ),
)
SPEC_BY_KEY = {spec.key: spec for spec in ASSET_SPECS}
DEFAULT_PATHS = {
    "checkpoints": DEFAULT_DOWNLOAD_DIRECTORY / "checkpoints.zip",
    "norman": DEFAULT_DOWNLOAD_DIRECTORY / "norman.zip",
    "combosciplex": DEFAULT_DOWNLOAD_DIRECTORY / "combosciplex.zip",
}
REQUIRED_EXTRACTED_MEMBERS = {
    "norman": (
        "norman/norman.h5ad",
        "norman/split_results.pkl",
    ),
    "combosciplex": ("combosciplex/combosciplex.h5ad",),
}


class AssetAuthenticationError(RuntimeError):
    """Raised when an archive violates the registered asset contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssetAuthenticationError(message)


def _sha256_stream(handle: BinaryIO, chunk_size: int = 8 << 20) -> str:
    """Hash a seekable binary stream from its current position."""

    digest = hashlib.sha256()
    while block := handle.read(chunk_size):
        digest.update(block)
    return digest.hexdigest()


def _safe_member_name(info: zipfile.ZipInfo) -> str:
    """Return a normalized member name or reject an extraction-unsafe name."""

    name = info.filename
    _require(bool(name), "ZIP contains an empty member name")
    _require("\x00" not in name, f"ZIP member contains NUL: {name!r}")

    # ZIP uses POSIX separators, but hostile archives often use backslashes to
    # bypass checks on POSIX hosts.  Treat both forms as path separators.
    portable_name = name.replace("\\", "/")
    drive, _ = ntpath.splitdrive(name)
    _require(not drive, f"ZIP member has a Windows drive or UNC path: {name!r}")
    path = PurePosixPath(portable_name)
    _require(not path.is_absolute(), f"ZIP member has an absolute path: {name!r}")
    _require(".." not in path.parts, f"ZIP member traverses a parent: {name!r}")
    _require(
        path.as_posix() not in {"", "."},
        f"ZIP member has no usable relative path: {name!r}",
    )

    unix_mode = (info.external_attr >> 16) & 0xFFFF
    _require(
        not stat.S_ISLNK(unix_mode),
        f"ZIP member is a symbolic link: {name!r}",
    )
    _require(
        not (info.flag_bits & 0x1),
        f"ZIP member is encrypted and cannot be authenticated: {name!r}",
    )
    return path.as_posix()


def _validate_zip(
    handle: BinaryIO,
    asset_key: str,
    required_members: Sequence[str] = (),
) -> tuple[dict[str, int | bool], dict[str, dict[str, int | str]]]:
    """Validate ZIP metadata and member CRCs without extracting any file."""

    handle.seek(0)
    try:
        with zipfile.ZipFile(handle, mode="r") as archive:
            members = archive.infolist()
            _require(members, f"{asset_key} ZIP has no members")

            normalized_names: set[str] = set()
            checkpoint_count = 0
            file_member_count = 0
            directory_member_count = 0
            compressed_bytes = 0
            uncompressed_bytes = 0
            required = set(required_members)
            required_member_metadata: dict[str, dict[str, int | str]] = {}
            for info in members:
                normalized_name = _safe_member_name(info)
                _require(
                    normalized_name not in normalized_names,
                    f"{asset_key} ZIP has a duplicate normalized member: "
                    f"{normalized_name!r}",
                )
                normalized_names.add(normalized_name)

                compressed_bytes += info.compress_size
                uncompressed_bytes += info.file_size
                if info.is_dir():
                    directory_member_count += 1
                else:
                    file_member_count += 1
                    if normalized_name in required:
                        # Hash the decompressed member stream from this exact
                        # authenticated archive.  Extracted files are later
                        # required to match this digest, not merely the central
                        # directory's byte length.
                        with archive.open(info, mode="r") as member_handle:
                            member_sha256 = _sha256_stream(member_handle)
                        required_member_metadata[normalized_name] = {
                            "expected_size_bytes": info.file_size,
                            "expected_sha256": member_sha256,
                            "zip_crc32": f"{info.CRC:08x}",
                        }
                    if PurePosixPath(normalized_name).name == "checkpoint.pt":
                        checkpoint_count += 1

            missing_required = sorted(required - set(required_member_metadata))
            _require(
                not missing_required,
                f"{asset_key} ZIP is missing required members: {missing_required}",
            )

            bad_member = archive.testzip()
            _require(
                bad_member is None,
                f"{asset_key} ZIP member failed CRC validation: {bad_member!r}",
            )
    except AssetAuthenticationError:
        raise
    except (
        OSError,
        RuntimeError,
        NotImplementedError,
        zipfile.BadZipFile,
        zlib.error,
    ) as error:
        raise AssetAuthenticationError(
            f"{asset_key} is not a structurally valid readable ZIP: {error}"
        ) from error

    return (
        {
            "member_count": len(members),
            "file_member_count": file_member_count,
            "directory_member_count": directory_member_count,
            "compressed_member_bytes": compressed_bytes,
            "uncompressed_member_bytes": uncompressed_bytes,
            "checkpoint_count": checkpoint_count,
            "all_member_crcs_valid": True,
            "all_member_paths_safe": True,
            "contains_no_symlink_members": True,
        },
        dict(sorted(required_member_metadata.items())),
    )


def authenticate_archive(path: Path, spec: AssetSpec) -> dict[str, Any]:
    """Authenticate one archive against ``spec`` and return its descriptor."""

    _require(path.exists(), f"Missing {spec.key} archive: {path}")
    _require(not path.is_symlink(), f"{spec.key} archive must not be a symlink")

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise AssetAuthenticationError(
            f"Unable to open {spec.key} archive: {error}"
        ) from error

    with os.fdopen(descriptor, "rb") as handle:
        before = os.fstat(handle.fileno())
        _require(stat.S_ISREG(before.st_mode), f"{spec.key} archive is not regular")
        _require(
            before.st_size == spec.expected_size_bytes,
            f"Unexpected {spec.key} archive size: {before.st_size}; "
            f"expected {spec.expected_size_bytes}",
        )

        digest = _sha256_stream(handle)
        if spec.expected_sha256 is not None:
            _require(
                digest == spec.expected_sha256,
                f"Unexpected {spec.key} archive SHA-256: {digest}; "
                f"expected {spec.expected_sha256}",
            )
        zip_summary, required_member_metadata = _validate_zip(
            handle,
            spec.key,
            REQUIRED_EXTRACTED_MEMBERS.get(spec.key, ()),
        )

        after = os.fstat(handle.fileno())
        _require(
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            == (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns),
            f"{spec.key} archive changed during authentication",
        )

    return {
        "official_filename": spec.official_filename,
        "local_filename": path.name,
        "google_drive_file_id": spec.google_drive_file_id,
        "expected_size_bytes": spec.expected_size_bytes,
        "expected_sha256": spec.expected_sha256,
        "observed_size_bytes": before.st_size,
        "sha256": digest,
        "zip": zip_summary,
        "required_extracted_members": required_member_metadata,
    }


def _authenticate_extracted_file(
    root: Path,
    member_name: str,
    expected_size_bytes: int,
    expected_sha256: str,
    archive_key: str,
) -> dict[str, Any]:
    """Authenticate one required extracted file without exposing its root."""

    relative = PurePosixPath(member_name)
    _require(
        not relative.is_absolute() and ".." not in relative.parts,
        f"Unsafe registered extracted member: {member_name!r}",
    )

    candidate = root
    for part in relative.parts:
        candidate = candidate / part
        _require(
            not candidate.is_symlink(),
            f"Extracted member path contains a symlink: {member_name!r}",
        )

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(candidate, flags)
    except OSError as error:
        raise AssetAuthenticationError(
            f"Unable to open extracted member {member_name!r}: {error}"
        ) from error

    with os.fdopen(descriptor, "rb") as handle:
        before = os.fstat(handle.fileno())
        _require(
            stat.S_ISREG(before.st_mode),
            f"Extracted member is not a regular file: {member_name!r}",
        )
        _require(
            before.st_size == expected_size_bytes,
            f"Unexpected extracted member size for {member_name!r}: "
            f"{before.st_size}; expected {expected_size_bytes}",
        )
        digest = _sha256_stream(handle)
        _require(
            digest == expected_sha256,
            f"Extracted member SHA-256 does not match ZIP member for "
            f"{member_name!r}: {digest}; expected {expected_sha256}",
        )
        after = os.fstat(handle.fileno())
        _require(
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            == (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns),
            f"Extracted member changed during authentication: {member_name!r}",
        )

    return {
        "archive_key": archive_key,
        "archive_member": member_name,
        "local_relative_path": member_name,
        "expected_size_bytes_from_zip": expected_size_bytes,
        "expected_sha256_from_zip": expected_sha256,
        "observed_size_bytes": before.st_size,
        "sha256": digest,
        "sha256_matches_zip_member": True,
    }


def authenticate_extracted_files(
    root: Path,
    assets: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Authenticate all registered extracted members below ``root``."""

    root = Path(root)
    _require(root.exists(), "Extracted root does not exist")
    _require(not root.is_symlink(), "Extracted root must not be a symlink")
    _require(root.is_dir(), "Extracted root must be a directory")

    authenticated: dict[str, dict[str, Any]] = {}
    for archive_key in sorted(REQUIRED_EXTRACTED_MEMBERS):
        _require(
            archive_key in assets,
            f"Missing authenticated archive for extracted files: {archive_key}",
        )
        registered = assets[archive_key].get("required_extracted_members")
        _require(
            isinstance(registered, Mapping),
            f"Missing required-member bindings for archive: {archive_key}",
        )
        for member_name in REQUIRED_EXTRACTED_MEMBERS[archive_key]:
            binding = registered.get(member_name)
            _require(
                isinstance(binding, Mapping),
                f"Missing cryptographic member binding: {member_name!r}",
            )
            expected_size = binding.get("expected_size_bytes")
            expected_sha256 = binding.get("expected_sha256")
            expected_crc32 = binding.get("zip_crc32")
            _require(
                isinstance(expected_size, int)
                and not isinstance(expected_size, bool)
                and expected_size >= 0,
                f"Invalid bound member size: {member_name!r}",
            )
            _require(
                isinstance(expected_sha256, str)
                and SHA256_PATTERN.fullmatch(expected_sha256) is not None,
                f"Missing bound member SHA-256: {member_name!r}",
            )
            _require(
                isinstance(expected_crc32, str)
                and CRC32_PATTERN.fullmatch(expected_crc32) is not None,
                f"Missing bound member CRC32: {member_name!r}",
            )
            authenticated[member_name] = _authenticate_extracted_file(
                root,
                member_name,
                expected_size,
                expected_sha256,
                archive_key,
            )
    return dict(sorted(authenticated.items()))


def authenticate_assets(
    paths: Mapping[str, Path],
    *,
    specs: Sequence[AssetSpec] = ASSET_SPECS,
    extracted_root: Path | None = None,
) -> dict[str, Any]:
    """Authenticate all registered archives and build a deterministic receipt."""

    expected_keys = {spec.key for spec in specs}
    _require(
        set(paths) == expected_keys,
        "Asset path keys must exactly match registered specs: "
        f"{sorted(expected_keys)}",
    )
    _require(
        len(expected_keys) == len(specs),
        "Asset specification keys must be unique",
    )

    assets = {
        spec.key: authenticate_archive(Path(paths[spec.key]), spec)
        for spec in sorted(specs, key=lambda item: item.key)
    }
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "passed",
        "contract": {
            "required_extracted_member_binding": (
                "decompressed_sha256+uncompressed_size+zip_crc32"
            ),
            "legacy_size_only_receipts_accepted": False,
        },
        "checks": {
            "all_archives_are_regular_non_symlinks": True,
            "all_archive_sizes_match_official_byte_lengths": True,
            "all_archive_sha256_digests_computed": True,
            "all_registered_archive_sha256_digests_match": True,
            "all_zip_central_directories_parsed": True,
            "all_zip_member_crcs_valid": True,
            "all_zip_member_paths_safe": True,
            "all_zip_members_are_unencrypted": True,
            "all_zip_members_are_not_symlinks": True,
            "receipt_omits_local_directory_paths": True,
        },
        "assets": assets,
    }
    if extracted_root is not None:
        receipt["extracted_files"] = authenticate_extracted_files(
            extracted_root,
            assets,
        )
        receipt["checks"]["all_required_extracted_files_are_regular"] = True
        receipt["checks"]["all_extracted_sizes_match_zip_metadata"] = True
        receipt["checks"]["all_extracted_sha256_digests_computed"] = True
        receipt["checks"][
            "all_extracted_sha256_digests_match_zip_members"
        ] = True
        receipt["checks"][
            "all_required_member_bindings_include_size_sha256_and_crc32"
        ] = True
    return receipt


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically publish a complete JSON receipt without overwriting a path."""

    path = Path(path)
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

        # A hard link exposes the already-complete inode in one operation and
        # fails with EEXIST instead of replacing an existing receipt.
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(f"Refusing to overwrite receipt: {path}") from error
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Authenticate the three official scDFM ZIP archives."
    )
    parser.add_argument(
        "--checkpoints",
        type=Path,
        default=DEFAULT_PATHS["checkpoints"],
        help=f"Checkpoint ZIP (default: {DEFAULT_PATHS['checkpoints']}).",
    )
    parser.add_argument(
        "--norman",
        type=Path,
        default=DEFAULT_PATHS["norman"],
        help=f"Norman ZIP (default: {DEFAULT_PATHS['norman']}).",
    )
    parser.add_argument(
        "--combosciplex",
        type=Path,
        default=DEFAULT_PATHS["combosciplex"],
        help=f"ComboSciPlex ZIP (default: {DEFAULT_PATHS['combosciplex']}).",
    )
    parser.add_argument(
        "--extracted-root",
        type=Path,
        default=None,
        help=(
            "Optional root containing norman/ and combosciplex/; required "
            "files are checked against their ZIP metadata."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional new JSON receipt; existing paths are never replaced.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    receipt = authenticate_assets(
        {
            "checkpoints": args.checkpoints,
            "norman": args.norman,
            "combosciplex": args.combosciplex,
        },
        extracted_root=args.extracted_root,
    )
    if args.output is not None:
        write_json_atomic(args.output, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return receipt


if __name__ == "__main__":
    main()
