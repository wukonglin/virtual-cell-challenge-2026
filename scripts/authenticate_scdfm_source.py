#!/usr/bin/env python3
"""Authenticate the pinned official scDFM source checkout.

The checkout is intentionally kept below the Git-ignored dataset tree.  This
tool proves that the source used by an experiment is the reviewed upstream
revision, has the reviewed tree and MIT license, contains the required runtime
files, and has no local changes.  It never modifies the checkout and emits no
receipt file unless ``--output`` is provided.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Sequence


SCHEMA = "vcc-scdfm-source-authentication-v1"
DEFAULT_SOURCE = Path("dataset/external_repos/scDFM")
OFFICIAL_REPOSITORY = "https://github.com/AI4Science-WestlakeU/scDFM.git"
EXPECTED_COMMIT = "2cf6bca1f044e74c4e1dc586892c0495880cf125"
EXPECTED_TREE = "a04130b07020505a609158cbe31e9e65083c0d79"
EXPECTED_LICENSE_SHA256 = (
    "dedf7a6ce9cb9f1062d6f387a6a164320a442d61591627f81ce6aedc7c8e2fdb"
)
REQUIRED_FILES = (
    ".gitignore",
    "LICENSE",
    "README.md",
    "environment.yml",
    "run.sh",
    "config/config_flow.py",
    "src/data_process/data.py",
    "src/flow_matching/path/affine.py",
    "src/flow_matching/solver/ode_solver.py",
    "src/models/instantiate_model.py",
    "src/models/origin/model.py",
    "src/script/run.py",
    "src/tokenizer/gene_tokenizer.py",
    "src/utils/utils.py",
)


class AuthenticationError(RuntimeError):
    """Raised when a source checkout differs from the pinned contract."""


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    """Return the SHA-256 digest of one regular file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _git(source: Path, *arguments: str) -> str:
    """Run one read-only Git command and return stripped standard output."""

    try:
        completed = subprocess.run(
            ["git", "-C", str(source), *arguments],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        raise AuthenticationError(f"Unable to execute Git: {error}") from error
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise AuthenticationError(
            f"Git command failed ({' '.join(arguments)}): {detail}"
        )
    return completed.stdout.strip()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuthenticationError(message)


def _validate_required_files(source: Path, required_files: Sequence[str]) -> None:
    """Require every declared runtime input to be tracked and regular."""

    tracked = set(_git(source, "ls-files", "-z").split("\0"))
    tracked.discard("")
    for relative_name in required_files:
        relative = Path(relative_name)
        _require(
            not relative.is_absolute() and ".." not in relative.parts,
            f"Unsafe required-file path: {relative_name}",
        )
        path = source / relative
        _require(path.is_file(), f"Missing required scDFM file: {relative_name}")
        _require(
            not path.is_symlink(),
            f"Required scDFM file must not be a symbolic link: {relative_name}",
        )
        _require(
            relative.as_posix() in tracked,
            f"Required scDFM file is not tracked: {relative_name}",
        )


def authenticate_source(
    source: Path,
    *,
    expected_commit: str = EXPECTED_COMMIT,
    expected_tree: str = EXPECTED_TREE,
    expected_license_sha256: str = EXPECTED_LICENSE_SHA256,
    required_files: Sequence[str] = REQUIRED_FILES,
) -> dict[str, Any]:
    """Authenticate ``source`` and return a deterministic receipt payload."""

    source = source.resolve()
    _require(source.is_dir(), f"Missing scDFM source checkout: {source}")
    _require((source / ".git").exists(), f"scDFM source is not a Git checkout: {source}")

    repository_root = Path(_git(source, "rev-parse", "--show-toplevel")).resolve()
    _require(
        repository_root == source,
        "scDFM source path is not the repository root",
    )

    commit = _git(source, "rev-parse", "HEAD^{commit}")
    tree = _git(source, "rev-parse", "HEAD^{tree}")
    _require(
        commit == expected_commit,
        f"Unexpected scDFM commit: {commit}; expected {expected_commit}",
    )
    _require(
        tree == expected_tree,
        f"Unexpected scDFM tree: {tree}; expected {expected_tree}",
    )

    _validate_required_files(source, required_files)

    status = _git(source, "status", "--porcelain=v1", "--untracked-files=all")
    _require(not status, f"scDFM source checkout is dirty: {status.splitlines()}")

    license_path = source / "LICENSE"
    license_text = license_path.read_text(encoding="utf-8")
    _require(
        license_text.startswith("MIT License\n")
        and "Permission is hereby granted, free of charge" in license_text,
        "scDFM LICENSE does not contain the reviewed MIT license text",
    )
    license_sha256 = sha256_file(license_path)
    _require(
        license_sha256 == expected_license_sha256,
        "Unexpected scDFM LICENSE SHA-256: "
        f"{license_sha256}; expected {expected_license_sha256}",
    )

    required = tuple(Path(name).as_posix() for name in required_files)
    return {
        "schema": SCHEMA,
        "status": "passed",
        "checks": {
            "commit_matches": True,
            "license_is_reviewed_mit_text": True,
            "license_sha256_matches": True,
            "repository_root_matches_source": True,
            "required_files_are_present_regular_and_tracked": True,
            "tracked_and_untracked_worktree_clean": True,
            "tree_matches": True,
        },
        "expected": {
            "repository": OFFICIAL_REPOSITORY,
            "commit": expected_commit,
            "tree": expected_tree,
            "license_sha256": expected_license_sha256,
        },
        "source": {
            "path": str(source),
            "repository_commit": commit,
            "repository_tree": tree,
            "worktree_clean": True,
            "required_files": list(required),
            "license": {
                "path": "LICENSE",
                "spdx_identifier": "MIT",
                "size_bytes": license_path.stat().st_size,
                "sha256": license_sha256,
            },
        },
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write a new deterministic JSON receipt without replacing an artifact."""

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

        # Publish the complete inode in one operation.  Unlike os.replace(),
        # link() fails when another process has already created the receipt.
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
        description="Authenticate the pinned official scDFM source checkout."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"Official scDFM checkout (default: {DEFAULT_SOURCE}).",
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
    receipt = authenticate_source(args.source)
    if args.output is not None:
        write_json_atomic(args.output, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return receipt


if __name__ == "__main__":
    main()
