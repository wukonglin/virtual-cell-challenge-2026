"""Small CPU-only I/O primitives for the EXS-1 command.

These mirror the repository's established descriptor, atomic-JSON, and
on-disk-concatenation contracts without importing the GPU model stack.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from anndata.experimental import concat_on_disk


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(
    path: Path, chunk_size: int = 8 << 20
) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(chunk_size), b""
        ):
            digest.update(block)
    return digest.hexdigest()


def describe_file(
    path: Path, *, hash_file: bool = True
) -> dict[str, Any]:
    canonical = path.resolve()
    stat = canonical.stat()
    result: dict[str, Any] = {
        "path": str(canonical),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if hash_file:
        result["sha256"] = sha256_file(canonical)
    return result


def atomic_write_json(
    path: Path, payload: dict[str, Any]
) -> None:
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.tmp"
    )
    require(
        not temporary.exists(),
        f"Temporary JSON output exists: {temporary}",
    )
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def concatenate_paths(
    paths: list[Path],
    output: Path,
    max_loaded_elements: int,
) -> None:
    require(paths, "No H5AD chunks to concatenate")
    require(
        not output.exists(),
        f"Temporary output already exists: {output}",
    )
    concat_on_disk(
        paths,
        output,
        axis=0,
        join="inner",
        merge="same",
        uns_merge="same",
        max_loaded_elems=max_loaded_elements,
    )
