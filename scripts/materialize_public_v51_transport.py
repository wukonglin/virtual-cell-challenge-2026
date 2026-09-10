#!/usr/bin/env python3
"""Materialize path-rebased V5.1 sidecars without changing payload identity.

The public V5.1 panel and residual sidecars intentionally authenticate their
payloads by absolute path, size, and SHA-256.  A byte-identical transport to a
different filesystem therefore needs new sidecars; silently editing the source
sidecars would erase that lineage.  This utility verifies the source sidecars
and payload bytes, changes only the authenticated payload path, and records the
transport as explicit immutable provenance.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Sequence


PANEL_SCHEMA = "vcc-public-validation-manifest-v1"
RESIDUAL_SCHEMA = "vcc-public-state-residual-v1"
TRANSPORT_SCHEMA = "vcc-public-v51-transport-rebase-v1"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def describe_file(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": stat.st_size,
        "sha256": sha256_file(resolved),
    }


def parse_sha256(value: str) -> str:
    normalized = value.strip().lower()
    if SHA256_PATTERN.fullmatch(normalized) is None:
        raise argparse.ArgumentTypeError(
            "expected exactly 64 hexadecimal SHA-256 characters"
        )
    return normalized


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"JSON root is not an object: {path}")
    return payload


def build_rebased_sidecar(
    *,
    source_sidecar: Path,
    source_expected_sha256: str,
    expected_schema: str,
    payload_path: Path,
    provenance_key: str,
) -> dict[str, Any]:
    source_sidecar = source_sidecar.resolve(strict=True)
    payload_path = payload_path.resolve(strict=True)
    require(source_sidecar.is_file(), f"Source sidecar is not a file: {source_sidecar}")
    require(payload_path.is_file(), f"Transported payload is not a file: {payload_path}")

    source_descriptor = describe_file(source_sidecar)
    require(
        source_descriptor["sha256"] == source_expected_sha256,
        f"Source sidecar SHA-256 mismatch: {source_sidecar}",
    )
    source = load_json(source_sidecar)
    require(source.get("schema") == expected_schema, "Unexpected source sidecar schema")
    require(
        "transport_provenance" not in source,
        "Refusing to rebase an already transported sidecar",
    )

    provenance = source.get("provenance")
    require(isinstance(provenance, dict), "Source sidecar lacks provenance")
    recorded = provenance.get(provenance_key)
    require(
        isinstance(recorded, dict),
        f"Source sidecar lacks provenance.{provenance_key}",
    )
    original_path = recorded.get("path")
    recorded_sha256 = str(recorded.get("sha256", "")).lower()
    try:
        recorded_size = int(recorded.get("size_bytes", -1))
    except (TypeError, ValueError) as error:
        raise RuntimeError("Recorded payload size is invalid") from error
    require(
        isinstance(original_path, str) and Path(original_path).is_absolute(),
        "Recorded payload path must be absolute",
    )
    require(
        SHA256_PATTERN.fullmatch(recorded_sha256) is not None,
        "Recorded payload SHA-256 is invalid",
    )

    transported = describe_file(payload_path)
    require(
        transported["size_bytes"] == recorded_size,
        f"Transported payload size mismatch: {payload_path}",
    )
    require(
        transported["sha256"] == recorded_sha256,
        f"Transported payload SHA-256 mismatch: {payload_path}",
    )

    rebased = copy.deepcopy(source)
    rebased["provenance"][provenance_key]["path"] = transported["path"]
    rebased["transport_provenance"] = {
        "schema": TRANSPORT_SCHEMA,
        "lineage": "anvil-portable-reconstruction-v1",
        "operation": "authenticated-path-only-sidecar-rebase",
        "parent_schema_preserved": expected_schema,
        "source_sidecar": source_descriptor,
        "rebased_descriptor": {
            "json_pointer": f"/provenance/{provenance_key}",
            "original_path": original_path,
            "transported_path": transported["path"],
            "size_bytes": recorded_size,
            "sha256": recorded_sha256,
        },
        "payload_bytes_preserved": True,
        "scientific_content_changed": False,
    }
    return rebased


def _serialized_json(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def publish_jsons_no_overwrite(
    outputs: Sequence[tuple[Path, dict[str, Any]]],
) -> None:
    destinations = [path.absolute() for path, _ in outputs]
    require(len(destinations) == len(set(destinations)), "Output paths must be distinct")
    for destination in destinations:
        require(destination.suffix == ".json", "Every output must end in .json")
        require(
            not destination.exists() and not destination.is_symlink(),
            f"Refusing to overwrite output: {destination}",
        )
        destination.parent.mkdir(parents=True, exist_ok=True)

    staged: list[tuple[Path, Path]] = []
    published: list[Path] = []
    try:
        for (destination, payload), normalized in zip(
            outputs, destinations, strict=True
        ):
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{destination.name}.",
                suffix=".tmp",
                dir=normalized.parent,
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                handle.write(_serialized_json(payload))
                handle.flush()
                os.fsync(handle.fileno())
            staged.append((temporary, normalized))

        for temporary, destination in staged:
            os.link(temporary, destination)
            published.append(destination)
    except BaseException:
        for destination in published:
            destination.unlink(missing_ok=True)
        raise
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-json-source", type=Path, required=True)
    parser.add_argument(
        "--panel-json-expected-sha256", type=parse_sha256, required=True
    )
    parser.add_argument("--panel-csv", type=Path, required=True)
    parser.add_argument("--residual-json-source", type=Path, required=True)
    parser.add_argument(
        "--residual-json-expected-sha256", type=parse_sha256, required=True
    )
    parser.add_argument("--residual-npz", type=Path, required=True)
    parser.add_argument("--output-panel-json", type=Path, required=True)
    parser.add_argument("--output-residual-json", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    source_paths = {
        args.panel_json_source.resolve(strict=True),
        args.residual_json_source.resolve(strict=True),
    }
    for output in (args.output_panel_json, args.output_residual_json):
        require(
            output.absolute() not in source_paths,
            "Transport outputs must not replace source sidecars",
        )

    panel = build_rebased_sidecar(
        source_sidecar=args.panel_json_source,
        source_expected_sha256=args.panel_json_expected_sha256,
        expected_schema=PANEL_SCHEMA,
        payload_path=args.panel_csv,
        provenance_key="output_csv",
    )
    residual = build_rebased_sidecar(
        source_sidecar=args.residual_json_source,
        source_expected_sha256=args.residual_json_expected_sha256,
        expected_schema=RESIDUAL_SCHEMA,
        payload_path=args.residual_npz,
        provenance_key="output_npz",
    )
    publish_jsons_no_overwrite(
        (
            (args.output_panel_json, panel),
            (args.output_residual_json, residual),
        )
    )
    print(
        json.dumps(
            {
                "schema": TRANSPORT_SCHEMA,
                "panel_sidecar": describe_file(args.output_panel_json),
                "residual_sidecar": describe_file(args.output_residual_json),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
