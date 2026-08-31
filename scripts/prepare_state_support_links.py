#!/usr/bin/env python3
"""Create small STATE-compatible HDF5 wrappers without copying support matrices.

The official support archive mixes files with and without the AnnData
``uns/log1p`` marker even though all six matrices are log-normalized. Recent
cell-load releases reject that metadata inconsistency. This script preserves
the official files byte-for-byte and creates HDF5 external-link wrappers with
one consistent marker.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py


SUPPORT_FILES = (
    "competition_train.h5",
    "k562_gwps.h5",
    "rpe1.h5",
    "jurkat.h5",
    "k562.h5",
    "hepg2.h5",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("dataset/state_support/extracted"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/state_support/state_compatible"),
    )
    return parser.parse_args()


def create_wrapper(source: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing wrapper: {output}")

    source = source.resolve()
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"Refusing to overwrite temporary wrapper: {temporary}")
    with h5py.File(source, "r") as source_file, h5py.File(temporary, "w") as output_file:
        for key, value in source_file.attrs.items():
            output_file.attrs[key] = value

        for key in source_file.keys():
            if key != "uns":
                output_file[key] = h5py.ExternalLink(str(source), f"/{key}")

        uns = output_file.create_group("uns")
        source_uns = source_file.get("uns")
        if source_uns is not None:
            for key, value in source_uns.attrs.items():
                uns.attrs[key] = value
            for key in source_uns.keys():
                if key != "log1p":
                    uns[key] = h5py.ExternalLink(str(source), f"/uns/{key}")

        log1p = uns.create_group("log1p")
        log1p.attrs["encoding-type"] = "dict"
        log1p.attrs["encoding-version"] = "0.1.0"

    validate_wrapper(temporary)
    temporary.replace(output)


def validate_wrapper(path: Path) -> None:
    with h5py.File(path, "r") as wrapped:
        required = {"X", "obs", "var", "uns"}
        missing = required.difference(wrapped.keys())
        if missing:
            raise RuntimeError(f"{path} is missing groups: {sorted(missing)}")
        if "log1p" not in wrapped["uns"]:
            raise RuntimeError(f"{path} is missing uns/log1p")
        matrix = wrapped["X"]
        shape = matrix.shape if isinstance(matrix, h5py.Dataset) else tuple(matrix.attrs["shape"])
        if int(shape[1]) != 18_080:
            raise RuntimeError(f"Unexpected gene dimension in {path}: {shape}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename in SUPPORT_FILES:
        source = args.source_dir / filename
        output = args.output_dir / filename
        if not source.is_file():
            raise FileNotFoundError(source)
        create_wrapper(source, output)
        print(f"Created {output} -> {source.resolve()}")


if __name__ == "__main__":
    main()
