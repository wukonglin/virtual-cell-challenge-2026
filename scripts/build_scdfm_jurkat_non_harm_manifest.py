#!/usr/bin/env python3
"""Build the immutable V7 Jurkat non-harm manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from scdfm_jurkat_non_harm import (
    absolute_path,
    construct_manifest,
    registered_manifest_path,
    require,
    write_json_atomic,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/scdfm/vcc2026_v7_gamma1.toml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/scdfm/v7/splits/jurkat_non_harm.json"),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    require(
        absolute_path(args.output) == registered_manifest_path(args.config),
        "Output is not the manifest path registered by the V7 configuration",
    )
    manifest = construct_manifest(args.config)
    write_json_atomic(args.output, manifest)
    print(f"Wrote sealed Jurkat non-harm manifest: {args.output}")


if __name__ == "__main__":
    main()
