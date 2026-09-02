#!/usr/bin/env python3
"""Build the deterministic repository-tracked V7 portable training lock."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

from scdfm_portable_training_gate import build_portable_hash_lock


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/scdfm/v7/splits/jurkat_non_harm.json"),
    )
    parser.add_argument(
        "--preflight",
        type=Path,
        default=Path(
            "artifacts/scdfm/v7/training/jurkat_training_preflight.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/scdfm/v7/training/portable_training_gate_lock.json"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = build_portable_hash_lock(args.manifest, args.preflight)
    output = Path(os.path.abspath(os.fspath(args.output.expanduser())))
    output.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(output, flags, 0o644)
    except FileExistsError as error:
        raise FileExistsError(f"Refusing to overwrite portable lock: {output}") from error
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(f"Wrote portable V7 training lock: {output}")


if __name__ == "__main__":
    main()
