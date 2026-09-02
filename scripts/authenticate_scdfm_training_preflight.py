#!/usr/bin/env python3
"""Reconstruct the V7 training preflight in the pinned offline audit env."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from scdfm_jurkat_non_harm import authenticate_training_preflight, write_json_atomic


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/scdfm/vcc2026_v7_gamma1.toml"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/scdfm/v7/splits/jurkat_non_harm.json"),
    )
    parser.add_argument(
        "--preflight",
        type=Path,
        default=Path("artifacts/scdfm/v7/training/jurkat_training_preflight.json"),
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    receipt = authenticate_training_preflight(
        args.config,
        args.manifest,
        args.preflight,
    )
    receipt["offline_audit_only"] = True
    receipt["portable_trainer_integration_implemented"] = False
    receipt["training_ready"] = False
    if args.output is not None:
        write_json_atomic(args.output, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
