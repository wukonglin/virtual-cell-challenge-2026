#!/usr/bin/env python3
"""Build and evaluate the registered EXS-1 expression-scale sweep.

The transform command is deliberately truth-blind: its parser has no truth
argument and it opens only an authenticated treated prediction plus the
controls-only generator input. For every predicted cell it attenuates the
predicted composition toward the paired source-control composition and
integerizes to either the original prediction library or the paired control
library.

The evaluate command is a separate scorer-side operation. It authenticates
completed cell-eval2 bundles, reconstructs the three registered cohort metrics
with the repository's existing aggregation helpers, and applies the gate in
docs/research/NEXT_ROUTE_DECISION.md. Missing arms remain explicitly unscored;
this program never substitutes expected or historical values for measurements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Sequence

import anndata as ad
import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sp

from exs1_local_io import atomic_write_json, concatenate_paths
from exs1_local_io import describe_file, require, sha256_file
from exs1_score_adapter import (
    MSE_METRIC,
    NMAE_METRIC,
    _mean_summary,
    _read_results,
    _validate_cell_eval_sidecars,
)


SCHEMA = "vcc-exs1-expression-scale-repair-v1"
EVALUATION_SCHEMA = "vcc-exs1-expression-scale-evaluation-v1"
DATA_SCHEMA = "vcc-public-validation-data-v1"
VIEWS_SCHEMA = "vcc-public-scoring-views-v1"
CONTROL_LABEL = "non-targeting"
PDS_METRIC = "pds_cosine"
EXPECTED_CELL_EVAL_COMMIT = "5e64833518a6603a0301cbe28185d49c30f4a986"
EXPECTED_CELL_EVAL_VERSION = "0.16.0"
EXPECTED_PDEX_VERSION = "0.3.0"
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")

# This grid is an implementation-level frozen sweep. The memo registered the
# success gate, not particular attenuation points.
DEFAULT_ATTENUATIONS = (
    "0.0000",
    "0.0200",
    "0.0500",
    "0.0800",
    "0.1000",
    "0.1500",
    "0.2000",
    "0.3500",
    "0.5000",
    "0.7500",
    "1.0000",
)
LIBRARY_POLICIES = ("keep_prediction", "match_source_control")
REQUIRED_OBS_COLUMNS = (
    "context",
    "target_gene",
    "source_control_row",
    "source_control_cell_id",
)
REGISTERED_GATE = {
    "source": "docs/research/NEXT_ROUTE_DECISION.md#exs-1--expression-scale-repair",
    "arm_pass_logic": (
        "mse_at_most_threshold AND pds_within_absolute_tolerance "
        "AND nmae_at_most_threshold"
    ),
    "experiment_pass_logic": "any_arm_passes",
    "metrics": {
        MSE_METRIC: {"operator": "<=", "threshold": 0.900},
        PDS_METRIC: {
            "operator": "absolute_difference<=",
            "reference": 0.564526198439242,
            "tolerance": 1e-6,
        },
        NMAE_METRIC: {"operator": "<=", "threshold": 1.017691308794254},
    },
}


@dataclass(frozen=True)
class AuthenticatedFile:
    """Immutable identity captured before a long read."""

    path: Path
    size_bytes: int
    mtime_ns: int
    sha256: str

    def descriptor(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "size_bytes": self.size_bytes,
            "mtime_ns": self.mtime_ns,
            "sha256": self.sha256,
        }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    transform = subparsers.add_parser(
        "transform", help="Create truth-blind EXS-1 prediction arms."
    )
    transform.add_argument("--prediction", type=Path, required=True)
    transform.add_argument("--prediction-expected-sha256", required=True)
    transform.add_argument("--controls-only", type=Path, required=True)
    transform.add_argument("--controls-expected-sha256", required=True)
    transform.add_argument("--output-dir", type=Path, required=True)
    transform.add_argument(
        "--attenuation",
        action="append",
        default=None,
        metavar="FACTOR",
        help=(
            "Repeat for a custom factor in [0,1], with at most four decimal "
            "places. The deterministic implementation grid is used when omitted."
        ),
    )
    transform.add_argument(
        "--library-policy",
        action="append",
        choices=LIBRARY_POLICIES,
        default=None,
        help="Repeat to restrict the two per-cell library variants.",
    )
    transform.add_argument("--chunk-rows", type=int, default=256)
    transform.add_argument(
        "--concat-max-loaded-elements", type=int, default=20_000_000
    )
    transform.add_argument(
        "--compression", choices=("lzf", "gzip", "none"), default="lzf"
    )

    evaluate = subparsers.add_parser(
        "evaluate", help="Authenticate scorer outputs and apply the EXS-1 gate."
    )
    evaluate.add_argument("--experiment-json", type=Path, required=True)
    evaluate.add_argument("--experiment-expected-sha256", required=True)
    evaluate.add_argument(
        "--score",
        action="append",
        nargs=5,
        default=[],
        metavar=(
            "ARM",
            "RESULTS_CSV",
            "RESULTS_SHA256",
            "VIEWS_JSON",
            "VIEWS_SHA256",
        ),
        help="One authenticated cell-eval2 result bundle; repeat for each arm.",
    )
    evaluate.add_argument("--cell-eval-checkout", type=Path, required=True)
    evaluate.add_argument("--output-json", type=Path, required=True)
    evaluate.add_argument(
        "--allow-partial",
        action="store_true",
        help="Emit null gates for arms whose scorer bundle is not supplied.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return _parser().parse_args(argv)


def _expected_sha256(value: str, label: str) -> str:
    normalized = str(value).strip().lower()
    require(
        SHA256_RE.fullmatch(normalized) is not None,
        f"Malformed expected SHA-256 for {label}",
    )
    return normalized


def _authenticate_regular_file(
    path: Path, expected: str, label: str
) -> AuthenticatedFile:
    """Reject links and authenticate bytes under an unchanged file identity."""

    require(not path.is_symlink(), f"{label} must not be a symbolic link: {path}")
    canonical = path.resolve()
    require(canonical.is_file(), f"Missing {label}: {canonical}")
    expected = _expected_sha256(expected, label)
    before = canonical.stat()
    observed = sha256_file(canonical)
    after = canonical.stat()
    require(
        (before.st_size, before.st_mtime_ns)
        == (after.st_size, after.st_mtime_ns),
        f"{label} changed while being authenticated",
    )
    require(observed == expected, f"{label} SHA-256 mismatch")
    return AuthenticatedFile(
        canonical, after.st_size, after.st_mtime_ns, observed
    )


def _reauthenticate(record: AuthenticatedFile, label: str) -> None:
    current = record.path.stat()
    require(
        (current.st_size, current.st_mtime_ns)
        == (record.size_bytes, record.mtime_ns),
        f"{label} changed during the run",
    )
    require(
        sha256_file(record.path) == record.sha256,
        f"{label} bytes changed during the run",
    )


def _authenticate_descriptor(record: Any, label: str) -> dict[str, Any]:
    require(isinstance(record, dict), f"{label} descriptor is absent")
    path = Path(str(record.get("path", ""))).resolve()
    expected = _expected_sha256(str(record.get("sha256", "")), label)
    require(path.is_file(), f"Missing {label}: {path}")
    require(not path.is_symlink(), f"{label} must not be a symbolic link")
    before = path.stat()
    require(
        before.st_size == int(record.get("size_bytes", -1)),
        f"{label} size mismatch",
    )
    observed = sha256_file(path)
    after = path.stat()
    require(
        (before.st_size, before.st_mtime_ns)
        == (after.st_size, after.st_mtime_ns),
        f"{label} changed while being authenticated",
    )
    require(observed == expected, f"{label} SHA-256 mismatch")
    if "mtime_ns" in record:
        require(
            after.st_mtime_ns == int(record["mtime_ns"]),
            f"{label} mtime mismatch",
        )
    return {
        "path": str(path),
        "size_bytes": after.st_size,
        "sha256": observed,
    }


def _same_descriptor(left: Any, right: Any, label: str) -> None:
    require(
        isinstance(left, dict) and isinstance(right, dict),
        f"{label} descriptor is absent",
    )
    for field in ("path", "size_bytes", "sha256"):
        if field == "path":
            lhs = str(Path(str(left.get(field, ""))).resolve())
            rhs = str(Path(str(right.get(field, ""))).resolve())
        else:
            lhs, rhs = left.get(field), right.get(field)
        require(lhs == rhs, f"{label} descriptor differs in {field}")


def _load_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file(), f"Missing {label}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Invalid {label}: {path}") from error
    require(isinstance(payload, dict), f"{label} must be a JSON object")
    return payload


def _parse_attenuations(
    values: Sequence[str] | None,
) -> tuple[Decimal, ...]:
    raw = tuple(values) if values is not None else DEFAULT_ATTENUATIONS
    parsed: list[Decimal] = []
    for value in raw:
        try:
            factor = Decimal(str(value))
        except InvalidOperation as error:
            raise RuntimeError(f"Invalid attenuation factor: {value}") from error
        require(
            factor.is_finite()
            and Decimal(0) <= factor <= Decimal(1),
            f"Attenuation outside [0,1]: {value}",
        )
        require(
            factor == factor.quantize(Decimal("0.0001")),
            f"Attenuation needs at most four decimal places: {value}",
        )
        parsed.append(factor.quantize(Decimal("0.0001")))
    require(bool(parsed), "At least one attenuation factor is required")
    require(
        len(parsed) == len(set(parsed)),
        "Attenuation factors contain duplicates",
    )
    return tuple(sorted(parsed))


def _arm_name(factor: Decimal, policy: str) -> str:
    basis_points = int(factor * Decimal(10_000))
    return f"attenuation_{basis_points:04d}bp__{policy}"


def _canonical_counts(matrix: Any, label: str) -> sp.csr_matrix:
    """Materialize a bounded block as canonical signed integer CSR counts."""

    if sp.issparse(matrix):
        original_dtype = np.dtype(matrix.dtype)
        require(
            np.issubdtype(original_dtype, np.integer),
            f"{label} storage is not integer counts",
        )
        result = sp.csr_matrix(matrix, dtype=np.int64, copy=True)
    else:
        dense = np.asarray(matrix)
        require(
            np.issubdtype(dense.dtype, np.integer),
            f"{label} storage is not integer counts",
        )
        result = sp.csr_matrix(dense.astype(np.int64, copy=False))
    result.sum_duplicates()
    result.sort_indices()
    result.eliminate_zeros()
    require(
        not result.data.size or np.all(result.data >= 0),
        f"{label} contains negative counts",
    )
    return result


def _largest_remainder(
    indices: np.ndarray,
    weights: np.ndarray,
    library: int,
    n_genes: int,
) -> sp.csr_matrix:
    """Integerize one sparse composition with gene-index tie breaking."""

    require(library > 0, "Per-cell output library must be positive")
    require(
        library <= np.iinfo(np.int32).max,
        "Per-cell output library exceeds int32",
    )
    require(
        indices.ndim == weights.ndim == 1
        and len(indices) == len(weights),
        "Bad sparse composition",
    )
    require(
        len(indices) > 0
        and np.all(np.isfinite(weights))
        and np.all(weights >= 0),
        "Invalid sparse composition",
    )
    total = float(weights.sum(dtype=np.float64))
    require(
        np.isfinite(total) and total > 0.0,
        "Sparse composition has no mass",
    )
    scaled = (weights / total) * float(library)
    base = np.floor(scaled).astype(np.int64)
    missing = int(library - int(base.sum(dtype=np.int64)))
    require(
        0 <= missing <= len(base),
        "Largest-remainder deficit is invalid",
    )
    if missing:
        fractions = scaled - base
        order = np.lexsort(
            (indices.astype(np.int64, copy=False), -fractions)
        )
        base[order[:missing]] += 1
    keep = base > 0
    data = base[keep]
    require(
        not data.size
        or int(data.max()) <= np.iinfo(np.int32).max,
        "Integerized count exceeds int32",
    )
    indptr = np.asarray([0, int(keep.sum())], dtype=np.int64)
    return sp.csr_matrix(
        (
            data.astype(np.int32),
            indices[keep].astype(np.int32),
            indptr,
        ),
        shape=(1, n_genes),
        dtype=np.int32,
    )


def attenuate_count_row(
    prediction_row: sp.csr_matrix,
    control_row: sp.csr_matrix,
    *,
    attenuation: float,
    library_policy: str,
    n_genes: int,
) -> sp.csr_matrix:
    """Return one deterministic EXS-1 row; public for focused tests."""

    require(
        0.0 <= attenuation <= 1.0 and np.isfinite(attenuation),
        "Invalid attenuation",
    )
    require(
        library_policy in LIBRARY_POLICIES,
        "Unknown library policy",
    )
    prediction_row = _canonical_counts(
        prediction_row, "prediction row"
    )
    control_row = _canonical_counts(control_row, "control row")
    require(
        prediction_row.shape == control_row.shape == (1, n_genes),
        "Paired row shape mismatch",
    )
    pred_library = int(prediction_row.sum(dtype=np.int64))
    control_library = int(control_row.sum(dtype=np.int64))
    require(
        pred_library > 0 and control_library > 0,
        "Input libraries must be positive",
    )

    # Preserve two exact endpoints without floating-point round trips.
    if (
        attenuation == 1.0
        and library_policy == "keep_prediction"
    ):
        return prediction_row.astype(np.int32)
    if (
        attenuation == 0.0
        and library_policy == "match_source_control"
    ):
        return control_row.astype(np.int32)

    pred_indices = prediction_row.indices.astype(
        np.int64, copy=False
    )
    control_indices = control_row.indices.astype(
        np.int64, copy=False
    )
    indices = np.union1d(pred_indices, control_indices).astype(
        np.int64, copy=False
    )
    pred_weights = np.zeros(len(indices), dtype=np.float64)
    control_weights = np.zeros(len(indices), dtype=np.float64)
    pred_weights[np.searchsorted(indices, pred_indices)] = (
        prediction_row.data / pred_library
    )
    control_weights[np.searchsorted(indices, control_indices)] = (
        control_row.data / control_library
    )
    weights = (
        attenuation * pred_weights
        + (1.0 - attenuation) * control_weights
    )
    library = (
        pred_library
        if library_policy == "keep_prediction"
        else control_library
    )
    result = _largest_remainder(
        indices, weights, library, n_genes
    )
    require(
        int(result.sum(dtype=np.int64)) == library,
        "Integerization changed the target library",
    )
    return result


def _axis_hash(values: Iterable[str], label: str) -> str:
    digest = hashlib.sha256(
        f"{SCHEMA}|{label}|string-axis-v1\n".encode("utf-8")
    )
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _logical_csr_hash(data: ad.AnnData, chunk_rows: int) -> str:
    digest = hashlib.sha256(
        f"{SCHEMA}|logical-csr-v1|{data.n_obs}|{data.n_vars}".encode(
            "utf-8"
        )
    )
    for start in range(0, data.n_obs, chunk_rows):
        stop = min(start + chunk_rows, data.n_obs)
        block = _canonical_counts(
            data.X[start:stop, :], "logical-hash block"
        )
        for row in range(block.shape[0]):
            left = int(block.indptr[row])
            right = int(block.indptr[row + 1])
            digest.update(
                np.asarray([right - left], dtype="<u8").tobytes()
            )
            digest.update(
                np.asarray(
                    block.indices[left:right], dtype="<u8"
                ).tobytes()
            )
            digest.update(
                np.asarray(
                    block.data[left:right], dtype="<i8"
                ).tobytes()
            )
    return digest.hexdigest()
