#!/usr/bin/env python3
"""Run the registered PDT-1 public-transfer direction experiment on CPU.

The experiment is intentionally pseudobulk-only. It never writes a prediction
artifact and it never renders single-cell counts. A0--A4 are constructed solely
from authenticated ``g100`` pseudobulk, an authenticated K562 GWPS effect atlas,
and constants frozen in ``docs/research/NEXT_ROUTE_DECISION.md``. The sealed
HepG2 truth is loaded only after those non-oracle arms exist; it is then used by
the pinned cell-eval2 scorer and, separately, to construct the A5 ceiling.

Every scientific file requires an expected SHA-256. The atlas receipt is
cross-bound to its NPZ and effect-space contract, the scorer checkout must be
clean at its expected commit, axes are authenticated in the output, and receipt
publication is atomic and no-overwrite.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import secrets
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import numpy as np

import diagnose_pds_reachability as pds


SCHEMA = "vcc-pdt1-public-transfer-v1"
ATLAS_SCHEMA = "vcc-bulk-effect-atlas-v1"
ATLAS_EFFECT_SPACE = "log1p-group-sum-cp50000-target-minus-batch-matched-control"
ATLAS_REQUIRED_KEYS = (
    "effects",
    "matched_control_profiles",
    "target_names",
    "gene_names",
    "target_cell_counts",
    "batch_names",
    "target_batch_counts",
    "control_batch_counts",
    "target_sum",
)

# Frozen by NEXT_ROUTE_DECISION.md; these are deliberately not CLI knobs.
SHARED_REMOVAL_FRACTION = 1.05
SHUFFLE_SEED = 20260901
REGISTERED_A0_PDS = 0.564526198439242
REGISTERED_A1_PDS = 0.576667
REGISTERED_PDS_FLOOR = 0.6263
REGISTERED_SHUFFLE_MARGIN = 0.0400
REGISTERED_ANTICORRELATED_CEILING = 0.300
REGISTERED_TARGET_COUNT = 300
REGISTERED_GENE_COUNT = 7_107
REGISTERED_CANDIDATE = "public_hepg2_p4_g100_raw_candidate"
REGISTERED_CELL_EVAL_COMMIT = "5e64833518a6603a0301cbe28185d49c30f4a986"
REGISTERED_INPUT_SHA256 = {
    "prediction_view": "7fe4da1f3f844b54303a255ece2ed4e1f9f5d0f2a5bfa5062694a0928abc7065",
    "truth_view": "f6c532dd1bad5b649ebb3ce596e6e2d2187f99427da786cbadbf08c34a8ebf0b",
    "agg_results": "69462b6d8e5da17a782736beba872173e36ed0fda07ce4066c3bf2c4def52649",
}
REGISTERED_AXIS_INPUTS = {
    "target_manifest": (
        Path("dataset/public_v51/split_manifest.csv"),
        "7ddbd3b992f7f4630fd4c9b12263c62866a0ab2f7180ccad59a986eea91847d4",
    ),
    "gene_manifest": (
        Path("dataset/public_v51/common_genes.csv"),
        "3278051032a01f94f8a38153e3ae2e19cae6e7673e3186d3b78e2d3bd1e46b86",
    ),
}
A0_REPRODUCTION_ATOL = 1e-9
A1_ROUNDED_ATOL = 5e-7
ORACLE_ATOL = 1e-12
GATE_FLOAT_ATOL = 1e-12
# The independent scorer agreement is intentionally tighter than display rounding.
SCORER_CROSSCHECK_ATOL = pds.CROSS_CHECK_ATOL


class PDT1Error(RuntimeError):
    """A fail-closed PDT-1 precondition was not satisfied."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PDT1Error(message)


@dataclass(frozen=True)
class AuthenticatedFile:
    """Identity captured while hashing one regular, non-symlink input."""

    path: Path
    size_bytes: int
    sha256: str
    device: int
    inode: int
    mtime_ns: int

    def as_receipt(self, repository_root: Path) -> dict[str, Any]:
        return {
            "path": pds.repo_relative(self.path, repository_root),
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class EffectAtlas:
    effects: np.ndarray
    target_names: tuple[str, ...]
    gene_names: tuple[str, ...]
    npz_keys: tuple[str, ...]


@dataclass(frozen=True)
class NonOracleArms:
    """A0--A4 plus their public-only construction audit."""

    means: Mapping[str, np.ndarray]
    audit: Mapping[str, Any]


def _normalise_sha256(value: str, label: str) -> str:
    rendered = value.strip().lower()
    require(
        len(rendered) == 64
        and all(character in "0123456789abcdef" for character in rendered),
        f"{label} must be a 64-character hexadecimal SHA-256",
    )
    return rendered


def authenticate_regular_file(
    path: Path,
    expected_sha256: str,
    *,
    label: str,
) -> AuthenticatedFile:
    """Hash one descriptor while refusing symlinks and non-regular inputs."""

    resolved = Path(os.path.abspath(os.fspath(path.expanduser())))
    expected = _normalise_sha256(expected_sha256, f"expected {label} hash")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(resolved, flags)
    except OSError as error:
        raise PDT1Error(f"cannot open authenticated {label}: {resolved}: {error}") from error
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), f"{label} is not a regular file: {resolved}")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    require(
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        f"{label} changed while it was being hashed: {resolved}",
    )
    actual = digest.hexdigest()
    require(actual == expected, f"{label} SHA-256 mismatch: expected {expected}, found {actual}")
    return AuthenticatedFile(
        path=resolved,
        size_bytes=int(after.st_size),
        sha256=actual,
        device=int(after.st_dev),
        inode=int(after.st_ino),
        mtime_ns=int(after.st_mtime_ns),
    )


def require_unchanged(record: AuthenticatedFile, *, label: str) -> None:
    """Reject path replacement or mutation between authentication and consumption."""

    try:
        current = record.path.lstat()
    except OSError as error:
        raise PDT1Error(f"authenticated {label} disappeared: {record.path}") from error
    require(not stat.S_ISLNK(current.st_mode), f"authenticated {label} became a symlink")
    require(stat.S_ISREG(current.st_mode), f"authenticated {label} is no longer regular")
    require(
        (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns)
        == (record.device, record.inode, record.size_bytes, record.mtime_ns),
        f"authenticated {label} changed after hashing: {record.path}",
    )


def axis_sha256(values: Sequence[str]) -> str:
    """Canonical hash of an ordered string axis, including its boundaries."""

    encoded = json.dumps(
        [str(value) for value in values], ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(b"vcc-ordered-string-axis-v1\n" + encoded).hexdigest()


def _string_axis(values: np.ndarray, label: str) -> tuple[str, ...]:
    array = np.asarray(values)
    require(array.ndim == 1, f"{label} must be one-dimensional")
    rendered = tuple(array.astype(str).tolist())
    require(bool(rendered), f"{label} is empty")
    require(all(value.strip() == value and value for value in rendered), f"{label} has blanks")
    require(len(rendered) == len(set(rendered)), f"{label} has duplicate identifiers")
    return rendered


def _validate_bulk(
    perturbations: Sequence[str],
    means: np.ndarray,
    genes: Sequence[str],
    *,
    label: str,
) -> tuple[tuple[str, ...], np.ndarray, tuple[str, ...]]:
    perts = tuple(str(value) for value in perturbations)
    gene_axis = tuple(str(value) for value in genes)
    values = np.asarray(means, dtype=np.float64)
    require(bool(perts) and len(perts) == len(set(perts)), f"{label} perturbations are invalid")
    require(
        bool(gene_axis) and len(gene_axis) == len(set(gene_axis)),
        f"{label} gene axis is invalid",
    )
    require(
        values.shape == (len(perts), len(gene_axis)),
        f"{label} pseudobulk shape {values.shape} does not match its axes",
    )
    require(np.isfinite(values).all(), f"{label} pseudobulk contains non-finite values")
    require(perts.count(pds.CONTROL_LABEL) == 1, f"{label} must contain one control row")
    return perts, values, gene_axis


def _control_and_targets(
    perturbations: Sequence[str],
) -> tuple[int, np.ndarray, tuple[str, ...]]:
    perts = tuple(str(value) for value in perturbations)
    control = perts.index(pds.CONTROL_LABEL)
    target_rows = np.asarray([index for index in range(len(perts)) if index != control])
    targets = tuple(perts[index] for index in target_rows)
    return control, target_rows, targets


def read_authenticated_csv_axis(
    record: AuthenticatedFile,
    column: str,
    *,
    label: str,
) -> tuple[str, ...]:
    """Read one named axis column from an already authenticated CSV."""

    require_unchanged(record, label=label)
    try:
        with record.path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            require(reader.fieldnames is not None, f"{label} has no CSV header")
            require(column in reader.fieldnames, f"{label} is missing column {column!r}")
            values = tuple(str(row[column]) for row in reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise PDT1Error(f"cannot parse authenticated {label}: {error}") from error
    require_unchanged(record, label=label)
    require(bool(values), f"{label} axis is empty")
    require(all(value and value.strip() == value for value in values), f"{label} has blanks")
    require(len(values) == len(set(values)), f"{label} has duplicate identifiers")
    return values


def _validate_atlas_receipt(
    receipt: Mapping[str, Any],
    atlas_record: AuthenticatedFile,
    *,
    npz_keys: Sequence[str],
    targets: int,
    genes: int,
) -> None:
    require(receipt.get("schema") == ATLAS_SCHEMA, "K562 atlas receipt schema mismatch")
    output = receipt.get("output")
    contract = receipt.get("contract")
    require(isinstance(output, Mapping), "K562 atlas receipt has no output descriptor")
    require(isinstance(contract, Mapping), "K562 atlas receipt has no contract")
    require(
        str(output.get("sha256", "")).lower() == atlas_record.sha256,
        "K562 atlas receipt does not bind the supplied NPZ hash",
    )
    require(int(output.get("bytes", -1)) == atlas_record.size_bytes, "K562 atlas size mismatch")
    require(
        tuple(output.get("npz_keys", ())) == tuple(npz_keys),
        "K562 atlas NPZ key order differs from its receipt",
    )
    require(
        contract.get("effect_space") == ATLAS_EFFECT_SPACE,
        "K562 atlas effect-space contract is not the registered log-fold space",
    )
    require(
        float(contract.get("normalization_target_sum", -1.0)) == pds.BULK_TARGET_SUM,
        "K562 atlas normalization target is not CP50000",
    )
    require(int(contract.get("targets", -1)) == targets, "K562 atlas target count mismatch")
    require(int(contract.get("genes", -1)) == genes, "K562 atlas gene count mismatch")


def load_authenticated_atlas(
    atlas_record: AuthenticatedFile,
    receipt_record: AuthenticatedFile,
) -> tuple[EffectAtlas, Mapping[str, Any]]:
    """Load and cross-bind the bulk-atlas artifact registered for PDT-1."""

    require_unchanged(atlas_record, label="K562 atlas")
    with np.load(atlas_record.path, allow_pickle=False) as payload:
        keys = tuple(payload.files)
        require(keys == ATLAS_REQUIRED_KEYS, "K562 atlas has an unexpected NPZ schema")
        target_names = _string_axis(payload["target_names"], "K562 atlas target axis")
        gene_names = _string_axis(payload["gene_names"], "K562 atlas gene axis")
        # Preserve the producer's float32 storage; assignment into float64
        # pseudobulks widens only the mapped block and avoids doubling this atlas.
        effects = np.asarray(payload["effects"], dtype=np.float32)
        require(
            effects.shape == (len(target_names), len(gene_names)),
            "K562 atlas effect matrix does not match its axes",
        )
        require(np.isfinite(effects).all(), "K562 atlas effects contain non-finite values")
        controls = np.asarray(payload["matched_control_profiles"])
        counts = np.asarray(payload["target_cell_counts"])
        batches = _string_axis(payload["batch_names"], "K562 atlas batch axis")
        target_batches = np.asarray(payload["target_batch_counts"])
        control_batches = np.asarray(payload["control_batch_counts"])
        target_sum = np.asarray(payload["target_sum"], dtype=np.float64).reshape(-1)
        require(controls.shape == effects.shape, "K562 matched controls have the wrong shape")
        require(counts.shape == (len(target_names),), "K562 target counts have the wrong shape")
        require(
            target_batches.shape == (len(target_names), len(batches)),
            "K562 target-by-batch counts have the wrong shape",
        )
        require(
            control_batches.shape == (len(batches),),
            "K562 control-by-batch counts have the wrong shape",
        )
        require(
            target_sum.shape == (1,) and float(target_sum[0]) == pds.BULK_TARGET_SUM,
            "K562 atlas target_sum is not exactly CP50000",
        )
    require_unchanged(atlas_record, label="K562 atlas")

    require_unchanged(receipt_record, label="K562 atlas receipt")
    try:
        with receipt_record.path.open("r", encoding="utf-8") as handle:
            receipt = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PDT1Error(f"cannot parse K562 atlas receipt: {error}") from error
    require(isinstance(receipt, Mapping), "K562 atlas receipt must be a JSON object")
    require_unchanged(receipt_record, label="K562 atlas receipt")
    _validate_atlas_receipt(
        receipt,
        atlas_record,
        npz_keys=keys,
        targets=len(target_names),
        genes=len(gene_names),
    )
    return EffectAtlas(effects, target_names, gene_names, keys), receipt


def build_non_oracle_arms(
    perturbations: Sequence[str],
    genes: Sequence[str],
    g100_means: np.ndarray,
    atlas: EffectAtlas,
) -> NonOracleArms:
    """Construct A0--A4 without accepting or accessing HepG2 truth.

    A2 replaces each available target's gene-mapped *trans* coordinates with
    its K562 log-fold effect. Every panel target gene is excluded from transfer,
    matching the scorer's panel exclusion. Absent targets and genes retain g100.
    A4 permutes source assignments over the sorted same-target overlap.
    """

    perts, baseline, gene_axis = _validate_bulk(perturbations, g100_means, genes, label="g100")
    control, target_rows, targets = _control_and_targets(perts)
    scoring_keep = pds.panel_excluded_mask(gene_axis, targets)
    source_gene_lookup = {name: index for index, name in enumerate(atlas.gene_names)}
    destination_columns = np.asarray(
        [
            index
            for index, gene in enumerate(gene_axis)
            if scoring_keep[index] and gene in source_gene_lookup
        ],
        dtype=np.int64,
    )
    source_columns = np.asarray(
        [source_gene_lookup[gene_axis[index]] for index in destination_columns],
        dtype=np.int64,
    )
    require(destination_columns.size > 0, "K562 atlas has no mapped trans genes")

    source_target_lookup = {name: index for index, name in enumerate(atlas.target_names)}
    direct_targets = tuple(sorted(set(targets).intersection(source_target_lookup)))
    require(len(direct_targets) >= 2, "K562 atlas needs at least two same-target overlaps")
    destination_row_lookup = {name: perts.index(name) for name in direct_targets}

    a0 = baseline.copy()
    control_profile = a0[control].copy()
    baseline_effects = a0[target_rows] - control_profile[None, :]
    a1_effects = pds.remove_shared(
        baseline_effects,
        pds.shared_direction(baseline_effects),
        SHARED_REMOVAL_FRACTION,
    )
    a1 = a0.copy()
    a1[target_rows] = control_profile[None, :] + a1_effects
    a1[control] = control_profile

    def transfer(assignments: Mapping[str, str]) -> np.ndarray:
        arm = a0.copy()
        for destination_target, source_target in assignments.items():
            row = destination_row_lookup[destination_target]
            source_row = source_target_lookup[source_target]
            arm[row, destination_columns] = (
                control_profile[destination_columns]
                + atlas.effects[source_row, source_columns]
            )
        arm[control] = control_profile
        return arm

    a2 = transfer({target: target for target in direct_targets})
    a2_effects = a2[target_rows] - control_profile[None, :]
    a3_effects = pds.remove_shared(
        a2_effects,
        pds.shared_direction(a2_effects),
        SHARED_REMOVAL_FRACTION,
    )
    a3 = a2.copy()
    a3[target_rows] = control_profile[None, :] + a3_effects
    a3[control] = control_profile

    permutation = np.random.default_rng(SHUFFLE_SEED).permutation(len(direct_targets))
    shuffled_assignment = {
        target: direct_targets[int(permutation[index])]
        for index, target in enumerate(direct_targets)
    }
    require(
        any(target != source for target, source in shuffled_assignment.items()),
        "registered target shuffle unexpectedly produced the identity permutation",
    )
    a4 = transfer(shuffled_assignment)

    for arm_name, values in (("A0", a0), ("A1", a1), ("A2", a2), ("A3", a3), ("A4", a4)):
        require(np.isfinite(values).all(), f"{arm_name} contains non-finite values")
        require(
            np.array_equal(values[control], control_profile),
            f"{arm_name} changed the g100 control pseudobulk",
        )

    arm_means = {"A0": a0, "A1": a1, "A2": a2, "A3": a3, "A4": a4}
    for values in arm_means.values():
        values.flags.writeable = False

    return NonOracleArms(
        means=MappingProxyType(arm_means),
        audit={
            "truth_available_to_constructor": False,
            "non_oracle_arrays_write_protected": True,
            "allowed_inputs": [
                "authenticated_g100_pseudobulk",
                "authenticated_k562_gwps_effect_atlas",
                "registered_constants",
            ],
            "shared_removal_fraction": SHARED_REMOVAL_FRACTION,
            "shared_direction_space": "full_registered_7107_gene_axis",
            "A1_shared_direction_fitted_from": "A0_target_effects_only",
            "A3_shared_direction_fitted_from": "A2_target_effects_only",
            "shuffle_seed": SHUFFLE_SEED,
            "shuffle_algorithm": (
                "numpy.default_rng(PCG64).permutation_over_lexicographically_sorted_"
                "same_target_overlap"
            ),
            "direct_target_count": len(direct_targets),
            "direct_targets_sha256": axis_sha256(direct_targets),
            "mapped_trans_gene_count": int(destination_columns.size),
            "mapped_trans_genes_sha256": axis_sha256(
                tuple(gene_axis[index] for index in destination_columns)
            ),
            "panel_target_gene_coordinates_transferred": 0,
            "trans_definition": "exclude_every_panel_target_gene",
            "source_effect_scaling": "none_copy_authenticated_K562_atlas_values",
            "source_effect_clipping": "none_pseudobulk_direction_test_only",
            "recipient_control": "g100_control_later_required_bit_identical_to_sealed_truth",
            "unmapped_targets_retain_g100": len(targets) - len(direct_targets),
            "unmapped_genes_retain_g100": len(gene_axis) - int(destination_columns.size),
            "a2_assignment": "same_target",
            "a4_assignment_sha256": axis_sha256(
                tuple(f"{target}->{shuffled_assignment[target]}" for target in direct_targets)
            ),
            "a4_fixed_point_count": sum(
                target == shuffled_assignment[target] for target in direct_targets
            ),
        },
    )


def build_oracle_arm(real_means: np.ndarray) -> np.ndarray:
    """Construct A5 separately; this function is never called by A0--A4."""

    values = np.asarray(real_means, dtype=np.float64)
    require(values.ndim == 2 and np.isfinite(values).all(), "oracle input is invalid")
    return values.copy()


def independent_arm_metrics(
    predicted_means: np.ndarray,
    real_means: np.ndarray,
    perturbations: Sequence[str],
    genes: Sequence[str],
) -> dict[str, Any]:
    """Re-derive PDS and the registered anti-correlation statistic."""

    perts, predicted, gene_axis = _validate_bulk(
        perturbations, predicted_means, genes, label="predicted arm"
    )
    real_perts, real, real_gene_axis = _validate_bulk(
        perturbations, real_means, genes, label="real"
    )
    require(perts == real_perts and gene_axis == real_gene_axis, "scoring axes differ")
    control, target_rows, targets = _control_and_targets(perts)
    keep = pds.panel_excluded_mask(gene_axis, targets)
    require(int(keep.sum()) > 0, "panel exclusion removed every scoring gene")
    real_effects = (real - real[control])[target_rows][:, keep]
    predicted_effects = (predicted - real[control])[target_rows][:, keep]
    similarity = pds.cosine_similarity_matrix(predicted_effects, real_effects)
    matched = np.diag(similarity)
    structure = pds.similarity_structure(similarity)
    separation = structure.separation_z
    return {
        "independent_pds": structure.pds,
        "ranked_gene_count": int(keep.sum()),
        "matched_cosine": {
            "mean": float(matched.mean()),
            "median": float(np.median(matched)),
            "minimum": float(matched.min()),
            "maximum": float(matched.max()),
            "anti_correlated_count": int((matched <= 0.0).sum()),
            "anti_correlated_fraction": float((matched <= 0.0).mean()),
        },
        "similarity_structure": {
            "matched_sd": structure.matched_sd,
            "mismatched_mean": structure.mismatched_mean,
            "mismatched_sd": structure.mismatched_sd,
            "separation_z": None if not np.isfinite(separation) else float(separation),
            "mean_competitors_nearer": structure.mean_competitors_nearer,
        },
    }


def score_arm(
    predicted_means: np.ndarray,
    real_means: np.ndarray,
    perturbations: Sequence[str],
    genes: Sequence[str],
    scorer: Callable[[np.ndarray], float],
) -> dict[str, Any]:
    """Score with cell-eval2 and require the diagnostic's independent agreement."""

    scorer_value = float(scorer(np.asarray(predicted_means, dtype=np.float64)))
    require(np.isfinite(scorer_value), "cell-eval2 returned a non-finite PDS")
    require(0.0 <= scorer_value <= 1.0, "cell-eval2 returned PDS outside [0, 1]")
    metrics = independent_arm_metrics(predicted_means, real_means, perturbations, genes)
    independent = float(metrics["independent_pds"])
    require(
        abs(scorer_value - independent) <= SCORER_CROSSCHECK_ATOL,
        f"cell-eval2 PDS {scorer_value!r} disagrees with independent {independent!r}",
    )
    return {"pds_cosine": scorer_value, **metrics}


def registered_decision(arm_metrics: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Apply all three preregistered gates to A2 and A3."""

    require(all(name in arm_metrics for name in ("A2", "A3", "A4")), "gate arms missing")
    shuffled = float(arm_metrics["A4"]["pds_cosine"])
    evaluations: dict[str, Any] = {}
    for arm in ("A2", "A3"):
        pds_value = float(arm_metrics[arm]["pds_cosine"])
        matched = arm_metrics[arm].get("matched_cosine")
        require(isinstance(matched, Mapping), f"{arm} matched-cosine metrics missing")
        anti_fraction = float(matched["anti_correlated_fraction"])
        margin = pds_value - shuffled
        checks = {
            "pds_at_least_0_6263": pds_value + GATE_FLOAT_ATOL >= REGISTERED_PDS_FLOOR,
            "pds_at_least_a4_plus_0_0400": margin + GATE_FLOAT_ATOL >= REGISTERED_SHUFFLE_MARGIN,
            "anti_correlated_fraction_below_0_300": (
                anti_fraction < REGISTERED_ANTICORRELATED_CEILING
            ),
        }
        evaluations[arm] = {
            "pds_cosine": pds_value,
            "margin_over_A4": margin,
            "anti_correlated_fraction": anti_fraction,
            "checks": checks,
            "passed": all(checks.values()),
        }
    winners = [arm for arm in ("A2", "A3") if evaluations[arm]["passed"]]
    return {
        "status": "PASS" if winners else "FAIL",
        "passed": bool(winners),
        "passing_arms": winners,
        "registered_thresholds": {
            "pds_cosine_minimum": REGISTERED_PDS_FLOOR,
            "margin_over_A4_minimum": REGISTERED_SHUFFLE_MARGIN,
            "anti_correlated_fraction_strict_maximum": REGISTERED_ANTICORRELATED_CEILING,
        },
        "arm_evaluations": evaluations,
    }


def validate_registered_reproductions(
    arm_metrics: Mapping[str, Mapping[str, Any]],
    recorded_a0: float,
) -> dict[str, Any]:
    """Fail before interpretation if A0, A1, or A5 no longer matches registration."""

    values = {name: float(arm_metrics[name]["pds_cosine"]) for name in ("A0", "A1", "A5")}
    checks = {
        "agg_results_records_registered_A0": (
            abs(float(recorded_a0) - REGISTERED_A0_PDS) <= A0_REPRODUCTION_ATOL
        ),
        "A0_matches_agg_results": (
            abs(values["A0"] - float(recorded_a0)) <= A0_REPRODUCTION_ATOL
        ),
        "A0_matches_registered_value": (
            abs(values["A0"] - REGISTERED_A0_PDS) <= A0_REPRODUCTION_ATOL
        ),
        "A1_matches_registered_rounded_expectation": (
            abs(values["A1"] - REGISTERED_A1_PDS) <= A1_ROUNDED_ATOL
        ),
        "A5_is_exact_oracle_ceiling": abs(values["A5"] - 1.0) <= ORACLE_ATOL,
    }
    require(all(checks.values()), f"registered PDT-1 reproduction failed: {checks}")
    return {
        "checks": checks,
        "recorded_A0_pds": float(recorded_a0),
        "expected": {"A0": REGISTERED_A0_PDS, "A1_rounded": REGISTERED_A1_PDS, "A5": 1.0},
        "observed": values,
        "tolerances": {
            "A0": A0_REPRODUCTION_ATOL,
            "A1_rounded": A1_ROUNDED_ATOL,
            "A5": ORACLE_ATOL,
        },
    }


def write_json_no_overwrite(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically publish JSON while refusing pre-existing paths and races."""

    destination = Path(os.path.abspath(os.fspath(path.expanduser())))
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite PDT-1 receipt: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{secrets.token_hex(12)}.tmp"
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(temporary, flags, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination, follow_symlinks=False)
        except FileExistsError as error:
            raise FileExistsError(f"refusing to overwrite PDT-1 receipt: {destination}") from error
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _resolve(path: Path, repository_root: Path) -> Path:
    candidate = path if path.is_absolute() else repository_root / path
    return Path(os.path.abspath(os.fspath(candidate.expanduser())))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--prediction-view", type=Path, required=True, help="g100 H5AD")
    parser.add_argument("--prediction-view-sha256", required=True)
    parser.add_argument("--truth-view", type=Path, required=True, help="sealed HepG2 H5AD")
    parser.add_argument("--truth-view-sha256", required=True)
    parser.add_argument(
        "--agg-results", type=Path, required=True, help="original g100 agg_results.csv"
    )
    parser.add_argument("--agg-results-sha256", required=True)
    parser.add_argument("--k562-atlas", type=Path, required=True)
    parser.add_argument("--k562-atlas-sha256", required=True)
    parser.add_argument("--k562-atlas-receipt", type=Path, required=True)
    parser.add_argument("--k562-atlas-receipt-sha256", required=True)
    parser.add_argument("--cell-eval-checkout", type=Path, default=Path("external/cell-eval2"))
    parser.add_argument("--expected-cell-eval-commit", required=True)
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    repository_root = Path(os.path.abspath(os.fspath(args.repo_root.expanduser())))
    require(repository_root.is_dir(), f"repository root is missing: {repository_root}")
    output = _resolve(args.output_json, repository_root)
    require(not output.exists() and not output.is_symlink(), f"refusing to overwrite: {output}")
    require(
        str(args.candidate_label) == REGISTERED_CANDIDATE,
        "candidate label is not registered g100",
    )
    require(
        str(args.expected_cell_eval_commit).lower() == REGISTERED_CELL_EVAL_COMMIT,
        "cell-eval2 commit is not the registered scorer commit",
    )

    supplied_registered_hashes = {
        "prediction_view": _normalise_sha256(
            args.prediction_view_sha256, "expected prediction_view hash"
        ),
        "truth_view": _normalise_sha256(args.truth_view_sha256, "expected truth_view hash"),
        "agg_results": _normalise_sha256(args.agg_results_sha256, "expected agg_results hash"),
    }
    require(
        supplied_registered_hashes == REGISTERED_INPUT_SHA256,
        "g100 scorer inputs do not match the registered diagnostic identities",
    )

    paths = {
        "prediction_view": _resolve(args.prediction_view, repository_root),
        "truth_view": _resolve(args.truth_view, repository_root),
        "agg_results": _resolve(args.agg_results, repository_root),
        "k562_atlas": _resolve(args.k562_atlas, repository_root),
        "k562_atlas_receipt": _resolve(args.k562_atlas_receipt, repository_root),
        **{
            name: _resolve(relative_path, repository_root)
            for name, (relative_path, _) in REGISTERED_AXIS_INPUTS.items()
        },
    }
    expected_hashes = {
        "prediction_view": args.prediction_view_sha256,
        "truth_view": args.truth_view_sha256,
        "agg_results": args.agg_results_sha256,
        "k562_atlas": args.k562_atlas_sha256,
        "k562_atlas_receipt": args.k562_atlas_receipt_sha256,
        **{name: digest for name, (_, digest) in REGISTERED_AXIS_INPUTS.items()},
    }
    records = {
        name: authenticate_regular_file(path, expected_hashes[name], label=name)
        for name, path in paths.items()
    }

    checkout = _resolve(args.cell_eval_checkout, repository_root)
    cell_eval_commit = pds.authenticate_cell_eval(checkout, args.expected_cell_eval_commit)
    sys.path.insert(0, str(checkout / "src"))
    import anndata as ad  # noqa: PLC0415 - deferred until authentication
    from cell_eval2.metrics.discrimination import discrimination_score  # noqa: PLC0415
    from cell_eval2.prep import pseudobulk_bulk_lognorm  # noqa: PLC0415

    def load_bulk(
        record: AuthenticatedFile, label: str
    ) -> tuple[tuple[str, ...], np.ndarray, tuple[str, ...]]:
        require_unchanged(record, label=label)
        data = ad.read_h5ad(record.path)
        try:
            perturbations, means = pseudobulk_bulk_lognorm(
                data, pds.PERT_COL, bulk_target_sum=pds.BULK_TARGET_SUM
            )
            genes = data.var_names.to_numpy().astype(str)
        finally:
            del data
        require_unchanged(record, label=label)
        return _validate_bulk(perturbations, means, genes, label=label)

    # No truth values exist in the A0--A4 constructor's call graph or signature.
    pred_perts, pred_means, pred_genes = load_bulk(records["prediction_view"], "g100")
    _, _, pred_targets = _control_and_targets(pred_perts)
    registered_targets = read_authenticated_csv_axis(
        records["target_manifest"], "target_gene", label="registered target manifest"
    )
    registered_genes = read_authenticated_csv_axis(
        records["gene_manifest"], "gene_name", label="registered gene manifest"
    )
    require(len(pred_targets) == REGISTERED_TARGET_COUNT, "g100 is not the 300-target panel")
    require(
        len(pred_genes) == REGISTERED_GENE_COUNT,
        "g100 is not on the registered 7,107-gene scoring axis",
    )
    require(
        pred_targets == tuple(sorted(registered_targets)),
        "g100 perturbation axis differs from the registered panel",
    )
    require(
        pred_genes == registered_genes,
        "g100 gene axis differs from the registered common axis",
    )
    atlas, atlas_receipt = load_authenticated_atlas(
        records["k562_atlas"], records["k562_atlas_receipt"]
    )
    non_oracle = build_non_oracle_arms(pred_perts, pred_genes, pred_means, atlas)

    # Truth enters only after A0--A4 are complete; A5 has an isolated constructor.
    real_perts, real_means, real_genes = load_bulk(records["truth_view"], "sealed HepG2 truth")
    require(real_perts == pred_perts, "g100 and truth perturbation axes differ")
    require(real_genes == pred_genes, "g100 and truth gene axes differ")
    control, _, _ = _control_and_targets(real_perts)
    require(
        np.array_equal(pred_means[control], real_means[control]),
        "g100 control pseudobulk is not bit-identical to sealed truth control",
    )
    oracle = build_oracle_arm(real_means)

    def scorer(means: np.ndarray) -> float:
        per_target = discrimination_score(
            pred_bulk=(np.asarray(real_perts), means),
            real_bulk=(np.asarray(real_perts), real_means),
            genes=np.asarray(real_genes),
            pert_col=pds.PERT_COL,
            control=pds.CONTROL_LABEL,
            **pds.DISCRIMINATION_KWARGS,
        )
        values = np.asarray(list(per_target.values()), dtype=np.float64)
        require(
            values.size == REGISTERED_TARGET_COUNT,
            "cell-eval2 returned the wrong target count",
        )
        require(np.isfinite(values).all(), "cell-eval2 returned non-finite target PDS")
        return float(values.mean())

    arm_arrays = {**non_oracle.means, "A5": oracle}
    arm_metrics = {
        name: score_arm(values, real_means, real_perts, real_genes, scorer)
        for name, values in arm_arrays.items()
    }

    require_unchanged(records["agg_results"], label="g100 agg_results")
    recorded_a0 = pds.recorded_scorer_pds(records["agg_results"].path)
    require_unchanged(records["agg_results"], label="g100 agg_results")
    reproduction = validate_registered_reproductions(arm_metrics, recorded_a0)
    decision = registered_decision(arm_metrics)

    for name, record in records.items():
        require_unchanged(record, label=name)
    git_commit, git_clean = pds.git_head(repository_root)
    next_route = repository_root / "docs" / "research" / "NEXT_ROUTE_DECISION.md"
    diagnostic = repository_root / "scripts" / "diagnose_pds_reachability.py"
    implementation = Path(__file__).resolve()
    require(
        next_route.is_file() and diagnostic.is_file() and implementation.is_file(),
        "registered PDT-1 sources are missing",
    )
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "completed",
        "experiment": "PDT-1_public-transfer_direction_test",
        "candidate": str(args.candidate_label),
        "scope": {
            "cpu_only": True,
            "pseudobulk_only": True,
            "candidate_artifact_written": False,
            "single_cell_counts_rendered": False,
            "official_submission_allowed": False,
        },
        "registration": {
            "source": {
                "path": pds.repo_relative(next_route, repository_root),
                "sha256": pds.sha256_file(next_route),
            },
            "arms": {
                "A0": "g100_as_scored",
                "A1": "g100_shared_component_removed_fraction_1.05",
                "A2": "same_target_K562_GWPS_trans_effect_on_HepG2_axis_g100_elsewhere",
                "A3": "A2_shared_component_removed_fraction_1.05",
                "A4": "A2_target_to_effect_assignment_permuted_seed_20260901",
                "A5": "sealed_truth_oracle_ceiling",
            },
        },
        "inputs": {name: record.as_receipt(repository_root) for name, record in records.items()},
        "scorer": {
            "checkout": pds.repo_relative(checkout, repository_root),
            "commit": cell_eval_commit,
            "preset": "vcc2026",
            "bulk_target_sum": pds.BULK_TARGET_SUM,
            "discrimination": dict(pds.DISCRIMINATION_KWARGS),
            "diagnostic_source": {
                "path": pds.repo_relative(diagnostic, repository_root),
                "sha256": pds.sha256_file(diagnostic),
            },
            "every_arm_independently_rederived": True,
        },
        "axes": {
            "perturbations_including_control": len(real_perts),
            "targets": len(pred_targets),
            "genes": len(real_genes),
            "ranked_genes_after_panel_exclusion": arm_metrics["A0"]["ranked_gene_count"],
            "perturbation_axis_sha256": axis_sha256(real_perts),
            "target_axis_sha256": axis_sha256(pred_targets),
            "gene_axis_sha256": axis_sha256(real_genes),
            "prediction_truth_axes_identical": True,
            "prediction_axes_match_authenticated_manifests": True,
            "prediction_truth_control_pseudobulk_bit_identical": True,
            "ordered_axis_hash_schema": "vcc-ordered-string-axis-v1",
            "k562_target_axis_sha256": axis_sha256(atlas.target_names),
            "k562_gene_axis_sha256": axis_sha256(atlas.gene_names),
        },
        "atlas_contract": {
            "receipt_schema": atlas_receipt["schema"],
            "effect_space": atlas_receipt["contract"]["effect_space"],
            "normalization_target_sum": atlas_receipt["contract"]["normalization_target_sum"],
            "npz_keys": list(atlas.npz_keys),
        },
        "construction": {
            "non_oracle": dict(non_oracle.audit),
            "oracle": {
                "arm": "A5",
                "constructed_in_separate_function": True,
                "constructed_after_A0_through_A4": True,
                "truth_values_exposed_to_non_oracle_constructor": False,
            },
        },
        "arms": arm_metrics,
        "reproduction": reproduction,
        "decision": decision,
        "provenance": {
            "git_commit": git_commit,
            "git_clean": git_clean,
            "implementation_source": {
                "path": pds.repo_relative(implementation, repository_root),
                "sha256": pds.sha256_file(implementation),
            },
            "python": platform.python_version(),
            "numpy": np.__version__,
            "argv": sys.argv,
        },
    }
    json.dumps(receipt, sort_keys=True, allow_nan=False)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repository_root = Path(os.path.abspath(os.fspath(args.repo_root.expanduser())))
    output = _resolve(args.output_json, repository_root)
    receipt = run(args)
    write_json_no_overwrite(output, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
