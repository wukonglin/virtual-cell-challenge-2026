#!/usr/bin/env python3
"""Diagnose how much perturbation-discrimination headroom a scored candidate leaves.

``pds_cosine`` is the largest single contributor to the official VCC 2026 overall
score. It consumes only the per-group count-summed ``bulk_lognorm`` pseudobulk and
ranks each target's predicted signed effect against every target's real effect by
cosine distance. Two consequences follow directly from that definition and are
measured here rather than assumed:

1. a pure amplitude rescale of the predicted effect is exactly inert, because
   cosine distance is scale free per row;
2. no transformation that preserves each group's per-gene integer count sum can
   move the metric at all.

This module therefore separates two very different failure modes that produce the
same low score: *missing or dispersed* per-target signal, and *structured,
target-correlated error* that leaves the 300 predicted profiles nearly interchangeable
under ranking. It distinguishes them with two simulated references that both carry the
candidate's own directional accuracy but have perfectly independent error:

- a HOMOGENEOUS reference, where every target gets the candidate's mean accuracy;
- a HETEROGENEOUS reference, where each target gets its own signed matched cosine,
  so targets the candidate predicts backwards stay backwards.

The homogeneous reference alone is misleading: it discards the dispersion of per-target
accuracy, which for a real candidate is large. The gap between the two references is the
share of the shortfall attributable to per-target signal deficit; the gap between the
heterogeneous reference and the candidate is the share attributable to error structure.
Both must be reported, and the accuracy must be measured in the space the scorer
actually ranks — after the panel target genes are removed, not before.

The diagnostic is read only. It authenticates the pinned cell-eval2 checkout and
uses that checkout's own pseudobulk and discrimination implementations, so the
reproduced value is the scorer's rather than a re-implementation. A second,
independent from-scratch ranking is computed as a cross-check and both must agree.

It creates no candidate, renders no counts, and authorizes nothing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

RECEIPT_SCHEMA = "vcc-pds-reachability-diagnostic-v1"
BULK_TARGET_SUM = 50_000.0
CONTROL_LABEL = "non-targeting"
PERT_COL = "target_gene"
DEFAULT_SEED = 20260901
# Three registered seeds; the simulated references are reported per seed and as a mean,
# because a single draw of an isotropic error field is not a measurement.
REFERENCE_SEED_OFFSETS = (0, 1, 2)
# The vcc2026 preset's discrimination block, restated so that reading this file tells
# you what is being reproduced. tests/test_diagnose_pds_reachability.py fails if these
# drift from external/cell-eval2/src/cell_eval2/configs/vcc2026.yaml.
DISCRIMINATION_KWARGS: dict[str, Any] = {
    "distance": "cosine",
    "rank_denominator": "n-1",
    "tie_policy": "midrank",
    "exclude_target_gene": True,
    "exclusion_scope": "panel",
    "control_source": "real",
}
# Agreement required between the scorer's own value, the independent re-derivation,
# and the value recorded by the original scoring run.
REPRODUCTION_ATOL = 1e-9
CROSS_CHECK_ATOL = 1e-12
# Below this, the mismatched cosine spread is float noise rather than a measurement.
DEGENERATE_SPREAD_RTOL = 1e-12


class DiagnosticError(RuntimeError):
    """A precondition of the diagnostic failed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DiagnosticError(message)


# --------------------------------------------------------------------------------------
# Pure numeric core. No filesystem, no scorer import; unit tested in isolation.
# --------------------------------------------------------------------------------------


def unit_rows(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation that leaves an all-zero row at zero."""
    values = np.asarray(matrix, dtype=np.float64)
    _require(values.ndim == 2, "unit_rows expects a 2-D array")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    safe = np.where(norms == 0.0, 1.0, norms)
    return values / safe


def cosine_similarity_matrix(pred_effects: np.ndarray, real_effects: np.ndarray) -> np.ndarray:
    """``S[i, j] = cos(pred_effect_i, real_effect_j)``."""
    return unit_rows(pred_effects) @ unit_rows(real_effects).T


def midrank_pds(similarity: np.ndarray) -> np.ndarray:
    """Per-target ``1 - rank / (n - 1)`` from a square cosine-similarity matrix.

    Cosine distance is ``1 - similarity``, so a *larger* similarity is a *smaller*
    distance. The rank of the true match is the number of strictly nearer competitors
    plus half the number of exact ties, which is the mid-rank convention the v2 scorer
    uses: a fully tied row scores exactly 0.5 rather than an alphabetical index.
    """
    values = np.asarray(similarity, dtype=np.float64)
    _require(values.ndim == 2 and values.shape[0] == values.shape[1], "similarity must be square")
    n = values.shape[0]
    _require(n >= 2, "at least two perturbations are required")
    matched = np.diag(values)
    strictly_nearer = (values > matched[:, None]).sum(axis=1)
    tied = (values == matched[:, None]).sum(axis=1) - 1  # exclude the self cell
    rank = strictly_nearer.astype(np.float64) + 0.5 * tied.astype(np.float64)
    return 1.0 - rank / float(n - 1)


def shared_direction(effects: np.ndarray) -> np.ndarray:
    """The unit common-response direction of a set of per-target effects."""
    mean = np.asarray(effects, dtype=np.float64).mean(axis=0, keepdims=True)
    return unit_rows(mean)[0]


def remove_shared(effects: np.ndarray, direction: np.ndarray, fraction: float) -> np.ndarray:
    """Subtract ``fraction`` of each row's projection onto ``direction``."""
    values = np.asarray(effects, dtype=np.float64)
    axis = np.asarray(direction, dtype=np.float64)
    projection = (values @ axis)[:, None] * axis[None, :]
    return values - float(fraction) * projection


def effective_rank(matrix: np.ndarray) -> float:
    """Shannon effective rank of the squared singular-value spectrum."""
    spectrum = np.linalg.svd(np.asarray(matrix, dtype=np.float64), compute_uv=False)
    power = spectrum**2
    total = power.sum()
    if total <= 0.0:
        return 0.0
    share = power / total
    nonzero = share[share > 0.0]
    return float(np.exp(-(nonzero * np.log(nonzero)).sum()))


@dataclass(frozen=True)
class SimilarityStructure:
    """How separable the true match is from its competitors, in plain cosine terms."""

    matched_mean: float
    matched_sd: float
    mismatched_mean: float
    mismatched_sd: float
    separation_z: float
    mean_competitors_nearer: float
    pds: float

    def as_dict(self) -> dict[str, float]:
        return {
            "matched_mean": self.matched_mean,
            "matched_sd": self.matched_sd,
            "mismatched_mean": self.mismatched_mean,
            "mismatched_sd": self.mismatched_sd,
            "separation_z": self.separation_z,
            "mean_competitors_nearer": self.mean_competitors_nearer,
            "pds": self.pds,
        }


def similarity_structure(similarity: np.ndarray) -> SimilarityStructure:
    """Decompose a PDS score into the separation that produced it.

    ``separation_z`` is the gap between the matched and mismatched cosine means in
    units of the mismatched spread. It is the quantity that actually determines the
    rank, and it is what distinguishes a weak-signal candidate from one whose error is
    correlated across targets.
    """
    values = np.asarray(similarity, dtype=np.float64)
    n = values.shape[0]
    matched = np.diag(values)
    off_diagonal = values[~np.eye(n, dtype=bool)]
    mismatched_mean = float(off_diagonal.mean())
    mismatched_sd = float(off_diagonal.std())
    separation = float(matched.mean() - mismatched_mean)
    # A spread at float-noise level relative to the values themselves carries no
    # information, and dividing by it would report either a meaningless 0.0 or a
    # spurious enormous z. Declare it undefined instead.
    degenerate = mismatched_sd <= DEGENERATE_SPREAD_RTOL * max(1.0, abs(mismatched_mean))
    competitors = (off_diagonal.reshape(n, n - 1) > matched[:, None]).sum(axis=1)
    return SimilarityStructure(
        matched_mean=float(matched.mean()),
        matched_sd=float(matched.std()),
        mismatched_mean=mismatched_mean,
        mismatched_sd=mismatched_sd,
        separation_z=float("nan") if degenerate else separation / mismatched_sd,
        mean_competitors_nearer=float(competitors.mean()),
        pds=float(midrank_pds(values).mean()),
    )


def panel_excluded_mask(genes: Sequence[str], panel_targets: Iterable[str]) -> np.ndarray:
    """Boolean keep-mask over ``genes`` after removing every panel target gene.

    This mirrors ``exclusion_scope='panel'``: the panel's target genes leave the ranked
    feature space once, before any distance is computed, so a submission cannot score by
    anti-correlating with the other targets' knockdowns.
    """
    excluded = set(panel_targets)
    return np.array([gene not in excluded for gene in genes], dtype=bool)


def simulate_directional_prediction(
    real_effects: np.ndarray,
    accuracy: float | np.ndarray,
    generator: np.random.Generator,
) -> np.ndarray:
    """A prediction with the given directional accuracy and *perfectly independent* error.

    ``accuracy`` is either a scalar applied to every target (the HOMOGENEOUS reference)
    or one signed value per target (the HETEROGENEOUS reference). Signed values matter:
    a negative accuracy means the target is predicted anti-correlated with its own truth,
    and preserving that is what keeps the reference honest.

    A candidate scoring far below the *heterogeneous* reference has structured,
    target-correlated error. A candidate near it is simply short of per-target signal.
    """
    signal = unit_rows(real_effects)
    values = np.asarray(accuracy, dtype=np.float64)
    _require(bool(np.all(np.abs(values) <= 1.0)), "accuracy must lie in [-1, 1]")
    if values.ndim == 0:
        values = np.full((signal.shape[0],), float(values))
    _require(values.shape == (signal.shape[0],), "per-target accuracy must have one value per row")
    weights = values[:, None]
    noise = unit_rows(generator.standard_normal(signal.shape))
    return weights * signal + np.sqrt(np.clip(1.0 - weights * weights, 0.0, 1.0)) * noise


def top1_retrieval(similarity: np.ndarray) -> int:
    """How many targets have their own real effect as the single nearest competitor."""
    values = np.asarray(similarity, dtype=np.float64)
    return int((values.argmax(axis=1) == np.arange(values.shape[0])).sum())


def mean_pairwise_cosine(effects: np.ndarray) -> float:
    """Mean off-diagonal cosine among a set of effects: how collapsed the set is."""
    unit = unit_rows(effects)
    gram = unit @ unit.T
    upper = np.triu_indices(gram.shape[0], 1)
    return float(gram[upper].mean())


# --------------------------------------------------------------------------------------
# Provenance helpers.
# --------------------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def repo_relative(path: Path, repo_root: Path) -> str:
    """Repository-relative POSIX path, so tracked receipts carry no absolute path."""
    resolved = Path(os.path.abspath(path))
    try:
        return resolved.relative_to(repo_root).as_posix()
    except ValueError:
        raise DiagnosticError(f"path escapes the repository root: {path}") from None


def git_head(repo_root: Path) -> tuple[str, bool]:
    head = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=normal"],
        check=True, capture_output=True, text=True,
    ).stdout
    return head, status.strip() == ""


def authenticate_cell_eval(checkout: Path, expected_commit: str) -> str:
    _require(checkout.joinpath(".git").exists(), f"not a git checkout: {checkout}")
    actual = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    _require(
        actual == expected_commit,
        f"cell-eval2 commit mismatch: expected {expected_commit}, found {actual}",
    )
    dirty = subprocess.run(
        ["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=normal"],
        check=True, capture_output=True, text=True,
    ).stdout
    _require(dirty.strip() == "", "cell-eval2 checkout is dirty; refusing to score against it")
    return actual


def recorded_scorer_pds(agg_results_csv: Path) -> float:
    """The ``pds_cosine`` mean recorded by the original authenticated scoring run."""
    with open(agg_results_csv, newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("statistic") == "mean":
                value = row.get("pds_cosine")
                _require(value not in (None, ""), "agg_results.csv has no pds_cosine mean")
                return float(value)
    raise DiagnosticError("agg_results.csv has no 'mean' statistic row")


# --------------------------------------------------------------------------------------
# Orchestration.
# --------------------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--prediction-view", type=Path, required=True)
    parser.add_argument("--truth-view", type=Path, required=True)
    parser.add_argument("--agg-results", type=Path, required=True,
                        help="cell_eval2/agg_results.csv from the original scoring run")
    parser.add_argument("--cell-eval-checkout", type=Path, default=Path("external/cell-eval2"))
    parser.add_argument("--expected-cell-eval-commit", required=True)
    parser.add_argument("--candidate-label", required=True,
                        help="identifier of the scored candidate, recorded in the receipt")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--shared-removal-grid", type=float, nargs="+",
        default=[round(0.05 * i, 2) for i in range(0, 41)],
    )
    parser.add_argument(
        "--amplitude-grid", type=float, nargs="+",
        default=[0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(os.path.abspath(args.repo_root))
    output = Path(os.path.abspath(args.output_json))
    _require(not output.exists(), f"refusing to overwrite an existing receipt: {output}")

    checkout = Path(os.path.abspath(
        args.cell_eval_checkout if args.cell_eval_checkout.is_absolute()
        else repo_root / args.cell_eval_checkout
    ))
    cell_eval_commit = authenticate_cell_eval(checkout, args.expected_cell_eval_commit)
    sys.path.insert(0, str(checkout / "src"))

    import anndata as ad  # noqa: PLC0415 - deferred until the checkout is authenticated
    from cell_eval2.metrics.discrimination import discrimination_score  # noqa: PLC0415
    from cell_eval2.prep import pseudobulk_bulk_lognorm  # noqa: PLC0415

    def load_bulk(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        adata = ad.read_h5ad(path)
        perts, means = pseudobulk_bulk_lognorm(adata, PERT_COL, bulk_target_sum=BULK_TARGET_SUM)
        genes = adata.var_names.to_numpy().astype(str)
        del adata
        return np.asarray(perts).astype(str), np.asarray(means, dtype=np.float64), genes

    real_perts, real_means, real_genes = load_bulk(args.truth_view)
    pred_perts, pred_means, pred_genes = load_bulk(args.prediction_view)
    _require(np.array_equal(real_genes, pred_genes), "prediction and truth gene axes differ")
    _require(np.array_equal(real_perts, pred_perts), "prediction and truth perturbation sets differ")

    genes = real_genes
    control_rows = np.flatnonzero(real_perts == CONTROL_LABEL)
    _require(control_rows.size == 1, f"expected exactly one {CONTROL_LABEL!r} row")
    control = int(control_rows[0])
    target_mask = np.ones(real_perts.size, dtype=bool)
    target_mask[control] = False
    targets = real_perts[target_mask]

    control_identical = bool(np.array_equal(real_means[control], pred_means[control]))

    def scorer_pds(means: np.ndarray) -> float:
        per_target = discrimination_score(
            pred_bulk=(real_perts, means),
            real_bulk=(real_perts, real_means),
            genes=genes,
            pert_col=PERT_COL,
            control=CONTROL_LABEL,
            **DISCRIMINATION_KWARGS,
        )
        return float(np.nanmean(np.asarray(list(per_target.values()), dtype=np.float64)))

    baseline = scorer_pds(pred_means)
    recorded = recorded_scorer_pds(args.agg_results)
    _require(
        abs(baseline - recorded) <= max(REPRODUCTION_ATOL, 10.0 ** -len(str(recorded).split(".")[-1])),
        f"reproduced pds {baseline!r} does not match the recorded {recorded!r}",
    )

    # Independent re-derivation on the scorer's own ranked feature space.
    keep = panel_excluded_mask(genes, targets)
    real_effects = (real_means - real_means[control])[target_mask][:, keep]
    pred_effects = (pred_means - real_means[control])[target_mask][:, keep]
    structure = similarity_structure(cosine_similarity_matrix(pred_effects, real_effects))
    _require(
        abs(structure.pds - baseline) <= CROSS_CHECK_ATOL,
        f"independent re-derivation {structure.pds!r} disagrees with the scorer {baseline!r}",
    )

    # Amplitude invariance: a pure rescale of the predicted effect cannot move a
    # per-row cosine ranking. Measured, not assumed.
    amplitude: dict[str, float] = {}
    for factor in args.amplitude_grid:
        means = real_means[control][None, :] + factor * (pred_means - real_means[control])
        means[control] = pred_means[control]
        amplitude[f"{factor:g}"] = scorer_pds(means)

    # Common-response removal: how much is recoverable by post-processing alone.
    axis = shared_direction((pred_means - real_means[control])[target_mask])
    shared_removal: dict[str, float] = {}
    for fraction in args.shared_removal_grid:
        effects = remove_shared(pred_means - real_means[control], axis, fraction)
        means = real_means[control][None, :] + effects
        means[control] = pred_means[control]
        shared_removal[f"{fraction:g}"] = scorer_pds(means)

    # Reference arms.
    oracle = scorer_pds(real_means.copy())
    flat = np.repeat(pred_means[control][None, :], pred_means.shape[0], axis=0)
    control_only = scorer_pds(flat)

    # The same prediction scored WITHOUT the panel exclusion, to size what the exclusion
    # costs. This is not an alternative score; the competition always excludes.
    without_exclusion = float(np.nanmean(np.asarray(list(discrimination_score(
        pred_bulk=(real_perts, pred_means), real_bulk=(real_perts, real_means), genes=genes,
        pert_col=PERT_COL, control=CONTROL_LABEL,
        **{**DISCRIMINATION_KWARGS, "exclude_target_gene": False},
    ).values()), dtype=np.float64)))

    similarity = cosine_similarity_matrix(pred_effects, real_effects)
    matched = np.diag(similarity)
    full_effects_real = (real_means - real_means[control])[target_mask]
    full_effects_pred = (pred_means - real_means[control])[target_mask]

    # Two simulated references, both with perfectly independent error, over three seeds.
    references: dict[str, dict[str, Any]] = {}
    for name, accuracy in (
        ("homogeneous", float(matched.mean())),
        ("heterogeneous_signed", matched.copy()),
    ):
        scores = []
        for offset in REFERENCE_SEED_OFFSETS:
            generator = np.random.default_rng(args.seed + offset)
            simulated = simulate_directional_prediction(real_effects, accuracy, generator)
            scores.append(similarity_structure(cosine_similarity_matrix(simulated, real_effects)).pds)
        references[name] = {
            "pds_by_seed": {str(args.seed + o): s for o, s in zip(REFERENCE_SEED_OFFSETS, scores)},
            "pds_mean": float(np.mean(scores)),
        }

    heterogeneous = references["heterogeneous_signed"]["pds_mean"]
    homogeneous = references["homogeneous"]["pds_mean"]

    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "status": "passed",
        "scope": {
            "purpose": "pds_reachability_diagnostic",
            "read_only": True,
            "candidate_created": False,
            "counts_rendered": False,
            "official_submission_allowed": False,
        },
        "candidate": args.candidate_label,
        "seed": args.seed,
        "scorer": {
            "checkout": repo_relative(checkout, repo_root),
            "commit": cell_eval_commit,
            "preset": "vcc2026",
            "bulk_target_sum": BULK_TARGET_SUM,
            "discrimination": dict(DISCRIMINATION_KWARGS),
        },
        "git": dict(zip(("commit", "clean"), git_head(repo_root))),
        "inputs": {
            name: {
                "path": repo_relative(path, repo_root),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for name, path in (
                ("prediction_view", args.prediction_view),
                ("truth_view", args.truth_view),
                ("agg_results", args.agg_results),
            )
        },
        "axes": {
            "perturbations_including_control": int(real_perts.size),
            "targets": int(targets.size),
            "genes": int(genes.size),
            "ranked_genes_after_panel_exclusion": int(keep.sum()),
            "prediction_control_pseudobulk_identical_to_truth": control_identical,
        },
        "reproduction": {
            "scorer_pds_cosine": baseline,
            "recorded_pds_cosine": recorded,
            "independent_rederivation": structure.pds,
        },
        "reference_arms": {
            "oracle_pred_equals_real": oracle,
            "control_only_no_skill": control_only,
        },
        "amplitude_invariance": {
            "pds_by_factor": amplitude,
            "exactly_inert": len({round(v, 15) for v in amplitude.values()}) == 1,
        },
        "shared_response_removal": {
            "pds_by_fraction": shared_removal,
            "best_fraction": max(shared_removal, key=shared_removal.__getitem__),
            "best_gain": max(shared_removal.values()) - baseline,
        },
        "panel_exclusion_cost": {
            "pds_without_exclusion": without_exclusion,
            "pds_with_exclusion": baseline,
            "cost": without_exclusion - baseline,
            "top1_retrieval_with_exclusion": top1_retrieval(similarity),
            "top1_retrieval_without_exclusion": top1_retrieval(
                cosine_similarity_matrix(full_effects_pred, full_effects_real)
            ),
            "top1_retrieval_chance": 1,
        },
        "structure": {
            "candidate": structure.as_dict(),
            "matched_cosine_distribution": {
                "mean": float(matched.mean()),
                "median": float(np.median(matched)),
                "min": float(matched.min()),
                "max": float(matched.max()),
                "fraction_not_positive": float((matched <= 0.0).mean()),
            },
            "independent_error_references": references,
            "shortfall_attribution": {
                "per_target_signal_deficit": homogeneous - heterogeneous,
                "target_correlated_error": heterogeneous - baseline,
                "note": (
                    "Both references carry the candidate's own directional accuracy with "
                    "perfectly independent error. The homogeneous one discards per-target "
                    "dispersion; the heterogeneous one keeps each target's signed matched "
                    "cosine. The larger term names the dominant failure mode."
                ),
            },
            "common_response": {
                "prediction_shared_axis_vs_real": float(
                    shared_direction(pred_effects) @ shared_direction(real_effects)
                ),
                "mean_pairwise_cosine_prediction": mean_pairwise_cosine(pred_effects),
                "mean_pairwise_cosine_real": mean_pairwise_cosine(real_effects),
            },
            "effective_rank_prediction": effective_rank(unit_rows(pred_effects)),
            "effective_rank_real": effective_rank(unit_rows(real_effects)),
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
