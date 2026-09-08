#!/usr/bin/env python3
"""How much per-target directional accuracy does each scored component actually require?

Two of the six official components -- ``pds_cosine`` and ``expr_mse_unbiased_capped_norm``
-- are functions of the predicted pseudobulk delta alone, so the accuracy each one demands
can be measured rather than guessed. They demand wildly different amounts, and knowing
which is cheap is what decides where modelling effort goes.

Three things are computed here, all on a sealed public panel with real labels:

1. **The analytic floor for expression error.** Writing the predicted delta as
   ``d = a * ||t|| * u`` with ``u`` a unit direction at cosine ``c`` to the true delta
   ``t``, the error ratio is ``1 + a^2 - 2*a*c``. Minimised over amplitude at ``a = c``,
   giving a floor of ``1 - c^2``. Expression error therefore cannot beat the no-skill
   baseline at all until ``c`` is large, and no amount of amplitude calibration
   substitutes for direction.

2. **A measured sweep** over (aligned signal fraction, amplitude ratio), scored through
   the pinned scorer's own discrimination implementation. This gives the real requirement
   curve for ``pds_cosine`` alongside the expression-error ratio, so the two can be read
   against each other.

3. **The reliability-gating ceiling.** A candidate whose per-target accuracy is dispersed
   -- some targets predicted forwards, some backwards -- ranks far worse than a uniform
   one at the same mean. That invites shrinking the unreliable targets toward the control.
   This measures how much an ORACLE gate recovers, which is the upper bound, and then
   whether any gate computable WITHOUT the truth recovers anything at all against a
   random-gating control.

The module is read only. Sealed truth is read for measurement and for the oracle bound
only; nothing here fits a model, selects a feature, or chooses a checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

RECEIPT_SCHEMA = "vcc-score-requirement-diagnostic-v1"
DEFAULT_SEED = 20260901
# Published per-context reference constants for expr_mse_unbiased_capped_norm, from the
# vendored metrics brief section 8. Used only to convert a raw ratio into a scaled score
# for reporting; the measurement itself does not depend on them.
MSE_BASELINE = 0.989
MSE_REPLICATE = 0.0365


class RequirementError(RuntimeError):
    """A precondition of the diagnostic failed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RequirementError(message)


# --------------------------------------------------------------------------------------
# Pure numeric core.
# --------------------------------------------------------------------------------------


def amplitude_ratio(predicted: np.ndarray, truth: np.ndarray) -> float:
    """``||predicted|| / ||truth||`` over the whole panel."""
    denominator = float(np.linalg.norm(truth))
    _require(denominator > 0.0, "truth has zero norm")
    return float(np.linalg.norm(predicted)) / denominator


def expression_error_ratio(predicted: np.ndarray, truth: np.ndarray) -> float:
    """``||predicted - truth||^2 / ||truth||^2``, the leading term of the scored metric.

    The shipped ``expr_mse_unbiased_capped_norm`` subtracts a delete-one jackknife from
    both legs, so this is the noise-free counterpart rather than the scored value. It is
    the right quantity for a requirement curve, because the jackknife correction depends
    on cell counts and dispersion rather than on directional accuracy.
    """
    denominator = float((np.asarray(truth, dtype=np.float64) ** 2).sum())
    _require(denominator > 0.0, "truth has zero energy")
    return float(((np.asarray(predicted, dtype=np.float64) - truth) ** 2).sum() / denominator)


def best_expression_error_ratio(accuracy: float) -> float:
    """The lowest error ratio reachable at directional accuracy ``accuracy``.

    ``min_a (1 + a^2 - 2*a*c) = 1 - c^2`` at ``a = c``.
    """
    _require(-1.0 <= accuracy <= 1.0, "accuracy must lie in [-1, 1]")
    return 1.0 - float(accuracy) ** 2


def optimal_amplitude(accuracy: float) -> float:
    """The amplitude ratio that minimises expression error at a given accuracy."""
    _require(-1.0 <= accuracy <= 1.0, "accuracy must lie in [-1, 1]")
    return float(accuracy)


def scaled_from_raw(raw: float, baseline: float, replicate: float) -> float:
    """Reference scaling ``s = (u - b) / (r - b)``, clamped to [0, 1] as the scorer does
    for this one component."""
    _require(baseline != replicate, "baseline and replicate anchor coincide")
    return float(min(1.0, max(0.0, (raw - baseline) / (replicate - baseline))))


def required_accuracy_for_error_ratio(target_ratio: float) -> float:
    """Inverse of the floor: the accuracy without which ``target_ratio`` is unreachable."""
    _require(0.0 < target_ratio <= 1.0, "target ratio must lie in (0, 1]")
    return float(np.sqrt(1.0 - target_ratio))


def unit_rows(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    _require(values.ndim == 2, "unit_rows expects a 2-D array")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.where(norms == 0.0, 1.0, norms)


def blend_toward_truth(
    truth: np.ndarray, accuracy: float, generator: np.random.Generator
) -> np.ndarray:
    """A unit-direction field at the given cosine to ``truth``, with isotropic remainder."""
    _require(0.0 <= accuracy <= 1.0, "accuracy must lie in [0, 1]")
    noise = unit_rows(generator.standard_normal(np.shape(truth)))
    return unit_rows(accuracy * unit_rows(truth) + np.sqrt(1.0 - accuracy**2) * noise)


def gate_targets(delta: np.ndarray, reliability: np.ndarray, drop: int) -> np.ndarray:
    """Shrink the ``drop`` least reliable targets to the control (a zero delta).

    A zeroed row ties its whole comparison at cosine distance 1.0 and therefore scores
    exactly 0.5 under mid-rank ties -- chance, not a penalty. That is what makes gating
    attractive in principle: a target predicted backwards scores below chance, so removing
    it should help.
    """
    values = np.array(delta, dtype=np.float64, copy=True)
    scores = np.asarray(reliability, dtype=np.float64)
    _require(scores.shape[0] == values.shape[0], "one reliability score per target")
    _require(0 <= drop <= values.shape[0], "drop count out of range")
    if drop == 0:
        return values
    finite = np.where(np.isnan(scores), -np.inf, scores)
    values[np.argsort(finite, kind="stable")[:drop]] = 0.0
    return values


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
    resolved = Path(os.path.abspath(path))
    try:
        return resolved.relative_to(repo_root).as_posix()
    except ValueError:
        raise RequirementError(f"path escapes the repository root: {path}") from None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--prediction-view", type=Path, required=True)
    parser.add_argument("--truth-view", type=Path, required=True)
    parser.add_argument("--cell-eval-checkout", type=Path, default=Path("external/cell-eval2"))
    parser.add_argument("--expected-cell-eval-commit", required=True)
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--accuracy-grid", type=float, nargs="+",
                        default=[0.0, 0.01, 0.02, 0.03, 0.05, 0.10, 0.20, 0.30, 0.50])
    parser.add_argument("--amplitude-grid", type=float, nargs="+",
                        default=[0.02, 0.05, 0.10, 0.32, 1.0, 2.0])
    parser.add_argument("--random-gate-repeats", type=int, default=5)
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
    actual = subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"],
                            check=True, capture_output=True, text=True).stdout.strip()
    _require(actual == args.expected_cell_eval_commit,
             f"cell-eval2 commit mismatch: expected {args.expected_cell_eval_commit}, found {actual}")
    sys.path.insert(0, str(checkout / "src"))
    sys.path.insert(0, str(repo_root / "scripts"))

    import anndata as ad  # noqa: PLC0415
    from cell_eval2.metrics.discrimination import discrimination_score  # noqa: PLC0415
    from cell_eval2.prep import pseudobulk_bulk_lognorm  # noqa: PLC0415
    from diagnose_pds_reachability import (  # noqa: PLC0415
        BULK_TARGET_SUM, CONTROL_LABEL, DISCRIMINATION_KWARGS, PERT_COL,
        cosine_similarity_matrix, panel_excluded_mask,
    )

    def load(path: Path):
        adata = ad.read_h5ad(path)
        perts, means = pseudobulk_bulk_lognorm(adata, PERT_COL, bulk_target_sum=BULK_TARGET_SUM)
        gene_names = adata.var_names.to_numpy().astype(str)
        del adata
        return np.asarray(perts).astype(str), np.asarray(means, dtype=np.float64), gene_names

    real_perts, real_means, genes = load(args.truth_view)
    pred_perts, pred_means, pred_genes = load(args.prediction_view)
    _require(np.array_equal(genes, pred_genes), "gene axes differ")
    _require(np.array_equal(real_perts, pred_perts), "perturbation sets differ")
    control = int(np.flatnonzero(real_perts == CONTROL_LABEL)[0])
    mask = np.ones(real_perts.size, dtype=bool)
    mask[control] = False
    keep = panel_excluded_mask(genes, real_perts[mask])

    truth = (real_means - real_means[control])[mask]
    candidate = (pred_means - real_means[control])[mask]
    truth_norms = np.linalg.norm(truth, axis=1, keepdims=True)

    def pds(delta: np.ndarray) -> float:
        means = np.empty_like(real_means)
        means[mask] = real_means[control][None, :] + delta
        means[control] = pred_means[control]
        per_target = discrimination_score(
            pred_bulk=(real_perts, means), real_bulk=(real_perts, real_means), genes=genes,
            pert_col=PERT_COL, control=CONTROL_LABEL, **DISCRIMINATION_KWARGS,
        )
        return float(np.nanmean(np.asarray(list(per_target.values()), dtype=np.float64)))

    matched = np.diag(cosine_similarity_matrix(candidate[:, keep], truth[:, keep]))
    current_accuracy = float(matched.mean())
    generator = np.random.default_rng(args.seed)

    sweep = []
    for accuracy in args.accuracy_grid:
        direction = blend_toward_truth(truth, accuracy, generator)
        realised = float(
            (unit_rows(direction[:, keep]) * unit_rows(truth[:, keep])).sum(axis=1).mean()
        )
        row: dict[str, Any] = {"requested_accuracy": accuracy, "realised_accuracy": realised,
                               "pds_cosine": pds(direction * truth_norms), "by_amplitude": {}}
        for amplitude in args.amplitude_grid:
            row["by_amplitude"][f"{amplitude:g}"] = expression_error_ratio(
                direction * truth_norms * amplitude, truth)
        sweep.append(row)

    # Reliability gating. The oracle knows each target's true matched cosine; the proxies
    # do not. A proxy that fails to beat random gating carries no reliability information.
    counts = [0, 30, 60, 90, 112, 150, 200, 250]
    oracle = {str(k): pds(gate_targets(candidate, matched, k)) for k in counts}
    flipped = candidate.copy()
    flipped[matched < 0.0] *= -1.0

    predicted_norm = np.linalg.norm(candidate[:, keep], axis=1)
    basal = np.array([
        real_means[control][np.flatnonzero(genes == t)[0]] if (genes == t).any() else np.nan
        for t in real_perts[mask]
    ])
    proxies = {
        "predicted_delta_norm_keep_largest": predicted_norm,
        "predicted_delta_norm_keep_smallest": -predicted_norm,
        "target_gene_basal_expression": basal,
    }
    proxy_results = {
        name: {str(k): pds(gate_targets(candidate, score, k)) for k in (60, 112, 200)}
        for name, score in proxies.items()
    }
    random_control = {}
    for k in (60, 112, 200):
        draws = []
        for _ in range(args.random_gate_repeats):
            score = generator.standard_normal(candidate.shape[0])
            draws.append(pds(gate_targets(candidate, score, k)))
        random_control[str(k)] = {"mean": float(np.mean(draws)), "sd": float(np.std(draws))}

    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "status": "passed",
        "scope": {"purpose": "score_requirement_diagnostic", "read_only": True,
                  "truth_used_for": "measurement and the oracle upper bound only",
                  "candidate_created": False, "official_submission_allowed": False},
        "candidate": args.candidate_label,
        "seed": args.seed,
        "scorer": {"checkout": repo_relative(checkout, repo_root), "commit": actual,
                   "preset": "vcc2026"},
        "inputs": {
            name: {"path": repo_relative(path, repo_root), "size_bytes": path.stat().st_size,
                   "sha256": sha256_file(path)}
            for name, path in (("prediction_view", args.prediction_view),
                               ("truth_view", args.truth_view))
        },
        "current": {
            "matched_cosine": current_accuracy,
            "anti_correlated_fraction": float((matched <= 0.0).mean()),
            "amplitude_ratio": amplitude_ratio(candidate, truth),
            "pds_cosine": pds(candidate),
            "expression_error_ratio": expression_error_ratio(candidate, truth),
            "amplitude_ratio_minimising_expression_error": optimal_amplitude(current_accuracy),
        },
        "expression_error_floor": {
            "formula": "min over amplitude of ||a*u - t||^2 / ||t||^2 is 1 - c^2 at a = c",
            "by_accuracy": {
                f"{c:g}": {
                    "floor_ratio": best_expression_error_ratio(c),
                    "scaled_score": scaled_from_raw(best_expression_error_ratio(c),
                                                    MSE_BASELINE, MSE_REPLICATE),
                }
                for c in (0.025, 0.05, 0.10, 0.20, 0.30, 0.40, 0.60)
            },
            "accuracy_required_for_top50_median_scaled_0_0759":
                required_accuracy_for_error_ratio(
                    MSE_BASELINE + 0.0759 * (MSE_REPLICATE - MSE_BASELINE)),
        },
        "requirement_sweep": sweep,
        "reliability_gating": {
            "oracle_by_targets_zeroed": oracle,
            "oracle_best": max(oracle.values()),
            "oracle_sign_flip": pds(flipped),
            "deployable_proxies": proxy_results,
            "random_gate_control": random_control,
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
