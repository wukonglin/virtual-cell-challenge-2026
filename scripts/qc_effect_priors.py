#!/usr/bin/env python3
"""Audit and optionally blend sparse perturbation-effect priors.

The submission generator consumes a dense ``effects`` tensor whose non-zero
entries are sparse log1p-space effects with an explicit or legacy normalization
contract. This utility validates that contract, reports label-free diagnostics
that can expose collapsed STATE predictions, and can create a convex ensemble
only when both priors use the same effect space.

The diagnostics are not substitutes for VCC scores.  In particular, no hidden
challenge labels are available locally.  They are pre-submission safety checks
and should be combined with evaluation on a public held-out cell context.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse


REQUIRED_KEYS = ("effects", "contexts", "targets", "genes")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "priors",
        nargs="+",
        type=Path,
        help="effect-prior NPZ files to audit; the first is the primary candidate",
    )
    parser.add_argument(
        "--controls-dir",
        type=Path,
        default=Path("dataset/controls"),
        help="official controls directory used to verify target and gene axes",
    )
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--max-effects", type=int, default=161)
    parser.add_argument(
        "--ensemble-alpha",
        type=float,
        default=None,
        help="weight of the first prior in an optional two-prior convex ensemble",
    )
    parser.add_argument("--ensemble-output", type=Path, default=None)
    parser.add_argument("--ensemble-report", type=Path, default=None)
    parser.add_argument("--minimum-abs-effect", type=float, default=0.005)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def quantiles(values: np.ndarray, points: tuple[float, ...]) -> dict[str, float]:
    if values.size == 0:
        return {str(point): math.nan for point in points}
    return {
        str(point): float(value)
        for point, value in zip(points, np.quantile(values, points))
    }


@dataclass(frozen=True)
class EffectPrior:
    path: Path
    effects: np.ndarray
    contexts: tuple[str, ...]
    targets: tuple[str, ...]
    genes: tuple[str, ...]
    effect_space: str
    effect_target_sum: float | None
    effect_gene_mask: np.ndarray
    explicit_effect_contract: bool


def load_prior(path: Path) -> EffectPrior:
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(set(REQUIRED_KEYS) - set(archive.files))
        if missing:
            raise ValueError(f"{path}: missing required keys {missing}")
        effects = archive["effects"].astype(np.float32)
        contexts = tuple(archive["contexts"].astype(str).tolist())
        targets = tuple(archive["targets"].astype(str).tolist())
        genes = tuple(archive["genes"].astype(str).tolist())
        metadata_present = tuple(
            key in archive.files
            for key in ("effect_space", "effect_target_sum", "effect_gene_mask")
        )
        if any(metadata_present) and not all(metadata_present):
            raise ValueError(f"{path}: incomplete effect-space metadata")
        explicit_effect_contract = all(metadata_present)
        if explicit_effect_contract:
            effect_space = str(archive["effect_space"].item())
            effect_target_sum = float(archive["effect_target_sum"].item())
            raw_effect_gene_mask = np.asarray(archive["effect_gene_mask"])
            if raw_effect_gene_mask.shape != (len(genes),):
                raise ValueError(f"{path}: effect_gene_mask has the wrong shape")
            if not np.isin(raw_effect_gene_mask, (False, True)).all():
                raise ValueError(f"{path}: effect_gene_mask must be boolean-like")
            effect_gene_mask = raw_effect_gene_mask.astype(np.bool_)
        else:
            effect_space = "legacy-log1p-cp10k-delta"
            effect_target_sum = None
            effect_gene_mask = np.ones(len(genes), dtype=np.bool_)
    expected = (len(contexts), len(targets), len(genes))
    if effects.shape != expected:
        raise ValueError(f"{path}: effects shape {effects.shape} != {expected}")
    return EffectPrior(
        path,
        effects,
        contexts,
        targets,
        genes,
        effect_space,
        effect_target_sum,
        effect_gene_mask,
        explicit_effect_contract,
    )


def target_indices(prior: EffectPrior) -> np.ndarray:
    gene_index = {gene: i for i, gene in enumerate(prior.genes)}
    missing = sorted(set(prior.targets) - set(gene_index))
    if missing:
        raise ValueError(
            f"{prior.path}: {len(missing)} targets are absent from the gene axis"
        )
    return np.asarray([gene_index[target] for target in prior.targets], dtype=np.int64)


def effective_rank_from_gram(gram: np.ndarray) -> tuple[float, float]:
    """Return spectral effective rank and the leading-component energy share."""
    n = gram.shape[0]
    centering = np.eye(n, dtype=np.float64) - np.full((n, n), 1.0 / n)
    centered = centering @ gram @ centering
    eigenvalues = np.linalg.eigvalsh(centered)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    total = float(eigenvalues.sum())
    if total <= 1e-12:
        return 0.0, 1.0
    probabilities = eigenvalues[eigenvalues > total * 1e-12] / total
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return float(np.exp(entropy)), float(eigenvalues[-1] / total)


def cosine_from_gram(gram: np.ndarray) -> np.ndarray:
    norms = np.sqrt(np.maximum(np.diag(gram), 0.0))
    denominator = norms[:, None] * norms[None, :]
    cosine = np.zeros_like(gram, dtype=np.float64)
    np.divide(gram, denominator, out=cosine, where=denominator > 1e-12)
    return np.clip(cosine, -1.0, 1.0)


def context_structure(
    effects: np.ndarray,
    panel_feature_mask: np.ndarray,
) -> dict[str, Any]:
    matrix = sparse.csr_matrix(effects[:, panel_feature_mask], dtype=np.float64)
    gram = (matrix @ matrix.T).toarray()
    cosine = cosine_from_gram(gram)
    off_diagonal = cosine[~np.eye(cosine.shape[0], dtype=bool)]
    nearest = np.max(
        np.where(np.eye(cosine.shape[0], dtype=bool), -np.inf, cosine), axis=1
    )
    norms = np.sqrt(np.maximum(np.diag(gram), 0.0))
    n_profiles = matrix.shape[0]
    centered_squared = (
        np.diag(gram)
        - 2.0 * np.sum(gram, axis=1) / n_profiles
        + np.sum(gram) / (n_profiles * n_profiles)
    )
    centered_norm = np.sqrt(np.maximum(centered_squared, 0.0))
    specificity = centered_norm / np.maximum(norms, 1e-12)
    effective_rank, leading_share = effective_rank_from_gram(gram)
    return {
        "panel_excluded_profile_norm": quantiles(
            norms, (0.0, 0.1, 0.5, 0.9, 1.0)
        ),
        "off_diagonal_cosine": quantiles(
            off_diagonal, (0.0, 0.1, 0.5, 0.9, 1.0)
        ),
        "nearest_other_cosine": quantiles(
            nearest, (0.0, 0.1, 0.5, 0.9, 1.0)
        ),
        "target_specificity_ratio": quantiles(
            specificity, (0.0, 0.1, 0.5, 0.9, 1.0)
        ),
        "spectral_effective_rank": effective_rank,
        "leading_component_energy_share": leading_share,
    }


def cross_context_cosines(
    effects: np.ndarray,
    panel_feature_mask: np.ndarray,
) -> np.ndarray:
    values: list[np.ndarray] = []
    for left in range(effects.shape[0]):
        x = effects[left, :, panel_feature_mask].astype(np.float64)
        x_norm = np.linalg.norm(x, axis=0)
        for right in range(left + 1, effects.shape[0]):
            y = effects[right, :, panel_feature_mask].astype(np.float64)
            y_norm = np.linalg.norm(y, axis=0)
            denominator = x_norm * y_norm
            cosine = np.zeros(effects.shape[1], dtype=np.float64)
            np.divide(
                np.sum(x * y, axis=0),
                denominator,
                out=cosine,
                where=denominator > 1e-12,
            )
            values.append(np.clip(cosine, -1.0, 1.0))
    return np.concatenate(values) if values else np.empty(0, dtype=np.float64)


def audit_prior(
    prior: EffectPrior,
    official_contexts: tuple[str, ...],
    official_targets: tuple[str, ...],
    official_genes: tuple[str, ...],
    max_effects: int,
) -> dict[str, Any]:
    effects = prior.effects
    target_idx = target_indices(prior)
    nonzero = effects != 0
    selected_counts = np.count_nonzero(nonzero, axis=2)
    diagonal = effects[
        np.arange(len(prior.contexts))[:, None],
        np.arange(len(prior.targets))[None, :],
        target_idx[None, :],
    ]
    off_target_mask = nonzero.copy()
    off_target_mask[
        np.arange(len(prior.contexts))[:, None],
        np.arange(len(prior.targets))[None, :],
        target_idx[None, :],
    ] = False
    off_target_values = effects[off_target_mask]

    panel_feature_mask = np.ones(len(prior.genes), dtype=bool)
    panel_feature_mask[target_idx] = False
    structures = {
        context: context_structure(effects[c], panel_feature_mask)
        for c, context in enumerate(prior.contexts)
    }
    cross_context = cross_context_cosines(effects, panel_feature_mask)

    hard_checks = {
        "contexts_exact": prior.contexts == official_contexts,
        "targets_exact": prior.targets == official_targets,
        "gene_axis_exact": prior.genes == official_genes,
        "finite": bool(np.isfinite(effects).all()),
        "every_group_nonzero": bool(np.all(selected_counts > 0)),
        "per_group_sparsity_within_contract": bool(
            np.all(selected_counts <= max_effects)
        ),
        "target_transcript_selected": bool(np.all(diagonal != 0)),
    }
    soft_warnings: list[str] = []
    for context, structure in structures.items():
        if structure["spectral_effective_rank"] < 5.0:
            soft_warnings.append(
                f"{context}: panel-excluded effects have spectral effective rank < 5"
            )
        if structure["nearest_other_cosine"]["0.5"] > 0.98:
            soft_warnings.append(
                f"{context}: median nearest-target cosine > 0.98 suggests response collapse"
            )
        if structure["panel_excluded_profile_norm"]["0.1"] <= 1e-8:
            soft_warnings.append(
                f"{context}: at least 10% of panel-excluded response profiles are zero"
            )
    if cross_context.size and float(np.quantile(cross_context, 0.5)) > 0.995:
        soft_warnings.append(
            "median same-target cross-context cosine > 0.995; context conditioning may be inactive"
        )
    if off_target_values.size and float(np.max(np.abs(off_target_values))) > 2.0:
        soft_warnings.append("one or more off-target effects have absolute magnitude > 2")

    return {
        "path": str(prior.path),
        "sha256": sha256_file(prior.path),
        "shape": list(effects.shape),
        "effect_contract": {
            "space": prior.effect_space,
            "target_sum": prior.effect_target_sum,
            "normalization_genes": int(np.count_nonzero(prior.effect_gene_mask)),
            "explicit": prior.explicit_effect_contract,
        },
        "hard_checks": hard_checks,
        "failed_hard_checks": sorted(
            name for name, passed in hard_checks.items() if not passed
        ),
        "soft_warnings": soft_warnings,
        "selected_effects_per_group": quantiles(
            selected_counts, (0.0, 0.1, 0.5, 0.9, 1.0)
        ),
        "target_transcript_effect": quantiles(
            diagonal, (0.0, 0.1, 0.5, 0.9, 1.0)
        ),
        "off_target_effect": quantiles(
            off_target_values, (0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0)
        ),
        "off_target_absolute_effect": quantiles(
            np.abs(off_target_values), (0.0, 0.1, 0.5, 0.9, 0.99, 1.0)
        ),
        "panel_target_genes_excluded_from_structure_diagnostics": len(target_idx),
        "context_structure": structures,
        "same_target_cross_context_cosine": quantiles(
            cross_context, (0.0, 0.1, 0.5, 0.9, 1.0)
        ),
    }


def compare_priors(primary: EffectPrior, other: EffectPrior) -> dict[str, Any]:
    if (
        primary.contexts != other.contexts
        or primary.targets != other.targets
        or primary.genes != other.genes
    ):
        raise ValueError(f"cannot compare misaligned priors: {primary.path}, {other.path}")
    target_idx = target_indices(primary)
    values: list[float] = []
    sign_agreements: list[float] = []
    top_jaccards: list[float] = []
    scale_ratios: list[float] = []
    for c in range(len(primary.contexts)):
        for p in range(len(primary.targets)):
            left = primary.effects[c, p].astype(np.float64).copy()
            right = other.effects[c, p].astype(np.float64).copy()
            left[target_idx[p]] = 0.0
            right[target_idx[p]] = 0.0
            denominator = np.linalg.norm(left) * np.linalg.norm(right)
            values.append(
                float(np.dot(left, right) / denominator) if denominator > 1e-12 else 0.0
            )
            union = (left != 0) | (right != 0)
            if np.any(union):
                active = union & (left != 0) & (right != 0)
                sign_agreements.append(
                    float(np.mean(np.sign(left[active]) == np.sign(right[active])))
                    if np.any(active)
                    else 0.0
                )
                left_set = set(np.flatnonzero(left))
                right_set = set(np.flatnonzero(right))
                top_jaccards.append(
                    len(left_set & right_set) / max(len(left_set | right_set), 1)
                )
                scale_ratios.append(
                    float(np.linalg.norm(left) / max(np.linalg.norm(right), 1e-12))
                )
    return {
        "primary": str(primary.path),
        "other": str(other.path),
        "same_effect_contract": bool(
            primary.effect_space == other.effect_space
            and primary.effect_target_sum == other.effect_target_sum
            and np.array_equal(primary.effect_gene_mask, other.effect_gene_mask)
        ),
        "off_target_cosine": quantiles(
            np.asarray(values), (0.0, 0.1, 0.5, 0.9, 1.0)
        ),
        "shared_gene_sign_agreement": quantiles(
            np.asarray(sign_agreements), (0.0, 0.1, 0.5, 0.9, 1.0)
        ),
        "selected_gene_jaccard": quantiles(
            np.asarray(top_jaccards), (0.0, 0.1, 0.5, 0.9, 1.0)
        ),
        "off_target_l2_scale_ratio": quantiles(
            np.asarray(scale_ratios), (0.0, 0.1, 0.5, 0.9, 1.0)
        ),
    }


def build_ensemble(
    primary: EffectPrior,
    other: EffectPrior,
    alpha: float,
    max_effects: int,
    minimum_abs_effect: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("--ensemble-alpha must be in [0, 1]")
    if max_effects < 1:
        raise ValueError("--max-effects must be positive")
    if (
        primary.contexts != other.contexts
        or primary.targets != other.targets
        or primary.genes != other.genes
    ):
        raise ValueError("ensemble inputs do not share identical axes")
    if (
        primary.effect_space != other.effect_space
        or primary.effect_target_sum != other.effect_target_sum
        or not np.array_equal(primary.effect_gene_mask, other.effect_gene_mask)
    ):
        raise ValueError("cannot ensemble priors with different effect-space contracts")

    target_idx = target_indices(primary)
    blended = alpha * primary.effects + (1.0 - alpha) * other.effects
    output = np.zeros_like(blended, dtype=np.float32)
    selected_counts = np.zeros(blended.shape[:2], dtype=np.int32)
    keep_off_target = max_effects - 1
    for c in range(blended.shape[0]):
        for p in range(blended.shape[1]):
            row = blended[c, p].astype(np.float32, copy=True)
            row[target_idx[p]] = 0.0
            candidates = np.flatnonzero(np.abs(row) >= minimum_abs_effect)
            if len(candidates) > keep_off_target:
                order = np.argpartition(np.abs(row[candidates]), -keep_off_target)[
                    -keep_off_target:
                ]
                candidates = candidates[order]
            output[c, p, candidates] = row[candidates]
            # The count generator enforces the knockdown fold separately.  A fixed
            # non-zero marker keeps the target transcript selected in every group.
            output[c, p, target_idx[p]] = np.float32(math.log(0.20))
            selected_counts[c, p] = int(len(candidates) + 1)
    return output, selected_counts


def atomic_savez(path: Path, **arrays: Any) -> None:
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if args.max_effects <= 0:
        raise ValueError("--max-effects must be positive")
    priors = [load_prior(path) for path in args.priors]

    official_genes = tuple(
        pd.read_csv(args.controls_dir / "gene_names.csv")["gene_name"]
        .astype(str)
        .tolist()
    )
    official_targets = tuple(
        pd.read_csv(args.controls_dir / "pert_counts.csv")["target_gene"]
        .astype(str)
        .tolist()
    )
    official_contexts = ("A", "B", "C")

    report: dict[str, Any] = {
        "artifact_type": "effect_prior_pre_submission_qc",
        "created_at_unix": time.time(),
        "scope": (
            "label-free safety diagnostics; not an estimate of hidden VCC scores"
        ),
        "official_metric_note": (
            "PDS structure diagnostics exclude every panel target gene, matching "
            "the current cell-eval2 vcc2026 feature-space rule."
        ),
        "priors": [
            audit_prior(
                prior,
                official_contexts,
                official_targets,
                official_genes,
                args.max_effects,
            )
            for prior in priors
        ],
        "comparisons_to_primary": [
            compare_priors(priors[0], prior) for prior in priors[1:]
        ],
    }

    ensemble_report: dict[str, Any] | None = None
    if args.ensemble_alpha is not None or args.ensemble_output is not None:
        if len(priors) != 2:
            raise ValueError("ensemble generation requires exactly two input priors")
        if args.ensemble_alpha is None or args.ensemble_output is None:
            raise ValueError(
                "--ensemble-alpha and --ensemble-output must be provided together"
            )
        if args.ensemble_output.exists() and not args.force:
            raise FileExistsError(
                f"{args.ensemble_output} exists; pass --force to replace it"
            )
        effects, selected_counts = build_ensemble(
            priors[0],
            priors[1],
            args.ensemble_alpha,
            args.max_effects,
            args.minimum_abs_effect,
        )
        args.ensemble_output.parent.mkdir(parents=True, exist_ok=True)
        ensemble_arrays: dict[str, Any] = {
            "effects": effects,
            "contexts": np.asarray(priors[0].contexts),
            "targets": np.asarray(priors[0].targets),
            "genes": np.asarray(priors[0].genes),
            "selected_counts": selected_counts,
            "ensemble_alpha": np.float32(args.ensemble_alpha),
            "ensemble_sources": np.asarray(
                [str(prior.path) for prior in priors]
            ),
        }
        if priors[0].explicit_effect_contract:
            ensemble_arrays.update(
                effect_space=np.asarray(priors[0].effect_space),
                effect_target_sum=np.asarray(
                    priors[0].effect_target_sum, dtype=np.float64
                ),
                effect_gene_mask=priors[0].effect_gene_mask,
            )
        atomic_savez(args.ensemble_output, **ensemble_arrays)
        ensemble = load_prior(args.ensemble_output)
        ensemble_report = audit_prior(
            ensemble,
            official_contexts,
            official_targets,
            official_genes,
            args.max_effects,
        )
        report["ensemble"] = {
            "alpha_primary": args.ensemble_alpha,
            "minimum_abs_effect": args.minimum_abs_effect,
            "output": str(args.ensemble_output),
            "selected_effects_per_group": quantiles(
                selected_counts, (0.0, 0.1, 0.5, 0.9, 1.0)
            ),
            "audit": ensemble_report,
        }

    if args.report is not None:
        if args.report.exists() and not args.force:
            raise FileExistsError(f"{args.report} exists; pass --force to replace it")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.report.with_name(args.report.name + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, args.report)
    if args.ensemble_report is not None:
        if ensemble_report is None:
            raise ValueError("--ensemble-report requires ensemble generation")
        if args.ensemble_report.exists() and not args.force:
            raise FileExistsError(
                f"{args.ensemble_report} exists; pass --force to replace it"
            )
        args.ensemble_report.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.ensemble_report.with_name(args.ensemble_report.name + ".tmp")
        temporary.write_text(
            json.dumps(ensemble_report, indent=2, sort_keys=True) + "\n"
        )
        os.replace(temporary, args.ensemble_report)

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
