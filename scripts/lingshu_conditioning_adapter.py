"""Low-rank, regularized Lingshu effect conditioning; NOT a flow/submission model.

All learned transforms use training targets only. The effect latent is a
candidate conditioning representation for future public-data flow experiments.
This module never loads expression data, generates cells, or uploads predictions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class AdapterSpec:
    input_rank: int = 8
    effect_rank: int = 4
    penalty: float = 1.0
    mean_gain: float = 1.0
    residual_gain: float = 0.25

    def validate(self):
        if self.input_rank < 1 or self.effect_rank < 1:
            raise ValueError("Ranks must be positive")
        if not np.isfinite(self.penalty) or self.penalty <= 0:
            raise ValueError("A finite positive ridge penalty is required")
        if not all(np.isfinite(v) and 0 <= v <= 1 for v in (self.mean_gain, self.residual_gain)):
            raise ValueError("Gains must lie in [0, 1]")


def matrix(value):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or min(value.shape) < 1 or not np.isfinite(value).all():
        raise ValueError("Expected a finite nonempty matrix")
    return value


def pca_basis(centered, max_rank):
    """No whitening: avoid amplifying tiny-variance gene-symbol directions."""
    _, singular, right = np.linalg.svd(centered, full_matrices=False)
    cutoff = max(centered.shape) * np.finfo(np.float64).eps * singular[0]
    rank = min(max_rank, centered.shape[0] - 1, int(np.sum(singular > cutoff)))
    return right[:rank].T.copy()


class TrainingBasis:
    """Expensive PCA shared across a fixed grid, always within ONE train fold."""

    def __init__(self, features, effects, *, max_input_rank=32, max_effect_rank=16):
        x, y = matrix(features), matrix(effects)
        if len(x) != len(y) or len(x) < 3:
            raise ValueError("At least three aligned training targets are required")
        if max_input_rank < 1 or max_effect_rank < 1:
            raise ValueError("Ranks must be positive")
        self.feature_center = x.mean(axis=0)
        # Unit mean squared ROW norm, not element RMS. Otherwise lambda would
        # be effectively ~3584x weaker for the frozen Lingshu feature dimension.
        self.feature_scale = max(float(np.sqrt(np.mean(np.sum((x - self.feature_center) ** 2, axis=1)))), 1e-12)
        centered_x = (x - self.feature_center) / self.feature_scale
        self.feature_basis = pca_basis(centered_x, max_input_rank)
        self.effect_center = y.mean(axis=0)
        centered_y = y - self.effect_center
        self.effect_basis = pca_basis(centered_y, max_effect_rank)
        self.x = centered_x @ self.feature_basis
        self.y = centered_y @ self.effect_basis
        self.n = len(x)

    def fit(self, spec):
        spec.validate()
        xb = self.feature_basis[:, :spec.input_rank]
        yb = self.effect_basis[:, :spec.effect_rank]
        x, y = self.x[:, :xb.shape[1]], self.y[:, :yb.shape[1]]
        if spec.residual_gain == 0 or x.shape[1] == 0 or y.shape[1] == 0:
            coefficients = np.zeros((x.shape[1], y.shape[1]))
        else:
            coefficients = np.linalg.solve(x.T @ x / self.n + spec.penalty * np.eye(x.shape[1]),
                                           x.T @ y / self.n)
        return LingshuConditioningAdapter(spec, self.feature_center.copy(), self.feature_scale,
                                         xb.copy(), self.effect_center.copy(), yb.copy(), coefficients)


@dataclass
class LingshuConditioningAdapter:
    spec: AdapterSpec
    feature_center: np.ndarray
    feature_scale: float
    feature_basis: np.ndarray
    effect_center: np.ndarray
    effect_basis: np.ndarray
    coefficients: np.ndarray

    def condition(self, features):
        """Supervised low-rank response latent, not cell-context information."""
        x = matrix(features)
        if x.shape[1] != len(self.feature_center):
            raise ValueError("Feature dimension mismatch")
        return self.spec.residual_gain * (((x - self.feature_center) / self.feature_scale)
                                         @ self.feature_basis @ self.coefficients)

    def predict_effect(self, features):
        return self.spec.mean_gain * self.effect_center + self.condition(features) @ self.effect_basis.T

    def save_research_only(self, path: Path, *, metadata=None):
        """Exclusive, pickle-free export. A saved fit is NOT a promotion receipt."""
        import json
        with Path(path).open("xb") as handle:
            np.savez_compressed(handle,
                schema=np.asarray("vcc-lingshu-low-rank-conditioning-research-v4"),
                spec=np.asarray(json.dumps(asdict(self.spec), sort_keys=True)),
                feature_center=self.feature_center, feature_scale=np.asarray(self.feature_scale),
                feature_basis=self.feature_basis, effect_center=self.effect_center,
                effect_basis=self.effect_basis, coefficients=self.coefficients,
                metadata=np.asarray(json.dumps(metadata or {}, sort_keys=True)),
                requires_authenticated_sidecars=np.asarray(True),
                submission_allowed=np.asarray(False), independent_test=np.asarray(False))
