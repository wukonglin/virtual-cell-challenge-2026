"""Opt-in v2 target standardization; legacy sealed experiments stay unchanged.

Exact constant columns are identified using observed TRAINING rows only. Their
mean is copied from the first row and their scale is one, ensuring exact zero
centered values even when repeated floating-point summation would round. Missing
modalities remain masked. PCA is deliberately unsupported until independently
registered and tested. This module performs no data loading or model fitting.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from public_flow_conditioning import (
    AlignedTargets, TargetTransform, _train_indices, fit_target_transform,
)

METHOD = "exact-constant-training-columns-first-observed-mean-unit-scale-v2"


@dataclass(frozen=True)
class StableTargetTransform(TargetTransform):
    """Compatible transform with explicit opt-in method provenance."""
    exact_constant_columns: tuple[int, ...] = ()

    def provenance(self) -> dict:
        return {**super().provenance(), "standardization_method": METHOD,
                "exact_constant_columns": list(self.exact_constant_columns),
                "constant_detection_scope": "observed_declared_training_targets_per_modality",
                "pca_supported": False}


def fit_stable_target_transform(
    aligned: AlignedTargets, *, train_ids: Sequence[str], fold_id: str,
    excluded_ids: Sequence[str] = (), scale: bool = True,
    pca_components: int | None = None,
) -> StableTargetTransform:
    """Train-only legacy rules plus exact zero-variance numerical handling.

Nonconstant columns use the legacy algorithm unchanged. Constant training
columns are temporarily zeroed before legacy fitting to avoid overflow from
summing repeated large constants; returned moments are restored to the original
feature scale. The returned transform accepts the ORIGINAL aligned features.
Neither held-out features nor recipient expression estimate any moments.
"""
    if pca_components is not None:
        raise ValueError("PCA is not supported by this opt-in stable v2 transform")
    if type(scale) is not bool:
        raise ValueError("scale must be boolean")
    # Enforce target admission/exclusions before any fitting arithmetic.
    training, indices = _train_indices(aligned, train_ids, excluded_ids)
    constant = np.zeros(aligned.values.shape[1], dtype=bool)
    first = np.zeros(aligned.values.shape[1], dtype=np.float64)
    offset = 0
    for modality, width in enumerate(aligned.widths):
        observed = indices[aligned.present[indices, modality]]
        if not len(observed):
            raise ValueError(f"No observed training features for modality {aligned.modality_names[modality]}")
        block = aligned.values[observed, offset:offset + width]
        # Equality, not range subtraction, is intentional: finite extrema may
        # overflow if subtracted, while exact constancy requires no arithmetic.
        exact = np.all(block == block[0], axis=0)
        constant[offset:offset + width] = exact
        first[offset:offset + width] = block[0]
        offset += width
    safe_values = aligned.values.copy()
    safe_values[:, constant] = 0.0
    safe = replace(aligned, values=safe_values)
    fitted = fit_target_transform(safe, train_ids=training, fold_id=fold_id,
                                  excluded_ids=excluded_ids, scale=scale)
    mean, scales = fitted.mean.copy(), fitted.scale.copy()
    mean[constant], scales[constant] = first[constant], 1.0
    mean.setflags(write=False)
    scales.setflags(write=False)
    return StableTargetTransform(
        mean=mean, scale=scales, components=None, signature=fitted.signature,
        fold_id=fitted.fold_id, training_ids_sha256=fitted.training_ids_sha256,
        observed_training_rows=fitted.observed_training_rows,
        exact_constant_columns=tuple(int(i) for i in np.flatnonzero(constant)),
    )
