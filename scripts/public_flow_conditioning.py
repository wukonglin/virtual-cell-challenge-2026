"""NumPy conditioning primitives: no data loading, flow training, or file I/O.

Source hashes identify caller-authenticated inputs, not authentication performed
here. Register train folds first. Context inputs must already be an explicitly
identified, matched CONTROL-ONLY subset; mixed batches are refused, not filtered.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import re
from typing import Sequence

import numpy as np


def _label(value: str, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a nonempty, unpadded string")
    return value


def _ids(values: Sequence[str], name: str, *, unique=True, empty=False) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence, not a scalar string")
    result = tuple(_label(item, name) for item in values)
    if (not result and not empty) or (unique and len(set(result)) != len(result)):
        raise ValueError(f"{name} must be nonempty and unique")
    return result


def _sha(value: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("Expected a lowercase SHA-256 source identity")
    return value


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _matrix(values, shape=None) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 2 or raw.dtype.kind not in "iuf":
        raise ValueError("Features must be a real numeric two-dimensional array")
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.array(raw, dtype=np.float64, order="C", copy=True)
    if (shape is not None and result.shape != shape) or not np.isfinite(result).all():
        raise ValueError("Feature shape mismatch or nonfinite values")
    result.setflags(write=False)
    return result


def _array_sha(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values, dtype="<f8")).hexdigest()


@dataclass(frozen=True)
class FeatureTable:
    """One pinned modality, e.g. GenePT or GO; safe arrays only, never pickle."""

    name: str
    target_ids: tuple[str, ...]
    feature_ids: tuple[str, ...]
    values: np.ndarray
    source_id: str
    source_sha256: str
    fingerprint: str = field(init=False)

    def __post_init__(self):
        _label(self.name, "modality")
        _label(self.source_id, "source_id")
        _sha(self.source_sha256)
        targets = _ids(self.target_ids, "target_ids", empty=True)
        features = _ids(self.feature_ids, "feature_ids")
        values = _matrix(self.values, (len(targets), len(features)))
        object.__setattr__(self, "target_ids", targets)
        object.__setattr__(self, "feature_ids", features)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "fingerprint", _digest({
            "name": self.name, "source_id": self.source_id,
            "source_sha256": self.source_sha256, "targets": targets,
            "features": features, "values_sha256": _array_sha(values),
        }))


@dataclass(frozen=True)
class AlignedTargets:
    target_ids: tuple[str, ...]
    values: np.ndarray
    modality_names: tuple[str, ...]
    widths: tuple[int, ...]
    present: np.ndarray
    source_fingerprints: tuple[str, ...]
    operation: str = "identity"

    def __post_init__(self):
        targets = _ids(self.target_ids, "target_ids")
        names = _ids(self.modality_names, "modality_names")
        widths = tuple(self.widths)
        if len(widths) != len(names) or any(type(w) is not int or w < 1 for w in widths):
            raise ValueError("Each modality must have a positive integer width")
        signatures = tuple(_sha(s) for s in self.source_fingerprints)
        if len(signatures) != len(names):
            raise ValueError("Missing modality provenance")
        values = _matrix(self.values, (len(targets), sum(widths)))
        present = np.array(self.present, copy=True)
        if present.dtype != np.bool_ or present.shape != (len(targets), len(names)):
            raise ValueError("present must be an explicit boolean target-by-modality mask")
        offset = 0
        for index, width in enumerate(widths):
            if np.any(values[~present[:, index], offset:offset + width] != 0):
                raise ValueError("Missing modality values must be zero with an explicit mask")
            offset += width
        present.setflags(write=False)
        for name, value in (("target_ids", targets), ("modality_names", names),
                            ("widths", widths), ("values", values), ("present", present),
                            ("source_fingerprints", signatures)):
            object.__setattr__(self, name, value)
        _label(self.operation, "operation")

    @property
    def unknown_target(self) -> np.ndarray:
        """Unknown in every supplied modality, distinct from a known zero vector."""
        return ~self.present.any(axis=1)

    @property
    def signature(self) -> tuple:
        return self.modality_names, self.widths, self.source_fingerprints, self.operation


def align_target_features(target_ids: Sequence[str], tables: Sequence[FeatureTable]) -> AlignedTargets:
    targets = _ids(target_ids, "target_ids")
    tables = tuple(tables)
    if not tables or any(not isinstance(table, FeatureTable) for table in tables):
        raise ValueError("Supply at least one explicit FeatureTable")
    names = _ids([table.name for table in tables], "modality_names")
    blocks, masks = [], []
    for table in tables:
        lookup = {target: index for index, target in enumerate(table.target_ids)}
        block = np.zeros((len(targets), len(table.feature_ids)), dtype=np.float64)
        mask = np.array([target in lookup for target in targets], dtype=bool)
        for index, target in enumerate(targets):
            if mask[index]:
                block[index] = table.values[lookup[target]]
        blocks.append(block)
        masks.append(mask)
    return AlignedTargets(targets, np.concatenate(blocks, axis=1), names,
                          tuple(len(table.feature_ids) for table in tables),
                          np.column_stack(masks), tuple(table.fingerprint for table in tables))


def _train_indices(aligned: AlignedTargets, train_ids, excluded_ids=()) -> tuple[tuple[str, ...], np.ndarray]:
    train = _ids(train_ids, "train_ids")
    excluded = _ids(excluded_ids, "excluded_ids", empty=True)
    if set(train) & set(excluded):
        raise ValueError("Training IDs overlap explicitly excluded IDs")
    lookup = {target: index for index, target in enumerate(aligned.target_ids)}
    if any(target not in lookup for target in train):
        raise ValueError("Training target missing from the aligned target axis")
    return train, np.array([lookup[target] for target in train], dtype=np.int64)


@dataclass(frozen=True)
class TargetTransform:
    """Fitted once on declared train IDs; transform never estimates statistics."""

    mean: np.ndarray
    scale: np.ndarray
    components: np.ndarray | None
    signature: tuple
    fold_id: str
    training_ids_sha256: str
    observed_training_rows: tuple[int, ...]

    def transform(self, aligned: AlignedTargets) -> np.ndarray:
        if aligned.signature != self.signature:
            raise ValueError("Feature source, modality layout, or ablation changed")
        result = (aligned.values - self.mean) / self.scale
        offset = 0
        for index, width in enumerate(aligned.widths):
            result[~aligned.present[:, index], offset:offset + width] = 0
            offset += width
        if self.components is not None:
            result = result @ self.components.T
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite transformed features")
        return result

    def provenance(self) -> dict:
        return {"fold_id": self.fold_id, "training_ids_sha256": self.training_ids_sha256,
                "observed_training_rows": self.observed_training_rows,
                "feature_signature_sha256": _digest(self.signature),
                "mean_sha256": _array_sha(self.mean), "scale_sha256": _array_sha(self.scale),
                "components_sha256": None if self.components is None else _array_sha(self.components)}


def fit_target_transform(aligned: AlignedTargets, *, train_ids: Sequence[str], fold_id: str,
                         excluded_ids: Sequence[str] = (), scale: bool = True,
                         pca_components: int | None = None) -> TargetTransform:
    """Center observed training rows per modality; optionally scale and apply PCA.

    Missing blocks stay zero and are NOT observed zero vectors when fitting
    moments. Keep ``aligned.present`` beside the returned model features. PCA
    uses only declared training rows; this function cannot independently
    authenticate the caller's split manifest. ``scale=False`` still centers.
    """
    _label(fold_id, "fold_id")
    if type(scale) is not bool:
        raise ValueError("scale must be boolean")
    train, indices = _train_indices(aligned, train_ids, excluded_ids)
    mean, scales = np.zeros(aligned.values.shape[1]), np.ones(aligned.values.shape[1])
    counts, offset = [], 0
    for index, width in enumerate(aligned.widths):
        observed = indices[aligned.present[indices, index]]
        if not len(observed):
            raise ValueError(f"No observed training features for modality {aligned.modality_names[index]}")
        block = aligned.values[observed, offset:offset + width]
        mean[offset:offset + width] = block.mean(axis=0)
        if scale:
            std = block.std(axis=0)
            scales[offset:offset + width] = np.where(std > 0, std, 1.0)
        counts.append(len(observed))
        offset += width
    if not np.isfinite(mean).all() or not np.isfinite(scales).all():
        raise ValueError("Training moments overflowed")
    mean.setflags(write=False)
    scales.setflags(write=False)
    transform = TargetTransform(mean, scales, None, aligned.signature, fold_id,
                                _digest(train), tuple(counts))
    components = None
    if pca_components is not None:
        maximum = min(len(train) - 1, aligned.values.shape[1])
        if type(pca_components) is not int or not 1 <= pca_components <= maximum:
            raise ValueError("PCA components must be within train-only centered matrix dimensions")
        # Select training rows BEFORE arithmetic/SVD involving input values.
        subset = AlignedTargets(train, aligned.values[indices], aligned.modality_names,
                                aligned.widths, aligned.present[indices],
                                aligned.source_fingerprints, aligned.operation)
        _, _, vt = np.linalg.svd(transform.transform(subset), full_matrices=False)
        components = vt[:pca_components].copy()
        for row in components:
            if row[np.argmax(np.abs(row))] < 0:
                row *= -1
        components.setflags(write=False)
    return TargetTransform(mean, scales, components, aligned.signature, fold_id,
                           _digest(train), tuple(counts))


def ablate_target_features(aligned: AlignedTargets, *, mode: str, seed: int = 0,
                          train_ids: Sequence[str] = (), fold_id: str | None = None) -> AlignedTargets:
    """Apply once to a registered target cohort before fold fitting.

    Shuffle jointly within identical availability strata, preserving missingness
    and cross-modality pairing. Constant means use training IDs only. Availability
    masks are retained in every arm, including zero: this is a feature-value test.
    """
    if mode not in {"shuffle", "zero", "constant"} or type(seed) is not int or seed < 0:
        raise ValueError("Invalid ablation mode or seed")
    values = np.zeros_like(aligned.values)
    operation = mode
    if mode == "shuffle":
        rng = np.random.default_rng(seed)
        for pattern in np.unique(aligned.present, axis=0):
            rows = np.flatnonzero(np.all(aligned.present == pattern, axis=1))
            values[rows] = aligned.values[rng.permutation(rows)]
        operation = f"shuffle:{seed}:{_digest(aligned.target_ids)}"
    elif mode == "constant":
        _label(fold_id, "fold_id")
        train, indices = _train_indices(aligned, train_ids)
        offset = 0
        for index, width in enumerate(aligned.widths):
            observed = indices[aligned.present[indices, index]]
            if not len(observed):
                raise ValueError("Cannot estimate a modality constant without training observations")
            values[aligned.present[:, index], offset:offset + width] = (
                aligned.values[observed, offset:offset + width].mean(axis=0))
            offset += width
        operation = f"constant:{fold_id}:{_digest(train)}"
    return AlignedTargets(aligned.target_ids, values, aligned.modality_names, aligned.widths,
                          aligned.present, aligned.source_fingerprints, operation)


@dataclass(frozen=True)
class ControlSummary:
    values: np.ndarray
    n_controls: int
    provenance: dict


def summarize_matched_controls(values, *, cell_ids: Sequence[str], context_ids: Sequence[str],
                               match_keys: Sequence[str], is_control, requested_context: str,
                               requested_match_key: str, feature_ids: Sequence[str], source_id: str,
                               source_sha256: str, representation: str,
                               include_std: bool = True) -> ControlSummary:
    """Mean and optional population std of ONE matched control-only population.

    ``match_keys`` encode the caller's registered donor/assay/batch matching unit.
    Different contexts or matching units are rejected, never pooled. Metadata is
    checked before converting/reading the value array. Representation is an
    explicit counts/log1p/frozen-embedding declaration, not an inferred scale.
    """
    cells = _ids(cell_ids, "cell_ids")
    contexts = _ids(context_ids, "context_ids", unique=False)
    matches = _ids(match_keys, "match_keys", unique=False)
    features = _ids(feature_ids, "feature_ids")
    _label(requested_context, "requested_context")
    _label(requested_match_key, "requested_match_key")
    _label(source_id, "source_id")
    _sha(source_sha256)
    _label(representation, "representation")
    flags = np.asarray(is_control)
    if flags.dtype != np.bool_ or flags.shape != (len(cells),) or not flags.all():
        raise ValueError("An explicit control-only boolean mask is required; treated rows are forbidden")
    if len(contexts) != len(cells) or any(item != requested_context for item in contexts):
        raise ValueError("Missing matched context controls; foreign contexts cannot be pooled")
    if len(matches) != len(cells) or any(item != requested_match_key for item in matches):
        raise ValueError("Controls do not belong to the requested matching unit")
    if type(include_std) is not bool:
        raise ValueError("include_std must be boolean")
    matrix = _matrix(values, (len(cells), len(features)))
    summary = matrix.mean(axis=0)
    if include_std:
        summary = np.concatenate([summary, matrix.std(axis=0)])
    if not np.isfinite(summary).all():
        raise ValueError("Control summary overflowed")
    summary.setflags(write=False)
    return ControlSummary(summary, len(cells), {
        "source_id": source_id, "source_sha256": source_sha256,
        "input_values_sha256": _array_sha(matrix), "cell_ids_sha256": _digest(cells),
        "feature_ids_sha256": _digest(features), "context_id": requested_context,
        "match_key": requested_match_key, "representation": representation,
        "statistic": "mean+population_std" if include_std else "mean",
        "control_only": True, "n_controls": len(cells),
    })
