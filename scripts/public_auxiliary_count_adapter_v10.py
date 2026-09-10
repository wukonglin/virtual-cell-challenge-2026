"""Bounded, control-anchored 482-gene flow -> full 18,533-gene count adapter.

Pure in-memory functions: no file reads, checkpoints, metadata downloads, cloud
calls, inference, or submission. The caller must authenticate the exact official
axis, model/normalization axes, controls-only role and checkpoint separately.

Predictions use log1p(CP10000) over 7,097 measured normalization genes. Inverse
normalization uses EACH ORIGINAL CONTROL'S measured library, not the 482-gene
or full 18,533-gene total. This is a declared count-rendering assumption; it
does not recover independent ground-truth UMI counts or conserve the final
normalization/full library after the other 18,051 genes are held fixed.

Only modeled coordinates may change. No clipping, thinning, target-forcing,
library redistribution, aliasing, missing-gene zero filling or seed retries.
Numeric/density/memory failures refuse the complete block rather than repair it.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math

import numpy as np
from scipy import sparse

SCHEMA = "public-auxiliary-native-count-adapter-v10"
DEFAULT_AXIS_SIZES = (18533, 482, 7097)
CP_SCALE = 10000.0
MAX_CELL_COUNTS = 1_000_000
MAX_LOG_CP_FLOAT32 = np.float32(np.log1p(CP_SCALE))


def require(value, message):
    if not value:
        raise ValueError(message)


def _axis(values, name):
    require(not isinstance(values, (str, bytes)), name + " must be a sequence")
    result = tuple(values)
    require(result and all(isinstance(v, str) and v and v == v.strip() for v in result)
            and len(result) == len(set(result)), "Invalid exact " + name)
    return result


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class GeneMapping:
    full_genes: tuple[str, ...]
    modeled_genes: tuple[str, ...]
    normalization_genes: tuple[str, ...]
    modeled_indices: np.ndarray
    normalization_indices: np.ndarray
    expected_sizes: tuple[int, int, int]

    def provenance(self):
        return {"full_gene_count": len(self.full_genes), "modeled_gene_count": len(self.modeled_genes),
                "normalization_gene_count": len(self.normalization_genes),
                "full_gene_order_sha256": _digest(self.full_genes),
                "modeled_gene_order_sha256": _digest(self.modeled_genes),
                "normalization_gene_order_sha256": _digest(self.normalization_genes),
                "modeled_indices_sha256": _digest(self.modeled_indices.tolist()),
                "normalization_indices_sha256": _digest(self.normalization_indices.tolist()),
                "dimensions_match_current_vcc": self.expected_sizes == DEFAULT_AXIS_SIZES}


def build_gene_mapping(full_genes, modeled_genes, normalization_genes, *,
                       modeled_indices=None, normalization_indices=None,
                       expected_sizes=DEFAULT_AXIS_SIZES):
    """Exact symbol joins only; no implicit reordering of prediction columns.

    Optional supplied index arrays must exactly match the symbol join. The
    ``expected_sizes`` override supports explicitly sized synthetic tests; it
    never claims that a nonstandard axis is ready for a VCC submission.
    """
    full = _axis(full_genes, "full gene axis")
    modeled = _axis(modeled_genes, "modeled gene axis")
    norm = _axis(normalization_genes, "normalization gene axis")
    expected = tuple(expected_sizes)
    require(len(expected) == 3 and all(type(v) is int and v > 0 for v in expected)
            and expected[1] <= expected[2] <= expected[0], "Invalid expected axis sizes")
    require((len(full), len(modeled), len(norm)) == expected, "Gene axis dimensions differ")
    require(set(modeled) <= set(norm) <= set(full), "Missing/unaligned genes; no alias or zero-fill allowed")
    positions = {gene: i for i, gene in enumerate(full)}
    mapped = []
    for names, supplied, label in ((modeled, modeled_indices, "modeled"),
                                    (norm, normalization_indices, "normalization")):
        correct = np.asarray([positions[g] for g in names], dtype=np.int64)
        if supplied is not None:
            a = np.asarray(supplied)
            require(a.shape == correct.shape and a.dtype.kind in "iu" and np.array_equal(a, correct),
                    "Supplied " + label + " index mapping differs from exact gene order")
        correct.setflags(write=False)
        mapped.append(correct)
    return GeneMapping(full, modeled, norm, mapped[0], mapped[1], expected)


def _validate_mapping(mapping):
    require(isinstance(mapping, GeneMapping), "Explicit GeneMapping required")
    return build_gene_mapping(mapping.full_genes, mapping.modeled_genes, mapping.normalization_genes,
        modeled_indices=mapping.modeled_indices, normalization_indices=mapping.normalization_indices,
        expected_sizes=mapping.expected_sizes)


@dataclass(frozen=True)
class CountLimits:
    max_batch_cells: int = 512
    max_cell_counts: int = MAX_CELL_COUNTS
    max_genes_per_cell: int = 12000
    max_density: float = 0.75
    max_output_nnz: int = 8_000_000
    max_input_bytes: int = 128 << 20
    max_working_bytes: int = 512 << 20
    max_output_bytes: int = 128 << 20

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name != "max_density":
                require(type(value) is int and value > 0, "Invalid positive integer limit " + name)
        require(self.max_cell_counts <= MAX_CELL_COUNTS, "Cannot exceed the one-million cell-count limit")
        require(type(self.max_density) in (int, float) and math.isfinite(self.max_density)
                and 0 < self.max_density <= 1, "Invalid density limit")


def _storage_bytes(raw):
    if sparse.issparse(raw):
        # CSR/CSC are deliberately the only accepted sparse layouts: inspect
        # bounded storage and full structure BEFORE dtype conversion/canonicalization.
        require(sparse.isspmatrix_csr(raw) or sparse.isspmatrix_csc(raw)
                or isinstance(raw, (sparse.csr_array, sparse.csc_array)), "CSR/CSC sparse controls required")
        return int(raw.data.nbytes + raw.indices.nbytes + raw.indptr.nbytes)
    require(isinstance(raw, np.ndarray), "A NumPy dense or CSR/CSC control matrix is required")
    return int(raw.nbytes)


def _check_sparse_structure(raw):
    """Inspect CSR/CSC without SciPy check_format's input-pruning side effect."""
    data, indices, indptr = raw.data, raw.indices, raw.indptr
    major, minor = raw.shape if raw.format == "csr" else raw.shape[::-1]
    require(data.ndim == indices.ndim == indptr.ndim == 1
            and len(data) == len(indices) and len(indptr) == major+1
            and indices.dtype.kind in "iu" and indptr.dtype.kind in "iu",
            "Invalid sparse structural arrays")
    require(indptr[0] == 0 and np.all(indptr[1:] >= indptr[:-1])
            and indptr[-1] <= len(data), "Invalid sparse row/column pointer")
    used = int(indptr[-1])
    require(np.all(indices[:used] >= 0) and np.all(indices[:used] < minor),
            "Sparse column/row index outside gene matrix")


def _canonical_controls(raw, mapping, limits):
    require(isinstance(limits, CountLimits), "Explicit CountLimits required")
    shape = getattr(raw, "shape", ())
    require(len(shape) == 2 and 0 < shape[0] <= limits.max_batch_cells
            and shape[1] == len(mapping.full_genes), "Raw control matrix shape/batch bound differs")
    storage = _storage_bytes(raw)
    require(storage <= limits.max_input_bytes, "Input storage byte bound exceeded")
    if sparse.issparse(raw):
        _check_sparse_structure(raw)
        require(raw.nnz <= limits.max_output_nnz, "Input sparse nnz bound exceeded")
        values, input_nnz = raw.data, raw.nnz
    else:
        values, input_nnz = raw, int(np.count_nonzero(raw))
    require(values.dtype.kind in "iuf" and np.isfinite(values).all()
            and np.all(values >= 0) and np.all(values <= limits.max_cell_counts)
            and np.equal(values, np.floor(values)).all(), "Controls must contain exact finite nonnegative integer counts")
    # Covers owned CSR copies, norm slice, possible replacement/output storage,
    # dense input conversion and float64 modeled expectation/rounding buffers.
    potential_nnz = min(shape[0]*shape[1], input_nnz + shape[0]*len(mapping.modeled_genes))
    estimate = (3*storage + 40*input_nnz + 24*potential_nnz
                + 64*shape[0]*len(mapping.modeled_genes) + 64*(shape[0]+shape[1])
                + (8*shape[0]*shape[1] if not sparse.issparse(raw) else 0))
    require(estimate <= limits.max_working_bytes, "Conservative working-memory byte bound exceeded")
    # Each original nonzero is already <=1M and nnz is bounded, so int64
    # duplicate summation cannot overflow before per-cell totals are checked.
    matrix = sparse.csr_matrix(raw, dtype=np.int64, copy=True)
    matrix.sum_duplicates()
    matrix.sort_indices()
    matrix.eliminate_zeros()
    library = np.asarray(matrix.sum(axis=1, dtype=np.int64)).reshape(-1)
    require(np.all(library > 0) and np.all(library <= limits.max_cell_counts),
            "Input cells must have positive libraries within one million counts")
    require(np.all(np.diff(matrix.indptr) <= limits.max_genes_per_cell)
            and matrix.nnz <= limits.max_output_nnz and matrix.nnz <= limits.max_density*shape[0]*shape[1],
            "Input density/nonzero bounds exceeded")
    matrix = matrix.astype(np.int32)
    return matrix, library, {"input_storage_bytes": storage, "working_bytes_upper_bound": int(estimate)}


def _control_encoding(raw, mapping):
    library = np.asarray(raw[:, mapping.normalization_indices].sum(axis=1, dtype=np.int64)).reshape(-1)
    counts = raw[:, mapping.modeled_indices].toarray().astype(np.int32, copy=False)
    scaled = np.zeros(counts.shape, dtype=np.float64)
    valid = library > 0
    scaled[valid] = counts[valid].astype(np.float64) * (CP_SCALE/library[valid, None])
    # Identical operation order to registered cache: raw * (10000/library),
    # then log1p in float64, finally project/cast to float32.
    encoded = np.log1p(scaled).astype(np.float32)
    require(np.isfinite(encoded).all(), "Nonfinite encoded controls")
    return encoded, library, counts


def encode_controls(raw_counts, mapping, *, control_genes, limits=None):
    """Return exact registered model inputs and original measured-library depth."""
    mapping = _validate_mapping(mapping)
    require(_axis(control_genes, "control gene axis") == mapping.full_genes,
            "Control columns differ from the declared full gene order")
    limits = CountLimits() if limits is None else limits
    raw, _, _ = _canonical_controls(raw_counts, mapping, limits)
    encoded, library, _ = _control_encoding(raw, mapping)
    return encoded, library


def _normalized_domain(predicted):
    require(isinstance(predicted, np.ndarray) and predicted.ndim == 2
            and min(predicted.shape) > 0 and predicted.dtype == np.float32 and np.isfinite(predicted).all()
            and np.all(predicted >= 0), "Predictions must be finite nonnegative float32 log-CP values")
    require(np.all(predicted <= MAX_LOG_CP_FLOAT32), "Prediction exceeds the physical per-gene log-CP domain")
    log = predicted.astype(np.float64)
    normalized = np.expm1(log)
    # A valid real log-CP value can round to float32 by <= half an adjacent
    # spacing. exp(y)-exp(y-half_ulp) upper-bounds the resulting positive mass
    # error; include a small explicit float64 arithmetic allowance. This is a
    # validation tolerance ONLY, never a clamp or redistribution of predictions.
    higher = np.nextafter(predicted, np.float32(np.inf)).astype(np.float64)
    lower = np.nextafter(predicted, np.float32(-np.inf)).astype(np.float64)
    half_ulp = 0.5*np.maximum(higher-log, log-lower)
    tolerance = (np.exp(log)*(-np.expm1(-half_ulp))
                 + 8*np.finfo(np.float64).eps*np.maximum(1.0, normalized)).sum(axis=1)
    total = normalized.sum(axis=1)
    require(np.all(total <= CP_SCALE+tolerance), "Modeled normalized mass exceeds CP10000 beyond float-roundoff tolerance")
    return normalized, total, tolerance


def inverse_control_depth(predicted_log, control_encoded, control_modeled_counts,
                          normalization_libraries, *, strength=1.0):
    """Original-depth count expectations with exact float32 zero-effect identity.

    Strength is interpolation in EXPECTED RAW COUNTS, not an undocumented
    log-expression adjustment. A prediction coordinate exactly equal to its
    float32 encoded control returns the original integer count before rounding.
    No tolerance suppresses a nonzero representable model effect.

    A zero-head flow's sqrt->square computation can itself change positive
    coordinates by float32 roundoff. Those are NOT promised bitwise identity;
    use strength=0 or exact encoded controls for the exact raw-control baseline.
    """
    require(type(strength) in (int, float) and math.isfinite(strength) and 0 <= strength <= 1,
            "Strength must lie in [0,1]")
    normalized, mass, tolerance = _normalized_domain(predicted_log)
    encoded, counts = np.asarray(control_encoded), np.asarray(control_modeled_counts)
    library = np.asarray(normalization_libraries)
    require(encoded.shape == predicted_log.shape and encoded.dtype == np.float32
            and np.isfinite(encoded).all() and np.all(encoded >= 0), "Invalid control encoding")
    require(counts.shape == predicted_log.shape and counts.dtype.kind in "iu"
            and np.all(counts >= 0) and np.all(counts <= MAX_CELL_COUNTS), "Invalid modeled control counts")
    require(library.shape == (len(predicted_log),) and library.dtype.kind in "iu"
            and np.all(library >= 0) and np.all(library <= MAX_CELL_COUNTS), "Invalid normalization library")
    require(np.all(counts.astype(np.int64).sum(axis=1) <= library), "Modeled control counts exceed measured library")
    zero_library = library == 0
    reconstructed = np.zeros(counts.shape, dtype=np.float64)
    valid = ~zero_library
    reconstructed[valid] = counts[valid].astype(np.float64)*(CP_SCALE/library[valid, None])
    require(np.array_equal(encoded, np.log1p(reconstructed).astype(np.float32)),
            "Control encoding differs from original modeled counts and measured depth")
    require(not np.any(predicted_log[zero_library] > 0), "Positive prediction with zero measured control library is not identifiable")
    identity = predicted_log == encoded
    expected = normalized*(library.astype(np.float64)[:, None]/CP_SCALE)
    expected[identity] = counts[identity]
    if strength == 0:
        expected = counts.astype(np.float64)
    elif strength != 1:
        expected = (1.0-strength)*counts.astype(np.float64) + strength*expected
        expected[identity] = counts[identity]
    require(np.isfinite(expected).all() and np.all(expected >= 0), "Invalid inverse count expectation")
    return expected, {"zero_effect_coordinates": int(identity.sum()), "modeled_coordinates": int(identity.size),
        "all_coordinates_zero_effect": bool(identity.all()), "zero_measured_library_cells": int(zero_library.sum()),
        "max_modeled_normalized_mass": float(mass.max(initial=0)),
        "max_normalized_mass_roundoff_tolerance": float(tolerance.max(initial=0)),
        "roundoff_tolerance_policy": "sum(exp(log)*[1-exp(-half_max_adjacent_float32_spacing)])+8eps64*max(1,expm1(log))",
        "count_strength": float(strength)}


def stochastic_round(expected, *, seed):
    """Unbiased per-coordinate floor+Bernoulli; fixed row-major RNG order."""
    require(type(seed) is int and 0 <= seed < 2**64, "Seed must be an unsigned 64-bit integer")
    expected = np.asarray(expected)
    require(expected.ndim == 2 and expected.size > 0 and expected.dtype.kind == "f"
            and np.isfinite(expected).all() and np.all(expected >= 0)
            and np.all(expected <= MAX_CELL_COUNTS), "Invalid bounded rounding expectation")
    base = np.floor(expected).astype(np.int64)
    rng = np.random.default_rng(seed)
    emitted = base + (rng.random(expected.shape) < expected-base)
    require(np.all(emitted <= MAX_CELL_COUNTS), "Rounded coordinate exceeds count limit")
    return emitted.astype(np.int32)


def flow_to_counts(raw_counts, predicted_log, mapping, *, control_genes,
                   prediction_genes, seed, strength=1.0, limits=None):
    """Return canonical int32 CSR counts and an auditable, JSON-safe QC record.

    The input may be integer storage or exact integer-valued floating storage.
    Unmodeled entries retain their exact integer VALUES (sparse storage is
    canonicalized). All possible Bernoulli outcomes must meet hard row-count,
    density and storage bounds BEFORE RNG draws. There is no favorable-seed retry.
    Reproducibility requires the same row order, modeled axis, seed and batching.
    """
    mapping = _validate_mapping(mapping)
    require(_axis(control_genes, "control gene axis") == mapping.full_genes,
            "Control columns differ from the declared full gene order")
    require(_axis(prediction_genes, "prediction gene axis") == mapping.modeled_genes,
            "Prediction columns differ from the declared modeled gene order")
    require(type(seed) is int and 0 <= seed < 2**64, "Seed must be an unsigned 64-bit integer")
    limits = CountLimits() if limits is None else limits
    raw, source_library, memory = _canonical_controls(raw_counts, mapping, limits)
    require(isinstance(predicted_log, np.ndarray)
            and predicted_log.shape == (raw.shape[0], len(mapping.modeled_genes)), "Prediction matrix shape differs")
    encoded, normalization_library, original_modeled = _control_encoding(raw, mapping)
    expected, inverse_qc = inverse_control_depth(predicted_log, encoded, original_modeled,
                                                 normalization_library, strength=strength)
    is_modeled = np.zeros(raw.shape[1], dtype=bool)
    is_modeled[mapping.modeled_indices] = True
    retained = raw.copy()
    retained.data[is_modeled[retained.indices]] = 0
    retained.eliminate_zeros()
    retained_library = np.asarray(retained.sum(axis=1, dtype=np.int64)).reshape(-1)
    # Conservative ceiling bounds are checked before drawing. Rejecting based
    # on a particular random realization would otherwise encourage biased retries.
    upper = np.ceil(expected)
    lower_library = retained_library + np.floor(expected).sum(axis=1)
    upper_library = retained_library + upper.sum(axis=1)
    require(np.all(lower_library > 0), "A possible rounded output cell is empty")
    require(np.all(upper_library <= limits.max_cell_counts), "A possible rounded output exceeds one million cell counts")
    upper_nnz_row = np.diff(retained.indptr) + np.count_nonzero(expected > 0, axis=1)
    upper_nnz = int(upper_nnz_row.sum())
    upper_storage = 8*upper_nnz + 8*(raw.shape[0]+1)
    require(np.all(upper_nnz_row <= limits.max_genes_per_cell)
            and upper_nnz <= limits.max_density*raw.shape[0]*raw.shape[1]
            and upper_nnz <= limits.max_output_nnz, "Possible output density/nonzero bounds exceeded")
    require(upper_storage <= limits.max_output_bytes, "Possible output CSR storage byte bound exceeded")
    rounded = stochastic_round(expected, seed=seed)
    modeled = sparse.csr_matrix(rounded)
    replacement = sparse.csr_matrix((modeled.data, mapping.modeled_indices[modeled.indices], modeled.indptr),
                                    shape=raw.shape, dtype=np.int32)
    replacement.sort_indices()
    result = (retained + replacement).tocsr()
    result.sum_duplicates()
    result.sort_indices()
    result.eliminate_zeros()
    output_library = np.asarray(result.sum(axis=1, dtype=np.int64)).reshape(-1)
    require(result.dtype == np.int32 and result.has_canonical_format
            and np.all(result.data > 0) and np.all(output_library > 0)
            and np.all(output_library <= limits.max_cell_counts), "Invalid integer count output")
    # Verify protected/unmodeled values without densifying the native gene axis.
    unmodeled = np.flatnonzero(~is_modeled)
    require((result[:, unmodeled] != raw[:, unmodeled]).nnz == 0, "Unmodeled counts changed")
    identity_output = (result != raw).nnz == 0
    if strength == 0 or inverse_qc["all_coordinates_zero_effect"]:
        require(identity_output, "Zero-effect raw-count identity failed")
    result_bytes = int(result.data.nbytes+result.indices.nbytes+result.indptr.nbytes)
    require(result_bytes <= upper_storage <= limits.max_output_bytes, "Output storage estimate failed")
    source_norm = normalization_library
    output_norm = np.asarray(result[:, mapping.normalization_indices].sum(axis=1, dtype=np.int64)).reshape(-1)
    qc = {"schema": SCHEMA, "gene_mapping": mapping.provenance(), "limits": asdict(limits), **memory, **inverse_qc,
        "rows": raw.shape[0], "seed": seed, "rng": "numpy.default_rng.PCG64; fixed row-major modeled-axis draws",
        "expected_count_integerization": "unbiased floor+Bernoulli; exact integer identity coordinates copied before rounding",
        "normalization_library_policy": "original control sum over exact registered normalization genes",
        "unmodeled_gene_policy": "preserve exact original integer values; canonicalize sparse storage",
        "unmodeled_gene_count": int(len(unmodeled)), "unmodeled_changed_entries": 0,
        "zero_effect_raw_count_identity": bool(identity_output), "source_total_counts": int(source_library.sum()),
        "output_total_counts": int(output_library.sum()), "expected_total_counts": float(retained_library.sum()+expected.sum()),
        "source_min_library": int(source_library.min()), "source_max_library": int(source_library.max()),
        "output_min_library": int(output_library.min()), "output_max_library": int(output_library.max()),
        "max_absolute_native_library_drift": int(np.abs(output_library-source_library).max()),
        "mean_absolute_native_library_drift_fraction": float(np.mean(np.abs(output_library-source_library)/source_library)),
        "max_absolute_normalization_library_drift": int(np.abs(output_norm-source_norm).max()),
        "source_nnz": int(raw.nnz), "output_nnz": int(result.nnz), "output_nnz_upper_bound": upper_nnz,
        "output_storage_bytes": result_bytes, "output_storage_upper_bound": upper_storage,
        "output_density": float(result.nnz/(raw.shape[0]*raw.shape[1])),
        "clipped_coordinates": 0, "dropped_counts": 0, "redistributed_counts": 0,
        "seed_retries": 0, "full_or_normalization_library_conservation_claimed": False,
        "independent_true_count_recovery_claimed": False, "official_file_validation_performed": False,
        "submission_ready": False}
    return result, qc
