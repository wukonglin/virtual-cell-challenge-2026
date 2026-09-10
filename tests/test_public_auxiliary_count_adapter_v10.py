"""Synthetic native-count adapter tests; no public expression or model loading."""
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import public_auxiliary_count_adapter_v10 as adapter


FULL = tuple("ABCDEFGH")
MODELED = ("F", "B")
NORMALIZATION = ("G", "A", "F", "C", "B")


@pytest.fixture
def mapping():
    return adapter.build_gene_mapping(FULL, MODELED, NORMALIZATION, expected_sizes=(8, 2, 5))


@pytest.fixture
def limits():
    return adapter.CountLimits(max_density=1.0, max_genes_per_cell=8)


@pytest.fixture
def raw():
    return np.asarray([[5, 10, 15, 7, 0, 20, 50, 3], [0, 0, 10, 5, 2, 0, 30, 4]], dtype=np.int32)


def emit(raw, prediction, mapping, limits, **kwargs):
    return adapter.flow_to_counts(raw, prediction, mapping, control_genes=FULL,
                                 prediction_genes=MODELED, seed=53, limits=limits, **kwargs)


def encode(raw, mapping, limits):
    return adapter.encode_controls(raw, mapping, control_genes=FULL, limits=limits)


def test_exact_mapping_order_and_default_production_dimensions(mapping):
    np.testing.assert_array_equal(mapping.modeled_indices, [5, 1])
    np.testing.assert_array_equal(mapping.normalization_indices, [6, 0, 5, 2, 1])
    assert not mapping.modeled_indices.flags.writeable
    assert not mapping.provenance()["dimensions_match_current_vcc"]
    with pytest.raises(ValueError, match="dimensions"):
        adapter.build_gene_mapping(FULL, MODELED, NORMALIZATION)


@pytest.mark.parametrize("full,modeled,norm,indices,match", [
    (tuple("ABCDEFGA"), MODELED, NORMALIZATION, None, "full"),
    (FULL, ("F", "F"), NORMALIZATION, None, "modeled"),
    (FULL, MODELED, ("G", "A", "F", "C", "C"), None, "normalization"),
    (FULL, ("F", "b"), NORMALIZATION, None, "Missing"),
    (FULL, ("F", " B"), NORMALIZATION, None, "exact"),
    (FULL, ("F", "H"), NORMALIZATION, None, "Missing"),
    (FULL, MODELED, ("G", "A", "F", "C", "outside"), None, "Missing"),
    (FULL, MODELED, NORMALIZATION, [1, 5], "mapping"),
    (FULL, MODELED, NORMALIZATION, [5., 1.], "mapping"),
    (FULL, MODELED, NORMALIZATION, [True, False], "mapping"),
])
def test_axis_corruption_or_aliasing_rejected(full, modeled, norm, indices, match):
    with pytest.raises(ValueError, match=match):
        adapter.build_gene_mapping(full, modeled, norm, modeled_indices=indices, expected_sizes=(8, 2, 5))


def test_hand_computed_normalization_uses_measured_not_full_or_modeled_library(raw, mapping, limits):
    encoded, library = encode(raw, mapping, limits)
    np.testing.assert_array_equal(library, [100, 40])
    expected = np.log1p(np.asarray([[2000., 1000.], [0., 0.]])).astype(np.float32)
    np.testing.assert_array_equal(encoded, expected)
    assert not np.array_equal(library, raw.sum(1))
    assert not np.array_equal(library, raw[:, mapping.modeled_indices].sum(1))
    prediction = np.log1p(np.asarray([[825., 1260.], [100., 200.]])).astype(np.float32)
    expected_counts, qc = adapter.inverse_control_depth(prediction, encoded,
                                                        raw[:, mapping.modeled_indices], library)
    np.testing.assert_allclose(expected_counts, [[8.25, 12.6], [.4, .8]], rtol=1e-6, atol=1e-7)
    assert qc["zero_effect_coordinates"] == 0


def test_normalization_arithmetic_matches_frozen_training_primitive(raw, mapping, limits):
    from prepare_public_flow_pilot import normalize_counts
    encoded, _ = encode(raw, mapping, limits)
    norm = normalize_counts(raw[:, mapping.normalization_indices])
    positions = [NORMALIZATION.index(g) for g in MODELED]
    np.testing.assert_array_equal(encoded, norm[:, positions])


@pytest.mark.parametrize("storage", ["dense_int32", "dense_int64", "dense_float32", "dense_float64", "dense_uint64", "csr", "csc", "csr_array"])
def test_zero_effect_integer_identity_every_storage(raw, mapping, limits, storage):
    if storage.startswith("dense_"):
        supplied = raw.astype(storage.split("_", 1)[1])
    else:
        supplied = {"csr": sparse.csr_matrix, "csc": sparse.csc_matrix,
                    "csr_array": sparse.csr_array}[storage](raw)
    encoded, _ = encode(supplied, mapping, limits)
    output, qc = emit(supplied, encoded, mapping, limits)
    np.testing.assert_array_equal(output.toarray(), raw)
    assert output.dtype == np.int32 and output.has_canonical_format
    assert qc["zero_effect_raw_count_identity"]
    assert qc["unmodeled_changed_entries"] == 0
    assert qc["max_absolute_native_library_drift"] == 0


def test_strength_zero_exact_identity_and_partial_strength_count_space(raw, mapping, limits):
    encoded, library = encode(raw, mapping, limits)
    prediction = np.log1p(np.asarray([[100., 200.], [500., 500.]])).astype(np.float32)
    output, qc = emit(raw, prediction, mapping, limits, strength=0)
    np.testing.assert_array_equal(output.toarray(), raw)
    assert qc["zero_effect_raw_count_identity"] and qc["count_strength"] == 0
    full, _ = adapter.inverse_control_depth(prediction, encoded, raw[:, mapping.modeled_indices], library)
    half, _ = adapter.inverse_control_depth(prediction, encoded, raw[:, mapping.modeled_indices], library, strength=.5)
    np.testing.assert_allclose(half, .5*(full+raw[:, mapping.modeled_indices]))


def test_unmodeled_values_exact_and_library_drift_is_reported(raw, mapping, limits):
    prediction = np.log1p(np.asarray([[500., 250.], [300., 200.]])).astype(np.float32)
    output, qc = emit(raw, prediction, mapping, limits)
    unmodeled = [i for i, gene in enumerate(FULL) if gene not in MODELED]
    np.testing.assert_array_equal(output[:, unmodeled].toarray(), raw[:, unmodeled])
    assert qc["max_absolute_native_library_drift"] > 0
    assert not qc["full_or_normalization_library_conservation_claimed"]
    assert not qc["independent_true_count_recovery_claimed"]
    assert qc["clipped_coordinates"] == qc["dropped_counts"] == qc["redistributed_counts"] == qc["seed_retries"] == 0


def test_reordered_full_axis_with_correct_maps_preserves_gene_results(raw, mapping, limits):
    prediction = np.log1p(np.asarray([[400., 200.], [250., 100.]])).astype(np.float32)
    reference, _ = emit(raw, prediction, mapping, limits)
    reordered = adapter.build_gene_mapping(FULL[::-1], MODELED, NORMALIZATION, expected_sizes=(8, 2, 5))
    result, _ = adapter.flow_to_counts(raw[:, ::-1], prediction, reordered,
        control_genes=FULL[::-1], prediction_genes=MODELED, seed=53, limits=limits)
    np.testing.assert_array_equal(result.toarray()[:, ::-1], reference.toarray())
    with pytest.raises(ValueError, match="Control columns"):
        adapter.flow_to_counts(raw, prediction, reordered, control_genes=FULL,
                               prediction_genes=MODELED, seed=53, limits=limits)
    with pytest.raises(ValueError, match="Prediction columns"):
        adapter.flow_to_counts(raw, prediction, mapping, control_genes=FULL,
                               prediction_genes=MODELED[::-1], seed=53, limits=limits)


def test_sparse_noncanonical_input_counts_preserved_without_mutating_input(mapping, limits):
    # Two entries in unmodeled A sum to five; explicit zero is removable storage.
    raw = sparse.csr_matrix((np.asarray([2, 3, 10, 20, 0], dtype=np.int32),
                            np.asarray([0, 0, 1, 5, 7]), np.asarray([0, 5])), shape=(1, 8))
    before_data, before_indices, before_indptr = raw.data.copy(), raw.indices.copy(), raw.indptr.copy()
    encoded, _ = encode(raw, mapping, limits)
    output, _ = emit(raw, encoded, mapping, limits)
    np.testing.assert_array_equal(output.toarray(), [[5, 10, 0, 0, 0, 20, 0, 0]])
    np.testing.assert_array_equal(raw.data, before_data)
    np.testing.assert_array_equal(raw.indices, before_indices)
    np.testing.assert_array_equal(raw.indptr, before_indptr)


def test_sparse_surplus_capacity_is_not_pruned_on_the_caller_object(mapping, limits):
    raw = sparse.csr_matrix((np.asarray([20, 5, 10], dtype=np.int32),
                            np.asarray([5, 0, 1]), np.asarray([0, 3])), shape=(1, 8))
    raw.data = np.concatenate((raw.data, np.asarray([0, 0], dtype=np.int32)))
    raw.indices = np.concatenate((raw.indices, np.asarray([0, 0], dtype=np.int32)))
    original = (raw.data, raw.indices, raw.indptr)
    before = tuple(v.copy() for v in original)
    encoded, _ = encode(raw, mapping, limits)
    output, _ = emit(raw, encoded, mapping, limits)
    np.testing.assert_array_equal(output.toarray(), [[5, 10, 0, 0, 0, 20, 0, 0]])
    for current, previous, value in zip((raw.data, raw.indices, raw.indptr), original, before):
        assert current is previous
        np.testing.assert_array_equal(current, value)


@pytest.mark.parametrize("corrupt", ["index", "pointer", "negative_pointer"])
def test_malformed_sparse_structure_fails_before_sparse_arithmetic(raw, mapping, limits, corrupt):
    source = sparse.csr_matrix(raw)
    if corrupt == "index":
        source.indices[0] = 100
    elif corrupt == "pointer":
        source.indptr[1] = source.indptr[-1]+1
    else:
        source.indptr[-1] = -1
    with pytest.raises(ValueError, match="Sparse|sparse"):
        emit(source, np.zeros((2, 2), dtype=np.float32), mapping, limits)


def test_zero_measured_library_preserves_only_identifiable_zero_outputs(mapping, limits):
    raw = np.asarray([[0, 0, 0, 7, 0, 0, 0, 2]], dtype=np.int32)
    encoded, library = encode(raw, mapping, limits)
    np.testing.assert_array_equal(encoded, np.zeros((1, 2), dtype=np.float32))
    np.testing.assert_array_equal(library, [0])
    output, qc = emit(raw, encoded, mapping, limits)
    np.testing.assert_array_equal(output.toarray(), raw)
    assert qc["zero_measured_library_cells"] == 1
    prediction = encoded.copy()
    prediction[0, 0] = .1
    with pytest.raises(ValueError, match="zero measured"):
        emit(raw, prediction, mapping, limits)


def test_physical_per_gene_and_modeled_mass_domain_with_derived_float_tolerance():
    encoded = np.asarray([[adapter.MAX_LOG_CP_FLOAT32, 0.]], dtype=np.float32)
    counts, library = np.asarray([[100, 0]], dtype=np.int32), np.asarray([100], dtype=np.int64)
    expected, qc = adapter.inverse_control_depth(encoded, encoded, counts, library)
    assert expected[0, 0] == 100
    assert 0 < qc["max_normalized_mass_roundoff_tolerance"] < .01
    above = encoded.copy()
    above[0, 0] = np.nextafter(adapter.MAX_LOG_CP_FLOAT32, np.float32(np.inf))
    with pytest.raises(ValueError, match="per-gene"):
        adapter.inverse_control_depth(above, encoded, counts, library)
    impossible = np.full((1, 2), np.float32(np.log1p(6000)))
    with pytest.raises(ValueError, match="mass exceeds"):
        adapter.inverse_control_depth(impossible, encoded, counts, library)


def test_exact_identity_does_not_hide_representable_small_effect(raw, mapping, limits):
    encoded, library = encode(raw, mapping, limits)
    changed = encoded.copy()
    changed[0, 0] = np.nextafter(changed[0, 0], np.float32(np.inf))
    expected, qc = adapter.inverse_control_depth(changed, encoded, raw[:, mapping.modeled_indices], library)
    assert qc["zero_effect_coordinates"] == 3
    assert expected[0, 0] != raw[0, 5]


def test_zero_head_flow_pipeline_scopes_identity_to_strength_zero_or_bitwise_input(mapping, limits):
    import torch
    import public_auxiliary_flow_v10 as flow
    raw = np.zeros((64, 8), dtype=np.int32)
    raw[:, 5] = np.arange(1, 65)
    raw[:, 1] = 7
    raw[:, 0] = 1000-raw[:, 5]-raw[:, 1]
    raw[:, 3] = 1
    encoded, library = encode(raw, mapping, limits)
    model = flow.AuxiliaryConditionalFlow(flow.PilotConfig(2, 3, hidden_dim=8, layers=1, steps=1))
    prediction = flow.sample_heun(model, torch.from_numpy(encoded), torch.ones(64, 3),
                                 torch.zeros(64, 4), steps=1).numpy()
    np.testing.assert_allclose(prediction, encoded, atol=1e-6, rtol=1e-6)
    expected, qc = adapter.inverse_control_depth(prediction, encoded, raw[:, mapping.modeled_indices], library)
    exact = prediction == encoded
    assert qc["zero_effect_coordinates"] == int(exact.sum())
    assert np.any(~exact)  # Multiple positive float32 values really round differently.
    assert np.any(expected[~exact] != raw[:, mapping.modeled_indices][~exact])
    baseline, baseline_qc = emit(raw, prediction, mapping, limits, strength=0)
    np.testing.assert_array_equal(baseline.toarray(), raw)
    assert baseline_qc["zero_effect_raw_count_identity"]


def test_zero_effect_exact_at_one_million_counts(mapping, limits):
    raw = np.asarray([[0, 0, 0, 0, 0, 1_000_000, 0, 0]], dtype=np.int32)
    encoded, _ = encode(raw, mapping, limits)
    result, _ = emit(raw, encoded, mapping, limits)
    np.testing.assert_array_equal(result.toarray(), raw)


def test_output_over_one_million_rejected_before_any_rng(mapping, limits, monkeypatch):
    raw = np.asarray([[0, 10, 0, 999980, 0, 10, 0, 0]], dtype=np.int32)
    prediction = np.asarray([[adapter.MAX_LOG_CP_FLOAT32, 0]], dtype=np.float32)
    def forbidden(*args, **kwargs):
        raise AssertionError("RNG must not be consulted after a deterministic preflight failure")
    monkeypatch.setattr(adapter, "stochastic_round", forbidden)
    with pytest.raises(ValueError, match="one million"):
        emit(raw, prediction, mapping, limits)


def test_possible_empty_output_rejected_not_seed_selected(mapping, limits):
    raw = np.asarray([[0, 0, 0, 0, 0, 1, 0, 0]], dtype=np.int32)
    prediction = np.log1p(np.asarray([[100., 0.]])).astype(np.float32)
    with pytest.raises(ValueError, match="empty"):
        emit(raw, prediction, mapping, limits)


def test_stochastic_round_unbiased_and_seeded_without_global_rng_mutation():
    expected = np.broadcast_to(np.asarray([[3.25, .7]], dtype=np.float64), (100000, 2))
    np.random.seed(802)
    before = np.random.get_state()
    first = adapter.stochastic_round(expected, seed=32)
    second = adapter.stochastic_round(expected, seed=32)
    different = adapter.stochastic_round(expected, seed=33)
    after = np.random.get_state()
    assert np.array_equal(first, second) and not np.array_equal(first, different)
    np.testing.assert_allclose(first.mean(0), [3.25, .7], atol=.004)
    assert before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2:] == after[2:]
    assert first.dtype == np.int32


@pytest.mark.parametrize("mutation,match", [
    (lambda a: a.astype(np.float64)+.1, "integer"),
    (lambda a: np.where(a > 0, -a, a), "integer"),
    (lambda a: np.where(a > 0, np.nan, a), "integer"),
    (lambda a: np.where(a > 0, np.inf, a), "integer"),
    (lambda a: a.astype(bool), "integer"),
    (lambda a: a.astype(np.complex64), "integer"),
    (lambda a: np.full_like(a, 200000), "libraries"),
    (lambda a: np.zeros_like(a), "positive libraries"),
    (lambda a: a[:, :-1], "shape"),
])
def test_invalid_raw_counts_rejected_before_prediction(raw, mapping, limits, mutation, match):
    with pytest.raises(ValueError, match=match):
        emit(mutation(raw), np.zeros((2, 2), dtype=np.float32), mapping, limits)


@pytest.mark.parametrize("kind,match", [("negative", "nonnegative"), ("nan", "finite"),
    ("double", "float32"), ("wrong_shape", "shape")])
def test_invalid_predictions_rejected(raw, mapping, limits, kind, match):
    prediction = np.zeros((2, 2), dtype=np.float32)
    if kind == "negative":
        prediction[0, 0] = -.01
    elif kind == "nan":
        prediction[0, 0] = np.nan
    elif kind == "double":
        prediction = prediction.astype(np.float64)
    else:
        prediction = prediction[:, :1]
    with pytest.raises(ValueError, match=match):
        emit(raw, prediction, mapping, limits)


@pytest.mark.parametrize("field,value,match", [
    ("max_batch_cells", 1, "batch"), ("max_input_bytes", 1, "Input storage"),
    ("max_working_bytes", 1, "working-memory"), ("max_output_bytes", 1, "storage"),
    ("max_genes_per_cell", 2, "density"), ("max_density", .1, "density"),
    ("max_output_nnz", 3, "bounds"),
])
def test_resource_density_limits_fail_without_altering_counts(raw, mapping, limits, field, value, match):
    before = raw.copy()
    encoded, _ = encode(raw, mapping, limits)
    with pytest.raises(ValueError, match=match):
        emit(raw, encoded, mapping, replace(limits, **{field: value}))
    np.testing.assert_array_equal(raw, before)


@pytest.mark.parametrize("value", [-.1, 1.01, True, float("nan"), float("inf")])
def test_invalid_strength_rejected(raw, mapping, limits, value):
    encoded, _ = encode(raw, mapping, limits)
    with pytest.raises(ValueError, match="Strength"):
        emit(raw, encoded, mapping, limits, strength=value)


@pytest.mark.parametrize("value", [-1, True, 2**64, 2.5])
def test_invalid_seed_rejected(value):
    with pytest.raises(ValueError, match="Seed"):
        adapter.stochastic_round(np.ones((1, 1)), seed=value)


@pytest.mark.parametrize("shape", [(0, 2), (2, 0)])
def test_empty_inverse_prediction_rejected(shape):
    with pytest.raises(ValueError, match="Predictions"):
        adapter.inverse_control_depth(np.zeros(shape, dtype=np.float32),
            np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.int32),
            np.ones(shape[0], dtype=np.int64))


def test_forged_control_encoding_or_mapping_rejected(raw, mapping, limits):
    encoded, library = encode(raw, mapping, limits)
    forged = encoded.copy()
    forged[0, 0] += .1
    with pytest.raises(ValueError, match="Control encoding differs"):
        adapter.inverse_control_depth(encoded, forged, raw[:, mapping.modeled_indices], library)
    altered = replace(mapping, modeled_indices=mapping.modeled_indices[::-1])
    with pytest.raises(ValueError, match="mapping"):
        emit(raw, encoded, altered, limits)


def test_synthetic_real_dimension_sparse_identity_and_18051_untouched():
    full = tuple(f"gene{i}" for i in range(18533))
    modeled = tuple(full[i] for i in range(100, 582))
    norm = tuple(full[:7097])
    mapping = adapter.build_gene_mapping(full, modeled, norm)
    raw = sparse.csr_matrix((np.asarray([9, 11, 13, 17], dtype=np.int32),
                             (np.asarray([0, 0, 1, 1]), np.asarray([100, 8000, 101, 18000]))), shape=(2, 18533))
    encoded, _ = adapter.encode_controls(raw, mapping, control_genes=full)
    result, qc = adapter.flow_to_counts(raw, encoded, mapping, control_genes=full,
                                      prediction_genes=modeled, seed=5)
    assert (result != raw).nnz == 0
    assert qc["unmodeled_gene_count"] == 18051
    assert qc["gene_mapping"]["dimensions_match_current_vcc"]
    assert not qc["submission_ready"] and not qc["official_file_validation_performed"]


def test_no_io_pickle_network_thinning_or_upload_code():
    text = Path(adapter.__file__).read_text()
    for token in ("torch.load(", "torch.save(", "requests.", "subprocess.", "wandb.", "open(", "np.clip("):
        assert token not in text
