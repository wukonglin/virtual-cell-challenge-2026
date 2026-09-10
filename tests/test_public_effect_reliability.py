"""Synthetic-only reliability estimands, pool isolation, weighting and RNG."""
from copy import deepcopy
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import public_effect_reliability as reliability


def arrays(*, n=8, targets=("A", "B"), batches=2, effect=1., random=False):
    rng = np.random.default_rng(117)
    source_axis, target_axis, batch_axis = [], [], []
    pools = {name: [] for name in ("control", "context_control", "treated")}
    rows = {name: [] for name in pools}; labels = {name: [] for name in pools}
    for source in ("replogle_k562", "nadig_jurkat"):
        for target_index, target in enumerate(targets):
            for batch in range(batches):
                group = len(source_axis)
                source_axis.append(source); target_axis.append(target); batch_axis.append(f"batch{batch}")
                for name, offset, size in (("control", 0, 32), ("context_control", 32, 32), ("treated", 64+target_index*40, n)):
                    if random and name != "treated":
                        # Pools are globally identical across targets within source/batch.
                        local = np.random.default_rng(19+batch*7+(0 if source == "replogle_k562" else 101)+(0 if name == "control" else 1))
                        value = local.uniform(1, 4, (size, 3))
                    elif random:
                        value = rng.uniform(1, 4, (size, 3)) + effect
                    else:
                        value = np.full((size, 3), 2. + (effect if name == "treated" else 0.))
                    pools[name].append(value.astype(np.float32))
                    rows[name].extend(range(batch*1000+offset, batch*1000+offset+size))
                    labels[name].extend([group]*size)
    result = {"genes": np.array(["G1", "G2", "G3"]), "sources": np.array(source_axis), "targets": np.array(target_axis), "batches": np.array(batch_axis),
              "sham_control": np.array([[np.nan]]), "context": np.array([[np.nan]])}
    for name in pools:
        result[name] = np.concatenate(pools[name]); result[name+"_rows"] = np.array(rows[name], dtype=np.int64)
        result["group" if name == "treated" else name+"_group"] = np.array(labels[name], dtype=np.int64)
    return result


def test_constant_known_effect_exact_signal_and_identity_null():
    result = reliability.compute_reliability(arrays(effect=1.))
    metrics = result["overall"]
    for name in ("observed_group_effect_mse", "observed_target_pooled_effect_mse", "observed_between_batch_crossdot", "observed_within_batch_half_crossdot"):
        assert metrics[name]["mean"] == pytest.approx(1)
    assert metrics["null_group_effect_mse"]["mean"] == 0
    assert metrics["null_within_batch_half_agreement"]["mean"] is None
    assert metrics["observed_within_batch_half_agreement"]["mean"] == 1
    assert result["fitting_performed"] is False and result["sham_values_used_in_statistics"] is False
    assert result["input_counts"]["control_unique_cells"] == 128
    assert result["input_counts"]["control_entries"] == 256
    assert set(result["aggregate"]) == set(reliability.AGGREGATE_KEYS)
    assert len(result["sources"]) == 2
    assert len(result["seed_summaries"]) == 32
    assert result["aggregate"]["real_group_energy"] == 1
    assert result["sources"][0]["aggregate"]["null_half_agreement"] is None


def test_negative_crossdots_and_disagreement_are_not_truncated():
    result = reliability.half_statistics(np.array([1., -1]), np.array([-1., 1]))
    assert result == {"crossdot": -1., "agreement": -1., "disagreement_mse": 4.}
    assert reliability.between_batch_crossdot([[1, -1], [-1, 1]]) == -1
    assert reliability.half_statistics(np.zeros(2), np.zeros(2))["agreement"] is None


def test_pool_first_then_square_differs_from_mean_group_energy():
    data = arrays(targets=("A",), effect=0.)
    for group in range(4):
        data["treated"][data["group"] == group] += 1 if group % 2 == 0 else -1
    result = reliability.compute_reliability(data)["overall"]
    assert result["observed_group_effect_mse"]["mean"] == 1
    assert result["observed_target_pooled_effect_mse"]["mean"] == 0
    assert result["observed_between_batch_crossdot"]["mean"] == -1


def test_fixed_seeds_reproduce_and_do_not_mutate_input():
    data = arrays(n=9, random=True)
    original = deepcopy(data)
    first, second = reliability.compute_reliability(data), reliability.compute_reliability(data)
    assert first == second
    for key in data:
        np.testing.assert_array_equal(data[key], original[key])
    assert first["groups"][0]["treated_half_sizes"] == [4, 5]
    assert first["groups"][0]["anchor_half_sizes"] == [16, 16]


def test_full32_null_energy_invariant_across_partitions():
    result = reliability.compute_reliability(arrays(n=32, random=True))
    for group in result["groups"]:
        metric = group["metrics"]["null_group_effect_mse"]
        assert metric["max"]-metric["min"] < 1e-12
    assert result["per_seed"][0]["seed"] == 20260920
    assert result["per_seed"][-1]["seed"] == 20260951


def test_global_ntc_permutations_are_once_per_source_batch_seed(monkeypatch):
    original = reliability._permutation
    calls = []
    def spy(size, seed, namespace, source, batch, target=None):
        calls.append((seed, namespace, source, batch, target))
        return original(size, seed, namespace, source, batch, target)
    monkeypatch.setattr(reliability, "_permutation", spy)
    reliability.compute_reliability(arrays())
    ntc = [call for call in calls if call[1] != "treated"]
    assert len(ntc) == 32*2*2*2
    assert len(ntc) == len(set(ntc))
    assert len([call for call in calls if call[1] == "treated"]) == 32*8


def test_context_and_anchor_halves_are_disjoint_not_shared_full_reference(monkeypatch):
    data = arrays(targets=("A",), n=8, effect=0.)
    # Anchor rows alternate0/4; deterministic identity order first16mean0,last16mean4.
    for group in range(4):
        row = np.flatnonzero(data["control_group"] == group)
        data["control"][row[:16]] = 0
        data["control"][row[16:]] = 4
    monkeypatch.setattr(reliability, "_permutation", lambda size, *a, **kw: np.arange(size))
    result = reliability.compute_reliability(data)["overall"]
    assert result["observed_group_effect_mse"]["mean"] == 0
    assert result["observed_within_batch_half_crossdot"]["mean"] == -4
    assert result["null_within_batch_half_crossdot"]["mean"] == -4
    assert result["observed_within_batch_half_disagreement_mse"]["mean"] == 16


def test_cell_row_order_does_not_change_seed_partition_statistics():
    data = arrays(n=11, random=True)
    expected = reliability.compute_reliability(data)
    for name in ("control", "context_control", "treated"):
        order = np.arange(len(data[name]))[::-1]
        data[name] = data[name][order]
        data[name+"_rows"] = data[name+"_rows"][order]
        label = "group" if name == "treated" else name+"_group"
        data[label] = data[label][order]
    assert reliability.compute_reliability(data) == expected


@pytest.mark.parametrize("change", ["negative", "nonfinite", "pool_overlap", "pool_reuse_changed", "treated_repeat", "source", "target_mismatch", "wrong_n"])
def test_bad_or_nonindependent_arrays_rejected(change):
    data = arrays()
    if change == "negative": data["treated"][0,0] = -1
    elif change == "nonfinite": data["control"][0,0] = np.nan
    elif change == "pool_overlap": data["context_control_rows"][0] = data["control_rows"][0]
    elif change == "pool_reuse_changed": data["context_control"][64,0] = 99
    elif change == "treated_repeat": data["treated_rows"][16] = data["treated_rows"][0]
    elif change == "source": data["sources"][0] = "hepg2"
    elif change == "target_mismatch": data["targets"][4:] = "C"
    elif change == "wrong_n": data["group"][0] = 1
    with pytest.raises(ValueError): reliability.compute_reliability(data)


@pytest.mark.parametrize("seeds", [(1,), reliability.SEEDS[::-1], tuple(range(32))])
def test_unregistered_seed_schedules_fail(seeds):
    with pytest.raises(ValueError, match="preregistered"):
        reliability.compute_reliability(arrays(), seeds=seeds)


def test_partition_ranges_are_descriptive_and_signed():
    summary = reliability.summarize_partitions([-2., None, 2.])
    assert summary["mean"] == 0
    assert summary["min"] == -2 and summary["max"] == 2
    assert summary["defined_partitions"] == 2
    assert summary["partitions"] == 3
    assert summary["range_is_confidence_interval"] is False
