"""Synthetic-only common-target, disjoint-pool, sealed row admission tests."""
import copy
import json
from pathlib import Path
import sys
import h5py
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_public_control_anchor_v2 as prep


@pytest.fixture
def cohort(tmp_path, monkeypatch):
    policy = {**prep.POLICY, "expected_common_targets": 2, "max_common_targets": 2,
              "min_batches_per_target_source": 2, "max_batches_per_target_source": 2,
              "min_treated_per_batch": 2, "max_treated_per_group": 3,
              "context_controls_per_batch": 2, "anchor_controls_per_batch": 2, "sham_controls_per_batch": 2,
              "core_controls_per_batch": 4, "total_controls_with_sham_per_batch": 6, "output_gene_count": 3}
    monkeypatch.setattr(prep, "POLICY", policy)
    seal = tmp_path / "seal.json"
    prep.old.write_json(seal, {"partition": {"global_exclusion_targets": [f"DENIED{i}" for i in range(772)]}})
    monkeypatch.setattr(prep.old, "SEALS", {"seal.json": prep.old.sha_file(seal)})
    monkeypatch.setattr(prep.old, "PRIOR_TARGETS", [])
    specs = []
    for source in prep.SOURCE_IDS:
        path = tmp_path / (source + ".h5ad")
        rows = [(batch, target, i) for batch in ("batch1", "batch2") for target, n in (("control", 6), ("A", 3), ("B", 3), ("DENIED0", 2)) for i in range(n)]
        expression = np.tile([1., 2., 3., 4., 5.], (len(rows), 1))
        expression[[i for i, (_, target, _) in enumerate(rows) if target == "DENIED0"]] = np.nan
        with h5py.File(path, "w") as handle:
            obs, var = handle.create_group("obs"), handle.create_group("var")
            obs.create_dataset("cell", data=np.array([f"{source}-{b}-{t}-{i}" for b, t, i in rows], dtype="S"))
            obs.create_dataset("target", data=np.array([t for _, t, _ in rows], dtype="S"))
            obs.create_dataset("batch", data=np.array([b for b, _, _ in rows], dtype="S"))
            var.create_dataset("gene", data=np.array(["G1", "G2", "G3", "DUP", "DUP"], dtype="S"))
            handle.create_dataset("X", data=expression)
        receipt = tmp_path / (source + ".json")
        spec = {"id": source, "role": "train", "path": str(path), "sha256": prep.old.sha_file(path), "size_bytes": path.stat().st_size,
                "cell_key": "cell", "target_key": "target", "batch_key": "batch", "gene_key": "gene", "control_label": "control"}
        prep.old.write_json(receipt, {"files": [{"name": path.name, "sha256": spec["sha256"], "size_bytes": spec["size_bytes"]}]})
        spec.update(receipt_name=path.name, receipt_path=str(receipt), receipt_sha256=prep.old.sha_file(receipt))
        specs.append(spec)
    relative = "artifacts/public_flow/h1_dev_pilot_v1/sources.json"
    old_manifest = tmp_path / relative
    old_manifest.parent.mkdir(parents=True)
    prep.old.write_json(old_manifest, {"sources": specs})
    monkeypatch.setattr(prep, "PROTECTED_SEALS", {"seal.json": prep.old.sha_file(seal), relative: prep.old.sha_file(old_manifest)})
    protocol = tmp_path / "protocol.md"
    protocol.write_text("Synthetic immutable protocol; independent training reference.")
    return tmp_path, specs, protocol


def registered(cohort):
    root, specs, protocol = cohort
    output = root / "v2"
    audit = prep.register(root, output, protocol)
    return output, audit


def materialized(cohort):
    output, audit = registered(cohort)
    cache = output / "cache.npz"
    receipt = prep.materialize(output / "contract.json", audit["contract_sha256"], output / "sources.json",
                               audit["source_manifest_sha256"], cache, repo=cohort[0])
    return output, audit, cache, receipt


def forbid_x(monkeypatch):
    original = h5py.Dataset.__getitem__
    def guarded(dataset, key):
        if dataset.name == "/X" or dataset.name.startswith("/X/"):
            pytest.fail("Expression decoded before row/source authorization")
        return original(dataset, key)
    monkeypatch.setattr(h5py.Dataset, "__getitem__", guarded)


def test_registration_metadata_only_common_cohort_and_pools(cohort, monkeypatch):
    forbid_x(monkeypatch)
    output, audit = registered(cohort)
    contract = prep.old.load_json(output / "contract.json")
    assert audit["metadata_only"] and audit["common_target_count"] == 2
    assert contract["common_targets"] == ["A", "B"]
    assert contract["shared_genes"] == ["G1", "G2", "G3"]
    assert set(s["id"] for s in contract["sources"]) == set(prep.SOURCE_IDS)
    for source in contract["sources"]:
        assert len(source["groups"]) == 4
        batches = {}
        for group in source["groups"]:
            pools = [set(group[key]) for key in ("context_control_rows", "control_rows", "sham_control_rows")]
            assert all(not pools[i] & pools[j] for i in range(3) for j in range(i))
            if group["batch"] in batches:
                assert pools == batches[group["batch"]]
            batches[group["batch"]] = pools


def test_materialize_only_admitted_rows_and_context_reconstruction(cohort, monkeypatch):
    original = h5py.Dataset.__getitem__
    calls = []
    def guarded(dataset, key):
        if dataset.name == "/X":
            calls.append((dataset.file.filename, key[0]))
            assert int(key[0]) % 14 < 12  # Poisoned excluded rows are last two per batch.
        return original(dataset, key)
    monkeypatch.setattr(h5py.Dataset, "__getitem__", guarded)
    output, audit, cache, receipt = materialized(cohort)
    arrays, contract = prep.load_cache(cache, receipt["cache_sha256"], output / "contract.json", audit["contract_sha256"],
                                      output / "sources.json", audit["source_manifest_sha256"], repo=cohort[0])
    assert calls and receipt["rpe1_h1_hepg2_challenge2026_feng_opened"] is False
    assert arrays["context"].shape == (8, 6)
    assert arrays["control"].shape == (16, 3)
    assert arrays["treated"].shape == (24, 3)
    assert "independent TRAINING anchor/reference" in contract["policy"]["fit_response"]


def test_authenticate_second_source_before_any_expression(cohort, monkeypatch):
    output, audit = registered(cohort)
    second = Path(cohort[1][1]["path"])
    with second.open("ab") as stream:
        stream.write(b"changed")
    forbid_x(monkeypatch)
    with pytest.raises(ValueError, match="authentication"):
        prep.materialize(output / "contract.json", audit["contract_sha256"], output / "sources.json", audit["source_manifest_sha256"], output / "cache.npz", repo=cohort[0])


@pytest.mark.parametrize("mutation", ["pool_overlap", "cross_target_pool", "excluded_target", "drop_source_target", "policy", "reserved_role", "expression_flag"])
def test_contract_rejects_unsafe_admission_before_x(cohort, monkeypatch, mutation):
    output, audit = registered(cohort)
    contract = prep.old.load_json(output / "contract.json")
    manifest = prep.old.load_json(output / "sources.json")
    group = contract["sources"][0]["groups"][0]
    if mutation == "pool_overlap": group["control_rows"] = group["context_control_rows"]
    elif mutation == "cross_target_pool":
        other = next(g for g in contract["sources"][0]["groups"] if g["target"] != group["target"] and g["batch"] == group["batch"])
        other["control_rows"], other["context_control_rows"] = other["context_control_rows"], other["control_rows"]
    elif mutation == "excluded_target": group["target"] = "DENIED0"
    elif mutation == "drop_source_target": contract["sources"][0]["groups"] = [g for g in contract["sources"][0]["groups"] if g["target"] == "A"]
    elif mutation == "policy": contract["policy"]["min_treated_per_batch"] = 1
    elif mutation == "reserved_role": contract["sources"][0]["role"] = "dev"
    elif mutation == "expression_flag": contract["rpe1_expression_admitted"] = True
    forbid_x(monkeypatch)
    with pytest.raises(ValueError): prep.validate_contract(contract, manifest, repo=cohort[0])


@pytest.mark.parametrize("mutation", ["context", "row", "group", "negative", "nan", "gene_order", "source_axis"])
def test_cache_rejects_mismatched_values_axes_or_context(cohort, mutation):
    output, audit, cache, receipt = materialized(cohort)
    arrays, contract = prep.load_cache(cache, receipt["cache_sha256"], output / "contract.json", audit["contract_sha256"], output / "sources.json", audit["source_manifest_sha256"], repo=cohort[0])
    if mutation == "context": arrays["context"][0, 0] += .1
    elif mutation == "row": arrays["control_rows"][0] += 1
    elif mutation == "group": arrays["control_group"][0] = -1
    elif mutation == "negative": arrays["control"][0, 0] = -1
    elif mutation == "nan": arrays["treated"][0, 0] = np.nan
    elif mutation == "gene_order": arrays["genes"] = arrays["genes"][::-1]
    elif mutation == "source_axis": arrays["sources"][0] = "hepg2"
    with pytest.raises(ValueError): prep.validate_cache(arrays, contract, audit["contract_sha256"], audit["source_manifest_sha256"])


def test_protocol_and_protected_seals_fail_closed(cohort):
    output, audit = registered(cohort)
    cohort[2].write_text("changed protocol")
    with pytest.raises(ValueError, match="SHA"):
        prep.load_contract(output / "contract.json", audit["contract_sha256"], output / "sources.json", audit["source_manifest_sha256"], repo=cohort[0])


def test_fresh_outputs_only(cohort):
    output, audit, cache, receipt = materialized(cohort)
    with pytest.raises(FileExistsError): prep.register(cohort[0], output, cohort[2])
    with pytest.raises(FileExistsError): prep.materialize(output / "contract.json", audit["contract_sha256"], output / "sources.json", audit["source_manifest_sha256"], cache, repo=cohort[0])
