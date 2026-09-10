"""Synthetic-only row-isolation, provenance and raw-count preparation tests."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_public_flow_pilot as prep


def make_h5(path, *, target="A", sparse=False, prefix="cell", legacy=False):
    counts = np.tile(np.array([1., 2., 3.]), (18, 1))
    counts[8:16] += 1
    counts[16:] = np.nan  # Excluded rows MUST NOT be selected or deserialized.
    with h5py.File(path, "w") as handle:
        obs, var = handle.create_group("obs"), handle.create_group("var")
        obs.create_dataset("cell", data=np.array([f"{prefix}-{i}" for i in range(18)], dtype="S"))
        categories = np.array(["control", target, "DENIED"], dtype="S")
        codes = np.array([0] * 8 + [1] * 8 + [2] * 2, dtype=np.int32)
        if legacy:
            obs.create_dataset("target", data=codes)
            obs.create_group("__categories").create_dataset("target", data=categories)
        else:
            cat = obs.create_group("target")
            cat.create_dataset("categories", data=categories)
            cat.create_dataset("codes", data=codes)
        obs.create_dataset("batch", data=np.array(["batch-1"] * 18, dtype="S"))
        var.create_dataset("symbol", data=np.array(["G1", "G2", "G3"], dtype="S"))
        if sparse:
            x = handle.create_group("X")
            x.attrs["encoding-type"] = "csr_matrix"
            x.attrs["shape"] = counts.shape
            x.create_dataset("data", data=counts.ravel())
            x.create_dataset("indices", data=np.tile(np.arange(3), 18))
            x.create_dataset("indptr", data=np.arange(0, 55, 3))
        else:
            handle.create_dataset("X", data=counts)
    return counts


def spec_for(path, source="replogle_k562", role=None):
    return {"id": source, "role": prep.SOURCE_ROLES[source] if role is None else role, "path": str(path),
            "size_bytes": path.stat().st_size, "sha256": prep.sha_file(path),
            "cell_key": "cell", "target_key": "target", "batch_key": "batch",
            "gene_key": "symbol", "control_label": "control"}


def fixture_sources(tmp_path, monkeypatch, sparse=False, profile="rpe1_dev"):
    seal = tmp_path / "seal.json"
    prep.write_json(seal, {"partition": {"global_exclusion_targets": ["DENIED"] + [f"exclude-{i}" for i in range(771)]}})
    monkeypatch.setattr(prep, "SEALS", {"seal.json": prep.sha_file(seal)})
    monkeypatch.setattr(prep, "PRIOR_TARGETS", [])
    specs = []
    for source, role in prep.SOURCE_PROFILES[profile].items():
        path = tmp_path / (source + ".h5ad")
        target = "B" if profile == "rpe1_dev" and source == "vcc2025_h1_train" else "A"
        make_h5(path, target=target, sparse=sparse, prefix=source)
        spec = spec_for(path, source, role=role)
        receipt = tmp_path / (source + ".receipt.json")
        prep.write_json(receipt, {"files": [{"name": path.name, "size_bytes": spec["size_bytes"], "sha256": spec["sha256"]}]})
        spec.update(receipt_path=str(receipt), receipt_name=path.name, receipt_sha256=prep.sha_file(receipt))
        specs.append(spec)
    manifest = tmp_path / "sources.json"
    prep.write_json(manifest, {"source_profile": profile, "sources": specs})
    return manifest, specs


def forbid_numeric_reads(monkeypatch):
    original = h5py.Dataset.__getitem__
    calls = []
    def guard(dataset, key):
        if dataset.name == "/X" or dataset.name.startswith("/X/"):
            calls.append((dataset.name, key))
            raise AssertionError("Numeric X read before authorization")
        return original(dataset, key)
    monkeypatch.setattr(h5py.Dataset, "__getitem__", guard)
    return calls


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("legacy", [False, True])
def test_metadata_and_registration_do_not_read_x(tmp_path, monkeypatch, sparse, legacy):
    path = tmp_path / "input.h5ad"
    make_h5(path, sparse=sparse, legacy=legacy)
    spec = spec_for(path)
    calls = forbid_numeric_reads(monkeypatch)
    with h5py.File(path, "r") as handle:
        meta = prep.metadata(handle, spec)
    groups = prep.choose_groups(meta, source="synthetic", control_label="control", denied=["DENIED"], seed=4)
    prep.validate_groups(meta, groups, control_label="control", denied=["DENIED"])
    assert len(groups) == 1
    assert groups[0]["control_rows"] == list(range(8))
    assert groups[0]["treated_rows"] == list(range(8, 16))
    assert not calls


@pytest.mark.parametrize("sparse", [False, True])
def test_row_reader_never_reads_excluded_rows(tmp_path, monkeypatch, sparse):
    path = tmp_path / "input.h5ad"
    counts = make_h5(path, sparse=sparse)
    original = h5py.Dataset.__getitem__
    calls = []
    def guard(dataset, key):
        if dataset.name == "/X":
            calls.append(key[0])
            assert key[0] in (0, 8)
        elif dataset.name in ("/X/data", "/X/indices"):
            calls.append((key.start, key.stop))
            assert (key.start, key.stop) in ((0, 3), (24, 27))
        return original(dataset, key)
    monkeypatch.setattr(h5py.Dataset, "__getitem__", guard)
    with h5py.File(path, "r") as handle:
        result = prep.read_authorized_rows(handle, [0, 8], authorized=[0, 8], columns=[2, 0])
    np.testing.assert_array_equal(result, counts[[0, 8]][:, [2, 0]])
    assert calls


@pytest.mark.parametrize("rows", [[0, 16], [0, 0], [8, 0], [True], [np.int64(0)]])
def test_bad_request_fails_before_numeric_read(tmp_path, monkeypatch, rows):
    path = tmp_path / "input.h5ad"
    make_h5(path)
    calls = forbid_numeric_reads(monkeypatch)
    with h5py.File(path, "r") as handle, pytest.raises(ValueError):
        prep.read_authorized_rows(handle, rows, authorized=[0, 8], columns=[0, 1])
    assert not calls


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("bad_value", [1.0000000001, -1., np.nan, np.inf])
def test_count_validation_precedes_lossy_cast(tmp_path, sparse, bad_value):
    path = tmp_path / "input.h5ad"
    make_h5(path, sparse=sparse)
    with h5py.File(path, "r+") as handle:
        if sparse:
            handle["X/data"][0] = bad_value
        else:
            handle["X"][0, 0] = bad_value
    with h5py.File(path, "r") as handle, pytest.raises(ValueError, match="raw UMI"):
        prep.read_authorized_rows(handle, [0], authorized=[0], columns=[0, 1, 2])


@pytest.mark.parametrize("legacy", [False, True])
def test_negative_category_code_rejected(tmp_path, legacy):
    path = tmp_path / "input.h5ad"
    make_h5(path, legacy=legacy)
    with h5py.File(path, "r+") as handle:
        handle["obs/target" if legacy else "obs/target/codes"][0] = -1
    with h5py.File(path, "r") as handle, pytest.raises(ValueError, match="category codes"):
        prep.metadata(handle, spec_for(path))


def test_exact_batch_control_matching():
    meta = {"cells": np.array([f"c{i}" for i in range(24)]),
            "targets": np.array(["control"] * 8 + ["A"] * 8 + ["B"] * 8),
            "batches": np.array(["one"] * 16 + ["two"] * 8)}
    groups = prep.choose_groups(meta, source="s", control_label="control", denied=[], seed=8)
    assert {g["target"] for g in groups} == {"A"}
    bad = copy.deepcopy(groups)
    bad[0]["batch"] = "two"
    with pytest.raises(ValueError, match="Wrong target/control/batch"):
        prep.validate_groups(meta, bad, control_label="control", denied=[])


@pytest.mark.parametrize("sparse", [False, True])
def test_register_then_materialize_only_selected_rows(tmp_path, monkeypatch, sparse):
    manifest, specs = fixture_sources(tmp_path, monkeypatch, sparse)
    contract = tmp_path / "contract.json"
    with monkeypatch.context() as scoped:
        calls = forbid_numeric_reads(scoped)
        prep.register(SimpleNamespace(sources=manifest, repo=tmp_path, output=contract, genes=2, seed=8))
        assert not calls
    content = prep.load_json(contract)
    assert len(content["sources"]) == 3
    assert {s["id"] for s in content["sources"] if s["role"] == "train"} == {"replogle_k562", "vcc2025_h1_train"}
    assert {g["target"] for s in content["sources"] if s["role"] == "dev" for g in s["groups"]} == {"A"}
    output = tmp_path / "cache.npz"
    prep.materialize(SimpleNamespace(contract=contract, contract_sha256=prep.sha_file(contract), output=output))
    with np.load(output, allow_pickle=False) as arrays:
        assert arrays["train_treated"].shape == (16, 2)
        assert arrays["dev_treated"].shape == (8, 2)
        assert np.isfinite(arrays["train_treated"]).all()
        assert set(arrays["train_targets"]) == {"A", "B"}
        assert set(arrays["dev_targets"]) == {"A"}
    assert prep.load_json(str(output) + ".receipt.json")["cache_sha256"] == prep.sha_file(output)
    with pytest.raises(ValueError, match="already exists"):
        prep.materialize(SimpleNamespace(contract=contract, contract_sha256=prep.sha_file(contract), output=output))


@pytest.mark.parametrize("tamper", ["source_sha", "metadata_sha", "contract_sha", "denied_rows"])
def test_materialize_failures_before_x(tmp_path, monkeypatch, tamper):
    manifest, specs = fixture_sources(tmp_path, monkeypatch)
    contract = tmp_path / "contract.json"
    prep.register(SimpleNamespace(sources=manifest, repo=tmp_path, output=contract, genes=2, seed=8))
    body = prep.load_json(contract)
    if tamper == "source_sha":
        body["sources"][0]["sha256"] = "0" * 64
    elif tamper == "metadata_sha":
        body["sources"][0]["metadata_sha256"] = "0" * 64
    elif tamper == "denied_rows":
        body["sources"][0]["groups"][0]["treated_rows"] = [16]
    changed = tmp_path / "changed-contract.json"
    prep.write_json(changed, body)
    checksum = "0" * 64 if tamper == "contract_sha" else prep.sha_file(changed)
    calls = forbid_numeric_reads(monkeypatch)
    with pytest.raises(ValueError):
        prep.materialize(SimpleNamespace(contract=changed, contract_sha256=checksum, output=tmp_path / "bad-cache.npz"))
    assert not calls
    assert not (tmp_path / "bad-cache.npz").exists()


def test_receipt_name_size_sha_and_duplicate_items_rejected(tmp_path, monkeypatch):
    manifest, specs = fixture_sources(tmp_path, monkeypatch)
    spec = specs[0]
    prep.verify_receipt(spec)
    for field, value in (("size_bytes", 1), ("sha256", "0" * 64), ("receipt_name", "other.h5ad")):
        bad = {**spec, field: value}
        with pytest.raises(ValueError):
            prep.verify_receipt(bad)
    receipt = prep.load_json(spec["receipt_path"])
    receipt["files"].append(dict(receipt["files"][0]))
    dup_path = tmp_path / "duplicate-receipt.json"
    prep.write_json(dup_path, receipt)
    with pytest.raises(ValueError):
        prep.verify_receipt({**spec, "receipt_path": str(dup_path), "receipt_sha256": prep.sha_file(dup_path)})


def test_duplicate_h1_source_identity_rejected_before_x(tmp_path, monkeypatch):
    manifest, specs = fixture_sources(tmp_path, monkeypatch)
    repeated = dict(specs[0], id="vcc2025_h1_train", role="train")
    new_manifest = tmp_path / "duplicate-sources.json"
    prep.write_json(new_manifest, {"sources": [specs[0], repeated, specs[2]]})
    calls = forbid_numeric_reads(monkeypatch)
    with pytest.raises(ValueError):
        prep.register(SimpleNamespace(sources=new_manifest, repo=tmp_path, output=tmp_path / "no-contract.json", genes=2, seed=8))
    assert not calls


@pytest.mark.parametrize("link_kind", ["external", "soft", "virtual", "external_storage"])
def test_non_self_contained_hdf5_rejected(tmp_path, link_kind):
    path = tmp_path / "input.h5ad"
    external = tmp_path / "external.h5"
    make_h5(path)
    with h5py.File(external, "w") as handle:
        handle.create_dataset("values", data=np.ones((18, 3)))
    with h5py.File(path, "r+") as handle:
        del handle["X"]
        if link_kind == "external":
            handle["X"] = h5py.ExternalLink(str(external), "/values")
        elif link_kind == "soft":
            handle.create_dataset("other", data=np.ones((18, 3)))
            handle["X"] = h5py.SoftLink("/other")
        elif link_kind == "virtual":
            layout = h5py.VirtualLayout(shape=(18, 3), dtype=np.float64)
            layout[:] = h5py.VirtualSource(str(external), "values", shape=(18, 3))
            handle.create_virtual_dataset("X", layout)
        else:
            handle.create_dataset("X", shape=(18, 3), dtype=np.float64,
                                  external=[(str(tmp_path / "outside.bin"), 0, h5py.h5f.UNLIMITED)])
    with h5py.File(path, "r") as handle, pytest.raises(ValueError):
        prep.metadata(handle, spec_for(path))


def test_duplicate_json_keys_and_symlink_source_rejected(tmp_path):
    source = tmp_path / "source.json"
    source.write_text('{"a": 1, "a": 2}')
    with pytest.raises(ValueError, match="Duplicate JSON"):
        prep.load_json(source)
    symlink = tmp_path / "link.json"
    symlink.symlink_to(source)
    with pytest.raises(OSError):
        prep.sha_file(symlink)


def test_normalization_uses_full_shared_gene_axis():
    values = prep.normalize_counts(np.array([[1., 2., 7.]]))
    np.testing.assert_allclose(values, np.log1p([[1000., 2000., 7000.]]), rtol=1e-6)
    with pytest.raises(ValueError, match="Zero-library"):
        prep.normalize_counts(np.zeros((1, 3)))


@pytest.mark.parametrize("sparse", [False, True])
def test_h1_dev_profile_never_fits_h1(tmp_path, monkeypatch, sparse):
    manifest, specs = fixture_sources(tmp_path, monkeypatch, sparse, profile="h1_dev")
    contract = tmp_path / "h1-dev-contract.json"
    with monkeypatch.context() as scoped:
        calls = forbid_numeric_reads(scoped)
        prep.register(SimpleNamespace(sources=manifest, repo=tmp_path, output=contract, genes=2, seed=8))
        assert not calls
    content = prep.load_json(contract)
    assert content["source_profile"] == "h1_dev"
    assert content["fit_contexts"] == ["K562", "Jurkat"]
    assert content["development_context"] == "H1"
    assert {s["id"] for s in content["sources"] if s["role"] == "train"} == {"replogle_k562", "nadig_jurkat"}
    assert {s["id"] for s in content["sources"] if s["role"] == "dev"} == {"vcc2025_h1_train"}
    assert sum(s["id"] == "vcc2025_h1_train" for s in content["sources"]) == 1
    train_targets = {g["target"] for s in content["sources"] if s["role"] == "train" for g in s["groups"]}
    dev_targets = {g["target"] for s in content["sources"] if s["role"] == "dev" for g in s["groups"]}
    assert dev_targets <= train_targets
    output = tmp_path / "h1-dev-cache.npz"
    prep.materialize(SimpleNamespace(contract=contract, contract_sha256=prep.sha_file(contract), output=output))
    with np.load(output, allow_pickle=False) as arrays:
        assert set(arrays["train_sources"]) == {"replogle_k562", "nadig_jurkat"}
        assert set(arrays["dev_sources"]) == {"vcc2025_h1_train"}
        assert arrays["train_treated"].shape == (16, 2)
        assert arrays["dev_treated"].shape == (8, 2)


@pytest.mark.parametrize("tamper", ["fit_h1", "dev_jurkat", "duplicate_h1", "wrong_profile"])
def test_h1_profile_role_or_identity_tampering_fails_before_x(tmp_path, monkeypatch, tamper):
    manifest, specs = fixture_sources(tmp_path, monkeypatch, profile="h1_dev")
    config = prep.load_json(manifest)
    if tamper == "fit_h1":
        config["sources"][-1]["role"] = "train"
    elif tamper == "dev_jurkat":
        config["sources"][1]["role"] = "dev"
    elif tamper == "duplicate_h1":
        config["sources"][1] = {**config["sources"][-1], "id": "nadig_jurkat", "role": "train"}
    else:
        config["source_profile"] = "rpe1_dev"
    changed = tmp_path / "wrong-h1-profile.json"
    prep.write_json(changed, config)
    calls = forbid_numeric_reads(monkeypatch)
    with pytest.raises(ValueError):
        prep.register(SimpleNamespace(sources=changed, repo=tmp_path, output=tmp_path / "no-contract.json", genes=2, seed=8))
    assert not calls


def test_sealed_jurkat_receipt_requires_original_pin_and_source_binding(tmp_path, monkeypatch):
    path = tmp_path / "jurkat.h5ad"
    make_h5(path)
    spec = spec_for(path, "nadig_jurkat", role="train")
    source = {"filename": path.name, "size_bytes": spec["size_bytes"], "sha256": spec["sha256"]}
    receipt = tmp_path / "sealed-preflight.json"
    prep.write_json(receipt, {"datasets": [{"dataset": "nadig_jurkat", "source": source}]})
    sealed_sha = prep.sha_file(receipt)
    key = "artifacts/scdfm/v7/training/jurkat_training_preflight.json"
    monkeypatch.setattr(prep, "SEALS", {key: sealed_sha})
    spec.update(receipt_path=str(receipt), receipt_name=path.name,
                receipt_sha256=sealed_sha, receipt_kind="sealed_v7_jurkat")
    prep.verify_receipt(spec)
    for field, value in (("sha256", "0" * 64), ("size_bytes", 1), ("receipt_name", "different.h5ad")):
        with pytest.raises(ValueError):
            prep.verify_receipt({**spec, field: value})
    changed = tmp_path / "unsealed-preflight.json"
    prep.write_json(changed, {"datasets": [{"dataset": "nadig_jurkat", "source": source}], "changed": True})
    with pytest.raises(ValueError, match="original sealed identity"):
        prep.verify_receipt({**spec, "receipt_path": str(changed), "receipt_sha256": prep.sha_file(changed)})


def test_integer_batch_labels_supported_without_relabeling_targets(tmp_path):
    path = tmp_path / "integer-batch.h5ad"
    make_h5(path)
    with h5py.File(path, "r+") as handle:
        del handle["obs/batch"]
        handle["obs"].create_dataset("batch", data=np.array([4] * 18))
    with h5py.File(path, "r") as handle:
        meta = prep.metadata(handle, spec_for(path))
    assert set(meta["batches"]) == {"4"}
    assert set(meta["targets"]) == {"control", "A", "DENIED"}
