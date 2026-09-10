"""Synthetic v7 selected-row, safe archive, parent preservation and cache tests."""
from __future__ import annotations

import copy
import io
import json
import os
from pathlib import Path
import sys
import zipfile

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import public_replicated_cache_v7 as cache
import prepare_public_flow_pilot as old
import public_training_readiness as safe
from test_prepare_public_replicated_v7 import cohort, registered as register_cohort


def expression_file(tmp_path, encoding="dense"):
    raw = np.array([[1., 2., 3., 4., 5.], [6., 7., 8., 9., 10.],
                    [np.nan]*5, [3., 0., 2., 8., 1.], [-9.]*5, [2., 1., 4., 3., 6.]])
    path = tmp_path / (encoding + ".h5ad")
    with h5py.File(path, "w") as handle:
        if encoding == "dense": handle.create_dataset("X", data=raw)
        else:
            group = handle.create_group("X"); group.attrs["encoding-type"] = "csr_matrix"; group.attrs["shape"] = raw.shape
            group.create_dataset("data", data=raw.reshape(-1))
            group.create_dataset("indices", data=np.tile(np.arange(5, dtype=np.int64), len(raw)))
            group.create_dataset("indptr", data=np.arange(0, raw.size+1, 5, dtype=np.int64))
    return path, raw


def read(handle, rows=None, **kwargs):
    args = {"authorized_rows": [0, 1, 3, 5], "n_obs": 6, "n_vars": 5,
            "shared_columns": [4, 0, 2], "output_indices": [2, 0], "chunk_rows": 2}
    args.update(kwargs)
    return cache.read_normalized_output_rows(handle, [0, 1, 3, 5] if rows is None else rows, **args)


def forbid_x_values(monkeypatch):
    original = h5py.Dataset.__getitem__
    def guarded(dataset, key):
        if dataset.name == "/X" or dataset.name.startswith("/X/"):
            pytest.fail("Whole request must be admitted before any X values")
        return original(dataset, key)
    monkeypatch.setattr(h5py.Dataset, "__getitem__", guarded)


@pytest.mark.parametrize("encoding", ["dense", "csr"])
@pytest.mark.parametrize("chunk_rows", [1, 3, 256])
def test_chunked_normalization_is_exact_v2_and_excluded_poison_is_unread(tmp_path, monkeypatch, encoding, chunk_rows):
    path, raw = expression_file(tmp_path, encoding)
    original = h5py.Dataset.__getitem__; calls = []
    def guarded(dataset, key):
        if dataset.name == "/X":
            row = key[0]; assert type(row) in (int, np.int64) and int(row) in {0, 1, 3, 5}; calls.append(int(row))
        if dataset.name == "/X/data":
            assert isinstance(key, slice)
            assert not set(range(key.start, key.stop)) & (set(range(10, 15)) | set(range(20, 25)))
            calls.append(key.start//5)
        return original(dataset, key)
    monkeypatch.setattr(h5py.Dataset, "__getitem__", guarded)
    with h5py.File(path, "r") as handle: result = read(handle, chunk_rows=chunk_rows)
    expected = old.normalize_counts(raw[[0, 1, 3, 5]][:, [4, 0, 2]])[:, [2, 0]]
    np.testing.assert_array_equal(result, expected)
    assert result.dtype == np.float32 and result.shape == (4, 2)
    assert sorted(calls) == [0, 1, 3, 5]


@pytest.mark.parametrize("rows", [[0, 1, 3, 4], [0, 1, 3, 6], [0, 1, 3, -1],
                                  [0, 1, 3, True], [0, 1, 3, 3], [1, 0, 3, 5], [0, 1, 3, 5.]])
def test_entire_row_request_validated_before_first_x(tmp_path, monkeypatch, rows):
    path, _ = expression_file(tmp_path); forbid_x_values(monkeypatch)
    with h5py.File(path, "r") as handle, pytest.raises(ValueError): read(handle, rows)


@pytest.mark.parametrize("kwargs", [{"shared_columns": [4, 0, 5]}, {"shared_columns": [4, 0, 0]},
    {"shared_columns": [4., 0., 2.]}, {"shared_columns": [[4, 0, 2]]},
    {"output_indices": [0, 3]}, {"output_indices": [0, 0]}, {"output_indices": [True, 0]},
    {"chunk_rows": 0}, {"chunk_rows": True}])
def test_invalid_axes_or_chunk_bound_fail_before_x(tmp_path, monkeypatch, kwargs):
    path, _ = expression_file(tmp_path); forbid_x_values(monkeypatch)
    with h5py.File(path, "r") as handle, pytest.raises(ValueError): read(handle, **kwargs)


@pytest.mark.parametrize("kwargs", [{"n_obs": 7}, {"n_vars": 6}])
def test_dense_shape_must_match_registered_metadata(tmp_path, monkeypatch, kwargs):
    path, _ = expression_file(tmp_path); forbid_x_values(monkeypatch)
    with h5py.File(path, "r") as handle, pytest.raises(ValueError): read(handle, **kwargs)


@pytest.mark.parametrize("mutation", ["shape", "float_pointer", "short_pointer", "negative_pointer", "oversized_span", "duplicate_index", "outside_index"])
def test_malformed_csr_structure_rejected(tmp_path, mutation):
    path, _ = expression_file(tmp_path, "csr")
    with h5py.File(path, "r+") as handle:
        x = handle["X"]
        if mutation == "shape": x.attrs["shape"] = (5, 5)
        elif mutation in {"float_pointer", "short_pointer"}:
            pointer = x["indptr"][:]; del x["indptr"]
            x.create_dataset("indptr", data=pointer.astype(float) if mutation == "float_pointer" else pointer[:-1])
        elif mutation == "negative_pointer": x["indptr"][0] = -1
        elif mutation == "oversized_span": x["indptr"][1] = 6
        elif mutation == "duplicate_index": x["indices"][1] = x["indices"][0]
        else: x["indices"][1] = 5
    original = h5py.Dataset.__getitem__
    if mutation == "oversized_span":
        # The invalid span must be refused before an oversized expression slice.
        def guarded(dataset, key):
            if dataset.name == "/X/data": pytest.fail("Oversized CSR span reached data read")
            return original(dataset, key)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(h5py.Dataset, "__getitem__", guarded)
            with h5py.File(path, "r") as handle, pytest.raises(ValueError): read(handle)
    else:
        with h5py.File(path, "r") as handle, pytest.raises(ValueError): read(handle)


@pytest.mark.parametrize("values", [[0.]*5, [1e308]*5])
def test_zero_or_overflowing_shared_library_is_not_silently_zero_normalized(tmp_path, values):
    path, _ = expression_file(tmp_path)
    with h5py.File(path, "r+") as handle: handle["X"][0] = values
    with np.errstate(over="ignore"), h5py.File(path, "r") as handle, pytest.raises(ValueError): read(handle)


def npy_payload(array):
    stream = io.BytesIO(); np.save(stream, array, allow_pickle=True); return stream.getvalue()


def zip_payload(path, members):
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in members: archive.writestr(name, payload)
    return safe.digest(path)


def test_safe_npz_exact_schema_and_nonobject_values(tmp_path):
    path = tmp_path/"safe.npz"; expected = np.arange(6, dtype=np.float32).reshape(2, 3)
    sha = zip_payload(path, [("values.npy", npy_payload(expected))])
    values = cache.read_npz(path, sha, {"values"})
    assert set(values) == {"values"}; np.testing.assert_array_equal(values["values"], expected)


@pytest.mark.parametrize("attack", ["object", "structured", "extra_member", "missing_member", "duplicate", "declared_shape", "bad_hash"])
def test_safe_npz_rejects_payload_schema_or_allocation_attacks(tmp_path, attack):
    path = tmp_path/"attack.npz"; payload = npy_payload(np.zeros((1,), dtype=np.float32))
    members = [("values.npy", payload)]
    if attack == "object": members[0] = ("values.npy", npy_payload(np.array([{"unsafe": True}], dtype=object)))
    elif attack == "structured": members[0] = ("values.npy", npy_payload(np.zeros(1, dtype=[("x", "f4")])))
    elif attack == "extra_member": members.append(("extra.npy", payload))
    elif attack == "missing_member": members = []
    elif attack == "duplicate": members.append(members[0])
    elif attack == "declared_shape":
        header = io.BytesIO(); np.lib.format.write_array_header_1_0(header, {"descr": "<f8", "fortran_order": False, "shape": (10**9,)})
        members = [("values.npy", header.getvalue()+b"\0"*8)]
    if attack == "duplicate":
        with pytest.warns(UserWarning, match="Duplicate"): sha = zip_payload(path, members)
    else: sha = zip_payload(path, members)
    with pytest.raises(ValueError): cache.read_npz(path, "0"*64 if attack == "bad_hash" else sha, {"values"})


def test_symlink_archive_leaf_rejected(tmp_path):
    path = tmp_path/"safe.npz"; sha = zip_payload(path, [("a.npy", npy_payload(np.zeros(1)))])
    link = tmp_path/"link.npz"; link.symlink_to(path)
    with pytest.raises(ValueError): cache.read_npz(link, sha, {"a"})


def test_fifo_cannot_block_before_regular_file_check(tmp_path, monkeypatch):
    path = tmp_path/"fifo"; os.mkfifo(path); original = os.open
    def guarded(name, flags, *args, **kwargs):
        if Path(name) == path: assert flags & os.O_NONBLOCK, "FIFO must be opened nonblocking before fstat"
        return original(name, flags, *args, **kwargs)
    monkeypatch.setattr(os, "open", guarded)
    with pytest.raises(ValueError): cache.read_npz(path, "0"*64, {"a"})


@pytest.fixture
def registered_cache(cohort, monkeypatch):
    """Make an independently whole-row-normalized, valid synthetic v2 parent."""
    parent, bindings = cohort["parent_contract"], cohort["parent_v2"]
    csha, ssha = bindings["contract"]["sha256"], bindings["sources"]["sha256"]
    arrays = cache.empty_cache(parent, csha, ssha)
    arrays["schema"] = np.asarray(cache.v2.CACHE_SCHEMA)
    offsets = cache.offsets_for(parent); group_offset = 0
    for source in parent["sources"]:
        meta = cohort["metas"][source["id"]]
        columns = [list(meta["genes"]).index(g) for g in parent["shared_genes"]]
        rows = sorted({row for g in source["groups"] for field in cache.POOLS.values() for row in g[field]})
        with h5py.File(source["path"], "r") as handle:
            normalized = old.normalize_counts(old.read_authorized_rows(handle, rows, authorized=rows, columns=columns))
        normalized = normalized[:, [parent["shared_genes"].index(g) for g in parent["output_genes"]]]
        lookup = {row: i for i, row in enumerate(rows)}
        for local, group in enumerate(source["groups"]):
            gid = group_offset + local
            for name, field in cache.POOLS.items():
                left, right = offsets[name][gid:gid+2]
                values = normalized[[lookup[row] for row in group[field]]]
                arrays[name][left:right] = values
                if name == "context_control":
                    statistics = values.astype(np.float64)
                    arrays["context"][gid] = np.concatenate((statistics.mean(0), statistics.std(0))).astype(np.float32)
        group_offset += len(source["groups"])
    cache.v2.validate_cache(arrays, parent, csha, ssha)
    path = Path(bindings["cache"]["path"])
    np.savez_compressed(path, **arrays)
    bindings["cache"]["sha256"] = safe.digest(path)
    receipt_path = Path(bindings["cache_receipt"]["path"])
    receipt_path.write_bytes(old.canonical({"cache_sha256": bindings["cache"]["sha256"],
        "contract_sha256": csha, "source_manifest_sha256": ssha}))
    bindings["cache_receipt"]["sha256"] = safe.digest(receipt_path)
    monkeypatch.setattr(cache.prep, "PARENT_PINS", {k: (Path(v["path"]).name, v["sha256"]) for k,v in bindings.items()})
    def parent_contract(path, sha, sources, source_sha, *, repo=None):
        assert Path(path) == Path(bindings["contract"]["path"]) and sha == csha
        assert Path(sources) == Path(bindings["sources"]["path"]) and source_sha == ssha
        safe.read_json(sources, source_sha)
        return safe.read_json(path, sha)
    monkeypatch.setattr(cache.v2, "load_contract", parent_contract)
    monkeypatch.setattr(cache.socket, "gethostname", lambda: "cbsuvlaminck3.biohpc.cornell.edu")
    output, audit, contract = register_cohort(cohort)
    return {"cohort": cohort, "out": output, "audit": audit, "contract": contract, "parent_arrays": arrays,
            "path": output/"cache.npz"}


def materialize(fixture):
    out, audit = fixture["out"], fixture["audit"]
    return cache.materialize(out/"contract.json", audit["contract_sha256"], out/"sources.json",
        audit["source_manifest_sha256"], fixture["path"], repo=fixture["cohort"]["root"])


def load(fixture, sha):
    out, audit = fixture["out"], fixture["audit"]
    return cache.load_cache(fixture["path"], sha, out/"contract.json", audit["contract_sha256"],
                           out/"sources.json", audit["source_manifest_sha256"], repo=fixture["cohort"]["root"])


def test_full_materialization_reads_unique_admitted_rows_and_replays_every_parent_pool(registered_cache, monkeypatch):
    fixture = registered_cache; cohort = fixture["cohort"]; original = h5py.Dataset.__getitem__; calls = []
    admitted = {s["id"]: {r for g in s["groups"] for f in cache.POOLS.values() for r in g[f]} for s in fixture["contract"]["sources"]}
    def guarded(dataset, key):
        if dataset.name == "/X":
            source = next(s for s, meta in cohort["metas"].items() if len(meta["cells"]) == dataset.shape[0])
            assert key[0] in admitted[source]
            calls.append((source, key[0]))
        return original(dataset, key)
    monkeypatch.setattr(h5py.Dataset, "__getitem__", guarded)
    receipt = materialize(fixture)
    arrays, contract = load(fixture, receipt["cache_sha256"])
    assert len(calls) == len(set(calls)) == sum(map(len, admitted.values()))
    assert receipt["completed"] is True and receipt["parent_subset_replay"] == {
        "passed": True, "groups_verified": 16, "all_pools_bit_identical": True,
        "context_bit_identical": True, "raw_sources_reread_by_comparison": False}
    assert arrays["context"].shape == (22, 6) and arrays["control"].shape == (22*32, 3)
    assert any(m["parent_group_id"] != m["expanded_group_id"] for m in contract["parent_group_mapping"])
    assert receipt["array_bytes"] == contract["resource_estimate"]["cache_arrays_upper_bound_bytes"]
    assert receipt["expanded_archive_bytes"] > receipt["array_bytes"]
    assert receipt["raw_sources_authenticated_before_expression"] is True
    assert all(receipt[k] is False for k in ("gpu_used", "model_fitting_performed", "posttraining_performed",
        "download_performed", "submission_performed", "cloud_synced", "rpe1_h1_hepg2_challenge2026_feng_opened"))


def test_every_expanded_cell_matches_independent_fixture_formula_and_new_row_permutation_is_detected(registered_cache):
    fixture = registered_cache
    receipt = materialize(fixture)
    arrays, contract = load(fixture, receipt["cache_sha256"])
    admitted_identities = set()
    old_identities = {(s["id"], row) for s in fixture["cohort"]["parent_contract"]["sources"]
                      for g in s["groups"] for field in cache.POOLS.values() for row in g[field]}

    def assert_expected_cells(values):
        # Resolve groups semantically, never assume parent-prefix or source order.
        group_ids = {(str(source), str(target), str(batch)): i for i, (source, target, batch) in
                     enumerate(zip(values["sources"], values["targets"], values["batches"]))}
        for source in contract["sources"]:
            sid = source["id"]
            genes = list(fixture["cohort"]["metas"][sid]["genes"])
            shared = [genes.index(g) for g in contract["shared_genes"]]
            output = [contract["shared_genes"].index(g) for g in contract["output_genes"]]
            salt = 0 if sid == "replogle_k562" else 17
            for group in source["groups"]:
                gid = group_ids[(sid, group["target"], group["batch"])]
                for pool, field in cache.POOLS.items():
                    originals = np.asarray(group[field], dtype=np.int64)
                    indices = np.flatnonzero(values["group" if pool == "treated" else pool+"_group"] == gid)
                    np.testing.assert_array_equal(values[pool+"_rows"][indices], originals)
                    # Independent known fixture counts; no production row reader,
                    # normalization helper, parent values or extraction offsets.
                    raw = np.column_stack((originals+1+salt, originals % 7+1+salt,
                        (3*originals) % 11+1+2*salt, originals % 5+1, originals % 3+1)).astype(np.float64)
                    common = raw[:, shared]
                    total = np.sum(common, axis=1, dtype=np.float64)
                    expected = np.log1p(common*(10000.0/total[:, None])).astype(np.float32)[:, output]
                    np.testing.assert_array_equal(values[pool][indices], expected)
                    admitted_identities.update((sid, int(row)) for row in originals)

    assert_expected_cells(arrays)
    assert admitted_identities - old_identities  # Covers truly new physical cells, not only additional group references.
    old_groups = {m["expanded_group_id"] for m in contract["parent_group_mapping"]}
    gid = next(i for i in range(len(arrays["targets"])) if i not in old_groups)
    selected = np.flatnonzero(arrays["group"] == gid)
    assert len(selected) >= 2
    changed = {k: v.copy() for k, v in arrays.items()}
    changed["treated"][selected] = arrays["treated"][selected[::-1]]
    assert not np.array_equal(changed["treated"][selected], arrays["treated"][selected])
    # Such a permutation preserves shape, labels, dtypes and all frozen parent
    # values. Local structural/parent checks are not claimed to reread new X.
    cache.validate_cache(changed, contract, fixture["audit"]["contract_sha256"], fixture["audit"]["source_manifest_sha256"])
    assert cache.verify_parent_subset(changed, contract, repo=fixture["cohort"]["root"])["passed"]
    with pytest.raises(AssertionError): assert_expected_cells(changed)


def test_second_source_authentication_failure_precedes_any_expression(registered_cache, monkeypatch):
    second = Path(registered_cache["cohort"]["specs"][1]["path"])
    with second.open("ab") as stream: stream.write(b"changed synthetic source bytes")
    forbid_x_values(monkeypatch)
    with pytest.raises(ValueError, match="authentication"): materialize(registered_cache)
    assert not registered_cache["path"].exists() and not Path(str(registered_cache["path"])+".receipt.json").exists()


@pytest.mark.parametrize("part", ["context", "original_row", "negative", "source_axis"])
def test_rehashed_corrupt_cache_still_fails_semantic_validation(registered_cache, part):
    receipt = materialize(registered_cache); path = registered_cache["path"]
    arrays = cache.read_npz(path, receipt["cache_sha256"])
    if part == "context": arrays["context"][0, 0] += .1
    elif part == "original_row": arrays["control_rows"][0] += 1
    elif part == "negative": arrays["treated"][0, 0] = -.1
    else: arrays["sources"][0] = "hepg2"
    np.savez_compressed(path, **arrays); newsha = safe.digest(path)
    with zipfile.ZipFile(path) as archive: expanded = sum(i.file_size for i in archive.infolist())
    receipt.update(cache_sha256=newsha, cache_bytes=path.stat().st_size,
                   array_bytes=sum(a.nbytes for a in arrays.values()), expanded_archive_bytes=expanded)
    Path(str(path)+".receipt.json").write_bytes(old.canonical(receipt))
    with pytest.raises(ValueError): load(registered_cache, newsha)


@pytest.mark.parametrize("part", ["anchor_values", "context_values", "mapping"])
def test_parent_subset_rejects_coherent_rehashed_changes_even_if_local_cache_is_consistent(registered_cache, part):
    receipt = materialize(registered_cache)
    arrays, contract = load(registered_cache, receipt["cache_sha256"])
    if part == "mapping": contract["parent_group_mapping"][0]["expanded_group_id"] += 1
    else:
        name = "control" if part == "anchor_values" else "context_control"
        first_source = contract["sources"][0]["id"]
        old_gid = contract["parent_group_mapping"][0]["expanded_group_id"]
        offsets = cache.offsets_for(contract)
        original = arrays[name+"_rows"][offsets[name][old_gid]]
        labels = arrays[name+"_group"]
        selected = (arrays[name+"_rows"] == original) & (arrays["sources"][labels] == first_source)
        arrays[name][selected, 0] += .01
        if name == "context_control":
            for gid in range(len(arrays["targets"])):
                left, right = offsets[name][gid:gid+2]
                values = arrays[name][left:right].astype(np.float64)
                arrays["context"][gid] = np.concatenate((values.mean(0), values.std(0))).astype(np.float32)
        cache.validate_cache(arrays, contract, registered_cache["audit"]["contract_sha256"],
                             registered_cache["audit"]["source_manifest_sha256"])
    with pytest.raises(ValueError): cache.verify_parent_subset(arrays, contract, repo=registered_cache["cohort"]["root"])


@pytest.mark.parametrize("part", ["activity_bool", "completion_bool", "groups_bool", "size_float", "header_bytes",
    "rss", "hostname", "source_rows", "chunks", "parent_replay", "population", "unknown_field"])
def test_modified_receipt_cannot_attest_to_valid_cache(registered_cache, part):
    receipt = materialize(registered_cache); sha = receipt["cache_sha256"]
    if part == "activity_bool": receipt["model_fitting_performed"] = 0
    elif part == "completion_bool": receipt["completed"] = 1
    elif part == "groups_bool": receipt["groups"] = True
    elif part == "size_float": receipt["array_bytes"] = float(receipt["array_bytes"])
    elif part == "header_bytes": receipt["expanded_archive_bytes"] += 1
    elif part == "rss": receipt["peak_rss_bytes"] = cache.MAX_RSS+1
    elif part == "hostname": receipt["hostname"] = "aida.cac.cornell.edu"
    elif part == "source_rows": receipt["source_extraction"][0]["unique_rows"] += 1
    elif part == "chunks": receipt["row_chunk_limit"] = 257
    elif part == "parent_replay": receipt["parent_subset_replay"]["all_pools_bit_identical"] = 1
    elif part == "population": receipt["population_entries"]["control"] += 1
    else: receipt["extra"] = "unexpected"
    Path(str(registered_cache["path"])+".receipt.json").write_bytes(old.canonical(receipt))
    with pytest.raises(ValueError): load(registered_cache, sha)


@pytest.mark.parametrize("which", ["cache", "receipt"])
def test_existing_outputs_are_not_overwritten(registered_cache, which):
    path = registered_cache["path"] if which == "cache" else Path(str(registered_cache["path"])+".receipt.json")
    path.write_bytes(b"original synthetic artifact")
    with pytest.raises(ValueError, match="Fresh"): materialize(registered_cache)
    assert path.read_bytes() == b"original synthetic artifact"


def test_parent_replay_failure_publishes_no_cache_or_receipt(registered_cache, monkeypatch):
    def rejected(*args, **kwargs): raise ValueError("Synthetic parent replay failed")
    monkeypatch.setattr(cache, "verify_parent_subset", rejected)
    with pytest.raises(ValueError, match="parent replay"): materialize(registered_cache)
    assert not registered_cache["path"].exists() and not Path(str(registered_cache["path"])+".receipt.json").exists()
