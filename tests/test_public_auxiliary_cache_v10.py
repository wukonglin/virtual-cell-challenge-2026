"""Synthetic-only auxiliary materialization/role, archive and admission tests."""
from __future__ import annotations

import copy
from datetime import datetime, timezone, timedelta
import io
import os
from pathlib import Path
import sys

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import public_auxiliary_cache_v10 as cache
from test_prepare_public_auxiliary_v9 import cohort


def save(path, value):
    path.write_bytes(cache.old.canonical(value))
    return cache.safe.digest(path)


@pytest.fixture
def prepared(cohort, monkeypatch):
    """Real v9 candidate reconstruction, tiny three-gene dense/CSR public mocks."""
    parents = cohort["parent"]
    candidates = cache.candidate.select_candidates(cohort["metas"], parents[1], parents[2], parents[3], parents[4])
    expected, allowed = {}, {}
    for source, source_record, spec in zip(candidates, parents[1]["sources"], parents[2]["sources"]):
        source_id = source["id"]
        rows, _ = cache.selected_rows(source)
        allowed[source_id] = rows
        size = len(cohort["metas"][source_id]["cells"])
        # Poison EVERY unregistered treated/tuning/sham row; selected rows vary.
        raw = np.full((size, 3), np.nan, dtype=np.float64)
        raw[rows] = np.asarray([[1 + r % 3, 2 + r % 7, 3 + r % 11] for r in rows], dtype=np.float64)
        expected[source_id] = cache.old.normalize_counts(raw[rows][:, [0, 2]])[:, [0]]
        with h5py.File(spec["path"], "r+") as handle:
            del handle["X"]
            if cohort["metas"][source_id]["encoding"] == "dense":
                handle.create_dataset("X", data=raw)
            else:
                x = handle.create_group("X")
                x.attrs["shape"] = raw.shape
                x.attrs["encoding-type"] = "csr_matrix"
                x.create_dataset("data", data=raw.reshape(-1))
                x.create_dataset("indices", data=np.tile(np.arange(3, dtype=np.int64), size))
                x.create_dataset("indptr", data=np.arange(0, raw.size + 1, 3, dtype=np.int64))
        for item in (source_record, spec):
            item["sha256"] = cache.safe.digest(spec["path"])
            item["size_bytes"] = Path(spec["path"]).stat().st_size
    monkeypatch.setattr(cache, "PARENT_SHARED_GENE_COUNT", 3)
    monkeypatch.setattr(cache, "PARENT_OUTPUT_GENE_COUNT", 2)
    monkeypatch.setattr(cache, "SHARED_GENE_COUNT", 2)
    monkeypatch.setattr(cache, "OUTPUT_GENE_COUNT", 1)
    # G1 is absent officially: retain G0,G2 for normalization, only G0 output.
    official = cohort["root"] / "official_gene_names.csv"
    official.write_text("gene_name\nG2\nEXTRA\nG0\n")
    monkeypatch.setattr(cache, "OFFICIAL_AXIS", official)
    monkeypatch.setattr(cache, "OFFICIAL_AXIS_SHA", cache.safe.digest(official))
    monkeypatch.setattr(cache, "OFFICIAL_AXIS_BYTES", official.stat().st_size)
    monkeypatch.setattr(cache, "OFFICIAL_GENE_COUNT", 3)
    monkeypatch.setattr(cache, "runtime_preflight", lambda: None)
    monkeypatch.setattr(cache, "_codes", lambda: {"synthetic-cache.py": "d" * 64})
    out = cohort["root"] / "candidate"
    result = cache.candidate.register(cohort["root"], out, cohort["protocol"])
    protocol = cohort["root"] / "materialization.md"
    protocol.write_text("Separate synthetic selected-expression materialization, not training.")
    registration = cohort["root"] / "cache_registration"
    sha = cache.register(out / "candidate_plan.json", result["candidate_plan_sha256"], protocol, registration)
    return {**cohort, "candidate_dir": out, "plan_sha": result["candidate_plan_sha256"],
        "contract": registration / "contract.json", "contract_sha": sha,
        "cache_dir": cohort["root"] / "cache", "cache_protocol": protocol,
        "expected": expected, "allowed": allowed}


def execute(prepared):
    receipt = cache.materialize(prepared["contract"], prepared["contract_sha"], prepared["cache_dir"])
    sha = cache.safe.digest(prepared["cache_dir"] / "complete.json")
    return receipt, sha


def read(prepared, sha):
    return cache.load_caches(prepared["contract"], prepared["contract_sha"], prepared["cache_dir"], sha)


def guarded_X(monkeypatch, callback):
    actual = h5py.Dataset.__getitem__
    def guarded(self, key):
        if self.name == "/X" or self.name.startswith("/X/"):
            callback(self, key)
        return actual(self, key)
    monkeypatch.setattr(h5py.Dataset, "__getitem__", guarded)


def test_registration_no_expression_read_or_cache_decode(prepared, monkeypatch):
    guarded_X(monkeypatch, lambda *a: pytest.fail("Registration read X"))
    monkeypatch.setattr(np, "load", lambda *a, **kw: pytest.fail("Registration decoded an expression cache"))
    fresh = prepared["root"] / "second_registration"
    sha = cache.register(prepared["candidate_dir"] / "candidate_plan.json", prepared["plan_sha"],
        prepared["cache_protocol"], fresh)
    contract, _ = cache.validate_contract(fresh / "contract.json", sha)
    assert contract["expression_materialization_authorized"] is True
    assert contract["training_authorized"] is False
    assert sorted(p.name for p in fresh.iterdir()) == ["contract.json", "registration_complete.json"]


def test_full_both_source_hashes_and_metadata_before_any_X(prepared, monkeypatch):
    hashes, reconstructed, reads = [], [], []
    actual_hash = cache.old.sha_stream
    actual_metadata = cache.candidate.validate_metadata
    def hash_stream(stream):
        result = actual_hash(stream)
        hashes.append(result)
        return result
    def metadata(plan, metas, **kw):
        assert len(hashes) == 2
        result = actual_metadata(plan, metas, **kw)
        reconstructed.append(True)
        return result
    def access(dataset, key):
        assert len(hashes) == 2 and reconstructed == [True]
        if dataset.name == "/X":
            reads.append(int(key[0]))
        elif dataset.name == "/X/data":
            assert key.start % 3 == key.stop % 3 == 0
            assert key.stop - key.start == 3
            reads.append(key.start // 3)
    monkeypatch.setattr(cache.old, "sha_stream", hash_stream)
    monkeypatch.setattr(cache.candidate, "validate_metadata", metadata)
    guarded_X(monkeypatch, access)
    receipt, sha = execute(prepared)
    assert sorted(reads) == sorted(r for rows in prepared["allowed"].values() for r in rows)
    assert receipt["raw_sources_authenticated_before_expression"]
    assert len(receipt["artifacts"]) == 2
    assert all(x["unique_rows"] == 120 for x in receipt["source_extraction"])
    caches, _ = read(prepared, sha)
    for source, arrays in caches.items():
        np.testing.assert_array_equal(arrays["expression"], prepared["expected"][source])
        assert not arrays["expression"].flags.writeable
        assert arrays["original_rows"].tolist() == prepared["allowed"][source]
        assert set(arrays["row_role"]) == {1, 2, 3}
        assert np.sum(arrays["row_role"] == 1) == 24
        assert np.sum(arrays["row_role"] == 2) == 32
        assert np.sum(arrays["row_role"] == 3) == 64
        assert arrays["fit_reference_indices"].shape == (2, 16)
        assert arrays["context_indices"].shape == (2, 32)
        assert arrays["treated_offsets"].tolist() == [0, 12, 24]
        assert len(np.unique(arrays["original_rows"])) == len(arrays["expression"])


@pytest.mark.parametrize("position", [0, 1])
def test_either_bad_full_source_hash_blocks_all_X(prepared, monkeypatch, position):
    guarded_X(monkeypatch, lambda *a: pytest.fail("X reached before BOTH source hashes passed"))
    actual = cache.old.sha_stream
    count = []
    def broken(stream):
        count.append(1)
        return "e" * 64 if len(count) == position + 1 else actual(stream)
    monkeypatch.setattr(cache.old, "sha_stream", broken)
    with pytest.raises(ValueError, match="Source authentication failed"):
        execute(prepared)
    assert not prepared["cache_dir"].exists()


def test_metadata_reconstruction_failure_blocks_all_X(prepared, monkeypatch):
    guarded_X(monkeypatch, lambda *a: pytest.fail("X reached before full candidate metadata reconstruction"))
    def broken(*a, **kw):
        raise ValueError("synthetic candidate row mismatch")
    monkeypatch.setattr(cache.candidate, "validate_metadata", broken)
    with pytest.raises(ValueError, match="candidate row mismatch"):
        execute(prepared)
    assert not prepared["cache_dir"].exists()


def test_open_handle_change_rejected_before_X(prepared, monkeypatch):
    guarded_X(monkeypatch, lambda *a: pytest.fail("Changed source reached X"))
    actual = cache.bounded.signature
    raw_stat = Path(prepared["parent"][2]["sources"][0]["path"]).stat()
    raw_identity = (raw_stat.st_dev, raw_stat.st_ino)
    calls = []
    def changed(stream):
        result = actual(stream)
        if result[:2] != raw_identity:
            return result
        calls.append(result)
        return (*result[:-1], result[-1] + 1) if len(calls) == 2 else result
    monkeypatch.setattr(cache.bounded, "signature", changed)
    with pytest.raises(ValueError, match="changed during complete hashing"):
        execute(prepared)


@pytest.mark.parametrize("field", ["policy", "source_routing", "capacity", "code_sha256", "output_genes", "training_authorized"])
def test_resealed_contract_mutation_rejected(prepared, field):
    record = cache.safe.read_json(prepared["contract"])
    record[field] = None
    sha = save(prepared["contract"], record)
    save(prepared["contract"].with_name("registration_complete.json"), cache._registration_completion(sha))
    with pytest.raises(ValueError):
        cache.validate_contract(prepared["contract"], sha)


@pytest.mark.parametrize("which", ["protocol", "code", "completion"])
def test_registration_inputs_and_completion_cannot_change(prepared, monkeypatch, which):
    if which == "protocol":
        prepared["cache_protocol"].write_text("Changed scientific materialization protocol")
    elif which == "code":
        monkeypatch.setattr(cache, "_codes", lambda: {"synthetic-cache.py": "f" * 64})
    else:
        save(prepared["contract"].with_name("registration_complete.json"), cache._registration_completion("f" * 64))
    with pytest.raises(ValueError):
        cache.validate_contract(prepared["contract"], prepared["contract_sha"])


@pytest.mark.parametrize("target", ["registration", "materialization"])
def test_fresh_outputs_required(prepared, target):
    if target == "registration":
        with pytest.raises(ValueError):
            cache.register(prepared["candidate_dir"] / "candidate_plan.json", prepared["plan_sha"],
                prepared["cache_protocol"], prepared["contract"].parent)
    else:
        prepared["cache_dir"].mkdir()
        with pytest.raises(ValueError, match="Fresh auxiliary"):
            execute(prepared)


@pytest.fixture
def arrays(prepared):
    contract, plan = cache.validate_contract(prepared["contract"], prepared["contract_sha"])
    plan = {**plan, "shared_genes": contract["shared_genes"], "output_genes": contract["output_genes"],
        "_binding_sha256": contract["candidate_plan"]["sha256"]}
    source = plan["sources"][0]
    arrays = cache.empty_cache(source, plan, prepared["contract_sha"])
    arrays["expression"][:] = prepared["expected"][source["id"]]
    cache.summarize_context(arrays)
    cache.validate_cache(arrays, source, plan, prepared["contract_sha"])
    return arrays, source, plan


@pytest.mark.parametrize("field", ["source", "contract_sha256", "plan_sha256", "genes", "original_rows", "row_role",
    "targets", "group_ids", "group_pool", "pool_ids", "pool_batches", "batches", "treated_offsets",
    "treated_indices", "fit_reference_indices", "context_indices", "context"])
def test_cache_axis_role_source_and_context_mutations_rejected(prepared, arrays, field):
    values, source, plan = arrays
    if values[field].dtype.kind == "U":
        values[field].flat[0] = "changed"
    else:
        values[field].flat[0] += 1
    with pytest.raises(ValueError):
        cache.validate_cache(values, source, plan, prepared["contract_sha"])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 10.0])
def test_expression_bounds_fail(prepared, arrays, value):
    values, source, plan = arrays
    values["expression"][0, 0] = value
    with pytest.raises(ValueError, match="normalized expression"):
        cache.validate_cache(values, source, plan, prepared["contract_sha"])


def test_dtype_mutation_and_extra_cache_field_fail(prepared, arrays):
    values, source, plan = arrays
    changed = {**values, "expression": values["expression"].astype(np.float64)}
    with pytest.raises(ValueError, match="dtype/shape"):
        cache.validate_cache(changed, source, plan, prepared["contract_sha"])
    with pytest.raises(ValueError, match="members"):
        cache.validate_cache({**values, "tuning_anchor": np.zeros(1)}, source, plan, prepared["contract_sha"])


def test_controls_deduplicated_and_context_is_not_reference(prepared, arrays):
    values, _, _ = arrays
    refs = values["fit_reference_indices"].ravel()
    contexts = values["context_indices"].ravel()
    treated = values["treated_indices"]
    assert not set(refs) & set(contexts) and not set(treated) & (set(refs) | set(contexts))
    assert len(values["expression"]) == len(refs) + len(contexts) + len(treated)
    for index, rows in enumerate(values["context_indices"]):
        context = values["expression"][rows].astype(np.float64)
        expected = np.r_[context.mean(0), context.std(0)].astype(np.float32)
        np.testing.assert_array_equal(values["context"][index], expected)


def test_explicit_completion_hash_required(prepared):
    execute(prepared)
    with pytest.raises(ValueError, match="invalid_sha256|hash_mismatch"):
        read(prepared, None)


@pytest.mark.parametrize("field", ["training_authorized", "source_extraction", "capacity", "hostname",
    "peak_rss_bytes", "elapsed_seconds", "raw_sources_authenticated_before_expression"])
def test_resealed_completion_mutations_rejected(prepared, field):
    receipt, _ = execute(prepared)
    mutations = {"training_authorized": True, "source_extraction": [], "capacity": {}, "hostname": "login-node",
        "peak_rss_bytes": cache.MAX_RSS + 1, "elapsed_seconds": cache.MAX_SECONDS + 1,
        "raw_sources_authenticated_before_expression": False}
    receipt[field] = mutations[field]
    sha = save(prepared["cache_dir"] / "complete.json", receipt)
    with pytest.raises(ValueError):
        read(prepared, sha)


@pytest.mark.parametrize("mutation", ["symlink", "corrupt", "oversize", "outside_path", "extra"])
def test_any_archive_mutation_blocks_all_npz_decode(prepared, monkeypatch, mutation):
    receipt, sha = execute(prepared)
    artifact = list(receipt["artifacts"].values())[-1]
    path = Path(artifact["path"])
    if mutation == "symlink":
        alternate = path.with_name("saved.npz")
        path.rename(alternate)
        path.symlink_to(alternate)
    elif mutation == "corrupt":
        path.write_bytes(b"wrong archive")
    elif mutation == "oversize":
        artifact["expanded_archive_bytes"] = cache.MAX_ARCHIVE + 1
        sha = save(prepared["cache_dir"] / "complete.json", receipt)
    elif mutation == "outside_path":
        artifact["path"] = str(path.parent.parent / path.name)
        sha = save(prepared["cache_dir"] / "complete.json", receipt)
    else:
        (prepared["cache_dir"] / "unexpected.txt").write_text("not a registered member")
    monkeypatch.setattr(np, "load", lambda *a, **kw: pytest.fail("Archive decoded before complete roster preflight"))
    with pytest.raises(ValueError):
        read(prepared, sha)


def test_archive_typed_allocation_binding_replayed(prepared):
    receipt, _ = execute(prepared)
    first = next(iter(receipt["artifacts"].values()))
    first["array_bytes"] += 1
    sha = save(prepared["cache_dir"] / "complete.json", receipt)
    with pytest.raises(ValueError, match="allocation binding"):
        read(prepared, sha)


def test_resources_checkpoint_before_materialization_output(prepared, monkeypatch):
    monkeypatch.setattr(cache, "MAX_RSS", 1)
    # Changed resource policy also invalidates contract, preventing work.
    with pytest.raises(ValueError):
        execute(prepared)
    assert not prepared["cache_dir"].exists()


def test_runtime_rejects_head_node(monkeypatch):
    monkeypatch.setattr(cache.socket, "gethostname", lambda: "aida-login")
    with pytest.raises(ValueError, match="approved compute node"):
        cache.runtime_preflight()


@pytest.mark.parametrize("variable", ["CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"])
def test_runtime_rejects_unregistered_devices_threads(monkeypatch, variable):
    monkeypatch.setattr(cache.socket, "gethostname", lambda: "cbsuvlaminck3")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        monkeypatch.setenv(name, "2")
    monkeypatch.setenv(variable, "3")
    with pytest.raises(ValueError):
        cache.runtime_preflight()


def test_registration_api_does_not_authorize_training():
    assert cache.policy()["training_authorized"] is False
    assert cache.policy()["separate_training_contract_required"] is True
    assert cache.policy()["context_role"] == "conditioning only, never response or prior targets"


def test_official_intersection_preserves_parent_order_and_all_rows(prepared):
    contract, parent = cache.validate_contract(prepared["contract"], prepared["contract_sha"])
    before = copy.deepcopy(parent)
    projected, report = cache.project_axes(parent)
    assert parent == before
    assert projected["sources"] == parent["sources"] and projected["source_routing"] == parent["source_routing"]
    assert parent["shared_genes"] == ["G0", "G1", "G2"] and parent["output_genes"] == ["G0", "G1"]
    assert projected["shared_genes"] == ["G0", "G2"] and projected["output_genes"] == ["G0"]
    assert cache.official_gene_axis() == ["G2", "EXTRA", "G0"]
    assert report["dropped_shared_genes"] == ["G1"] and report["dropped_output_genes"] == ["G1"]
    assert report["zero_filling_performed"] is report["new_rows_authorized"] is report["aliasing_performed"] is False
    assert contract["axis_projection"] == report
    for source, capacity in contract["capacity"].items():
        assert capacity["deduplicated_output_expression_bytes"] == 120 * 1 * 4
        assert contract["parent_capacity"][source]["unique_candidate_rows"] == capacity["unique_candidate_rows"]


@pytest.mark.parametrize("mutation", ["hash", "header", "duplicate", "whitespace", "count", "symlink"])
def test_official_gene_metadata_is_exact_authenticated_csv(prepared, monkeypatch, mutation):
    path = cache.OFFICIAL_AXIS
    if mutation == "hash":
        path.write_text("gene_name\nG2\nG1\nG0\n")
    elif mutation == "symlink":
        alternate = path.with_name("outside.csv")
        path.rename(alternate)
        path.symlink_to(alternate)
    else:
        payload = {"header": "symbol\nG2\nEXTRA\nG0\n", "duplicate": "gene_name\nG2\nG2\nG0\n",
            "whitespace": "gene_name\n G2\nEXTRA\nG0\n", "count": "gene_name\nG2\nG0\n"}[mutation]
        path.write_text(payload)
        monkeypatch.setattr(cache, "OFFICIAL_AXIS_SHA", cache.safe.digest(path))
        monkeypatch.setattr(cache, "OFFICIAL_AXIS_BYTES", path.stat().st_size)
    with pytest.raises(ValueError):
        cache.official_gene_axis()


def test_changed_official_axis_blocks_expression_even_if_rehashed(prepared, monkeypatch):
    guarded_X(monkeypatch, lambda *a: pytest.fail("Changed official axis reached expression"))
    path = cache.OFFICIAL_AXIS
    path.write_text("gene_name\nG0\nEXTRA\nG2\n")
    monkeypatch.setattr(cache, "OFFICIAL_AXIS_SHA", cache.safe.digest(path))
    monkeypatch.setattr(cache, "OFFICIAL_AXIS_BYTES", path.stat().st_size)
    with pytest.raises(ValueError, match="contract reconstruction"):
        execute(prepared)


def test_original_parent_axes_cannot_be_replaced_by_projection_before_v9_replay(prepared, monkeypatch):
    original = cache.candidate.validate_metadata
    calls = []
    def validate(plan, metas, **kwargs):
        assert plan["shared_genes"] == ["G0", "G1", "G2"]
        assert plan["output_genes"] == ["G0", "G1"]
        calls.append(True)
        return original(plan, metas, **kwargs)
    monkeypatch.setattr(cache.candidate, "validate_metadata", validate)
    _, sha = execute(prepared)
    arrays, _ = read(prepared, sha)
    assert calls == [True]
    assert all(a["genes"].tolist() == ["G0"] and a["expression"].shape == (120, 1) for a in arrays.values())


def test_cli_requires_explicit_completion_sha(tmp_path):
    with pytest.raises(SystemExit):
        cache.main(["verify", "--output-dir", str(tmp_path), "--contract", str(tmp_path / "contract.json"),
            "--contract-sha256", "a" * 64])
