"""Synthetic count writer tests; no competition generation or uploads."""
from dataclasses import replace
import os
from pathlib import Path
import stat
import sys

import anndata as ad
from anndata.io import read_elem, write_elem
import h5py
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import public_auxiliary_vcc_writer_v10 as writer

GENES = ["Z", "A", "B", "D"]
TARGETS = ["T2", "T1"]
CONFIG = writer.WriterConfig(expected_genes=4, expected_targets=2, cells_per_group=2,
    groups_per_shard=2, validation_rows=1)


@pytest.fixture
def new_writer(tmp_path, monkeypatch):
    monkeypatch.setattr(writer, "runtime_preflight", lambda: None)
    result = writer.SubmissionWriter(tmp_path / "writer", GENES, TARGETS, "a" * 64, CONFIG)
    yield result
    result.close()


def block(number=0):
    return sparse.csr_matrix(np.array([[number + 1, 0, 3, 0], [0, number + 4, 0, 5]], dtype=np.int32))


def fill(w):
    for i, (context, target) in enumerate(w.keys):
        w.append(context, target, block(i))


def test_complete_streaming_writer_preserves_every_integer_and_axis(new_writer):
    w = new_writer
    expected = sparse.vstack([block(i) for i in range(6)]).toarray()
    fill(w)
    receipt = w.finalize()
    output = Path(receipt["output_path"])
    assert output == w.out / "prediction.h5ad"
    actual = ad.read_h5ad(output)
    np.testing.assert_array_equal(actual.X.toarray(), expected)
    assert actual.var_names.tolist() == GENES
    assert actual.obs_names.is_unique
    assert actual.obs.groupby(["context", "target_gene"], observed=True).size().tolist() == [2] * 6
    assert receipt["validation"]["shape"] == [12, 4]
    assert receipt["validation"]["nnz"] == 24
    assert receipt["validation"]["groups"] == 6
    assert receipt["validation"]["sparsity_cap_applied"] is False
    assert receipt["validation"]["unmodeled_genes_modified_by_writer"] is False
    assert receipt["submission_performed"] is receipt["official_cli_validated"] is False
    assert writer.safe.digest(output) == receipt["output_sha256"]
    assert writer.safe.digest(w.out / "groups.jsonl") == receipt["group_journal_sha256"]
    assert len(list((w.out / "groups").glob("*.h5ad"))) == 6
    assert len(list((w.out / "shards").glob("*.h5ad"))) == 3
    assert all(stat.S_IMODE(p.stat().st_mode) == 0o600 for p in w.out.rglob("*") if p.is_file())
    assert all(stat.S_IMODE(p.stat().st_mode) == 0o700 for p in w.out.rglob("*") if p.is_dir())
    assert receipt["validation"] == writer.validate_output(output, GENES, TARGETS, CONFIG, expected_nnz=24)


@pytest.mark.parametrize("count", [0, 1, 5])
def test_incomplete_panel_cannot_finalize(new_writer, count):
    for i, (c, t) in enumerate(new_writer.keys[:count]):
        new_writer.append(c, t, block(i))
    with pytest.raises(ValueError, match="quotas"):
        new_writer.finalize()
    assert not (new_writer.out / "complete.json").exists()


@pytest.mark.parametrize("key", [("context_A", "T2"), ("B", "T2"), ("A", "T1"), ("A", "non-targeting")])
def test_append_rejects_context_target_order_and_wrong_labels(new_writer, key):
    with pytest.raises(ValueError, match="Exact next"):
        new_writer.append(*key, block())
    assert not list((new_writer.out / "groups").iterdir())


def test_repeated_group_and_repeated_finalization_fail(new_writer):
    new_writer.append("A", "T2", block())
    with pytest.raises(ValueError):
        new_writer.append("A", "T2", block())
    for c, t in new_writer.keys[1:]:
        new_writer.append(c, t, block())
    new_writer.finalize()
    with pytest.raises(ValueError):
        new_writer.finalize()
    with pytest.raises(ValueError):
        new_writer.append("A", "T2", block())


@pytest.mark.parametrize("kind", ["dense", "float", "negative", "zero", "oversized", "shape", "duplicate", "unsorted", "forged_sorted_flag"])
def test_invalid_count_block_fails_before_output(new_writer, kind):
    value = block()
    if kind == "dense": value = value.toarray()
    elif kind == "float": value = value.astype(np.float32)
    elif kind == "negative": value.data[0] = -1
    elif kind == "zero": value.data[0] = 0
    elif kind == "oversized": value.data[0] = 1000001
    elif kind == "shape": value = value[:, :3]
    elif kind == "duplicate": value.indices[1] = value.indices[0]
    elif kind == "unsorted": value.indices[:2] = [2, 0]
    else:
        value.indices[:2] = [2, 0]
        value.has_sorted_indices = True
        value.has_canonical_format = True
    with pytest.raises(ValueError):
        new_writer.append("A", "T2", value)
    assert not list((new_writer.out / "groups").iterdir())


def test_zero_and_excessive_libraries_rejected(new_writer):
    zero = sparse.csr_matrix(np.array([[0, 0, 0, 0], [1, 2, 3, 4]], dtype=np.int32))
    with pytest.raises(ValueError):
        new_writer.append("A", "T2", zero)
    excess = sparse.csr_matrix(np.full((2, 4), 300000, dtype=np.int32))
    with pytest.raises(ValueError):
        new_writer.append("A", "T2", excess)


@pytest.mark.parametrize("mutation", ["gene_order", "missing_gene", "wrong_target", "duplicate_id", "negative", "fractional_dtype", "zero_library", "duplicate_index", "unsorted_index", "indptr", "extra_sparse_member"])
def test_final_streaming_validator_rejects_schema_or_count_tampering(new_writer, mutation):
    fill(new_writer)
    receipt = new_writer.finalize()
    path = receipt["output_path"]
    with h5py.File(path, "r+") as h:
        if mutation in ("gene_order", "missing_gene"):
            genes = GENES[::-1] if mutation == "gene_order" else GENES[:-1]
            write_elem(h, "var", pd.DataFrame(index=pd.Index(genes, name="gene_name")))
        elif mutation in ("wrong_target", "duplicate_id"):
            obs = read_elem(h["obs"])
            if mutation == "wrong_target":
                obs["target_gene"] = obs["target_gene"].astype(str)
                obs.iloc[0, obs.columns.get_loc("target_gene")] = "T1"
            else:
                ids = obs.index.tolist(); ids[0] = ids[1]; obs.index = ids
            write_elem(h, "obs", obs)
        elif mutation == "negative": h["X/data"][0] = -1
        elif mutation == "fractional_dtype":
            values = h["X/data"][:].astype(np.float32); values[0] = .5
            del h["X/data"]; h["X"].create_dataset("data", data=values)
        elif mutation == "zero_library": h["X/indptr"][1] = 0
        elif mutation == "duplicate_index": h["X/indices"][1] = h["X/indices"][0]
        elif mutation == "unsorted_index": h["X/indices"][:2] = [2, 0]
        elif mutation == "indptr": h["X/indptr"][1] = 9999999
        else: h["X"].create_dataset("unexpected", data=[1])
    with pytest.raises(ValueError):
        writer.validate_output(path, GENES, TARGETS, CONFIG, expected_nnz=24)


def test_invalid_final_pointers_rejected_before_data_read(new_writer, monkeypatch):
    fill(new_writer); path = new_writer.finalize()["output_path"]
    with h5py.File(path, "r+") as h: h["X/indptr"][1] = 9999999
    actual = h5py.Dataset.__getitem__
    def guarded(self, key):
        if self.name in ("/X/data", "/X/indices"):
            pytest.fail("Malformed pointer reached sparse payload")
        return actual(self, key)
    monkeypatch.setattr(h5py.Dataset, "__getitem__", guarded)
    with pytest.raises(ValueError, match="pointer"):
        writer.validate_output(path, GENES, TARGETS, CONFIG)


def test_final_validator_bounds_each_sparse_read_and_supports_int64(new_writer, monkeypatch):
    fill(new_writer); path = new_writer.finalize()["output_path"]
    with h5py.File(path, "r+") as h:
        for field in ("indices", "indptr"):
            values = h["X/" + field][:].astype(np.int64)
            del h["X/" + field]; h["X"].create_dataset(field, data=values)
    actual = h5py.Dataset.__getitem__; calls = []
    def guarded(self, key):
        if self.name in ("/X/data", "/X/indices"):
            assert isinstance(key, slice) and key.stop - key.start <= CONFIG.validation_rows * len(GENES)
            calls.append((self.name, key.start, key.stop))
        return actual(self, key)
    monkeypatch.setattr(h5py.Dataset, "__getitem__", guarded)
    report = writer.validate_output(path, GENES, TARGETS, CONFIG)
    assert report["nnz"] == 24 and len(calls) == 24


def test_actual_concat_promotes_pointers_at_large_nnz_boundary(new_writer, monkeypatch):
    # Lower only the dependency module's threshold to exercise its real int64
    # writer path without allocating billions of entries. Other NumPy users
    # (SciPy and our validator) retain their real integer limits.
    import anndata.experimental.merge as merge
    from types import SimpleNamespace
    class ThresholdProxy:
        def __getattr__(self, name):
            return getattr(np, name)
        def iinfo(self, dtype):
            return SimpleNamespace(max=12) if dtype is np.int32 else np.iinfo(dtype)
    monkeypatch.setattr(merge, "np", ThresholdProxy())
    fill(new_writer)
    path = new_writer.finalize()["output_path"]
    with h5py.File(path, "r") as handle:
        assert handle["X/indptr"].dtype == np.int64
    np.testing.assert_array_equal(ad.read_h5ad(path).X.toarray(), sparse.vstack([block(i) for i in range(6)]).toarray())


def test_append_does_not_modify_caller_counts(new_writer):
    counts = block().astype(np.int64)
    before = (counts.data.copy(), counts.indices.copy(), counts.indptr.copy())
    new_writer.append("A", "T2", counts)
    for expected, actual in zip(before, (counts.data, counts.indices, counts.indptr)):
        np.testing.assert_array_equal(actual, expected)


def test_concat_refuses_axis_changed_block_even_with_rehashed_binding(new_writer):
    w = new_writer
    w.append("A", "T2", block())
    record = w.pending[0]
    with h5py.File(record["path"], "r+") as h:
        write_elem(h, "var", pd.DataFrame(index=pd.Index(GENES[::-1], name="gene_name")))
    record["sha256"] = writer.safe.digest(record["path"])
    with pytest.raises(ValueError, match="full gene axis"):
        w.append("A", "T1", block())
    assert not list((w.out / "shards").iterdir())


def test_concat_and_output_never_overwrite_existing_target(new_writer):
    fill(new_writer)
    path = new_writer.out / "prediction.h5ad"
    path.write_bytes(b"user-owned output")
    with pytest.raises(ValueError, match="Fresh concat"):
        new_writer.finalize()
    assert path.read_bytes() == b"user-owned output"


def test_preexisting_group_is_preserved(new_writer):
    path = new_writer.out / "groups/g000000.h5ad"
    path.write_bytes(b"user file")
    with pytest.raises(FileExistsError):
        new_writer.append("A", "T2", block())
    assert path.read_bytes() == b"user file"


def test_source_symlink_is_not_followed(new_writer):
    new_writer.append("A", "T2", block())
    path = Path(new_writer.pending[0]["path"])
    moved = path.with_name("retained.h5ad"); path.rename(moved); path.symlink_to(moved)
    with pytest.raises(ValueError):
        new_writer.append("A", "T1", block())


@pytest.mark.parametrize("field,value", [("expected_genes", 18534), ("expected_targets", 301),
    ("cells_per_group", 401), ("validation_rows", 257), ("max_nnz", 4750000001),
    ("max_counts_per_cell", 1000001), ("groups_per_shard", True)])
def test_official_caps_cannot_be_raised(field, value):
    with pytest.raises(ValueError): writer.WriterConfig(**{field: value})


def test_exact_axes_nonempty_unique_no_controls_and_hash_required(tmp_path, monkeypatch):
    monkeypatch.setattr(writer, "runtime_preflight", lambda: None)
    for genes, targets, sha in [(["Z"] * 4, TARGETS, "a" * 64), (GENES, ["T1", "non-targeting"], "a" * 64),
            (GENES, TARGETS, None), (GENES, ["T1", "T|bad"], "a" * 64)]:
        with pytest.raises(ValueError): writer.SubmissionWriter(tmp_path / "bad", genes, targets, sha, CONFIG)
    assert not (tmp_path / "bad").exists()


def test_nnz_budget_rejected_before_block_write(new_writer):
    new_writer.config = replace(CONFIG, max_nnz=3)
    with pytest.raises(ValueError, match="nnz cap"):
        new_writer.append("A", "T2", block())
    assert not list((new_writer.out / "groups").iterdir())


def test_read_control_metadata_handles_nullable_indexes_without_X(tmp_path, monkeypatch):
    path = tmp_path / "controls.h5ad"
    obs = pd.DataFrame({"target_gene": pd.Categorical(["non-targeting"] * 6),
        "context": pd.Categorical(["A"] * 6), "ntc_id": pd.Categorical(["n0"] * 3 + ["n1"] * 3)},
        index=pd.Index([f"c{i}" for i in range(6)]))
    ad.AnnData(X=sparse.csr_matrix(np.ones((6, 4), np.float32)), obs=obs,
        var=pd.DataFrame(index=pd.Index(GENES))).write_h5ad(path)
    actual = h5py.Dataset.__getitem__
    def guarded(self, key):
        if self.name == "/X" or self.name.startswith("/X/"): pytest.fail("Control metadata read expression")
        return actual(self, key)
    monkeypatch.setattr(h5py.Dataset, "__getitem__", guarded)
    with h5py.File(path, "r") as handle:
        result = writer.read_control_metadata(handle, GENES, "A", expected_cells=6, expected_ntc_ids=2)
        assert result["control_only"] and not result["expression_values_read"]
        assert result["ntc_ids"] == ["n0"] * 3 + ["n1"] * 3
        with pytest.raises(ValueError): writer.read_control_metadata(handle, GENES[::-1], "A", expected_cells=6, expected_ntc_ids=2)
        with pytest.raises(ValueError): writer.read_control_metadata(handle, GENES, "B", expected_cells=6, expected_ntc_ids=2)


def test_head_node_rejected(monkeypatch):
    monkeypatch.setattr(writer.socket, "gethostname", lambda: "aida-login")
    with pytest.raises(ValueError, match="compute host"):
        writer.runtime_preflight()
