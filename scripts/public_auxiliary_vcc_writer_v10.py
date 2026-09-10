"""Bounded count-only VCC H5AD writing; no models, credentials or upload.

The caller authenticates source controls, axes and inference provenance. This
module preserves its exact gene order and quotas, checks integer count blocks,
and independently streams the final CSR before issuing a completion receipt.
No sparsity cap, count redistribution, gene intersection or zero filling occurs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import socket
import stat
import tempfile
import time

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from anndata.experimental import concat_on_disk
from anndata.io import read_elem, write_elem

import public_training_readiness as safe
import prepare_public_flow_pilot as old
import prepare_public_replicated_v7 as paths
import public_replicated_cache_v7 as bounded

SCHEMA = "public-auxiliary-vcc-count-writer-v10"
require, equal = paths.require, paths.equal
CONTEXTS = ("A", "B", "C")


@dataclass(frozen=True)
class WriterConfig:
    expected_genes: int = 18533
    expected_targets: int = 300
    contexts: tuple[str, ...] = CONTEXTS
    cells_per_group: int = 400
    groups_per_shard: int = 10
    validation_rows: int = 64
    max_nnz: int = 4750000000
    max_counts_per_cell: int = 1000000
    concat_max_loaded_elements: int = 20000000
    max_rss_bytes: int = 16 << 30
    max_output_bytes: int = 128 << 30
    max_seconds: int = 21600

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name != "contexts":
                require(type(value) is int and value > 0, "Positive integer writer budget/dimension required: " + name)
        require(isinstance(self.contexts, tuple) and self.contexts == CONTEXTS,
            "Exactly A, B, C contexts required")
        require(self.expected_genes <= 18533 and self.expected_targets <= 300
            and self.cells_per_group <= 400 and self.groups_per_shard <= 10
            and self.validation_rows <= 256 and self.max_nnz <= 4750000000
            and self.max_counts_per_cell <= 1000000, "Writer limits may only be reduced for bounded tests")


def _names(values, count, label):
    require(isinstance(values, (list, tuple)) and len(values) == count
        and all(type(v) is str and v and v == v.strip() and "\x00" not in v and "|" not in v for v in values)
        and len(set(values)) == len(values), "Exact unique " + label + " required")
    return list(values)


def runtime_preflight():
    require(socket.gethostname().split(".")[0] in {"cbsuvlaminck3", "cbsuvlaminck6"},
        "Use an approved compute host, never a head/login node")


def _sha_json(value):
    return hashlib.sha256(old.canonical(value)).hexdigest()


def _dump(path, value):
    require(len(old.canonical(value)) <= safe.MAX_JSON_BYTES, "Writer JSON allocation bound")
    paths.dump_new(path, value)


def _private_h5(path):
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    return os.fdopen(os.open(path, flags, 0o600), "w+b")


def _obs(keys, cells_per_group):
    contexts = [c for c, _ in keys for _ in range(cells_per_group)]
    targets = [t for _, t in keys for _ in range(cells_per_group)]
    ids = [f"{c}|{t}|{i:03d}" for c, t in keys for i in range(cells_per_group)]
    return pd.DataFrame({"context": pd.Categorical(contexts), "target_gene": pd.Categorical(targets)},
        index=pd.Index(ids, name="cell_id"))


def read_control_metadata(handle, genes, context, *, expected_cells=18400, expected_ntc_ids=46):
    """Annotation-only; caller must authenticate the raw handle before counts.

    Official controls use nullable-string HDF groups for indexes. read_elem
    supports this encoding, unlike the older public-Replogle annotation helper.
    """
    require(context in CONTEXTS, "Unknown control context")
    require(type(expected_cells) is int and 0 < expected_cells <= 18400
        and type(expected_ntc_ids) is int and 0 < expected_ntc_ids <= 46,
        "Bounded official control metadata dimensions required")
    old.require_self_contained_h5(handle)
    obs, var = read_elem(handle["obs"]), read_elem(handle["var"])
    require(list(obs) == ["target_gene", "context", "ntc_id"] and not list(var), "Official control annotation schema differs")
    require(len(obs) == expected_cells and obs.index.is_unique and var.index.is_unique
        and not obs.index.hasnans and not var.index.hasnans and not obs.isna().any().any(),
        "Invalid/missing control annotation identities")
    require(var.index.astype(str).tolist() == list(genes), "Official control gene axis changed")
    require(set(obs["target_gene"].astype(str)) == {"non-targeting"}
        and set(obs["context"].astype(str)) == {context}, "Non-control/wrong-context cells admitted")
    counts = obs["ntc_id"].astype(str).value_counts()
    require(len(counts) == expected_ntc_ids and expected_cells % expected_ntc_ids == 0
        and (counts == expected_cells // expected_ntc_ids).all(), "Control NTC strata differ")
    x = handle["X"]
    require(isinstance(x, h5py.Group) and x.attrs.get("encoding-type") in ("csr_matrix", b"csr_matrix")
        and tuple(x.attrs["shape"]) == (expected_cells, len(genes)), "Official control CSR shape differs")
    return {"cells": obs.index.astype(str).tolist(), "ntc_ids": obs["ntc_id"].astype(str).tolist(),
        "context": context, "genes": list(genes), "control_only": True, "expression_values_read": False}


def validate_block(matrix, width, rows, config):
    require(sparse.isspmatrix_csr(matrix) and matrix.shape == (rows, width), "Exact CSR block dimensions required")
    require(matrix.data.dtype.kind in "iu" and matrix.data.dtype.itemsize <= 8
        and matrix.indices.dtype.kind in "iu" and matrix.indptr.dtype.kind in "iu", "Integer count CSR required")
    require(matrix.indptr.shape == (rows + 1,) and int(matrix.indptr[0]) == 0
        and int(matrix.indptr[-1]) == len(matrix.data) == len(matrix.indices)
        and np.all(matrix.indptr[1:] >= matrix.indptr[:-1])
        and np.all(np.diff(matrix.indptr.astype(np.int64)) <= width), "Invalid CSR block pointers")
    require(np.all(matrix.data > 0) and np.all(matrix.data <= config.max_counts_per_cell)
        and np.all(matrix.indices >= 0) and np.all(matrix.indices < width), "Invalid block values/indices or explicit zeros")
    require(matrix.has_canonical_format and matrix.has_sorted_indices, "Canonical duplicate-free sorted CSR required")
    for left, right in zip(matrix.indptr[:-1], matrix.indptr[1:]):
        left, right = int(left), int(right)
        require(right > left, "Zero count library")
        require(np.all(matrix.indices[left + 1:right] > matrix.indices[left:right - 1]),
            "Duplicate/unsorted CSR entries cannot rely on cached canonical flags")
    totals = np.asarray(matrix.sum(axis=1, dtype=np.int64)).ravel()
    require(np.all(totals > 0) and np.all(totals <= config.max_counts_per_cell), "Zero/oversized count library")
    return {"rows": rows, "nnz": int(matrix.nnz), "total_counts": int(totals.sum(dtype=np.int64)),
        "minimum_library": int(totals.min()), "maximum_library": int(totals.max())}


def _header(handle, genes, keys, config):
    old.require_self_contained_h5(handle)
    require(handle.attrs.get("encoding-type") in ("anndata", b"anndata"), "AnnData encoding required")
    obs, var = read_elem(handle["obs"]), read_elem(handle["var"])
    expected = _obs(keys, config.cells_per_group)
    require(list(var) == [] and var.index.is_unique and var.index.astype(str).tolist() == genes,
        "Exact full gene axis required before concatenation; no intersection/reindex/drop allowed")
    require(list(obs) == ["context", "target_gene"] and obs.index.is_unique
        and obs.index.astype(str).tolist() == expected.index.tolist(), "Observation identity/order/column mismatch")
    for key in ("context", "target_gene"):
        require(not obs[key].isna().any() and obs[key].astype(str).tolist() == expected[key].astype(str).tolist(),
            "Wrong per-context/target quota or row order")
    x = handle["X"]
    require(isinstance(x, h5py.Group) and x.attrs.get("encoding-type") in ("csr_matrix", b"csr_matrix")
        and tuple(x.attrs["shape"]) == (len(expected), len(genes)) and set(x) == {"data", "indices", "indptr"},
        "Exact full-axis CSR encoding required")
    data, indices, pointers = (x[k] for k in ("data", "indices", "indptr"))
    require(data.ndim == indices.ndim == pointers.ndim == 1 and len(data) == len(indices)
        and len(pointers) == len(expected) + 1 and data.dtype.kind in "iu" and data.dtype.itemsize <= 8
        and indices.dtype.kind == pointers.dtype.kind == "i" and indices.dtype.itemsize <= 8
        and pointers.dtype.itemsize <= 8, "Invalid final CSR dimensions/dtypes")
    require(len(data) <= config.max_nnz, "Official nnz cap exceeded")
    if len(data) >= np.iinfo(np.int32).max:
        require(pointers.dtype.itemsize >= 8, "Large CSR requires int64 indptr")
    return x, len(expected)


def validate_output(path, genes, targets, config=WriterConfig(), expected_nnz=None):
    """Fully stream final data/indices/pointers with <=validation_rows per read."""
    genes = _names(genes, config.expected_genes, "genes")
    targets = _names(targets, config.expected_targets, "targets")
    require("non-targeting" not in targets, "Submitted controls are forbidden")
    keys = [(c, t) for c in config.contexts for t in targets]
    total_counts, minimum, maximum = 0, None, 0
    with safe.regular_reader(path) as stream:
        before = bounded.signature(stream)
        with h5py.File(stream, "r") as handle:
            x, nrows = _header(handle, genes, keys, config)
            nnz = len(x["data"])
            require(expected_nnz is None or (type(expected_nnz) is int and nnz == expected_nnz),
                "Final nnz differs from emitted blocks")
            previous = 0
            for start in range(0, nrows, config.validation_rows):
                stop = min(start + config.validation_rows, nrows)
                ptr = x["indptr"][start:stop + 1].astype(np.int64)
                require(ptr[0] == previous and np.all(ptr >= 0) and np.all(ptr <= nnz)
                    and np.all(ptr[1:] >= ptr[:-1]) and np.all(np.diff(ptr) <= len(genes)),
                    "Malformed final CSR pointer or oversized row before value access")
                left, right = int(ptr[0]), int(ptr[-1])
                data, indices = x["data"][left:right], x["indices"][left:right]
                require(np.all(data > 0) and np.all(data <= config.max_counts_per_cell)
                    and np.all(indices >= 0) and np.all(indices < len(genes)), "Invalid final stored counts/indices")
                for a, b in zip(ptr[:-1] - left, ptr[1:] - left):
                    a, b = int(a), int(b)
                    require(b > a and np.all(indices[a + 1:b] > indices[a:b - 1]),
                        "Zero cell, duplicate or unsorted final gene indices")
                    total = int(data[a:b].sum(dtype=np.int64))
                    require(0 < total <= config.max_counts_per_cell, "Final count library exceeds official bounds")
                    total_counts += total
                    minimum = total if minimum is None else min(minimum, total)
                    maximum = max(maximum, total)
                previous = right
            require(previous == nnz, "Unassigned final sparse values")
        require(bounded.signature(stream) == before, "Final output changed during streaming validation")
    return {"shape": [len(keys) * config.cells_per_group, len(genes)], "groups": len(keys),
        "cells_per_group": config.cells_per_group, "nnz": nnz, "total_counts": total_counts,
        "minimum_library": minimum, "maximum_library": maximum, "full_axis_exact": True,
        "all_quotas_exact": True, "integer_nonnegative_counts": True, "canonical_csr": True,
        "unmodeled_genes_modified_by_writer": False, "sparsity_cap_applied": False,
        "gene_order_sha256": _sha_json(genes), "target_order_sha256": _sha_json(targets)}


class SubmissionWriter:
    def __init__(self, output_dir, genes, targets, generation_contract_sha, config=WriterConfig()):
        runtime_preflight()
        require(isinstance(config, WriterConfig), "Explicit writer configuration required")
        require(isinstance(generation_contract_sha, str) and safe.HASH.fullmatch(generation_contract_sha),
            "Generation contract SHA256 required")
        self.config = config
        self.genes = _names(genes, config.expected_genes, "genes")
        self.targets = _names(targets, config.expected_targets, "targets")
        require("non-targeting" not in self.targets, "Submitted controls forbidden")
        self.keys = [(c, t) for c in config.contexts for t in self.targets]
        self.out = paths.canonical_path(output_dir)
        require(not os.path.lexists(self.out), "Fresh writer directory required")
        self.out.mkdir(mode=0o700)
        (self.out / "groups").mkdir(mode=0o700)
        (self.out / "shards").mkdir(mode=0o700)
        self.contract_sha = generation_contract_sha
        self.started = time.monotonic()
        self.records, self.pending, self.shards = [], [], []
        self.nnz = 0
        self.finished = False
        self.journal = (self.out / "groups.jsonl").open("x", buffering=1)
        os.chmod(self.out / "groups.jsonl", 0o600)
        _dump(self.out / "writer.json", {"schema": SCHEMA, "generation_contract_sha256": generation_contract_sha,
            "config": asdict(config), "genes": self.genes, "targets": self.targets,
            "versions": {n: importlib.metadata.version(n) for n in ("anndata", "numpy", "scipy", "h5py", "pandas")},
            "submission_performed": False})
        self._resources()

    def _resources(self):
        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
        elapsed = time.monotonic() - self.started
        require(peak <= self.config.max_rss_bytes and elapsed <= self.config.max_seconds, "Writer RAM/wall-clock limit")
        total, stack = 0, [self.out]
        while stack:
            for entry in os.scandir(stack.pop()):
                info = entry.stat(follow_symlinks=False)
                if stat.S_ISDIR(info.st_mode):
                    stack.append(entry.path)
                else:
                    require(stat.S_ISREG(info.st_mode), "Writer artifacts must be regular files, never links/FIFOs")
                    total += info.st_size
        require(total <= self.config.max_output_bytes, "Writer disk bound exceeded")
        return {"peak_rss_bytes": peak, "elapsed_seconds": elapsed, "output_bytes": total}

    def _concat(self, records, filename):
        keys = []
        for record in records:
            require(safe.digest(record["path"]) == record["sha256"], "Concat input bytes changed")
            with safe.regular_reader(record["path"]) as stream, h5py.File(stream, "r") as handle:
                _header(handle, self.genes, [tuple(x) for x in record["keys"]], self.config)
            keys.extend(record["keys"])
        # concat_on_disk uses mode=w internally. Its sole output is inside this
        # newly created private staging directory; publication uses O_EXCL-like
        # hard-link creation and never overwrites the public destination.
        stage_dir = Path(tempfile.mkdtemp(prefix="concat_", dir=self.out))
        staged = stage_dir / "assembled.h5ad"
        concat_on_disk([r["path"] for r in records], staged, axis=0, join="inner", merge="same", uns_merge="same",
            max_loaded_elems=self.config.concat_max_loaded_elements)
        require(not staged.is_symlink() and staged.is_file(), "Expected regular staged H5AD")
        os.chmod(staged, 0o600)
        with safe.regular_reader(staged) as stream, h5py.File(stream, "r") as handle:
            _header(handle, self.genes, [tuple(x) for x in keys], self.config)
        destination = self.out / filename
        require(not os.path.lexists(destination), "Fresh concat publication target required")
        os.link(staged, destination, follow_symlinks=False)
        self._resources()
        return {"path": str(destination), "sha256": safe.digest(destination), "keys": keys,
            "nnz": sum(r["nnz"] for r in records)}

    def append(self, context, target, counts):
        require(not self.finished and len(self.records) < len(self.keys), "Writer already complete/full")
        require((context, target) == self.keys[len(self.records)], "Exact next context/target required; no duplicates or reorder")
        qc = validate_block(counts, len(self.genes), self.config.cells_per_group, self.config)
        require(self.nnz + qc["nnz"] <= self.config.max_nnz, "Official global nnz cap exceeded")
        ordinal = len(self.records)
        path = self.out / "groups" / f"g{ordinal:06d}.h5ad"
        matrix = counts.astype(np.int32, copy=True)
        with _private_h5(path) as stream:
            with h5py.File(stream, "w") as handle:
                handle.attrs.update({"encoding-type": "anndata", "encoding-version": "0.1.0"})
                write_elem(handle, "X", matrix, dataset_kwargs={"compression": "lzf"})
                write_elem(handle, "obs", _obs([(context, target)], self.config.cells_per_group))
                write_elem(handle, "var", pd.DataFrame(index=pd.Index(self.genes, name="gene_name")))
                for field in ("obsm", "varm", "obsp", "varp", "layers"):
                    write_elem(handle, field, {})
                write_elem(handle, "uns", {"method": "public source-specific GO flow; control-anchored raw counts",
                    "generation_contract_sha256": self.contract_sha, "writer_schema": SCHEMA})
            stream.flush()
            os.fsync(stream.fileno())
        record = {"ordinal": ordinal, "path": str(path), "sha256": safe.digest(path),
            "keys": [[context, target]], "nnz": qc["nnz"], "counts": qc}
        self.records.append(record)
        self.pending.append(record)
        self.nnz += qc["nnz"]
        self.journal.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        self.journal.flush()
        os.fsync(self.journal.fileno())
        self._resources()
        if len(self.pending) == self.config.groups_per_shard:
            self._flush()
        return qc

    def _flush(self):
        if self.pending:
            record = self._concat(self.pending, f"shards/s{len(self.shards):04d}.h5ad")
            self.shards.append(record)
            self.pending = []

    def finalize(self):
        require(not self.finished and len(self.records) == len(self.keys), "All exact official quotas required before finalization")
        self._flush()
        result = self._concat(self.shards, "prediction.h5ad")
        validation = validate_output(result["path"], self.genes, self.targets, self.config, self.nnz)
        require(safe.digest(result["path"]) == result["sha256"], "Final output changed after concatenation")
        self.journal.flush()
        os.fsync(self.journal.fileno())
        self.journal.close()
        receipt = {"schema": SCHEMA, "completed": True, "generation_contract_sha256": self.contract_sha,
            "completed_at": datetime.now(timezone.utc).isoformat(), "output_path": result["path"],
            "output_sha256": result["sha256"], "output_bytes": Path(result["path"]).stat().st_size,
            "writer_sha256": safe.digest(self.out / "writer.json"),
            "group_journal_sha256": safe.digest(self.out / "groups.jsonl"), "validation": validation,
            "resources": self._resources(), "submission_performed": False, "official_cli_validated": False,
            "intermediates_preserved": True}
        _dump(self.out / "complete.json", receipt)
        self.finished = True
        return receipt

    def close(self):
        if not self.journal.closed:
            self.journal.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
