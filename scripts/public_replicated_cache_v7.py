"""Bounded, authenticated replication-expanded public cache; no model fitting.

All source bytes and complete metadata row roles are checked before any X
values. Normalization is identical to v2 but performed in bounded row chunks.
Every old cache group must replay exactly before a new receipt is published.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import resource
import socket
import zipfile

import h5py
import numpy as np

import prepare_public_control_anchor_v2 as v2
import prepare_public_flow_pilot as old
import prepare_public_replicated_v7 as prep
import public_training_readiness as safe
from train_public_flow_pilot import write_new

MAX_ARCHIVE = 512 << 20
MAX_RSS = 8 << 30
CHUNK_ROWS = 256
RECEIPT_SCHEMA = "public-replicated-control-anchor-cache-receipt-v7"
CACHE_KEYS = v2.CACHE_KEYS
POOLS = v2.POOLS


def require(condition, message):
    if not condition:
        raise ValueError(message)


def equal(actual, expected, message):
    require(json.dumps(actual, sort_keys=True, allow_nan=False) ==
            json.dumps(expected, sort_keys=True, allow_nan=False), message)


def signature(stream):
    info = os.fstat(stream.fileno())
    return tuple(getattr(info, name) for name in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns"))


def check_rss():
    # Approved execution hosts are Linux, where ru_maxrss is measured in KiB.
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    require(peak <= MAX_RSS, "Extraction exceeded the registered 8GiB peak-RSS limit")
    return peak


def _indices(values, maximum, label):
    require(isinstance(values, (list, tuple)) and bool(values)
            and all(type(x) is int and 0 <= x < maximum for x in values)
            and len(set(values)) == len(values), "Invalid " + label)
    return list(values)


def _layout(handle, n_obs, n_vars):
    require(type(n_obs) is int and type(n_vars) is int and n_obs > 0 and n_vars > 0,
            "Registered expression dimensions required")
    x = handle["X"]
    shape = tuple(x.shape) if isinstance(x, h5py.Dataset) else tuple(x.attrs["shape"])
    require(shape == (n_obs, n_vars), "Expression shape differs from registered metadata")
    if isinstance(x, h5py.Dataset):
        require(x.dtype.kind in "iuf" and x.dtype.itemsize <= 8, "Unsupported dense raw-count dtype")
    else:
        require(x.attrs.get("encoding-type") in ("csr_matrix", b"csr_matrix"),
                "Only dense or CSR raw counts are supported")
        require(set(x) == {"data", "indices", "indptr"}, "Exact CSR member schema required")
        data, indices, pointers = x["data"], x["indices"], x["indptr"]
        require(data.ndim == indices.ndim == pointers.ndim == 1
                and len(data) == len(indices) and len(pointers) == n_obs + 1
                and data.dtype.kind in "iuf" and data.dtype.itemsize <= 8
                and indices.dtype.kind in "iu" and pointers.dtype.kind in "iu",
                "Invalid CSR dimensions or dtypes")
    return x


def read_normalized_output_rows(handle, rows, *, authorized_rows, n_obs, n_vars,
                                shared_columns, output_indices, chunk_rows=CHUNK_ROWS):
    """Validate the whole request, then normalize per-row over the full shared axis."""
    requested = _indices(rows, n_obs, "requested rows")
    authorized = _indices(authorized_rows, n_obs, "authorized rows")
    require(requested == sorted(requested) and authorized == sorted(authorized)
            and set(requested) <= set(authorized), "Unauthorized, unordered or duplicate row request")
    columns = _indices(shared_columns, n_vars, "shared gene columns")
    output = _indices(output_indices, len(columns), "output gene indices")
    require(type(chunk_rows) is int and 1 <= chunk_rows <= CHUNK_ROWS, "Invalid row chunk budget")
    x = _layout(handle, n_obs, n_vars)
    # Pointer reads follow full row authorization and source authentication by caller.
    # Bound each selected sparse row BEFORE old.read_authorized_rows slices its payload.
    if isinstance(x, h5py.Group):
        for row in requested:
            start, end = map(int, x["indptr"][row:row + 2])
            require(0 <= start <= end <= len(x["data"]) and end - start <= n_vars,
                    "Invalid or oversized authorized CSR row")
    result = np.empty((len(requested), len(output)), dtype=np.float32)
    for offset in range(0, len(requested), chunk_rows):
        part = requested[offset:offset + chunk_rows]
        counts = old.read_authorized_rows(handle, part, authorized=authorized, columns=columns)
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                totals = counts.sum(axis=1, dtype=np.float64)
                require(np.isfinite(totals).all() and np.all(totals > 0), "Invalid shared-axis library sums")
                values = old.normalize_counts(counts)
        except FloatingPointError as error:
            raise ValueError("Nonfinite raw-count normalization arithmetic") from error
        require(values.dtype == np.float32 and np.isfinite(values).all() and np.all(values >= 0),
                "Normalization produced invalid expression")
        result[offset:offset + len(part)] = values[:, output]
        check_rss()
    return result


def roster(contract):
    return [(source, group) for source in contract["sources"] for group in source["groups"]]


def offsets_for(contract):
    rows = roster(contract)
    return {name: np.cumsum([0] + [len(group[field]) for _, group in rows], dtype=np.int64)
            for name, field in POOLS.items()}


def empty_cache(contract, contract_sha, source_sha):
    rows = roster(contract)
    arrays = {name: np.array(value) for name, value in {
        "schema": prep.CACHE_SCHEMA, "representation": prep.REPRESENTATION,
        "scope": "calibration", "contract_sha256": contract_sha,
        "source_manifest_sha256": source_sha}.items()}
    arrays.update(genes=np.asarray(contract["output_genes"]),
                  targets=np.asarray([g["target"] for _, g in rows]),
                  sources=np.asarray([s["id"] for s, _ in rows]),
                  batches=np.asarray([g["batch"] for _, g in rows]))
    width = len(contract["output_genes"])
    planned = sum(a.nbytes for a in arrays.values()) + len(rows) * width * 2 * 4
    for name, field in POOLS.items():
        planned += sum(len(g[field]) for _, g in rows) * (width * 4 + 16)
    require(planned + 65536 <= MAX_ARCHIVE, "Planned cache exceeds archive allocation limit")
    arrays["context"] = np.empty((len(rows), width * 2), dtype=np.float32)
    for name, field in POOLS.items():
        counts = [len(g[field]) for _, g in rows]
        arrays[name] = np.empty((sum(counts), width), dtype=np.float32)
        arrays[name + "_rows"] = np.asarray([r for _, g in rows for r in g[field]], dtype=np.int64)
        arrays["group" if name == "treated" else name + "_group"] = np.repeat(
            np.arange(len(rows), dtype=np.int64), counts)
    return arrays


def validate_cache(arrays, contract, contract_sha, source_sha):
    require(set(arrays) == CACHE_KEYS, "Unexpected replicated cache fields")
    for name, expected in {"schema": prep.CACHE_SCHEMA, "representation": prep.REPRESENTATION,
                           "scope": "calibration", "contract_sha256": contract_sha,
                           "source_manifest_sha256": source_sha}.items():
        value = arrays[name]
        require(value.shape == () and value.dtype.kind == "U" and str(value) == expected,
                "Replicated cache identity mismatch")
    rows, width = roster(contract), len(contract["output_genes"])
    for name, expected in {"genes": contract["output_genes"],
                           "targets": [g["target"] for _, g in rows],
                           "sources": [s["id"] for s, _ in rows],
                           "batches": [g["batch"] for _, g in rows]}.items():
        require(arrays[name].dtype.kind == "U" and arrays[name].shape == (len(expected),)
                and arrays[name].tolist() == expected, "Replicated cache axis mismatch")
    offsets = offsets_for(contract)
    seen = {}
    for name, field in POOLS.items():
        counts = [len(g[field]) for _, g in rows]
        labels = arrays["group" if name == "treated" else name + "_group"]
        originals, values = arrays[name + "_rows"], arrays[name]
        require(labels.dtype == np.int64 and np.array_equal(labels, np.repeat(
            np.arange(len(rows), dtype=np.int64), counts)), "Unauthorized cache group ordering")
        require(originals.dtype == np.int64 and originals.shape == (sum(counts),)
                and originals.tolist() == [r for _, g in rows for r in g[field]],
                "Unauthorized original-row mapping")
        require(values.dtype == np.float32 and values.shape == (sum(counts), width)
                and np.isfinite(values).all() and np.all(values >= 0)
                and np.all(values <= np.float32(np.log1p(10000))),
                "Cache expression must be finite nonnegative float32 log1p-CP10k")
        for group_id, (source, _) in enumerate(rows):
            left, right = offsets[name][group_id:group_id + 2]
            for original, value in zip(originals[left:right], values[left:right]):
                key = (source["id"], int(original))
                if key in seen:
                    require(np.array_equal(seen[key], value), "Repeated physical cell expression differs")
                else:
                    seen[key] = value
    context = arrays["context"]
    require(context.dtype == np.float32 and context.shape == (len(rows), width * 2)
            and np.isfinite(context).all(), "Invalid context summary")
    for group_id in range(len(rows)):
        left, right = offsets["context_control"][group_id:group_id + 2]
        value = arrays["context_control"][left:right].astype(np.float64)
        expected = np.concatenate((value.mean(0), value.std(0))).astype(np.float32)
        require(np.array_equal(context[group_id], expected), "Context does not replay from its independent pool")
    require(sum(a.nbytes for a in arrays.values()) + 65536 <= MAX_ARCHIVE,
            "Expanded cache arrays exceed registered limit")
    return arrays


def read_npz(path, sha, keys=CACHE_KEYS, *, return_metadata=False):
    require(isinstance(sha, str) and safe.HASH.fullmatch(sha), "Explicit SHA-256 required")
    with safe.regular_reader(path) as stream:
        before = signature(stream)
        require(before[2] <= MAX_ARCHIVE, "Oversized compressed cache")
        payload = stream.read(MAX_ARCHIVE + 1)
        require(signature(stream) == before, "Cache changed while reading")
    require(len(payload) <= MAX_ARCHIVE and hashlib.sha256(payload).hexdigest() == sha,
            "Cache SHA-256 mismatch")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = archive.infolist()
        require(len(members) == len(keys) and {m.filename for m in members} == {k + ".npy" for k in keys},
                "Exact safe NPZ member roster required")
        expanded_size = sum(m.file_size for m in members)
        require(expanded_size <= MAX_ARCHIVE, "Oversized expanded cache")
        for member in members:
            with archive.open(member) as stream:
                version = np.lib.format.read_magic(stream)
                require(version in {(1, 0), (2, 0)}, "Unsupported safe NPY version")
                read_header = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                               else np.lib.format.read_array_header_2_0)
                shape, _, dtype = read_header(stream)
                require(not dtype.hasobject and dtype.fields is None
                        and all(type(n) is int and n >= 0 for n in shape),
                        "Unsafe NPY dtype or shape")
                require(math.prod(shape) * dtype.itemsize == member.file_size - stream.tell(),
                        "NPY allocation differs from bounded payload")
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in keys}
    details = {"cache_bytes": len(payload), "expanded_archive_bytes": expanded_size,
               "array_bytes": sum(value.nbytes for value in arrays.values())}
    return (arrays, details) if return_metadata else arrays


def verify_parent_subset(arrays, contract, *, repo=None):
    bindings = contract["parent_v2"]
    parent = v2.load_contract(Path(bindings["contract"]["path"]), bindings["contract"]["sha256"],
                              Path(bindings["sources"]["path"]), bindings["sources"]["sha256"], repo=repo)
    previous = read_npz(Path(bindings["cache"]["path"]), bindings["cache"]["sha256"])
    v2.validate_cache(previous, parent, bindings["contract"]["sha256"], bindings["sources"]["sha256"])
    old_rows, new_rows = roster(parent), roster(contract)
    expected = []
    lookup = {(s["id"], g["target"], g["batch"]): i for i, (s, g) in enumerate(new_rows)}
    require(len(lookup) == len(new_rows), "Duplicate expanded group identity")
    for old_id, (source, group) in enumerate(old_rows):
        new_id = lookup[(source["id"], group["target"], group["batch"])]
        equal(new_rows[new_id][1], group, "Parent group membership changed")
        expected.append({"parent_group_id": old_id, "expanded_group_id": new_id})
    equal(contract["parent_group_mapping"], expected, "Parent mapping changed")
    require(np.array_equal(previous["genes"], arrays["genes"]), "Parent output gene axis changed")
    old_offsets, new_offsets = offsets_for(parent), offsets_for(contract)
    for mapping in expected:
        oi, ni = mapping["parent_group_id"], mapping["expanded_group_id"]
        require(np.array_equal(previous["context"][oi], arrays["context"][ni]),
                "Parent context arithmetic changed")
        for name in POOLS:
            old_left, old_right = old_offsets[name][oi:oi + 2]
            left, right = new_offsets[name][ni:ni + 2]
            for key in (name, name + "_rows"):
                require(previous[key].dtype == arrays[key].dtype
                        and np.array_equal(previous[key][old_left:old_right], arrays[key][left:right]),
                        "Parent normalized values or original rows changed")
    return {"passed": True, "groups_verified": len(expected), "all_pools_bit_identical": True,
            "context_bit_identical": True, "raw_sources_reread_by_comparison": False}


def load_cache(cache_path, cache_sha, contract_path, contract_sha, sources_path, source_sha, *, repo=None):
    contract = prep.load_contract(contract_path, contract_sha, sources_path, source_sha, repo=repo)
    receipt = safe.read_json(Path(str(cache_path) + ".receipt.json"))
    expected = {"schema": RECEIPT_SCHEMA, "completed": True,
                "contract_sha256": contract_sha, "source_manifest_sha256": source_sha,
                "cache_sha256": cache_sha, "model_fitting_performed": False,
                "posttraining_performed": False, "submission_performed": False, "download_performed": False,
                "raw_sources_authenticated_before_expression": True, "gpu_used": False,
                "cloud_synced": False, "rpe1_h1_hepg2_challenge2026_feng_opened": False,
                "representation": prep.REPRESENTATION, "scope": "calibration", "stage": "data_preparation",
                "row_chunk_limit": CHUNK_ROWS, "groups": len(roster(contract)),
                "cache_path": str(Path(cache_path).resolve()),
                "source_extraction": [{"source": spec["id"],
                    "unique_rows": len({row for group in spec["groups"] for field in POOLS.values() for row in group[field]}),
                    "row_chunks": math.ceil(len({row for group in spec["groups"] for field in POOLS.values() for row in group[field]}) / CHUNK_ROWS)}
                    for spec in contract["sources"]]}
    for key, value in expected.items():
        equal(receipt.get(key), value, "Cache receipt binding mismatch")
    extra = {"completed_at", "cache_bytes", "array_bytes", "expanded_archive_bytes", "population_entries",
             "parent_subset_replay", "peak_rss_bytes", "hostname"}
    require(set(receipt) == set(expected) | extra, "Cache receipt field roster changed")
    safe.timestamp(receipt["completed_at"])
    require(isinstance(receipt["hostname"], str)
            and receipt["hostname"].split(".")[0] in {"cbsuvlaminck3", "cbsuvlaminck6"},
            "Cache preparation host was not approved")
    require(type(receipt["peak_rss_bytes"]) is int and 0 < receipt["peak_rss_bytes"] <= MAX_RSS,
            "Cache preparation exceeded registered RAM limit")
    arrays, details = read_npz(cache_path, cache_sha, return_metadata=True)
    validate_cache(arrays, contract, contract_sha, source_sha)
    for key, value in details.items():
        equal(receipt.get(key), value, "Cache receipt allocation or archive size changed")
    equal(receipt.get("population_entries"), {name: len(arrays[name]) for name in POOLS},
          "Cache receipt population sizes changed")
    replay = verify_parent_subset(arrays, contract, repo=repo)
    equal(receipt.get("parent_subset_replay"), replay, "Parent replay receipt mismatch")
    return arrays, contract


def materialize(contract_path, contract_sha, sources_path, source_sha, output, *, repo=None):
    output = Path(output)
    receipt_path = Path(str(output) + ".receipt.json")
    require(not os.path.lexists(output) and not os.path.lexists(receipt_path), "Fresh cache and receipt required")
    contract = prep.load_contract(contract_path, contract_sha, sources_path, source_sha, repo=repo)
    arrays = empty_cache(contract, contract_sha, source_sha)
    offsets = offsets_for(contract)
    opened, metas, extraction = [], {}, []
    with ExitStack() as stack:
        # Authenticate every source handle before reading any X value from either source.
        for spec in contract["sources"]:
            stream = stack.enter_context(safe.regular_reader(spec["path"]))
            before = signature(stream)
            require(before[2] == spec["size_bytes"] and old.sha_stream(stream) == spec["sha256"],
                    "Source authentication failed BEFORE expression access")
            require(signature(stream) == before, "Source changed during hashing")
            handle = stack.enter_context(h5py.File(stream, "r"))
            meta = old.metadata(handle, spec)
            require(old.metadata_digest(meta) == spec["metadata_sha256"], "Source metadata changed")
            metas[spec["id"]] = meta
            opened.append((spec, stream, handle, before))
            print(json.dumps({"stage": "source_authenticated", "source": spec["id"]}), flush=True)
        prep.validate_metadata(contract, metas)
        for _, stream, _, before in opened:
            require(signature(stream) == before, "Source changed before expression access")
        group_offset = 0
        for spec, stream, handle, before in opened:
            meta = metas[spec["id"]]
            selected = sorted({row for group in spec["groups"] for field in POOLS.values() for row in group[field]})
            lookup = {gene: index for index, gene in enumerate(meta["genes"])}
            counts = Counter(meta["genes"])
            require(all(counts[g] == 1 for g in contract["shared_genes"]), "Nonunique shared measured gene")
            shared_columns = [lookup[g] for g in contract["shared_genes"]]
            output_indices = [contract["shared_genes"].index(g) for g in contract["output_genes"]]
            values = read_normalized_output_rows(handle, selected, authorized_rows=selected,
                n_obs=spec["cells_total"], n_vars=spec["genes_total"], shared_columns=shared_columns,
                output_indices=output_indices)
            row_lookup = {row: index for index, row in enumerate(selected)}
            for local_id, group in enumerate(spec["groups"]):
                gid = group_offset + local_id
                for name, field in POOLS.items():
                    left, right = offsets[name][gid:gid + 2]
                    arrays[name][left:right] = values[[row_lookup[row] for row in group[field]]]
                    if name == "context_control":
                        context = arrays[name][left:right].astype(np.float64)
                        arrays["context"][gid] = np.concatenate((context.mean(0), context.std(0))).astype(np.float32)
            group_offset += len(spec["groups"])
            require(signature(stream) == before, "Source changed during expression extraction")
            extraction.append({"source": spec["id"], "unique_rows": len(selected),
                               "row_chunks": math.ceil(len(selected) / CHUNK_ROWS)})
            print(json.dumps({"stage": "source_materialized", **extraction[-1]}), flush=True)
            del values
        for _, stream, _, before in opened:
            require(signature(stream) == before, "Source changed before finalization")
    validate_cache(arrays, contract, contract_sha, source_sha)
    parent_audit = verify_parent_subset(arrays, contract, repo=repo)
    prep.load_contract(contract_path, contract_sha, sources_path, source_sha, repo=repo)
    for value in arrays.values():
        value.setflags(write=False)
    check_rss()
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    payload = buffer.getvalue()
    require(len(payload) <= MAX_ARCHIVE, "Compressed cache exceeds registered archive limit")
    # Includes NPY headers, not merely sum(array.nbytes).
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        expanded = sum(m.file_size for m in archive.infolist())
    require(expanded <= MAX_ARCHIVE, "Expanded archive including headers exceeds limit")
    peak = check_rss()
    write_new(output, payload)
    cache_sha = hashlib.sha256(payload).hexdigest()
    require(safe.digest(output) == cache_sha, "Written cache failed byte verification")
    receipt = {"schema": RECEIPT_SCHEMA, "completed": True,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "cache_path": str(output.resolve()), "cache_sha256": cache_sha, "cache_bytes": len(payload),
        "array_bytes": sum(a.nbytes for a in arrays.values()), "expanded_archive_bytes": expanded,
        "contract_sha256": contract_sha, "source_manifest_sha256": source_sha,
        "representation": prep.REPRESENTATION, "scope": "calibration", "stage": "data_preparation",
        "groups": len(arrays["targets"]), "population_entries": {n: len(arrays[n]) for n in POOLS},
        "source_extraction": extraction, "parent_subset_replay": parent_audit,
        "raw_sources_authenticated_before_expression": True, "row_chunk_limit": CHUNK_ROWS,
        "peak_rss_bytes": peak, "hostname": socket.gethostname(), "gpu_used": False,
        "model_fitting_performed": False, "posttraining_performed": False,
        "submission_performed": False, "download_performed": False, "cloud_synced": False,
        "rpe1_h1_hepg2_challenge2026_feng_opened": False}
    write_new(receipt_path, old.canonical(receipt))
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("materialize", "verify"))
    for name in ("contract", "sources", "cache"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--cache-sha256")
    args = parser.parse_args(argv)
    require(socket.gethostname().split(".")[0] in {"cbsuvlaminck3", "cbsuvlaminck6"},
            "Use an approved BioHPC compute host, never a login/head node")
    if args.action == "materialize":
        result = materialize(args.contract, args.contract_sha256, args.sources,
                             args.source_manifest_sha256, args.cache)
    else:
        if not args.cache_sha256:
            parser.error("Verification requires --cache-sha256")
        arrays, _ = load_cache(args.cache, args.cache_sha256, args.contract, args.contract_sha256,
                              args.sources, args.source_manifest_sha256)
        result = {"verified": True, "groups": len(arrays["targets"]),
                  "model_fitting_performed": False, "submission_performed": False}
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
