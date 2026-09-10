"""Deduplicated, source-specific auxiliary expression preparation; never training.

The v9 candidate plan alone does not permit expression reads. This separate
registration seals the materialization policy and dependency closure first.
Both complete raw files and the reconstructed candidate metadata must then
authenticate before any selected expression value can be read.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import csv
from datetime import datetime, timezone, timedelta
import hashlib
import io
import math
import os
from pathlib import Path
import resource
import socket
import time
import zipfile

import h5py
import numpy as np

import prepare_public_auxiliary_v9 as candidate
import public_replicated_cache_v7 as bounded
import public_training_readiness as safe

SCHEMA = "public-auxiliary-materialization-v10"
CACHE_SCHEMA = "public-auxiliary-deduplicated-cache-v10"
RECEIPT_SCHEMA = "public-auxiliary-materialization-receipt-v10"
REPRESENTATION = "log1p(CP10000); normalize over frozen shared measured genes before output subset"
CHUNK_ROWS = 256
MAX_ARCHIVE = 512 << 20
MAX_DISK = 1 << 30
MAX_RSS = 8 << 30
MAX_SECONDS = 3600
PARENT_SHARED_GENE_COUNT = 7563
PARENT_OUTPUT_GENE_COUNT = 512
SHARED_GENE_COUNT = 7097
OUTPUT_GENE_COUNT = 482
OFFICIAL_GENE_COUNT = 18533
OFFICIAL_AXIS = Path(__file__).resolve().parents[1] / "dataset/controls/gene_names.csv"
OFFICIAL_AXIS_SHA = "25bfa66715e186bebabce7ac788bbcea47e2bf59ca70be1f8f3a06f2f0e47201"
OFFICIAL_AXIS_BYTES = 119295
MODULES = tuple(dict.fromkeys(("public_auxiliary_cache_v10.py", "public_replicated_cache_v7.py", *candidate.MODULES)))
KEYS = frozenset({"schema", "representation", "source", "contract_sha256", "plan_sha256", "genes",
    "original_rows", "row_role", "expression", "group_ids", "targets", "batches", "pool_ids",
    "pool_batches", "group_pool", "treated_offsets", "treated_indices", "fit_reference_indices",
    "context_indices", "context"})
require, equal = candidate.require, candidate.equal
canonical_path, old = candidate.canonical_path, candidate.old


def runtime_preflight():
    require(socket.gethostname().split(".")[0] in {"cbsuvlaminck3", "cbsuvlaminck6"},
        "Auxiliary preparation requires an approved compute node, never login/head")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Auxiliary preparation does not use a GPU")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        require(os.environ.get(name) == "2", "Auxiliary preparation requires two numerical threads")
    from threadpoolctl import threadpool_info
    require(all(p["num_threads"] <= 2 for p in threadpool_info()), "Active numerical thread pool exceeds two")


def checkpoint(started):
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    elapsed = time.monotonic() - started
    require(0 < peak <= MAX_RSS and 0 <= elapsed <= MAX_SECONDS,
        "Auxiliary preparation exceeded registered RSS/time checkpoints")
    return {"peak_rss_bytes": peak, "elapsed_seconds": elapsed}


def policy():
    return {"stage": "data_preparation", "expression_materialization_authorized": True,
        "training_authorized": False, "separate_training_contract_required": True,
        "source_routing": "independent source caches; frozen v9 source-specific warmstart routes only",
        "raw_sources": "both complete open source files hashed before any X values; same handles retained",
        "metadata": "full frozen v9 candidate reconstruction before any X values",
        "allowed_rows": "exact deduplicated candidate treated, fit_reference16 and context32 only",
        "forbidden_rows": "all panel-treated, protected-treated, tuning-anchor and sham rows",
        "context_role": "conditioning only, never response or prior targets",
        "normalization": REPRESENTATION, "shared_gene_count": SHARED_GENE_COUNT,
        "output_gene_count": OUTPUT_GENE_COUNT, "dtype": "float32", "row_chunk_limit": CHUNK_ROWS,
        "axis_projection": "exact official gene_name membership, preserving parent shared/output order; no aliasing or zero filling",
        "parent_shared_gene_count": PARENT_SHARED_GENE_COUNT, "parent_output_gene_count": PARENT_OUTPUT_GENE_COUNT,
        "official_gene_count": OFFICIAL_GENE_COUNT,
        "physical_identity": "source identifier plus exact original row number",
        "cache_layout": "one expression matrix per source with unique original rows and explicit group/pool indices",
        "row_roles": {"treated": 1, "fit_reference": 2, "context": 3},
        "context_summary": "float64 population mean/std of context32 only, concatenated then float32",
        "cpu_threads_max": 2, "gpu_used": False, "peak_rss_limit_bytes": MAX_RSS,
        "elapsed_limit_seconds": MAX_SECONDS, "archive_limit_bytes": MAX_ARCHIVE,
        "output_disk_limit_bytes": MAX_DISK, "runtime_limits": "RSS/time checkpoints; preflight numerical threads",
        "full_gene_count_emitter": False, "model_fitting_performed": False,
        "posttraining_performed": False, "download_performed": False,
        "submission_performed": False, "cloud_synced": False, "automatic_promotion": False}


def _codes():
    return {name: safe.digest(Path(__file__).with_name(name)) for name in MODULES}


def official_gene_axis():
    """Only authenticated public submission metadata, never challenge expression."""
    path = canonical_path(OFFICIAL_AXIS)
    with safe.regular_reader(path) as stream:
        before = bounded.signature(stream)
        require(before[2] == OFFICIAL_AXIS_BYTES and before[2] <= safe.MAX_JSON_BYTES,
            "Official gene metadata byte size differs")
        payload = stream.read(safe.MAX_JSON_BYTES + 1)
        require(bounded.signature(stream) == before and hashlib.sha256(payload).hexdigest() == OFFICIAL_AXIS_SHA,
            "Official gene metadata SHA256 or open identity differs")
    rows = list(csv.reader(io.StringIO(payload.decode("utf-8"), newline="")))
    require(rows and rows[0] == ["gene_name"] and all(len(row) == 1 for row in rows[1:]),
        "Exact single-column official gene_name CSV required")
    genes = [row[0] for row in rows[1:]]
    require(len(genes) == OFFICIAL_GENE_COUNT and len(set(genes)) == len(genes)
        and all(g and g == g.strip() for g in genes), "Official gene axis must contain unique exact symbols")
    return genes


def project_axes(plan):
    """Project axes only; the original plan remains intact for all v9 row replay."""
    require(len(plan["shared_genes"]) == PARENT_SHARED_GENE_COUNT
        and len(plan["output_genes"]) == PARENT_OUTPUT_GENE_COUNT,
        "Frozen parent auxiliary gene axis dimensions changed")
    genes = official_gene_axis()
    names = set(genes)
    shared = [g for g in plan["shared_genes"] if g in names]
    output = [g for g in plan["output_genes"] if g in names]
    require(len(shared) == SHARED_GENE_COUNT and len(output) == OUTPUT_GENE_COUNT
        and len(set(shared)) == len(shared) and len(set(output)) == len(output)
        and set(output) <= set(shared), "Official-intersection auxiliary axis dimensions changed")
    projection = {"method": "exact gene symbol intersection preserving original shared/output order",
        "official_metadata": {"path": str(canonical_path(OFFICIAL_AXIS)), "sha256": OFFICIAL_AXIS_SHA},
        "official_metadata_bytes": OFFICIAL_AXIS_BYTES, "official_gene_count": len(genes),
        "official_gene_order_sha256": hashlib.sha256(old.canonical(genes)).hexdigest(),
        "parent_shared_genes": plan["shared_genes"], "parent_output_genes": plan["output_genes"],
        "dropped_shared_genes": [g for g in plan["shared_genes"] if g not in names],
        "dropped_output_genes": [g for g in plan["output_genes"] if g not in names],
        "new_rows_authorized": False, "aliasing_performed": False, "zero_filling_performed": False}
    return {**plan, "shared_genes": shared, "output_genes": output}, projection


def _build(plan_path, plan_sha, plan, protocol, created_at):
    safe.timestamp(created_at)
    view, projection = project_axes(plan)
    capacities = {source: {k: v for k, v in counts.items()
        if k != "deduplicated512_float32_expression_bytes_if_later_authorized"}
        for source, counts in plan["capacity"].items()}
    for counts in capacities.values():
        counts["deduplicated_output_expression_bytes"] = counts["unique_candidate_rows"] * OUTPUT_GENE_COUNT * 4
    return {"schema": SCHEMA, "created_at": created_at, "policy": policy(),
        "candidate_plan": {"path": str(plan_path), "sha256": plan_sha},
        "protocol": {"path": str(protocol), "sha256": safe.digest(protocol)}, "code_sha256": _codes(),
        "source_routing": plan["source_routing"], "capacity": capacities, "parent_capacity": plan["capacity"],
        "axis_projection": projection,
        "sources": [{"id": s["id"], "acquisition": s["acquisition"],
            "metadata_sha256": s["metadata_sha256"], "cache_filename": s["id"] + ".npz"} for s in plan["sources"]],
        "shared_genes": view["shared_genes"], "output_genes": view["output_genes"],
        "expression_materialization_authorized": True, "training_authorized": False,
        "model_fitting_performed": False, "posttraining_performed": False, "submission_performed": False}


def _registration_completion(sha):
    return {"schema": SCHEMA, "registration_completed": True, "contract_sha256": sha,
        "raw_expression_values_read": False, "training_authorized": False}


def register(plan_path, plan_sha, protocol_path, output_dir):
    """Create a fresh materialization contract; metadata and hashes only, no X."""
    runtime_preflight()
    started = time.monotonic()
    plan_path, protocol, out = map(canonical_path, (plan_path, protocol_path, output_dir))
    require(not os.path.lexists(out), "Fresh materialization registration directory required")
    plan = candidate.load_registration(plan_path, plan_sha)
    record = _build(plan_path, plan_sha, plan, protocol, datetime.now(timezone.utc).isoformat())
    checkpoint(started)
    equal(_build(plan_path, plan_sha, candidate.load_registration(plan_path, plan_sha),
        protocol, record["created_at"]), record, "Materialization registration inputs changed")
    checkpoint(started)
    out.mkdir(mode=0o700)
    candidate._dump(out / "contract.json", record)
    sha = safe.digest(out / "contract.json")
    candidate._dump(out / "registration_complete.json", _registration_completion(sha))
    return sha


def validate_contract(path, sha):
    path = canonical_path(path)
    require(path.name == "contract.json", "Canonical auxiliary contract.json leaf required")
    record = safe.read_json(path, sha)
    require(record.get("schema") == SCHEMA, "Wrong auxiliary materialization schema")
    plan_path = canonical_path(record["candidate_plan"]["path"])
    plan_sha = record["candidate_plan"]["sha256"]
    plan = candidate.load_registration(plan_path, plan_sha)
    expected = _build(plan_path, plan_sha, plan, canonical_path(record["protocol"]["path"]), record["created_at"])
    equal(record, expected, "Complete auxiliary materialization contract reconstruction")
    require(datetime.fromisoformat(record["created_at"]) <= datetime.now(timezone.utc) + timedelta(seconds=300),
        "Materialization registration is in the future")
    equal(safe.read_json(path.with_name("registration_complete.json")), _registration_completion(sha),
        "Auxiliary registration completion binding")
    return record, plan


def selected_rows(source):
    roles = [{r for g in source["groups"] for r in g["treated_rows"]},
        {r for p in source["control_pools"] for r in p["fit_reference_rows"]},
        {r for p in source["control_pools"] for r in p["context_rows"]}]
    require(all(roles) and all(not roles[a] & roles[b] for a, b in ((0, 1), (0, 2), (1, 2))),
        "Candidate source physical roles overlap")
    require(all(type(r) is int and r >= 0 for rows in roles for r in rows), "Invalid original row identity")
    return sorted(set.union(*roles)), roles


def empty_cache(source, plan, contract_sha):
    """Deterministic source-local array layout, independent of original-row range."""
    rows, roles = selected_rows(source)
    positions = {r: i for i, r in enumerate(rows)}
    pools, groups = source["control_pools"], source["groups"]
    pool_lookup = {p["pool_id"]: i for i, p in enumerate(pools)}
    require(len(pool_lookup) == len(pools), "Duplicate source pool identity")
    arrays = {key: np.asarray(value) for key, value in {
        "schema": CACHE_SCHEMA, "representation": REPRESENTATION, "source": source["id"],
        "contract_sha256": contract_sha, "plan_sha256": plan["_binding_sha256"],
        "genes": plan["output_genes"], "group_ids": [g["group_id"] for g in groups],
        "targets": [g["target"] for g in groups], "batches": [g["batch"] for g in groups],
        "pool_ids": [p["pool_id"] for p in pools], "pool_batches": [p["batch"] for p in pools]}.items()}
    arrays.update(original_rows=np.asarray(rows, dtype=np.int64),
        row_role=np.asarray([next(i + 1 for i, role in enumerate(roles) if row in role) for row in rows], dtype=np.uint8),
        group_pool=np.asarray([pool_lookup[g["pool_id"]] for g in groups], dtype=np.int64),
        treated_offsets=np.cumsum([0] + [len(g["treated_rows"]) for g in groups], dtype=np.int64),
        treated_indices=np.asarray([positions[r] for g in groups for r in g["treated_rows"]], dtype=np.int64),
        fit_reference_indices=np.asarray([[positions[r] for r in p["fit_reference_rows"]] for p in pools], dtype=np.int64),
        context_indices=np.asarray([[positions[r] for r in p["context_rows"]] for p in pools], dtype=np.int64))
    planned = sum(a.nbytes for a in arrays.values()) + (len(rows) + 2 * len(pools)) * len(plan["output_genes"]) * 4
    require(planned + 65536 <= MAX_ARCHIVE, "Planned deduplicated archive exceeds bound")
    arrays["expression"] = np.empty((len(rows), len(plan["output_genes"])), dtype=np.float32)
    arrays["context"] = np.empty((len(pools), 2 * len(plan["output_genes"])), dtype=np.float32)
    require(set(arrays) == KEYS, "Internal auxiliary array schema mismatch")
    return arrays


def summarize_context(arrays):
    for index, rows in enumerate(arrays["context_indices"]):
        values = arrays["expression"][rows].astype(np.float64)
        arrays["context"][index] = np.concatenate((values.mean(0), values.std(0))).astype(np.float32)


def validate_cache(arrays, source, plan, contract_sha):
    require(set(arrays) == KEYS, "Unexpected auxiliary cache members")
    expected = empty_cache(source, plan, contract_sha)
    for key, array in arrays.items():
        require(isinstance(array, np.ndarray) and array.shape == expected[key].shape
            and array.dtype == expected[key].dtype, "Auxiliary cache dtype/shape mismatch: " + key)
        if key not in {"expression", "context"}:
            require(np.array_equal(array, expected[key]), "Auxiliary source/role/axis mapping changed: " + key)
    expression = arrays["expression"]
    require(np.isfinite(expression).all() and np.all(expression >= 0)
        and np.all(expression <= np.float32(np.log1p(10000))), "Invalid auxiliary normalized expression")
    expected["expression"] = expression
    summarize_context(expected)
    require(np.array_equal(arrays["context"], expected["context"]), "Context does not replay from independent context32")
    require(sum(a.nbytes for a in arrays.values()) + 65536 <= MAX_ARCHIVE, "Expanded auxiliary array limit")
    return arrays


def _serialize(path, arrays):
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    payload = buffer.getvalue()
    require(len(payload) <= MAX_ARCHIVE, "Compressed auxiliary archive limit")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        expanded = sum(m.file_size for m in archive.infolist())
    require(expanded <= MAX_ARCHIVE, "Expanded auxiliary archive with headers exceeds limit")
    bounded.write_new(path, payload)
    sha = hashlib.sha256(payload).hexdigest()
    require(safe.digest(path) == sha, "Written auxiliary cache changed")
    return {"path": str(path), "sha256": sha, "cache_bytes": len(payload), "expanded_archive_bytes": expanded,
        "array_bytes": sum(a.nbytes for a in arrays.values())}


def _receipt_base(contract_sha, contract, output):
    return {"schema": RECEIPT_SCHEMA, "completed": True, "contract_sha256": contract_sha,
        "candidate_plan_sha256": contract["candidate_plan"]["sha256"], "output_dir": str(output),
        "stage": "data_preparation", "representation": REPRESENTATION, "capacity": contract["capacity"],
        "row_chunk_limit": CHUNK_ROWS, "source_ids": [s["id"] for s in contract["sources"]],
        "raw_sources_authenticated_before_expression": True, "candidate_metadata_reconstructed_before_expression": True,
        "deduplicated_physical_rows": True, "training_authorized": False, "model_fitting_performed": False,
        "posttraining_performed": False, "submission_performed": False, "download_performed": False,
        "cloud_synced": False, "gpu_used": False, "rpe1_h1_hepg2_challenge2026_feng_opened": False}


def materialize(contract_path, contract_sha, output_dir):
    runtime_preflight()
    started = time.monotonic()
    out = canonical_path(output_dir)
    require(not os.path.lexists(out), "Fresh auxiliary cache output directory required")
    contract, plan = validate_contract(contract_path, contract_sha)
    bound_plan = {**plan, "shared_genes": contract["shared_genes"], "output_genes": contract["output_genes"],
        "_binding_sha256": contract["candidate_plan"]["sha256"]}
    caches, opened, metas, extractions = {}, [], {}, []
    with ExitStack() as stack:
        # Do not move any expression/pointer read above completion of this loop.
        for source in plan["sources"]:
            spec = source["acquisition"]
            stream = stack.enter_context(safe.regular_reader(spec["path"]))
            before = bounded.signature(stream)
            require(before[2] == spec["size_bytes"] and old.sha_stream(stream) == spec["sha256"],
                "Source authentication failed BEFORE any auxiliary X access")
            require(bounded.signature(stream) == before, "Source changed during complete hashing")
            handle = stack.enter_context(h5py.File(stream, "r"))
            meta = old.metadata(handle, spec)
            require(old.metadata_digest(meta) == source["metadata_sha256"], "Authenticated source metadata changed")
            opened.append((source, stream, handle, before))
            metas[source["id"]] = meta
            checkpoint(started)
            print(old.canonical({"stage": "auxiliary_source_authenticated", "source": source["id"]}).decode(), end="", flush=True)
        candidate.validate_metadata(plan, metas)
        for _, stream, _, before in opened:
            require(bounded.signature(stream) == before, "Source changed before auxiliary extraction")
        for source, stream, handle, before in opened:
            meta = metas[source["id"]]
            arrays = empty_cache(source, bound_plan, contract_sha)
            rows = arrays["original_rows"].tolist()
            counts, lookup = Counter(meta["genes"]), {g: i for i, g in enumerate(meta["genes"])}
            require(all(counts[g] == 1 for g in bound_plan["shared_genes"]), "Nonunique shared measured gene")
            shared_columns = [lookup[g] for g in bound_plan["shared_genes"]]
            output_indices = [bound_plan["shared_genes"].index(g) for g in bound_plan["output_genes"]]
            # Chunk externally too: checkpoint the registered wall time/RSS every <=256 rows.
            for offset in range(0, len(rows), CHUNK_ROWS):
                part = rows[offset:offset + CHUNK_ROWS]
                arrays["expression"][offset:offset + len(part)] = bounded.read_normalized_output_rows(
                    handle, part, authorized_rows=rows, n_obs=len(meta["cells"]), n_vars=len(meta["genes"]),
                    shared_columns=shared_columns, output_indices=output_indices, chunk_rows=CHUNK_ROWS)
                checkpoint(started)
                require(bounded.signature(stream) == before, "Source changed during auxiliary extraction")
            summarize_context(arrays)
            validate_cache(arrays, source, bound_plan, contract_sha)
            caches[source["id"]] = arrays
            extractions.append({"source": source["id"], "unique_rows": len(rows),
                "row_chunks": math.ceil(len(rows) / CHUNK_ROWS)})
            print(old.canonical({"stage": "auxiliary_source_materialized", **extractions[-1]}).decode(), end="", flush=True)
        for _, stream, _, before in opened:
            require(bounded.signature(stream) == before, "Source changed before auxiliary finalization")
    equal(validate_contract(contract_path, contract_sha)[0], contract, "Contract changed during auxiliary materialization")
    checkpoint(started)
    out.mkdir(mode=0o700)
    artifacts = {}
    for source in contract["sources"]:
        artifacts[source["id"]] = _serialize(out / source["cache_filename"], caches[source["id"]])
        checkpoint(started)
    require(sum(a["cache_bytes"] for a in artifacts.values()) + (16 << 20) <= MAX_DISK,
        "Auxiliary materialization output disk bound exceeded")
    result = {**_receipt_base(contract_sha, contract, out), "completed_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(), **checkpoint(started), "artifacts": artifacts, "source_extraction": extractions}
    candidate._dump(out / "complete.json", result)
    return result


def load_caches(contract_path, contract_sha, output_dir, completion_sha):
    """Replay every cache role/axis/value bound under an explicitly pinned receipt."""
    runtime_preflight()
    require(isinstance(completion_sha, str) and safe.HASH.fullmatch(completion_sha),
        "invalid_sha256: explicit auxiliary completion digest required")
    started = time.monotonic()
    contract, plan = validate_contract(contract_path, contract_sha)
    bound_plan = {**plan, "shared_genes": contract["shared_genes"], "output_genes": contract["output_genes"],
        "_binding_sha256": contract["candidate_plan"]["sha256"]}
    out = canonical_path(output_dir)
    receipt = safe.read_json(out / "complete.json", completion_sha)
    base = _receipt_base(contract_sha, contract, out)
    extra = {"completed_at", "hostname", "peak_rss_bytes", "elapsed_seconds", "artifacts", "source_extraction"}
    require(set(receipt) == set(base) | extra, "Unexpected auxiliary completion receipt fields")
    for key, value in base.items():
        equal(receipt[key], value, "Auxiliary completion receipt binding changed: " + key)
    safe.timestamp(receipt["completed_at"])
    require(datetime.fromisoformat(contract["created_at"]) <= datetime.fromisoformat(receipt["completed_at"])
        <= datetime.now(timezone.utc) + timedelta(seconds=300), "Invalid auxiliary completion timestamp")
    require(isinstance(receipt["hostname"], str) and receipt["hostname"].split(".")[0]
        in {"cbsuvlaminck3", "cbsuvlaminck6"}, "Unapproved auxiliary preparation host")
    require(type(receipt["peak_rss_bytes"]) is int and 0 < receipt["peak_rss_bytes"] <= MAX_RSS,
        "Auxiliary receipt exceeded registered RSS")
    require(type(receipt["elapsed_seconds"]) in (int, float) and math.isfinite(receipt["elapsed_seconds"])
        and 0 <= receipt["elapsed_seconds"] <= MAX_SECONDS, "Auxiliary receipt exceeded registered time")
    require(set(receipt["artifacts"]) == set(receipt["source_ids"]), "Auxiliary source artifact roster changed")
    equal(sorted(p.name for p in out.iterdir()), sorted(["complete.json"] + [s["cache_filename"] for s in contract["sources"]]),
        "Auxiliary output directory roster changed")
    # Authenticate every member before decoding any archive.
    total = 0
    for source in contract["sources"]:
        artifact = receipt["artifacts"][source["id"]]
        require(set(artifact) == {"path", "sha256", "cache_bytes", "expanded_archive_bytes", "array_bytes"},
            "Unexpected auxiliary archive binding fields")
        require(artifact["path"] == str(out / source["cache_filename"]), "Auxiliary archive path changed")
        require(all(type(artifact[k]) is int and 0 < artifact[k] <= MAX_ARCHIVE
            for k in ("cache_bytes", "expanded_archive_bytes", "array_bytes")), "Invalid archive size binding")
        with safe.regular_reader(artifact["path"]) as stream:
            require(os.fstat(stream.fileno()).st_size == artifact["cache_bytes"], "Auxiliary archive bytes changed")
        require(safe.digest(artifact["path"]) == artifact["sha256"], "Auxiliary archive digest mismatch")
        total += artifact["cache_bytes"]
    require(total + (16 << 20) <= MAX_DISK, "Auxiliary completion disk bound")
    caches, extractions = {}, []
    for source in plan["sources"]:
        artifact = receipt["artifacts"][source["id"]]
        arrays, details = bounded.read_npz(artifact["path"], artifact["sha256"], KEYS, return_metadata=True)
        equal(details, {k: artifact[k] for k in details}, "Auxiliary archive allocation binding mismatch")
        validate_cache(arrays, source, bound_plan, contract_sha)
        for values in arrays.values():
            values.setflags(write=False)
        caches[source["id"]] = arrays
        extractions.append({"source": source["id"], "unique_rows": len(arrays["original_rows"]),
            "row_chunks": math.ceil(len(arrays["original_rows"]) / CHUNK_ROWS)})
        checkpoint(started)
    equal(receipt["source_extraction"], extractions, "Auxiliary original-row/chunk conservation changed")
    require(safe.digest(out / "complete.json") == completion_sha, "Completion changed during cache loading")
    return caches, contract


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "materialize", "verify"))
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--plan-sha256")
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--contract-sha256")
    parser.add_argument("--completion-sha256")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.action == "register":
        if not all((args.plan, args.plan_sha256, args.protocol)):
            parser.error("Registration requires --plan, --plan-sha256 and --protocol")
        result = {"contract_sha256": register(args.plan, args.plan_sha256, args.protocol, args.output_dir)}
    else:
        if not all((args.contract, args.contract_sha256)):
            parser.error("Materialization/verification requires --contract and --contract-sha256")
        if args.action == "materialize":
            result = materialize(args.contract, args.contract_sha256, args.output_dir)
        else:
            if not args.completion_sha256:
                parser.error("Verification requires explicit --completion-sha256")
            caches, _ = load_caches(args.contract, args.contract_sha256, args.output_dir, args.completion_sha256)
            result = {"verified": True, "unique_rows": {s: len(a["expression"]) for s, a in caches.items()},
                "training_authorized": False, "model_fitting_performed": False}
    print(old.canonical(result).decode(), end="", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
