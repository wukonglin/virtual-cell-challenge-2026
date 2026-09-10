"""Independent five-group, native-control anchoring audit; no generation or upload."""
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import socket
import time

import anndata
import h5py
import numpy as np
from scipy import sparse

import public_training_readiness as safe
import public_auxiliary_vcc_writer_v10 as writer
import package_public_auxiliary_vcc_v10 as package

SHA = "2d8c218e99a42b65481424c88d431ed093c60575c171f2b9e539dc1a12f9cd81"
CONTROLS = {
    "A":"f22e71968487d3da9402769dcd0c73a47851fad4cd7dd25510283c923031fe3e",
    "B":"41557555b3febf6db19fd930b4dd7db88941840077bf985d14dfff959c782380",
    "C":"e090e7c7cfdd0006f689214ba38ba0cd2669578e89c9bc325b65f0bd940c02e0",
}
ORDINALS = (0, 299, 300, 599, 600)
require = safe.require


def compare(raw, prediction, modeled):
    require(sparse.isspmatrix_csr(raw) and sparse.isspmatrix_csr(prediction)
        and raw.shape == prediction.shape == (400,18533), "Exact400 by18533 CSR blocks required")
    for matrix in (raw,prediction):
        require(np.isfinite(matrix.data).all() and (matrix.data >= 0).all()
            and np.equal(matrix.data,np.floor(matrix.data)).all(), "Finite nonnegative integer counts required")
    modeled = np.asarray(modeled,dtype=np.int64)
    require(modeled.shape == (482,) and len(np.unique(modeled)) == 482, "Exact482 modeled gene indices required")
    unmodeled = np.flatnonzero(~np.isin(np.arange(18533),modeled))
    require(len(unmodeled) == 18051, "Exact18051 unmodeled genes required")
    raw, prediction = raw.astype(np.int64), prediction.astype(np.int64)
    difference = prediction[:,unmodeled]-raw[:,unmodeled]
    difference.eliminate_zeros()
    require(difference.nnz == 0, "Unmodeled native control counts changed")
    modeled_difference = prediction[:,modeled]-raw[:,modeled]
    modeled_difference.eliminate_zeros()
    before = np.asarray(raw.sum(axis=1)).ravel()
    after = np.asarray(prediction.sum(axis=1)).ravel()
    require((before > 0).all() and (after > 0).all() and (after <= 1000000).all(), "Valid bounded cell libraries required")
    drift = after-before
    relative = drift.astype(np.float64)/before
    return {"cells":400,"genes":18533,"modeled_genes":482,"unmodeled_genes":18051,
        "unmodeled_changed_entries":0,"unmodeled_counts_exactly_preserved":True,
        "modeled_changed_entries":int(modeled_difference.nnz),
        "modeled_genes_changed":int(len(np.unique(modeled_difference.indices))),
        "cells_with_modeled_changes":int(np.count_nonzero(np.diff(modeled_difference.indptr))),
        "nonnegative_integer_counts":True,"negative_entries":0,"fractional_entries":0,
        "control_library_min":int(before.min()),"control_library_max":int(before.max()),
        "prediction_library_min":int(after.min()),"prediction_library_max":int(after.max()),
        "library_drift_min":int(drift.min()),"library_drift_max":int(drift.max()),
        "library_drift_mean":float(drift.mean()),"library_drift_total":int(drift.sum()),
        "library_relative_drift_min":float(relative.min()),"library_relative_drift_max":float(relative.max()),
        "library_relative_drift_mean":float(relative.mean())}


def audit():
    started = time.monotonic()
    require(socket.gethostname().split(".")[0] == "cbsuvlaminck3", "Approved compute host required")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU-only audit required")
    for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        require(os.environ.get(name) == "2", "Two numerical threads required")
    os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:2]))
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS,((4 << 30) if hard == -1 else min(4 << 30,hard),hard))
    base = package.BASE
    registration = base / "inference_registration/inference_contract.json"
    inference = safe.read_json(registration,SHA)
    require(inference.get("schema") == "public-auxiliary-vcc-inference-v10", "Sealed inference schema required")
    for name,sha in inference["code_sha256"].items():
        package.authenticate(package.ROOT / "scripts" / name,sha)
    genes,targets = package.metadata()
    require(inference["genes"] == genes and inference["targets"] == targets, "Exact official axes required")
    positions = {gene:i for i,gene in enumerate(genes)}
    modeled = [positions[gene] for gene in inference["modeled_genes"]]
    folder = base / "inference/counts"
    journal_path = folder / "groups.jsonl"
    prefix = []
    records = {}
    with safe.regular_reader(journal_path) as stream:
        for ordinal in range(max(ORDINALS)+1):
            line = stream.readline(1 << 16)
            require(line.endswith(b"\n") and len(line) < (1 << 16), "Complete bounded published journal prefix required")
            prefix.append(line)
            if ordinal in ORDINALS:
                row = json.loads(line)
                require(row["ordinal"] == ordinal, "Expected journal ordinal required")
                context, target = "ABC"[ordinal//300],targets[ordinal%300]
                require(row["keys"] == [[context,target]]
                    and row["path"] == str(folder / "groups" / f"g{ordinal:06d}.h5ad"), "Expected immutable group required")
                records[ordinal] = row
    prefix = b"".join(prefix)
    writer_meta = safe.read_json(folder / "writer.json")
    require(writer_meta["generation_contract_sha256"] == SHA and writer_meta["genes"] == genes
        and writer_meta["targets"] == targets, "Matching writer registration required")
    reports = []
    with ExitStack() as stack:
        sources,groups = {},{}
        # Authenticate all3 official native control files before ANY X decode.
        for context,sha in CONTROLS.items():
            binding = inference["controls"][context]
            path = package.ROOT / "dataset/controls" / f"context_{context}.h5ad"
            require(binding["path"] == str(path) and binding["sha256"] == sha, "Exact registered control source required")
            stream = stack.enter_context(safe.regular_reader(path))
            sig = package.signature(stream)
            require(hashlib.file_digest(stream,"sha256").hexdigest() == sha
                and package.signature(stream) == sig, "Full same-open control authentication failed")
            stream.seek(0)
            sources[context] = (stream,sig)
        # Also authenticate every selected saved output before expression access.
        for ordinal,row in records.items():
            stream = stack.enter_context(safe.regular_reader(row["path"]))
            sig = package.signature(stream)
            require(hashlib.file_digest(stream,"sha256").hexdigest() == row["sha256"]
                and package.signature(stream) == sig, "Published group hash differs")
            stream.seek(0)
            groups[ordinal] = (stream,sig)
        for stream,sig in [*sources.values(),*groups.values()]:
            require(package.signature(stream) == sig,"Source changed before first expression read")
        for context,(stream,sig) in sources.items():
            with h5py.File(stream,"r") as handle:
                writer.read_control_metadata(handle,genes,context)
                rows = inference["controls"][context]["base_rows"]
                require(len(rows) == len(set(rows)) == 400 and all(type(i) is int and 0 <= i < 18400 for i in rows),
                    "Exact registered400 control donors required")
                raw = anndata.io.sparse_dataset(handle["X"])[rows]
                require(package.signature(stream) == sig,"Control changed during selected donor read")
            for ordinal in ORDINALS:
                if "ABC"[ordinal//300] != context:
                    continue
                output_stream,output_sig = groups[ordinal]
                target = targets[ordinal%300]
                with h5py.File(output_stream,"r") as handle:
                    obs,var = anndata.io.read_elem(handle["obs"]),anndata.io.read_elem(handle["var"])
                    require(var.index.tolist() == genes and obs.index.tolist() == [f"{context}|{target}|{i:03d}" for i in range(400)]
                        and set(obs["context"].astype(str)) == {context}
                        and set(obs["target_gene"].astype(str)) == {target}, "Exact saved-group identities and gene axis required")
                    prediction = anndata.io.read_elem(handle["X"])
                    qc = writer.validate_block(prediction,18533,400,writer.WriterConfig())
                    require(qc["nnz"] == records[ordinal]["nnz"], "Published group nnz differs")
                    summary = compare(raw,prediction,modeled)
                require(package.signature(output_stream) == output_sig,"Output group changed during audit")
                reports.append({"ordinal":ordinal,"context":context,"target":target,
                    "group":{"path":records[ordinal]["path"],"sha256":records[ordinal]["sha256"]},
                    "base_rows":rows,"comparison":summary})
        for stream,sig in [*sources.values(),*groups.values()]:
            require(package.signature(stream) == sig,"Authenticated source changed during audit")
    # The journal may append concurrently; its already-published prefix must not change.
    with safe.regular_reader(journal_path) as stream:
        require(stream.read(len(prefix)) == prefix,"Published journal prefix changed")
    require(safe.read_json(registration,SHA) == inference,"Sealed inference changed during audit")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    require(peak <= 4 << 30 and time.monotonic()-started <= 600,"Bounded audit resource limit")
    out = base / "partial_control_anchor_audit_20260910"
    require(not os.path.lexists(out),"Fresh audit receipt directory required")
    out.mkdir(mode=0o700)
    receipt = {"schema":"public-auxiliary-native-control-anchor-audit-v10","completed":True,
        "completed_at":datetime.now(timezone.utc).isoformat(),"inference_contract":{"path":str(registration),"sha256":SHA},
        "source_control_sha256":CONTROLS,"journal_prefix_sha256":hashlib.sha256(prefix).hexdigest(),
        "journal_prefix_lines":601,"journal_prefix_bytes":len(prefix),"audited_groups":reports,
        "total_audited_predictions":2000,"registered_native_control_rows_read":1200,
        "all_unmodeled_counts_exactly_preserved":True,"all_counts_nonnegative_integer":True,
        "source_hashes_verified_before_any_expression":True,"challenge_treated_expression_read":False,
        "synthetic_counts_read":False,"scientific_improvement_established":False,
        "score_evaluation_performed":False,"submission_performed":False,"partial_output_audit":True,
        "code_sha256":{Path(__file__).name:safe.digest(Path(__file__))},
        "resources":{"cpu_affinity":sorted(os.sched_getaffinity(0)),"peak_rss_bytes":peak,
            "elapsed_seconds":time.monotonic()-started}}
    package.dump_new(out / "complete.json",receipt)
    print(json.dumps({"receipt":str(out/"complete.json"),"sha256":safe.digest(out/"complete.json"),
        "groups":[{"ordinal":r["ordinal"],"context":r["context"],"target":r["target"],**r["comparison"]} for r in reports],
        "resources":receipt["resources"]},sort_keys=True),flush=True)


if __name__ == "__main__":
    audit()
