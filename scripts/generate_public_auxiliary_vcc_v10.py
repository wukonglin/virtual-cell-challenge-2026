"""Exploratory VCC inference from frozen, public-only GO/context flow parents.

No training, challenge-treated reads, downloads, credential access or upload.
Register first; execute only the registered control rows and fixed ensemble.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import csv
import hashlib
import importlib.metadata
import io
import os
from pathlib import Path
import resource
import shutil
import time

import anndata
import h5py
import numpy as np
import torch

import build_go_target_features as go
import prepare_public_joint_target_source_v3 as go_source
import public_auxiliary_cache_v10 as cache
import public_auxiliary_count_adapter_v10 as adapter
import public_auxiliary_flow_v10 as flow
import public_auxiliary_vcc_writer_v10 as writer
import public_training_readiness as safe
import run_public_auxiliary_flow_v10 as training

SCHEMA = "public-auxiliary-vcc-inference-v10"
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/public_flow/auxiliary_v10_20260910"
TRAINING_SHA = "590a5916c6a9c76e263f2c8e4e45ed9ba41def7682a6694ce82d87b1b7074f2b"
COMPLETION_SHA = "78e0e7d8e0fd443f6ba6ca3b4d3d02aeef42e166b50908aa28a3e1100a2b9a67"
CONTROL_SHAS = {
    "A": "f22e71968487d3da9402769dcd0c73a47851fad4cd7dd25510283c923031fe3e",
    "B": "41557555b3febf6db19fd930b4dd7db88941840077bf985d14dfff959c782380",
    "C": "e090e7c7cfdd0006f689214ba38ba0cd2669578e89c9bc325b65f0bd940c02e0",
}
MANIFEST_SHA = "2c2b5d425df817ca35e9edf08282d12494ce42913ba824375bf0f98a1cc676f5"
TARGETS_SHA = "f57edd7b912ebd718efc7ee9d0f334772513e7cc418d133ce525470e373b3276"
SEED = 20260927
require, equal, dump = cache.require, cache.equal, training.dump


def binding(path):
    path = cache.canonical_path(path)
    return {"path": str(path), "sha256": safe.digest(path)}


def codes():
    names = ("generate_public_auxiliary_vcc_v10.py", "public_auxiliary_count_adapter_v10.py",
             "public_auxiliary_vcc_writer_v10.py", *training.MODULES)
    return {name: safe.digest(Path(__file__).with_name(name)) for name in dict.fromkeys(names)}


def policy():
    return {"candidate": "GO_context_flow_v10_exploratory", "stage": "inference",
        "ensemble": {source: 0.5 for source in flow.SOURCES}, "arm": "true", "heun_steps": 16,
        "strength": 1.0, "seed": SEED, "context_cells": 32, "base_cells": 400,
        "context_rule": "float64 mean/population std, then float32; same 482-gene encoding as training",
        "donor_rule": "SHA256 canonical [seed, namespace, context, cell_id]; first32 context, next400 base",
        "base_reuse": "same 400 native control cells for every target within a context",
        "target_features": "same fit-only GO vocabularies; binary L2, GO availability, GenePT absent flag",
        "count_rule": "inverse logCP using original 7097-gene control depth; seeded unbiased rounding",
        "unmodeled_genes": "all 18051 unmodeled native counts preserved exactly",
        "library_conservation": False, "clipping_or_thinning": False,
        "challenge_treated_used": False, "genept_used": False, "promoterai_used": False,
        "posttraining_performed": False, "scientific_improvement_established": False,
        "exploratory_submission_authorized_by_user": True, "upload_performed_by_generator": False,
        "cpu_threads": 2, "gpu_uuid": training.GPU_UUID, "gpu_visible": training.GPU_VISIBLE,
        "gpu_allocated_limit_bytes": 6 << 30, "rss_limit_bytes": 16 << 30,
        "output_limit_bytes": 100 << 30, "wall_time_seconds": 7200,
        "numpy_version": np.__version__, "torch_version": torch.__version__,
        "anndata_version": importlib.metadata.version("anndata")}


def metadata():
    genes = cache.official_gene_axis()
    path = ROOT / "dataset/controls/pert_counts.csv"
    require(safe.digest(path) == TARGETS_SHA, "Official target metadata differs")
    with safe.regular_reader(path) as stream:
        rows = list(csv.reader(io.StringIO(stream.read(1 << 20).decode(), newline="")))
    require(rows[0] == ["target_gene"] and all(len(row) == 1 for row in rows[1:]), "Exact target CSV required")
    targets = [row[0] for row in rows[1:]]
    require(len(targets) == 300 and len(set(targets)) == 300
        and all(t and t == t.strip() for t in targets), "Exact 300 unique targets required")
    manifest = safe.read_json(ROOT / "dataset/controls/manifest.json", MANIFEST_SHA)
    for key, value in {"season":"2026", "partition":"val", "panel_id":"vcc2026-val-1",
        "contexts":["A","B","C"], "pert_col":"target_gene", "context_col":"context",
        "control_label":"non-targeting", "n_genes":18533, "n_constructs":300,
        "cells_per_pert":400}.items():
        equal(manifest.get(key), value, "Official manifest " + key)
    return genes, targets


def donor_roles(ids, context, *, context_count=32, base_count=400):
    require(context in CONTROL_SHAS and len(ids) == len(set(ids))
        and all(isinstance(x, str) and x for x in ids), "Unique official control identities required")
    require(type(context_count) is int and context_count > 1 and type(base_count) is int
        and base_count > 0 and context_count+base_count <= len(ids), "Invalid donor counts")
    order = sorted(range(len(ids)), key=lambda i: (
        hashlib.sha256(cache.old.canonical([SEED,"vcc-flow-v10",context,ids[i]])).digest(), ids[i]))
    return {"context_rows": order[:context_count], "base_rows": order[context_count:context_count+base_count]}


def control_metadata(handle, genes, context):
    """Annotation/layout only; never decode X values here."""
    writer.read_control_metadata(handle, genes, context)  # rejects external/soft links before decoding
    obs = anndata.io.read_elem(handle["obs"])
    var = anndata.io.read_elem(handle["var"])
    require(len(obs) == 18400 and var.index.tolist() == genes, "Official control axes differ")
    require(obs.index.is_unique and not obs.index.hasnans, "Unique complete control IDs required")
    require(set(obs.columns) == {"target_gene","context","ntc_id"}, "Unexpected control metadata")
    require(not obs.isna().any().any() and set(obs["target_gene"].astype(str)) == {"non-targeting"}
        and set(obs["context"].astype(str)) == {context}, "Controls must be non-targeting and context matched")
    counts = obs["ntc_id"].value_counts()
    require(len(counts) == 46 and (counts == 400).all(), "Official control guide quotas differ")
    x = handle["X"]
    require(x.attrs.get("encoding-type") == "csr_matrix" and tuple(x.attrs["shape"]) == (18400,18533)
        and set(x) == {"data","indices","indptr"}, "Official CSR layout differs")
    ids = obs.index.astype(str).tolist()
    return {"obs_ids_sha256": hashlib.sha256(cache.old.canonical(ids)).hexdigest(),
            **donor_roles(ids, context)}


def official_features(record, original_features, targets):
    """Transform new public target annotations without expanding fitted vocabularies."""
    index = go_source._go_index(ROOT / go_source.GO_RELATIVE)
    result, coverage = {}, {}
    for source in flow.SOURCES:
        roles = record["sources"][source]
        original = sorted(roles["fit_targets"] + roles["tune_targets"])
        union = sorted(set(original) | set(targets))
        built = go.build_features(index, target_ids=union, train_target_ids=roles["fit_targets"],
            fold_id="auxiliary-v10-"+source, sources=go_source.GO_EXPECTED)
        frozen = cache.bounded.read_npz(record["features"][source]["path"],
            record["features"][source]["sha256"], training.GO_KEYS)
        require(np.array_equal(built.arrays["go_ids"], frozen["go_ids"]), "Inference expanded the fitted GO vocabulary")
        positions = {name:i for i,name in enumerate(built.arrays["target_ids"].tolist())}
        values = built.arrays["features"].astype(np.float32)
        values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1.)
        values = np.column_stack((values, built.arrays["present"].astype(np.float32),
                                 np.zeros(len(values), np.float32))).astype(np.float32)
        require(np.array_equal(values[[positions[t] for t in original]], original_features[source]),
                "Original fitted target features must replay exactly")
        result[source] = values[[positions[t] for t in targets]].copy()
        coverage[source] = {"targets":len(targets), "go_present":int(result[source][:,-2].sum()),
            "go_terms":len(built.arrays["go_ids"]), "genept_present":0,
            "features_sha256":hashlib.sha256(result[source].tobytes()).hexdigest()}
    return result, coverage


def parents():
    path = BASE / "training_registration/training_contract.json"
    record, original = training.validate(path, TRAINING_SHA)
    complete = safe.read_json(BASE / "pretraining/complete.json", COMPLETION_SHA)
    require(complete.get("completed") is True and complete.get("training_contract_sha256") == TRAINING_SHA
        and complete.get("optimizer_steps") == 3600 and complete.get("tracking_events") == 3606,
        "Completed public pretraining required")
    require(len(complete["runs"]) == 6 and {(r["source"],r["arm"]) for r in complete["runs"]}
        == {(s,a) for s in flow.SOURCES for a in training.ARMS}, "Exact trained arm roster required")
    checkpoints, metrics = {}, {}
    for item in complete["runs"]:
        summary = safe.read_json(item["summary_path"], item["summary_sha256"])
        require(summary["training_contract_sha256"] == TRAINING_SHA
            and summary["saved_checkpoint_replayed"] is True, "Parent checkpoint was not replayed")
        if item["arm"] == "true":
            checkpoints[item["source"]] = summary["checkpoint"]
            metrics[item["source"]] = item["evaluation"]
    materialization, _ = cache.validate_contract(record["materialization"]["path"], record["materialization"]["sha256"])
    return record, original, checkpoints, metrics, materialization


def build_record(created_at):
    """Read metadata and hashes only. No expression or checkpoint tensor decoding."""
    safe.timestamp(created_at)
    require(safe.timestamp(created_at).timestamp() <= time.time()+300, "Future registration timestamp")
    record, original, checkpoints, metrics, materialization = parents()
    genes, targets = metadata()
    mapping = adapter.build_gene_mapping(genes, materialization["output_genes"], materialization["shared_genes"])
    require(not set(targets) & {t for r in record["sources"].values() for t in r["fit_targets"]},
            "Official target treated responses must not occur in source fitting")
    features, coverage = official_features(record, original, targets)
    controls = {}
    for context, sha in CONTROL_SHAS.items():
        path = cache.canonical_path(ROOT / f"dataset/controls/context_{context}.h5ad")
        with safe.regular_reader(path) as stream:
            signature = cache.bounded.signature(stream)
            require(hashlib.file_digest(stream, "sha256").hexdigest() == sha, "Official control file hash differs")
            stream.seek(0)
            with h5py.File(stream, "r") as handle:
                roles = control_metadata(handle, genes, context)
            require(cache.bounded.signature(stream) == signature, "Control file changed during metadata registration")
        controls[context] = {"path":str(path), "sha256":sha, **roles}
    for checkpoint in checkpoints.values():
        meta = safe.binding(checkpoint)
        require(safe.digest(Path(checkpoint["path"]).with_name("checkpoint.npz")) == meta["checkpoint_sha256"],
                "Parent checkpoint hash differs")
    result = {"schema":SCHEMA, "created_at":created_at, "policy":policy(), "code_sha256":codes(),
        "training_contract":binding(BASE / "training_registration/training_contract.json"),
        "training_completion":binding(BASE / "pretraining/complete.json"), "checkpoints":checkpoints,
        "public_validation_metrics":metrics, "controls":controls, "genes":genes, "targets":targets,
        "modeled_genes":materialization["output_genes"], "normalization_genes":materialization["shared_genes"],
        "mapping":mapping.provenance(), "feature_coverage":coverage,
        "manifest_sha256":MANIFEST_SHA, "targets_sha256":TARGETS_SHA,
        "expression_read_during_registration":False}
    return result, features, mapping


def register(out):
    cache.runtime_preflight()
    out = cache.canonical_path(out)
    require(not os.path.lexists(out), "Fresh inference registration required")
    record, _, _ = build_record(datetime.now(timezone.utc).isoformat())
    out.mkdir(mode=0o700)
    dump(out / "inference_contract.json", record)
    sha = safe.digest(out / "inference_contract.json")
    dump(out / "complete.json", {"schema":SCHEMA, "registered":True, "inference_contract_sha256":sha})
    return sha


def validate(path, sha):
    path = cache.canonical_path(path)
    require(path.name == "inference_contract.json", "Canonical inference contract required")
    record = safe.read_json(path, sha)
    expected, features, mapping = build_record(record["created_at"])
    equal(record, expected, "Complete inference contract reconstruction")
    equal(safe.read_json(path.with_name("complete.json")),
        {"schema":SCHEMA, "registered":True, "inference_contract_sha256":sha}, "Inference registration receipt")
    return record, features, mapping


def context_summary(encoded):
    require(encoded.dtype == np.float32 and encoded.ndim == 2 and len(encoded) == 32
        and np.isfinite(encoded).all() and (encoded >= 0).all(), "Exactly32 encoded controls required")
    x = encoded.astype(np.float64)
    return np.concatenate((x.mean(0), x.std(0))).astype(np.float32)


def round_seed(context, target):
    return int.from_bytes(hashlib.sha256(cache.old.canonical([SEED,"count-rounding",context,target])).digest()[:8], "big")


def infer(models, control, features, context, *, device):
    require(set(models) == set(flow.SOURCES) == set(features), "Exact two-source inference ensemble")
    base = torch.as_tensor(control, dtype=torch.float32, device=device)
    pooled = torch.as_tensor(np.broadcast_to(context, (len(control),len(context))).copy(), device=device)
    prediction = torch.zeros_like(base)
    for source in flow.SOURCES:
        feature = torch.as_tensor(np.broadcast_to(features[source], (len(control),len(features[source]))).copy(), device=device)
        prediction += 0.5 * flow.sample_heun(models[source],base,feature,pooled,steps=16)
    return prediction.cpu().numpy()


def resources(out, started):
    elapsed = time.monotonic()-started
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    gpu = torch.cuda.max_memory_allocated(0)
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file() and not p.is_symlink())
    require(elapsed <= policy()["wall_time_seconds"] and rss <= policy()["rss_limit_bytes"]
        and gpu <= policy()["gpu_allocated_limit_bytes"] and size <= policy()["output_limit_bytes"],
        "Inference resource checkpoint exceeded")
    return {"elapsed_seconds":elapsed,"peak_rss_bytes":rss,"gpu_allocated_bytes":gpu,"output_bytes":size}


def run(path, sha, out):
    started = time.monotonic()
    torch.set_num_threads(2)
    out = cache.canonical_path(out)
    require(not os.path.lexists(out), "Fresh inference output required")
    with training.cpu_cache_phase():
        cache.runtime_preflight()
        record, features, mapping = validate(path,sha)
    require(os.environ.get("CUDA_VISIBLE_DEVICES") in {training.GPU_VISIBLE,training.GPU_UUID}
        and torch.cuda.is_available() and torch.cuda.device_count() == 1, "Registered actual CUDA GPU required")
    props = torch.cuda.get_device_properties(0)
    require("GPU-"+str(props.uuid).removeprefix("GPU-") == training.GPU_UUID
        and props.name == "NVIDIA GeForce RTX 2080 Ti", "GPU identity differs")
    torch.cuda.set_per_process_memory_fraction((6 << 30)/props.total_memory,0)
    require(shutil.disk_usage(out.parent).free >= 120 << 30, "At least120GiB free disk required")
    models = {}
    for source in flow.SOURCES:
        config = flow.PilotConfig(gene_dim=482,target_dim=features[source].shape[1])
        model, optimizer, _ = training.restore_checkpoint(record["checkpoints"][source],TRAINING_SHA,"cuda:0",config)
        models[source] = model
        del optimizer
    out.mkdir(mode=0o700)
    dump(out / "started.json", {"schema":SCHEMA,"inference_contract_sha256":sha,"pid":os.getpid(),
        "started_at":datetime.now(timezone.utc).isoformat(),"policy":policy()})
    limits = adapter.CountLimits(max_genes_per_cell=18533,max_density=1.0)
    sink = writer.SubmissionWriter(out / "counts",record["genes"],record["targets"],sha)
    reports = []
    with ExitStack() as stack:
        handles = {}
        # All three native controls authenticate before the first X value is read.
        for context, entry in record["controls"].items():
            stream = stack.enter_context(safe.regular_reader(entry["path"]))
            signature = cache.bounded.signature(stream)
            require(hashlib.file_digest(stream,"sha256").hexdigest() == entry["sha256"], "Control hash before expression differs")
            stream.seek(0)
            handle = stack.enter_context(h5py.File(stream,"r"))
            equal(control_metadata(handle,record["genes"],context),
                {k:v for k,v in entry.items() if k not in {"path","sha256"}}, "Control role reconstruction")
            handles[context] = (stream, signature, handle)
        for stream,signature,_ in handles.values():
            require(cache.bounded.signature(stream) == signature, "Control changed before first expression read")
        for context,(stream,signature,handle) in handles.items():
            roles = record["controls"][context]
            matrix = anndata.io.sparse_dataset(handle["X"])
            raw_context = matrix[roles["context_rows"]]
            raw_base = matrix[roles["base_rows"]]
            require(cache.bounded.signature(stream) == signature, "Control changed during selected row reads")
            encoded_context,_ = adapter.encode_controls(raw_context,mapping,control_genes=record["genes"],limits=limits)
            encoded,_ = adapter.encode_controls(raw_base,mapping,control_genes=record["genes"],limits=limits)
            pooled = context_summary(encoded_context)
            # Actual native-count identity check on all400 selected donors.
            identity, identity_qc = adapter.flow_to_counts(raw_base,encoded,mapping,control_genes=record["genes"],
                prediction_genes=record["modeled_genes"],seed=SEED,strength=0.0,limits=limits)
            require((identity != raw_base).nnz == 0, "Actual native-control identity replay failed")
            for number,target in enumerate(record["targets"]):
                prediction = infer(models,encoded,{s:features[s][number] for s in flow.SOURCES},pooled,device="cuda:0")
                counts,qc = adapter.flow_to_counts(raw_base,prediction,mapping,control_genes=record["genes"],
                    prediction_genes=record["modeled_genes"],seed=round_seed(context,target),strength=1.0,limits=limits)
                sink.append(context,target,counts)
                reports.append({"context":context,"target":target,"qc":qc})
                if (number+1) % 25 == 0:
                    print(cache.old.canonical({"context":context,"targets_completed":number+1,
                        "resources":resources(out,started)}).decode(),flush=True)
            require(cache.bounded.signature(stream) == signature, "Open native controls changed during inference")
            dump(out / ("context_"+context+"_complete.json"), {"context":context,"groups":300,
                "identity_qc":identity_qc,"context_rows":roles["context_rows"],"base_rows":roles["base_rows"]})
    receipt = sink.finalize()
    require(record["code_sha256"] == codes(), "Code changed during generation")
    dump(out / "count_adapter_qc.json", {"schema":SCHEMA,"groups":reports})
    dump(out / "complete.json", {"schema":SCHEMA,"completed":True,"inference_contract_sha256":sha,
        "completed_at":datetime.now(timezone.utc).isoformat(),"output":receipt,
        "count_adapter_qc":binding(out / "count_adapter_qc.json"),"resources":resources(out,started),
        "official_cli_validation_performed":False,"submission_performed":False,
        "scientific_improvement_established":False,"posttraining_performed":False})
    return safe.digest(out / "complete.json")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action",required=True)
    r = sub.add_parser("register"); r.add_argument("--out",required=True)
    r = sub.add_parser("run")
    for key in ("contract","sha","out"): r.add_argument("--"+key,required=True)
    args = p.parse_args()
    print(register(args.out) if args.action == "register" else run(args.contract,args.sha,args.out),flush=True)


if __name__ == "__main__":
    main()
