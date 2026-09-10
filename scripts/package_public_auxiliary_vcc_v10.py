"""Authenticate one frozen exploratory prediction and run official VCC prep once.

Run with .venv-vcc/bin/python on approved BioHPC compute, never a login node.
No training, challenge-source reads, downloads, credentials or submission.
Only the new private output directory is written; original counts are retained.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import csv
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import re
import resource
import shutil
import signal
import socket
import stat
import sys
import tarfile
import tempfile
import time

import public_training_readiness as safe

SCHEMA = "public-auxiliary-vcc-package-v10"
INFERENCE_SCHEMA = "public-auxiliary-vcc-inference-v10"
WRITER_SCHEMA = "public-auxiliary-vcc-count-writer-v10"
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/public_flow/auxiliary_v10_20260910"
TRAINING_SHA = "590a5916c6a9c76e263f2c8e4e45ed9ba41def7682a6694ce82d87b1b7074f2b"
TRAINING_COMPLETION_SHA = "78e0e7d8e0fd443f6ba6ca3b4d3d02aeef42e166b50908aa28a3e1100a2b9a67"
WRITER_CODE_SHA = "e3d6b47028a5549157c86f86f6737f7b33502b6ba3e534df9939fc282ba5c485"
OFFICIAL_VERSION = "0.2.0"
OFFICIAL_MODULES = {
    "__init__.py": "a516e9051e39d96a2635265373602d0730ce05f1ac2bec5641880657e71a101d",
    "prep.py": "bd99d6972936aadec4d0da403af36917da1f30018ae227a80a6e34072a96cb55",
    "sizing.py": "0a5bab4f559b086b093ba004066bb071a869fdfa46b6af0719349fea0e773fe4",
    "vccfile.py": "84a3c3e2468df53758e3054edc3b601b5fbb13f73d8748e0693c1caad4773a0b",
}
METADATA_SHAS = {
    "gene_names.csv": "25bfa66715e186bebabce7ac788bbcea47e2bf59ca70be1f8f3a06f2f0e47201",
    "pert_counts.csv": "f57edd7b912ebd718efc7ee9d0f334772513e7cc418d133ce525470e373b3276",
    "manifest.json": "2c2b5d425df817ca35e9edf08282d12494ce42913ba824375bf0f98a1cc676f5",
}
RAM_LIMIT = 96 << 30
OUTPUT_LIMIT = 128 << 30
MIN_FREE_DISK = 120 << 30
WALL_SECONDS = 21600
require = safe.require


def canonical_path(path):
    path = Path(path)
    require(path.is_absolute() and not path.is_symlink(), "Absolute non-symlink path required")
    return path.resolve(strict=False)


def signature(stream):
    value = os.fstat(stream.fileno())
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def authenticate(path, sha):
    path = canonical_path(path)
    require(isinstance(sha, str) and safe.HASH.fullmatch(sha), "Valid SHA256 required")
    with safe.regular_reader(path) as stream:
        before = signature(stream)
        value = hashlib.file_digest(stream, "sha256").hexdigest()
        require(signature(stream) == before and value == sha, "File hash or open identity differs")
    return {"path": str(path), "sha256": sha, "bytes": before[2]}


def dump_new(path, value):
    payload = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    require(len(payload) <= safe.MAX_JSON_BYTES, "Bounded packaging receipt required")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    with os.fdopen(os.open(path, flags, 0o600), "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _bound_json(binding, path, sha):
    require(binding == {"path": str(canonical_path(path)), "sha256": sha}, "Fixed parent binding differs")
    return safe.read_json(path, sha)


def _csv(path, sha, header, count):
    authenticate(path, sha)
    with safe.regular_reader(path) as stream:
        rows = list(csv.reader(io.StringIO(stream.read(1 << 20).decode(), newline="")))
    require(rows and rows[0] == [header] and len(rows) == count + 1
        and all(len(row) == 1 for row in rows[1:]), "Exact official CSV dimensions required")
    names = [row[0] for row in rows[1:]]
    require(len(set(names)) == count and all(x and x == x.strip() for x in names), "Unique official names required")
    return names


def metadata():
    folder = ROOT / "dataset/controls"
    genes = _csv(folder / "gene_names.csv", METADATA_SHAS["gene_names.csv"], "gene_name", 18533)
    targets = _csv(folder / "pert_counts.csv", METADATA_SHAS["pert_counts.csv"], "target_gene", 300)
    manifest = safe.read_json(folder / "manifest.json", METADATA_SHAS["manifest.json"])
    expected = {"season": "2026", "partition": "val", "panel_id": "vcc2026-val-1",
        "contexts": ["A", "B", "C"], "pert_col": "target_gene", "context_col": "context",
        "control_label": "non-targeting", "n_genes": 18533, "n_constructs": 300, "cells_per_pert": 400}
    require(all(manifest.get(k) == v for k, v in expected.items()), "Official manifest differs")
    return genes, targets


def code_pins(inference, training):
    codes = inference.get("code_sha256")
    require(isinstance(codes, dict) and len(codes) >= 6, "Complete inference code closure required")
    required = {"generate_public_auxiliary_vcc_v10.py", "public_auxiliary_count_adapter_v10.py",
        "public_auxiliary_vcc_writer_v10.py", "run_public_auxiliary_flow_v10.py",
        "public_auxiliary_flow_v10.py", "public_auxiliary_cache_v10.py"}
    require(required <= codes.keys() and isinstance(training.get("code_sha256"), dict)
        and all(codes.get(name) == sha for name, sha in training["code_sha256"].items()),
        "Frozen training code closure differs")
    require(codes["public_auxiliary_vcc_writer_v10.py"] == WRITER_CODE_SHA, "Frozen count writer differs")
    for name, sha in codes.items():
        require(isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9_]+\.py", name), "Unsafe code pin name")
        authenticate(ROOT / "scripts" / name, sha)
    return codes


def validate_inputs(generation_path, generation_sha, inference_path, inference_sha):
    """No numerical imports or expression decoding; authenticate closed provenance."""
    generation_path, inference_path = map(canonical_path, (generation_path, inference_path))
    require(generation_path.name == "complete.json" and inference_path.name == "inference_contract.json",
        "Canonical completion and inference contract names required")
    generation = safe.read_json(generation_path, generation_sha)
    inference = safe.read_json(inference_path, inference_sha)
    require(generation.get("schema") == INFERENCE_SCHEMA and generation.get("completed") is True
        and generation.get("inference_contract_sha256") == inference_sha
        and inference.get("schema") == INFERENCE_SCHEMA, "Completed registered generation required")
    require(safe.read_json(inference_path.with_name("complete.json")) == {"schema": INFERENCE_SCHEMA,
        "registered": True, "inference_contract_sha256": inference_sha}, "Inference registration receipt differs")
    for key in ("submission_performed", "official_cli_validation_performed", "posttraining_performed",
                "scientific_improvement_established"):
        require(generation.get(key) is False, "Generation state differs: " + key)
    policy = inference.get("policy", {})
    for key in ("challenge_treated_used", "genept_used", "promoterai_used", "posttraining_performed",
                "scientific_improvement_established", "upload_performed_by_generator", "clipping_or_thinning"):
        require(policy.get(key) is False, "Inference provenance differs: " + key)
    require(policy.get("stage") == "inference" and policy.get("arm") == "true"
        and policy.get("heun_steps") == 16 and policy.get("strength") == 1.0
        and policy.get("ensemble") == {"nadig_jurkat": .5, "replogle_k562": .5}
        and policy.get("exploratory_submission_authorized_by_user") is True,
        "Fixed exploratory two-source ensemble required")
    training = _bound_json(inference.get("training_contract"),
        BASE / "training_registration/training_contract.json", TRAINING_SHA)
    trained = _bound_json(inference.get("training_completion"),
        BASE / "pretraining/complete.json", TRAINING_COMPLETION_SHA)
    require(trained.get("completed") is True and trained.get("training_contract_sha256") == TRAINING_SHA
        and trained.get("optimizer_steps") == 3600 and trained.get("tracking_events") == 3606,
        "Fixed completed public pretraining required")
    codes = code_pins(inference, training)
    genes, targets = metadata()
    require(inference.get("genes") == genes and inference.get("targets") == targets
        and len(inference.get("modeled_genes", [])) == 482
        and len(inference.get("normalization_genes", [])) == 7097
        and inference.get("manifest_sha256") == METADATA_SHAS["manifest.json"]
        and inference.get("targets_sha256") == METADATA_SHAS["pert_counts.csv"], "Inference official axes differ")
    for key in ("modeled_genes", "normalization_genes"):
        axis = inference[key]
        require(len(set(axis)) == len(axis) and set(axis) <= set(genes), "Projected inference axis differs")
    require(set(inference["modeled_genes"]) <= set(inference["normalization_genes"]), "Output genes outside normalization")
    counts = generation_path.parent / "counts"
    receipt_path = counts / "complete.json"
    receipt = safe.read_json(receipt_path)
    require(receipt == generation.get("output") and receipt.get("schema") == WRITER_SCHEMA
        and receipt.get("completed") is True and receipt.get("generation_contract_sha256") == inference_sha,
        "Embedded count-writer receipt differs")
    for key in ("submission_performed", "official_cli_validated"):
        require(receipt.get(key) is False, "Writer state differs")
    writer = safe.read_json(counts / "writer.json", receipt.get("writer_sha256"))
    require(writer.get("schema") == WRITER_SCHEMA and writer.get("generation_contract_sha256") == inference_sha
        and writer.get("genes") == genes and writer.get("targets") == targets, "Writer axes or provenance differ")
    cfg = writer.get("config", {})
    for key, value in {"expected_genes":18533, "expected_targets":300, "contexts":["A","B","C"],
        "cells_per_group":400, "max_nnz":4750000000, "max_counts_per_cell":1000000}.items():
        require(cfg.get(key) == value, "Writer count safety differs: " + key)
    report = receipt.get("validation", {})
    for key, value in {"shape":[360000,18533], "groups":900, "cells_per_group":400,
        "full_axis_exact":True, "all_quotas_exact":True, "integer_nonnegative_counts":True,
        "canonical_csr":True, "unmodeled_genes_modified_by_writer":False, "sparsity_cap_applied":False}.items():
        require(type(report.get(key)) is type(value) and report[key] == value, "Writer validation differs: " + key)
    require(type(report.get("nnz")) is int and 360000 <= report["nnz"] <= 4750000000
        and 0 < report.get("minimum_library", 0) <= report.get("maximum_library", 0) <= 1000000,
        "Writer numerical safety differs")
    output = canonical_path(counts / "prediction.h5ad")
    require(receipt.get("output_path") == str(output), "Exact generated count path required")
    input_binding = authenticate(output, receipt.get("output_sha256"))
    require(0 < input_binding["bytes"] == receipt.get("output_bytes") <= OUTPUT_LIMIT, "Input size differs")
    journal = authenticate(counts / "groups.jsonl", receipt.get("group_journal_sha256"))
    qc_binding = generation.get("count_adapter_qc")
    require(isinstance(qc_binding, dict) and qc_binding.get("path") == str(generation_path.parent / "count_adapter_qc.json"),
        "Count-adapter QC path differs")
    safe.binding(qc_binding)
    return {"generation": generation, "inference": inference, "training": training, "writer": receipt,
        "input": input_binding, "writer_completion": {"path": str(receipt_path), "sha256": safe.digest(receipt_path)},
        "journal": journal, "code_sha256": codes}


def runtime_preflight(parent):
    require(socket.gethostname().split(".")[0] in {"cbsuvlaminck3", "cbsuvlaminck6"},
        "Use an approved BioHPC compute host, never a head/login node")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU-only packaging requires hidden GPUs")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        require(os.environ.get(name) == "2", "Numerical CPU thread bound must be2: " + name)
    require(len(os.sched_getaffinity(0)) <= 8, "Bind packaging to at most8 CPU cores")
    require(shutil.disk_usage(parent).free >= MIN_FREE_DISK, "At least120GiB free package filesystem required")


def resources(out, started):
    size = 0
    for path in out.rglob("*"):
        require(not path.is_symlink(), "No symlinks in private package output")
        if path.is_file():
            size += path.stat().st_size
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    elapsed = time.monotonic() - started
    require(size <= OUTPUT_LIMIT and rss <= RAM_LIMIT and elapsed <= WALL_SECONDS,
        "Packaging resource budget exceeded")
    return {"peak_rss_bytes": rss, "output_bytes": size, "elapsed_seconds": elapsed}


@contextmanager
def resource_guard(out, started):
    """96GiB address-space ceiling,128GiB per-file ceiling and live checkpoints."""
    require(signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0), "Isolated packaging process required")
    prior = {kind: resource.getrlimit(kind) for kind in (resource.RLIMIT_AS, resource.RLIMIT_FSIZE)}
    handler = signal.getsignal(signal.SIGALRM)
    try:
        for kind, cap in ((resource.RLIMIT_AS, RAM_LIMIT), (resource.RLIMIT_FSIZE, OUTPUT_LIMIT)):
            soft, hard = prior[kind]
            lowered = min(cap, soft) if soft != resource.RLIM_INFINITY else cap
            if hard != resource.RLIM_INFINITY:
                lowered = min(lowered, hard)
            resource.setrlimit(kind, (lowered, hard))
        signal.signal(signal.SIGALRM, lambda *_: resources(out, started))
        signal.setitimer(signal.ITIMER_REAL, 10, 10)
        yield
        resources(out, started)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, handler)
        for kind, value in prior.items():
            resource.setrlimit(kind, value)


def official_loader():
    require(Path(sys.prefix).resolve() == (ROOT / ".venv-vcc").resolve(), "Use the pinned .venv-vcc interpreter")
    package = importlib.import_module("vcc")
    folder = Path(package.__file__).resolve().parent
    require(folder == (ROOT / ".venv-vcc/lib/python3.11/site-packages/vcc").resolve()
        and package.__version__ == OFFICIAL_VERSION, "Official CLI installation differs")
    for name, sha in OFFICIAL_MODULES.items():
        authenticate(folder / name, sha)
    prep = importlib.import_module("vcc.prep")
    require(Path(prep.__file__).resolve() == folder / "prep.py", "Official prep import path differs")
    container = importlib.import_module("vcc.vccfile")
    require(Path(container.__file__).resolve() == folder / "vccfile.py", "Official container import path differs")
    return prep.run_prep, {str(folder / name): sha for name, sha in OFFICIAL_MODULES.items()}, container.validate_vcc


def prep_kwargs(input_path, output_path):
    return dict(input_path=str(input_path), genes_path=str(ROOT / "dataset/controls/gene_names.csv"),
        output_path=str(output_path), perts_path=str(ROOT / "dataset/controls/pert_counts.csv"),
        verify_targets=True, check_cell_counts=True, cells_per_pert=400,
        pert_col="target_gene", celltype_col=None, ntc_name="non-targeting",
        output_pert_col="target_gene", output_celltype_col="celltype",
        context_col="context", output_context_col="context", required_contexts=("A","B","C"),
        encoding=32, allow_discrete=False, require_counts=True, reject_controls=True,
        expected_gene_dim=18533, max_cell_dim=400000, max_nnz=4750000000,
        max_counts_per_cell=1000000, dry_run=False, force=False)


def validate_prep_result(value, input_path, output_path, nnz):
    expected = {"input": str(input_path), "output": str(output_path), "n_cells":360000,
        "n_genes":18533, "encoding":32, "nnz":nnz, "pert_col":"target_gene",
        "output_pert_col":"target_gene", "celltype_col":None, "normalization":"counts-preserved",
        "context_col":"context", "cells_per_context":{c:120000 for c in "ABC"},
        "verified_targets":True, "reordered_genes":False, "dry_run":False, "vcc_member":"pred.h5ad.zst"}
    require(isinstance(value, dict), "Official prep result object required")
    for key, wanted in expected.items():
        require(type(value.get(key)) is type(wanted) and value[key] == wanted, "Official prep result differs: " + key)
    allowed_drops = {"uns[method]", "uns[generation_contract_sha256]", "uns[writer_schema]"}
    require(isinstance(value.get("dropped"), list) and len(value["dropped"]) == len(set(value["dropped"]))
        and set(value["dropped"]) <= allowed_drops,
        "Official prep unexpectedly dropped prediction content")


def validate_archive(path, nnz):
    """Inspect bounded tar metadata only; no second expression decode/prep run."""
    with safe.regular_reader(path) as stream, tarfile.open(fileobj=stream, mode="r:") as archive:
        members = []
        for member in archive:
            members.append(member)
            require(len(members) <= 2 and member.isfile(), "Exactly two regular VCC members required")
        require([m.name for m in members] == ["meta.json", "pred.h5ad.zst"]
            and 0 < members[0].size <= 65536 and 0 < members[1].size <= OUTPUT_LIMIT,
            "Official VCC archive layout differs")
        with archive.extractfile(members[0]) as content:
            meta = json.loads(content.read(65537))
    expected = {"schema":1, "nnz":nnz, "n_obs":360000, "n_vars":18533, "cli_version":OFFICIAL_VERSION}
    require(meta == expected, "Official VCC metadata differs")
    return meta


def package(generation_complete, generation_sha, inference_contract, inference_sha, output_dir):
    started = time.monotonic()
    out = canonical_path(output_dir)
    require(out.parent.is_dir() and not os.path.lexists(out), "Fresh package output with existing parent required")
    require(out.is_relative_to(ROOT) and out != ROOT, "Keep packaging on the project filesystem")
    runtime_preflight(out.parent)
    evidence = validate_inputs(generation_complete, generation_sha, inference_contract, inference_sha)
    input_path = Path(evidence["input"]["path"])
    require(not out.is_relative_to(input_path.parent) and not input_path.is_relative_to(out), "Do not overwrite count inputs")
    own_code = {"package_public_auxiliary_vcc_v10.py": safe.digest(Path(__file__)),
        "public_training_readiness.py": safe.digest(Path(safe.__file__))}
    out.mkdir(mode=0o700)
    temporary_root = out / "temporary"
    temporary_root.mkdir(mode=0o700)
    output_path = out / "prediction.vcc"
    prior_tmp, prior_umask = tempfile.tempdir, os.umask(0o077)
    try:
        tempfile.tempdir = str(temporary_root)
        with resource_guard(out, started):
            runner, official_codes, validate_container = official_loader()
            arguments = prep_kwargs(input_path, output_path)
            dump_new(out / "package_intent.json", {"schema": SCHEMA,
                "generation_completion": {"path":str(canonical_path(generation_complete)), "sha256":generation_sha},
                "inference_contract": {"path":str(canonical_path(inference_contract)), "sha256":inference_sha},
                "input":evidence["input"], "prep_arguments":arguments, "official_cli_version":OFFICIAL_VERSION,
                "official_code_sha256":official_codes, "code_sha256":own_code,
                "limits":{"ram_bytes":RAM_LIMIT,"output_bytes":OUTPUT_LIMIT,"wall_seconds":WALL_SECONDS},
                "submission_performed":False})
            result = runner(**arguments).to_dict()  # Exactly one official validation+prep invocation.
            validate_prep_result(result, input_path, output_path, evidence["writer"]["validation"]["nnz"])
            require(output_path.is_file() and not output_path.is_symlink(), "Official package missing")
            os.chmod(output_path, 0o600)
            validate_container(output_path)
            archive = validate_archive(output_path, result["nnz"])
            require(safe.read_json(canonical_path(generation_complete), generation_sha) == evidence["generation"]
                and safe.read_json(canonical_path(inference_contract), inference_sha) == evidence["inference"],
                "Frozen lineage changed during prep")
            authenticate(input_path, evidence["input"]["sha256"])
            code_pins(evidence["inference"], evidence["training"])
            for name, sha in own_code.items():
                authenticate(ROOT / "scripts" / name, sha)
            for path, sha in official_codes.items():
                authenticate(path, sha)
            metadata()
            package_sha = safe.digest(output_path)
            usage = resources(out, started)
            receipt = {"schema":SCHEMA,"completed":True,"completed_at":datetime.now(timezone.utc).isoformat(),
                "package":{"path":str(output_path),"sha256":package_sha},
                "package_bytes":output_path.stat().st_size,
                "generation_completion":{"path":str(canonical_path(generation_complete)),"sha256":generation_sha},
                "inference_contract":{"path":str(canonical_path(inference_contract)),"sha256":inference_sha},
                "training_contract":evidence["inference"]["training_contract"],
                "training_completion":evidence["inference"]["training_completion"],
                "count_writer_completion":evidence["writer_completion"], "input":evidence["input"],
                "official_cli_version":OFFICIAL_VERSION,"official_code_sha256":official_codes,
                "prep_result":result,"archive_metadata":archive,"code_sha256":own_code,
                "inference_code_sha256":evidence["code_sha256"],"resources":usage,
                "official_cli_validation_performed":True,"submission_performed":False,
                "official_container_validated":True,
                "scientific_improvement_established":False,"input_preserved":True}
            dump_new(out / "complete.json", receipt)
            return receipt
    finally:
        tempfile.tempdir = prior_tmp
        os.umask(prior_umask)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("generation-complete", "generation-sha", "inference-contract", "inference-sha", "out"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    result = package(args.generation_complete, args.generation_sha, args.inference_contract, args.inference_sha, args.out)
    print(json.dumps({"completed":True, "package":result["package"],
        "completion_sha256":safe.digest(Path(args.out) / "complete.json")}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
