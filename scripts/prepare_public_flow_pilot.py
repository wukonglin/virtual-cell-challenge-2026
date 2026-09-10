"""Register metadata-only public folds, then read only authorized H5AD rows.

Two separate invocations are mandatory: ``register`` writes a no-clobber row
contract before any expression access; ``materialize`` authenticates that
contract and each complete source before reading the selected cell rows.
This is a small GO-conditioning engineering pilot, not a submission pipeline.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np


SCHEMA = "public-flow-row-contract-v1"
REPRESENTATION = "log1p_cp10k_shared_measured_genes"
SEALS = {
    "artifacts/scdfm/v7/splits/jurkat_non_harm.json": "3114e59b5430d140cfeeaccd8b2f6974ceb1c14ba254ec41597057ce7a70c439",
    "artifacts/scdfm/v7/training/jurkat_training_preflight.json": "b5dfc61625625e8b70f126e9eb563f12f62ec629e8a138ef6045688c0f233ba1",
    "artifacts/scdfm/v7/training/portable_training_gate_lock.json": "f9c5b8675a1a46c561c056574009722d1faa4bfaf8b4e12e7d7017d3c5a9628d",
}
PRIOR_TARGETS = "ACAT2 CHMP3 DLG5 EIF3H EWSR1 FDPS HDAC8 HMGB2 IDE JAZF1 KIF1B KIF20A KRT18 LRPPRC MED12 METTL14 MTA1 NFE2L1 OXCT1 PAXIP1 PBX1 PLCB3 PMS1 PPP2R3C PRCP RNF2 SMARCA5 SMARCB1 STRAP STX4 SUPT4H1 TCF7L2 TRAPPC6A TSC22D4 USF2 VCL ZNF598".split()
SOURCE_ROLES = {"replogle_k562": "train", "vcc2025_h1_train": "train", "replogle_rpe1": "dev"}
SOURCE_PROFILES = {"rpe1_dev": SOURCE_ROLES,
                   "h1_dev": {"replogle_k562": "train", "nadig_jurkat": "train", "vcc2025_h1_train": "dev"}}


def canonical(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha_file(path):
    with open_regular(path) as stream:
        return sha_stream(stream)


def sha_stream(stream):
    stream.seek(0)
    result = hashlib.sha256()
    for block in iter(lambda: stream.read(8 << 20), b""):
        result.update(block)
    stream.seek(0)
    return result.hexdigest()


@contextmanager
def open_regular(path):
    import stat
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("Expected a regular source file")
        yield stream


def load_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result
    with open_regular(path) as stream:
        return json.load(stream, object_pairs_hook=pairs)


def write_json(path, value):
    with Path(path).open("xb") as stream:
        stream.write(canonical(value))


def verify_receipt(spec):
    if sha_file(spec["receipt_path"]) != spec["receipt_sha256"]:
        raise ValueError("Acquisition receipt changed")
    receipt = load_json(spec["receipt_path"])
    if spec.get("receipt_kind") == "sealed_v7_jurkat":
        if spec["receipt_sha256"] != SEALS["artifacts/scdfm/v7/training/jurkat_training_preflight.json"]:
            raise ValueError("Jurkat source must retain its original sealed identity")
        matching = [item["source"] for item in receipt["datasets"] if item["dataset"] == "nadig_jurkat"]
        if len(matching) != 1 or matching[0]["filename"] != spec["receipt_name"]:
            raise ValueError("Wrong sealed Jurkat source")
    else:
        matching = [item for item in receipt["files"] if item["name"] == spec["receipt_name"]]
    if len(matching) != 1 or any(matching[0][key] != spec[key] for key in ("size_bytes", "sha256")):
        raise ValueError("Source identity differs from verified acquisition receipt")
    if Path(spec["path"]).name != Path(spec["receipt_name"]).name:
        raise ValueError("Source filename differs from receipt")


def strings(values):
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.dtype.kind not in "OSU":
        raise ValueError("Expected one-dimensional string annotations")
    result = np.array([v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in raw])
    if any(not v or v.strip() != v for v in result):
        raise ValueError("Empty/padded annotation is not supported")
    return result


def annotation(frame, key, *, integer_labels=False):
    obj = frame[key]
    if isinstance(obj, h5py.Group):
        if set(obj) != {"categories", "codes"}:
            raise ValueError("Unsupported categorical encoding")
        categories = strings(obj["categories"][:])
        codes = np.asarray(obj["codes"][:])
    elif "__categories" in frame and key in frame["__categories"]:
        categories = strings(frame["__categories"][key][:])
        codes = np.asarray(obj[:])
    else:
        raw = obj[:]
        if integer_labels and raw.dtype.kind in "iu" and raw.ndim == 1:
            return raw.astype(str)
        return strings(raw)
    if codes.ndim != 1 or codes.dtype.kind not in "iu" or np.any(codes < 0) or np.any(codes >= len(categories)):
        raise ValueError("Invalid/missing category codes")
    return categories[codes]


def require_self_contained_h5(handle):
    """A container hash cannot authenticate linked files or virtual datasets."""
    seen = set()
    def visit(group):
        address = h5py.h5o.get_info(group.id).addr
        if address in seen:
            raise ValueError("Repeated/cyclic HDF group links are unsupported")
        seen.add(address)
        for key in group:
            if not isinstance(group.get(key, getlink=True), h5py.HardLink):
                raise ValueError("External and soft HDF links are forbidden")
            obj = group[key]
            if isinstance(obj, h5py.Group):
                visit(obj)
            elif obj.is_virtual or obj.external:
                raise ValueError("Virtual/external-storage datasets are forbidden")
    visit(handle)


def metadata(handle, spec):
    require_self_contained_h5(handle)
    obs, var = handle["obs"], handle["var"]
    fields = {"cells": annotation(obs, spec["cell_key"]),
              "targets": annotation(obs, spec["target_key"]),
              "batches": annotation(obs, spec["batch_key"], integer_labels=True),
              "genes": annotation(var, spec["gene_key"])}
    if len(set(fields["cells"])) != len(fields["cells"]):
        raise ValueError("Duplicate cell IDs")
    if any(len(fields[k]) != len(fields["cells"]) for k in ("targets", "batches")):
        raise ValueError("Annotation lengths differ")
    # Shape and encoding attributes only: never access X numeric values here.
    x = handle["X"]
    shape = tuple(x.shape) if isinstance(x, h5py.Dataset) else tuple(x.attrs["shape"])
    encoding = "dense" if isinstance(x, h5py.Dataset) else x.attrs.get("encoding-type", "")
    if isinstance(encoding, bytes):
        encoding = encoding.decode()
    if shape != (len(fields["cells"]), len(fields["genes"])) or encoding not in ("dense", "csr_matrix"):
        raise ValueError("Unsupported expression shape/encoding")
    fields["encoding"] = encoding
    return fields


def metadata_digest(meta):
    return digest({k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in meta.items()})


def ranked(values, seed, namespace):
    return sorted(values, key=lambda v: (hashlib.sha256(f"{seed}|{namespace}|{v}".encode()).digest(), str(v)))


def choose_groups(meta, *, source, control_label, denied, seed, max_targets=16,
                  max_batches=2, min_cells=8, max_controls=64, max_treated=32,
                  eligible_targets=None):
    """Choose only from annotations, with exact source/batch-matched controls."""
    if set(denied) & {control_label}:
        raise ValueError("Control label conflicts with target exclusions")
    if min(max_targets, max_batches, min_cells, max_controls, max_treated) < 1 or min(max_controls, max_treated) < min_cells:
        raise ValueError("Invalid sampling bounds")
    targets, batches, cells = (meta[k] for k in ("targets", "batches", "cells"))
    controls = {b: np.flatnonzero((targets == control_label) & (batches == b)) for b in np.unique(batches)}
    candidates = set(targets) - set(denied) - {control_label}
    if eligible_targets is not None:
        candidates &= set(eligible_targets)
    groups, selected_targets = [], 0
    for target in ranked(candidates, seed, "target"):
        rows = np.flatnonzero(targets == target)
        eligible = [b for b in np.unique(batches[rows])
                    if len(controls[b]) >= min_cells and np.count_nonzero(batches[rows] == b) >= min_cells]
        if not eligible:
            continue
        for batch in ranked(eligible, seed, source + "|" + target)[:max_batches]:
            treated = rows[batches[rows] == batch]
            def pick(indices, count, kind):
                order = ranked(indices.tolist(), seed, source + "|" + batch + "|" + kind)
                # Hash cell identity, not expression, and keep matrix row order.
                order = sorted(order, key=lambda i: hashlib.sha256(f"{seed}|{source}|{cells[i]}".encode()).digest())
                return sorted(order[:count])
            groups.append({"target": str(target), "batch": str(batch),
                           "control_rows": pick(controls[batch], max_controls, "control"),
                           "treated_rows": pick(treated, max_treated, "treated")})
        selected_targets += 1
        if selected_targets == max_targets:
            break
    if not groups:
        raise ValueError(f"No eligible matched-control groups for {source}")
    return groups


def validate_groups(meta, groups, *, control_label, denied):
    if not groups:
        raise ValueError("Empty source groups")
    pairs, seen_treated = set(), set()
    for group in groups:
        target, batch = group["target"], group["batch"]
        if target in set(denied) | {control_label} or (target, batch) in pairs:
            raise ValueError("Excluded target or duplicate group")
        pairs.add((target, batch))
        for key, label in (("control_rows", control_label), ("treated_rows", target)):
            rows = group[key]
            if not rows or any(type(i) is not int or i < 0 or i >= len(meta["cells"]) for i in rows) or rows != sorted(set(rows)):
                raise ValueError("Invalid authorized row indices")
            if np.any(meta["targets"][rows] != label) or np.any(meta["batches"][rows] != batch):
                raise ValueError("Wrong target/control/batch in authorized rows")
            if key == "treated_rows":
                if seen_treated.intersection(rows):
                    raise ValueError("Repeated treated cells across groups")
                seen_treated.update(rows)


def read_authorized_rows(handle, rows, *, authorized, columns):
    """Check the entire request BEFORE the first expression value read."""
    if any(type(i) is not int for i in rows) or not set(rows) <= set(authorized) or rows != sorted(set(rows)):
        raise ValueError("Unauthorized or duplicate row read")
    x = handle["X"]
    columns = np.asarray(columns)
    if columns.ndim != 1 or columns.dtype.kind not in "iu" or len(set(columns)) != len(columns):
        raise ValueError("Invalid measured-gene columns")
    width = x.shape[1] if isinstance(x, h5py.Dataset) else int(x.attrs["shape"][1])
    if np.any(columns < 0) or np.any(columns >= width):
        raise ValueError("Measured gene outside source axis")
    result = np.empty((len(rows), len(columns)), dtype=np.float64)
    if isinstance(x, h5py.Dataset):
        for out, row in enumerate(rows):
            result[out] = x[row, :][columns]
    else:
        encoding = x.attrs.get("encoding-type")
        if encoding not in ("csr_matrix", b"csr_matrix"):
            raise ValueError("Only dense and CSR are supported")
        lookup = np.full(width, -1, dtype=np.int64)
        lookup[columns] = np.arange(len(columns))
        for out, row in enumerate(rows):
            start, end = map(int, x["indptr"][row:row + 2])
            if not 0 <= start <= end <= len(x["data"]) or len(x["indices"]) != len(x["data"]):
                raise ValueError("Invalid CSR row pointers")
            indices, values = x["indices"][start:end], x["data"][start:end]
            if indices.dtype.kind not in "iu" or np.any(indices < 0) or np.any(indices >= width) or len(set(indices)) != len(indices):
                raise ValueError("Invalid/duplicate CSR gene index")
            mapped = lookup[indices]
            result[out] = 0
            result[out, mapped[mapped >= 0]] = values[mapped >= 0]
    if not np.isfinite(result).all() or np.any(result < 0) or np.any(result != np.floor(result)):
        raise ValueError("Expected nonnegative integer raw UMI counts; no normalized fallback")
    return result


def normalize_counts(counts):
    total = counts.sum(axis=1, dtype=np.float64)
    if np.any(total <= 0):
        raise ValueError("Zero-library cell on common measured gene axis")
    return np.log1p(counts.astype(np.float64) * (10000 / total[:, None])).astype(np.float32)


def source_manifest(args):
    """Construct a bounded allowlist from completed acquisition receipts."""
    profile = getattr(args, "profile", "rpe1_dev")
    roles = SOURCE_PROFILES[profile]
    entries = [
        ("replogle_k562", "dataset/raw/replogle_2022_singlecell", "K562_essential_raw_singlecell_01.h5ad"),
        ("vcc2025_h1_train", "vcc_2025_h1", "train/adata_Training.h5ad"),
        ("replogle_rpe1", "dataset/raw/replogle_2022_singlecell", "rpe1_raw_singlecell_01.h5ad"),
    ]
    if profile == "h1_dev":
        entries = [entries[0], ("nadig_jurkat", "dataset/raw/replogle_nadig", "GSE264667_jurkat_raw_singlecell_01.h5ad"), entries[1]]
    sources = []
    for identifier, directory, name in entries:
        root = args.repo.resolve() / directory
        receipt_path = root / "acquisition_complete.json"
        if profile == "h1_dev" and identifier == "replogle_k562":
            receipt_path = root / "acquisition_k562_complete.json"
        if identifier == "nadig_jurkat":
            receipt_path = args.repo.resolve() / "artifacts/scdfm/v7/training/jurkat_training_preflight.json"
            if sha_file(receipt_path) != SEALS["artifacts/scdfm/v7/training/jurkat_training_preflight.json"]:
                raise ValueError("Jurkat preflight seal changed")
        receipt = load_json(receipt_path)
        if identifier == "nadig_jurkat":
            items = [s["source"] for s in receipt["datasets"] if s["dataset"] == identifier]
        elif receipt.get("status") != "complete":
            raise ValueError("Source acquisition has not completed")
        else:
            items = [r for r in receipt["files"] if r["name"] == name]
        if len(items) != 1:
            raise ValueError("Source absent/duplicated in completed acquisition")
        item = items[0]
        h1 = identifier == "vcc2025_h1_train"
        sources.append({"id": identifier, "role": roles[identifier], "path": str(root / name),
                        "size_bytes": item["size_bytes"], "sha256": item["sha256"],
                        "receipt_path": str(receipt_path), "receipt_sha256": sha_file(receipt_path), "receipt_name": name,
                        "cell_key": "_index" if h1 else "cell_barcode", "target_key": "target_gene" if h1 else "gene",
                        "batch_key": "batch" if h1 else "gem_group", "gene_key": "_index" if h1 else "gene_name",
                        "control_label": "non-targeting"})
        if identifier == "nadig_jurkat":
            sources[-1]["receipt_kind"] = "sealed_v7_jurkat"
    write_json(args.output, {"schema": "public-flow-source-manifest-v1", "source_profile": profile, "sources": sources})
    print(json.dumps({"source_manifest_sha256": sha_file(args.output)}), flush=True)


def register(args):
    if type(args.genes) is not int or args.genes < 1:
        raise ValueError("Output gene count must be positive")
    config = load_json(args.sources)
    profile = config.get("source_profile", "rpe1_dev")
    roles = SOURCE_PROFILES[profile]
    if set(s["id"] for s in config["sources"]) != set(roles) or len(config["sources"]) != 3:
        raise ValueError("Pilot sources differ from the explicit source profile")
    if len({s["sha256"] for s in config["sources"]}) != 3 or len({str(Path(s["path"]).resolve()) for s in config["sources"]}) != 3:
        raise ValueError("Duplicate source bytes/path would double-count a dataset")
    for relative, expected in SEALS.items():
        if sha_file(args.repo / relative) != expected:
            raise ValueError("Existing protected split seal changed")
    original = load_json(args.repo / next(iter(SEALS)))
    denied = sorted(set(original["partition"]["global_exclusion_targets"]) | set(PRIOR_TARGETS))
    if len(denied) != 772:
        raise ValueError("Unexpected protected exclusion union")
    sources, metas, shared = [], {}, None
    for spec in config["sources"]:
        if spec["role"] != roles[spec["id"]]:
            raise ValueError("Reserved context cannot enter fitting")
        if Path(spec["path"]).stat().st_size != spec["size_bytes"]:
            raise ValueError("Source size differs from acquisition receipt")
        verify_receipt(spec)
        with open_regular(spec["path"]) as stream, h5py.File(stream, "r") as handle:
            meta = metadata(handle, spec)
        metas[spec["id"]] = meta
        counts = Counter(meta["genes"])
        unique = {g for g, count in counts.items() if count == 1}
        shared = unique if shared is None else shared & unique
        sources.append({**spec, "metadata_sha256": metadata_digest(meta),
                        "cells_total": len(meta["cells"]), "genes_total": len(meta["genes"]),
                        "duplicate_gene_symbols_excluded": sum(c > 1 for c in counts.values())})
    shared = sorted(shared)
    if len(shared) < args.genes:
        raise ValueError("Insufficient shared uniquely identified genes")
    output_genes = sorted(ranked(shared, args.seed, "output-gene")[:args.genes])
    training_targets = set()
    for spec in sorted(sources, key=lambda s: s["role"] == "dev"):
        meta = metas[spec["id"]]
        eligible_targets = None
        if spec["role"] == "dev":
            eligible_targets = training_targets
        elif profile == "h1_dev":
            eligible_targets = set(metas["vcc2025_h1_train"]["targets"]) - {"non-targeting"} - set(denied)
        groups = choose_groups(meta, source=spec["id"], control_label=spec["control_label"], denied=denied,
                               seed=args.seed, eligible_targets=eligible_targets)
        validate_groups(meta, groups, control_label=spec["control_label"], denied=denied)
        if spec["role"] == "train":
            training_targets.update(g["target"] for g in groups)
        spec["groups"] = groups
    contract = {"schema": SCHEMA, "source_profile": profile, "seed": args.seed, "representation": REPRESENTATION,
                "purpose": "bounded GO-conditioning engineering pilot; not untouched test or submission",
                "fit_contexts": ["K562", "Jurkat"] if profile == "h1_dev" else ["K562", "H1"],
                "development_context": "H1" if profile == "h1_dev" else "RPE1",
                "recipient_treated_allowed": False, "challenge_2026_treated_allowed": False,
                "h1_deduplication": "only newly downloaded train; exclude old identical support copy and weights",
                "excluded_targets": denied, "protected_seals": SEALS,
                "policy": {"max_targets_per_source": 16, "max_batches_per_target": 2, "min_cells": 8,
                           "max_controls_per_group": 64, "max_treated_per_group": 32,
                           "matching": "exact source and batch; labeled non-targeting only",
                           "genes": "unique shared symbols; fixed SHA256-ranked subset; no effect selection",
                           "normalization": "CP10000 over full shared measured axis, then log1p, then output subset",
                           "go": "direct primary terms; exact symbols; no IEA/ND/NOT/obsolete/alternate; fit vocabulary train only",
                           "optimizer": "fresh per-arm AdamW; no prior weights; fixed steps; no dev checkpoint selection",
                           "training": {"steps": 200, "batch_size": 64, "hidden_size": 128, "layers": 2,
                                        "learning_rate": 0.0003, "weight_decay": 0.0001, "gradient_clip_norm": 1.0, "seed": args.seed,
                                        "arms": ["true", "constant", "shuffled"], "shuffle_seed": 20260910,
                                        "sampler": "uniform source then target then matched batch; unpaired cells"},
                           "evaluation": {"scope": "development only; no score claim", "when": "fixed final step only",
                                          "heun_steps": 16, "clipping": False,
                                          "centroid_mse": "mean over modeled genes of squared population-mean difference",
                                          "variance_ratio": "mean predicted population gene variance / mean treated population gene variance",
                                          "mmd2": "biased RBF MMD; kernel exp(-mean gene squared distance / 2); bandwidth fixed at 1 on log1p CP10k",
                                          "scale": REPRESENTATION}},
                "shared_genes": shared, "output_genes": output_genes, "sources": sources,
                "source_manifest_sha256": sha_file(args.sources)}
    write_json(args.output, contract)
    print(json.dumps({"contract_sha256": sha_file(args.output), "shared_genes": len(shared),
                      "output_genes": len(output_genes), "groups": {s["id"]: len(s["groups"]) for s in sources}}))


def materialize(args):
    if sha_file(args.contract) != args.contract_sha256:
        raise ValueError("Contract SHA mismatch")
    contract = load_json(args.contract)
    if contract["schema"] != SCHEMA or contract["representation"] != REPRESENTATION:
        raise ValueError("Unsupported row contract")
    if Path(args.output).exists():
        raise ValueError("Cache already exists")
    buckets = {split: {k: [] for k in ("control", "control_group", "treated", "group", "targets", "sources", "context")} for split in ("train", "dev")}
    shared = contract["shared_genes"]
    output_indices = [shared.index(g) for g in contract["output_genes"]]
    roles = SOURCE_PROFILES[contract.get("source_profile", "rpe1_dev")]
    for spec in contract["sources"]:
        if spec["role"] != roles[spec["id"]]:
            raise ValueError("Invalid fit/development role")
        with open_regular(spec["path"]) as stream:
            before = os.fstat(stream.fileno())
            if before.st_size != spec["size_bytes"] or sha_stream(stream) != spec["sha256"]:
                raise ValueError("Raw source authentication failed BEFORE expression read")
            with h5py.File(stream, "r") as handle:
                meta = metadata(handle, spec)
                if metadata_digest(meta) != spec["metadata_sha256"]:
                    raise ValueError("Metadata changed")
                validate_groups(meta, spec["groups"], control_label=spec["control_label"], denied=contract["excluded_targets"])
                lookup = {g: i for i, g in enumerate(meta["genes"])}
                columns = [lookup[g] for g in shared]
                authorized = sorted({r for g in spec["groups"] for k in ("control_rows", "treated_rows") for r in g[k]})
                values = normalize_counts(read_authorized_rows(handle, authorized, authorized=authorized, columns=columns))[:, output_indices]
                row_lookup = {r: i for i, r in enumerate(authorized)}
                bucket = buckets[spec["role"]]
                for group in spec["groups"]:
                    group_id = len(bucket["targets"])
                    control = values[[row_lookup[r] for r in group["control_rows"]]]
                    treated = values[[row_lookup[r] for r in group["treated_rows"]]]
                    bucket["control"].append(control)
                    bucket["control_group"].extend([group_id] * len(control))
                    bucket["treated"].append(treated)
                    bucket["group"].extend([group_id] * len(treated))
                    bucket["targets"].append(group["target"])
                    bucket["sources"].append(spec["id"])
                    bucket["context"].append(np.concatenate((control.mean(axis=0), control.std(axis=0))))
            after = os.fstat(stream.fileno())
            if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                raise ValueError("Source changed during authorized read")
        print(json.dumps({"source": spec["id"], "authorized_rows_read": len(authorized)}), flush=True)
    arrays = {"schema": np.array("public-flow-pilot-cache-v1"), "representation": np.array(REPRESENTATION),
              "genes": np.array(contract["output_genes"]), "contract_sha256": np.array(args.contract_sha256),
              "source_manifest_sha256": np.array(contract["source_manifest_sha256"])}
    for split, bucket in buckets.items():
        for key, value in bucket.items():
            arrays[split + "_" + key] = np.concatenate(value) if key in ("control", "treated") else np.asarray(value)
    with Path(args.output).open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    write_json(str(args.output) + ".receipt.json", {"schema": "public-flow-cache-receipt-v1", "cache_sha256": sha_file(args.output),
               "contract_sha256": args.contract_sha256, "representation": REPRESENTATION,
               "rows": {s: {k: len(arrays[s + "_" + k]) for k in ("control", "treated", "targets")} for s in buckets}})
    print(json.dumps({"cache_sha256": sha_file(args.output)}), flush=True)


def export_targets(args):
    if sha_file(args.contract) != args.contract_sha256:
        raise ValueError("Contract SHA mismatch")
    contract = load_json(args.contract)
    targets = sorted({g["target"] for s in contract["sources"] for g in s["groups"]})
    train_targets = sorted({g["target"] for s in contract["sources"] if s["role"] == "train" for g in s["groups"]})
    if set(targets) & set(contract["excluded_targets"]):
        raise ValueError("Protected target in export")
    args.output_dir.mkdir(mode=0o700)
    write_json(args.output_dir / "all.json", targets)
    write_json(args.output_dir / "train.json", train_targets)
    print(json.dumps({"all_targets": len(targets), "train_targets": len(train_targets)}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    manifest_parser = sub.add_parser("source-manifest")
    manifest_parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    manifest_parser.add_argument("--output", type=Path, required=True)
    manifest_parser.add_argument("--profile", choices=tuple(SOURCE_PROFILES), default="rpe1_dev")
    register_parser = sub.add_parser("register")
    register_parser.add_argument("--sources", type=Path, required=True)
    register_parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    register_parser.add_argument("--output", type=Path, required=True)
    register_parser.add_argument("--genes", type=int, default=512)
    register_parser.add_argument("--seed", type=int, default=20260909)
    materialize_parser = sub.add_parser("materialize")
    materialize_parser.add_argument("--contract", type=Path, required=True)
    materialize_parser.add_argument("--contract-sha256", required=True)
    materialize_parser.add_argument("--output", type=Path, required=True)
    targets_parser = sub.add_parser("export-targets")
    targets_parser.add_argument("--contract", type=Path, required=True)
    targets_parser.add_argument("--contract-sha256", required=True)
    targets_parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    return {"source-manifest": source_manifest, "register": register, "materialize": materialize, "export-targets": export_targets}[args.action](args)


if __name__ == "__main__":
    main()
