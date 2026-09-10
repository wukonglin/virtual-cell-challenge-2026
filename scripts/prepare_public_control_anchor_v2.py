"""Metadata-first common-target public calibration with disjoint NTC pools.

``register`` never decodes X. ``materialize`` requires the explicit frozen
contract SHA, authenticates BOTH source handles before the first selected X
read, and refuses all sources except K562/Jurkat. RPE1/H1/HepG2/2026 and Feng
expression remain deferred. Existing v1 files and implementation are immutable.
"""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
import io
import json
import os
from pathlib import Path
import socket

import h5py
import numpy as np
import prepare_public_flow_pilot as old
from train_public_flow_pilot import authenticated_json, authenticated_bytes, load_npz, write_new

CACHE_SCHEMA = "public-control-anchor-cache-v2"
CONTRACT_SCHEMA = "public-control-anchor-row-contract-v2"
MANIFEST_SCHEMA = "public-control-anchor-source-manifest-v2"
REPRESENTATION = old.REPRESENTATION
SOURCE_IDS = ("replogle_k562", "nadig_jurkat")
PROTECTED_SEALS = {
    **old.SEALS,
    "artifacts/public_flow/h1_dev_pilot_v1/sources.json": "aa836a654761b3c1639526f0d757aea202afb0fddf1ffd94bd873c3b2bb74fbb",
    "artifacts/public_flow/h1_dev_pilot_v1/contract.json": "64214578c0b20837f29a374fc719e6cc3edfc644d0e184d4ff4cd915110ee33c",
    "artifacts/public_flow/h1_dev_pilot_v1/cache.npz": "408dda72ff7f7135d92ec824940ba2116966adc0ef5aa651869e98d03f45d52c",
    "artifacts/public_flow/h1_dev_pilot_v1/go/features.npz": "b478be572e57a05e74a0833a9a7124cedb48d38c4a19b5a609b0d39fd672fd15",
    "artifacts/public_flow/h1_dev_pilot_v1/go/receipt.json": "98c4f013ef3a2457393a4b1a9fab8d482e6a6478297c7e652d9573201310c383",
}
DEPENDENCIES = ("prepare_public_control_anchor_v2.py", "prepare_public_flow_pilot.py", "train_public_flow_pilot.py",
                "public_flow_conditioning.py", "public_crispri_flow.py", "build_go_target_features.py")
POLICY = {
    "seed": 20260909, "expected_common_targets": 56, "max_common_targets": 64,
    "min_batches_per_target_source": 2, "max_batches_per_target_source": 4,
    "min_treated_per_batch": 8, "max_treated_per_group": 32,
    "context_controls_per_batch": 32, "anchor_controls_per_batch": 32, "sham_controls_per_batch": 8,
    "core_controls_per_batch": 64, "total_controls_with_sham_per_batch": 72,
    "output_gene_count": 512,
    "target_selection": "intersection of eligible K562/Jurkat targets BEFORE shared seeded target ranking",
    "batch_matching": "exact source and gem_group; explicitly labeled non-targeting only",
    "control_pool_selection": "source/batch global seeded cell-ID ordering; target-independent disjoint context/anchor/sham",
    "fit_response": "TRAINING treated mean minus independent TRAINING anchor/reference-pool mean; held-source anchors prediction/evaluation only",
    "context": "context-pool mean and population std only; no anchor/sham expression",
    "normalization": "CP10000 over full K562/Jurkat shared uniquely named measured axis; log1p; fixed output subset",
    "output_genes": "unique shared symbols, seeded SHA256 ranking, no response selection",
    "validation": "both K562/Jurkat leave-one-source-out directions; same targets; fixed lambda10 gain1; no tuning",
    "heldout_context_labels_allowed_for_fitting": False,
    "rpe1": "all expression deferred this round; prior project exposure, not an untouched test",
    "h1_hepg2_challenge2026_feng": "all expression excluded this round",
    "comparability": "new shared axis and new sampled cells; not directly comparable with v1 H1 metrics",
    "sham_limit": "eight outcome NTCs per source/batch are noisy; shared control groups are not independent replicates",
}
POOLS = {"context_control": "context_control_rows", "control": "control_rows", "sham_control": "sham_control_rows", "treated": "treated_rows"}
CACHE_KEYS = {"schema", "representation", "contract_sha256", "source_manifest_sha256", "scope", "genes", "targets", "sources", "batches", "context"} | {
    key for name in POOLS for key in (name, "group" if name == "treated" else name + "_group", name + "_rows")}


def json_bytes(value):
    return old.canonical(value)


def repo_path(repo=None):
    return Path(repo).resolve() if repo is not None else Path(__file__).resolve().parents[1]


def verify_seals(repo):
    for relative, expected in PROTECTED_SEALS.items():
        if old.sha_file(repo / relative) != expected:
            raise ValueError("Protected v7/v1 seal changed")


def exclusions(repo):
    original = old.load_json(repo / next(iter(old.SEALS)))
    values = sorted(set(original["partition"]["global_exclusion_targets"]) | set(old.PRIOR_TARGETS))
    if len(values) != 772:
        raise ValueError("The protected exclusion union must remain exactly 772")
    return values


def source_specs(repo):
    """Reuse authenticated acquisition identities, never include H1 source."""
    relative = "artifacts/public_flow/h1_dev_pilot_v1/sources.json"
    original = authenticated_json(repo / relative, PROTECTED_SEALS[relative])
    result = []
    for source in SOURCE_IDS:
        matches = [s for s in original["sources"] if s["id"] == source]
        if len(matches) != 1:
            raise ValueError("Calibration source missing or duplicated")
        spec = {**matches[0], "role": "calibration"}
        old.verify_receipt(spec)
        result.append(spec)
    return result


def dependency_hashes():
    return {name: old.sha_file(Path(__file__).with_name(name)) for name in DEPENDENCIES}


def eligible_batches(meta, denied, control_label="non-targeting"):
    counts = Counter(zip(meta["targets"], meta["batches"]))
    controls = {batch: n for (target, batch), n in counts.items() if target == control_label}
    found = {}
    for (target, batch), count in counts.items():
        if (target not in set(denied) | {control_label} and count >= POLICY["min_treated_per_batch"]
                and controls.get(batch, 0) >= POLICY["total_controls_with_sham_per_batch"]):
            found.setdefault(str(target), []).append(str(batch))
    return {target: sorted(batches) for target, batches in found.items()
            if len(batches) >= POLICY["min_batches_per_target_source"]}


def ranked_rows(meta, indices, source, batch):
    seed = POLICY["seed"]
    return sorted((int(row) for row in indices), key=lambda row: (hashlib.sha256(
        f"{seed}|{source}|{batch}|{meta['cells'][row]}".encode()).digest(), str(meta["cells"][row])))


def select_cohort(metas, specs, denied):
    eligible = {spec["id"]: eligible_batches(metas[spec["id"]], denied, spec["control_label"]) for spec in specs}
    common = set.intersection(*(set(eligible[source]) for source in SOURCE_IDS))
    if len(common) != POLICY["expected_common_targets"] or len(common) > POLICY["max_common_targets"]:
        raise ValueError("Common-target metadata count differs from the approved bounded cohort")
    targets = old.ranked(common, POLICY["seed"], "v2-common-target")
    chosen = []
    for spec in specs:
        source, meta = spec["id"], metas[spec["id"]]
        pools, groups = {}, []
        for target in targets:
            batches = old.ranked(eligible[source][target], POLICY["seed"], source + "|" + target)[:POLICY["max_batches_per_target_source"]]
            for batch in batches:
                if batch not in pools:
                    candidates = np.flatnonzero((meta["targets"] == spec["control_label"]) & (meta["batches"] == batch))
                    ordered = ranked_rows(meta, candidates, source, batch)
                    nc, na = POLICY["context_controls_per_batch"], POLICY["anchor_controls_per_batch"]
                    ns = POLICY["sham_controls_per_batch"]
                    pools[batch] = {"context_control_rows": sorted(ordered[:nc]), "control_rows": sorted(ordered[nc:nc+na]),
                                    "sham_control_rows": sorted(ordered[nc+na:nc+na+ns])}
                treated = np.flatnonzero((meta["targets"] == target) & (meta["batches"] == batch))
                rows = sorted(ranked_rows(meta, treated, source, batch)[:POLICY["max_treated_per_group"]])
                groups.append({"target": target, "batch": batch, **pools[batch], "treated_rows": rows})
        unique_genes = Counter(meta["genes"])
        chosen.append({**spec, "metadata_sha256": old.metadata_digest(meta), "cells_total": len(meta["cells"]),
                       "genes_total": len(meta["genes"]), "duplicate_gene_symbols_excluded": sum(n > 1 for n in unique_genes.values()),
                       "groups": groups})
    return sorted(targets), chosen


def validate_contract(contract, manifest, *, repo=None):
    repo = repo_path(repo)
    if (contract.get("schema") != CONTRACT_SCHEMA or manifest.get("schema") != MANIFEST_SCHEMA
            or contract.get("representation") != REPRESENTATION or contract.get("scope") != "calibration"
            or manifest.get("scope") != "calibration"
            or contract.get("policy") != POLICY or contract.get("protected_seals") != PROTECTED_SEALS
            or contract.get("excluded_targets") != exclusions(repo)
            or contract.get("raw_expression_read_during_registration") is not False
            or contract.get("full_source_hash_required_before_materialization") is not True
            or contract.get("rpe1_expression_admitted") is not False
            or contract.get("h1_hepg2_challenge2026_feng_expression_admitted") is not False):
        raise ValueError("Contract schema, frozen policy or protected exclusions differ")
    verify_seals(repo)
    if contract.get("implementation_sha256") != dependency_hashes():
        raise ValueError("Frozen preparation implementation changed")
    protocol = contract.get("protocol", {})
    authenticated_bytes(Path(protocol["path"]), protocol["sha256"], 1 << 20)
    expected_specs = source_specs(repo)
    if manifest.get("sources") != expected_specs:
        raise ValueError("Only authenticated K562/Jurkat calibration sources are allowed")
    if len({spec["sha256"] for spec in expected_specs}) != 2 or len({str(Path(s["path"]).resolve()) for s in expected_specs}) != 2:
        raise ValueError("Duplicate dataset bytes or source path")
    sources = contract.get("sources")
    if not isinstance(sources, list) or len(sources) != 2:
        raise ValueError("Exactly two calibration sources required")
    shared, output, targets = (contract.get(name) for name in ("shared_genes", "output_genes", "common_targets"))
    for axis in (shared, output, targets):
        if not isinstance(axis, list) or not axis or any(not isinstance(s, str) or not s or s != s.strip() for s in axis) or axis != sorted(set(axis)):
            raise ValueError("Axes must be sorted, unique, nonempty strings")
    if (len(output) != POLICY["output_gene_count"] or not set(output) <= set(shared)
            or len(targets) != POLICY["expected_common_targets"] or set(targets) & set(contract["excluded_targets"])):
        raise ValueError("Gene/target axes differ from approved bounds")
    expected_output = sorted(old.ranked(shared, POLICY["seed"], "v2-output-gene")[:POLICY["output_gene_count"]])
    if output != expected_output:
        raise ValueError("Output genes were not selected by the frozen metadata-only rule")
    for source, original in zip(sources, expected_specs):
        if any(source.get(key) != value for key, value in original.items()):
            raise ValueError("Source role/identity differs from authenticated manifest")
        groups = source.get("groups")
        cells = source.get("cells_total")
        if not isinstance(groups, list) or type(cells) is not int or cells < 1:
            raise ValueError("Malformed source groups/cell count")
        bytarget, pairs, batch_pools, treated_seen = Counter(), set(), {}, set()
        all_controls = set()
        for group in groups:
            if set(group) != {"target", "batch", *POOLS.values()}:
                raise ValueError("Malformed matching group schema")
            target, batch = group["target"], group["batch"]
            if target not in targets or not isinstance(batch, str) or not batch or (target, batch) in pairs:
                raise ValueError("Invalid target, batch or repeated matching group")
            pairs.add((target, batch)); bytarget[target] += 1
            for name, field in POOLS.items():
                rows = group[field]
                minimum = POLICY["min_treated_per_batch"] if name == "treated" else POLICY[{"control": "anchor", "context_control": "context", "sham_control": "sham"}[name] + "_controls_per_batch"]
                maximum = POLICY["max_treated_per_group"] if name == "treated" else minimum
                if (not isinstance(rows, list) or not minimum <= len(rows) <= maximum
                        or any(type(row) is not int or not 0 <= row < cells for row in rows) or rows != sorted(set(rows))):
                    raise ValueError("Invalid control/treated pool row list or size")
            pool = {key: group[key] for key in ("context_control_rows", "control_rows", "sham_control_rows")}
            if batch in batch_pools and pool != batch_pools[batch]:
                raise ValueError("Control pools must be globally identical within source/batch")
            poolsets = [set(rows) for rows in pool.values()]
            if any(poolsets[i] & poolsets[j] for i in range(3) for j in range(i)):
                raise ValueError("Context, anchor and sham controls must be disjoint")
            if treated_seen & set(group["treated_rows"]):
                raise ValueError("Treated cells repeated across groups")
            treated_seen.update(group["treated_rows"])
            batch_pools[batch] = pool
            all_controls.update(set.union(*poolsets))
        if treated_seen & all_controls:
            raise ValueError("Control/treated roles overlap")
        if set(bytarget) != set(targets) or any(not POLICY["min_batches_per_target_source"] <= n <= POLICY["max_batches_per_target_source"] for n in bytarget.values()):
            raise ValueError("Every common target needs two to four batches in BOTH sources")
        for i, (batch, pool) in enumerate(batch_pools.items()):
            rows = set.union(*(set(v) for v in pool.values()))
            for previous in list(batch_pools.values())[:i]:
                if rows & set.union(*(set(v) for v in previous.values())):
                    raise ValueError("A control cell cannot appear in different batches")
    return contract


def register(repo, output_dir, protocol_path):
    repo, output_dir = repo_path(repo), Path(output_dir)
    if os.path.lexists(output_dir):
        raise FileExistsError("Fresh registration directory required")
    verify_seals(repo)
    denied, specs, metas = exclusions(repo), source_specs(repo), {}
    shared = None
    for spec in specs:
        with old.open_regular(spec["path"]) as stream:
            if os.fstat(stream.fileno()).st_size != spec["size_bytes"]:
                raise ValueError("Source size differs from authenticated receipt")
            with h5py.File(stream, "r") as handle:
                meta = old.metadata(handle, spec)
        metas[spec["id"]] = meta
        counts = Counter(meta["genes"])
        unique = {g for g, n in counts.items() if n == 1}
        shared = unique if shared is None else shared & unique
    targets, sources = select_cohort(metas, specs, denied)
    shared = sorted(shared)
    if len(shared) < POLICY["output_gene_count"]:
        raise ValueError("Insufficient shared measured genes")
    manifest = {"schema": MANIFEST_SCHEMA, "scope": "calibration", "sources": specs}
    source_sha = hashlib.sha256(json_bytes(manifest)).hexdigest()
    protocol_path = Path(protocol_path).resolve()
    protocol_sha = old.sha_file(protocol_path)
    contract = {"schema": CONTRACT_SCHEMA, "representation": REPRESENTATION, "scope": "calibration", "policy": POLICY,
                "source_manifest_sha256": source_sha, "sources": sources, "excluded_targets": denied,
                "protected_seals": PROTECTED_SEALS, "implementation_sha256": dependency_hashes(),
                "protocol": {"path": str(protocol_path), "sha256": protocol_sha},
                "common_targets": targets, "shared_genes": shared,
                "output_genes": sorted(old.ranked(shared, POLICY["seed"], "v2-output-gene")[:POLICY["output_gene_count"]]),
                "raw_expression_read_during_registration": False, "full_source_hash_required_before_materialization": True,
                "rpe1_expression_admitted": False, "h1_hepg2_challenge2026_feng_expression_admitted": False}
    validate_contract(contract, manifest, repo=repo)
    output_dir.mkdir(mode=0o700)
    for name, content in (("sources.json", manifest), ("contract.json", contract), ("targets.json", targets)):
        write_new(output_dir / name, json_bytes(content))
    audit = {"metadata_only": True, "common_target_count": len(targets), "shared_gene_count": len(shared),
             "output_gene_count": len(contract["output_genes"]), "contract_sha256": old.sha_file(output_dir / "contract.json"),
             "source_manifest_sha256": source_sha, "targets_sha256": old.sha_file(output_dir / "targets.json"),
             "source_groups": {s["id"]: len(s["groups"]) for s in sources},
             "unique_authorized_rows": {s["id"]: len({r for g in s["groups"] for field in POOLS.values() for r in g[field]}) for s in sources}}
    write_new(output_dir / "metadata_audit.json", json_bytes(audit))
    return audit


def load_contract(contract_path, contract_sha256, sources_path, source_sha256, *, repo=None):
    contract = authenticated_json(Path(contract_path), contract_sha256)
    manifest = authenticated_json(Path(sources_path), source_sha256)
    if contract.get("source_manifest_sha256") != source_sha256:
        raise ValueError("Source-manifest digest differs from frozen row contract")
    validate_contract(contract, manifest, repo=repo)
    return contract


def validate_cache(arrays, contract, contract_sha256, source_sha256):
    if set(arrays) != CACHE_KEYS:
        raise ValueError("Unexpected v2 cache keys")
    expected = {"schema": CACHE_SCHEMA, "representation": REPRESENTATION, "scope": "calibration",
                "contract_sha256": contract_sha256, "source_manifest_sha256": source_sha256}
    for key, value in expected.items():
        if arrays[key].shape != () or arrays[key].dtype.kind != "U" or str(arrays[key]) != value:
            raise ValueError("Cache scalar identity differs")
    roster = [(source, group) for source in contract["sources"] for group in source["groups"]]
    axes = {"genes": contract["output_genes"], "sources": [s["id"] for s, _ in roster],
            "targets": [g["target"] for _, g in roster], "batches": [g["batch"] for _, g in roster]}
    for key, expected_values in axes.items():
        if arrays[key].dtype.kind != "U" or arrays[key].shape != (len(expected_values),) or arrays[key].tolist() != expected_values:
            raise ValueError("Cache group/gene axis differs from frozen rows")
    width = len(contract["output_genes"])
    for name, field in POOLS.items():
        group_key = "group" if name == "treated" else name + "_group"
        counts = [len(g[field]) for _, g in roster]
        labels, rows, values = arrays[group_key], arrays[name + "_rows"], arrays[name]
        if (labels.dtype != np.int64 or not np.array_equal(labels, np.repeat(np.arange(len(roster), dtype=np.int64), counts))
                or rows.dtype != np.int64 or rows.shape != (sum(counts),)
                or rows.tolist() != [row for _, g in roster for row in g[field]]):
            raise ValueError("Cache row IDs/group counts differ from authorized pools")
        if values.dtype != np.float32 or values.shape != (sum(counts), width) or not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("Cache expression must be finite nonnegative float32 on the frozen axis")
        seen = {}
        for group, (source, _) in enumerate(roster):
            chosen = labels == group
            for row, value in zip(rows[chosen], values[chosen]):
                identity = (source["id"], int(row))
                if identity in seen and not np.array_equal(seen[identity], value):
                    raise ValueError("Repeated source-row cache values differ")
                seen[identity] = value
    context = arrays["context"]
    if context.dtype != np.float32 or context.shape != (len(roster), width*2) or not np.isfinite(context).all():
        raise ValueError("Invalid v2 context matrix")
    for group in range(len(roster)):
        values = arrays["context_control"][arrays["context_control_group"] == group].astype(np.float64)
        expected_context = np.concatenate((values.mean(0), values.std(0))).astype(np.float32)
        if not np.array_equal(context[group], expected_context):
            raise ValueError("Context must be computed solely from its independent context pool")
    return arrays


def load_cache(cache_path, cache_sha256, contract_path, contract_sha256, sources_path, source_sha256, *, repo=None):
    contract = load_contract(contract_path, contract_sha256, sources_path, source_sha256, repo=repo)
    arrays = load_npz(Path(cache_path), cache_sha256, CACHE_KEYS)
    return validate_cache(arrays, contract, contract_sha256, source_sha256), contract


def materialize(contract_path, contract_sha256, sources_path, source_sha256, output, *, repo=None):
    output = Path(output)
    if os.path.lexists(output) or os.path.lexists(str(output) + ".receipt.json"):
        raise FileExistsError("Fresh cache and receipt required")
    contract = load_contract(contract_path, contract_sha256, sources_path, source_sha256, repo=repo)
    buckets = {name: [] for name in CACHE_KEYS - {"schema", "representation", "scope", "contract_sha256", "source_manifest_sha256", "genes"}}
    with ExitStack() as stack:
        opened = []
        # Authenticate ALL selected source handles and group metadata before ANY X value.
        for spec in contract["sources"]:
            stream = stack.enter_context(old.open_regular(spec["path"]))
            stat = os.fstat(stream.fileno())
            if stat.st_size != spec["size_bytes"] or old.sha_stream(stream) != spec["sha256"]:
                raise ValueError("Raw source authentication failed BEFORE any expression read")
            handle = stack.enter_context(h5py.File(stream, "r"))
            meta = old.metadata(handle, spec)
            if old.metadata_digest(meta) != spec["metadata_sha256"]:
                raise ValueError("Source annotations changed")
            for name, field in POOLS.items():
                controls = "context_control_rows" if name == "treated" else field
                converted = [{"target": g["target"], "batch": g["batch"], "control_rows": g[controls], "treated_rows": g["treated_rows"]} for g in spec["groups"]]
                old.validate_groups(meta, converted, control_label=spec["control_label"], denied=contract["excluded_targets"])
            opened.append((spec, stream, handle, meta, stat))
        for spec, stream, handle, meta, before in opened:
            shared = contract["shared_genes"]
            counts = Counter(meta["genes"])
            if any(counts[g] != 1 for g in shared):
                raise ValueError("Shared gene axis is not uniquely measured")
            lookup = {gene: i for i, gene in enumerate(meta["genes"])}
            columns = [lookup[g] for g in shared]
            selected = sorted({row for g in spec["groups"] for field in POOLS.values() for row in g[field]})
            values = old.normalize_counts(old.read_authorized_rows(handle, selected, authorized=selected, columns=columns))
            values = values[:, [shared.index(g) for g in contract["output_genes"]]]
            row_lookup = {row: i for i, row in enumerate(selected)}
            for group in spec["groups"]:
                index = len(buckets["targets"])
                buckets["targets"].append(group["target"]); buckets["sources"].append(spec["id"]); buckets["batches"].append(group["batch"])
                for name, field in POOLS.items():
                    selected_values = values[[row_lookup[row] for row in group[field]]]
                    buckets[name].append(selected_values)
                    buckets[name + "_rows"].extend(group[field])
                    buckets["group" if name == "treated" else name + "_group"].extend([index] * len(group[field]))
                    if name == "context_control":
                        statistics = selected_values.astype(np.float64)
                        buckets["context"].append(np.concatenate((statistics.mean(0), statistics.std(0))).astype(np.float32))
            after = os.fstat(stream.fileno())
            if any(getattr(before, key) != getattr(after, key) for key in ("st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")):
                raise ValueError("Source changed during authorized expression read")
    arrays = {"schema": np.array(CACHE_SCHEMA), "representation": np.array(REPRESENTATION), "scope": np.array("calibration"),
              "contract_sha256": np.array(contract_sha256), "source_manifest_sha256": np.array(source_sha256), "genes": np.array(contract["output_genes"])}
    for name, values in buckets.items():
        arrays[name] = np.concatenate(values) if name in POOLS else np.asarray(values)
    validate_cache(arrays, contract, contract_sha256, source_sha256)
    buffer = io.BytesIO(); np.savez_compressed(buffer, **arrays)
    if len(buffer.getvalue()) > 512 << 20 or sum(v.nbytes for v in arrays.values()) > 512 << 20:
        raise ValueError("Cache exceeds bounded 512MiB archive/expanded array budget")
    write_new(output, buffer.getvalue())
    receipt = {"schema": "public-control-anchor-cache-receipt-v2", "cache_sha256": old.sha_file(output),
               "contract_sha256": contract_sha256, "source_manifest_sha256": source_sha256,
               "scope": "calibration", "raw_sources_opened": list(SOURCE_IDS), "groups": len(arrays["targets"]),
               "population_entries": {name: len(arrays[name]) for name in POOLS}, "rpe1_h1_hepg2_challenge2026_feng_opened": False}
    write_new(Path(str(output) + ".receipt.json"), json_bytes(receipt))
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    reg = commands.add_parser("register")
    reg.add_argument("--repo", type=Path, default=repo_path())
    reg.add_argument("--output-dir", type=Path, required=True)
    reg.add_argument("--protocol", type=Path, required=True)
    mat = commands.add_parser("materialize")
    mat.add_argument("--repo", type=Path, default=repo_path())
    for name in ("contract", "sources", "output"):
        mat.add_argument("--" + name, type=Path, required=True)
    mat.add_argument("--contract-sha256", required=True)
    mat.add_argument("--source-manifest-sha256", required=True)
    args = parser.parse_args(argv)
    if socket.gethostname().split(".")[0] not in {"cbsuvlaminck3", "cbsuvlaminck6"}:
        raise RuntimeError("Use an approved BioHPC compute host; never an HPC login/head node")
    if args.command == "register":
        result = register(args.repo, args.output_dir, args.protocol)
    else:
        result = materialize(args.contract, args.contract_sha256, args.sources, args.source_manifest_sha256, args.output, repo=args.repo)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
