"""Register all eligible public batches while preserving every frozen v2 row.

Registration reads H5AD annotations and X shape/encoding only, never X values.
The separate materializer must authenticate BOTH complete source files and
call validate_metadata before reading any expression. No GO, fitting, network,
credential access or automatic workflow is implemented here.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import socket

import h5py
import numpy as np
import prepare_public_control_anchor_v2 as v2
import public_training_readiness as safe
import run_public_nested_v6 as parent_v6

old = v2.old
CACHE_SCHEMA = "public-replicated-control-anchor-cache-v7"
CONTRACT_SCHEMA = "public-replicated-control-anchor-row-contract-v7"
MANIFEST_SCHEMA = "public-replicated-control-anchor-source-manifest-v7"
REPRESENTATION = v2.REPRESENTATION
SOURCE_IDS = v2.SOURCE_IDS
POOLS = dict(v2.POOLS)
PARENT_RELATIVE = "artifacts/public_flow/control_anchor_v2_20260909"
V6_RELATIVE = "artifacts/public_flow/nested_regularization_v6_20260910/execution/contract.json"
PARENT_PINS = {
    "contract": ("contract.json", "a94565b29d14d98588e8dbf58ad25481e1ebc5e20b209db643667d8a5344bbf1"),
    "sources": ("sources.json", "b7e7d0a300fe4baeaf8d8963894ef2aa6b6681c1e8cb346f7e31c2515e7a78e7"),
    "cache": ("cache.npz", "13ed1ea60e0dff4d516098e9ef5c4d0520c7847ce2bdfc733082f645a5ee7186"),
    "cache_receipt": ("cache.npz.receipt.json", "24d4863ad2e1b11ed74285d5d8379bec82a3b9d9e9b2a9c637853c2ec7170253"),
}
V6_SHA = "cf8968e580a362b9fe552a5e0eb7757385c7f3172a24b845a25f61508541af4c"
EXPECTED_GROUPS = {"replogle_k562": 953, "nadig_jurkat": 814}
EXPECTED_GROUP_DIGESTS = {
    "replogle_k562": "7e4ec0e9641a0a231f0b1fef8d22e69dd6af86d08208cec0d2ce75f2dca06e25",
    "nadig_jurkat": "2f6a0e2a621958b8a80007de24d28c92bf0c76d1299977082d06fc11b9241590",
}
EXPECTED_SHARED = 7563
EXPECTED_PARENT_GROUPS = 398
DEPENDENCIES = tuple(dict.fromkeys(("prepare_public_replicated_v7.py", "public_replicated_cache_v7.py", *parent_v6.MODULES)))
POLICY = {**v2.POLICY, "max_batches_per_target_source": None,
    "batch_selection": "all eligible batches in unchanged v2 seeded order; only the former four-batch cap is removed",
    "stage": "metadata registration and separately authorized cache materialization only",
    "validation": "data preparation and exact parent-subset replay only; no model fitting or evaluation at this stage",
    "comparability": "same v2 measured/output axis and bit-preserved parent rows; expanded cohort is not a new score or an untouched test",
    "new_targets_or_gene_axis": False, "go_or_model_fitting": False,
    "raw_row_chunk_size": 256, "cache_archive_limit_bytes": 512 << 20,
    "extraction_ram_limit_bytes": 8 << 30, "cache_artifact_limit_bytes": 2 << 30}


def equal(actual, expected, label):
    if json.dumps(actual, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True, allow_nan=False):
        raise ValueError("Registered " + label + " differs")


def require(value, message):
    if not value:
        raise ValueError(message)


def repo_path(repo=None):
    return Path(repo).resolve() if repo is not None else Path(__file__).resolve().parents[1]


def canonical_path(path):
    path = Path(path).absolute()
    require(not path.is_symlink(), "Nonsymlink artifact leaf required")
    return path.resolve()


def json_bytes(value):
    return old.canonical(value)


def dump_new(path, value):
    payload = json_bytes(value)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())


def dependency_hashes():
    return {name: safe.digest(Path(__file__).with_name(name)) for name in DEPENDENCIES}


def _parents(repo):
    repo = repo_path(repo)
    bindings = {name: {"path": str(repo / PARENT_RELATIVE / filename), "sha256": digest}
        for name, (filename, digest) in PARENT_PINS.items()}
    prior = safe.binding(bindings["contract"])
    manifest = safe.binding(bindings["sources"])
    receipt = safe.binding(bindings["cache_receipt"])
    require(receipt.get("cache_sha256") == bindings["cache"]["sha256"]
        and receipt.get("contract_sha256") == bindings["contract"]["sha256"]
        and receipt.get("source_manifest_sha256") == bindings["sources"]["sha256"], "Frozen v2 cache receipt binding differs")
    v2.validate_contract(prior, manifest, repo=repo)
    execution_binding = {"path": str(repo / V6_RELATIVE), "sha256": V6_SHA}
    sealed = safe.binding(execution_binding)
    execution, roles, _ = parent_v6.validate_contract(Path(execution_binding["path"]), V6_SHA)
    equal(execution, sealed, "v6 execution parent")
    for key, name in (("cache_sha256", "cache"), ("contract_sha256", "contract"), ("source_manifest_sha256", "sources")):
        require(roles["inputs"][key] == bindings[name]["sha256"], "V6 does not bind the frozen v2 parent")
    return prior, manifest, bindings, execution_binding


def _meta_sources(metas, specs, denied, parent):
    require(isinstance(metas, dict) and set(metas) == set(SOURCE_IDS), "Exactly two public metadata sources required")
    eligible = {s["id"]: v2.eligible_batches(metas[s["id"]], denied, s["control_label"]) for s in specs}
    common = sorted(set.intersection(*(set(eligible[s]) for s in SOURCE_IDS)))
    equal(common, parent["common_targets"], "unchanged common target cohort")
    require(len(common) == POLICY["expected_common_targets"], "Unexpected common target count")
    targets = old.ranked(common, POLICY["seed"], "v2-common-target")
    sources = []
    for spec in specs:
        source, meta = spec["id"], metas[spec["id"]]
        original = next(s for s in parent["sources"] if s["id"] == source)
        metadata_sha = old.metadata_digest(meta)
        require(metadata_sha == original["metadata_sha256"], "Source metadata changed from the frozen v2 identity")
        # One metadata pass instead of a full-array scan for each new batch.
        admitted = set(targets) | {spec["control_label"]}
        positions = {}
        for row, (target, batch) in enumerate(zip(meta["targets"], meta["batches"])):
            if target in admitted:
                positions.setdefault((str(target), str(batch)), []).append(row)
        groups, pools = [], {}
        for target in targets:
            batches = old.ranked(eligible[source][target], POLICY["seed"], source + "|" + target)
            for batch in batches:  # The ONLY sampling change is absence of [:4].
                if batch not in pools:
                    candidates = positions[(spec["control_label"], batch)]
                    ranked = v2.ranked_rows(meta, candidates, source, batch)
                    nc, na, ns = (POLICY[k] for k in ("context_controls_per_batch", "anchor_controls_per_batch", "sham_controls_per_batch"))
                    pools[batch] = {"context_control_rows": sorted(ranked[:nc]), "control_rows": sorted(ranked[nc:nc+na]),
                        "sham_control_rows": sorted(ranked[nc+na:nc+na+ns])}
                treated = positions[(target, batch)]
                rows = sorted(v2.ranked_rows(meta, treated, source, batch)[:POLICY["max_treated_per_group"]])
                groups.append({"target": target, "batch": batch, **pools[batch], "treated_rows": rows})
        counts = Counter(meta["genes"])
        sources.append({**spec, "metadata_sha256": metadata_sha, "cells_total": len(meta["cells"]),
            "genes_total": len(meta["genes"]), "duplicate_gene_symbols_excluded": sum(n > 1 for n in counts.values()), "groups": groups})
    require({s["id"]: len(s["groups"]) for s in sources} == EXPECTED_GROUPS, "Expanded metadata group counts differ from approved bounds")
    equal({s["id"]: old.digest(s["groups"]) for s in sources}, EXPECTED_GROUP_DIGESTS, "independently audited expanded group digests")
    return sources


def _axes(metas, parent):
    axes = [{g for g, n in Counter(metas[s]["genes"]).items() if n == 1} for s in SOURCE_IDS]
    shared = sorted(set.intersection(*axes))
    output = sorted(old.ranked(shared, POLICY["seed"], "v2-output-gene")[:POLICY["output_gene_count"]])
    equal(shared, parent["shared_genes"], "unchanged shared measured axis")
    equal(output, parent["output_genes"], "unchanged output axis")
    require(len(shared) == EXPECTED_SHARED and len(output) == POLICY["output_gene_count"], "Unexpected gene-axis size")
    return shared, output


def group_roster(sources):
    return [{"group_id": i, "source": source["id"], "target": group["target"], "batch": group["batch"]}
        for i, (source, group) in enumerate((s, g) for s in sources for g in s["groups"])]


def parent_mapping(parent, sources):
    expanded = {}
    for i, (source, group) in enumerate((s, g) for s in sources for g in s["groups"]):
        key = (source["id"], group["target"], group["batch"])
        require(key not in expanded, "Repeated expanded source/target/batch group")
        expanded[key] = (i, group)
    mapping = []
    for i, (source, group) in enumerate((s, g) for s in parent["sources"] for g in s["groups"]):
        key = (source["id"], group["target"], group["batch"])
        require(key in expanded, "Frozen v2 group absent from expansion")
        expanded_id, actual = expanded[key]
        equal(actual, group, "preserved v2 control/treated row lists")
        mapping.append({"parent_group_id": i, "expanded_group_id": expanded_id})
    require(len(mapping) == EXPECTED_PARENT_GROUPS and len({m["expanded_group_id"] for m in mapping}) == EXPECTED_PARENT_GROUPS,
        "Exact frozen parent subset mapping required")
    return mapping


def resource_estimate(sources, shared, output):
    groups = sum(len(s["groups"]) for s in sources)
    entries = {name: sum(len(g[field]) for s in sources for g in s["groups"]) for name, field in POOLS.items()}
    populations = sum(entries.values()) * (len(output) * 4 + 2 * 8)
    context = groups * len(output) * 2 * 4
    roster = group_roster(sources)
    string_axes = [[g[k] for g in roster] for k in ("target", "source", "batch")] + [output]
    strings = sum(4 * len(a) * max(map(len, a)) for a in string_axes)
    scalar_strings = 4 * sum(map(len, (CACHE_SCHEMA, REPRESENTATION, "calibration", "0"*64, "0"*64)))
    array_bytes = populations + context + strings + scalar_strings
    archive_overhead = 2 << 20
    require(array_bytes + archive_overhead < POLICY["cache_archive_limit_bytes"], "Expanded cache exceeds the512MiB bound")
    return {"groups": groups, "population_entries": entries,
        "cache_arrays_upper_bound_bytes": array_bytes, "archive_overhead_allowance_bytes": archive_overhead,
        "archive_budget_bytes": POLICY["cache_archive_limit_bytes"],
        "raw_row_chunk_size": POLICY["raw_row_chunk_size"],
        "raw_dense_chunk_upper_bound_bytes": POLICY["raw_row_chunk_size"] * max(s["genes_total"] for s in sources) * 8,
        "shared_dense_chunk_upper_bound_bytes": POLICY["raw_row_chunk_size"] * len(shared) * 8,
        "extraction_ram_budget_bytes": POLICY["extraction_ram_limit_bytes"],
        "cache_artifact_budget_bytes": POLICY["cache_artifact_limit_bytes"]}


def _validate_groups(sources, parent, specs):
    require(isinstance(sources, list) and len(sources) == 2 and [s["id"] for s in sources] == list(SOURCE_IDS), "Source ordering changed")
    for source, spec in zip(sources, specs):
        original = next(s for s in parent["sources"] if s["id"] == source["id"])
        equal({k: v for k, v in source.items() if k != "groups"}, {k: v for k, v in original.items() if k != "groups"}, "source metadata/acquisition identity")
        require(type(source.get("cells_total")) is int and source["cells_total"] > 0, "Invalid source cell count")
        groups = source.get("groups")
        require(isinstance(groups, list) and len(groups) == EXPECTED_GROUPS[source["id"]], "Unexpected expanded group count")
        seen, bytarget, batches, controls, treated = set(), Counter(), {}, {}, set()
        for g in groups:
            require(isinstance(g, dict) and set(g) == {"target", "batch", *POOLS.values()}, "Unexpected group fields")
            target, batch = g["target"], g["batch"]
            require(target in parent["common_targets"] and isinstance(batch, str) and batch and batch == batch.strip()
                and (target, batch) not in seen, "Invalid target/batch identity")
            seen.add((target, batch)); bytarget[target] += 1
            for name, field in POOLS.items():
                values = g[field]
                minimum = POLICY["min_treated_per_batch"] if name == "treated" else POLICY[{"control":"anchor", "context_control":"context", "sham_control":"sham"}[name] + "_controls_per_batch"]
                maximum = POLICY["max_treated_per_group"] if name == "treated" else minimum
                require(isinstance(values, list) and minimum <= len(values) <= maximum
                    and all(type(row) is int and 0 <= row < source["cells_total"] for row in values)
                    and values == sorted(set(values)), "Invalid typed original row list")
            pool = {field:g[field] for name, field in POOLS.items() if name != "treated"}
            require(batch not in batches or all(pool[k] == batches[batch][k] for k in pool), "Target-dependent source/batch control pool")
            batches[batch] = pool
            for field, values in pool.items():
                for row in values:
                    require(row not in controls or controls[row] == (batch, field), "Original control cell crosses batch/pool roles")
                    controls[row] = (batch, field)
            require(not treated & set(g["treated_rows"]), "Treated original row repeated across groups")
            treated.update(g["treated_rows"])
        require(not treated & set(controls), "Treated/control roles overlap")
        require(set(bytarget) == set(parent["common_targets"]) and min(bytarget.values()) >= POLICY["min_batches_per_target_source"],
            "Missing common target or insufficient replication")


def validate_contract(contract, manifest, *, repo=None):
    parent, original_manifest, bindings, v6_binding = _parents(repo_path(repo))
    wanted_manifest = {"schema": MANIFEST_SCHEMA, "scope": "calibration", "sources": original_manifest["sources"]}
    equal(manifest, wanted_manifest, "source manifest")
    fixed = {"schema": CONTRACT_SCHEMA, "representation": REPRESENTATION, "scope": "calibration", "policy": POLICY,
        "source_manifest_sha256": hashlib.sha256(json_bytes(manifest)).hexdigest(), "parent_v2": bindings,
        "parent_v6_execution": v6_binding, "protected_seals": parent["protected_seals"],
        "implementation_sha256": dependency_hashes(), "common_targets": parent["common_targets"],
        "shared_genes": parent["shared_genes"], "output_genes": parent["output_genes"], "excluded_targets": parent["excluded_targets"],
        "raw_expression_read_during_registration": False, "full_source_hash_required_before_materialization": True,
        "metadata_reconstruction_required_before_expression": True, "rpe1_expression_admitted": False,
        "h1_hepg2_challenge2026_feng_expression_admitted": False, "go_or_model_fitting_performed": False,
        "future_training_registered": False, "download_performed": False, "submission_performed": False}
    require(isinstance(contract, dict) and set(contract) == set(fixed) | {"sources", "protocol", "group_roster", "parent_group_mapping", "resource_estimate"},
        "Unexpected replicated contract fields")
    for key, value in fixed.items(): equal(contract.get(key), value, key)
    safe.binding(contract["protocol"], json_document=False)
    require(len(contract["excluded_targets"]) == 772 and len(contract["common_targets"]) == POLICY["expected_common_targets"]
        and len(contract["shared_genes"]) == EXPECTED_SHARED and len(contract["output_genes"]) == POLICY["output_gene_count"], "Frozen cohort axes changed")
    _validate_groups(contract["sources"], parent, manifest["sources"])
    equal({s["id"]: old.digest(s["groups"]) for s in contract["sources"]}, EXPECTED_GROUP_DIGESTS,
        "independently audited expanded group digests")
    equal(contract["group_roster"], group_roster(contract["sources"]), "expanded group roster")
    equal(contract["parent_group_mapping"], parent_mapping(parent, contract["sources"]), "preserved parent mapping")
    equal(contract["resource_estimate"], resource_estimate(contract["sources"], contract["shared_genes"], contract["output_genes"]), "resource estimate")
    return contract


def validate_metadata(contract, metas):
    """Reconstruct full metadata selection BEFORE a materializer reads X."""
    equal(contract.get("policy"), POLICY, "metadata reconstruction policy")
    require(contract.get("schema") == CONTRACT_SCHEMA and contract.get("representation") == REPRESENTATION
        and contract.get("scope") == "calibration", "Wrong metadata reconstruction schema")
    for name in ("contract", "sources", "cache", "cache_receipt"):
        require(contract["parent_v2"][name]["sha256"] == PARENT_PINS[name][1], "Frozen parent digest changed")
    parent = safe.binding(contract["parent_v2"]["contract"])
    manifest = safe.binding(contract["parent_v2"]["sources"])
    sources = _meta_sources(metas, manifest["sources"], parent["excluded_targets"], parent)
    shared, output = _axes(metas, parent)
    equal(contract["sources"], sources, "complete metadata-selected source/groups")
    equal(contract["shared_genes"], shared, "shared genes")
    equal(contract["output_genes"], output, "output genes")
    equal(contract["common_targets"], parent["common_targets"], "common targets")
    equal(contract["excluded_targets"], parent["excluded_targets"], "protected exclusions")
    equal(contract["group_roster"], group_roster(sources), "metadata group roster")
    equal(contract["parent_group_mapping"], parent_mapping(parent, sources), "metadata parent subset")


def read_metadata_sources(specs):
    metas = {}
    for spec in specs:
        with safe.regular_reader(spec["path"]) as stream:
            before = os.fstat(stream.fileno())
            require(before.st_size == spec["size_bytes"], "Source byte size differs from frozen acquisition")
            with h5py.File(stream, "r") as handle:
                meta = old.metadata(handle, spec)
            after = os.fstat(stream.fileno())
            require(all(getattr(before, key) == getattr(after, key) for key in ("st_ino", "st_dev", "st_size", "st_mtime_ns", "st_ctime_ns")),
                "Source changed while reading annotations")
        metas[spec["id"]] = meta
    return metas


def load_contract(contract_path, contract_sha, sources_path, source_sha, *, repo=None):
    contract_path, sources_path = canonical_path(contract_path), canonical_path(sources_path)
    require(contract_path.name == "contract.json" and sources_path == contract_path.with_name("sources.json"),
        "Contract and sources must belong to one completed registration directory")
    contract = safe.read_json(contract_path, contract_sha)
    manifest = safe.read_json(sources_path, source_sha)
    require(contract.get("source_manifest_sha256") == source_sha, "Source manifest digest differs")
    validate_contract(contract, manifest, repo=repo)
    complete = safe.read_json(contract_path.with_name("complete.json"))
    audit_binding = {"path": str(contract_path.with_name("metadata_audit.json")), "sha256": complete.get("metadata_audit_sha256")}
    audit = safe.binding(audit_binding)
    targets_sha = hashlib.sha256(json_bytes(contract["common_targets"])).hexdigest()
    safe.binding({"path": str(contract_path.with_name("targets.json")), "sha256": targets_sha}, json_document=False)
    equal(audit, metadata_audit(contract, contract_sha, source_sha, targets_sha), "completed metadata audit")
    equal(complete, {"schema": CONTRACT_SCHEMA, "completed": True, "contract_sha256": contract_sha,
        "source_manifest_sha256": source_sha, "metadata_audit_sha256": audit_binding["sha256"],
        "expression_values_read": False}, "registration completion binding")
    return contract


def metadata_audit(contract, contract_sha, source_sha, targets_sha):
    return {"schema": "public-replicated-control-anchor-metadata-audit-v7", "metadata_only": True,
        "common_target_count": len(contract["common_targets"]), "shared_gene_count": len(contract["shared_genes"]),
        "output_gene_count": len(contract["output_genes"]), "contract_sha256": contract_sha, "source_manifest_sha256": source_sha,
        "targets_sha256": targets_sha, "source_groups": {s["id"]:len(s["groups"]) for s in contract["sources"]},
        "parent_groups_preserved": len(contract["parent_group_mapping"]), "resource_estimate": contract["resource_estimate"],
        "unique_authorized_rows": {s["id"]:len({r for g in s["groups"] for field in POOLS.values() for r in g[field]}) for s in contract["sources"]},
        "expression_values_read": False, "go_or_model_fitting_performed": False, "future_training_registered": False}


def register(repo, output_dir, protocol_path):
    repo, output_dir, protocol_path = repo_path(repo), canonical_path(output_dir), canonical_path(protocol_path)
    if os.path.lexists(output_dir): raise FileExistsError("Fresh replicated registration directory required")
    parent, original_manifest, bindings, v6_binding = _parents(repo)
    codes = dependency_hashes()
    protocol = {"path": str(protocol_path), "sha256": safe.digest(protocol_path)}
    specs = original_manifest["sources"]
    metas = read_metadata_sources(specs)
    sources = _meta_sources(metas, specs, parent["excluded_targets"], parent)
    shared, output = _axes(metas, parent)
    manifest = {"schema": MANIFEST_SCHEMA, "scope": "calibration", "sources": specs}
    source_sha = hashlib.sha256(json_bytes(manifest)).hexdigest()
    contract = {"schema": CONTRACT_SCHEMA, "representation": REPRESENTATION, "scope": "calibration", "policy": POLICY,
        "source_manifest_sha256": source_sha, "sources": sources, "common_targets": parent["common_targets"],
        "shared_genes": shared, "output_genes": output, "excluded_targets": parent["excluded_targets"],
        "protected_seals": parent["protected_seals"], "implementation_sha256": codes, "protocol": protocol,
        "parent_v2": bindings, "parent_v6_execution": v6_binding,
        "group_roster": group_roster(sources), "parent_group_mapping": parent_mapping(parent, sources),
        "resource_estimate": resource_estimate(sources, shared, output),
        "raw_expression_read_during_registration": False, "full_source_hash_required_before_materialization": True,
        "metadata_reconstruction_required_before_expression": True, "rpe1_expression_admitted": False,
        "h1_hepg2_challenge2026_feng_expression_admitted": False, "go_or_model_fitting_performed": False,
        "future_training_registered": False, "download_performed": False, "submission_performed": False}
    validate_contract(contract, manifest, repo=repo)
    validate_metadata(contract, metas)
    output_dir.mkdir(mode=0o700)
    for name, value in (("sources.json", manifest), ("contract.json", contract), ("targets.json", contract["common_targets"])):
        dump_new(output_dir / name, value)
    audit = metadata_audit(contract, safe.digest(output_dir / "contract.json"), source_sha, safe.digest(output_dir / "targets.json"))
    dump_new(output_dir / "metadata_audit.json", audit)
    dump_new(output_dir / "complete.json", {"schema": CONTRACT_SCHEMA, "completed": True,
        "contract_sha256": audit["contract_sha256"], "source_manifest_sha256": source_sha,
        "metadata_audit_sha256": safe.digest(output_dir / "metadata_audit.json"), "expression_values_read": False})
    return audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=repo_path())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args(argv)
    if socket.gethostname().split(".")[0] not in {"cbsuvlaminck3", "cbsuvlaminck6"}:
        parser.exit(2, "Metadata registration requires an approved BioHPC compute host.\n")
    print(json.dumps(register(args.repo, args.output_dir, args.protocol), sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
