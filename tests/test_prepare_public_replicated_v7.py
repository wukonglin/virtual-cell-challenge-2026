"""Synthetic H5 metadata registration; expression reads forbidden in register."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_public_replicated_v7 as gate


def save(path, value):
    path.write_bytes(gate.json_bytes(value))
    return gate.safe.digest(path)


@pytest.fixture
def cohort(tmp_path, monkeypatch):
    policy = {**gate.v2.POLICY, "expected_common_targets": 2, "max_common_targets": 2, "output_gene_count": 3}
    monkeypatch.setattr(gate.v2, "POLICY", policy)
    monkeypatch.setattr(gate, "POLICY", {**gate.POLICY, "expected_common_targets": 2, "max_common_targets": 2, "output_gene_count": 3})
    monkeypatch.setattr(gate, "EXPECTED_GROUPS", {"replogle_k562": 10, "nadig_jurkat": 12})
    monkeypatch.setattr(gate, "EXPECTED_SHARED", 3)
    monkeypatch.setattr(gate, "EXPECTED_PARENT_GROUPS", 16)
    monkeypatch.setattr(gate, "dependency_hashes", lambda: {"synthetic_pinned_implementation": "a" * 64})
    denied = [f"DENIED{i:03}" for i in range(772)]
    specs, metas = [], {}
    for source, nbatches in zip(gate.SOURCE_IDS, (5, 6)):
        rows = []
        for batch in range(nbatches + 1):
            label = f"batch{batch}"
            control_count = 72 if batch < nbatches else 71
            for target, count in (("control", control_count), ("A", 40 if batch == 0 else 8), ("B", 8),
                                  (denied[0], 8), ("SMALL", 7)):
                rows += [(label, target, index) for index in range(count)]
        path = tmp_path / (source + ".h5ad")
        row_number = np.arange(len(rows), dtype=np.int64)
        source_salt = 0 if source == "replogle_k562" else 17
        # Distinct nonproportional integer counts make wrong row/source joins
        # visible after shared-axis normalization, unlike a uniform fixture.
        expression = np.column_stack((row_number + 1 + source_salt,
            row_number % 7 + 1 + source_salt,
            (3 * row_number) % 11 + 1 + 2 * source_salt,
            row_number % 5 + 1, row_number % 3 + 1)).astype(np.float32)
        expression[[i for i, (_, target, _) in enumerate(rows) if target in {denied[0], "SMALL"}]] = np.nan
        with h5py.File(path, "w") as handle:
            obs, var = handle.create_group("obs"), handle.create_group("var")
            obs.create_dataset("cell", data=np.array([f"{source}-{b}-{t}-{i}" for b,t,i in rows], dtype="S"))
            obs.create_dataset("target", data=np.array([t for _,t,_ in rows], dtype="S"))
            obs.create_dataset("batch", data=np.array([b for b,_,_ in rows], dtype="S"))
            var.create_dataset("gene", data=np.array(["G1", "G2", "G3", "DUP", "DUP"], dtype="S"))
            handle.create_dataset("X", data=expression)
        spec = {"id": source, "role": "calibration", "path": str(path), "sha256": gate.safe.digest(path),
            "size_bytes": path.stat().st_size, "cell_key": "cell", "target_key": "target", "batch_key": "batch", "gene_key": "gene",
            "control_label": "control", "receipt_name": path.name, "receipt_path": str(tmp_path / (source + ".receipt.json"))}
        spec["receipt_sha256"] = save(Path(spec["receipt_path"]), {"files":[{"name":path.name,"sha256":spec["sha256"],"size_bytes":spec["size_bytes"]}]})
        with h5py.File(path, "r") as handle: metas[source] = gate.old.metadata(handle, spec)
        specs.append(spec)
    targets, selected = gate.v2.select_cohort(metas, specs, denied)
    expanded_policy = {**policy, "max_batches_per_target_source": 100}
    monkeypatch.setattr(gate.v2, "POLICY", expanded_policy)
    _, expanded = gate.v2.select_cohort(metas, specs, denied)
    monkeypatch.setattr(gate.v2, "POLICY", policy)
    monkeypatch.setattr(gate, "EXPECTED_GROUP_DIGESTS", {s["id"]:gate.old.digest(s["groups"]) for s in expanded})
    parent = {"schema": gate.v2.CONTRACT_SCHEMA, "scope": "calibration", "representation": gate.REPRESENTATION,
        "sources": selected, "common_targets": targets, "shared_genes": ["G1", "G2", "G3"], "output_genes": ["G1", "G2", "G3"],
        "excluded_targets": denied, "protected_seals": {"synthetic": "b" * 64}}
    manifest = {"schema": gate.v2.MANIFEST_SCHEMA, "scope": "calibration", "sources": specs}
    parent_dir = tmp_path / "parent-v2"
    parent_dir.mkdir()
    bindings = {}
    for key, filename, payload in (("contract", "contract.json", parent), ("sources", "sources.json", manifest)):
        bindings[key] = {"path": str(parent_dir / filename), "sha256": save(parent_dir / filename, payload)}
    cache = parent_dir / "cache.npz"
    cache.write_bytes(b"Opaque synthetic parent: registration must never decode this")
    bindings["cache"] = {"path": str(cache), "sha256": gate.safe.digest(cache)}
    receipt = {"cache_sha256": bindings["cache"]["sha256"], "contract_sha256": bindings["contract"]["sha256"],
        "source_manifest_sha256": bindings["sources"]["sha256"]}
    bindings["cache_receipt"] = {"path": str(parent_dir / "cache.npz.receipt.json"),
        "sha256": save(parent_dir / "cache.npz.receipt.json", receipt)}
    monkeypatch.setattr(gate, "PARENT_PINS", {k:(Path(v["path"]).name,v["sha256"]) for k,v in bindings.items()})
    prior6 = tmp_path / "v6.json"
    v6_binding = {"path": str(prior6), "sha256": save(prior6, {"schema": "synthetic-v6-execution"})}
    monkeypatch.setattr(gate, "_parents", lambda repo: (copy.deepcopy(parent), copy.deepcopy(manifest), copy.deepcopy(bindings), copy.deepcopy(v6_binding)))
    protocol = tmp_path / "protocol.md"
    protocol.write_text("Synthetic registration-only expanded public-batch protocol")
    return {"root": tmp_path, "specs": specs, "metas": metas, "parent_contract": parent,
        "parent_manifest": manifest, "protocol": protocol, "parent_v2": bindings}


def registered(cohort):
    out = cohort["root"] / "v7"
    audit = gate.register(cohort["root"], out, cohort["protocol"])
    contract = gate.load_contract(out / "contract.json", audit["contract_sha256"], out / "sources.json",
        audit["source_manifest_sha256"], repo=cohort["root"])
    return out, audit, contract


def forbid_expression(monkeypatch):
    original = h5py.Dataset.__getitem__
    def guarded(dataset, key):
        if dataset.name == "/X" or dataset.name.startswith("/X/"):
            pytest.fail("Expression decoded by metadata-only registration")
        return original(dataset, key)
    monkeypatch.setattr(h5py.Dataset, "__getitem__", guarded)
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("NPZ decoded by metadata-only registration"))


def test_registration_removes_only_batch_cap_and_preserves_parent(cohort, monkeypatch):
    forbid_expression(monkeypatch)
    _, audit, contract = registered(cohort)
    assert audit["metadata_only"] and audit["source_groups"] == gate.EXPECTED_GROUPS
    assert audit["parent_groups_preserved"] == 16
    assert contract["common_targets"] == ["A", "B"]
    assert contract["shared_genes"] == contract["output_genes"] == ["G1", "G2", "G3"]
    assert contract["policy"]["max_batches_per_target_source"] is None
    assert contract["policy"]["max_treated_per_group"] == 32
    assert all(len(g["treated_rows"]) <= 32 for s in contract["sources"] for g in s["groups"])
    old = [(s["id"], g) for s in cohort["parent_contract"]["sources"] for g in s["groups"]]
    new = [(s["id"], g) for s in contract["sources"] for g in s["groups"]]
    for m in contract["parent_group_mapping"]:
        assert old[m["parent_group_id"]] == new[m["expanded_group_id"]]
    assert all(g["target"] not in {"SMALL", "DENIED000"} for _,g in new)
    for source, groups in ((s["id"], s["groups"]) for s in contract["sources"]):
        bad_batch = "batch5" if source == "replogle_k562" else "batch6"
        assert all(g["batch"] != bad_batch for g in groups)
    gate.validate_metadata(contract, cohort["metas"])


def test_same_source_batch_pools_disjoint_and_global(cohort):
    _, _, contract = registered(cohort)
    for s in contract["sources"]:
        pools = {}
        for g in s["groups"]:
            pool = {k:g[k] for k in ("control_rows", "context_control_rows", "sham_control_rows")}
            if g["batch"] in pools: assert pools[g["batch"]] == pool
            pools[g["batch"]] = pool
            assert [len(pool[k]) for k in pool] == [32,32,8]
            assert len(set().union(*map(set,pool.values()))) == 72


def test_resource_bound_accounts_for_all_cache_arrays(cohort):
    _, _, contract = registered(cohort)
    resource = contract["resource_estimate"]
    assert resource["raw_row_chunk_size"] == 256
    assert resource["cache_arrays_upper_bound_bytes"] + resource["archive_overhead_allowance_bytes"] < 512 << 20
    assert resource["extraction_ram_budget_bytes"] == 8 << 30
    assert resource["cache_artifact_budget_bytes"] == 2 << 30
    assert resource["groups"] == 22
    assert resource["population_entries"]["control"] == 22*32


@pytest.mark.parametrize("part", ["policy_bool", "cap", "row_bool", "row_order", "pool_overlap", "cross_target_pool", "source_order",
    "source_sha", "metadata_sha", "excluded", "gene_axis", "parent_mapping", "mapping_bool", "resource", "role", "extra", "expression"])
def test_contract_strict_rejection_before_expression(cohort, monkeypatch, part):
    out, _, contract = registered(cohort)
    manifest = gate.safe.read_json(out / "sources.json")
    source = contract["sources"][0]; group = source["groups"][0]
    if part == "policy_bool": contract["policy"]["new_targets_or_gene_axis"] = 0
    elif part == "cap": contract["policy"]["max_batches_per_target_source"] = 4
    elif part == "row_bool": group["control_rows"][0] = True
    elif part == "row_order": group["treated_rows"].reverse()
    elif part == "pool_overlap": group["control_rows"] = group["context_control_rows"]
    elif part == "cross_target_pool":
        other = next(g for g in source["groups"] if g["target"] != group["target"] and g["batch"] == group["batch"])
        other["control_rows"],other["context_control_rows"] = other["context_control_rows"],other["control_rows"]
    elif part == "source_order": contract["sources"].reverse()
    elif part == "source_sha": source["sha256"] = "f"*64
    elif part == "metadata_sha": source["metadata_sha256"] = "f"*64
    elif part == "excluded": contract["excluded_targets"].pop()
    elif part == "gene_axis": contract["output_genes"].reverse()
    elif part == "parent_mapping": contract["parent_group_mapping"][0]["expanded_group_id"] += 1
    elif part == "mapping_bool": contract["parent_group_mapping"][0]["parent_group_id"] = False
    elif part == "resource": contract["resource_estimate"]["cache_arrays_upper_bound_bytes"] = 0
    elif part == "role": source["role"] = "train"
    elif part == "extra": contract["extra"] = "unexpected"
    elif part == "expression": contract["raw_expression_read_during_registration"] = True
    forbid_expression(monkeypatch)
    with pytest.raises(ValueError): gate.validate_contract(contract, manifest, repo=cohort["root"])


@pytest.mark.parametrize("part", ["targets", "batches", "genes", "cells", "missing_source", "encoding"])
def test_changed_passed_metadata_reconstruction_rejected(cohort, monkeypatch, part):
    _, _, contract = registered(cohort)
    metas = copy.deepcopy(cohort["metas"])
    first = gate.SOURCE_IDS[0]
    if part == "missing_source": del metas[first]
    elif part == "encoding": metas[first]["encoding"] = "csr_matrix"
    else: metas[first][part][0] = "CHANGED"
    forbid_expression(monkeypatch)
    with pytest.raises(ValueError): gate.validate_metadata(contract, metas)


def test_metadata_rejects_alternative_unparent_treated_rows_even_when_typed(cohort):
    _, _, contract = registered(cohort)
    mapping = {m["expanded_group_id"] for m in contract["parent_group_mapping"]}
    roster = [(s,g) for s in contract["sources"] for g in s["groups"]]
    index = next(i for i in range(len(roster)) if i not in mapping)
    source, group = roster[index]
    meta = cohort["metas"][source["id"]]
    outsider = int(np.flatnonzero(meta["targets"] == "DENIED000")[0])
    group["treated_rows"] = sorted(group["treated_rows"][1:] + [outsider])
    with pytest.raises(ValueError, match="metadata-selected"): gate.validate_metadata(contract, cohort["metas"])


def test_independent_expanded_group_digest_checked(cohort, monkeypatch):
    monkeypatch.setattr(gate, "EXPECTED_GROUP_DIGESTS", {s:"0"*64 for s in gate.SOURCE_IDS})
    with pytest.raises(ValueError, match="audited expanded"): registered(cohort)
    assert not (cohort["root"] / "v7").exists()


def test_protocol_stage_does_not_claim_training_or_a_new_score(cohort):
    assert "no model fitting or evaluation" in gate.POLICY["validation"]
    assert "not a new score" in gate.POLICY["comparability"]


def test_original_metadata_source_size_checked_before_h5_open(cohort, monkeypatch):
    with Path(cohort["specs"][0]["path"]).open("ab") as stream: stream.write(b"changed")
    monkeypatch.setattr(h5py, "File", lambda *a, **k:pytest.fail("H5 opened after source-size mismatch"))
    with pytest.raises(ValueError, match="byte size"): registered(cohort)


def test_registration_does_not_create_outputs_if_source_metadata_changed(cohort, monkeypatch):
    spec = cohort["specs"][0]
    with h5py.File(spec["path"], "r+") as h: h["obs/target"][0] = b"CHANGED"
    forbid_expression(monkeypatch)
    with pytest.raises(ValueError): registered(cohort)
    assert not (cohort["root"] / "v7").exists()


def test_resource_overflow_rejected_before_allocation(cohort, monkeypatch):
    monkeypatch.setitem(gate.POLICY, "cache_archive_limit_bytes", 1)
    with pytest.raises(ValueError, match="512MiB"): registered(cohort)


def test_fresh_paths_and_protocol_pin(cohort):
    out, audit, _ = registered(cohort)
    with pytest.raises(FileExistsError): registered(cohort)
    cohort["protocol"].write_text("changed")
    with pytest.raises(ValueError): gate.load_contract(out/"contract.json",audit["contract_sha256"],out/"sources.json",audit["source_manifest_sha256"],repo=cohort["root"])


@pytest.mark.parametrize("part", ["complete", "audit", "targets"])
def test_partial_registration_refused(cohort, part):
    out, audit, _ = registered(cohort)
    filename = {"complete":"complete.json", "audit":"metadata_audit.json", "targets":"targets.json"}[part]
    (out / filename).rename(out / ("unavailable-" + filename))
    with pytest.raises(ValueError): gate.load_contract(out/"contract.json",audit["contract_sha256"],out/"sources.json",audit["source_manifest_sha256"],repo=cohort["root"])


@pytest.mark.parametrize("part", ["completed_bool", "contract_sha", "source_sha", "audit_rehashed", "targets_sha", "targets_bytes", "metadata_bool"])
def test_completion_and_rehashed_audit_bindings(cohort, part):
    out, original, _ = registered(cohort)
    complete = gate.safe.read_json(out/"complete.json")
    audit = gate.safe.read_json(out/"metadata_audit.json")
    if part == "completed_bool": complete["completed"] = 1
    elif part == "contract_sha": complete["contract_sha256"] = "0"*64
    elif part == "source_sha": complete["source_manifest_sha256"] = "0"*64
    elif part == "audit_rehashed": audit["parent_groups_preserved"] = 0
    elif part == "targets_sha": audit["targets_sha256"] = "0"*64
    elif part == "targets_bytes": save(out/"targets.json", ["FORGED"])
    elif part == "metadata_bool": audit["metadata_only"] = 1
    complete["metadata_audit_sha256"] = save(out/"metadata_audit.json", audit)
    save(out/"complete.json", complete)
    with pytest.raises(ValueError): gate.load_contract(out/"contract.json",original["contract_sha256"],out/"sources.json",original["source_manifest_sha256"],repo=cohort["root"])


@pytest.mark.parametrize("kind", ["fifo", "symlink"])
def test_nonregular_metadata_json_fails_without_blocking(cohort, kind):
    path = cohort["root"] / "bad.json"
    if kind == "fifo": os.mkfifo(path)
    else: path.symlink_to(cohort["parent_v2"]["contract"]["path"])
    with pytest.raises(ValueError): gate.load_contract(path,"a"*64,path,"b"*64,repo=cohort["root"])


def test_cli_rejects_login_before_work(cohort, monkeypatch):
    monkeypatch.setattr(gate.socket,"gethostname",lambda:"aida.cac.cornell.edu")
    monkeypatch.setattr(gate,"register",lambda *a:pytest.fail("Registration on head node"))
    with pytest.raises(SystemExit): gate.main(["--repo",str(cohort["root"]),"--protocol",str(cohort["protocol"]),"--output-dir",str(cohort["root"]/"blocked")])
