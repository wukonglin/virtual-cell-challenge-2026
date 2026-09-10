"""Synthetic expanded roles; no real expression or outcomes are decoded."""
import copy
import os
from pathlib import Path
import sys

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_public_expanded_v8 as gate
import public_joint_target_source_v3 as joint
import public_nested_v6 as model


def save(path, value):
    path.write_bytes(gate.replicated.json_bytes(value))
    return gate.safe.digest(path)


def feature_binding(root, key):
    return {"feature_fold_id": key,
        **{name + suffix: str(root / key / (name + ".npz")) if suffix == "_path" else "a" * 64
           for name in ("npz", "receipt", "coverage") for suffix in ("_path", "_sha256")}}


@pytest.fixture
def cohort(tmp_path, monkeypatch):
    targets = [f"T{i:02}" for i in range(56)]
    excluded = [f"EXCLUDED{i:03}" for i in range(772)]
    sources = []
    for source in gate.replicated.SOURCE_IDS:
        groups = []
        for i, target in enumerate(targets):
            for batch in ("1", "0", "2"):  # Expansion is deliberately not a prefix.
                offset = int(batch) * 10000
                groups.append({"target": target, "batch": batch,
                    "control_rows": list(range(offset, offset + 32)),
                    "context_control_rows": list(range(offset + 32, offset + 64)),
                    "sham_control_rows": list(range(offset + 64, offset + 72)),
                    "treated_rows": list(range(offset + 100 + i * 8, offset + 108 + i * 8))})
        sources.append({"id": source, "cells_total": 30000, "groups": groups})
    parent_sources = [{**s, "groups": [g for g in s["groups"] if g["batch"] != "2"]} for s in sources]
    parent_roster = gate.replicated.group_roster(parent_sources)
    parent_outer = {"targets": targets, "excluded_targets": excluded,
        "group_roster": parent_roster, "folds": gate.outer_preparation.make_folds(targets, parent_roster),
        "inputs": {"cache_sha256": "b" * 64}, "input_paths": {"cache": str(tmp_path / "old.npz")},
        "go_source_sha256": dict(gate.outer_preparation.GO_EXPECTED),
        "go_acquisition_path": str(tmp_path / "public-go.json")}
    for fold in parent_outer["folds"]:
        fold["go"] = feature_binding(tmp_path, "outer_" + fold["fold_id"])
    parent_inner = {"targets": targets, "excluded_targets": excluded,
        "inner_folds": gate.nested_preparation.make_inner_folds(parent_outer),
        "anchor_pools": gate.nested_preparation.make_anchor_pools({"sources": parent_sources}, parent_outer)}
    for key, fold in parent_inner["inner_folds"].items():
        fold["go"] = feature_binding(tmp_path, "inner_" + key)
    roster = gate.replicated.group_roster(sources)
    lookup = {(g["source"], g["target"], g["batch"]): g["group_id"] for g in roster}
    row = {"common_targets": targets, "excluded_targets": excluded, "sources": sources, "group_roster": roster,
        "parent_group_mapping": [{"parent_group_id": old["group_id"],
            "expanded_group_id": lookup[old["source"], old["target"], old["batch"]]} for old in parent_roster]}
    bindings = {"v7": {key: {"path": str(tmp_path / filename), "sha256": str(i) * 64}
        for i, (key, filename) in enumerate((("cache", "cache.npz"), ("contract", "contract.json"),
            ("source_manifest", "sources.json"), ("cache_receipt", "cache.npz.receipt.json")))},
        "v6_execution": {"path": str(tmp_path / "v6.json"), "sha256": gate.V6_SHA},
        "v6_nested_registration": {"path": str(tmp_path / "v6roles.json"), "sha256": gate.V6_ROLE_SHA},
        "v3_outer_registration": {"path": str(tmp_path / "v3roles.json"), "sha256": gate.V3_ROLE_SHA}}
    monkeypatch.setattr(gate, "EXPECTED_GROUPS", {s["id"]: len(s["groups"]) for s in sources})
    monkeypatch.setattr(gate, "EXPECTED_POOLS", 6)
    monkeypatch.setattr(gate, "EXPECTED_PARENT_POOLS", 4)
    original_parents = gate._parents
    def parents():
        return copy.deepcopy((bindings, row, parent_outer, parent_inner))
    monkeypatch.setattr(gate, "_parents", parents)
    protocol = tmp_path / "protocol.md"
    protocol.write_text("Synthetic registered expanded public validation; no expression.")
    return {"root": tmp_path, "protocol": protocol, "row": row, "bindings": bindings,
        "outer": parent_outer, "inner": parent_inner, "parents": parents,
        "original_parents": original_parents}


def registered(cohort):
    out = cohort["root"] / "registration"
    sha = gate.register(cohort["protocol"], out)
    record, roles, nested = gate.validate_registration(out / "role_manifest.json", sha)
    return out, sha, record, roles, nested


def reseal(out, record):
    sha = save(out / "role_manifest.json", record)
    complete = gate.safe.read_json(out / "complete.json")
    complete["role_manifest_sha256"] = sha
    save(out / "complete.json", complete)
    return sha


def test_expanded_outer_and_inner_roles_are_compatible(cohort):
    _, _, record, roles, nested = registered(cohort)
    assert len(roles["group_roster"]) == 336 and len(nested["anchor_pools"]) == 6
    assert record["inputs"] == roles["inputs"] == nested["inputs"]
    assert record["input_paths"] == roles["input_paths"] == nested["input_paths"]
    assert record["inputs"] != cohort["outer"]["inputs"]
    axes = {"targets": np.array([g["target"] for g in roles["group_roster"]]),
        "sources": np.array([g["source"] for g in roles["group_roster"]])}
    folds = joint.joint_folds(tuple(axes["targets"]), tuple(axes["sources"]), roles)
    assert len(folds) == 8
    for outer in folds:
        inner = model.inner_fold(axes, outer, nested)
        assert len(outer.fit_targets) == 42 and len(outer.held_targets) == 14
        assert len(inner.fit_targets) == 28 and len(inner.held_targets) == 14
        assert inner.fit_source == inner.held_source == outer.fit_source
        assert not set(inner.fit_indices) & set(inner.evaluation_indices)
        info = nested["inner_folds"][outer.target_fold]
        direction = next(d for d in info["directions"] if d["direction_id"] == outer.fold_id)
        assert list(inner.fit_indices) == direction["fit_group_ids"]
        assert list(inner.evaluation_indices) == direction["tuning_group_ids"]
        assert set(inner.fit_indices + inner.evaluation_indices) == set(outer.fit_indices)


def test_original_reference_pools_preserved_and_new_pools_global(cohort):
    _, _, _, roles, nested = registered(cohort)
    pools = {(p["source"], p["batch"]): p for p in nested["anchor_pools"]}
    for parent in cohort["inner"]["anchor_pools"]:
        assert pools[parent["source"], parent["batch"]] == parent
    for source in cohort["row"]["sources"]:
        for group in source["groups"]:
            p = pools[source["id"], group["batch"]]
            fr, tune = p["fit_reference_rows"], p["tuning_anchor_rows"]
            assert len(fr) == len(tune) == 16 and not set(fr) & set(tune)
            assert sorted(fr + tune) == group["control_rows"]
            ranked = sorted(group["control_rows"], key=lambda row: (
                gate.nested_preparation._rank("nested-anchor-v6", source["id"], group["batch"], row), row))
            assert fr == sorted(ranked[:16])
    # Flat group IDs shifted by expansion, but original source/row roles did not.
    assert any(m["parent_group_id"] != m["expanded_group_id"] for m in cohort["row"]["parent_group_mapping"])
    assert roles["excluded_targets"] == cohort["outer"]["excluded_targets"]


def test_inner_view_uses_expanded_global_original_row_roles(cohort):
    _, _, _, roles, nested = registered(cohort)
    roster = roles["group_roster"]
    arrays = {"targets": np.array([g["target"] for g in roster]),
        "sources": np.array([g["source"] for g in roster]),
        "batches": np.array([g["batch"] for g in roster]), "genes": np.array(["G0", "G1"]),
        "context": np.zeros((len(roster), 4), dtype=np.float32)}
    all_groups = [g for s in cohort["row"]["sources"] for g in s["groups"]]
    for field, row_field, label in (("control", "control_rows", "control_group"),
        ("treated", "treated_rows", "group"), ("context_control", "context_control_rows", "context_control_group")):
        pairs = [(i, r) for i, g in enumerate(all_groups) for r in g[row_field]]
        arrays[row_field] = np.array([r for _, r in pairs], dtype=np.int64)
        arrays[label] = np.array([i for i, _ in pairs], dtype=np.int64)
        arrays[field] = np.array([[r / 30000, (r + 1) / 30000] for _, r in pairs], dtype=np.float32)
    folds = joint.joint_folds(tuple(arrays["targets"]), tuple(arrays["sources"]), roles)
    for outer in folds:
        inner = model.inner_fold(arrays, outer, nested)
        view = model.inner_view(arrays, inner, nested)
        fit_mask = np.isin(view["control_group"], inner.fit_indices)
        tune_mask = np.isin(view["control_group"], inner.evaluation_indices)
        fit_rows = set(view["control_rows"][fit_mask].tolist())
        tune_rows = set(view["control_rows"][tune_mask].tolist())
        assert not fit_rows & tune_rows
        assert len(fit_rows) == len(tune_rows) == 3 * 16
        for group in (*inner.fit_indices, *inner.evaluation_indices):
            assert np.sum(view["control_group"] == group) == 16
        assert np.array_equal(view["control"], arrays["control"][view["_control_parent_cache_rows"]])
        assert not view["control"].flags.writeable


def test_registration_and_validation_never_decode_npz_or_raw_sources(cohort, monkeypatch):
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("NPZ decoded during registration"))
    monkeypatch.setattr(h5py, "File", lambda *a, **k: pytest.fail("Raw HDF5 opened during registration"))
    monkeypatch.setattr(gate.outer_preparation, "_go_index", lambda *a, **k: pytest.fail("GO vocabulary rebuilt"))
    _, _, record, _, _ = registered(cohort)
    assert record["expression_decoded"] is False
    assert record["new_go_vocabulary_fitted"] is False
    assert record["outer_go_reused"] is record["inner_go_reused"] is True


def test_feature_loader_uses_reauthenticated_original_parent_roles(cohort, monkeypatch):
    _, _, record, _, _ = registered(cohort)
    called = []
    def outer(value):
        assert value == cohort["outer"]
        called.append("outer")
        return "OUTER"
    def inner(value):
        assert value == cohort["inner"]
        called.append("inner")
        return "INNER"
    monkeypatch.setattr(gate.outer_preparation, "load_feature_tables", outer)
    monkeypatch.setattr(gate.nested_preparation, "load_feature_tables", inner)
    assert gate.load_feature_tables(record) == ("OUTER", "INNER")
    assert called == ["outer", "inner"]
    record["nested"]["inner_folds"]["target_fold_0"]["go"]["npz_sha256"] = "f" * 64
    with pytest.raises(ValueError):
        gate.load_feature_tables(record)
    assert called == ["outer", "inner"]


@pytest.mark.parametrize("part", ["source", "count", "target", "excluded", "roster_bool", "duplicate",
    "outer_target", "outer_group", "inner_target", "inner_group", "pool_role", "pool_bool", "pool_count",
    "map_bool", "map_group", "map_count"])
def test_malformed_parent_metadata_fails_before_registering(cohort, part):
    row, outer, inner = cohort["row"], cohort["outer"], cohort["inner"]
    if part == "source": row["sources"][0]["id"] = "feng"
    elif part == "count": row["sources"][0]["groups"].pop()
    elif part == "target": row["common_targets"][0] = "UNREGISTERED"
    elif part == "excluded": row["excluded_targets"].pop()
    elif part == "roster_bool": row["group_roster"][0]["group_id"] = False
    elif part == "duplicate": row["sources"][0]["groups"][1] = copy.deepcopy(row["sources"][0]["groups"][0])
    elif part == "outer_target": outer["folds"][0]["fit_targets"][0] = outer["folds"][0]["held_targets"][0]
    elif part == "outer_group": outer["folds"][0]["directions"][0]["fit_group_ids"][0] = False
    elif part == "inner_target": inner["inner_folds"]["target_fold_0"]["fit_targets"].reverse()
    elif part == "inner_group": inner["inner_folds"]["target_fold_0"]["directions"][0]["fit_group_ids"].pop()
    elif part == "pool_role": inner["anchor_pools"][0]["fit_reference_rows"][0] = inner["anchor_pools"][0]["tuning_anchor_rows"][0]
    elif part == "pool_bool": row["sources"][0]["groups"][0]["control_rows"][0] = False
    elif part == "pool_count": inner["anchor_pools"].pop()
    elif part == "map_bool": row["parent_group_mapping"][0]["parent_group_id"] = False
    elif part == "map_group": row["parent_group_mapping"][0]["expanded_group_id"] = 2
    elif part == "map_count": row["parent_group_mapping"].pop()
    with pytest.raises((ValueError, KeyError)):
        gate.register(cohort["protocol"], cohort["root"] / "registration")
    assert not (cohort["root"] / "registration").exists()


@pytest.mark.parametrize("part", ["schema", "extra", "policy", "policy_bool", "parents", "input", "path",
    "outer_go", "inner_go", "outer_group", "inner_group", "target", "anchor", "flag", "code"])
def test_coherently_rehashed_registration_mutations_rejected(cohort, part):
    out, _, record, _, _ = registered(cohort)
    if part == "schema": record["schema"] = "legacy"
    elif part == "extra": record["new"] = True
    elif part == "policy": record["policy"]["cpu_threads_max"] = 1024
    elif part == "policy_bool": record["policy"]["raw_sources_opened"] = 0
    elif part == "parents": record["parents"]["v7"]["cache_receipt"]["sha256"] = "f" * 64
    elif part == "input": record["inputs"]["cache_sha256"] = "f" * 64
    elif part == "path": record["input_paths"]["cache"] = str(cohort["root"] / "other.npz")
    elif part == "outer_go": record["roles"]["folds"][0]["go"]["feature_fold_id"] = "all56"
    elif part == "inner_go": record["nested"]["inner_folds"]["target_fold_0"]["go"]["npz_sha256"] = "f" * 64
    elif part == "outer_group": record["roles"]["folds"][0]["directions"][0]["fit_group_ids"][0] = False
    elif part == "inner_group": record["nested"]["inner_folds"]["target_fold_0"]["directions"][0]["fit_group_ids"].pop()
    elif part == "target": record["nested"]["inner_folds"]["target_fold_0"]["fit_targets"].reverse()
    elif part == "anchor": record["nested"]["anchor_pools"][0]["fit_reference_rows"].reverse()
    elif part == "flag": record["expression_decoded"] = 0
    elif part == "code": record["code_sha256"]["prepare_public_expanded_v8.py"] = "f" * 64
    with pytest.raises(ValueError):
        gate.validate_registration(out / "role_manifest.json", reseal(out, record))


@pytest.mark.parametrize("part", ["missing", "false", "bool_int", "sha", "groups", "extra"])
def test_completion_fails_closed(cohort, part):
    out, sha, _, _, _ = registered(cohort)
    path = out / "complete.json"
    value = gate.safe.read_json(path)
    if part == "missing": path.unlink()
    else:
        if part == "false": value["completed"] = False
        elif part == "bool_int": value["expression_decoded"] = 0
        elif part == "sha": value["role_manifest_sha256"] = "f" * 64
        elif part == "groups": value["groups"] += 1
        elif part == "extra": value["new"] = True
        save(path, value)
    with pytest.raises(ValueError):
        gate.validate_registration(out / "role_manifest.json", sha)


def test_fresh_directory_and_regular_leaves_required(cohort):
    out, sha, _, _, _ = registered(cohort)
    with pytest.raises(FileExistsError): gate.register(cohort["protocol"], out)
    alias = cohort["root"] / "alias"
    alias.symlink_to(out, target_is_directory=True)
    with pytest.raises(ValueError): gate.register(cohort["protocol"], alias)
    protocol_alias = cohort["root"] / "alias.md"
    protocol_alias.symlink_to(cohort["protocol"])
    with pytest.raises(ValueError): gate.register(protocol_alias, cohort["root"] / "other")
    leaf = cohort["root"] / "role_manifest.json"
    leaf.symlink_to(out / "role_manifest.json")
    with pytest.raises(ValueError): gate.validate_registration(leaf, sha)
    leaf.unlink()
    os.mkfifo(leaf)
    with pytest.raises(ValueError): gate.validate_registration(leaf, sha)


def test_changed_metadata_during_registration_leaves_no_completion(cohort, monkeypatch):
    original = gate._parents
    calls = []
    def changing():
        result = original()
        calls.append(True)
        if len(calls) > 1:
            result[0]["v7"]["cache"]["sha256"] = "f" * 64
        return result
    monkeypatch.setattr(gate, "_parents", changing)
    out = cohort["root"] / "registration"
    with pytest.raises(ValueError): gate.register(cohort["protocol"], out)
    assert (out / "role_manifest.json").is_file()
    assert not (out / "complete.json").exists()


@pytest.mark.parametrize("part", ["valid", "cache", "receipt", "receipt_bool", "receipt_parent", "v6role", "v3role"])
def test_parent_authentication_requires_exact_cache_and_parent_bindings(cohort, monkeypatch, part):
    root = cohort["root"]
    cache = root / "cache.npz"
    cache.write_bytes(b"OPAQUE-NOT-NPZ")
    bindings = cohort["bindings"]["v7"]
    bindings["cache"]["sha256"] = gate.safe.digest(cache)
    receipt = {"schema": "public-replicated-control-anchor-cache-receipt-v7", "completed": True,
        "groups": 336, "stage": "data_preparation", "cache_path": str(cache),
        **{key + "_sha256": bindings[key]["sha256"] for key in ("cache", "contract", "source_manifest")},
        "model_fitting_performed": False, "posttraining_performed": False, "submission_performed": False,
        "rpe1_h1_hepg2_challenge2026_feng_opened": False}
    receipt_path = root / "cache.npz.receipt.json"
    if part == "receipt_bool": receipt["completed"] = 1
    elif part == "receipt_parent": receipt["contract_sha256"] = "f" * 64
    bindings["cache_receipt"]["sha256"] = save(receipt_path, receipt)
    monkeypatch.setattr(gate, "PINS", {key: (Path(value["path"]).name, value["sha256"]) for key, value in bindings.items()})
    monkeypatch.setattr(gate, "REPO", root)
    monkeypatch.setattr(gate, "V7_RELATIVE", ".")
    monkeypatch.setattr(gate.replicated, "load_contract", lambda *a, **k: cohort["row"])
    execution = {"nested_registration": cohort["bindings"]["v6_nested_registration"],
        "registration": cohort["bindings"]["v3_outer_registration"]}
    if part == "v6role": execution["nested_registration"]["sha256"] = "f" * 64
    elif part == "v3role": execution["registration"]["sha256"] = "f" * 64
    monkeypatch.setattr(gate.previous, "validate_contract", lambda *a, **k: (execution, cohort["outer"], cohort["inner"]))
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("Expression NPZ decoded"))
    if part == "cache": cache.write_bytes(b"OTHER")
    elif part == "receipt": receipt_path.write_text("{}")
    if part == "valid":
        _, row, outer, inner = cohort["original_parents"]()
        assert row == cohort["row"] and outer == cohort["outer"] and inner == cohort["inner"]
    else:
        with pytest.raises(ValueError): cohort["original_parents"]()


def test_registered_resource_bounds_and_no_promotion():
    policy = gate.policy()
    assert policy["cpu_threads_max"] == 8
    assert policy["peak_rss_limit_bytes"] == 16 << 30
    assert policy["artifact_limit_bytes"] == 32 << 30
    assert policy["per_npz_limit_bytes"] == 512 << 20
    assert policy["automatic_promotion"] is policy["submission_performed"] is False
