"""Synthetic metadata/role admission tests; no actual public expression read."""
import copy
from pathlib import Path
import sys

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_public_auxiliary_v9 as gate


def save(path, value):
    path.write_bytes(gate.old.canonical(value))
    return gate.safe.digest(path)


@pytest.fixture(params=["dense", "csr_matrix"])
def cohort(tmp_path, monkeypatch, request):
    targets = [f"T{i:02}" for i in range(56)]
    excluded = [f"X{i:03}" for i in range(772)]
    sources, specs, metas = [], [], {}
    for source_number, source in enumerate(gate.replicated.SOURCE_IDS):
        labels, batches = [], []
        for batch in ("0", "1"):
            for label, count in [("non-targeting", 72), *[(t, 8) for t in targets],
                    (f"AUX{source_number}", 12), (excluded[0], 8)]:
                labels.extend([label] * count)
                batches.extend([batch] * count)
        cells = [f"CELL{i:05}" for i in range(len(labels))]
        meta = {"cells": np.array(cells), "targets": np.array(labels), "batches": np.array(batches),
            "genes": np.array(["G0", "G1", "G2"]), "encoding": request.param}
        path = tmp_path / (source + ".h5ad")
        with h5py.File(path, "w") as handle:
            obs, var = handle.create_group("obs"), handle.create_group("var")
            for key in ("cells", "targets", "batches"):
                obs.create_dataset(key, data=np.asarray(meta[key], dtype=h5py.string_dtype()))
            var.create_dataset("genes", data=np.asarray(meta["genes"], dtype=h5py.string_dtype()))
            if request.param == "dense":
                handle.create_dataset("X", shape=(len(cells), 3), dtype="f4")
            else:
                x = handle.create_group("X")
                x.attrs["shape"] = (len(cells), 3)
                x.attrs["encoding-type"] = "csr_matrix"
                x.create_dataset("data", data=np.array([], dtype="f4"))
                x.create_dataset("indices", data=np.array([], dtype="i4"))
                x.create_dataset("indptr", data=np.zeros(len(cells) + 1, dtype="i4"))
        spec = {"id": source, "path": str(path), "sha256": gate.safe.digest(path),
            "size_bytes": path.stat().st_size, "cell_key": "cells", "target_key": "targets",
            "batch_key": "batches", "gene_key": "genes", "control_label": "non-targeting"}
        groups = []
        for target in targets:
            for batch in ("0", "1"):
                candidates = np.flatnonzero((meta["targets"] == "non-targeting") & (meta["batches"] == batch))
                ranked = gate.replicated.v2.ranked_rows(meta, candidates, source, batch)
                groups.append({"target": target, "batch": batch, "context_control_rows": sorted(ranked[:32]),
                    "control_rows": sorted(ranked[32:64]), "sham_control_rows": sorted(ranked[64:72]),
                    "treated_rows": np.flatnonzero((meta["targets"] == target) & (meta["batches"] == batch)).tolist()})
        sources.append({**spec, "metadata_sha256": gate.old.metadata_digest(meta),
            "cells_total": len(cells), "genes_total": 3, "groups": groups})
        specs.append(spec)
        metas[source] = meta
    roster = gate.replicated.group_roster(sources)
    roles = {"targets": targets, "excluded_targets": excluded, "group_roster": roster,
        "folds": gate.expanded.outer_preparation.make_folds(targets, roster)}
    row = {"common_targets": targets, "excluded_targets": excluded, "sources": sources,
        "shared_genes": ["G0", "G1", "G2"], "output_genes": ["G0", "G1"], "group_roster": roster}
    nested = {"targets": targets, "excluded_targets": excluded,
        "inner_folds": gate.expanded.nested_preparation.make_inner_folds(roles),
        "anchor_pools": gate.expanded.nested_preparation.make_anchor_pools(row, roles)}
    parent = ({"v81_execution": {"path": str(tmp_path / "parent.json"), "sha256": "a" * 64}},
        row, {"sources": specs}, roles, nested)
    monkeypatch.setattr(gate, "_parents", lambda repo: copy.deepcopy(parent))
    monkeypatch.setattr(gate, "_codes", lambda: {"synthetic.py": "b" * 64})
    monkeypatch.setattr(gate, "EXPECTED_TARGETS", {s: 1 for s in gate.replicated.SOURCE_IDS})
    monkeypatch.setattr(gate, "EXPECTED_GROUPS", {s: 2 for s in gate.replicated.SOURCE_IDS})
    monkeypatch.setattr(gate, "EXPECTED_TREATED", {s: 24 for s in gate.replicated.SOURCE_IDS})
    monkeypatch.setattr(gate, "EXPECTED_POOLS", {s: 2 for s in gate.replicated.SOURCE_IDS})
    protocol = tmp_path / "protocol.md"
    protocol.write_text("Synthetic candidate metadata only; no data/model authorization.")
    return {"root": tmp_path, "protocol": protocol, "parent": parent, "metas": metas}


def register(cohort):
    out = cohort["root"] / "registration"
    result = gate.register(cohort["root"], out, cohort["protocol"])
    record = gate.load_registration(out / "candidate_plan.json", result["candidate_plan_sha256"], repo=cohort["root"])
    return out, result, record


def test_registration_never_reads_X_or_decodes_parent_cache(cohort, monkeypatch):
    actual = h5py.Dataset.__getitem__
    def guarded(self, key):
        if self.name == "/X" or self.name.startswith("/X/"):
            pytest.fail("Expression access during metadata-only registration")
        return actual(self, key)
    monkeypatch.setattr(h5py.Dataset, "__getitem__", guarded)
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("Parent cache decoded"))
    out, result, record = register(cohort)
    assert result["candidate_targets"] == 2 and result["candidate_groups"] == 4
    assert result["candidate_treated_rows"] == 48
    assert record["training_authorized"] is record["expression_materialization_authorized"] is False
    assert not record["full_source_hash_performed_now"]
    assert sorted(p.name for p in out.iterdir()) == ["candidate_plan.json", "complete.json"]


def test_candidates_preserve_only_fit16_context32_and_source_routing(cohort):
    _, _, record = register(cohort)
    frozen = {(p["source"], p["batch"]): p for p in cohort["parent"][4]["anchor_pools"]}
    for source, direction in zip(record["sources"], record["source_routing"]):
        assert direction["fit_source"] == source["id"] != direction["held_source"]
        assert len(direction["clone_per_outer_fold"]) == 4
        assert set(source["targets"]).isdisjoint(record["panel_targets"] + record["protected_excluded_targets"])
        assert direction["candidate_group_ids"] == [g["group_id"] for g in source["groups"]]
        for pool in source["control_pools"]:
            p = frozen[source["id"], pool["batch"]]
            assert set(pool) == {"pool_id", "source", "batch", "fit_reference_rows", "context_rows"}
            assert pool["fit_reference_rows"] == p["fit_reference_rows"] and len(pool["fit_reference_rows"]) == 16
            assert pool["context_rows"] == p["context_rows"] and len(pool["context_rows"]) == 32
            assert set(pool["fit_reference_rows"] + pool["context_rows"]).isdisjoint(p["tuning_anchor_rows"] + p["sham_rows"])
        assert record["capacity"][source["id"]]["unique_candidate_rows"] == 120


def test_semantic_ids_have_no_global_roster_dependence(cohort):
    _, _, record = register(cohort)
    for source in record["sources"]:
        for g in source["groups"]:
            assert g["group_id"] == gate.identity("group", source["id"], g["target"], g["batch"])
            assert g["pool_id"] == gate.identity("pool", source["id"], g["batch"])
            assert g["group_id"] != gate.identity("group", record["source_routing"][0]["held_source"]
                if source["id"] == record["source_routing"][0]["fit_source"] else record["source_routing"][0]["fit_source"],
                g["target"], g["batch"])
    assert gate.identity("group", "replogle_k562", "A|B", "C") != gate.identity("group", "replogle_k562", "A", "B|C")


def test_whole_candidate_plan_is_unchanged_by_panel_group_reordering(cohort):
    _, row, manifest, roles, nested = copy.deepcopy(cohort["parent"])
    expected = gate.select_candidates(cohort["metas"], row, manifest, roles, nested)
    for source in row["sources"]:
        source["groups"].reverse()
    roster = gate.replicated.group_roster(row["sources"])
    row["group_roster"] = roster
    roles["group_roster"] = roster
    roles["folds"] = gate.expanded.outer_preparation.make_folds(row["common_targets"], roster)
    nested["inner_folds"] = gate.expanded.nested_preparation.make_inner_folds(roles)
    nested["anchor_pools"] = gate.expanded.nested_preparation.make_anchor_pools(row, roles)
    actual = gate.select_candidates(cohort["metas"], row, manifest, roles, nested)
    assert expected == actual


def test_changed_implementation_hash_rejected(cohort, monkeypatch):
    out, result, _ = register(cohort)
    monkeypatch.setattr(gate, "_codes", lambda: {"synthetic.py": "c" * 64})
    with pytest.raises(ValueError, match="complete candidate registration replay"):
        gate.load_registration(out / "candidate_plan.json", result["candidate_plan_sha256"], repo=cohort["root"])


@pytest.mark.parametrize("change", ["treated_to_context", "tuning_reference", "add_sham", "panel_target", "protected_target",
    "source_routing", "source_id", "group_id", "metadata_sha", "training", "materialization", "policy", "extra_field", "gene_axis"])
def test_resealed_tampering_fails_closed(cohort, change):
    out, _, record = register(cohort)
    s, pool, g = record["sources"][0], record["sources"][0]["control_pools"][0], record["sources"][0]["groups"][0]
    frozen = next(p for p in cohort["parent"][4]["anchor_pools"] if p["source"] == s["id"] and p["batch"] == pool["batch"])
    if change == "treated_to_context": g["treated_rows"][0] = pool["context_rows"][0]
    elif change == "tuning_reference": pool["fit_reference_rows"] = frozen["tuning_anchor_rows"]
    elif change == "add_sham": pool["sham_rows"] = frozen["sham_rows"]
    elif change == "panel_target": g["target"] = record["panel_targets"][0]
    elif change == "protected_target": g["target"] = record["protected_excluded_targets"][0]
    elif change == "source_routing": record["source_routing"][0]["held_source"] = s["id"]
    elif change == "source_id": g["source"] = "h1"
    elif change == "group_id": g["group_id"] = 0
    elif change == "metadata_sha": s["metadata_sha256"] = "0" * 64
    elif change == "training": record["training_authorized"] = True
    elif change == "materialization": record["expression_materialization_authorized"] = True
    elif change == "policy": record["policy"]["max_treated_per_group"] = 64
    elif change == "extra_field": record["pretrained"] = True
    elif change == "gene_axis": record["output_genes"].reverse()
    sha = save(out / "candidate_plan.json", record)
    save(out / "complete.json", gate._completion(sha, record))
    with pytest.raises(ValueError):
        gate.load_registration(out / "candidate_plan.json", sha, repo=cohort["root"])


@pytest.mark.parametrize("change", ["annotation", "dimensions", "controls", "panel", "source_order", "source_identity"])
def test_parent_or_metadata_change_rejected(cohort, change):
    bindings, row, manifest, roles, nested = cohort["parent"]
    if change == "annotation": cohort["metas"]["replogle_k562"]["targets"][0] = "other"
    elif change == "dimensions": row["sources"][0]["cells_total"] += 1
    elif change == "controls": nested["anchor_pools"][0]["fit_reference_rows"] = nested["anchor_pools"][0]["tuning_anchor_rows"]
    elif change == "panel": roles["targets"][0] = "replaced"
    elif change == "source_order": manifest["sources"].reverse()
    elif change == "source_identity": manifest["sources"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        gate.select_candidates(cohort["metas"], row, manifest, roles, nested)


def test_source_size_changed_fails_before_publication(cohort):
    path = Path(cohort["parent"][2]["sources"][0]["path"])
    with path.open("ab") as stream:
        stream.write(b"changed")
    out = cohort["root"] / "registration"
    with pytest.raises(ValueError, match="byte size"):
        gate.register(cohort["root"], out, cohort["protocol"])
    assert not out.exists()


def test_protocol_and_completion_tamper_rejected(cohort):
    out, result, record = register(cohort)
    save(out / "complete.json", {**gate._completion(result["candidate_plan_sha256"], record), "training_authorized": True})
    with pytest.raises(ValueError, match="completion"):
        gate.load_registration(out / "candidate_plan.json", result["candidate_plan_sha256"], repo=cohort["root"])
    save(out / "complete.json", gate._completion(result["candidate_plan_sha256"], record))
    cohort["protocol"].write_text("changed")
    with pytest.raises(ValueError):
        gate.load_registration(out / "candidate_plan.json", result["candidate_plan_sha256"], repo=cohort["root"])


def test_fresh_only_and_no_symlink_reads(cohort):
    out, result, _ = register(cohort)
    with pytest.raises(FileExistsError):
        gate.register(cohort["root"], out, cohort["protocol"])
    link = cohort["root"] / "candidate_plan.json"
    link.symlink_to(out / "candidate_plan.json")
    with pytest.raises(ValueError, match="Nonsymlink"):
        gate.load_registration(link, result["candidate_plan_sha256"], repo=cohort["root"])


def test_budget_failure_never_publishes_completion(cohort, monkeypatch):
    monkeypatch.setattr(gate, "MAX_JSON_BYTES", 20)
    out = cohort["root"] / "registration"
    with pytest.raises(ValueError, match="Oversized"):
        gate.register(cohort["root"], out, cohort["protocol"])
    assert not (out / "complete.json").exists()


@pytest.mark.parametrize("limit", ["MAX_RSS", "MAX_SECONDS"])
def test_runtime_checkpoint_failure_never_publishes_candidates(cohort, monkeypatch, limit):
    monkeypatch.setattr(gate, limit, -1)
    out = cohort["root"] / "registration"
    with pytest.raises(ValueError, match="exceeded"):
        gate.register(cohort["root"], out, cohort["protocol"])
    assert not out.exists()


@pytest.mark.parametrize("parts", [("group", "h1", "T", "B"), ("group", "replogle_k562", "T"),
    ("pool", "replogle_k562", " T"), ("pool", "replogle_k562", ""), ("group", "replogle_k562", True, "B")])
def test_invalid_semantic_identity_rejected(parts):
    with pytest.raises(ValueError): gate.identity(*parts)
