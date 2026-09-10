"""Synthetic-only sharded expanded-cohort execution, safety and replay tests."""
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import zipfile

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import public_expanded_v8 as v
from test_public_nested_v6 import fixture as nested_fixture

PROVENANCE = {"diagnostic_contract_sha256": "c" * 64, "role_manifest_sha256": "d" * 64}


def fixture():
    """Six batches per source/target, non-prefix IDs, shared controls, two genes."""
    base, roles, outer_tables, nested, inner_tables = nested_fixture()
    copies, groups = 3, len(base["targets"])
    arrays = {"genes": base["genes"].copy()}
    for key in ("targets", "sources", "context", "control", "treated", "context_control"):
        arrays[key] = np.concatenate([base[key].copy() for _ in range(copies)])
    arrays["batches"] = np.concatenate([np.asarray([str(b) + f"_rep{i}" for b in base["batches"]]) for i in range(copies)])
    for key in ("control_group", "group", "context_control_group"):
        arrays[key] = np.concatenate([base[key] + i * groups for i in range(copies)])
    for key in ("control_rows", "treated_rows", "context_control_rows"):
        arrays[key] = np.concatenate([base[key] + i * 10000000 for i in range(copies)])
    for fold in roles["folds"]:
        for row in fold["directions"]:
            for key in ("fit_group_ids", "joint_eval_group_ids", "held_target_group_ids_all_sources", "other_excluded_group_ids"):
                row[key] = [group + i * groups for i in range(copies) for group in row[key]]
    pools = []
    for i in range(copies):
        for old in nested["anchor_pools"]:
            pool = {"source": old["source"], "batch": old["batch"] + f"_rep{i}"}
            for key in ("fit_reference_rows", "tuning_anchor_rows", "context_rows", "sham_rows"):
                pool[key] = [row + i * 10000000 for row in old[key]]
            pools.append(pool)
    nested["anchor_pools"] = pools
    # Reverse semantic group order to ensure no code relies on old v2 prefix IDs.
    permutation = np.arange(len(arrays["targets"]))[::-1]
    inverse = np.argsort(permutation)
    for key in ("targets", "sources", "batches", "context"):
        arrays[key] = arrays[key][permutation]
    for key in ("control_group", "group", "context_control_group"):
        arrays[key] = inverse[arrays[key]]
    for fold in roles["folds"]:
        for row in fold["directions"]:
            for key in ("fit_group_ids", "joint_eval_group_ids", "held_target_group_ids_all_sources", "other_excluded_group_ids"):
                row[key] = sorted(int(inverse[group]) for group in row[key])
    return arrays, roles, outer_tables, nested, inner_tables


def execute(path, arm="true", expanded=False):
    args = fixture() if expanded else nested_fixture()
    summary = v.run_arm(*args, arm, path, PROVENANCE)
    return args, summary


def audit(path, args, summary):
    return v.audit_arm(path, summary, *args, PROVENANCE)


def persist_summary(path, summary):
    (path / "summary.json").write_text(v.previous.canonical(summary) + "\n")


def mutate_shard(path, summary, kind, transform, *, index=0):
    binding = summary["artifacts"][index][kind]
    target = path / binding["path"]
    with np.load(target, allow_pickle=False) as archive:
        values = {key: archive[key].copy() for key in archive.files}
    transform(values)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **values)
    payload = buffer.getvalue()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        expanded_bytes = sum(info.file_size for info in archive.infolist())
    summary["shard_compressed_bytes"] += len(payload) - binding["compressed_bytes"]
    summary["shard_expanded_bytes"] += expanded_bytes - binding["expanded_bytes"]
    binding.update(sha256=hashlib.sha256(payload).hexdigest(), compressed_bytes=len(payload),
                   expanded_bytes=expanded_bytes, array_bytes=sum(value.nbytes for value in values.values()))
    target.write_bytes(payload)
    persist_summary(path, summary)


def test_policy_wraps_science_without_changing_it():
    assert v.policy()["science"] == v.previous.policy()
    assert v.ARMS == v.previous.ARMS and v.GRID == (.1, 1., 10.)
    assert v.policy()["max_shard_compressed_bytes"] == 512 << 20
    assert v.policy()["max_output_bytes"] == 32 << 30
    assert v.policy()["automatic_promotion"] is False


@pytest.mark.parametrize("arm", v.ARMS)
def test_all_arms_expanded_six_batch_nonprefix_cohort_replay_without_refits(tmp_path, monkeypatch, arm):
    path = tmp_path / arm
    args, summary = execute(path, arm, expanded=True)
    assert len(args[0]["targets"]) == 48
    assert len({str(b) for b, s, t in zip(args[0]["batches"], args[0]["sources"], args[0]["targets"])
                if s == "J" and t == "A"}) == 6
    assert len(summary["artifacts"]) == (4 if arm == "control" else 16)
    assert summary["npz_shards"] == (8 if arm == "control" else 32)
    assert not (path / "weights.npz").exists()
    assert not (path / "predictions.npz").exists()
    assert len((path / "summary.json").read_bytes()) <= v.MAX_JSON_BYTES
    monkeypatch.setattr(v.previous.kernel, "fit_kernel_decoder", lambda *a, **k: pytest.fail("refit"))
    monkeypatch.setattr(np.linalg, "solve", lambda *a, **k: pytest.fail("ridge solve"))
    result = audit(path, args, summary)
    assert result["ridge_refits"] == 0
    assert result["out_of_fold_groups"] == 48
    assert result["out_of_fold_source_targets"] == 8
    assert result["fits_replayed"] == len(summary["artifacts"])
    assert result["npz_shards_verified"] == summary["npz_shards"]
    assert result["saved_head_replay"] and result["disjoint_inner_references_verified"]
    assert result["exact_artifact_roster_verified"] and result["output_budget_verified"]
    assert result["shard_headers_verified_before_decode"] and result["same_open_checksum_verified"]


@pytest.mark.parametrize("arm", ["control", "true", "shuffled_20260921"])
def test_same_arrays_match_frozen_unsharded_science_exactly(tmp_path, arm):
    args = nested_fixture()
    old = v.previous.run_arm(*args, arm, tmp_path / "old", PROVENANCE)
    new = v.run_arm(*args, arm, tmp_path / "new", PROVENANCE)
    for key in ("folds", "aggregate", "source_metrics"):
        assert v.previous.canonical(old[key]) == v.previous.canonical(new[key])
    assert audit(tmp_path / "new", args, new)["passed"]


@pytest.mark.parametrize("kind,key", [
    ("weights", "kernel_dual_coefficients"), ("weights", "prior_original_rows"),
    ("weights", "training_response"), ("weights", "kernel_regularization"),
    ("predictions", "nonnegative"), ("predictions", "control_parent_cache_rows"),
    ("predictions", "treated_parent_cache_rows"), ("predictions", "donor_original_rows")])
def test_rehashed_semantic_shard_tampering_rejected(tmp_path, kind, key):
    path = tmp_path / "run"
    args, summary = execute(path)
    def transform(values):
        values[key].flat[0] += 1
    mutate_shard(path, summary, kind, transform)
    with pytest.raises(ValueError):
        audit(path, args, summary)


@pytest.mark.parametrize("change", ["selected", "candidate_score", "candidate_order", "regularization", "phase",
    "fold_index", "prefix", "candidate_index", "binding_path", "binding_size_bool", "binding_size_wrong",
    "provenance", "policy", "flag", "new_field", "npz_shards", "shard_sum", "report_dtype"])
def test_rehashed_summary_role_and_selection_tampering_rejected(tmp_path, change):
    path = tmp_path / "run"
    args, summary = execute(path)
    if change == "selected": summary["folds"][0]["inner_selection"]["selected_regularization"] = 123.
    elif change == "candidate_score": summary["folds"][0]["inner_selection"]["candidates"][0]["metrics"]["target_pooled_mse"] += 1
    elif change == "candidate_order": summary["folds"][0]["inner_selection"]["candidates"].reverse()
    elif change == "regularization": summary["artifacts"][0]["regularization"] = 10.
    elif change == "phase": summary["artifacts"][0]["phase"] = "outer"
    elif change == "fold_index": summary["artifacts"][0]["fold_index"] = True
    elif change == "prefix": summary["artifacts"][0]["prefix"] = "fold_1_inner_0_"
    elif change == "candidate_index": summary["artifacts"][0]["candidate_index"] = 1
    elif change == "binding_path": summary["artifacts"][0]["weights"]["path"] = "../other.npz"
    elif change == "binding_size_bool": summary["artifacts"][0]["weights"]["compressed_bytes"] = True
    elif change == "binding_size_wrong": summary["artifacts"][0]["weights"]["expanded_bytes"] += 1
    elif change == "provenance": summary["provenance"]["role_manifest_sha256"] = "f" * 64
    elif change == "policy": summary["policy"]["science"]["regularization_grid"] = [10.]
    elif change == "flag": summary["challenge_treated_used"] = 0
    elif change == "new_field": summary["unknown"] = True
    elif change == "npz_shards": summary["npz_shards"] = True
    elif change == "shard_sum": summary["shard_compressed_bytes"] += 1
    elif change == "report_dtype": summary["folds"][0]["inner_selection"]["selected_regularization"] = True
    persist_summary(path, summary)
    with pytest.raises(ValueError):
        audit(path, args, summary)


@pytest.mark.parametrize("change", ["dtype", "genes", "extra", "missing"])
def test_rehashed_array_schema_tampering_rejected(tmp_path, change):
    path = tmp_path / "run"
    args, summary = execute(path)
    def transform(values):
        if change == "dtype": values["control_parent_cache_rows"] = values["control_parent_cache_rows"].astype(float)
        elif change == "genes": values["genes"] = values["genes"][::-1]
        elif change == "extra": values["extra"] = np.array([1])
        else: del values["group"]
    mutate_shard(path, summary, "predictions", transform)
    with pytest.raises(ValueError):
        audit(path, args, summary)


@pytest.mark.parametrize("part", ["missing", "extra", "symlink", "fifo", "steps", "summary", "unpersisted"])
def test_artifact_roster_and_regular_file_safety(tmp_path, part):
    path = tmp_path / "run"
    args, summary = execute(path, "control")
    target = path / summary["artifacts"][0]["weights"]["path"]
    if part == "missing": target.unlink()
    elif part == "extra": (path / "extra.npz").write_bytes(b"extra")
    elif part == "symlink":
        copy = tmp_path / "copy.npz"
        target.rename(copy)
        target.symlink_to(copy)
    elif part == "fifo":
        target.unlink()
        os.mkfifo(target)
    elif part == "steps":
        payload = b"{}\n"
        (path / "steps.jsonl").write_bytes(payload)
        summary["steps_sha256"] = hashlib.sha256(payload).hexdigest()
        persist_summary(path, summary)
    elif part == "summary": (path / "summary.json").write_text('{"schema": 1, "schema": 2}')
    else: summary["aggregate"]["target_pooled_mse"] += 1
    with pytest.raises(ValueError):
        audit(path, args, summary)


def test_shard_swap_with_rehashed_bindings_fails_semantic_replay(tmp_path):
    path = tmp_path / "run"
    args, summary = execute(path)
    first, second = summary["artifacts"][:2]
    for kind in ("weights", "predictions"):
        a, b = first[kind], second[kind]
        aa, bb = (path / a["path"]).read_bytes(), (path / b["path"]).read_bytes()
        (path / a["path"]).write_bytes(bb)
        (path / b["path"]).write_bytes(aa)
        first[kind] = {**b, "path": a["path"]}
        second[kind] = {**a, "path": b["path"]}
    persist_summary(path, summary)
    with pytest.raises(ValueError):
        audit(path, args, summary)


@pytest.mark.parametrize("change", ["last_path", "last_phase", "last_hash", "last_size_bool"])
def test_entire_roster_admitted_before_first_npz_decode(tmp_path, monkeypatch, change):
    path = tmp_path / "run"
    args, summary = execute(path, "control")
    row = summary["artifacts"][-1]
    if change == "last_path": row["predictions"]["path"] = "../outside.npz"
    elif change == "last_phase": row["phase"] = "inner"
    elif change == "last_hash": row["predictions"]["sha256"] = "z" * 64
    else: row["predictions"]["expanded_bytes"] = True
    persist_summary(path, summary)
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("Decoded before whole-roster admission"))
    with pytest.raises(ValueError):
        audit(path, args, summary)


def test_progress_has_each_fit_once_in_saved_order(tmp_path):
    args = nested_fixture()
    events = []
    summary = v.run_arm(*args, "true", tmp_path / "run", PROVENANCE, progress=events.append)
    saved = [json.loads(line) for line in (tmp_path / "run" / "steps.jsonl").read_text().splitlines()]
    assert saved == events and len(events) == 16
    for number in range(4):
        assert [e["phase"] for e in events[number*4:number*4+4]] == ["inner", "inner", "inner", "outer"]
        assert [e["regularization"] for e in events[number*4:number*4+3]] == list(v.GRID)
    assert audit(tmp_path / "run", args, summary)["passed"]


@pytest.mark.parametrize("budget", ["shard", "output", "json"])
def test_small_limits_fail_closed_without_completed_summary(tmp_path, monkeypatch, budget):
    args = nested_fixture()
    if budget == "shard": monkeypatch.setattr(v, "MAX_SHARD_BYTES", 100)
    elif budget == "output": monkeypatch.setattr(v, "MAX_OUTPUT_BYTES", 100)
    else: monkeypatch.setattr(v, "MAX_JSON_BYTES", 100)
    with pytest.raises(ValueError):
        v.run_arm(*args, "control", tmp_path / "run", PROVENANCE)
    assert not (tmp_path / "run" / "summary.json").exists()


@pytest.mark.parametrize("bad", [None, {}, {"diagnostic_contract_sha256": "x" * 64}, {"diagnostic_contract_sha256": True}])
def test_invalid_provenance_rejected_before_outputs(tmp_path, bad):
    with pytest.raises(ValueError):
        v.run_arm(*nested_fixture(), "true", tmp_path / "run", bad)
    assert not (tmp_path / "run").exists()


def test_existing_output_never_overwritten(tmp_path):
    path = tmp_path / "run"
    execute(path, "control")
    before = (path / "summary.json").read_bytes()
    with pytest.raises(FileExistsError):
        execute(path, "control")
    assert (path / "summary.json").read_bytes() == before


@pytest.mark.parametrize("bad", ["object", "duplicate", "header_shape", "compressed_budget", "expanded_budget"])
def test_unsafe_npz_rejected_before_decode(tmp_path, monkeypatch, bad):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        member = io.BytesIO()
        if bad == "header_shape":
            np.lib.format.write_array_header_1_0(member, {"shape": (1000000000,), "fortran_order": False, "descr": "<f8"})
        else:
            np.lib.format.write_array(member, np.array([object()], dtype=object) if bad == "object" else np.zeros(100), allow_pickle=True)
        archive.writestr("x.npy", member.getvalue())
        if bad == "duplicate":
            with pytest.warns(UserWarning): archive.writestr("x.npy", member.getvalue())
    payload = output.getvalue()
    path = tmp_path / "x.npz"
    path.write_bytes(payload)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        size = sum(info.file_size for info in archive.infolist())
    binding = {"path": path.name, "sha256": hashlib.sha256(payload).hexdigest(),
        "compressed_bytes": len(payload), "expanded_bytes": size, "array_bytes": 800}
    if bad == "compressed_budget": monkeypatch.setattr(v.cache_io, "MAX_ARCHIVE", len(payload)-1)
    if bad == "expanded_budget": monkeypatch.setattr(v.cache_io, "MAX_ARCHIVE", len(payload)+10)
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("unsafe decode"))
    with pytest.raises(ValueError): v._read_shard(tmp_path, binding, {"x"}, "x.npz")
