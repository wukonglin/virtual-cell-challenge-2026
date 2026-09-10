"""Synthetic nested target/reference roles; no cell-expression decoding."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_public_nested_v6 as gate


def save(path, value):
    path.write_bytes(gate.go._json_bytes(value))
    return gate.evidence.digest(path)


@pytest.fixture
def cohort(tmp_path, monkeypatch):
    targets = [f"T{i:02}" for i in range(56)]
    excluded = [f"EXCLUDED{i:03}" for i in range(772)]
    sources, roster = [], []
    for source in gate.preparation.v2.SOURCE_IDS:
        groups = []
        for i, target in enumerate(targets):
            for batch in range(2):
                offset = batch * 10000
                groups.append({"target": target, "batch": str(batch),
                    "control_rows": list(range(offset, offset + 32)),
                    "context_control_rows": list(range(offset + 32, offset + 64)),
                    "sham_control_rows": list(range(offset + 64, offset + 72)),
                    "treated_rows": list(range(offset + 100 + i * 8, offset + 108 + i * 8))})
                roster.append({"group_id": len(roster), "source": source, "target": target, "batch": str(batch)})
        sources.append({"id": source, "cells_total": 20000, "groups": groups})
    row = {"sources": sources, "excluded_targets": excluded}
    row_path = tmp_path / "rows.json"
    row_sha = save(row_path, row)
    roles = {"targets": targets, "excluded_targets": excluded, "group_roster": roster,
        "folds": gate.preparation.make_folds(targets, roster),
        "inputs": {"contract_sha256": row_sha, "cache_sha256": "c" * 64},
        "input_paths": {"contract": str(row_path), "cache": str(tmp_path / "opaque-never-decoded.npz")},
        "go_acquisition_path": str(tmp_path / "synthetic-go-receipt.json")}
    prior = {"registration": {"path": str(tmp_path / "outer-roles.json"), "sha256": "d" * 64}}
    prior_path = tmp_path / "prior.json"
    monkeypatch.setattr(gate, "PRIOR_SHA", save(prior_path, prior))
    monkeypatch.setattr(gate.parent, "validate_contract", lambda *a, **k: (copy.deepcopy(prior), copy.deepcopy(roles)))
    annotations = {target: frozenset({"GO:0000001", f"GO:{i+2:07}"}) for i, target in enumerate(targets)}
    missing = gate.make_inner_folds(roles)["target_fold_0"]["tuning_targets"][0]
    annotations[missing] -= {"GO:0000001"}
    index = gate.go.AnnotationIndex(annotations, {t: (f"UniProtKB:P{i}",) for i, t in enumerate(targets)}, {}, {},
        {"ontology_data_version": "synthetic", "include_iea": False, "ancestor_policy": "direct_only"})
    monkeypatch.setattr(gate.preparation, "_go_index", lambda path: index)
    protocol = tmp_path / "protocol.md"
    protocol.write_text("Synthetic nested protocol, sealed before any GO parsing")
    return {"root": tmp_path, "protocol": protocol, "prior": prior_path, "roles": roles,
        "row": row, "row_path": row_path, "missing": missing}


def registered(cohort):
    out = cohort["root"] / "v6"
    digest = gate.register(cohort["prior"], cohort["protocol"], out)
    record, roles = gate.validate_registration(out / "role_manifest.json", digest)
    return out, digest, record, roles


def reseal(out, record):
    digest = save(out / "role_manifest.json", record)
    complete = json.loads((out / "complete.json").read_text())
    complete["role_manifest_sha256"] = digest
    save(out / "complete.json", complete)
    return digest


def test_nested_targets_and_same_source_group_roles(cohort):
    _, _, record, outer = registered(cohort)
    for key, fold in record["inner_folds"].items():
        assert len(fold["fit_targets"]) == 28 and len(fold["tuning_targets"]) == 14
        assert not set(fold["fit_targets"]) & set(fold["tuning_targets"])
        assert sorted(fold["fit_targets"] + fold["tuning_targets"]) == fold["outer_fit_targets"]
        assert not set(fold["outer_held_targets"]) & set(fold["outer_fit_targets"])
        assert all(fold[field] == sorted(fold[field]) for field in ("fit_targets", "tuning_targets", "outer_fit_targets", "outer_held_targets"))
        for direction in fold["directions"]:
            fit, tune = (direction[k] for k in ("fit_group_ids", "tuning_group_ids"))
            assert not set(fit) & set(tune)
            assert sorted(fit + tune) == direction["outer_fit_group_ids"]
            assert not set(fit + tune) & set(direction["outer_joint_eval_group_ids"])
            assert {outer["group_roster"][i]["source"] for i in fit+tune} == {direction["fit_source"]}
            assert {outer["group_roster"][i]["target"] for i in fit} == set(fold["fit_targets"])
            assert {outer["group_roster"][i]["target"] for i in tune} == set(fold["tuning_targets"])
    assert record["inputs"] == outer["inputs"] and record["input_paths"] == outer["input_paths"]
    assert record["excluded_targets"] == cohort["row"]["excluded_targets"]


def test_anchors_global_original_source_rows_not_group_offsets(cohort):
    _, _, record, _ = registered(cohort)
    assert len(record["anchor_pools"]) == 4
    pools = {(p["source"], p["batch"]): p for p in record["anchor_pools"]}
    for source in cohort["row"]["sources"]:
        for group in source["groups"]:
            pool = pools[source["id"], group["batch"]]
            assert len(pool["fit_reference_rows"]) == len(pool["tuning_anchor_rows"]) == 16
            assert not set(pool["fit_reference_rows"]) & set(pool["tuning_anchor_rows"])
            assert sorted(pool["fit_reference_rows"] + pool["tuning_anchor_rows"]) == group["control_rows"]
            assert pool["context_rows"] == group["context_control_rows"]
            assert pool["sham_rows"] == group["sham_control_rows"]
    assert pools["replogle_k562", "0"]["fit_reference_rows"] != pools["nadig_jurkat", "0"]["fit_reference_rows"]


def test_exact_json_array_rank_contract(cohort):
    args = [20260925, "nested-anchor-v6", "source|with|bars", "batch", 17]
    expected = hashlib.sha256(json.dumps(args, ensure_ascii=False, separators=(",", ":")).encode()).digest()
    assert gate._rank(args[1], *args[2:]) == expected
    assert gate._rank("x", "a|b", "c") != gate._rank("x", "a", "b|c")
    roles = gate.make_inner_folds(cohort["roles"])
    for fold_id, fold in roles.items():
        ranked = sorted(fold["outer_fit_targets"], key=lambda t: (gate._rank("nested-target-v6", fold_id, t), t))
        assert fold["tuning_targets"] == sorted(ranked[:14])


def test_registration_and_validation_never_decode_any_npz(cohort, monkeypatch):
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("Unexpected NPZ decode"))
    _, _, record, _ = registered(cohort)
    assert record["expression_decoded"] is False
    assert record["response_model_fitting_performed"] is False


def test_plan_sealed_before_go_and_checked_afterward(cohort, monkeypatch):
    original = gate.preparation._go_index
    def spy(path):
        plan = cohort["root"] / "v6/fold_plan.json"
        assert plan.is_file()
        assert len(gate.evidence.read_json(plan)["anchor_pools"]) == 4
        return original(path)
    monkeypatch.setattr(gate.preparation, "_go_index", spy)
    registered(cohort)


def test_only_inner28_vocabulary_and_outer42_output_masks_rebuilt(cohort):
    _, _, record, _ = registered(cohort)
    tables = gate.load_feature_tables(record)
    for key, fold in record["inner_folds"].items():
        table = tables[key]
        assert len(table.feature_ids) == 29
        forbidden = {f"GO:{int(t[1:])+2:07}" for t in fold["tuning_targets"]+fold["outer_held_targets"]}
        assert not forbidden & set(table.feature_ids)
        receipt = gate.evidence.read_json(fold["go"]["receipt_path"])
        assert set(table.target_ids) == set(fold["outer_fit_targets"]) - set(receipt["missing_in_vocab_targets"])
        assert receipt["target_count"] == 42
        assert receipt["train_target_ids_sha256"] == gate.go._sha_json(fold["fit_targets"])
        coverage = gate.evidence.read_json(fold["go"]["coverage_path"])
        assert coverage["roles"]["fit"]["targets"] == 28
        assert coverage["roles"]["held"]["targets"] == 14
    table = tables["target_fold_0"]
    assert cohort["missing"] not in table.target_ids
    with np.load(record["inner_folds"]["target_fold_0"]["go"]["npz_path"], allow_pickle=False) as arrays:
        index = arrays["target_ids"].tolist().index(cohort["missing"])
        assert bool(arrays["known_symbol"][index]) and bool(arrays["has_accepted_annotation"][index])
        assert not bool(arrays["present"][index]) and not arrays["features"][index].any()
    coverage = gate.evidence.read_json(record["inner_folds"]["target_fold_0"]["go"]["coverage_path"])
    assert cohort["missing"] in coverage["roles"]["held"]["missing_in_vocab_targets"]


@pytest.mark.parametrize("part", ["target", "tuning", "source", "group", "anchor", "anchor_bool", "policy_bool", "input", "excluded", "vocab_reuse", "extra"])
def test_rehashed_manifest_role_mutations_fail_closed(cohort, part):
    out, _, record, _ = registered(cohort)
    fold = record["inner_folds"]["target_fold_0"]
    if part == "target": fold["fit_targets"][0] = fold["outer_held_targets"][0]
    elif part == "tuning": fold["tuning_targets"] = [fold["fit_targets"][0]] * 14
    elif part == "source": fold["directions"][0]["held_source"] = fold["directions"][0]["fit_source"]
    elif part == "group": fold["directions"][0]["fit_group_ids"][0] = fold["directions"][0]["tuning_group_ids"][0]
    elif part == "anchor": record["anchor_pools"][0]["fit_reference_rows"][0] = record["anchor_pools"][0]["tuning_anchor_rows"][0]
    elif part == "anchor_bool": record["anchor_pools"][0]["fit_reference_rows"][0] = False
    elif part == "policy_bool": record["policy"]["automatic_promotion"] = 0
    elif part == "input": record["inputs"]["cache_sha256"] = "e" * 64
    elif part == "excluded": record["excluded_targets"].pop()
    elif part == "vocab_reuse": record["outer_vocabulary_reused"] = True
    elif part == "extra": record["extra"] = "ignored?"
    digest = reseal(out, record)
    with pytest.raises(ValueError): gate.validate_registration(out / "role_manifest.json", digest)


@pytest.mark.parametrize("part", ["npz", "receipt", "coverage"])
def test_artifact_byte_tampering_fails(cohort, part):
    out, digest, record, _ = registered(cohort)
    path = Path(record["inner_folds"]["target_fold_0"]["go"][part + "_path"])
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError): gate.validate_registration(out / "role_manifest.json", digest)


@pytest.mark.parametrize("part", ["train_hash", "output_hash", "fold_id", "sources_bool", "scope"])
def test_rehashed_go_receipt_metadata_fails(cohort, part):
    out, _, record, _ = registered(cohort)
    feature = record["inner_folds"]["target_fold_0"]["go"]
    path = Path(feature["receipt_path"])
    receipt = gate.evidence.read_json(path)
    if part == "train_hash": receipt["train_target_ids_sha256"] = "0" * 64
    elif part == "output_hash": receipt["target_ids_sha256"] = "0" * 64
    elif part == "fold_id": receipt["fold_id"] = "outer-only-vocabulary"
    elif part == "sources_bool": receipt["sources"] = True
    elif part == "scope": receipt["vocabulary_fit_scope"] = "all_targets"
    feature["receipt_sha256"] = save(path, receipt)
    digest = reseal(out, record)
    with pytest.raises(ValueError): gate.validate_registration(out / "role_manifest.json", digest)


@pytest.mark.parametrize("part", ["completed_bool", "fold_count_bool", "missing", "timestamp_extra"])
def test_completion_exact_typed_binding(cohort, part):
    out, digest, _, _ = registered(cohort)
    path = out / "complete.json"
    value = gate.evidence.read_json(path)
    if part == "completed_bool": value["completed"] = 1
    elif part == "fold_count_bool": value["inner_folds"] = True
    elif part == "missing": del value["expression_decoded"]
    elif part == "timestamp_extra": value["unexpected"] = "extra"
    save(path, value)
    with pytest.raises(ValueError): gate.validate_registration(out / "role_manifest.json", digest)


@pytest.mark.parametrize("part", ["same_batch", "cross_batch", "bool_row", "duplicate", "overlap", "roster"])
def test_anchor_original_metadata_errors_rejected(cohort, part):
    row, roles = copy.deepcopy(cohort["row"]), copy.deepcopy(cohort["roles"])
    group = row["sources"][0]["groups"][0]
    if part == "same_batch": group["control_rows"][0] = 999
    elif part == "cross_batch":
        for g in row["sources"][0]["groups"]:
            if g["batch"] == "1": g["control_rows"] = list(range(32))
    elif part == "bool_row": group["control_rows"][0] = False
    elif part == "duplicate": group["control_rows"][1] = group["control_rows"][0]
    elif part == "overlap": group["context_control_rows"] = group["control_rows"]
    elif part == "roster": roles["group_roster"][0]["target"] = "WRONG"
    with pytest.raises(ValueError): gate.make_anchor_pools(row, roles)


def test_go_failure_preserves_plan_without_completed_registration(cohort, monkeypatch):
    def fail(path): raise ValueError("Synthetic GO parse failure")
    monkeypatch.setattr(gate.preparation, "_go_index", fail)
    with pytest.raises(ValueError): registered(cohort)
    assert (cohort["root"] / "v6/fold_plan.json").is_file()
    assert not (cohort["root"] / "v6/role_manifest.json").exists()
    assert not (cohort["root"] / "v6/complete.json").exists()


def test_mid_build_protocol_change_prevents_completion(cohort, monkeypatch):
    original = gate.preparation._go_index
    def changed(path):
        cohort["protocol"].write_text("Changed after role registration")
        return original(path)
    monkeypatch.setattr(gate.preparation, "_go_index", changed)
    with pytest.raises(ValueError, match="changed while"): registered(cohort)
    assert not (cohort["root"] / "v6/complete.json").exists()


def test_fresh_output_no_overwrite(cohort):
    registered(cohort)
    with pytest.raises(FileExistsError): registered(cohort)


def test_parent_directory_alias_canonical_paths_are_replayable(cohort):
    alias = cohort["root"] / "directory-alias"
    alias.symlink_to(cohort["root"], target_is_directory=True)
    out = alias / "v6"
    digest = gate.register(alias / "prior.json", alias / "protocol.md", out)
    record, _ = gate.validate_registration(out.resolve() / "role_manifest.json", digest)
    assert record["fold_plan_path"] == str(out.resolve() / "fold_plan.json")
    assert record["prior_v5_contract"]["path"] == str(cohort["prior"])


@pytest.mark.parametrize("part", ["policy_bool", "anchor", "inner_target"])
def test_rehashed_plan_and_manifest_still_reconstructed(cohort, part):
    out, _, record, _ = registered(cohort)
    plan = gate.evidence.read_json(out / "fold_plan.json")
    if part == "policy_bool":
        plan["policy"]["automatic_promotion"] = 0
        record["policy"]["automatic_promotion"] = 0
    elif part == "anchor":
        for value in (plan, record): value["anchor_pools"][0]["fit_reference_rows"] = value["anchor_pools"][0]["tuning_anchor_rows"]
    else:
        for value in (plan, record): value["inner_folds"]["target_fold_0"]["fit_targets"][0] = value["inner_folds"]["target_fold_0"]["tuning_targets"][0]
    record["fold_plan_sha256"] = save(out / "fold_plan.json", plan)
    digest = reseal(out, record)
    with pytest.raises(ValueError, match="Frozen nested"): gate.validate_registration(out / "role_manifest.json", digest)


def test_changed_original_rows_fail_before_go_parse(cohort, monkeypatch):
    cohort["row_path"].write_text("changed original metadata")
    monkeypatch.setattr(gate.preparation, "_go_index", lambda *a: pytest.fail("GO parsed before source metadata authentication"))
    with pytest.raises(ValueError): registered(cohort)
    assert not (cohort["root"] / "v6").exists()


@pytest.mark.parametrize("kind", ["symlink", "fifo"])
def test_nonregular_manifest_rejected_without_blocking(cohort, kind):
    path = cohort["root"] / "bad.json"
    if kind == "symlink": path.symlink_to(cohort["prior"])
    else: os.mkfifo(path)
    with pytest.raises(ValueError): gate.validate_registration(path, "a" * 64)


def test_login_host_cli_rejected_before_any_registration(cohort, monkeypatch):
    monkeypatch.setattr(gate.socket, "gethostname", lambda: "aida.cac.cornell.edu")
    monkeypatch.setattr(gate, "register", lambda *a: pytest.fail("Registration attempted on head node"))
    with pytest.raises(SystemExit): gate.main(["--prior-v5-contract", str(cohort["prior"]), "--protocol", str(cohort["protocol"]),
        "--output-dir", str(cohort["root"] / "blocked")])
