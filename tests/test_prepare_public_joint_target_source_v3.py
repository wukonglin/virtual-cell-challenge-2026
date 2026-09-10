"""Synthetic56-target folds, train-only GO vocabulary, opaque expression pins."""
import copy
import gzip
import json
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_public_joint_target_source_v3 as gate


@pytest.fixture
def cohort(tmp_path, monkeypatch):
    pilot = tmp_path / gate.PILOT_RELATIVE
    pilot.mkdir(parents=True)
    targets = [f"T{index:02}" for index in range(56)]
    row = {"common_targets": targets, "excluded_targets": [f"DENIED{index}" for index in range(772)],
           "sources": [{"id": source, "groups": [{"target": target, "batch": f"batch{batch}"} for target in targets for batch in range(2)]}
                       for source in gate.v2.SOURCE_IDS]}
    for key, relative in gate.FILES.items():
        path = pilot / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(gate.go._json_bytes(targets) if key == "targets" else b"Opaque synthetic expression/metadata fixture; never decode")
    monkeypatch.setattr(gate, "EXPECTED", {key+"_sha256": gate.sha256_file(pilot/relative) for key, relative in gate.FILES.items()})
    monkeypatch.setattr(gate.v2, "load_contract", lambda *args, **kwargs: copy.deepcopy(row))
    acquired = tmp_path / gate.GO_RELATIVE
    acquired.parent.mkdir(parents=True)
    obo = "format-version: 1.2\ndata-version: synthetic\n"
    for number in range(1,58):
        obo += f"\n[Term]\nid: GO:{number:07}\nnamespace: biological_process\n"
    gaf = "!gaf-version: 2.2\n!generated-by: synthetic\n!date-generated: 2026-09-10\n"
    for index, target in enumerate(targets):
        for term in (1,index+2):
            gaf += "\t".join(["UniProtKB",f"P{index}",target,"involved_in",f"GO:{term:07}","PMID:1","IDA","","P","synthetic","","protein","taxon:9606","20260910","UniProt","",""])+"\n"
    (acquired.parent/"go-basic.obo").write_text(obo)
    (acquired.parent/"HUMAN-uniprot.gaf.gz").write_bytes(gzip.compress(gaf.encode(), mtime=0))
    records = [{"path": name, "sha256": gate.sha256_file(acquired.parent/name), "size_bytes": (acquired.parent/name).stat().st_size}
               for name in ("go-basic.obo", "HUMAN-uniprot.gaf.gz")]
    acquired.write_text(json.dumps({"schema": "vcc-public-feature-acquisition-v1", "files": records}))
    monkeypatch.setattr(gate, "GO_EXPECTED", {"acquisition_sha256": gate.sha256_file(acquired),
        "obo_sha256": records[0]["sha256"], "gaf_sha256": records[1]["sha256"]})
    protocol = tmp_path / "protocol.md"
    protocol.write_text("Synthetic fixed joint-target-source protocol")
    return tmp_path, protocol, row


def registered(cohort):
    root, protocol, _ = cohort
    output = root / "v3"
    receipt = gate.register(root, output, protocol)
    return output, receipt


def test_exact_fourfold_roles_exclude_held_targets_from_both_sources(cohort):
    output, receipt = registered(cohort)
    record = gate.load_registration(output/"role_manifest.json", receipt["role_manifest_sha256"], repo=cohort[0])
    assert len(record["folds"]) == 4
    held_once = []
    for fold in record["folds"]:
        assert len(fold["fit_targets"]) == 42 and len(fold["held_targets"]) == 14
        assert not set(fold["fit_targets"]) & set(fold["held_targets"])
        held_once += fold["held_targets"]
        for direction in fold["directions"]:
            assert len(direction["fit_group_ids"]) == 84
            assert len(direction["joint_eval_group_ids"]) == 28
            assert len(direction["held_target_group_ids_all_sources"]) == 56
            assert not set(direction["fit_group_ids"]) & set(direction["held_target_group_ids_all_sources"])
            partitions = [set(direction[key]) for key in ("fit_group_ids", "held_target_group_ids_all_sources", "other_excluded_group_ids")]
            assert len(direction["other_excluded_group_ids"]) == 84
            assert sum(map(len, partitions)) == len(set.union(*partitions)) == len(record["group_roster"])
            assert set(direction["joint_eval_group_ids"]) <= partitions[1]
            other = [record["group_roster"][i] for i in direction["other_excluded_group_ids"]]
            assert {group["source"] for group in other} == {direction["held_source"]}
            assert {group["target"] for group in other} == set(fold["fit_targets"])
            chosen = [record["group_roster"][i] for i in direction["fit_group_ids"]]
            assert {group["source"] for group in chosen} == {direction["fit_source"]}
            assert {group["target"] for group in chosen} == set(fold["fit_targets"])
    assert sorted(held_once) == cohort[2]["common_targets"]


def test_registration_never_decodes_expression_or_any_existing_npz(cohort, monkeypatch):
    monkeypatch.setattr(np, "load", lambda *args, **kwargs: pytest.fail("Registration attempted NPZ decoding"))
    output, receipt = registered(cohort)
    record = gate.load_registration(output/"role_manifest.json", receipt["role_manifest_sha256"], repo=cohort[0])
    assert record["expression_decoded"] is False
    assert record["full56_vocabulary_reused"] is False


def test_roles_frozen_before_go_source_parsing(cohort, monkeypatch):
    original = gate._go_index
    def spy(path):
        assert (cohort[0]/"v3/fold_plan.json").is_file()
        return original(path)
    monkeypatch.setattr(gate, "_go_index", spy)
    registered(cohort)


def test_fold_go_never_contains_held_only_terms(cohort):
    output, receipt = registered(cohort)
    record = gate.load_registration(output/"role_manifest.json", receipt["role_manifest_sha256"], repo=cohort[0])
    tables = gate.load_feature_tables(record)
    for fold in record["folds"]:
        table = tables[fold["fold_id"]]
        assert len(table.target_ids) == 56
        assert len(table.feature_ids) == 43  # One shared +42training-only terms, not57.
        forbidden = {f"GO:{int(target[1:])+2:07}" for target in fold["held_targets"]}
        assert not forbidden & set(table.feature_ids)
        audit = json.loads(Path(fold["go"]["coverage_path"]).read_text())
        assert audit["vocabulary_size"] == 43
        assert audit["roles"]["held"]["present"] == 14
        assert all(item["accepted_terms_outside_fit_vocabulary"] == 1 for item in audit["held_nearest_fit_overlap"])
        assert all(item["jaccard"] == .5 for item in audit["held_nearest_fit_overlap"])
        assert all(item["nearest_fit_target"] == min(fold["fit_targets"]) for item in audit["held_nearest_fit_overlap"])


def test_fold_partition_independent_of_input_order():
    targets = [f"T{i:02}" for i in range(56)]
    assert gate.make_folds(targets, []) == gate.make_folds(targets[::-1], [])


@pytest.mark.parametrize("part", ["cache", "targets", "contract", "prior_suite"])
def test_changed_existing_input_blocks_before_go(cohort, monkeypatch, part):
    path = cohort[0]/gate.PILOT_RELATIVE/gate.FILES[part]
    path.write_bytes(b"changed")
    monkeypatch.setattr(gate, "_go_index", lambda *args: pytest.fail("Parsed GO before prior inputs authenticated"))
    with pytest.raises(ValueError, match="exact previously exposed"):
        registered(cohort)


@pytest.mark.parametrize("part", ["fit_target", "fit_group", "held_source", "missing_fold", "reuse56", "code", "protocol"])
def test_changed_roles_or_pins_fail_closed(cohort, part):
    output, receipt = registered(cohort)
    path = output/"role_manifest.json"
    record = json.loads(path.read_text())
    fold = record["folds"][0]
    if part == "fit_target": fold["fit_targets"][0] = fold["held_targets"][0]
    elif part == "fit_group": fold["directions"][0]["fit_group_ids"][0] = fold["directions"][0]["joint_eval_group_ids"][0]
    elif part == "held_source": fold["directions"][0]["held_source"] = fold["directions"][0]["fit_source"]
    elif part == "missing_fold": record["folds"].pop()
    elif part == "reuse56": record["full56_vocabulary_reused"] = True
    elif part == "code": record["code_sha256"][gate.MODULES[0]] = "a"*64
    elif part == "protocol": cohort[1].write_text("changed")
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError): gate.load_registration(path, gate.sha256_file(path), repo=cohort[0])


@pytest.mark.parametrize("artifact", ["npz", "receipt", "coverage"])
def test_fold_artifact_tampering_rejected(cohort, artifact):
    output, receipt = registered(cohort)
    record = json.loads((output/"role_manifest.json").read_text())
    target = Path(record["folds"][0]["go"][artifact+"_path"])
    with target.open("ab") as stream: stream.write(b"changed")
    with pytest.raises(ValueError, match="path/hash"):
        gate.load_registration(output/"role_manifest.json", receipt["role_manifest_sha256"], repo=cohort[0])


def test_missing_go_rows_have_explicit_undefined_nearest_match():
    index = gate.go.AnnotationIndex({"A": frozenset({"GO:0000001"})}, {"A": ("UniProtKB:P1",)}, {}, {},
                                    {"ontology_data_version": "synthetic"})
    result = gate.go.build_features(index, target_ids=["A", "B"], train_target_ids=["A"], fold_id="synthetic", sources={key:"a"*64 for key in gate.GO_EXPECTED})
    audit = gate.coverage_audit(result, ["A"], ["B"])
    assert audit["roles"]["held"]["missing_in_vocab_targets"] == ["B"]
    assert audit["held_nearest_fit_overlap"][0]["jaccard"] is None
    assert audit["selection_or_filtering_performed"] is False


def test_existing_registration_is_never_overwritten(cohort):
    registered(cohort)
    with pytest.raises(FileExistsError): registered(cohort)


def test_bad_go_source_hash_retains_metadata_plan_but_no_complete_manifest(cohort):
    gaf = cohort[0]/gate.GO_RELATIVE
    (gaf.parent/"HUMAN-uniprot.gaf.gz").write_bytes(b"bad")
    with pytest.raises(ValueError, match="SHA"):
        registered(cohort)
    assert (cohort[0]/"v3/fold_plan.json").exists()
    assert not (cohort[0]/"v3/role_manifest.json").exists()
