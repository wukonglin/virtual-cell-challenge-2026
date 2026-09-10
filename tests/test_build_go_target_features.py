"""Synthetic GO/GAF fixtures only; no real target list or expression data."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_go_target_features as go  # noqa: E402
from public_flow_conditioning import align_target_features  # noqa: E402


OBO = """format-version: 1.2
data-version: releases/2026-07-26

[Term]
id: GO:0000001
name: synthetic process
namespace: biological_process
alt_id: GO:0000011
is_a: GO:0000003 ! ignored parent

[Term]
id: GO:0000002
name: synthetic function
namespace: molecular_function

[Term]
id: GO:0000003
name: heldout-only process or parent
namespace: biological_process

[Term]
id: GO:0000004
name: obsolete synthetic process
namespace: biological_process
is_obsolete: true
replaced_by: GO:0000001

[Typedef]
id: part_of
name: part of
"""
HEADER = """!gaf-version: 2.2
!generated-by: UniProt
!date-generated: 2026-07-28
!go-version: http://purl.obolibrary.org/obo/go/releases/2026-07-26/extensions/go-plus.ofn
"""
SOURCES = {"gaf_sha256": "a" * 64, "obo_sha256": "b" * 64, "acquisition_sha256": "c" * 64}


def gaf(symbol="A", term="GO:0000001", *, evidence="IDA", relation="involved_in",
        taxon="taxon:9606", object_id="P00001", aspect="P", aliases="ALIAS", database="UniProtKB"):
    return "\t".join([database, object_id, symbol, relation, term, "PMID:123456", evidence, "",
                       aspect, "synthetic protein", aliases, "protein", taxon, "20260728", "UniProt", "", ""]) + "\n"


def parse(*rows, include_iea=False, obo=OBO):
    return go.parse_gaf(io.StringIO(HEADER + "".join(rows)), go.parse_obo(io.StringIO(obo)),
                        include_iea=include_iea)


def features(index, targets=("A", "B", "UNKNOWN"), train=("A",), **kwargs):
    return go.build_features(index, target_ids=targets, train_target_ids=train,
                             fold_id="synthetic-fold-0", sources=SOURCES, **kwargs)


def test_obo_primary_active_ids_only_and_no_ancestor_propagation():
    ontology = go.parse_obo(io.StringIO(OBO))
    assert ontology.active == {"GO:0000001": "P", "GO:0000002": "F", "GO:0000003": "P"}
    assert ontology.obsolete == {"GO:0000004"}
    assert ontology.alternate == {"GO:0000011"}
    index = parse(gaf())
    assert index.terms["A"] == {"GO:0000001"}
    assert "GO:0000003" not in features(index).table.feature_ids
    assert index.policy["ancestor_policy"] == "direct_only"


def test_ignored_ontology_cycles_do_not_recurse_or_add_terms():
    cyclic = OBO.replace("name: heldout-only process or parent", "is_a: GO:0000001\nname: heldout-only process or parent")
    assert parse(gaf(), obo=cyclic).terms["A"] == {"GO:0000001"}


@pytest.mark.parametrize("term,reason", [("GO:0000004", "excluded_obsolete_GO"),
                                         ("GO:0000011", "excluded_alternate_GO"),
                                         ("GO:9999999", "excluded_unresolved_GO"),
                                         ("invalid", "excluded_unresolved_GO")])
def test_obsolete_alternate_and_unresolved_are_explicitly_excluded(term, reason):
    index = parse(gaf(term=term))
    assert not index.terms
    assert index.counts[reason] == 1
    assert index.source_ids["A"] == ("UniProtKB:P00001",)


@pytest.mark.parametrize("relation", ["NOT|involved_in", "involved_in|NOT", "NOT"])
def test_not_qualifiers_never_become_positive_features(relation):
    index = parse(gaf(relation=relation))
    assert not index.terms
    assert index.counts["excluded_NOT"] == 1


def test_positive_annotation_is_not_cancelled_by_context_specific_not_row():
    index = parse(gaf(), gaf(relation="NOT|involved_in"))
    assert index.terms["A"] == {"GO:0000001"}
    assert index.counts["excluded_NOT"] == 1


def test_iea_opt_in_and_nd_always_excluded():
    rows = (gaf(evidence="IEA"), gaf("B", evidence="ND"), gaf("C", evidence="NEWCODE"))
    excluded = parse(*rows)
    assert not excluded.terms
    assert excluded.counts["excluded_evidence_IEA"] == 1
    included = parse(*rows, include_iea=True)
    assert included.terms == {"A": {"GO:0000001"}}
    assert included.counts["excluded_evidence_ND"] == 1
    assert included.counts["excluded_evidence_unknown"] == 1
    assert included.policy["include_iea"] is True
    assert "IEA" not in excluded.policy["accepted_evidence_codes"]
    assert "ND" not in included.policy["accepted_evidence_codes"]
    with pytest.raises(ValueError):
        parse(gaf(), include_iea="False")


@pytest.mark.parametrize("evidence", sorted(go.KNOWN_EVIDENCE - {"IEA", "ND"}))
def test_declared_non_iea_evidence_is_accepted(evidence):
    assert parse(gaf(evidence=evidence)).terms["A"] == {"GO:0000001"}


@pytest.mark.parametrize("taxon", ["taxon:10090", "9606", "taxon:9606|taxon:10090", "taxon:10090|taxon:9606", ""])
def test_strict_sole_human_taxon_policy(taxon):
    index = parse(gaf(taxon=taxon))
    assert not index.terms and not index.source_ids
    assert index.counts["excluded_nonsole_human_taxon"] == 1


def test_duplicate_pair_binary_union_and_multiple_source_objects_audited():
    index = parse(gaf(), gaf(), gaf(object_id="P00002"), gaf(term="GO:0000002", aspect="F"))
    result = features(index)
    assert index.counts["accepted_annotation_rows"] == 4
    assert index.counts["duplicate_symbol_term_rows"] == 2
    assert index.counts["unique_symbol_term_pairs"] == 2
    assert index.counts["symbols_with_multiple_source_objects"] == 1
    np.testing.assert_array_equal(result.arrays["features"][0], [1, 1])
    assert result.receipt["source_objects_by_target"]["A"] == ("UniProtKB:P00001", "UniProtKB:P00002")


def test_exact_case_sensitive_symbol_mapping_no_synonym_aliases():
    index = parse(gaf("A", aliases="ALIAS|Another"))
    result = features(index, targets=("A", "ALIAS", "a"))
    np.testing.assert_array_equal(result.arrays["known_symbol"], [True, False, False])
    assert result.table.target_ids == ("A",)


def test_holdout_only_go_terms_cannot_expand_training_vocabulary():
    first = features(parse(gaf("A"), gaf("B", term="GO:0000002", aspect="F")))
    changed = features(parse(gaf("A"), gaf("B", term="GO:0000003"), gaf("B", term="GO:0000002", aspect="F")))
    assert first.table.feature_ids == changed.table.feature_ids == ("GO:0000001",)
    np.testing.assert_array_equal(first.arrays["features"], changed.arrays["features"])
    np.testing.assert_array_equal(first.arrays["has_accepted_annotation"], [True, True, False])
    np.testing.assert_array_equal(first.arrays["known_symbol"], [True, True, False])
    np.testing.assert_array_equal(first.arrays["present"], [True, False, False])
    assert first.receipt["vocabulary_fit_scope"] == "explicit_training_targets_only"
    aligned = align_target_features(("A", "B", "UNKNOWN"), [first.table])
    np.testing.assert_array_equal(aligned.present[:, 0], [True, False, False])
    np.testing.assert_array_equal(aligned.unknown_target, [False, True, True])


def test_heldout_can_use_existing_training_vocabulary_without_refitting():
    result = features(parse(gaf("A"), gaf("B"), gaf("B", term="GO:0000003")))
    assert result.table.feature_ids == ("GO:0000001",)
    np.testing.assert_array_equal(result.arrays["features"], [[1], [1], [0]])
    np.testing.assert_array_equal(result.arrays["accepted_term_count"], [1, 2, 0])


def test_known_symbol_with_only_rejected_annotations_remains_explicit():
    result = features(parse(gaf("A"), gaf("B", relation="NOT|involved_in")))
    np.testing.assert_array_equal(result.arrays["known_symbol"], [True, True, False])
    np.testing.assert_array_equal(result.arrays["has_accepted_annotation"], [True, False, False])
    np.testing.assert_array_equal(result.arrays["present"], [True, False, False])


@pytest.mark.parametrize("changes", [
    {"target_ids": ("A", "A")}, {"train_target_ids": ("A", "A")},
    {"train_target_ids": ()}, {"train_target_ids": ("OUTSIDE",)},
    {"train_target_ids": ("UNKNOWN",)}, {"target_ids": {"A": 1}},
    {"target_ids": {"A", "B"}}, {"target_ids": "A"},
    {"fold_id": ""}, {"sources": {}}, {"sources": {**SOURCES, "gaf_sha256": "bad"}},
    {"max_dense_elements": 1}, {"max_dense_elements": True},
])
def test_build_rejects_unregistered_ambiguous_or_empty_training_contract(changes):
    kwargs = dict(target_ids=("A", "B", "UNKNOWN"), train_target_ids=("A",),
                  fold_id="synthetic", sources=SOURCES)
    kwargs.update(changes)
    with pytest.raises(ValueError):
        go.build_features(parse(gaf()), **kwargs)


@pytest.mark.parametrize("text", [
    OBO.replace("format-version: 1.2", "format-version: 1.4"),
    OBO.replace("data-version: releases/2026-07-26\n", ""),
    OBO.replace("id: GO:0000002", "id: GO:0000001"),
    OBO.replace("id: GO:0000002", "id: GO:bad"),
    OBO.replace("namespace: molecular_function", "namespace: unknown"),
    OBO.replace("is_obsolete: true", "is_obsolete: maybe"),
    OBO.replace("alt_id: GO:0000011", "alt_id: GO:0000002"),
])
def test_malformed_or_ambiguous_obo_rejected(text):
    with pytest.raises(ValueError):
        go.parse_obo(io.StringIO(text))


@pytest.mark.parametrize("text", [
    gaf(), HEADER.replace("2.2", "2.1") + gaf(),
    HEADER.replace("!generated-by: UniProt\n", "") + gaf(),
    HEADER + "too\tfew\tcolumns\n", HEADER + gaf().rstrip("\n") + "\textra\n",
    HEADER.replace("releases/2026-07-26", "releases/1999-01-01") + gaf(),
    HEADER + gaf(symbol=""), HEADER + gaf(object_id=""), HEADER + gaf(relation=""),
    HEADER + gaf(relation="enables|involved_in"),
])
def test_malformed_gaf_or_release_mismatch_rejected(text):
    with pytest.raises(ValueError):
        go.parse_gaf(io.StringIO(text), go.parse_obo(io.StringIO(OBO)))


def test_database_and_aspect_exclusions_are_counted():
    index = parse(gaf(database="OtherDB"), gaf(aspect="F"))
    assert index.counts["excluded_database"] == index.counts["excluded_aspect_mismatch"] == 1
    assert not index.terms


def test_safe_deterministic_npz_receipt_and_no_overwrite(tmp_path):
    result = features(parse(gaf(), gaf("B")))
    first = go.write_artifacts(result, tmp_path / "one")
    second = go.write_artifacts(result, tmp_path / "two")
    assert first["npz_sha256"] == second["npz_sha256"]
    assert first["feature_table_source_id"] == result.table.source_id
    assert first["feature_table_source_sha256"] == result.table.source_sha256
    with pytest.raises(ValueError):
        result.arrays["features"][0, 0] = 0
    raw = (tmp_path / "one" / "features.npz").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == first["npz_sha256"]
    with np.load(tmp_path / "one" / "features.npz", allow_pickle=False) as arrays:
        assert set(arrays.files) == set(result.arrays)
        for name in arrays.files:
            assert arrays[name].dtype.kind != "O"
            np.testing.assert_array_equal(arrays[name], result.arrays[name])
    assert json.loads((tmp_path / "one" / "receipt.json").read_text())["npz_size_bytes"] == len(raw)
    with pytest.raises(FileExistsError):
        go.write_artifacts(result, tmp_path / "one")


def test_object_arrays_cannot_be_serialized_as_pickle(tmp_path):
    result = features(parse(gaf()))
    result.arrays["unsafe"] = np.array([{"arbitrary": "object"}], dtype=object)
    with pytest.raises(ValueError, match="Object arrays"):
        go.write_artifacts(result, tmp_path / "unsafe")
    assert not (tmp_path / "unsafe" / "receipt.json").exists()


def source_bundle(tmp_path):
    gaf_path, obo_path = tmp_path / "HUMAN-uniprot.gaf.gz", tmp_path / "go-basic.obo"
    gaf_path.write_bytes(gzip.compress((HEADER + gaf()).encode(), mtime=0))
    obo_path.write_text(OBO)
    receipt = {"schema": "vcc-public-feature-acquisition-v1", "files": [
        {"path": path.name, "size_bytes": path.stat().st_size,
         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in (gaf_path, obo_path)]}
    path = tmp_path / "acquisition.json"
    path.write_text(json.dumps(receipt))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_source_authentication_and_synthetic_cli_end_to_end(tmp_path):
    receipt, digest = source_bundle(tmp_path)
    hashes = go.authenticated_sources(receipt, digest)
    assert hashes["acquisition_sha256"] == digest
    targets, train = tmp_path / "targets.json", tmp_path / "train.json"
    targets.write_text(json.dumps(["A", "UNKNOWN"]))
    train.write_text(json.dumps(["A"]))
    assert go.main(["--acquisition", str(receipt), "--acquisition-sha256", digest,
                    "--targets-json", str(targets), "--train-targets-json", str(train),
                    "--fold-id", "synthetic-only", "--output-dir", str(tmp_path / "features")]) == 0
    result = json.loads((tmp_path / "features" / "receipt.json").read_text())
    assert result["sources"] == hashes
    assert result["policy"]["include_iea"] is False
    assert result["response_model_trained"] is False and result["expression_read"] is False


@pytest.mark.parametrize("failure", ["receipt-sha", "source-bytes", "source-symlink", "source-path", "size-bool"])
def test_changed_or_unsafe_registered_sources_fail(tmp_path, failure):
    receipt_path, digest = source_bundle(tmp_path)
    if failure == "receipt-sha":
        digest = "0" * 64
    elif failure == "source-bytes":
        (tmp_path / "go-basic.obo").write_text(OBO + "\n")
    elif failure == "source-symlink":
        source = tmp_path / "go-basic.obo"
        renamed = tmp_path / "moved.obo"
        source.rename(renamed)
        source.symlink_to(renamed)
    else:
        receipt = json.loads(receipt_path.read_text())
        if failure == "source-path":
            receipt["files"][0]["path"] = "../outside"
        else:
            receipt["files"][0]["size_bytes"] = True
        receipt_path.write_text(json.dumps(receipt))
        digest = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        go.authenticated_sources(receipt_path, digest)
