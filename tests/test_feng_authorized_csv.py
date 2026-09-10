"""Synthetic gzip fixtures only: never open real public counts or use a GPU."""
from __future__ import annotations

import csv
from dataclasses import FrozenInstanceError
import gzip
import hashlib
import io
import os
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import feng_authorized_csv as feng  # noqa: E402


def _compressed(rows, delimiter=","):
    stream = io.StringIO(newline="")
    csv.writer(stream, delimiter=delimiter).writerows(rows)
    return gzip.compress(stream.getvalue().encode(), mtime=0)


def _fixture(tmp_path, *, screen="fitness", orphan=False, rows=None,
             duplicate_header=False, duplicate_metadata=False):
    prefixes = {"fitness": "ST-P1-D3_I1_", "nonfitness": "MD-P1-D6_I1_",
                "targeted": "P1_I1_"}
    cells = [prefixes[screen] + barcode + "-1" for barcode in ("AAAA", "CCCC", "GGGG")]
    metadata = [[feng.metadata_id(screen, cell), "batch", "TEST_ACGT", "line_1"]
                for cell in reversed(cells)]
    if duplicate_metadata:
        metadata.append(metadata[0])
    if orphan:
        cells.append(prefixes[screen] + "TTTT-1")
    header_cells = cells + [cells[0]] if duplicate_header else cells
    if rows is None:
        rows = [["G1", "1.0", "EXCLUDED_SECRET", "3e0"],
                ["UNSELECTED_GENE", "NOT_NUMERIC", "NAN", "-10"],
                ["G2", "4", "EXCLUDED_SECRET", "6"]]
        if orphan:
            rows = [row + ["ORPHAN_MUST_NEVER_PARSE"] for row in rows]
    source = tmp_path / "rna.csv.gz"
    meta = tmp_path / "metadata.tsv.gz"
    source.write_bytes(_compressed([["", *header_cells], *rows]))
    meta.write_bytes(_compressed([["Cell_ID", "Batch", "Guide_Call", "Cell_Line"],
                                 *metadata], delimiter="\t"))
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    selected = [cells[2], cells[0]]
    contract = {"schema": feng.SCHEMA, "screen": screen,
        "authorization_sha256": "a" * 64,
        "source_sha256": sha(source), "metadata_sha256": sha(meta),
        "header_axis_sha256": feng.canonical_sha256(header_cells),
        "rna_cell_ids": selected,
        "metadata_cell_ids": [feng.metadata_id(screen, x) for x in selected],
        "gene_ids": ["G2", "G1"],
        "excluded_unannotated_rna_ids": [cells[-1]] if orphan else []}
    return source, meta, contract


def _open(fixture, **overrides):
    source, metadata, contract = fixture
    kwargs = {"contract": contract,
        "expected_source_sha256": contract["source_sha256"],
        "expected_metadata_sha256": contract["metadata_sha256"],
        "expected_header_axis_sha256": contract["header_axis_sha256"],
        "expected_allowlist_sha256": feng.canonical_sha256(contract)}
    kwargs.update(overrides)
    return feng.open_authorized_feng_csv(source, metadata, **kwargs)


@pytest.mark.parametrize("screen", ["fitness", "nonfitness", "targeted"])
def test_exact_mapping_reorders_cells_and_genes_without_parsing_excluded_fields(tmp_path, monkeypatch, screen):
    fixture = _fixture(tmp_path, screen=screen)
    calls = []
    real_parse = feng._parse_count

    def parser(value):
        calls.append(value)
        assert "SECRET" not in value and value not in ("NOT_NUMERIC", "NAN", "-10")
        return real_parse(value)

    monkeypatch.setattr(feng, "_parse_count", parser)
    with _open(fixture) as prepared:
        assert not calls  # Authentication/header/metadata phase has no numeric reads.
        assert prepared.rna_cell_ids == tuple(fixture[2]["rna_cell_ids"])
        output = prepared.read_counts()
        np.testing.assert_array_equal(output, [[6, 3], [4, 1]])
        assert output.dtype == np.int64
    assert calls == ["3e0", "1.0", "6", "4"]


@pytest.mark.parametrize("screen,raw,expected", [
    ("fitness", "ST-P3-D4_I12_ACGT-1", "MP-ST-P3-D4_I12_ACGT-1"),
    ("nonfitness", "MD-P8-D6_I2_ACGT-1", "MP-MD-P8-D6_I2_ACGT-1"),
    ("targeted", "P4_I73_ACGT-1", "PC-P4-D3_I73_ACGT-1")])
def test_source_specific_id_string_rules(screen, raw, expected):
    assert feng.metadata_id(screen, raw) == expected


@pytest.mark.parametrize("screen,raw", [("other", "P1_I1_AAAA-1"),
    ("fitness", "MP-ST-P1-D3_I1_AAAA-1"), ("targeted", "PC-P1-D3_I1_AAAA-1"),
    ("nonfitness", "ST-P1-D3_I1_AAAA-1"), ("targeted", "P1_I1_BAD-1")])
def test_mapping_never_guesses_or_strips_unknown_prefixes(screen, raw):
    with pytest.raises(feng.FengAuthorizationError):
        feng.metadata_id(screen, raw)


@pytest.mark.parametrize("pin", ["expected_source_sha256", "expected_metadata_sha256",
                                  "expected_header_axis_sha256", "expected_allowlist_sha256"])
def test_external_pins_must_match_contract_before_any_numeric_reads(tmp_path, monkeypatch, pin):
    fixture = _fixture(tmp_path)
    monkeypatch.setattr(feng, "_parse_count", lambda _: pytest.fail("Numeric parse before authorization"))
    with pytest.raises(feng.FengAuthorizationError, match="SHA256|sha256"):
        with _open(fixture, **{pin: "b" * 64}):
            pytest.fail("Mismatched authorization must not prepare")


@pytest.mark.parametrize("field", ["source_sha256", "metadata_sha256", "header_axis_sha256"])
def test_forged_content_or_header_pin_is_rejected_after_contract_hash_matches(tmp_path, field):
    fixture = _fixture(tmp_path)
    fixture[2][field] = "b" * 64
    with pytest.raises(feng.FengAuthorizationError, match="SHA256"):
        with _open(fixture):
            pass


@pytest.mark.parametrize("authorization", ["", "0" * 64, "not-an-authorization", None])
def test_independent_prior_authorization_digest_is_required(tmp_path, authorization):
    fixture = _fixture(tmp_path)
    fixture[2]["authorization_sha256"] = authorization
    with pytest.raises(feng.FengAuthorizationError, match="authorization"):
        with _open(fixture):
            pass


def test_registered_orphan_is_excluded_never_selected_or_parsed(tmp_path):
    fixture = _fixture(tmp_path, screen="targeted", orphan=True)
    with _open(fixture) as prepared:
        np.testing.assert_array_equal(prepared.read_counts(), [[6, 3], [4, 1]])
    fixture[2]["rna_cell_ids"][0] = fixture[2]["excluded_unannotated_rna_ids"][0]
    fixture[2]["metadata_cell_ids"][0] = feng.metadata_id("targeted", fixture[2]["rna_cell_ids"][0])
    with pytest.raises(feng.FengAuthorizationError, match="orphan"):
        with _open(fixture):
            pass


def test_unregistered_or_stale_orphan_exclusions_fail(tmp_path):
    fixture = _fixture(tmp_path, orphan=True)
    fixture[2]["excluded_unannotated_rna_ids"] = []
    with pytest.raises(feng.FengAuthorizationError, match="orphan"):
        with _open(fixture):
            pass
    fixture = _fixture(tmp_path)
    fixture[2]["excluded_unannotated_rna_ids"] = ["ST-P1-D3_I1_TTTT-1"]
    with pytest.raises(feng.FengAuthorizationError, match="orphan"):
        with _open(fixture):
            pass


@pytest.mark.parametrize("option", ["duplicate_header", "duplicate_metadata"])
def test_duplicate_source_or_metadata_cell_ids_fail(tmp_path, option):
    fixture = _fixture(tmp_path, **{option: True})
    with pytest.raises(feng.FengAuthorizationError, match="[Dd]uplicate"):
        with _open(fixture):
            pass


@pytest.mark.parametrize("field", ["rna_cell_ids", "metadata_cell_ids", "gene_ids"])
def test_duplicate_authorized_ids_fail(tmp_path, field):
    fixture = _fixture(tmp_path)
    fixture[2][field][1] = fixture[2][field][0]
    with pytest.raises(feng.FengAuthorizationError, match="Duplicate"):
        with _open(fixture):
            pass


@pytest.mark.parametrize("value", ["-1", "0.5", "NaN", "Inf", "1_000", "secret",
                                   "9223372036854775808", "1e10000", " 1"])
def test_invalid_selected_numeric_values_fail(tmp_path, value):
    fixture = _fixture(tmp_path, rows=[["G1", value, "DO_NOT_PARSE", "1"],
                                      ["G2", "1", "DO_NOT_PARSE", "1"]])
    with _open(fixture) as prepared:
        with pytest.raises(feng.FengAuthorizationError, match="[Cc]ount"):
            prepared.read_counts()
        with pytest.raises(feng.FengAuthorizationError, match="failed"):
            prepared.read_counts()


@pytest.mark.parametrize("rows,match", [
    ([["G1", "1", "ignored", "2"]], "missing"),
    ([["G1", "1", "ignored", "2"], ["G1", "3", "ignored", "4"]], "Duplicate gene"),
    ([["G1", "1", "ignored"]], "column count")])
def test_gene_axis_and_row_schema_are_fail_closed(tmp_path, rows, match):
    with _open(_fixture(tmp_path, rows=rows)) as prepared:
        with pytest.raises(feng.FengAuthorizationError, match=match):
            prepared.read_counts()


def test_authorization_axes_cannot_mutate_and_reader_cannot_reopen(tmp_path):
    with _open(_fixture(tmp_path)) as prepared:
        with pytest.raises(FrozenInstanceError):
            prepared.gene_ids = ("UNAUTHORIZED",)
        prepared.read_counts()
        with pytest.raises(feng.FengAuthorizationError, match="consumed"):
            prepared.read_counts()
    with pytest.raises(feng.FengAuthorizationError, match="closed"):
        prepared.read_counts()


def test_output_and_line_memory_limits(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path)
    monkeypatch.setattr(feng, "MAX_OUTPUT_ELEMENTS", 3)
    with pytest.raises(feng.FengAuthorizationError, match="element limit"):
        with _open(fixture):
            pass
    monkeypatch.setattr(feng, "MAX_OUTPUT_ELEMENTS", 8_000_000)
    monkeypatch.setattr(feng, "MAX_ROW_BYTES", 8)
    with _open(fixture) as prepared:
        with pytest.raises(feng.FengAuthorizationError, match="byte limit"):
            prepared.read_counts()


def test_changed_source_after_preparation_is_rejected(tmp_path):
    fixture = _fixture(tmp_path)
    with _open(fixture) as prepared:
        with fixture[0].open("ab") as stream:
            stream.write(b"mutation")
        with pytest.raises(feng.FengAuthorizationError, match="changed"):
            prepared.read_counts()


def test_source_file_symlink_is_rejected(tmp_path):
    source, metadata, contract = _fixture(tmp_path)
    linked = tmp_path / "linked.gz"
    linked.symlink_to(source)
    with pytest.raises(feng.FengAuthorizationError, match="symlink"):
        with _open((linked, metadata, contract)):
            pass


def test_metadata_records_have_a_preparse_byte_limit(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path)
    monkeypatch.setattr(feng, "MAX_METADATA_RECORD_BYTES", 8)
    with pytest.raises(feng.FengAuthorizationError, match="byte limit"):
        with _open(fixture):
            pass


def test_selected_orphan_cannot_be_invented_in_the_contract(tmp_path):
    fixture = _fixture(tmp_path)
    raw = "ST-P1-D3_I1_TTTT-1"
    fixture[2]["rna_cell_ids"][0] = raw
    fixture[2]["metadata_cell_ids"][0] = feng.metadata_id("fitness", raw)
    with pytest.raises(feng.FengAuthorizationError, match="orphan"):
        with _open(fixture):
            pass


def test_selected_cells_must_be_present_in_source_header(tmp_path):
    source, meta, contract = _fixture(tmp_path)
    raw = "ST-P1-D3_I1_TTTT-1"
    metadata_rows = list(csv.reader(io.StringIO(gzip.decompress(meta.read_bytes()).decode()), delimiter="\t"))
    metadata_rows.append([feng.metadata_id("fitness", raw), "batch", "TEST_ACGT", "line_1"])
    meta.write_bytes(_compressed(metadata_rows, delimiter="\t"))
    contract["metadata_sha256"] = hashlib.sha256(meta.read_bytes()).hexdigest()
    contract["rna_cell_ids"][0] = raw
    contract["metadata_cell_ids"][0] = feng.metadata_id("fitness", raw)
    with pytest.raises(feng.FengAuthorizationError, match="absent from header"):
        with _open((source, meta, contract)):
            pass


def test_valid_exact_integer_formats_do_not_round(tmp_path):
    fixture = _fixture(tmp_path, rows=[["G1", "-0", "ignored", "1.5e1"],
                                      ["G2", "9223372036854775807", "ignored", "+2"]])
    with _open(fixture) as prepared:
        np.testing.assert_array_equal(prepared.read_counts(),
            np.array([[2, 15], [9223372036854775807, 0]], dtype=np.int64))


def test_blank_source_header_has_a_typed_error(tmp_path):
    source, metadata, contract = _fixture(tmp_path)
    source.write_bytes(gzip.compress(b"\n", mtime=0))
    contract["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    with pytest.raises(feng.FengAuthorizationError, match="header"):
        with _open((source, metadata, contract)):
            pass


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX FIFO test")
def test_nonregular_source_fails_without_blocking_on_fifo(tmp_path):
    _, metadata, contract = _fixture(tmp_path)
    fifo = tmp_path / "source.fifo"
    os.mkfifo(fifo)
    with pytest.raises(feng.FengAuthorizationError, match="regular file"):
        with _open((fifo, metadata, contract)):
            pass
