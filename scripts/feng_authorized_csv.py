"""Pinned, column-authorized Feng RNA CSV consumer; no acquisition or fitting.

The caller must first register source roles, the target denylist, donor folds,
matched verified controls and the exact cell/gene selection. A hash here binds
that prior authorization; it does not establish biological admissibility.

Unlike an H5 row reader, gzip CSV necessarily scans compressed bytes and row
text, including excluded fields. Only selected cells at registered gene rows
are converted to numbers. Excluded values never enter normalization or output.
Preparation reads hashes/header/assignment metadata only; read_counts() is a
separate explicit expression-read operation. Physical CSV records must occupy
one line, as in this deposit; line, axis and output sizes are bounded.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import csv
import gzip
import hashlib
import json
import os
import re
import stat

import numpy as np


SCHEMA = "vcc-feng-column-authorization-v1"
MAX_CELLS = 2_000_000
MAX_GENES = 100_000
MAX_HEADER_BYTES = 100_000_000
MAX_ROW_BYTES = 32_000_000
MAX_METADATA_RECORD_BYTES = 8_192
MAX_OUTPUT_ELEMENTS = 8_000_000
_FIELDS = {"schema", "screen", "authorization_sha256", "source_sha256",
           "metadata_sha256", "header_axis_sha256", "rna_cell_ids",
           "metadata_cell_ids", "gene_ids", "excluded_unannotated_rna_ids"}


class FengAuthorizationError(ValueError):
    """Invalid authorization, source identity, schema or selected counts."""


def _require(value, message):
    if not value:
        raise FengAuthorizationError(message)


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _digest(value, label):
    _require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
             and value != "0" * 64, f"Invalid {label} SHA256")
    return value


def _ids(value, label, *, maximum, empty=False):
    _require(isinstance(value, list) and len(value) <= maximum,
             f"Invalid or oversized {label} axis")
    _require(empty or bool(value), f"Empty {label} axis")
    _require(all(isinstance(x, str) and x and len(x) <= 256
                 and not any(c in x for c in "\r\n\x00") for x in value),
             f"Invalid {label} identifiers")
    _require(len(set(value)) == len(value), f"Duplicate {label} identifiers")
    return tuple(value)


def metadata_id(screen, rna_id):
    """Exact source-specific mapping, never a fuzzy/positional join."""
    if screen in ("fitness", "nonfitness"):
        kind = "ST" if screen == "fitness" else "MD"
        _require(re.fullmatch(kind + r"-P[0-9]+-D[0-9]+_I[0-9]+_[ACGT]+-1", rna_id),
                 "Unexpected genome-wide RNA cell ID")
        return "MP-" + rna_id
    _require(screen == "targeted", "Unknown Feng screen")
    match = re.fullmatch(r"P([0-9]+)_I([0-9]+)_([ACGT]+-1)", rna_id)
    _require(match is not None, "Unexpected targeted RNA cell ID")
    return f"PC-P{match[1]}-D3_I{match[2]}_{match[3]}"


def _identity(handle):
    info = os.fstat(handle.fileno())
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


@contextmanager
def _authenticated(path, expected):
    _digest(expected, "source")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(os.fspath(path), flags)
    except OSError as error:
        raise FengAuthorizationError("Cannot open regular source without following a file symlink") from error
    with os.fdopen(descriptor, "rb") as handle:
        _require(stat.S_ISREG(os.fstat(handle.fileno()).st_mode), "Source is not a regular file")
        original = _identity(handle)
        digest = hashlib.sha256()
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
        _require(_identity(handle) == original, "Source changed during authentication")
        _require(digest.hexdigest() == expected, "Source SHA256 mismatch")
        handle.seek(0)
        yield handle, original


def _record(stream, limit, *, delimiter=","):
    line = stream.readline(limit + 1)
    if not line:
        return None
    _require(len(line) <= limit, "CSV record exceeds byte limit")
    try:
        text = line.decode("utf-8")
        return next(csv.reader([text], strict=True, delimiter=delimiter))
    except (UnicodeError, csv.Error, StopIteration) as error:
        raise FengAuthorizationError("Malformed single-line UTF-8 CSV record") from error


def _parse_count(text):
    _require(len(text) <= 64 and re.fullmatch(
        r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?", text),
        "Malformed selected count")
    try:
        value = Decimal(text)
        _require(value.is_finite() and value >= 0 and value <= np.iinfo(np.int64).max
                 and value == value.to_integral_value(), "Selected count must be a finite nonnegative int64 integer")
        return int(value)
    except (InvalidOperation, OverflowError) as error:
        raise FengAuthorizationError("Malformed selected count") from error


@dataclass(frozen=True)
class _PreparedFengCSV:
    """Use only inside open_authorized_feng_csv; no expression is read on init."""
    _stream: object
    _handle: object
    _file_identity: tuple
    _positions: tuple
    _column_count: int
    rna_cell_ids: tuple
    metadata_cell_ids: tuple
    gene_ids: tuple
    authorization_sha256: str
    _state: str = "prepared"

    def read_counts(self):
        """Read the authorized matrix, selected cells × registered gene order."""
        _require(self._state == "prepared", "Reader is closed, consumed or failed")
        object.__setattr__(self, "_state", "consuming")
        try:
            _require(_identity(self._handle) == self._file_identity, "Source changed before count reads")
            output = np.zeros((len(self.rna_cell_ids), len(self.gene_ids)), dtype=np.int64)
            selected = {name: index for index, name in enumerate(self.gene_ids)}
            seen = set()
            while (row := _record(self._stream, MAX_ROW_BYTES)) is not None:
                _require(len(row) == self._column_count, "CSV row/header column count mismatch")
                gene = row[0]
                _require(isinstance(gene, str) and gene and len(gene) <= 256, "Invalid gene row ID")
                _require(gene not in seen, "Duplicate gene row ID")
                seen.add(gene)
                _require(len(seen) <= MAX_GENES, "Gene-row limit exceeded")
                if gene in selected:
                    output[:, selected[gene]] = [_parse_count(row[position]) for position in self._positions]
            _require(set(selected).issubset(seen), "Registered gene rows are missing")
            _require(_identity(self._handle) == self._file_identity, "Source changed during count reads")
        except Exception:
            object.__setattr__(self, "_state", "failed")
            raise
        object.__setattr__(self, "_state", "complete")
        return output


@contextmanager
def open_authorized_feng_csv(source_path, metadata_path, *, contract,
                            expected_source_sha256, expected_metadata_sha256,
                            expected_header_axis_sha256, expected_allowlist_sha256):
    """Authenticate sources and freeze a caller-approved column/gene selection.

    The allowlist digest is an externally pinned hash of the exact contract.
    authorization_sha256 must reference the caller's independently registered
    public-only source/split/control/target-exclusion authorization. This module
    does not create that authorization or select cells from their expression.
    """
    _require(isinstance(contract, dict) and set(contract) == _FIELDS, "Invalid authorization contract schema")
    _require(contract["schema"] == SCHEMA, "Invalid authorization schema version")
    _require(canonical_sha256(contract) == _digest(expected_allowlist_sha256, "allowlist"), "Allowlist SHA256 mismatch")
    for key, pinned in (("source_sha256", expected_source_sha256),
                        ("metadata_sha256", expected_metadata_sha256),
                        ("header_axis_sha256", expected_header_axis_sha256)):
        _require(contract[key] == _digest(pinned, key), f"Contract {key} mismatch")
    authorization = _digest(contract["authorization_sha256"], "prior authorization")
    rna_ids = _ids(contract["rna_cell_ids"], "selected RNA cell", maximum=MAX_CELLS)
    meta_ids = _ids(contract["metadata_cell_ids"], "selected metadata cell", maximum=MAX_CELLS)
    genes = _ids(contract["gene_ids"], "selected gene", maximum=MAX_GENES)
    orphans = _ids(contract["excluded_unannotated_rna_ids"], "excluded orphan", maximum=MAX_CELLS, empty=True)
    _require(len(rna_ids) == len(meta_ids), "Selected cell axes differ")
    _require(len(rna_ids) * len(genes) <= MAX_OUTPUT_ELEMENTS, "Authorized output exceeds element limit")
    _require(not set(rna_ids).intersection(orphans), "An excluded orphan cannot be selected")
    _require(tuple(metadata_id(contract["screen"], x) for x in rna_ids) == meta_ids, "Selected RNA/metadata mapping differs")
    with _authenticated(metadata_path, expected_metadata_sha256) as (meta_handle, meta_identity):
        metadata_ids = set()
        with gzip.GzipFile(fileobj=meta_handle, mode="rb") as compressed:
            columns = _record(compressed, MAX_METADATA_RECORD_BYTES, delimiter="\t")
            _require(columns == ["Cell_ID", "Batch", "Guide_Call", "Cell_Line"], "Metadata columns differ")
            while (row := _record(compressed, MAX_METADATA_RECORD_BYTES, delimiter="\t")) is not None:
                _require(len(row) == 4, "Malformed metadata row")
                cell = row[0]
                _require(cell and len(cell) <= 256 and cell not in metadata_ids,
                         "Invalid or duplicate metadata cell ID")
                metadata_ids.add(cell)
                _require(len(metadata_ids) <= MAX_CELLS, "Metadata cell limit exceeded")
        _require(_identity(meta_handle) == meta_identity, "Metadata changed while reading")
    _require(set(meta_ids).issubset(metadata_ids), "Selected orphan lacks assignment metadata")
    with _authenticated(source_path, expected_source_sha256) as (handle, original):
        with gzip.GzipFile(fileobj=handle, mode="rb") as stream:
            header = _record(stream, MAX_HEADER_BYTES)
            _require(bool(header) and header[0] == "", "Expected unnamed gene-ID header field")
            raw_ids = _ids(header[1:], "RNA header cell", maximum=MAX_CELLS)
            _require(canonical_sha256(list(raw_ids)) == expected_header_axis_sha256, "Header-axis SHA256 mismatch")
            _require(set(rna_ids).issubset(raw_ids), "Selected RNA cells absent from header")
            observed_orphans = {x for x in raw_ids if metadata_id(contract["screen"], x) not in metadata_ids}
            _require(observed_orphans == set(orphans), "Unregistered or stale orphan exclusion")
            lookup = {cell: index + 1 for index, cell in enumerate(raw_ids)}
            prepared = _PreparedFengCSV(stream, handle, original, tuple(lookup[x] for x in rna_ids),
                len(header), rna_ids, meta_ids, genes, authorization)
            try:
                yield prepared
            finally:
                object.__setattr__(prepared, "_state", "closed")
