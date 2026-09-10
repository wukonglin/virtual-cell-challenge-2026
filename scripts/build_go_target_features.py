"""Build fold-specific, direct-only GO features from pinned public GAF/OBO.

No network calls, expression reads, alias guessing, ancestor propagation, pickle
loading, or response-model fitting. See official format/evidence references in
``REFERENCES``. The CLI requires an explicit training target list and fold ID;
do not run it on a real target list before the experiment protocol is frozen.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import zipfile

import numpy as np

from public_flow_conditioning import FeatureTable

REFERENCES = {
    "gaf": "https://geneontology.org/docs/go-annotation-file-gaf-format-2.2/",
    "evidence": "https://geneontology.org/docs/guide-go-evidence-codes/",
    "obo": "https://owlcollab.github.io/oboformat/doc/GO.format.obo-1_2.html",
    "license": "https://geneontology.org/docs/go-citation-policy/",
}
GO_ID = re.compile(r"GO:[0-9]{7}")
ASPECTS = {"biological_process": "P", "molecular_function": "F", "cellular_component": "C"}
KNOWN_EVIDENCE = frozenset("EXP IDA IPI IMP IGI IEP HTP HDA HMP HGI HEP IBA IBD IKR IRD ISS ISO ISA ISM IGC RCA TAS NAS IC ND IEA".split())


def _names(values, label):
    if isinstance(values, (str, bytes, dict, set, frozenset)) or values is None:
        raise ValueError(f"{label} must be a list of exact symbols")
    values = tuple(values)
    if not values or any(not isinstance(v, str) or not v or v.strip() != v for v in values):
        raise ValueError(f"Invalid {label}")
    if len(set(values)) != len(values):
        raise ValueError(f"Duplicate {label}")
    return values


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _sha_json(value):
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _file_sha(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected regular non-symlink file: {path}")
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


@dataclass(frozen=True)
class Ontology:
    active: dict[str, str]
    obsolete: frozenset[str]
    alternate: frozenset[str]
    data_version: str


def parse_obo(lines) -> Ontology:
    """Read only primary GO identity, aspect and obsolescence; ignore all edges."""
    active, obsolete, alternate, seen = {}, set(), set(), set()
    header, term = {}, None

    def finish():
        if term is None:
            return
        identifier = term.get("id")
        if not identifier:
            raise ValueError("OBO Term lacks an ID")
        if not identifier.startswith("GO:"):
            return
        if GO_ID.fullmatch(identifier) is None or identifier in seen:
            raise ValueError("Malformed or duplicate OBO primary GO ID")
        seen.add(identifier)
        alternate.update(term.get("alt_id", []))
        if term.get("is_obsolete", "false") == "true":
            obsolete.add(identifier)
        elif term.get("is_obsolete", "false") != "false":
            raise ValueError("Malformed OBO obsolescence flag")
        elif term.get("namespace") not in ASPECTS:
            raise ValueError("Active GO term lacks a recognized namespace")
        else:
            active[identifier] = ASPECTS[term["namespace"]]

    in_header = True
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("!"):
            continue
        if line.startswith("["):
            finish()
            term = {} if line == "[Term]" else None
            in_header = False
            continue
        if ":" not in line:
            raise ValueError("Malformed OBO tag line")
        key, value = line.split(":", 1)
        value = value.strip()
        if in_header and key in {"format-version", "data-version"}:
            if key in header:
                raise ValueError("Duplicate OBO version header")
            header[key] = value
        elif term is not None and key in {"id", "namespace", "is_obsolete", "alt_id"}:
            value = value.split(" !", 1)[0].split(" {", 1)[0].strip()
            if key == "alt_id":
                if GO_ID.fullmatch(value) is None:
                    raise ValueError("Malformed OBO alternate GO ID")
                term.setdefault(key, []).append(value)
            else:
                if key in term:
                    raise ValueError("Duplicate OBO identity field")
                term[key] = value
    finish()
    if header.get("format-version") != "1.2" or not header.get("data-version") or not active:
        raise ValueError("Require a nonempty versioned OBO 1.2 ontology")
    if alternate & seen:
        raise ValueError("Ambiguous primary/alternate GO identity")
    return Ontology(active, frozenset(obsolete), frozenset(alternate), header["data-version"])


@dataclass(frozen=True)
class AnnotationIndex:
    terms: dict[str, frozenset[str]]
    source_ids: dict[str, tuple[str, ...]]
    counts: dict[str, int]
    headers: dict[str, str]
    policy: dict


def parse_gaf(lines, ontology: Ontology, *, include_iea=False) -> AnnotationIndex:
    if type(include_iea) is not bool:
        raise ValueError("include_iea must be explicitly boolean")
    allowed = KNOWN_EVIDENCE - {"ND"} - (set() if include_iea else {"IEA"})
    counts, headers, terms, source_ids = Counter(), {}, defaultdict(set), defaultdict(set)
    for number, raw in enumerate(lines, 1):
        line = raw.rstrip("\r\n")
        if not line:
            continue
        if line.startswith("!"):
            if ":" in line:
                key, value = line[1:].split(":", 1)
                key, value = key.strip(), value.strip()
                if key in {"gaf-version", "generated-by", "date-generated", "go-version"}:
                    if key in headers and headers[key] != value:
                        raise ValueError("Contradictory GAF headers")
                    headers[key] = value
            continue
        if headers.get("gaf-version") != "2.2":
            raise ValueError("Require GAF 2.2 header before annotations")
        fields = line.split("\t")
        if len(fields) != 17:
            raise ValueError(f"GAF line {number} does not have 17 fields")
        counts["annotation_rows"] += 1
        database, object_id, symbol, relation, go_id = fields[:5]
        evidence, aspect, taxon = fields[6], fields[8], fields[12]
        if database != "UniProtKB":
            counts["excluded_database"] += 1
            continue
        if taxon != "taxon:9606":
            counts["excluded_nonsole_human_taxon"] += 1
            continue
        _names((symbol,), "GAF symbol")
        if not object_id or object_id.strip() != object_id:
            raise ValueError("Missing/malformed source object ID")
        source_ids[symbol].add(database + ":" + object_id)
        qualifiers = relation.split("|")
        if "NOT" in qualifiers:
            counts["excluded_NOT"] += 1
            continue
        if len(qualifiers) != 1 or not relation or relation.strip() != relation:
            raise ValueError("Malformed positive GAF relation")
        if evidence not in allowed:
            counts["excluded_evidence_" + (evidence if evidence in KNOWN_EVIDENCE else "unknown")] += 1
            continue
        if go_id in ontology.obsolete:
            counts["excluded_obsolete_GO"] += 1
            continue
        if go_id in ontology.alternate:
            counts["excluded_alternate_GO"] += 1
            continue
        if go_id not in ontology.active:
            counts["excluded_unresolved_GO"] += 1
            continue
        if aspect != ontology.active[go_id]:
            counts["excluded_aspect_mismatch"] += 1
            continue
        counts["accepted_annotation_rows"] += 1
        if go_id in terms[symbol]:
            counts["duplicate_symbol_term_rows"] += 1
        terms[symbol].add(go_id)
    if any(not headers.get(key) for key in ("gaf-version", "generated-by", "date-generated")):
        raise ValueError("Missing mandatory GAF headers")
    if headers["gaf-version"] != "2.2":
        raise ValueError("Unsupported GAF version")
    if "go-version" in headers:
        match = re.search(r"/go/(releases/[^/]+)/", headers["go-version"])
        if match is None or match.group(1) != ontology.data_version:
            raise ValueError("GAF/OBO ontology release mismatch")
    counts["unique_symbol_term_pairs"] = sum(map(len, terms.values()))
    counts["human_symbols_seen"] = len(source_ids)
    counts["symbols_with_multiple_source_objects"] = sum(len(ids) > 1 for ids in source_ids.values())
    policy = {
        "schema": "vcc-go-direct-policy-v1", "ancestor_policy": "direct_only",
        "gene_mapping": "exact_case_sensitive_GAF_column_3_no_aliases",
        "go_mapping": "primary_active_ids_only_no_alt_or_obsolete_replacement",
        "taxon_policy": "sole_taxon_9606_excludes_interspecies_annotations",
        "evidence_policy": "recognized_codes_except_ND_with_explicit_IEA_choice",
        "include_iea": include_iea, "accepted_evidence_codes": sorted(allowed),
        "negation_policy": "exclude_NOT_rows", "duplicate_policy": "binary_symbol_term_union",
        "relation_policy": "collapse_positive_relations", "annotation_extensions": "ignored",
        "ontology_data_version": ontology.data_version,
    }
    return AnnotationIndex({s: frozenset(t) for s, t in terms.items()},
                           {s: tuple(sorted(ids)) for s, ids in source_ids.items()},
                           dict(sorted(counts.items())), headers, policy)


@dataclass(frozen=True)
class GOFeatures:
    table: FeatureTable
    arrays: dict[str, np.ndarray]
    receipt: dict


def build_features(index: AnnotationIndex, *, target_ids, train_target_ids, fold_id,
                   sources: dict, max_dense_elements=50_000_000) -> GOFeatures:
    targets = _names(target_ids, "target_ids")
    train = _names(train_target_ids, "train_target_ids")
    if not set(train) <= set(targets):
        raise ValueError("Training target IDs must be explicitly present on the requested axis")
    if not isinstance(fold_id, str) or not fold_id or fold_id.strip() != fold_id:
        raise ValueError("Explicit fold_id required")
    if set(sources) != {"gaf_sha256", "obo_sha256", "acquisition_sha256"} or any(
            not isinstance(s, str) or re.fullmatch(r"[0-9a-f]{64}", s) is None for s in sources.values()):
        raise ValueError("All authenticated source hashes are required")
    vocabulary = tuple(sorted({term for symbol in train for term in index.terms.get(symbol, ())}))
    if not vocabulary:
        raise ValueError("Training targets have no accepted GO terms; do not fabricate a vocabulary")
    if type(max_dense_elements) is not int or max_dense_elements < 1 or len(targets) * len(vocabulary) > max_dense_elements:
        raise ValueError("Dense GO feature allocation exceeds the explicit bound")
    columns = {term: i for i, term in enumerate(vocabulary)}
    features = np.zeros((len(targets), len(vocabulary)), dtype=np.uint8)
    for row, symbol in enumerate(targets):
        for term in index.terms.get(symbol, ()):
            if term in columns:
                features[row, columns[term]] = 1
    arrays = {
        "target_ids": np.asarray(targets, dtype=np.str_), "go_ids": np.asarray(vocabulary, dtype=np.str_),
        "train_target_ids": np.asarray(train, dtype=np.str_), "features": features,
        "known_symbol": np.array([s in index.source_ids for s in targets], dtype=bool),
        "has_accepted_annotation": np.array([bool(index.terms.get(s)) for s in targets], dtype=bool),
        "present": features.any(axis=1),
        "accepted_term_count": np.array([len(index.terms.get(s, ())) for s in targets], dtype=np.int64),
        "in_vocab_term_count": features.sum(axis=1, dtype=np.int64),
        "source_object_count": np.array([len(index.source_ids.get(s, ())) for s in targets], dtype=np.int64),
    }
    identity = {"sources": sources, "policy": index.policy, "fold_id": fold_id,
                "train_target_ids_sha256": _sha_json(train), "vocabulary_sha256": _sha_json(vocabulary)}
    covered = arrays["present"]
    table = FeatureTable("go", tuple(s for s, keep in zip(targets, covered) if keep), vocabulary,
                         features[covered], "GO-direct:" + index.policy["ontology_data_version"] + ":" + fold_id,
                         _sha_json(identity))
    for array in arrays.values():
        array.setflags(write=False)
    receipt = {"schema": "vcc-go-target-features-v1", "status": "built_features_only", **identity,
               "target_ids_sha256": _sha_json(targets), "feature_table_fingerprint": table.fingerprint,
               "feature_table_source_id": table.source_id, "feature_table_source_sha256": table.source_sha256,
               "target_count": len(targets), "train_target_count": len(train), "vocabulary_size": len(vocabulary),
               "vocabulary_go_ids": vocabulary, "gaf_headers": index.headers, "annotation_counts": index.counts,
               "coverage_counts": {key: int(arrays[key].sum()) for key in
                                   ("known_symbol", "has_accepted_annotation", "present")},
               "missing_in_vocab_targets": [s for s, keep in zip(targets, covered) if not keep],
               "source_objects_by_target": {s: index.source_ids.get(s, ()) for s in targets},
               "feature_dtype": "uint8_binary", "feature_table_missing_rows": "omitted_then_explicit_alignment_mask",
               "vocabulary_fit_scope": "explicit_training_targets_only", "expression_read": False,
               "response_model_trained": False, "references": REFERENCES, "license": "CC-BY-4.0"}
    return GOFeatures(table, arrays, receipt)


def authenticated_sources(acquisition_path: Path, expected_sha256: str):
    if _file_sha(acquisition_path) != expected_sha256:
        raise ValueError("Acquisition receipt SHA-256 mismatch")
    receipt = json.loads(acquisition_path.read_text())
    if receipt.get("schema") != "vcc-public-feature-acquisition-v1":
        raise ValueError("Unsupported acquisition receipt")
    names = {"HUMAN-uniprot.gaf.gz": "gaf_sha256", "go-basic.obo": "obo_sha256"}
    records = receipt.get("files")
    if not isinstance(records, list) or len(records) != 2 or any(not isinstance(r, dict) for r in records):
        raise ValueError("Require exactly the two registered GO source files")
    if {r.get("path") for r in records} != set(names):
        raise ValueError("Unexpected GO source filename")
    hashes = {"acquisition_sha256": expected_sha256}
    for record in records:
        path = acquisition_path.parent / record["path"]
        if type(record.get("size_bytes")) is not int or record["size_bytes"] <= 0:
            raise ValueError("Invalid source size")
        digest = _file_sha(path)
        if path.stat().st_size != record["size_bytes"] or digest != record.get("sha256"):
            raise ValueError("Registered GO source size/hash mismatch")
        hashes[names[record["path"]]] = digest
    return hashes


def write_artifacts(result: GOFeatures, output_dir: Path):
    """Create a new directory; a receipt is published only after safe NPZ bytes."""
    if os.path.lexists(output_dir):
        raise FileExistsError("Refusing to overwrite an existing feature artifact directory")
    output_dir.mkdir(mode=0o700)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, array in sorted(result.arrays.items()):
            member = io.BytesIO()
            np.lib.format.write_array(member, array, allow_pickle=False)
            entry = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o600 << 16
            archive.writestr(entry, member.getvalue())
    npz = buffer.getvalue()
    receipt = {**result.receipt, "npz_filename": "features.npz", "npz_size_bytes": len(npz),
               "npz_sha256": hashlib.sha256(npz).hexdigest()}
    for name, payload in (("features.npz", npz), ("receipt.json", _json_bytes(receipt))):
        with os.fdopen(os.open(output_dir / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisition", type=Path, required=True)
    parser.add_argument("--acquisition-sha256", required=True)
    parser.add_argument("--targets-json", type=Path, required=True, help="JSON list of exact target symbols")
    parser.add_argument("--train-targets-json", type=Path, required=True, help="Frozen training-only JSON symbol list")
    parser.add_argument("--fold-id", required=True)
    parser.add_argument("--include-iea", action="store_true", help="Explicit ablation; default excludes IEA and ND")
    parser.add_argument("--max-dense-elements", type=int, default=50_000_000)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    sources = authenticated_sources(args.acquisition, args.acquisition_sha256)
    with (args.acquisition.parent / "go-basic.obo").open() as stream:
        ontology = parse_obo(stream)
    with gzip.open(args.acquisition.parent / "HUMAN-uniprot.gaf.gz", "rt") as stream:
        index = parse_gaf(stream, ontology, include_iea=args.include_iea)
    result = build_features(index, target_ids=json.loads(args.targets_json.read_text()),
                            train_target_ids=json.loads(args.train_targets_json.read_text()),
                            fold_id=args.fold_id, sources=sources, max_dense_elements=args.max_dense_elements)
    receipt = write_artifacts(result, args.output_dir)
    print(json.dumps({key: receipt[key] for key in ("status", "target_count", "vocabulary_size", "coverage_counts", "npz_sha256")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
