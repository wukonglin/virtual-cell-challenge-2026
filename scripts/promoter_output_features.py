"""Independent expression-free promoter composition features for output genes.

NOT PromoterAI, learned embeddings, TF binding predictions, or a CRISPRi model.
No Illumina code, weights, scores, or licensed assets. Exact GENCODE symbols and
versioned IDs only. Ambiguous loci, versions and TSSs remain explicitly missing.
GTF is 1-based closed; fetch intervals are genomic-plus 0-based half-open.
Windows are [-upstream, downstream) relative to TSS in transcription direction.
Minus-strand DNA is reverse complemented exactly once here. Feature computation
is offline; optional registered CLI acquisition uses public GENCODE/Ensembl
endpoints only. No expression, alias matching, active-TSS inference, or axis changes.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np


SCHEMA = "independent-promoter-composition-v1"
FEATURE_IDS = ("gc_fraction_valid_bases", "cpg_fraction_valid_dinucleotides",
               "cpg_observed_expected", "ambiguous_base_fraction")
PRIMARY_CHROMS = frozenset([f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY", "chrM"])
_ATTRIBUTE = re.compile(r'\s*([^\s;]+)\s+(?:"([^"\r\n]*)"|([0-9]+))\s*;')
_COMPLEMENT = str.maketrans("ACGTRYSWKMBDHVN", "TGCAYRSWMKVHDBN")
_DNA = frozenset("ACGTRYSWKMBDHVN")


def _sha(value: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("Expected lowercase SHA-256")
    return value


def _canonical(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _axis(symbols: Sequence[str]) -> tuple[str, ...]:
    symbols = tuple(symbols)
    if (not symbols or any(not isinstance(s, str) or not s or s.strip() != s for s in symbols)
            or len(set(symbols)) != len(symbols)):
        raise ValueError("Output axis must contain unique nonempty exact symbols")
    return symbols


def axis_sha256(gene_symbols: Sequence[str]) -> str:
    return hashlib.sha256(_canonical(list(_axis(gene_symbols)))).hexdigest()


@dataclass(frozen=True)
class Transcript:
    gene_symbol: str
    gene_id: str
    transcript_id: str
    chrom: str
    start1: int
    end1: int
    strand: str
    mane_select: bool = False

    def __post_init__(self):
        if not self.gene_symbol or self.gene_symbol.strip() != self.gene_symbol:
            raise ValueError("Invalid gene symbol")
        for value, prefix in ((self.gene_id, "ENSG"), (self.transcript_id, "ENST")):
            if re.fullmatch(prefix + r"[0-9]{11}\.[1-9][0-9]*(?:_PAR_Y)?", value) is None:
                raise ValueError("Require explicit versioned human Ensembl IDs")
        if (self.strand not in ("+", "-") or not self.chrom or self.chrom.strip() != self.chrom
                or type(self.start1) is not int or type(self.end1) is not int
                or self.start1 < 1 or self.end1 < self.start1 or type(self.mane_select) is not bool):
            raise ValueError("Invalid transcript coordinates/strand")

    @property
    def tss0(self) -> int:
        return self.start1 - 1 if self.strand == "+" else self.end1 - 1


def parse_gtf(lines: Iterable[str], *, requested_symbols: Sequence[str]) -> tuple[Transcript, ...]:
    """Parse transcript annotations only; requested malformed records fail closed.

    Only the official multivalued tag, ont and ccdsid attributes may repeat.
    Repeated transcript IDs, duplicate identity attributes and missing version
    suffixes are forbidden. Unrequested rows are ignored after syntax checks.
    """
    requested = set(_axis(requested_symbols))
    transcripts, seen_ids = [], set()
    for line in lines:
        if line.startswith("#") or not line.strip():
            continue
        parts = line.rstrip("\r\n").split("\t")
        if len(parts) != 9:
            raise ValueError("GTF requires nine tab-separated fields")
        if parts[2] != "transcript":
            continue
        attrs, pos = {}, 0
        while pos < len(parts[8]):
            match = _ATTRIBUTE.match(parts[8], pos)
            if match is None:
                if not parts[8][pos:].strip():
                    break
                raise ValueError("Malformed GTF attributes")
            key, value, numeric = match.groups()
            if numeric is not None:
                if key != "level" or numeric not in {"1", "2", "3"}:
                    raise ValueError("Only GENCODE numeric level 1/2/3 may be unquoted")
                value = numeric
            if key in attrs and key not in {"tag", "ont", "ccdsid"}:
                raise ValueError("Duplicate GTF attribute")
            attrs.setdefault(key, []).append(value)
            pos = match.end()
        symbol = attrs.get("gene_name", [None])[0]
        if symbol not in requested:
            continue
        if not {"gene_id", "transcript_id"} <= attrs.keys():
            raise ValueError("Missing requested transcript identifiers")
        transcript = Transcript(symbol, attrs["gene_id"][0], attrs["transcript_id"][0],
                                parts[0], int(parts[3]), int(parts[4]), parts[6],
                                "MANE_Select" in attrs.get("tag", []))
        if transcript.transcript_id in seen_ids:
            raise ValueError("Duplicate transcript ID")
        seen_ids.add(transcript.transcript_id)
        transcripts.append(transcript)
    return tuple(transcripts)


def select_tss(symbol: str, transcripts: Sequence[Transcript]) -> tuple[Transcript | None, str]:
    """Unique exact locus, then unique MANE TSS or consensus annotated TSS.

    A reference MANE choice is not the cell context's observed active TSS.
    Transcripts sharing a TSS use smallest versioned ID as audit representative,
    not biological ranking. No expression-based or unversioned-ID selection.
    """
    candidates = [t for t in transcripts if t.gene_symbol == symbol]
    if not candidates:
        return None, "missing_exact_symbol"
    ids = {t.gene_id for t in candidates}
    if len(ids) != 1:
        reason = "ambiguous_gene_version" if len({v.split(".")[0] for v in ids}) == 1 else "ambiguous_gene_symbol"
        return None, reason
    if len({(t.chrom, t.strand) for t in candidates}) != 1:
        return None, "ambiguous_locus"
    if candidates[0].chrom not in PRIMARY_CHROMS:
        return None, "nonprimary_contig"
    mane = [t for t in candidates if t.mane_select]
    pool = mane or candidates
    if len({t.tss0 for t in pool}) != 1:
        return None, "ambiguous_mane_tss" if mane else "ambiguous_tss"
    return min(pool, key=lambda t: t.transcript_id), "mane_select_tss" if mane else "consensus_tss"


def promoter_interval(transcript: Transcript, *, chrom_size: int,
                      upstream: int = 1000, downstream: int = 100) -> tuple[int, int]:
    if (type(chrom_size) is not int or chrom_size < 1 or type(upstream) is not int
            or type(downstream) is not int or upstream < 1 or downstream < 1):
        raise ValueError("Window and chromosome bounds must be positive integers")
    if transcript.end1 > chrom_size:
        raise ValueError("Transcript outside reference chromosome")
    if transcript.strand == "+":
        start, end = transcript.tss0 - upstream, transcript.tss0 + downstream
    else:
        start, end = transcript.tss0 - downstream + 1, transcript.tss0 + upstream + 1
    if start < 0 or end > chrom_size:
        raise ValueError("Promoter crosses chromosome boundary; padding/clamping forbidden")
    return start, end


def orient_sequence(sequence: str, strand: str, *, expected_length: int) -> str:
    if not isinstance(sequence, str) or len(sequence) != expected_length or strand not in ("+", "-"):
        raise ValueError("Sequence length/strand mismatch")
    sequence = sequence.upper()
    if not set(sequence) <= _DNA:
        raise ValueError("Sequence must use IUPAC DNA without gaps/whitespace")
    return sequence if strand == "+" else sequence.translate(_COMPLEMENT)[::-1]


def sequence_features(sequence: str) -> tuple[np.ndarray, np.ndarray]:
    """GC/CpG with denominator-specific validity masks.

    CpG O/E = (#CG / valid adjacent pairs) / (p(C)*p(G)). Ambiguity breaks pairs:
    CNG has no CG. These descriptors are not methylation or variant scores.
    """
    sequence = orient_sequence(sequence, "+", expected_length=len(sequence))
    if not sequence:
        raise ValueError("Empty sequence")
    n = sum(base in "ACGT" for base in sequence)
    c, g = sequence.count("C"), sequence.count("G")
    pairs = sum(a in "ACGT" and b in "ACGT" for a, b in zip(sequence, sequence[1:]))
    cpg = sequence.count("CG")
    values = np.zeros(len(FEATURE_IDS), dtype=np.float32)
    mask = np.array([n > 0, pairs > 0, pairs > 0 and c > 0 and g > 0, True], dtype=bool)
    if mask[0]:
        values[0] = (c + g) / n
    if mask[1]:
        values[1] = cpg / pairs
    if mask[2]:
        values[2] = cpg * n * n / (pairs * c * g)
    values[3] = 1 - n / len(sequence)
    return values, mask


@dataclass(frozen=True)
class PromoterFeatures:
    gene_symbols: tuple[str, ...]
    values: np.ndarray
    valid: np.ndarray
    present: np.ndarray
    receipt: dict


def build_features(*, gene_symbols: Sequence[str], transcripts: Sequence[Transcript],
                   chrom_sizes: Mapping[str, int], fetch_plus: Callable[[str, int, int], str],
                   assembly: str, annotation_sha256: str, reference_sha256: str,
                   contract_sha256: str, upstream: int = 1000, downstream: int = 100,
                   max_ambiguous_fraction: float = 0.2) -> PromoterFeatures:
    """Fixed features in frozen output-gene order.

    reference_sha256 binds a coordinate-bearing DNA bundle (or verified manifest).
    fetch_plus returns genomic-plus DNA from it. Acquisition/authentication is a
    separate caller responsibility: hashes here record provenance and do not
    claim verification of arbitrary callback data or license acceptance.
    """
    genes = _axis(gene_symbols)
    if assembly != "GRCh38":
        raise ValueError("Only explicitly pinned GRCh38 supported; no implicit liftover")
    if not np.isfinite(max_ambiguous_fraction) or not 0 <= max_ambiguous_fraction < 1:
        raise ValueError("Invalid ambiguous-base threshold")
    if type(upstream) is not int or type(downstream) is not int or min(upstream, downstream) < 1:
        raise ValueError("Invalid promoter window")
    sources = {k: _sha(v) for k, v in {
        "annotation_sha256": annotation_sha256, "reference_sha256": reference_sha256,
        "contract_sha256": contract_sha256}.items()}
    values = np.zeros((len(genes), len(FEATURE_IDS)), dtype=np.float32)
    valid = np.zeros_like(values, dtype=bool)
    present, records = np.zeros(len(genes), dtype=bool), []
    for i, symbol in enumerate(genes):
        selected, reason = select_tss(symbol, transcripts)
        record = {"gene_symbol": symbol, "selection": reason, "present": False}
        records.append(record)
        if selected is None:
            continue
        record["transcript"] = asdict(selected)
        if selected.chrom not in chrom_sizes:
            record["missing_reason"] = "missing_reference_chromosome"
            continue
        try:
            start, end = promoter_interval(selected, chrom_size=chrom_sizes[selected.chrom],
                                          upstream=upstream, downstream=downstream)
        except ValueError as exc:
            record["missing_reason"] = str(exc)
            continue
        plus = fetch_plus(selected.chrom, start, end)
        sequence = orient_sequence(plus, selected.strand, expected_length=upstream + downstream)
        record.update(start0=start, end0=end, tss0=selected.tss0,
                      oriented_sequence_sha256=hashlib.sha256(sequence.encode()).hexdigest())
        row, row_mask = sequence_features(sequence)
        record["ambiguous_fraction"] = float(row[3])
        if row[3] > max_ambiguous_fraction:
            record["missing_reason"] = "excess_ambiguous_bases"
            continue
        values[i], valid[i], present[i], record["present"] = row, row_mask, True, True
    receipt = {"schema": SCHEMA, "feature_family": "independent_dna_composition_not_promoterai",
               "assembly": assembly, "gene_axis_sha256": axis_sha256(genes),
               "feature_ids": list(FEATURE_IDS), "sources": sources,
               "policy": {"upstream": upstream, "downstream": downstream,
                          "window": "transcription-relative [-upstream, downstream)",
                          "sequence_input": "genomic_plus_0based_halfopen",
                          "tss": "unique_mane_select_else_consensus",
                          "symbol_matching": "exact_no_aliases_no_version_stripping",
                          "max_ambiguous_fraction": max_ambiguous_fraction,
                          "expression_accessed": False, "fit_performed": False},
               "genes": records, "present_count": int(present.sum())}
    return PromoterFeatures(genes, values, valid, present, receipt)


def save_npz(path, features: PromoterFeatures) -> str:
    """No-clobber numeric/string artifact, readable with allow_pickle=False.

    Embedded receipt binds exact axis and sequence digests. Later consumers must
    authenticate returned hash, frozen contract and gene-axis hash.
    """
    with open(path, "xb") as stream:
        np.savez_compressed(stream, gene_symbols=np.array(features.gene_symbols, dtype=str),
                            feature_ids=np.array(FEATURE_IDS, dtype=str), values=features.values,
                            valid=features.valid, present=features.present,
                            receipt_json=np.array(_canonical(features.receipt).decode(), dtype=str))
    result = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def ensembl_region(chrom: str, start0: int, end0: int) -> str:
    """Explicit primary-chromosome naming map, not biological ID alias guessing."""
    if (chrom not in PRIMARY_CHROMS or type(start0) is not int or type(end0) is not int
            or start0 < 0 or end0 <= start0 or end0 - start0 > 1100):
        raise ValueError("Region must be a bounded primary-chromosome promoter")
    name = "MT" if chrom == "chrM" else chrom[3:]
    return f"{name}:{start0 + 1}..{end0}:1"


def sequence_batches(intervals: Sequence[tuple[str, int, int]]) -> list[list[str]]:
    """At most 512 unique 1,100bp regions, 50/request, no full-genome fallback."""
    if len(intervals) > 512:
        raise ValueError("Promoter campaign exceeds 512 windows")
    regions = list(dict.fromkeys(ensembl_region(*item) for item in intervals))
    return [regions[i:i+50] for i in range(0, len(regions), 50)]


def validate_ensembl_response(payload, regions: Sequence[str]) -> dict[str, str]:
    """Join by exact query, verify GRCh38/coordinates/strand/type, never by order."""
    if (not isinstance(payload, list) or not 1 <= len(regions) <= 50
            or len(set(regions)) != len(regions) or len(payload) != len(regions)):
        raise ValueError("Wrong batched sequence response shape")
    expected = set(regions)
    result = {}
    for item in payload:
        if not isinstance(item, dict) or item.get("query") not in expected or item["query"] in result:
            raise ValueError("Missing/duplicate/foreign sequence query")
        query = item["query"]
        match = re.fullmatch(r"([0-9]+|X|Y|MT):([0-9]+)\.\.([0-9]+):1", query)
        if match is None:
            raise ValueError("Invalid expected region")
        chrom, start, end = match.groups()
        # The endpoint's id explicitly reports coordinate system and assembly.
        expected_id = f"chromosome:GRCh38:{chrom}:{start}:{end}:1"
        if item.get("id") != expected_id or item.get("molecule") != "dna":
            raise ValueError("Assembly/coordinates/orientation/molecule mismatch")
        result[query] = orient_sequence(item.get("seq"), "+", expected_length=int(end)-int(start)+1)
    return result


# Fixed, public reference acquisition. No cloud credentials or model licenses.
ANNOTATION_URL = "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_48/gencode.v48.basic.annotation.gtf.gz"
ANNOTATION_BYTES = 33712235
ANNOTATION_MD5 = "a8ac6cd463006654d253fd8cb29d1913"
ASSEMBLY_URL = "https://rest.ensembl.org/info/assembly/homo_sapiens?content-type=application/json"
SEQUENCE_URL = "https://rest.ensembl.org/sequence/region/homo_sapiens?coord_system_version=GRCh38;coord_system=chromosome"


def _read_regular(path, *, cap: int) -> bytes:
    import os
    import stat
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("Expected a regular file")
        data = stream.read(cap + 1)
    if len(data) > cap:
        raise ValueError("File exceeds size bound")
    return data


def _json(data: bytes):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result
    return json.loads(data, object_pairs_hook=pairs)


def _write_new(path, data: bytes):
    with open(path, "xb") as stream:
        stream.write(data)


def register_reference(contract_path, *, contract_sha256: str, directory, reuse_annotation=None):
    """Freeze the exact axis, mapping rules, byte/call caps before acquisition."""
    from pathlib import Path
    raw = _read_regular(contract_path, cap=2_000_000)
    if hashlib.sha256(raw).hexdigest() != _sha(contract_sha256):
        raise ValueError("Frozen row contract hash mismatch")
    contract = _json(raw)
    genes = _axis(contract["output_genes"])
    if len(genes) > 512 or contract.get("schema") != "public-flow-row-contract-v1":
        raise ValueError("Wrong pilot axis or contract schema")
    plan = {"schema": "independent-promoter-reference-plan-v1", "assembly": "GRCh38",
            "generator_sha256": hashlib.sha256(_read_regular(__file__, cap=1_000_000)).hexdigest(),
            "contract_sha256": contract_sha256, "output_genes": list(genes),
            "gene_axis_sha256": axis_sha256(genes),
            "annotation": {"url": ANNOTATION_URL, "size_bytes": ANNOTATION_BYTES,
                           "md5": ANNOTATION_MD5,
                           "md5_source": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_48/MD5SUMS"},
            "assembly_url": ASSEMBLY_URL, "sequence_url": SEQUENCE_URL,
            "caps": {"annotation_bytes": 35_000_000, "api_bytes": 1_000_000,
                     "regions": 512, "region_bases": 1100, "regions_per_post": 50,
                     "api_requests": 12, "automatic_retries": 0},
            "policy": {"upstream": 1000, "downstream": 100,
                       "tss": "unique_mane_select_else_consensus", "symbols": "exact_only",
                       "ambiguous_loci_versions_tss": "missing_not_guessed",
                       "sequence_input": "genomic_plus_0based_halfopen",
                       "max_ambiguous_fraction": 0.2, "expression_accessed": False,
                       "paid_services": False, "gcp_used": False,
                       "licensed_model_assets": False},
            "primary_references": ["https://www.gencodegenes.org/human/release_48.html",
                "https://www.gencodegenes.org/pages/data_access.html",
                "https://www.ebi.ac.uk/about/terms-of-use/",
                "https://www.ensembl.org/info/about/legal/disclaimer.html",
                "https://rest.ensembl.org/documentation/info/sequence_region_post"]}
    if reuse_annotation is not None:
        cached = _read_regular(reuse_annotation, cap=ANNOTATION_BYTES)
        if len(cached) != ANNOTATION_BYTES or hashlib.md5(cached).hexdigest() != ANNOTATION_MD5:
            raise ValueError("Cached annotation differs from publisher checksum")
        plan["cached_annotation"] = {"path": str(Path(reuse_annotation).absolute()),
                                     "sha256": hashlib.sha256(cached).hexdigest()}
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    _write_new(directory / "plan.json", _canonical(plan))
    return plan


def validate_assembly_response(payload) -> dict[str, int]:
    if not isinstance(payload, dict) or payload.get("assembly_name") != "GRCh38.p14":
        raise ValueError("Reference metadata is not pinned GRCh38.p14")
    result = {}
    for item in payload.get("top_level_region", []):
        name = item.get("name")
        chrom = "chrM" if name == "MT" else "chr" + str(name)
        if chrom not in PRIMARY_CHROMS:
            continue
        if (item.get("coord_system") != "chromosome" or chrom in result
                or type(item.get("length")) is not int or item["length"] < 1100):
            raise ValueError("Invalid/duplicate primary chromosome metadata")
        result[chrom] = item["length"]
    if set(result) != PRIMARY_CHROMS:
        raise ValueError("Incomplete primary chromosome size metadata")
    return result


def acquire_reference(plan_path, *, plan_sha256: str):
    """Bounded fresh acquisition; any HTTP/schema failure stops without retry.

    Restarting this same campaign is deliberately forbidden after started.json;
    inspect its receipts and authorize an explicit recovery plan instead. Each
    response is retained with SHA-256 and GTF also has the publisher MD5 check.
    No checkpoint, expression or GCP operations occur.
    """
    import gzip
    import io
    from pathlib import Path
    import time
    import urllib.request
    raw = _read_regular(plan_path, cap=2_000_000)
    if hashlib.sha256(raw).hexdigest() != _sha(plan_sha256):
        raise ValueError("Reference plan hash mismatch")
    plan = _json(raw)
    if plan.get("generator_sha256") != hashlib.sha256(_read_regular(__file__, cap=1_000_000)).hexdigest():
        raise ValueError("Reference acquisition implementation changed after registration")
    if (plan.get("schema") != "independent-promoter-reference-plan-v1"
            or plan["annotation"] != {"url": ANNOTATION_URL, "size_bytes": ANNOTATION_BYTES,
                 "md5": ANNOTATION_MD5,
                 "md5_source": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_48/MD5SUMS"}
            or plan["assembly_url"] != ASSEMBLY_URL or plan["sequence_url"] != SEQUENCE_URL
            or plan["caps"] != {"annotation_bytes": 35_000_000, "api_bytes": 1_000_000,
                "regions": 512, "region_bases": 1100, "regions_per_post": 50,
                "api_requests": 12, "automatic_retries": 0}
            or plan["assembly"] != "GRCh38"):
        raise ValueError("Unsupported reference acquisition plan")
    genes = _axis(plan["output_genes"])
    if len(genes) > 512 or axis_sha256(genes) != plan["gene_axis_sha256"]:
        raise ValueError("Reference axis mismatch")
    directory = Path(plan_path).parent
    _write_new(directory / "started.json", _canonical({"plan_sha256": plan_sha256,
               "automatic_retries": 0, "started_unix_seconds": time.time()}))
    records, api_bytes, api_requests, last_api_time = [], 0, 0, 0.0

    def fetch(url, filename, cap, body=None):
        nonlocal api_bytes, api_requests, last_api_time
        is_api = url in (ASSEMBLY_URL, SEQUENCE_URL)
        if url not in (ANNOTATION_URL, ASSEMBLY_URL, SEQUENCE_URL):
            raise ValueError("Unapproved reference endpoint")
        if is_api:
            if api_requests >= 12:
                raise ValueError("API request cap reached")
            time.sleep(max(0, last_api_time + 1 - time.monotonic()))
            cap = min(cap, 1_000_000 - api_bytes)
            api_requests += 1
        request = urllib.request.Request(url, data=body,
                  headers={"Accept": "application/json" if is_api else "application/gzip",
                           "Content-Type": "application/json", "User-Agent": "VCC-public-promoter-reference/1"})
        with urllib.request.urlopen(request, timeout=45) as response:
            if response.status != 200 or response.geturl() != url:
                raise ValueError("Unexpected response status or redirect")
            content_length = response.headers.get("Content-Length")
            if content_length is not None and int(content_length) > cap:
                raise ValueError("HTTP content exceeds request byte cap")
            data = response.read(cap + 1)
            if len(data) > cap:
                raise ValueError("Response exceeds byte cap")
            rate_remaining = response.headers.get("X-RateLimit-Remaining")
            retry_after = response.headers.get("Retry-After")
        if is_api:
            api_bytes += len(data)
            last_api_time = time.monotonic()
        record = {"filename": filename, "url": url, "size_bytes": len(data),
                  "sha256": hashlib.sha256(data).hexdigest(),
                  "request_sha256": hashlib.sha256(body).hexdigest() if body else None}
        _write_new(directory / filename, data)
        _write_new(directory / (filename + ".receipt.json"), _canonical(record))
        records.append(record)
        if (rate_remaining is not None and float(rate_remaining) <= 0) or retry_after is not None:
            raise ValueError("Server requests rate-limit backoff; campaign stopped without retry")
        return data

    if "cached_annotation" in plan:
        cached = plan["cached_annotation"]
        annotation = _read_regular(cached["path"], cap=ANNOTATION_BYTES)
        if hashlib.sha256(annotation).hexdigest() != _sha(cached["sha256"]):
            raise ValueError("Cached annotation hash changed")
        records.append({"filename": cached["path"], "url": ANNOTATION_URL,
                        "size_bytes": len(annotation), "sha256": cached["sha256"],
                        "reuse_verified_local": True, "transferred_bytes": 0})
    else:
        annotation = fetch(ANNOTATION_URL, "gencode.v48.basic.annotation.gtf.gz", ANNOTATION_BYTES)
    if len(annotation) != ANNOTATION_BYTES or hashlib.md5(annotation).hexdigest() != ANNOTATION_MD5:
        raise ValueError("Annotation size/publisher MD5 mismatch")
    annotation_sha = hashlib.sha256(annotation).hexdigest()
    with gzip.GzipFile(fileobj=io.BytesIO(annotation)) as gz:
        with io.TextIOWrapper(gz) as stream:
            transcripts = parse_gtf(stream, requested_symbols=genes)
    chrom_sizes = validate_assembly_response(_json(fetch(ASSEMBLY_URL, "assembly.json", 100_000)))
    intervals, mapping = [], []
    for gene in genes:
        selected, reason = select_tss(gene, transcripts)
        entry = {"symbol": gene, "selection": reason}
        if selected is not None:
            entry["transcript"] = asdict(selected)
            try:
                start, end = promoter_interval(selected, chrom_size=chrom_sizes[selected.chrom])
                intervals.append((selected.chrom, start, end))
                entry["region"] = ensembl_region(selected.chrom, start, end)
            except ValueError as exc:
                entry["missing_reason"] = str(exc)
        mapping.append(entry)
    batches = sequence_batches(intervals)
    # Freeze exact sequence queries before requesting any DNA bodies.
    sequence_plan = {"plan_sha256": plan_sha256, "annotation_sha256": annotation_sha,
                     "mapping": mapping, "batches": batches, "chrom_sizes": chrom_sizes}
    _write_new(directory / "sequence_plan.json", _canonical(sequence_plan))
    sequences = {}
    for i, regions in enumerate(batches):
        body = _canonical({"regions": regions})
        response = fetch(SEQUENCE_URL, f"sequence_batch_{i:02d}.json", 100_000, body)
        sequences.update(validate_ensembl_response(_json(response), regions))
    bundle = {"schema": "grch38-promoter-plus-sequences-v1", "sequences": sequences,
              "plan_sha256": plan_sha256,
              "sequence_plan_sha256": hashlib.sha256(_canonical(sequence_plan)).hexdigest(),
              "response_receipts": records[1:]}
    bundle_bytes = _canonical(bundle)
    _write_new(directory / "sequence_bundle.json", bundle_bytes)
    features = build_features(gene_symbols=genes, transcripts=transcripts, chrom_sizes=chrom_sizes,
        fetch_plus=lambda chrom, start, end: sequences[ensembl_region(chrom, start, end)],
        assembly="GRCh38", annotation_sha256=annotation_sha,
        reference_sha256=hashlib.sha256(bundle_bytes).hexdigest(), contract_sha256=plan["contract_sha256"])
    feature_sha = save_npz(directory / "features.npz", features)
    receipt = {"schema": "independent-promoter-reference-complete-v1", "plan_sha256": plan_sha256,
               "files": records, "api_requests": api_requests, "api_bytes": api_bytes,
               "total_response_bytes": sum(r.get("transferred_bytes", r["size_bytes"]) for r in records),
               "feature_sha256": feature_sha, "features": features.receipt,
               "complete_unix_seconds": time.time()}
    _write_new(directory / "complete.json", _canonical(receipt))
    return receipt


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    registration = commands.add_parser("register")
    registration.add_argument("--contract", required=True)
    registration.add_argument("--contract-sha256", required=True)
    registration.add_argument("--directory", required=True)
    registration.add_argument("--reuse-annotation")
    acquisition = commands.add_parser("acquire")
    acquisition.add_argument("--plan", required=True)
    acquisition.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    if args.command == "register":
        result = register_reference(args.contract, contract_sha256=args.contract_sha256,
                                    directory=args.directory, reuse_annotation=args.reuse_annotation)
        print(json.dumps({"plan_sha256": hashlib.sha256(_canonical(result)).hexdigest(),
                          "output_genes": len(result["output_genes"])}))
    else:
        result = acquire_reference(args.plan, plan_sha256=args.plan_sha256)
        print(json.dumps({"total_response_bytes": result["total_response_bytes"],
                          "feature_sha256": result["feature_sha256"],
                          "present_count": result["features"]["present_count"]}))


if __name__ == "__main__":
    main()
