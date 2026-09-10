"""Independent promoter features: synthetic DNA/annotation fixtures only."""
from dataclasses import replace
import hashlib
import gzip
import io
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import promoter_output_features as p


def tx(**kwargs):
    args = dict(gene_symbol="A", gene_id="ENSG00000000001.2",
                transcript_id="ENST00000000001.3", chrom="chr1",
                start1=2001, end1=3000, strand="+")
    return p.Transcript(**(args | kwargs))


def gtf(**kwargs):
    t = tx(**kwargs)
    tags = 'tag "basic"; tag "MANE_Select";' if t.mane_select else 'tag "basic";'
    return (f'{t.chrom}\tTEST\ttranscript\t{t.start1}\t{t.end1}\t.\t{t.strand}\t.\t'
            f'gene_id "{t.gene_id}"; transcript_id "{t.transcript_id}"; '
            f'gene_name "{t.gene_symbol}"; {tags}\n')


def build(**kwargs):
    options = dict(gene_symbols=("A", "MISSING"), transcripts=(tx(),),
                   chrom_sizes={"chr1": 5000}, fetch_plus=lambda c, s, e: "ACGT" * ((e-s)//4),
                   assembly="GRCh38", annotation_sha256="a"*64, reference_sha256="b"*64,
                   contract_sha256="c"*64, upstream=8, downstream=4)
    return p.build_features(**(options | kwargs))


@pytest.mark.parametrize("strand,tss,start,end", [("+", 2000, 1000, 2100), ("-", 2999, 2900, 4000)])
def test_gtf_tss_and_strand_window(strand, tss, start, end):
    transcript = tx(strand=strand)
    assert transcript.tss0 == tss
    assert p.promoter_interval(transcript, chrom_size=5000) == (start, end)
    assert end-start == 1100


@pytest.mark.parametrize("kwargs", [{"gene_id": "ENSG00000000001"}, {"transcript_id": "ENST00000000001"},
                                    {"gene_id": "ENSG00000000001.0"}, {"strand": "."},
                                    {"start1": 0}, {"end1": 1000}, {"start1": True},
                                    {"mane_select": 1}])
def test_invalid_identifiers_or_coordinates_fail(kwargs):
    with pytest.raises(ValueError):
        tx(**kwargs)


def test_gtf_exact_only_mane_and_versions_retained():
    result = p.parse_gtf(io.StringIO("# header\n" + gtf(mane_select=True) + gtf(gene_symbol="OLD_ALIAS")),
                         requested_symbols=["A"])
    assert result == (tx(mane_select=True),)
    assert result[0].gene_id.endswith(".2")
    assert result[0].transcript_id.endswith(".3")


@pytest.mark.parametrize("text", [gtf()+gtf(), gtf().replace('gene_name "A";', 'gene_name "A"; gene_name "B";'),
                                 gtf().replace('gene_name "A";', 'gene_name A;'), "broken\n"])
def test_malformed_gtf_fails(text):
    with pytest.raises(ValueError):
        p.parse_gtf(io.StringIO(text), requested_symbols=["A"])


def test_gtf_nontranscript_rows_do_not_define_tss():
    assert not p.parse_gtf(io.StringIO(gtf().replace("\ttranscript\t", "\texon\t")), requested_symbols=["A"])


def test_official_gtf_unquoted_numeric_level():
    row = gtf().replace('tag "basic";', 'level 2; tag "basic";')
    assert p.parse_gtf(io.StringIO(row), requested_symbols=["A"]) == (tx(),)
    for bad in [row.replace("level 2", "level 9"), row.replace("level 2", "gene_name 2")]:
        with pytest.raises(ValueError):
            p.parse_gtf(io.StringIO(bad), requested_symbols=["A"])


def test_official_gtf_multivalued_ontology_and_ccds():
    row = gtf().rstrip() + ' ont "PGO:0000001"; ont "PGO:0000019"; ccdsid "CCDS1"; ccdsid "CCDS2";\n'
    assert p.parse_gtf(io.StringIO(row), requested_symbols=["A"]) == (tx(),)


def test_mane_fixed_choice_and_consensus_fallback():
    other = tx(transcript_id="ENST00000000002.1", start1=2100)
    assert p.select_tss("A", [tx(), other]) == (None, "ambiguous_tss")
    assert p.select_tss("A", [tx(mane_select=True), other])[0] == tx(mane_select=True)
    consensus = replace(other, start1=2001)
    assert p.select_tss("A", [tx(), consensus]) == (tx(), "consensus_tss")
    assert p.select_tss("A", [tx(mane_select=True), replace(other, mane_select=True)]) == (None, "ambiguous_mane_tss")


@pytest.mark.parametrize("other,reason", [
    (tx(gene_id="ENSG00000000002.1"), "ambiguous_gene_symbol"),
    (tx(gene_id="ENSG00000000001.3"), "ambiguous_gene_version"),
    (tx(chrom="chr2"), "ambiguous_locus"), (tx(strand="-"), "ambiguous_locus")])
def test_ambiguous_symbols_versions_and_loci_are_not_resolved(other, reason):
    assert p.select_tss("A", [tx(), other]) == (None, reason)


def test_missing_and_nonprimary_not_guessed():
    assert p.select_tss("a", [tx()]) == (None, "missing_exact_symbol")
    assert p.select_tss("A", [tx(chrom="chr1_alt")]) == (None, "nonprimary_contig")


@pytest.mark.parametrize("transcript,size", [(tx(start1=1), 5000), (tx(strand="-"), 3050), (tx(), 2000)])
def test_boundary_never_clamped_or_padded(transcript, size):
    with pytest.raises(ValueError):
        p.promoter_interval(transcript, chrom_size=size)


def test_reverse_complement_and_softmask():
    seq = "AaCGTNryswkmbdhv"
    oriented = p.orient_sequence(seq, "-", expected_length=len(seq))
    assert p.orient_sequence(oriented, "-", expected_length=len(seq)) == seq.upper()
    assert p.orient_sequence("AAGC", "-", expected_length=4) == "GCTT"


@pytest.mark.parametrize("seq,strand,length", [("ACGT", "+", 3), ("AC-T", "+", 4), ("ACGT", "?", 4), ("AC U", "+", 4)])
def test_bad_sequence_fails(seq, strand, length):
    with pytest.raises(ValueError):
        p.orient_sequence(seq, strand, expected_length=length)


def test_composition_math_and_denominators():
    values, mask = p.sequence_features("ACGT")
    np.testing.assert_allclose(values, [0.5, 1/3, 16/3, 0])
    assert mask.all()
    values, mask = p.sequence_features("CNG")
    np.testing.assert_allclose(values, [1, 0, 0, 1/3])
    assert mask.tolist() == [True, False, False, True]
    values, mask = p.sequence_features("AAAA")
    assert mask.tolist() == [True, True, False, True]
    np.testing.assert_allclose(values, 0)
    values, mask = p.sequence_features("NNN")
    assert mask.tolist() == [False, False, False, True]
    np.testing.assert_allclose(values, [0, 0, 0, 1])


def test_exact_axis_and_explicit_masks():
    f = build(gene_symbols=("MISSING", "A"))
    assert f.gene_symbols == ("MISSING", "A")
    assert f.present.tolist() == [False, True]
    assert not f.valid[0].any()
    assert not f.values[0].any()
    assert f.receipt["policy"]["expression_accessed"] is False
    assert f.receipt["policy"]["fit_performed"] is False
    assert f.receipt["gene_axis_sha256"] == p.axis_sha256(["MISSING", "A"])
    assert p.axis_sha256(["A", "MISSING"]) != p.axis_sha256(["MISSING", "A"])


def test_missing_maps_never_call_sequence_fetch():
    def forbidden(*args):
        raise AssertionError("fetch must not run")
    f = build(gene_symbols=("MISSING",), fetch_plus=forbidden)
    assert not f.present.any()
    f = build(chrom_sizes={}, fetch_plus=forbidden)
    assert f.receipt["genes"][0]["missing_reason"] == "missing_reference_chromosome"


def test_build_coordinates_and_orientation_receipt():
    calls = []
    def fetch(chrom, start, end):
        calls.append((chrom, start, end))
        return "AAGC" * 3
    f = build(transcripts=(tx(strand="-"),), fetch_plus=fetch)
    assert calls == [("chr1", 2996, 3008)]
    assert f.receipt["genes"][0]["oriented_sequence_sha256"] == hashlib.sha256(("GCTT"*3).encode()).hexdigest()


def test_excess_ambiguous_bases_explicit_missing():
    f = build(fetch_plus=lambda *args: "N"*12)
    assert not f.present.any()
    assert not f.valid.any()
    assert not f.values.any()
    assert f.receipt["genes"][0]["missing_reason"] == "excess_ambiguous_bases"


@pytest.mark.parametrize("kwargs", [{"assembly": "hg38"}, {"annotation_sha256": "wrong"},
                                    {"gene_symbols": ("A", "A")}, {"gene_symbols": ()},
                                    {"gene_symbols": (" A",)}, {"upstream": 0},
                                    {"max_ambiguous_fraction": 1}, {"max_ambiguous_fraction": float("nan")}])
def test_invalid_build_fails(kwargs):
    with pytest.raises(ValueError):
        build(**kwargs)


def test_artifact_is_pickle_free_and_no_clobber(tmp_path):
    f = build()
    path = tmp_path / "features.npz"
    digest = p.save_npz(path, f)
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    with np.load(path, allow_pickle=False) as data:
        assert data["gene_symbols"].tolist() == ["A", "MISSING"]
        np.testing.assert_array_equal(data["present"], f.present)
        assert json.loads(str(data["receipt_json"])) == f.receipt
        for key in data.files:
            assert data[key].dtype.kind != "O"
    with pytest.raises(FileExistsError):
        p.save_npz(path, f)


def test_par_y_suffix_retained_never_collapsed():
    t = tx(gene_id="ENSG00000000001.2_PAR_Y", transcript_id="ENST00000000001.3_PAR_Y", chrom="chrY")
    assert t.gene_id.endswith("_PAR_Y")
    assert p.select_tss("A", [tx(), t])[0] is None


def test_ensembl_coordinate_mapping_and_batch_limits():
    assert p.ensembl_region("chr1", 0, 1100) == "1:1..1100:1"
    assert p.ensembl_region("chrM", 10, 1110) == "MT:11..1110:1"
    batches = p.sequence_batches([("chr1", i, i+1100) for i in range(512)])
    assert list(map(len, batches)) == [50]*10 + [12]
    assert p.sequence_batches([("chr1", 0, 10)]*2) == [["1:1..10:1"]]
    with pytest.raises(ValueError):
        p.sequence_batches([("chr1", i, i+1100) for i in range(513)])


@pytest.mark.parametrize("interval", [("chr1_alt", 0, 10), ("chr1", -1, 10),
                                      ("chr1", 0, 1101), ("chr1", 2, 2), ("chr1", True, 4)])
def test_bad_request_bounds(interval):
    with pytest.raises(ValueError):
        p.ensembl_region(*interval)


def response(query="1:1..4:1", **kwargs):
    chrom, region, strand = query.split(":")
    start, end = region.split("..")
    return {"query": query, "id": f"chromosome:GRCh38:{chrom}:{start}:{end}:{strand}",
            "molecule": "dna", "seq": "ACGT"} | kwargs


def test_sequence_response_joins_query_not_position():
    one, two = response(), response("X:2..5:1", seq="GGGG")
    assert p.validate_ensembl_response([two, one], [one["query"], two["query"]]) == {
        "1:1..4:1": "ACGT", "X:2..5:1": "GGGG"}


@pytest.mark.parametrize("payload", [[], {}, [response(), response()],
    [response(query="2:1..4:1")], [response(id="chromosome:GRCh37:1:1:4:1")],
    [response(id="chromosome:GRCh38:1:1:4:-1")], [response(molecule="protein")],
    [response(seq="ACG")], [response(seq="AC G")], [response(seq=None)]])
def test_sequence_response_rejects_untrusted_mapping(payload):
    with pytest.raises(ValueError):
        p.validate_ensembl_response(payload, ["1:1..4:1"])


def assembly():
    return {"assembly_name": "GRCh38.p14", "top_level_region": [
        {"name": "MT" if c == "chrM" else c[3:], "coord_system": "chromosome", "length": 1_000_000}
        for c in sorted(p.PRIMARY_CHROMS)]}


def test_assembly_pinned_and_complete():
    assert set(p.validate_assembly_response(assembly())) == p.PRIMARY_CHROMS
    with pytest.raises(ValueError):
        p.validate_assembly_response(assembly() | {"assembly_name": "GRCh37"})
    with pytest.raises(ValueError):
        p.validate_assembly_response(assembly() | {"top_level_region": []})
    duplicate = assembly()
    duplicate["top_level_region"].append(duplicate["top_level_region"][0])
    with pytest.raises(ValueError):
        p.validate_assembly_response(duplicate)


def register(tmp_path, genes=("A", "MISSING")):
    path = tmp_path / "contract.json"
    raw = p._canonical({"schema": "public-flow-row-contract-v1", "output_genes": list(genes)})
    path.write_bytes(raw)
    directory = tmp_path / "promoter"
    plan = p.register_reference(path, contract_sha256=hashlib.sha256(raw).hexdigest(), directory=directory)
    return directory, hashlib.sha256(p._canonical(plan)).hexdigest()


def test_registration_fresh_directory_and_contract_binding(tmp_path):
    directory, digest = register(tmp_path)
    plan = json.loads((directory / "plan.json").read_bytes())
    assert digest == hashlib.sha256((directory / "plan.json").read_bytes()).hexdigest()
    assert plan["caps"]["api_requests"] == 12
    assert plan["caps"]["automatic_retries"] == 0
    assert plan["policy"]["licensed_model_assets"] is False
    with pytest.raises(FileExistsError):
        p.register_reference(tmp_path / "contract.json", contract_sha256=plan["contract_sha256"], directory=directory)
    with pytest.raises(ValueError, match="contract hash"):
        p.register_reference(tmp_path / "contract.json", contract_sha256="d"*64, directory=tmp_path / "bad")


def fake_http(monkeypatch, *, bad_sequence=False, oversize=False, rate_limited=False):
    import urllib.request
    annotation = gzip.compress(gtf(mane_select=True).encode(), mtime=0)
    monkeypatch.setattr(p, "ANNOTATION_BYTES", len(annotation))
    monkeypatch.setattr(p, "ANNOTATION_MD5", hashlib.md5(annotation).hexdigest())
    monkeypatch.setattr("time.sleep", lambda n: None)
    calls = []
    class Response:
        status = 200
        def __init__(self, url, body):
            self.url, self.body = url, body
            self.headers = {"Content-Length": str(len(body) + 1_000_000 if oversize else len(body))}
            if rate_limited:
                self.headers["Retry-After"] = "5"
        def geturl(self):
            return self.url
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def read(self, cap):
            return self.body[:cap]
    def open_url(request, timeout):
        calls.append(request.full_url)
        if request.full_url == p.ANNOTATION_URL:
            return Response(request.full_url, annotation)
        if request.full_url == p.ASSEMBLY_URL:
            return Response(request.full_url, p._canonical(assembly()))
        assert request.full_url == p.SEQUENCE_URL
        regions = json.loads(request.data)["regions"]
        payload = []
        for query in regions:
            item = response(query, seq="A"*1100)
            if bad_sequence:
                item["id"] = item["id"].replace("GRCh38", "GRCh37")
            payload.append(item)
        return Response(request.full_url, p._canonical(payload))
    monkeypatch.setattr(urllib.request, "urlopen", open_url)
    return calls


def test_complete_acquisition_synthetic_transport_only(tmp_path, monkeypatch):
    calls = fake_http(monkeypatch)
    directory, digest = register(tmp_path)
    result = p.acquire_reference(directory / "plan.json", plan_sha256=digest)
    assert result["features"]["present_count"] == 1
    assert result["api_requests"] == 2
    assert result["api_bytes"] < 1_000_000
    assert (directory / "sequence_plan.json").exists()
    assert (directory / "complete.json").exists()
    assert calls == [p.ANNOTATION_URL, p.ASSEMBLY_URL, p.SEQUENCE_URL]
    assert result["feature_sha256"] == hashlib.sha256((directory / "features.npz").read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        p.acquire_reference(directory / "plan.json", plan_sha256=digest)
    assert len(calls) == 3  # no second requests, no retry campaign


@pytest.mark.parametrize("option", ["bad_sequence", "oversize", "rate_limited"])
def test_acquisition_stops_no_partial_success_or_retries(tmp_path, monkeypatch, option):
    calls = fake_http(monkeypatch, **{option: True})
    directory, digest = register(tmp_path)
    with pytest.raises(ValueError):
        p.acquire_reference(directory / "plan.json", plan_sha256=digest)
    assert not (directory / "complete.json").exists()
    assert not (directory / "features.npz").exists()
    assert len(calls) <= 3
    with pytest.raises(FileExistsError):
        p.acquire_reference(directory / "plan.json", plan_sha256=digest)


def test_tampered_acquisition_plan_does_not_touch_network(tmp_path, monkeypatch):
    calls = fake_http(monkeypatch)
    directory, digest = register(tmp_path)
    path = directory / "plan.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="plan hash"):
        p.acquire_reference(path, plan_sha256=digest)
    assert not calls


def test_symlinked_plan_forbidden(tmp_path, monkeypatch):
    calls = fake_http(monkeypatch)
    directory, digest = register(tmp_path)
    alias = tmp_path / "plan_alias.json"
    alias.symlink_to(directory / "plan.json")
    with pytest.raises(OSError):
        p.acquire_reference(alias, plan_sha256=digest)
    assert not calls


def test_recovery_reuses_verified_annotation_without_redownload(tmp_path, monkeypatch):
    calls = fake_http(monkeypatch)
    directory, digest = register(tmp_path)
    p.acquire_reference(directory / "plan.json", plan_sha256=digest)
    oldplan = json.loads((directory / "plan.json").read_bytes())
    recovery = tmp_path / "recovery"
    plan = p.register_reference(tmp_path / "contract.json", contract_sha256=oldplan["contract_sha256"],
        directory=recovery, reuse_annotation=directory / "gencode.v48.basic.annotation.gtf.gz")
    calls.clear()
    result = p.acquire_reference(recovery / "plan.json", plan_sha256=hashlib.sha256(p._canonical(plan)).hexdigest())
    assert calls == [p.ASSEMBLY_URL, p.SEQUENCE_URL]
    assert result["total_response_bytes"] == result["api_bytes"]
    assert result["files"][0]["reuse_verified_local"] is True


def test_changed_recovery_annotation_rejected_before_download(tmp_path, monkeypatch):
    calls = fake_http(monkeypatch)
    directory, digest = register(tmp_path)
    p.acquire_reference(directory / "plan.json", plan_sha256=digest)
    oldplan = json.loads((directory / "plan.json").read_bytes())
    annotation = directory / "gencode.v48.basic.annotation.gtf.gz"
    recovery = tmp_path / "recovery"
    plan = p.register_reference(tmp_path / "contract.json", contract_sha256=oldplan["contract_sha256"],
        directory=recovery, reuse_annotation=annotation)
    annotation.write_bytes(b"tampered")
    calls.clear()
    with pytest.raises(ValueError, match="Cached annotation hash"):
        p.acquire_reference(recovery / "plan.json", plan_sha256=hashlib.sha256(p._canonical(plan)).hexdigest())
    assert not calls
