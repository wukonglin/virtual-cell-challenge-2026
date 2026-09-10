"""Synthetic packaging tests. The official prep function is NEVER invoked here."""
from contextlib import contextmanager
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import signal
import stat
import tarfile
import tempfile
from types import SimpleNamespace

import pytest

import package_public_auxiliary_vcc_v10 as pack


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return pack.safe.digest(path)


def tar_fixture(path, nnz=360000, *, extra=False, schema=1):
    meta = {"schema":schema,"nnz":nnz,"n_obs":360000,"n_vars":18533,"cli_version":"0.2.0"}
    with tarfile.open(path, "w") as archive:
        for name, payload in [("meta.json",json.dumps(meta).encode()),("pred.h5ad.zst",b"synthetic-not-expression")]:
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member,io.BytesIO(payload))
        if extra:
            archive.addfile(tarfile.TarInfo("unexpected"))


def result_fixture(input_path, output_path, nnz=360000):
    return {"input":str(input_path),"output":str(output_path),"n_cells":360000,"n_genes":18533,
        "encoding":32,"nnz":nnz,"pert_col":"target_gene","output_pert_col":"target_gene",
        "celltype_col":None,"normalization":"counts-preserved","context_col":"context",
        "cells_per_context":{c:120000 for c in "ABC"},"verified_targets":True,"reordered_genes":False,
        "dry_run":False,"vcc_member":"pred.h5ad.zst",
        "dropped":["uns[method]","uns[generation_contract_sha256]","uns[writer_schema]"],"notes":[]}


@pytest.fixture
def scenario(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    base = root / "artifacts/public_flow/auxiliary_v10_20260910"
    scripts = root / "scripts"
    scripts.mkdir()
    names = ("generate_public_auxiliary_vcc_v10.py","public_auxiliary_count_adapter_v10.py",
        "public_auxiliary_vcc_writer_v10.py","run_public_auxiliary_flow_v10.py",
        "public_auxiliary_flow_v10.py","public_auxiliary_cache_v10.py")
    for name in names:
        (scripts/name).write_text("# Synthetic lineage stub " + name)
    for path in (Path(pack.__file__),Path(pack.safe.__file__)):
        (scripts/path.name).write_bytes(path.read_bytes())
    codes = {name:pack.safe.digest(scripts/name) for name in names}
    monkeypatch.setattr(pack,"ROOT",root)
    monkeypatch.setattr(pack,"BASE",base)
    monkeypatch.setattr(pack,"WRITER_CODE_SHA",codes["public_auxiliary_vcc_writer_v10.py"])
    train_path = base / "training_registration/training_contract.json"
    train_sha = write_json(train_path,{"code_sha256":{name:codes[name] for name in names[3:]}})
    trained_path = base / "pretraining/complete.json"
    trained_sha = write_json(trained_path,{"completed":True,"training_contract_sha256":train_sha,
        "optimizer_steps":3600,"tracking_events":3606})
    monkeypatch.setattr(pack,"TRAINING_SHA",train_sha)
    monkeypatch.setattr(pack,"TRAINING_COMPLETION_SHA",trained_sha)
    folder = root / "dataset/controls"
    folder.mkdir(parents=True)
    genes = [f"GENE{i:05d}" for i in range(18533)]
    targets = genes[:300]
    (folder/"gene_names.csv").write_text("gene_name\n"+"\n".join(genes)+"\n")
    (folder/"pert_counts.csv").write_text("target_gene\n"+"\n".join(targets)+"\n")
    write_json(folder/"manifest.json",{"season":"2026","partition":"val","panel_id":"vcc2026-val-1",
        "contexts":["A","B","C"],"pert_col":"target_gene","context_col":"context",
        "control_label":"non-targeting","n_genes":18533,"n_constructs":300,"cells_per_pert":400})
    shas = {p.name:pack.safe.digest(p) for p in folder.iterdir()}
    monkeypatch.setattr(pack,"METADATA_SHAS",shas)
    policy = {k:False for k in ("challenge_treated_used","genept_used","promoterai_used","posttraining_performed",
        "scientific_improvement_established","upload_performed_by_generator","clipping_or_thinning")}
    policy.update(stage="inference",arm="true",heun_steps=16,strength=1.,
        ensemble={"nadig_jurkat":.5,"replogle_k562":.5},exploratory_submission_authorized_by_user=True)
    inference_path = base / "inference_registration/inference_contract.json"
    inference = {"schema":pack.INFERENCE_SCHEMA,"policy":policy,"code_sha256":codes,
        "training_contract":{"path":str(train_path),"sha256":train_sha},
        "training_completion":{"path":str(trained_path),"sha256":trained_sha},
        "genes":genes,"targets":targets,"modeled_genes":genes[:482],"normalization_genes":genes[:7097],
        "manifest_sha256":shas["manifest.json"],"targets_sha256":shas["pert_counts.csv"]}
    inference_sha = write_json(inference_path,inference)
    write_json(inference_path.with_name("complete.json"),{"schema":pack.INFERENCE_SCHEMA,
        "registered":True,"inference_contract_sha256":inference_sha})
    generation_path = base / "generation/complete.json"
    counts = generation_path.parent / "counts"
    counts.mkdir(parents=True)
    input_path = counts / "prediction.h5ad"
    input_path.write_bytes(b"synthetic input; never opened as AnnData")
    journal = counts / "groups.jsonl"
    journal.write_text("synthetic journal")
    writer = {"schema":pack.WRITER_SCHEMA,"generation_contract_sha256":inference_sha,"genes":genes,"targets":targets,
        "config":{"expected_genes":18533,"expected_targets":300,"contexts":["A","B","C"],
        "cells_per_group":400,"max_nnz":4750000000,"max_counts_per_cell":1000000}}
    writer_sha = write_json(counts / "writer.json",writer)
    report = {"shape":[360000,18533],"groups":900,"cells_per_group":400,"nnz":360000,
        "full_axis_exact":True,"all_quotas_exact":True,"integer_nonnegative_counts":True,"canonical_csr":True,
        "unmodeled_genes_modified_by_writer":False,"sparsity_cap_applied":False,
        "minimum_library":1,"maximum_library":9000}
    receipt = {"schema":pack.WRITER_SCHEMA,"completed":True,"generation_contract_sha256":inference_sha,
        "output_path":str(input_path),"output_sha256":pack.safe.digest(input_path),"output_bytes":input_path.stat().st_size,
        "writer_sha256":writer_sha,"group_journal_sha256":pack.safe.digest(journal),"validation":report,
        "submission_performed":False,"official_cli_validated":False}
    write_json(counts / "complete.json",receipt)
    qc_path = generation_path.parent / "count_adapter_qc.json"
    qc_sha = write_json(qc_path,{"groups":[]})
    generation = {"schema":pack.INFERENCE_SCHEMA,"completed":True,"inference_contract_sha256":inference_sha,
        "output":receipt,"count_adapter_qc":{"path":str(qc_path),"sha256":qc_sha},
        "submission_performed":False,"official_cli_validation_performed":False,
        "posttraining_performed":False,"scientific_improvement_established":False}
    generation_sha = write_json(generation_path,generation)
    calls = []
    def fake_prep(**kwargs):
        calls.append(kwargs)
        assert tempfile.tempdir == str(base / "package/temporary")
        tar_fixture(kwargs["output_path"])
        return SimpleNamespace(to_dict=lambda:result_fixture(kwargs["input_path"],kwargs["output_path"]))
    container_calls = []
    def fake_container(path):
        container_calls.append(path)
        assert path.is_file()
    monkeypatch.setattr(pack,"official_loader",lambda:(fake_prep,{},fake_container))
    monkeypatch.setattr(pack,"runtime_preflight",lambda _:None)
    @contextmanager
    def guard(*_):
        yield
    monkeypatch.setattr(pack,"resource_guard",guard)
    return {"root":root,"base":base,"generation_path":generation_path,"generation_sha":generation_sha,
        "inference_path":inference_path,"inference_sha":inference_sha,"input":input_path,
        "generation":generation,"inference":inference,"receipt":receipt,"calls":calls,
        "container_calls":container_calls,"out":base/"package"}


def args(s):
    return (s["generation_path"],s["generation_sha"],s["inference_path"],s["inference_sha"])


def test_full_authenticated_pipeline_uses_one_fake_prep_and_preserves_input(scenario):
    s = scenario
    before = s["input"].read_bytes()
    previous_temp = tempfile.tempdir
    receipt = pack.package(*args(s),s["out"])
    assert len(s["calls"]) == len(s["container_calls"]) == 1
    assert receipt == pack.safe.read_json(s["out"]/"complete.json")
    assert receipt["completed"] and receipt["official_cli_validation_performed"]
    assert receipt["official_container_validated"] and not receipt["submission_performed"]
    assert receipt["package"]["sha256"] == pack.safe.digest(s["out"]/"prediction.vcc")
    assert receipt["generation_completion"]["sha256"] == s["generation_sha"]
    assert s["input"].read_bytes() == before and tempfile.tempdir == previous_temp
    for name in ("prediction.vcc","complete.json","package_intent.json"):
        assert stat.S_IMODE((s["out"]/name).stat().st_mode) == 0o600
    assert stat.S_IMODE(s["out"].stat().st_mode) == 0o700
    assert stat.S_IMODE((s["out"]/"temporary").stat().st_mode) == 0o700


def test_prep_all_default_safety_enabled(scenario):
    s = scenario
    pack.package(*args(s),s["out"])
    kw = s["calls"][0]
    for key in ("verify_targets","check_cell_counts","require_counts","reject_controls"):
        assert kw[key] is True
    for key in ("allow_discrete","dry_run","force"):
        assert kw[key] is False
    assert kw["cells_per_pert"] == 400 and kw["required_contexts"] == ("A","B","C")
    assert kw["max_nnz"] == 4750000000 and kw["max_counts_per_cell"] == 1000000
    assert kw["encoding"] == 32 and kw["expected_gene_dim"] == 18533


@pytest.mark.parametrize("part",["input","code","gene_names.csv","pert_counts.csv","manifest.json","qc","journal","writer"])
def test_tamper_rejected_before_prep(scenario,part):
    s = scenario
    paths = {"input":s["input"],"code":s["root"]/"scripts/public_auxiliary_flow_v10.py",
        "qc":s["generation_path"].parent/"count_adapter_qc.json","journal":s["input"].parent/"groups.jsonl",
        "writer":s["input"].parent/"writer.json"}
    path = paths.get(part,s["root"]/"dataset/controls"/part)
    path.write_bytes(path.read_bytes()+b" ")
    with pytest.raises(ValueError):
        pack.package(*args(s),s["out"])
    assert not s["calls"] and not s["out"].exists()


@pytest.mark.parametrize("key,value",[("completed",False),("submission_performed",True),
    ("inference_contract_sha256","a"*64),("scientific_improvement_established",True),
    ("official_cli_validation_performed",True)])
def test_bad_generation_state_rejected(scenario,key,value):
    s = scenario
    s["generation"][key] = value
    s["generation_sha"] = write_json(s["generation_path"],s["generation"])
    with pytest.raises(ValueError):
        pack.package(*args(s),s["out"])
    assert not s["calls"]


@pytest.mark.parametrize("key,value",[("shape",[359999,18533]),("groups",899),("cells_per_group",399),
    ("full_axis_exact",False),("sparsity_cap_applied",True),("integer_nonnegative_counts",1),
    ("nnz",4750000001),("minimum_library",0),("maximum_library",1000001)])
def test_bad_writer_validation_rejected(scenario,key,value):
    s = scenario
    s["receipt"]["validation"][key] = value
    write_json(s["input"].parent/"complete.json",s["receipt"])
    s["generation_sha"] = write_json(s["generation_path"],s["generation"])
    with pytest.raises(ValueError):
        pack.package(*args(s),s["out"])
    assert not s["calls"]


@pytest.mark.parametrize("key,value",[("n_cells",1),("n_genes",482),("normalization","normalized-log1p"),
    ("reordered_genes",True),("verified_targets",False),("dry_run",True),("nnz",1),
    ("cells_per_context",{"A":120000}),("dropped",["layers"]),("encoding",64)])
def test_bad_official_result_rejected(key,value):
    result = result_fixture("input","output")
    result[key] = value
    with pytest.raises(ValueError):
        pack.validate_prep_result(result,"input","output",360000)


@pytest.mark.parametrize("drops",[[],["uns[method]"],
    ["uns[method]","uns[generation_contract_sha256]","uns[writer_schema]"]])
def test_only_surviving_provenance_uns_keys_may_be_dropped(drops):
    # anndata0.11 concat_on_disk currently omits uns entirely; allow that empty subset.
    result = result_fixture("input","output")
    result["dropped"] = drops
    pack.validate_prep_result(result,"input","output",360000)


@pytest.mark.parametrize("drops",[["uns"],["uns[unknown]"],["uns[method]","uns[method]"]])
def test_generic_unknown_duplicate_drop_reports_rejected(drops):
    result = result_fixture("input","output")
    result["dropped"] = drops
    with pytest.raises(ValueError,match="dropped"):
        pack.validate_prep_result(result,"input","output",360000)


def test_existing_output_never_overwritten(scenario):
    s = scenario
    s["out"].mkdir()
    (s["out"]/"mine").write_text("keep")
    with pytest.raises(ValueError,match="Fresh package"):
        pack.package(*args(s),s["out"])
    assert (s["out"]/"mine").read_text() == "keep" and not s["calls"]


def test_symlink_count_input_rejected(scenario):
    s = scenario
    moved = s["input"].with_name("moved.h5ad")
    s["input"].rename(moved)
    s["input"].symlink_to(moved)
    with pytest.raises(ValueError,match="non-symlink"):
        pack.package(*args(s),s["out"])
    assert not s["calls"]


def test_fake_prep_failure_restores_temp_and_no_completion(scenario,monkeypatch):
    s = scenario
    previous = tempfile.tempdir
    def fail(**kw):
        raise RuntimeError("synthetic prep failure")
    monkeypatch.setattr(pack,"official_loader",lambda:(fail,{},lambda _:None))
    with pytest.raises(RuntimeError,match="synthetic"):
        pack.package(*args(s),s["out"])
    assert not (s["out"]/"complete.json").exists()
    assert (s["out"]/"package_intent.json").exists() and tempfile.tempdir == previous


def test_input_change_during_prep_prevents_completion(scenario,monkeypatch):
    s = scenario
    original, _, checker = pack.official_loader()
    def change(**kw):
        result = original(**kw)
        s["input"].write_bytes(b"changed")
        return result
    monkeypatch.setattr(pack,"official_loader",lambda:(change,{},checker))
    with pytest.raises(ValueError,match="hash or open identity"):
        pack.package(*args(s),s["out"])
    assert not (s["out"]/"complete.json").exists()


@pytest.mark.parametrize("extra,schema",[(True,1),(False,2)])
def test_wrong_tar_container_rejected(tmp_path,extra,schema):
    path = tmp_path/"fixture.vcc"
    tar_fixture(path,extra=extra,schema=schema)
    with pytest.raises(ValueError):
        pack.validate_archive(path,360000)


def test_authenticate_detects_mid_hash_identity_change(tmp_path,monkeypatch):
    path = tmp_path/"file"
    path.write_bytes(b"fixture")
    sha = pack.safe.digest(path)
    original = pack.signature
    calls = []
    def fake(stream):
        calls.append(1)
        value = original(stream)
        return value if len(calls) == 1 else (*value[:-1],value[-1]+1)
    monkeypatch.setattr(pack,"signature",fake)
    with pytest.raises(ValueError,match="open identity"):
        pack.authenticate(path,sha)


@pytest.mark.parametrize("host",["aida","login.anvil.rcac.purdue.edu","cbsulogin","localhost"])
def test_head_nodes_rejected(tmp_path,monkeypatch,host):
    monkeypatch.setattr(pack.socket,"gethostname",lambda:host)
    with pytest.raises(ValueError,match="never a head/login"):
        pack.runtime_preflight(tmp_path)


def test_cpu_disk_preflight(tmp_path,monkeypatch):
    monkeypatch.setattr(pack.socket,"gethostname",lambda:"cbsuvlaminck3.biohpc.cornell.edu")
    monkeypatch.setattr(pack.os,"sched_getaffinity",lambda _:{0,1})
    monkeypatch.setattr(pack.shutil,"disk_usage",lambda _:SimpleNamespace(free=pack.MIN_FREE_DISK))
    for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        monkeypatch.setenv(name,"2")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","")
    pack.runtime_preflight(tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","0")
    with pytest.raises(ValueError,match="hidden GPUs"):
        pack.runtime_preflight(tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","")
    monkeypatch.setattr(pack.os,"sched_getaffinity",lambda _:set(range(9)))
    with pytest.raises(ValueError,match="at most8"):
        pack.runtime_preflight(tmp_path)


def test_guard_sets_and_restores_limits_without_real_process_mutation(tmp_path,monkeypatch):
    limits = {resource.RLIMIT_AS:(resource.RLIM_INFINITY,resource.RLIM_INFINITY),
        resource.RLIMIT_FSIZE:(64 << 30,resource.RLIM_INFINITY)}
    originals = copy.deepcopy(limits)
    monkeypatch.setattr(pack.resource,"getrlimit",lambda kind:limits[kind])
    monkeypatch.setattr(pack.resource,"setrlimit",lambda kind,pair:limits.__setitem__(kind,pair))
    monkeypatch.setattr(pack.signal,"getitimer",lambda _:(0.,0.))
    monkeypatch.setattr(pack.signal,"getsignal",lambda _:signal.SIG_DFL)
    monkeypatch.setattr(pack.signal,"signal",lambda *_:None)
    monkeypatch.setattr(pack.signal,"setitimer",lambda *_:None)
    monkeypatch.setattr(pack,"resources",lambda *_:{})
    with pack.resource_guard(tmp_path,0):
        assert limits[resource.RLIMIT_AS][0] == 96 << 30
        assert limits[resource.RLIMIT_FSIZE][0] == 64 << 30
    assert limits == originals


def test_json_writer_never_overwrites(tmp_path):
    path = tmp_path/"receipt.json"
    pack.dump_new(path,{"x":1})
    with pytest.raises(FileExistsError):
        pack.dump_new(path,{"x":2})
    assert pack.safe.read_json(path) == {"x":1}


def test_wrong_interpreter_rejected_before_official_import(monkeypatch):
    monkeypatch.setattr(pack.sys,"prefix","/synthetic/wrong-interpreter")
    with pytest.raises(ValueError,match="pinned .venv-vcc"):
        pack.official_loader()
