"""Synthetic artifact tests; no public expression, public archive, network or GPU."""
import json
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import materialize_genept_conditioning_v11 as m


@pytest.mark.parametrize("bad", [np.array([object()]),np.array([complex(1,2)]),np.array([float("nan")]),np.array([float("inf")])])
def test_writer_rejects_bad_arrays_before_writing(tmp_path,bad):
    path=tmp_path/"bad.npz"
    with pytest.raises(ValueError):
        m.write_npz(path,{"bad":bad})
    assert not path.exists()


def test_writer_replays_and_never_overwrites(tmp_path):
    path=tmp_path/"good.npz"
    arrays={"target_ids":np.array(["A","B"]),"embeddings":np.eye(2,dtype=np.float32),"present":np.ones(2,dtype=np.uint8)}
    receipt=m.write_npz(path,arrays)
    assert receipt["sha256"]==m.digest(path)
    with np.load(path,allow_pickle=False) as data:
        assert all(np.array_equal(data[k],v) for k,v in arrays.items())
    with pytest.raises(FileExistsError):
        m.write_npz(path,arrays)


def test_registration_is_metadata_only(tmp_path,monkeypatch):
    record={"schema":m.SCHEMA,"policy":m.policy(),"no_numeric_arrays":True}
    monkeypatch.setattr(m,"plan",lambda _:record)
    monkeypatch.setattr(m.numeric,"runtime_guard",lambda:None)
    def forbidden(*a,**k):
        raise AssertionError("Feature decode/stat fit before registration")
    monkeypatch.setattr(np,"load",forbidden)
    monkeypatch.setattr(m.conditioning,"fit_transform",forbidden)
    out=tmp_path/"registration"
    sha=m.register("synthetic-protocol",out)
    saved=m.numeric.bound_json(out/"contract.json",sha)
    assert {k:v for k,v in saved.items() if k!="registered_at"}==record
    with pytest.raises(ValueError,match="Fresh"):
        m.register("synthetic-protocol",out)


def test_complete_synthetic_feature_materialization(tmp_path,monkeypatch):
    # Same role sizes, feature dimensions and duplicate/missing structure, but
    # random synthetic embeddings. Only the authenticated metadata boundary is
    # stubbed; real quarantine, stats, shuffle, serialization and replay run.
    kfit=["LTB4R2","MED1","SKP2","UXT"]+[f"KF{i:03d}" for i in range(295)]
    jfit=["NOP9"]+[f"JF{i:03d}" for i in range(89)]
    khe=[f"KH{i:03d}" for i in range(75)]; jhe=[f"JH{i:03d}" for i in range(23)]
    official=["BRD1","BRPF1","MBD4"]+[f"O{i:03d}" for i in range(297)]
    ids=sorted(kfit+jfit+khe+jhe+official); index={t:i for i,t in enumerate(ids)}
    values=np.random.default_rng(711).standard_normal((787,1536)).astype(np.float32)
    present=np.ones(787,dtype=np.uint8); objects=np.arange(787,dtype=np.int32)
    for left,right in (("BRD1","BRPF1"),("LTB4R2","NOP9"),("MBD4","MED1"),("SKP2","UXT")):
        values[index[right]]=values[index[left]]; objects[index[right]]=objects[index[left]]
    for target in kfit[-3:]+jfit[-2:]:
        values[index[target]]=0; present[index[target]]=0; objects[index[target]]=-1
    base=tmp_path/"source"; (base/"numeric").mkdir(parents=True)
    arrays={"target_ids":np.asarray(ids),"embeddings":values,"present":present,
            "source_vector_object_id":objects,"source_vector_sha256":np.full(787,"synthetic",dtype="U64")}
    binding=m.write_npz(base/"numeric/features.npz",arrays)
    record={"schema":m.SCHEMA,"policy":m.policy(),"protocol":{"path":"synthetic"},
        "numeric_feature_bytes":binding["bytes"],"selected_targets":ids,"official_targets":official,
        "global_held_targets":sorted(khe+jhe),"sources":{
            "replogle_k562":{"fit_targets":sorted(kfit),"tune_targets":sorted(khe)},
            "nadig_jurkat":{"fit_targets":sorted(jfit),"tune_targets":sorted(jhe)}}}
    monkeypatch.setattr(m,"BASE",base); monkeypatch.setattr(m,"FEATURE_SHA",binding["sha256"])
    monkeypatch.setattr(m,"plan",lambda _:record)
    monkeypatch.setattr(m.numeric,"runtime_guard",lambda:None)
    monkeypatch.setattr(m.resource,"setrlimit",lambda *a:None)
    contract=tmp_path/"contract.json"; m.dump(contract,{**record,"registered_at":m.numeric.now()})
    out=tmp_path/"output"
    result=m.execute(contract,m.digest(contract),out)
    assert result["completed"] and result["all_numpy_artifacts_replayed"]
    assert result["coverage"]["official"]=={"total":300,"covered":297}
    assert result["sources"]["replogle_k562"]["available_fit_targets"]==292
    assert result["sources"]["nadig_jurkat"]["available_fit_targets"]==87
    assert result["responses_read"] is False and result["predictive_model_training_performed"] is False
    assert json.loads((out/"quarantine.json").read_text())["quarantined_target_names"]==m.QUARANTINE
    for source,roles in record["sources"].items():
        for seed in m.SEEDS:
            binding=result["sources"][source]["arms"][str(seed)]["report"]
            assert m.digest(binding["path"])==binding["sha256"]
            with np.load(out/source/f"arms_{seed}.npz",allow_pickle=False) as a:
                assert not a["masked"].any()
                assert np.array_equal(a["correct"][a["present"]==0],a["shuffled"][a["present"]==0])
                names=a["target_ids"].tolist(); donors=a["shuffled_donor_target_ids"].tolist()
                lookup={t:i for i,t in enumerate(names)}
                for target,donor in zip(names,donors):
                    assert (target in roles["fit_targets"])==(donor in roles["fit_targets"])
                    if a["present"][lookup[target]]: assert target!=donor
