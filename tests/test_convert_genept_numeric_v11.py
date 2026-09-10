"""Synthetic fixtures only; no real archive decoding, pickle loading, or network."""
from array import array
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle  # dumps creates small controlled fixtures; loads is never used
import struct

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("genept_converter", Path(__file__).parents[1]/"scripts/convert_genept_numeric_v11.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fixture():
    shared = [0.1, -0.2, 0.3]
    return {"A": shared, "Alias_A": shared, "B": [0.2, 0.3, 0.4]}


def payload():
    return pickle.dumps(fixture(), protocol=4)


def parse(value, **kwargs):
    return m.parse_vectors(value, dimension=3, **kwargs)


def test_shared_numeric_vectors_match_original():
    result, audit = parse(payload())
    assert audit["name_keys"] == 3
    assert audit["stored_vector_objects"] == 2
    assert result["A"] is result["Alias_A"]
    assert all(type(v) is array for v in result.values())
    for key, value in fixture().items():
        assert list(result[key]) == value


def test_long_memo_reference():
    original = {f"G{i:04d}": [float(i), 1., 2.] for i in range(300)}
    original["Z_ALIAS"] = original["G0299"]
    result, audit = parse(pickle.dumps(original, protocol=4))
    assert result["G0299"] is result["Z_ALIAS"]
    assert audit["opcode_counts"]["LONG_BINGET"] > 0


@pytest.mark.parametrize("cut", range(len(payload())))
def test_every_truncation_rejected(cut):
    with pytest.raises(ValueError):
        parse(payload()[:cut])


SUPPORTED = {0x95,0x7d,0x94,0x28,0x8c,0x5d,0x47,0x68,0x6a,0x65,0x75,0x2e}
@pytest.mark.parametrize("opcode", sorted(set(range(256))-SUPPORTED))
def test_every_other_opcode_rejected_before_argument(opcode):
    with pytest.raises(ValueError, match="Forbidden opcode"):
        parse(b"\x80\x04"+bytes([opcode]))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_rejected(value):
    with pytest.raises(ValueError, match="Nonfinite"):
        parse(pickle.dumps({"A": [value, 1., 2.], "B": [1., 2., 3.]}, protocol=4))


@pytest.mark.parametrize("dimension", [1,2,4,4097,0,-1,True,2.5])
def test_dimension_mismatch(dimension):
    with pytest.raises(ValueError):
        m.parse_vectors(payload(), dimension=dimension)


@pytest.mark.parametrize("limits", [{"max_names":2},{"max_vectors":1},{"max_memo":2},{"max_bytes":3}])
def test_resource_caps(limits):
    with pytest.raises(ValueError):
        parse(payload(), **limits)


@pytest.mark.parametrize("value", [b"\x80\x04h\x00.", b"\x80\x04}\x94h\x00.",
    b"\x80\x04}\x94}\x94.", b"\x80\x04G"+struct.pack(">d",1.)+b"\x94.",
    b"\x80\x04e.", b"\x80\x04u.", b"\x80\x04"+b"("*4097+b"."])
def test_invalid_stack_memo_root(value):
    with pytest.raises(ValueError):
        parse(value)


def test_duplicate_names_rejected():
    value = pickle.dumps({"A": [1.,2.,3.], "B": [4.,5.,6.]}, protocol=4)
    value = value.replace(b"\x8c\x01B", b"\x8c\x01A")
    with pytest.raises(ValueError, match="Duplicate"):
        parse(value)


def test_non_float_container_rejected():
    for value in ({"A":[1,2,3],"B":[4.,5.,6.]}, {"A":[[1.,2.,3.]],"B":[4.,5.,6.]}):
        with pytest.raises(ValueError):
            parse(pickle.dumps(value,protocol=4))


def test_frames_checked():
    good = payload()
    assert good[2] == 0x95
    length = struct.unpack_from("<Q", good, 3)[0]
    # Legal frame termination between opcodes need not cover the whole stream.
    for changed in (0, 4, length+1, 2**63):
        with pytest.raises(ValueError):
            parse(good[:3] + struct.pack("<Q",changed) + good[11:])


def test_missing_stop_and_trailing_bytes():
    with pytest.raises(ValueError):
        parse(payload()+b"extra")
    with pytest.raises(ValueError):
        parse(b"\x80\x04}(")


def test_nonexecuting_global_rejection(tmp_path):
    # This malicious byte stream is never passed to pickle.loads.
    target = tmp_path/"should_not_exist"
    code = b"cos\nsystem\n(S'touch " + str(target).encode() + b"'\ntR."
    with pytest.raises(ValueError, match="Forbidden"):
        parse(b"\x80\x04"+code)
    assert not target.exists()


def test_selection_preserves_values_missing_and_aliases(tmp_path):
    names,_ = parse(payload())
    result,receipt = m.select_features(names,["A","Alias_A","B","Missing"],dimension=3)
    assert result["embeddings"].dtype == np.float32
    np.testing.assert_array_equal(result["present"],[1,1,1,0])
    np.testing.assert_array_equal(result["source_vector_object_id"],[0,0,1,-1])
    np.testing.assert_array_equal(result["embeddings"][0],np.array(fixture()["A"],dtype=np.float32))
    np.testing.assert_array_equal(result["embeddings"][-1],np.zeros(3))
    assert receipt["missing"] == ["Missing"]
    assert 0 < receipt["max_float32_roundoff"] < 1e-7
    with (tmp_path/"features.npz").open("wb") as stream:
        np.savez_compressed(stream,**result)
    with np.load(tmp_path/"features.npz",allow_pickle=False) as replay:
        assert all(np.array_equal(replay[k],v) for k,v in result.items())


@pytest.mark.parametrize("targets", [[],["B","A"],["A","A"],[" A"],[1]])
def test_invalid_selection(targets):
    with pytest.raises(ValueError):
        m.select_features({},targets,dimension=3)


@pytest.mark.parametrize("vector", [[0.,0.,0.],[1e100,1.,2.],[float("nan"),1.,2.],[1.,2.]])
def test_bad_available_numeric_vector(vector):
    with pytest.raises(ValueError):
        m.select_features({"A":array("d",vector)},["A"],dimension=3)


def test_fresh_json_never_overwrites(tmp_path):
    path = tmp_path/"receipt.json"
    m.fresh_json(path,{"a":1})
    with pytest.raises(FileExistsError):
        m.fresh_json(path,{"a":2})
    assert json.loads(path.read_text()) == {"a":1}


def test_bound_json_hash(tmp_path):
    path=tmp_path/"input.json"
    path.write_text('{"a":1}')
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    assert m.bound_json(path,sha)=={"a":1}
    with pytest.raises(ValueError,match="digest"):
        m.bound_json(path,"0"*64)


def test_contract_change_rejected_before_output(tmp_path):
    path=tmp_path/"contract.json"
    m.fresh_json(path,{"schema":m.SCHEMA})
    with pytest.raises(ValueError,match="digest"):
        m._convert(path,"0"*64,tmp_path/"output",0)
    assert not (tmp_path/"output").exists()


def test_nofollow_regular_readers(tmp_path):
    target=tmp_path/"target.json"
    target.write_text("{}")
    link=tmp_path/"link.json"
    link.symlink_to(target)
    with pytest.raises(OSError):
        m.digest(link)
    with pytest.raises(ValueError):
        m.read_regular(target,1)
    import os
    fifo=tmp_path/"pipe"
    os.mkfifo(fifo)
    with pytest.raises(ValueError,match="Regular"):
        m.digest(fifo)


def test_approved_host_and_gpu_guard(monkeypatch):
    monkeypatch.setattr(m.socket,"gethostname",lambda:"aida-login")
    with pytest.raises(ValueError,match="compute host"):
        m.runtime_guard()
    monkeypatch.setattr(m.socket,"gethostname",lambda:"cbsuvlaminck3.biohpc.cornell.edu")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","2")
    with pytest.raises(ValueError,match="hidden"):
        m.runtime_guard()


def test_sharing_reports_content_and_object_independently():
    a=array("d",[1.,2.,3.]); b=array("d",[1.,2.,3.])
    arrays,_=m.select_features({"A":a,"Alias_A":a,"B":b},["A","Alias_A","B","Missing"],dimension=3)
    report=m.sharing_audit(arrays,{"s":{"fit_targets":["A"],"tune_targets":["B"]}},["Alias_A"])
    assert report["source_object"]["duplicate_groups"]==[["A","Alias_A"]]
    assert report["source_object"]["cross_fit_held_groups"]==[]
    assert report["binary64_content"]["cross_fit_held_groups"]==[["A","Alias_A","B"]]
    assert report["float32_content"]["cross_fit_held_groups"]==[["A","Alias_A","B"]]


def test_registration_binds_metadata_before_numeric_read(tmp_path,monkeypatch):
    parent=tmp_path/"parent.json"
    m.fresh_json(parent,{"sources":{"replogle_k562":{"fit_targets":["FIT"],"tune_targets":["HELD"]},
                                  "nadig_jurkat":{"fit_targets":["JFIT"],"tune_targets":["JHELD"]}}})
    targets=tmp_path/"targets.csv"
    targets.write_text("target_gene\n"+"\n".join(f"OFF{i:03d}" for i in range(300))+"\n")
    protocol=tmp_path/"protocol.md"
    protocol.write_text("metadata only")
    monkeypatch.setattr(m,"TRAIN_SHA",m.digest(parent))
    monkeypatch.setattr(m,"TARGET_SHA",m.digest(targets))
    # Archive deliberately absent: registration must never read numeric data.
    archive=tmp_path/"absent.zip"
    out=tmp_path/"registration"
    sha=m.register(archive,parent,targets,protocol,out)
    registered=m.bound_json(out/"contract.json",sha)
    assert len(registered["selected_targets"])==304
    assert registered["expression_read"] is False
    assert registered["converter_sha256"]==m.digest(m.__file__)
    with pytest.raises(ValueError,match="Fresh"):
        m.register(archive,parent,targets,protocol,out)
    protocol.write_text("changed")
    assert m.plan(archive,parent,targets,protocol)["protocol"] != registered["protocol"]
    targets.write_text("target_gene\nBAD\n")
    with pytest.raises(ValueError,match="changed"):
        m.plan(archive,parent,targets,protocol)
