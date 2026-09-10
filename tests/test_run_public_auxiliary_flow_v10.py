"""Synthetic registered-feature and safe optimizer checkpoint orchestration tests."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import sys
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import run_public_auxiliary_flow_v10 as run
from test_public_auxiliary_flow_v10 import arrays, data, features, tiny_config


def test_actual_optimizer_checkpoint_roundtrip(tmp_path,data,features):
    result=run.flow.train_source(data,features,tiny_config(),arm="true",device="cpu")
    binding=run.save_checkpoint(tmp_path,result,"a"*64)
    model,optimizer,metadata=run.restore_checkpoint(binding,"a"*64,"cpu",tiny_config())
    for key,value in result.model.state_dict().items():
        assert torch.equal(value,model.state_dict()[key])
    assert all(v["step"].item()==4 for v in optimizer.state.values())
    assert run.flow.evaluate_source(model,data,result.features,steps=2,batch_size=16)==result.evaluation
    with pytest.raises(ValueError,match="configuration"):
        run.restore_checkpoint(binding,"a"*64,"cpu",replace(tiny_config(),gene_dim=10**9))


def test_checkpoint_rejects_missing_optimizer_state(tmp_path,data,features):
    result=run.flow.train_source(data,features,tiny_config(),device="cpu")
    result.optimizer_state["state"].clear()
    binding=run.save_checkpoint(tmp_path,result,"a"*64)
    with pytest.raises(ValueError,match="parameter states"):
        run.restore_checkpoint(binding,"a"*64,"cpu",tiny_config())


@pytest.mark.parametrize("change",["lr","eps","weight_decay","negative_variance","fractional_step"])
def test_checkpoint_rejects_optimizer_lineage_changes(tmp_path,data,features,change):
    result=run.flow.train_source(data,features,tiny_config(),device="cpu")
    if change in {"lr","eps","weight_decay"}:
        result.optimizer_state["param_groups"][0][change]=.123
    elif change=="negative_variance":
        next(iter(result.optimizer_state["state"].values()))["exp_avg_sq"].flatten()[0]=-1
    else:
        next(iter(result.optimizer_state["state"].values()))["step"].fill_(4.5)
    binding=run.save_checkpoint(tmp_path,result,"a"*64)
    with pytest.raises(ValueError):run.restore_checkpoint(binding,"a"*64,"cpu",tiny_config())


@pytest.mark.parametrize("value",[float("nan"),float("inf"),object(),np.array([1.]),{True:1},torch.tensor([float("nan")])])
def test_pack_rejects_unsafe_state(value):
    with pytest.raises(ValueError):run.pack_state(value)


def test_tree_rejects_duplicate_unknown_and_unreferenced_tensors():
    tree,arrays=run.pack_state({"a":torch.tensor([1.]),2:(False,None,1.2)})
    restored=run.unpack_state(tree,arrays)
    assert restored[2]==(False,None,1.2)
    extra={**arrays,"extra":np.zeros(1)}
    with pytest.raises(ValueError,match="Unreferenced"):run.unpack_state(tree,extra)
    with pytest.raises(ValueError):run.unpack_state({"scalar":float("inf")},{})
    with pytest.raises(ValueError):run.unpack_state({"dict":[["k",{"scalar":1}],["k",{"scalar":2}]]},{})


@pytest.mark.parametrize("n,held",[(374,75),(113,23),(8,2)])
def test_auxiliary_target_split_stable_and_source_keyed(n,held):
    source={"id":"replogle_k562","targets":[f"T{i:04d}" for i in range(n)]}
    split=run.split_targets(source)
    assert len(split["tune_targets"])==held
    assert len(split["fit_targets"])==n-held
    assert not set(split["fit_targets"])&set(split["tune_targets"])
    assert run.split_targets(source)==split
    assert run.split_targets({**source,"id":"nadig_jurkat"})!=split


@pytest.fixture
def registered(tmp_path,monkeypatch):
    targets=[f"t{i}" for i in range(8)]
    plan={"sources":[{"id":s,"targets":targets} for s in run.flow.SOURCES]}
    base={"schema":run.SCHEMA,"created_at":"2026-09-10T18:00:00+00:00","policy":run.policy(),
        "code_sha256":{"fake.py":"c"*64},"materialization":{"path":"contract.json","sha256":"a"*64},
        "cache_directory":"cache","cache_completion_sha256":"b"*64,
        "candidate_plan":{},"protocol":{"path":str(tmp_path/"protocol.md"),"sha256":"d"*64},
        "go_sources":run.go_source.GO_EXPECTED,
        "sources":{s["id"]:run.split_targets(s) for s in plan["sources"]},
        "excluded_targets":["protected","panel"],"torch_version":torch.__version__,"numpy_version":np.__version__}
    index=run.go.AnnotationIndex(terms={t:frozenset([f"GO:{i:07d}"]) for i,t in enumerate(targets)},
        source_ids={t:("UniProtKB:fake",) for t in targets},counts={},headers={},
        policy={"ontology_data_version":"synthetic"})
    monkeypatch.setattr(run,"_base",lambda *args:(copy.deepcopy(base),copy.deepcopy(plan)))
    monkeypatch.setattr(run,"codes",lambda:copy.deepcopy(base["code_sha256"]))
    monkeypatch.setattr(run.go_source,"_go_index",lambda path:index)
    out=tmp_path/"registration"
    sha=run.register("contract.json","a"*64,"cache","b"*64,tmp_path/"protocol.md",out)
    return out,sha,base


def test_registration_reconstructs_train_only_GO_and_absent_GenePT(registered):
    out,sha,base=registered
    record,matrices=run.validate(out/"training_contract.json",sha)
    for source,values in matrices.items():
        assert values.dtype==np.float32 and values.shape==(8,8)
        assert np.all(values[:,-1]==0)
        # Synthetic held-only terms are never admitted to the fit vocabulary.
        for target in base["sources"][source]["tune_targets"]:
            row=int(target[1:]);assert np.all(values[row]==0)
        assert (values[:,-2]>0).sum()==6


@pytest.mark.parametrize("mutation",["path","extra_binding","roles","code","complete"])
def test_resealed_registration_tampering_rejected(registered,mutation):
    out,sha,base=registered
    path=out/"training_contract.json";record=json.loads(path.read_text())
    if mutation=="path":record["features"][run.flow.SOURCES[0]]["path"]="elsewhere.npz"
    elif mutation=="extra_binding":record["features"][run.flow.SOURCES[0]]["unexpected"]=True
    elif mutation=="roles":record["sources"][run.flow.SOURCES[0]]["fit_targets"]=["panel"]
    elif mutation=="code":record["code_sha256"]={"fake.py":"0"*64}
    path.write_bytes(run.cache.old.canonical(record));sha=run.safe.digest(path)
    complete={"schema":run.SCHEMA,"registered":True,"training_contract_sha256":sha}
    if mutation=="complete":complete["registered"]=False
    (out/"complete.json").write_bytes(run.cache.old.canonical(complete))
    with pytest.raises(ValueError):run.validate(path,sha)


def test_cpu_loading_phase_restores_GPU_visibility_even_on_failure(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","2")
    monkeypatch.setattr(torch.cuda,"is_initialized",lambda:False)
    with pytest.raises(RuntimeError):
        with run.cpu_cache_phase():
            assert run.os.environ["CUDA_VISIBLE_DEVICES"]==""
            raise RuntimeError("fixture")
    assert run.os.environ["CUDA_VISIBLE_DEVICES"]=="2"
    monkeypatch.setattr(torch.cuda,"is_initialized",lambda:True)
    with pytest.raises(ValueError):
        with run.cpu_cache_phase():pass


def test_WandB_distribution_metrics_survive_allowlist():
    metrics={"model_centroid_mse":.1,"control_centroid_mse":.2,"model_mmd2":.3,"control_mmd2":.4,
        "variance_ratio_to_control":.7,"model_zero_fraction":.6,"negative_fraction":0.}
    values=run.scalar_metrics(run.tracking_evaluation(metrics))
    assert values["validation/target/variance_ratio"]==.7
    assert values["validation/target/zero_fraction"]==.6


def test_resource_counter_never_follows_sdk_log_links(tmp_path,monkeypatch):
    (tmp_path/"external.log").symlink_to("/nonexistent/unreadable")
    monkeypatch.setattr(torch.cuda,"is_initialized",lambda:False)
    value=run.resources(tmp_path,run.time.monotonic())
    assert value["output_bytes"]==len("/nonexistent/unreadable")


@pytest.mark.parametrize("tamper",[None,"loss","stage","event_count","truncate"])
def test_tracking_replays_actual_step_values_not_only_counts(tmp_path,tamper):
    history=[{"step":1,"flow_matching_loss":.3,"gradient_norm":.4,"learning_rate":.01}]
    evaluation={"metrics":{"model_centroid_mse":.1,"control_centroid_mse":.2,
        "variance_ratio_to_control":.8,"model_zero_fraction":.5}}
    receipt={"stage":"pretrain","mode":"offline","group":"public-auxiliary-flow-v10",
        "execution_status":"completed","training_exit_code":0,"event_count":2,"tracking_errors":0,
        "cloud_synced":False,"raw_artifacts_uploaded":False,"console_code_environment_capture":False}
    events=[{"event_index":1,**run.scalar_metrics({"step":1,"loss":.3,"cfm":.3,"gradient_norm":.4,"learning_rate":.01})},
        {"event_index":2,**run.scalar_metrics(run.tracking_evaluation(evaluation["metrics"]))}]
    if tamper=="loss":events[0]["train/loss"]=.9
    elif tamper=="stage":receipt["stage"]="posttrain"
    elif tamper=="event_count":receipt["event_count"]=3
    elif tamper=="truncate":events.pop()
    run.dump(tmp_path/"tracking.json",receipt)
    (tmp_path/"metrics.jsonl").write_text("".join(json.dumps(e)+"\n" for e in events))
    if tamper:
        with pytest.raises(ValueError):run.verify_tracking(tmp_path,history,evaluation)
    else:assert run.verify_tracking(tmp_path,history,evaluation)["events_replayed"]==2
