"""Registered source-specific GO flow pretraining with safe checkpoint replay.

No challenge-treated data, network, online logging or submission. CPU cache
authentication precedes first CUDA initialization. Frozen earlier runs unchanged.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import stat
import time

import numpy as np
import torch
import build_go_target_features as go
import prepare_public_joint_target_source_v3 as go_source
import public_auxiliary_cache_v10 as cache
import public_auxiliary_flow_v10 as flow
import public_training_readiness as safe
from wandb_training import TrainingTracker, scalar_metrics

SCHEMA = "public-auxiliary-flow-execution-v10"
ARMS = ("true", "shuffled", "constant")
GPU_UUID = "GPU-9d152176-395d-7140-4fb7-71c9e32b7c25"
GPU_VISIBLE = "2"
SEED = 20260926
MODULES = tuple(dict.fromkeys(("run_public_auxiliary_flow_v10.py", "public_auxiliary_flow_v10.py",
    "build_go_target_features.py", "wandb_training.py", *cache.MODULES)))
GO_KEYS = frozenset(("target_ids", "go_ids", "train_target_ids", "features", "known_symbol",
    "has_accepted_annotation", "present", "accepted_term_count", "in_vocab_term_count", "source_object_count"))
equal, require = cache.equal, cache.require


def dump(path, value):
    cache.candidate._dump(Path(path), value)


def codes():
    return {name: safe.digest(Path(__file__).with_name(name)) for name in MODULES}


def split_targets(source):
    targets = source["targets"]
    require(targets == sorted(set(targets)) and len(targets) >= 5, "Unique sorted source targets required")
    ranked = sorted(targets, key=lambda t: (hashlib.sha256(cache.old.canonical([SEED, source["id"], t])).digest(), t))
    count = math.ceil(len(targets)/5)
    return {"fit_targets": sorted(ranked[count:]), "tune_targets": sorted(ranked[:count])}


def policy():
    return {"stage": "pretrain", "steps": 600, "batch_size": 128, "learning_rate": 1e-4,
        "weight_decay": 1e-4, "grad_clip": 1., "heun_steps": 16, "hidden_dim": 128,
        "layers": 2, "seed": SEED, "arms": list(ARMS), "target_holdout_fraction": .2,
        "checkpoint_selection": "fixed_final_only", "go_transform": "binary_L2_then_GO_present_and_GenePT_absent_flags",
        "genept_used": False, "promoterai_used": False, "cpu_threads": 2,
        "gpu_uuid": GPU_UUID, "gpu_visible": GPU_VISIBLE, "gpu_memory_limit": 6 << 30,
        "rss_limit": 16 << 30, "output_limit": 8 << 30, "per_arm_seconds": 7200,
        "wandb_mode": "offline", "wandb_stage": "pretrain", "source_specific": True,
        "challenge_treated_used": False, "posttraining_performed": False, "submission_performed": False,
        "count_emitter_available": False, "automatic_promotion": False}


def _base(contract_path, contract_sha, cache_dir, completion_sha, protocol, created_at):
    contract_path, cache_dir, protocol = map(cache.canonical_path, (contract_path, cache_dir, protocol))
    parent, plan = cache.validate_contract(contract_path, contract_sha)
    receipt = safe.read_json(cache_dir / "complete.json", completion_sha)
    require(receipt.get("contract_sha256") == contract_sha and receipt.get("completed") is True,
        "Completed matching materialization required")
    safe.timestamp(created_at)
    require(datetime.fromisoformat(created_at).timestamp() <= time.time()+300, "Future training registration")
    return {"schema": SCHEMA, "created_at": created_at, "policy": policy(), "code_sha256": codes(),
        "materialization": {"path": str(contract_path), "sha256": contract_sha},
        "cache_directory": str(cache_dir), "cache_completion_sha256": completion_sha,
        "candidate_plan": parent["candidate_plan"], "protocol": {"path": str(protocol), "sha256": safe.digest(protocol)},
        "go_sources": go_source.GO_EXPECTED, "sources": {s["id"]: split_targets(s) for s in plan["sources"]},
        "excluded_targets": sorted(set(plan["panel_targets"] + plan["protected_excluded_targets"])),
        "torch_version": torch.__version__, "numpy_version": np.__version__}, plan


def register(contract_path, contract_sha, cache_dir, completion_sha, protocol, out):
    out = cache.canonical_path(out)
    require(not os.path.lexists(out), "Fresh training registration required")
    record, plan = _base(contract_path, contract_sha, cache_dir, completion_sha, protocol,
        datetime.now(timezone.utc).isoformat())
    out.mkdir(mode=0o700)
    # Roles are durably fixed before source GO vocabularies are constructed.
    dump(out / "roles.json", {"sources": record["sources"], "excluded_targets": record["excluded_targets"]})
    index = go_source._go_index(Path(__file__).resolve().parents[1] / go_source.GO_RELATIVE)
    features = {}
    for source in plan["sources"]:
        name = source["id"]
        built = go.build_features(index, target_ids=source["targets"], train_target_ids=record["sources"][name]["fit_targets"],
            fold_id="auxiliary-v10-"+name, sources=go_source.GO_EXPECTED)
        directory = out / name
        go.write_artifacts(built, directory)
        features[name] = {"path": str(directory / "features.npz"), "sha256": safe.digest(directory / "features.npz"),
            "receipt": {"path": str(directory / "receipt.json"), "sha256": safe.digest(directory / "receipt.json")}}
    record["features"] = features
    record["roles_sha256"] = safe.digest(out / "roles.json")
    require(record["code_sha256"] == codes(), "Code changed during registration")
    dump(out / "training_contract.json", record)
    sha = safe.digest(out / "training_contract.json")
    dump(out / "complete.json", {"schema": SCHEMA, "registered": True, "training_contract_sha256": sha})
    return sha


def validate(path, sha):
    path = cache.canonical_path(path)
    require(path.name == "training_contract.json", "Canonical training contract required")
    record = safe.read_json(path, sha)
    base, plan = _base(record["materialization"]["path"], record["materialization"]["sha256"],
        record["cache_directory"], record["cache_completion_sha256"], record["protocol"]["path"], record["created_at"])
    equal({k:v for k,v in record.items() if k not in {"features", "roles_sha256"}}, base, "Training contract reconstruction")
    equal(safe.read_json(path.with_name("roles.json"), record["roles_sha256"]),
        {"sources": record["sources"], "excluded_targets": record["excluded_targets"]}, "Frozen auxiliary roles")
    equal(safe.read_json(path.with_name("complete.json")),
        {"schema": SCHEMA, "registered": True, "training_contract_sha256": sha}, "Training registration completion")
    require(set(record["features"]) == set(record["sources"]), "Exact source feature roster required")
    index = go_source._go_index(Path(__file__).resolve().parents[1] / go_source.GO_RELATIVE)
    matrices = {}
    for source in plan["sources"]:
        name = source["id"]
        binding = record["features"][name]
        require(set(binding) == {"path", "sha256", "receipt"}, "GO binding fields")
        require(binding["path"] == str(path.parent/name/"features.npz"), "Wrong GO artifact path")
        require(binding["receipt"]["path"] == str(path.parent/name/"receipt.json"), "Wrong GO receipt path")
        receipt = safe.binding(binding["receipt"])
        arrays = cache.bounded.read_npz(binding["path"], binding["sha256"], GO_KEYS)
        built = go.build_features(index, target_ids=source["targets"], train_target_ids=record["sources"][name]["fit_targets"],
            fold_id="auxiliary-v10-"+name, sources=go_source.GO_EXPECTED)
        for key in GO_KEYS:
            require(arrays[key].dtype == built.arrays[key].dtype and np.array_equal(arrays[key], built.arrays[key]),
                "GO features differ from train-target-only reconstruction")
        equal(receipt, {**built.receipt, "npz_filename": "features.npz",
            "npz_size_bytes": Path(binding["path"]).stat().st_size, "npz_sha256": binding["sha256"]}, "GO receipt reconstruction")
        x = arrays["features"].astype(np.float32)
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1.)
        matrices[name] = np.column_stack((x, arrays["present"].astype(np.float32), np.zeros(len(x), np.float32))).astype(np.float32)
    return record, matrices


def pack_state(value):
    arrays = {}
    def encode(v):
        if isinstance(v, torch.Tensor):
            a = v.detach().cpu().numpy().copy()
            require(a.dtype.kind in "fiu" and np.isfinite(a).all(), "Only finite numeric checkpoint tensors")
            key = f"tensor_{len(arrays):05d}"
            arrays[key] = a
            return {"tensor": key, "dtype": a.dtype.str, "shape": list(a.shape)}
        if isinstance(v, dict):
            require(all(type(k) in (str,int) for k in v), "Typed state dictionary keys required")
            return {"dict": [[k,encode(w)] for k,w in v.items()]}
        if isinstance(v, (list,tuple)):
            return {"tuple" if isinstance(v,tuple) else "list": [encode(w) for w in v]}
        require(v is None or type(v) in (str,bool,int,float), "Unsupported checkpoint metadata")
        require(type(v) is not float or math.isfinite(v), "Nonfinite checkpoint scalar")
        return {"scalar": v}
    return encode(value), arrays


def unpack_state(tree, arrays):
    seen = set()
    def decode(node, depth=0):
        require(depth < 20 and isinstance(node,dict), "Invalid checkpoint tree")
        if set(node) == {"tensor", "dtype", "shape"}:
            key = node["tensor"]
            require(key in arrays and key not in seen, "Missing/repeated state tensor")
            a = arrays[key]; seen.add(key)
            require(a.dtype.str == node["dtype"] and list(a.shape) == node["shape"]
                and a.dtype.kind in "fiu" and np.isfinite(a).all(), "Invalid checkpoint array")
            return torch.from_numpy(a.copy())
        require(len(node) == 1, "Unexpected checkpoint tree fields")
        if "dict" in node:
            pairs = node["dict"]
            require(isinstance(pairs,list) and all(isinstance(p,list) and len(p)==2 and type(p[0]) in (str,int) for p in pairs), "Invalid state map")
            require(len({p[0] for p in pairs}) == len(pairs), "Duplicate state keys")
            return {k:decode(v,depth+1) for k,v in pairs}
        for name, factory in (("list",list),("tuple",tuple)):
            if name in node:
                require(isinstance(node[name],list), "Invalid state sequence")
                return factory(decode(v,depth+1) for v in node[name])
        require("scalar" in node, "Unknown checkpoint node")
        v = node["scalar"]
        require(v is None or type(v) in (str,bool,int,float), "Invalid state scalar")
        require(type(v) is not float or math.isfinite(v), "Nonfinite state scalar")
        return v
    result = decode(tree)
    require(seen == set(arrays), "Unreferenced checkpoint tensors")
    return result


def save_checkpoint(out, result, contract_sha):
    tree, arrays = pack_state({"model": result.model.state_dict(), "optimizer": result.optimizer_state})
    require(sum(a.nbytes for a in arrays.values()) < 256 << 20, "Checkpoint allocation limit")
    with os.fdopen(os.open(out/"checkpoint.npz", os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600), "wb") as stream:
        np.savez_compressed(stream, **arrays)
    meta = {"schema": SCHEMA, "training_contract_sha256": contract_sha,
        "config": asdict(result.model.config), "tree": tree, "tensor_keys": sorted(arrays),
        "checkpoint_sha256": safe.digest(out/"checkpoint.npz"), "optimizer_steps": result.metadata["optimizer_steps"]}
    dump(out/"checkpoint.json", meta)
    return {"path": str(out/"checkpoint.json"), "sha256": safe.digest(out/"checkpoint.json")}


def restore_checkpoint(binding, contract_sha, device, expected_config):
    meta = safe.binding(binding)
    require(set(meta) == {"schema","training_contract_sha256","config","tree","tensor_keys","checkpoint_sha256","optimizer_steps"},
        "Exact checkpoint metadata fields required")
    require(meta["schema"] == SCHEMA and meta["training_contract_sha256"] == contract_sha, "Checkpoint contract mismatch")
    require(isinstance(expected_config,flow.PilotConfig), "Registered expected model configuration required")
    equal(meta["config"], asdict(expected_config), "Checkpoint architecture must match registered configuration")
    require(type(meta["optimizer_steps"]) is int and meta["optimizer_steps"] == expected_config.steps, "Fixed final step required")
    require(isinstance(meta["tensor_keys"],list) and len(meta["tensor_keys"]) <= 512
        and meta["tensor_keys"] == [f"tensor_{i:05d}" for i in range(len(meta["tensor_keys"]))], "Exact bounded tensor roster")
    arrays = cache.bounded.read_npz(Path(binding["path"]).with_name("checkpoint.npz"), meta["checkpoint_sha256"], set(meta["tensor_keys"]))
    state = unpack_state(meta["tree"], arrays)
    require(isinstance(state,dict) and set(state)=={"model","optimizer"}, "Model and optimizer state required")
    config = flow.PilotConfig(**meta["config"])
    model = flow.AuxiliaryConditionalFlow(config).to(device)
    model.load_state_dict(state["model"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    expected_groups = [{k:v for k,v in g.items() if k != "params"} for g in optimizer.param_groups]
    optimizer.load_state_dict(state["optimizer"])
    equal([{k:v for k,v in g.items() if k != "params"} for g in optimizer.param_groups], expected_groups,
        "AdamW hyperparameters must match registered configuration")
    require(set(optimizer.state) == set(model.parameters()), "All optimizer parameter states required")
    require(all(int(v["step"].item()) == meta["optimizer_steps"] for v in optimizer.state.values()), "AdamW step lineage mismatch")
    for parameter, state in optimizer.state.items():
        require(set(state)=={"step","exp_avg","exp_avg_sq"} and state["step"].numel()==1
            and float(state["step"].item())==meta["optimizer_steps"]
            and state["exp_avg"].shape==parameter.shape and state["exp_avg_sq"].shape==parameter.shape,
            "Invalid AdamW moment/step state")
        require(state["exp_avg"].dtype==parameter.dtype and state["exp_avg_sq"].dtype==parameter.dtype
            and torch.all(state["exp_avg_sq"] >= 0).item(), "Invalid AdamW moment values")
    return model, optimizer, meta


def tracking_evaluation(metrics):
    values = dict(metrics)
    values["variance_ratio"] = metrics["variance_ratio_to_control"]
    values["zero_fraction"] = metrics["model_zero_fraction"]
    return {"kind_metrics":{"target":values}}


def verify_tracking(directory, history, evaluation):
    receipt = safe.read_json(directory/"tracking.json")
    for key,value in {"stage":"pretrain","mode":"offline","group":"public-auxiliary-flow-v10",
        "execution_status":"completed","training_exit_code":0,"event_count":len(history)+1,
        "tracking_errors":0,"cloud_synced":False,"raw_artifacts_uploaded":False,
        "console_code_environment_capture":False}.items():
        equal(receipt.get(key),value,"Tracking receipt "+key)
    with safe.regular_reader(directory/"metrics.jsonl") as stream:
        payload=stream.read((8 << 20)+1)
    require(len(payload) <= 8 << 20,"Tracking journal size limit")
    observed=[json.loads(line) for line in payload.splitlines()]
    expected=[]
    for index,h in enumerate(history,1):
        expected.append({"event_index":index,**scalar_metrics({"step":h["step"],"loss":h["flow_matching_loss"],
            "cfm":h["flow_matching_loss"],"gradient_norm":h["gradient_norm"],"learning_rate":h["learning_rate"]})})
    expected.append({"event_index":len(history)+1,**scalar_metrics(tracking_evaluation(evaluation["metrics"]))})
    equal(observed,expected,"Every actual optimizer and final evaluation tracking event")
    return {"events_replayed":len(expected),"journal_sha256":safe.digest(directory/"metrics.jsonl")}


@contextmanager
def cpu_cache_phase():
    require(not torch.cuda.is_initialized(), "Authenticate caches before first CUDA initialization")
    previous = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try: yield
    finally:
        if previous is None: os.environ.pop("CUDA_VISIBLE_DEVICES",None)
        else: os.environ["CUDA_VISIBLE_DEVICES"] = previous


def resources(out, started):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    require(rss <= policy()["rss_limit"], "Training RSS limit")
    total, stack = 0, [out]
    while stack:
        for p in os.scandir(stack.pop()):
            info = p.stat(follow_symlinks=False)
            if stat.S_ISDIR(info.st_mode): stack.append(p.path)
            else: total += info.st_size  # SDK log links are counted, never followed.
    require(total <= policy()["output_limit"], "Training output limit")
    gpu = torch.cuda.max_memory_allocated(0) if torch.cuda.is_initialized() else 0
    require(gpu <= policy()["gpu_memory_limit"], "Training GPU memory limit")
    return {"peak_rss_bytes": rss, "gpu_peak_allocated_bytes": gpu, "output_bytes": total, "elapsed_seconds": time.monotonic()-started}


def run(path, sha, out):
    started = time.monotonic()
    torch.set_num_threads(2)
    out = cache.canonical_path(out)
    require(not os.path.lexists(out), "Fresh training output required")
    with cpu_cache_phase():
        cache.runtime_preflight()
        record, features = validate(path,sha)
        arrays, _ = cache.load_caches(record["materialization"]["path"], record["materialization"]["sha256"],
            record["cache_directory"], record["cache_completion_sha256"])
    require(os.environ.get("CUDA_VISIBLE_DEVICES") in {GPU_VISIBLE,GPU_UUID}, "Registered GPU selection required")
    require(torch.cuda.is_available() and torch.cuda.device_count()==1, "One actual CUDA GPU required")
    props = torch.cuda.get_device_properties(0)
    device_uuid = "GPU-" + str(props.uuid).removeprefix("GPU-")
    require(props.name == "NVIDIA GeForce RTX 2080 Ti" and device_uuid==GPU_UUID, "Registered GPU identity differs")
    torch.cuda.set_per_process_memory_fraction((6 << 30)/props.total_memory,0)
    out.mkdir(mode=0o700)
    dump(out/"started.json", {"schema": SCHEMA, "training_contract_sha256": sha, "pid": os.getpid(),
        "started_at": datetime.now(timezone.utc).isoformat(), "gpu_uuid": device_uuid, "policy": policy()})
    completed = []
    for source in flow.SOURCES:
        data = flow.prepare_source(arrays[source], **record["sources"][source], excluded_targets=record["excluded_targets"])
        config = flow.PilotConfig(gene_dim=arrays[source]["expression"].shape[1],target_dim=features[source].shape[1])
        for arm in ARMS:
            arm_started = time.monotonic()
            directory = out/(source+"__"+arm); directory.mkdir(mode=0o700)
            tracker = TrainingTracker(directory/"tracking", stage="pretrain", mode="offline",
                project="virtual-cell-challenge-2026", group="public-auxiliary-flow-v10", name=source+"-"+arm,
                config={"contract_sha256":sha,"steps":600,"batch_size":128,"learning_rate":1e-4,
                    "seed":SEED,"feature_mode":arm,"target_features_sha256":record["features"][source]["sha256"]})
            def progress(step, values):
                require(time.monotonic()-arm_started <= policy()["per_arm_seconds"], "Complete arm wall-time limit")
                tracker.log(scalar_metrics({"step":step,"loss":values["flow_matching_loss"],"cfm":values["flow_matching_loss"],
                    "gradient_norm":values["gradient_norm"],"learning_rate":values["learning_rate"]}))
                if step % 100 == 0:
                    print(cache.old.canonical({"source":source,"arm":arm,"step":step,"loss":values["flow_matching_loss"],
                        "resources":resources(out,started)}).decode(),flush=True)
            try:
                result = flow.train_source(data,features[source],config,arm=arm,device="cuda:0",progress=progress)
                tracker.summary(tracking_evaluation(result.evaluation["metrics"]))
                checkpoint = save_checkpoint(directory,result,sha)
                restored, optimizer, _ = restore_checkpoint(checkpoint,sha,"cuda:0",config)
                replay = flow.evaluate_source(restored,data,result.features,steps=16,batch_size=128,device="cuda:0")
                equal(replay,result.evaluation,"Restored final evaluation replay")
                require(time.monotonic()-arm_started <= policy()["per_arm_seconds"], "Complete arm replay wall-time limit")
                dump(directory/"history.json", {"steps":result.history})
                dump(directory/"summary.json", {"schema":SCHEMA,"training_contract_sha256":sha,"checkpoint":checkpoint,
                    "metadata":result.metadata,"evaluation":result.evaluation,"saved_checkpoint_replayed":True,
                    "history_sha256":safe.digest(directory/"history.json"),"posttraining_performed":False})
                tracker.finish(0)
                require(tracker.record["event_count"]==601 and tracker.record["tracking_errors"]==0,
                    "Every optimizer step and final evaluation must be tracked")
                tracking_audit=verify_tracking(directory/"tracking",result.history,result.evaluation)
                completed.append({"source":source,"arm":arm,"summary_path":str(directory/"summary.json"),
                    "summary_sha256":safe.digest(directory/"summary.json"),"tracking_sha256":safe.digest(directory/"tracking/tracking.json"),
                    "evaluation":result.evaluation["metrics"],"sampling_sha256":result.metadata["sampling_sha256"],
                    "tracking_audit":tracking_audit})
                del restored,optimizer,result
                torch.cuda.empty_cache()
            except BaseException:
                if not tracker.journal.closed: tracker.finish(1)
                raise
    for source in flow.SOURCES:
        require(len({r["sampling_sha256"] for r in completed if r["source"]==source})==1,"Arm sampling mismatch")
    require(record["code_sha256"]==codes(),"Code changed during training")
    dump(out/"complete.json", {"schema":SCHEMA,"completed":True,"completed_at":datetime.now(timezone.utc).isoformat(),
        "training_contract_sha256":sha,"runs":completed,"optimizer_steps":3600,"tracking_events":3606,
        "pretraining_performed":True,"posttraining_performed":False,"submission_performed":False,
        "full_gene_count_emitter_available":False,"resources":resources(out,started)})
    return safe.digest(out/"complete.json")


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest="action",required=True)
    r=sub.add_parser("register")
    for name in ("materialization-contract","materialization-sha","cache-dir","completion-sha","protocol","out"):
        r.add_argument("--"+name,required=True)
    r=sub.add_parser("run")
    for name in ("contract","sha","out"): r.add_argument("--"+name,required=True)
    a=p.parse_args()
    if a.action=="register":
        cache.runtime_preflight()
        value=register(a.materialization_contract,a.materialization_sha,a.cache_dir,a.completion_sha,a.protocol,a.out)
    else: value=run(a.contract,a.sha,a.out)
    print(value,flush=True)


if __name__=="__main__": main()
