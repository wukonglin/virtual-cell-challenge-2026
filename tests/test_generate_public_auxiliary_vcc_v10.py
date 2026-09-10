"""Synthetic inference integration tests: never public or challenge expression."""
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import generate_public_auxiliary_vcc_v10 as gen


def test_donors_are_exact_disjoint_deterministic_and_context_specific():
    ids = ["cell"+str(i) for i in range(1000)]
    roles = gen.donor_roles(ids,"A")
    assert roles == gen.donor_roles(ids,"A")
    assert len(roles["context_rows"]) == 32
    assert len(roles["base_rows"]) == 400
    assert len(set(roles["context_rows"]+roles["base_rows"])) == 432
    assert roles != gen.donor_roles(ids,"B")
    reverse = gen.donor_roles(ids[::-1],"A")
    for role in roles:
        assert [ids[i] for i in roles[role]] == [ids[::-1][i] for i in reverse[role]]


@pytest.mark.parametrize("ids,context,kw", [
    (["a"]*500,"A",{}),([str(i) for i in range(500)],"D",{}),
    ([str(i) for i in range(400)],"A",{}),(["",*map(str,range(500))],"A",{}),
    (list(range(500)),"A",{}),([str(i) for i in range(500)],"A",{"context_count":1}),
    ([str(i) for i in range(500)],"A",{"base_count":False}),
])
def test_invalid_donor_rosters_fail(ids,context,kw):
    with pytest.raises(ValueError): gen.donor_roles(ids,context,**kw)


def test_context_summary_matches_float64_population_formula():
    x = np.random.default_rng(7).uniform(size=(32,9)).astype(np.float32)
    expected = np.r_[x.astype(float).mean(0),x.astype(float).std(0)].astype(np.float32)
    np.testing.assert_array_equal(gen.context_summary(x),expected)
    assert gen.context_summary(x).dtype == np.float32


@pytest.mark.parametrize("x", [np.ones((31,2),np.float32),np.ones((32,2),np.float64),
    np.full((32,2),np.nan,np.float32),np.full((32,2),-1,np.float32)])
def test_context_summary_rejects_bad_cells(x):
    with pytest.raises(ValueError): gen.context_summary(x)


def test_rounding_seed_stable_target_and_context_specific():
    seeds = {gen.round_seed(c,t) for c in "ABC" for t in ["GENE1","GENE2"]}
    assert len(seeds) == 6
    assert gen.round_seed("A","GENE1") == gen.round_seed("A","GENE1")
    assert all(type(s) is int and 0 <= s < 2**64 for s in seeds)


def test_fixed_equal_ensemble_and_context_broadcast(monkeypatch):
    sources = gen.flow.SOURCES
    models = dict(zip(sources,[object(),object()]))
    features = {s:np.array([i,1],np.float32) for i,s in enumerate(sources)}
    control = np.arange(15,dtype=np.float32).reshape(5,3)
    context = np.arange(6,dtype=np.float32)
    calls = []
    def sample(model,base,feat,ctx,steps):
        calls.append(model)
        assert steps == 16
        np.testing.assert_array_equal(base.numpy(),control)
        np.testing.assert_array_equal(ctx.numpy(),np.broadcast_to(context,(5,6)))
        i = list(models.values()).index(model)
        np.testing.assert_array_equal(feat.numpy(),np.broadcast_to(features[sources[i]],(5,2)))
        return base + 2*i
    monkeypatch.setattr(gen.flow,"sample_heun",sample)
    np.testing.assert_array_equal(gen.infer(models,control,features,context,device="cpu"),control+1)
    assert calls == list(models.values())


def test_missing_ensemble_source_rejected():
    with pytest.raises(ValueError): gen.infer({},np.zeros((3,2)),{},np.zeros(4),device="cpu")


def test_real_torch_models_inference_nonnegative_and_no_fit():
    torch.set_num_threads(2)
    models = {s:gen.flow.AuxiliaryConditionalFlow(gen.flow.PilotConfig(gene_dim=3,target_dim=2)) for s in gen.flow.SOURCES}
    before = {s:{k:v.clone() for k,v in m.state_dict().items()} for s,m in models.items()}
    x = np.array([[0,1,2],[3,0,4]],np.float32)
    out = gen.infer(models,x,{s:np.ones(2,np.float32) for s in models},np.ones(6,np.float32),device="cpu")
    assert out.shape == x.shape and out.dtype == np.float32 and (out >= 0).all()
    np.testing.assert_allclose(out,x,rtol=2e-7)
    for s,m in models.items():
        assert all(torch.equal(v,before[s][k]) for k,v in m.state_dict().items())
        assert all(p.grad is None for p in m.parameters())


def test_go_union_uses_original_fitted_vocab_and_exact_replay(monkeypatch):
    record = {"sources":{s:{"fit_targets":["FIT"],"tune_targets":["HELD"]} for s in gen.flow.SOURCES},
        "features":{s:{"path":"synthetic","sha256":"a"*64} for s in gen.flow.SOURCES}}
    original = {s:np.array([[1,0,1,0],[0,1,1,0]],np.float32) for s in gen.flow.SOURCES}
    seen = []
    def build(index,*,target_ids,train_target_ids,fold_id,sources):
        seen.append((target_ids,train_target_ids))
        lookup = {"FIT":[1,0],"HELD":[0,1],"NEW":[1,1]}
        return SimpleNamespace(arrays={"go_ids":np.array(["GO:1","GO:2"]),"target_ids":np.array(target_ids),
            "features":np.array([lookup[t] for t in target_ids],np.float32),"present":np.ones(len(target_ids),bool)})
    monkeypatch.setattr(gen.go_source,"_go_index",lambda _:object())
    monkeypatch.setattr(gen.go,"build_features",build)
    monkeypatch.setattr(gen.cache.bounded,"read_npz",lambda *args:{"go_ids":np.array(["GO:1","GO:2"])})
    features,coverage = gen.official_features(record,original,["NEW"])
    assert seen == [(["FIT","HELD","NEW"],["FIT"])]*2
    for source in features:
        np.testing.assert_allclose(features[source],[[2**-.5,2**-.5,1,0]],rtol=1e-6)
        assert coverage[source]["go_present"] == 1 and coverage[source]["genept_present"] == 0
        assert coverage[source]["features_sha256"] == hashlib.sha256(features[source].tobytes()).hexdigest()
    original[gen.flow.SOURCES[0]][0,0] = .5
    with pytest.raises(ValueError,match="replay exactly"): gen.official_features(record,original,["NEW"])


def test_metadata_exact_axes_only_no_expression():
    # Authenticated small public CSV/JSON metadata is safe; no H5AD reads.
    genes,targets = gen.metadata()
    assert len(genes) == 18533 and len(set(genes)) == 18533
    assert len(targets) == 300 and len(set(targets)) == 300


def test_policy_honestly_labels_partial_gene_exploratory_candidate():
    p = gen.policy()
    assert p["ensemble"] == {s:.5 for s in gen.flow.SOURCES}
    assert p["strength"] == 1 and p["heun_steps"] == 16
    for key in ("challenge_treated_used","genept_used","promoterai_used","posttraining_performed",
                "scientific_improvement_established","library_conservation","clipping_or_thinning",
                "upload_performed_by_generator"):
        assert p[key] is False
