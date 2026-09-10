"""Synthetic controls/GO only; no production cache, model jobs or dev evaluation."""
from pathlib import Path
import json
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import public_control_anchor_v2 as v
from public_flow_conditioning import FeatureTable
from public_flow_stable_conditioning import METHOD


def anchor(control, delta, ntc=None, **kwargs):
    return v.anchor_expression(control, delta, is_ntc=np.zeros(len(control), dtype=bool) if ntc is None else ntc,
                               representation=v.REPRESENTATION, **kwargs)


def fixture():
    return dict(targets=("A", "B", "A", "B"), sources=("K", "K", "J", "J"),
        control_context=np.array([[1., 2, .1, .2], [1.1, 2.1, .1, .2], [2, 3, .2, .3], [2.1, 3.1, .2, .3]]),
        mean_delta=np.array([[.2, -.1], [-.3, .2], [.1, -.2], [-.2, .3]]),
        table=FeatureTable("go", ("A", "B"), ("term1", "term2"), np.eye(2), "synthetic", "a" * 64),
        fold_id="registered_public", permitted_training_sources=("K", "J"), heldout_sources=("H",),
        excluded_targets=("DENIED",), representation=v.REPRESENTATION)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_zero_delta_exact_elementwise_identity(dtype):
    control = np.array([[0., .1, .3], [.2, 0, .8]], dtype=dtype)
    prediction = anchor(control, np.zeros(3))
    np.testing.assert_array_equal(prediction.raw, control)
    np.testing.assert_array_equal(prediction.nonnegative, control)
    assert prediction.metrics["zero_effect_identity_verified"] is True
    assert prediction.metrics["projection_fraction"] == 0


def test_ntc_bypasses_perturbation_even_for_unknown_target():
    control = np.array([[0., .1], [.2, 0]])
    prediction = anchor(control, [-100, 100], ntc=np.array([True, True]))
    np.testing.assert_array_equal(prediction.nonnegative, control)
    np.testing.assert_array_equal(prediction.raw, control)
    np.testing.assert_array_equal(prediction.applied_delta, np.zeros_like(control))
    assert prediction.metrics["ntc_identity_verified"] is True


def test_partial_ntc_mask_is_row_specific():
    control = np.array([[0., .1], [.2, 0]])
    prediction = anchor(control, [-1, 2], ntc=np.array([True, False]))
    np.testing.assert_array_equal(prediction.nonnegative, [[0, .1], [0, 2]])
    assert prediction.metrics["projection_fraction"] == .25
    assert prediction.metrics["projection_fraction_perturbed"] == .5


def test_projection_is_explicit_and_quantified():
    control = np.array([[0., 1], [.5, 2]])
    prediction = anchor(control, [-1, .2])
    np.testing.assert_allclose(prediction.raw, [[-1, 1.2], [-.5, 2.2]])
    np.testing.assert_allclose(prediction.nonnegative, [[0, 1.2], [0, 2.2]])
    metrics = prediction.metrics
    assert metrics["projection_fraction"] == .5
    assert metrics["projection_mean_absolute_correction"] == .375
    assert metrics["projection_max_absolute_correction"] == 1
    assert metrics["raw_negative_magnitude_mean"] == .75
    assert metrics["raw_negative_rms"] == pytest.approx(np.sqrt(1.25 / 4))
    assert metrics["projection_centroid_distortion_rms"] == pytest.approx(.75 / np.sqrt(2))
    assert metrics["projected_negative_fraction"] == 0


def test_additive_residual_can_lift_zero_expression():
    prediction = anchor(np.zeros((3, 2)), [.2, 1.1])
    np.testing.assert_allclose(prediction.nonnegative, np.tile([.2, 1.1], (3, 1)))
    assert not np.array_equal(prediction.nonnegative, np.round(prediction.nonnegative))


def test_gain_zero_exact_identity():
    control = np.array([[0., 1], [.3, 2]])
    prediction = anchor(control, [-5, 7], gain=0)
    np.testing.assert_array_equal(prediction.nonnegative, control)


@pytest.mark.parametrize("gain", [-1, 1.1, np.nan, np.inf, True, "1"])
def test_invalid_gain_refused(gain):
    with pytest.raises(ValueError, match="gain"):
        anchor(np.ones((2, 2)), np.ones(2), gain=gain)


@pytest.mark.parametrize("mask", [[0, 1], [True], True, np.array([[True], [False]])])
def test_explicit_boolean_ntc_mask_required(mask):
    with pytest.raises(ValueError, match="NTC"):
        anchor(np.ones((2, 2)), np.ones(2), ntc=mask)


@pytest.mark.parametrize("delta", [[np.nan, 0], [np.inf, 0], [1, 2, 3], [[1, 2]], ['a', 'b']])
def test_bad_delta_refused(delta):
    with pytest.raises(ValueError):
        anchor(np.ones((2, 2)), delta)


@pytest.mark.parametrize("control", [[[np.nan, 0], [1, 2]], [[-1, 0], [1, 2]], [[np.inf, 0], [1, 2]]])
def test_invalid_control_refused(control):
    with pytest.raises(ValueError):
        anchor(control, [1, 2])


def test_overflow_refused_not_hidden_by_projection():
    with pytest.raises(ValueError, match="overflow"):
        anchor(np.full((2, 2), 1e308), np.full(2, 1e308))


@pytest.mark.parametrize("representation", ["counts", "log_counts", "CP10k", None])
def test_raw_count_or_undeclared_representation_refused(representation):
    with pytest.raises(ValueError, match="raw counts"):
        v.anchor_expression(np.ones((2, 2)), np.ones(2), is_ntc=np.zeros(2, dtype=bool), representation=representation)


def test_input_arrays_immutable_and_outputs_readonly():
    control, delta = np.array([[0., 1], [.3, 2]]), np.array([-.5, .2])
    before_c, before_d = control.copy(), delta.copy()
    prediction = anchor(control, delta)
    np.testing.assert_array_equal(control, before_c)
    np.testing.assert_array_equal(delta, before_d)
    assert not prediction.raw.flags.writeable
    assert not prediction.nonnegative.flags.writeable


def test_paired_population_diagnostics_report_raw_and_projected():
    control = np.array([[0., 1], [.3, 2], [.2, 1.3]])
    prediction = anchor(control, [-.5, .2])
    report = v.paired_population_diagnostics(prediction, control, control.copy(), representation=v.REPRESENTATION)
    assert report["raw"]["negative_fraction"] == .5
    assert report["nonnegative"]["negative_fraction"] == 0
    assert report["used_for_fitting_or_selection"] is False
    assert report["raw"]["model_centroid_mse"] != report["nonnegative"]["model_centroid_mse"]


def test_source_folds_keep_all_batches_together():
    targets, sources = ("A", "A", "B", "A", "C"), ("K", "K", "K", "J", "J")
    folds = v.source_holdout_folds(targets, sources)
    for fold in folds:
        assert all(sources[i] != fold.held_source for i in fold.fit_indices)
        assert all(sources[i] == fold.held_source for i in fold.evaluation_indices)
        assert set(fold.fit_indices).isdisjoint(fold.evaluation_indices)
        assert {targets[i] for i in fold.evaluation_indices} == {"A"}
        assert len(fold.excluded_missing_target_indices) == 1


def test_no_overlap_source_fold_is_explicitly_empty():
    folds = v.source_holdout_folds(("A", "B"), ("K", "J"))
    assert all(not fold.evaluation_indices for fold in folds)
    assert all(len(fold.excluded_missing_target_indices) == 1 for fold in folds)


@pytest.mark.parametrize("targets,sources", [(('A',), ('K',)), (('A','B'), ('K',)), (('A','B'), ('K','K'))])
def test_invalid_source_folds_refused(targets, sources):
    with pytest.raises(ValueError):
        v.source_holdout_folds(targets, sources)


@pytest.mark.parametrize("arm", v.ARMS)
def test_synthetic_fit_and_ntc_null(arm):
    inputs = fixture()
    model = v.fit_residual_decoder(**inputs, arm=arm)
    contexts = inputs['control_context'][:2]
    delta = model.predict_delta(('A','NTC'), contexts, is_ntc=np.array([False,True]))
    assert delta.shape == (2,2)
    np.testing.assert_array_equal(delta[1], [0,0])
    assert np.isfinite(delta).all()
    if arm in {'true','shuffled'}:
        assert model.target_provenance['standardization_method'] == METHOD
    if arm == 'control':
        np.testing.assert_array_equal(delta, np.zeros((2,2)))


@pytest.mark.parametrize("arm", v.ARMS)
def test_unknown_perturbed_target_refused(arm):
    inputs=fixture()
    model=v.fit_residual_decoder(**inputs,arm=arm)
    with pytest.raises(ValueError,match='overlapping'):
        model.predict_delta(('NEW',), inputs['control_context'][:1], is_ntc=np.array([False]))


@pytest.mark.parametrize('change', ['heldout', 'unauthorized', 'excluded'])
def test_training_admission_checked_before_effect_arrays(change):
    class DoNotRead:
        def __array__(self, *args, **kwargs):
            raise AssertionError('Expression must not be touched before admission')
    inputs=fixture()
    inputs['mean_delta']=DoNotRead()
    if change=='heldout':
        inputs['heldout_sources']=('K',)
    elif change=='unauthorized':
        inputs['permitted_training_sources']=('J',)
    else:
        inputs['excluded_targets']=('A',)
    with pytest.raises(ValueError,match='Unauthorized'):
        v.fit_residual_decoder(**inputs,arm='true')


def test_recipient_context_cannot_mutate_fitted_statistics():
    inputs=fixture()
    model=v.fit_residual_decoder(**inputs,arm='true')
    mean,scale,weights=model.context_mean.copy(),model.context_scale.copy(),model.coefficients.copy()
    model.predict_delta(('A',), np.ones((1,4))*100, is_ntc=np.array([False]))
    np.testing.assert_array_equal(mean,model.context_mean)
    np.testing.assert_array_equal(scale,model.context_scale)
    np.testing.assert_array_equal(weights,model.coefficients)


def test_policy_has_no_development_tuning_or_count_claim():
    policy=v.policy()
    assert policy['ridge_lambda']==10
    assert policy['gain']==1
    assert policy['hyperparameter_selection'] is False
    assert policy['count_emitter'] is False


def flat_cache():
    targets=np.array(['A','B','A','B'])
    sources=np.array(['K','K','J','J'])
    batches=np.array(['kbatch','kbatch','jbatch','jbatch'])
    controls,contexts,shams,treated,labels,context=[] ,[],[],[],[],[]
    for group in range(4):
        source_offset=0 if group<2 else .3
        population=np.array([[1+j/10+source_offset, .1+j/20] for j in range(8)],dtype=np.float32)
        contexts.append(population)
        controls.append(population+.05)
        shams.append(population+.1)
        treated.append(population+([.1,-.05] if targets[group]=='A' else [-.05,.1]))
        labels.extend([group]*8)
        context.append(np.r_[population.mean(0),population.std(0)])
    return {'genes':np.array(['gene1','gene2']),'targets':targets,'sources':sources,'batches':batches,
        'context':np.array(context),'context_control':np.concatenate(contexts),'context_control_group':np.array(labels),
        'control':np.concatenate(controls),'control_group':np.array(labels),
        'sham_control':np.concatenate(shams),'sham_control_group':np.array(labels),
        'treated':np.concatenate(treated).astype(np.float32),'group':np.array(labels)}


def provenance():
    return {'diagnostic_contract_sha256':'a'*64,'contract_sha256':'b'*64,'source_manifest_sha256':'c'*64}


@pytest.mark.parametrize('arm', v.ARMS)
def test_two_source_validation_synthetic_outputs(tmp_path,arm):
    arrays=flat_cache()
    table=fixture()['table']
    out=tmp_path/arm
    summary=v.run_validation(arrays,table,arm,out,excluded_targets=('DENIED',),provenance=provenance())
    assert summary['stage']=='validation'
    assert summary['development_used_for_fit_or_selection'] is False
    assert summary['exact_ntc_identity_verified'] is True
    assert summary['exact_zero_effect_identity_verified'] is True
    assert summary['H1_treated_read'] is False
    assert summary['count_emitter'] is False
    assert len(summary['outer_folds'])==2
    assert len((out/'steps.jsonl').read_text().splitlines())==2
    assert json.loads((out/'summary.json').read_text())['arm']==arm
    for fold in summary['outer_folds']:
        assert set(fold['fit_group_indices']).isdisjoint(fold['evaluation_group_indices'])
        assert fold['held_source'] not in fold['train_sources']
        assert len(fold['ntc_sham_groups'])==1
        assert fold['aggregate']['negative_fraction']==0
        assert fold['aggregate']['ntc_sham_centroid_mse']>0
        assert fold['aggregate']['context_anchor_centroid_mse']>0
        assert fold['aggregate']['observed_shift_rms']>0
        assert 'raw_shift_cosine' in fold['aggregate']
        if arm=='control':
            assert fold['aggregate']['predicted_shift_rms']==0
            assert fold['aggregate']['raw_predicted_shift_rms']==0
            assert fold['aggregate']['shift_cosine'] is None
    with np.load(out/'predictions.npz',allow_pickle=False) as saved:
        assert np.all(saved['fold_0_nonnegative']>=0)
        assert np.all(saved['fold_1_nonnegative']>=0)
    with pytest.raises(FileExistsError):
        v.run_validation(arrays,table,arm,out,excluded_targets=('DENIED',),provenance=provenance())


def test_sham_values_never_fit_decoder(tmp_path):
    first,second=flat_cache(),flat_cache()
    second['sham_control']+=4
    for label,arrays in [('first',first),('second',second)]:
        v.run_validation(arrays,fixture()['table'],'true',tmp_path/label,excluded_targets=(),provenance=provenance())
    with np.load(tmp_path/'first/weights.npz',allow_pickle=False) as a, np.load(tmp_path/'second/weights.npz',allow_pickle=False) as b:
        assert set(a.files)==set(b.files)
        for key in a.files:
            np.testing.assert_array_equal(a[key],b[key])


@pytest.mark.parametrize('field,group_key',[('control','control_group'),('treated','group')])
def test_held_source_effects_cannot_fit_its_own_decoder(tmp_path,field,group_key):
    first,second=flat_cache(),flat_cache()
    second[field][second['sources'][second[group_key]]=='J']+=2
    for label,arrays in [('first',first),('second',second)]:
        v.run_validation(arrays,fixture()['table'],'true',tmp_path/label,excluded_targets=(),provenance=provenance())
    with np.load(tmp_path/'first/weights.npz',allow_pickle=False) as a, np.load(tmp_path/'second/weights.npz',allow_pickle=False) as b:
        # Sorted source folds: fold_0 holds J out and trains exclusively on K.
        np.testing.assert_array_equal(a['fold_0_coefficients'],b['fold_0_coefficients'])
        # The opposite fold is allowed to fit J's independent reference/effects.
        assert not np.array_equal(a['fold_1_coefficients'],b['fold_1_coefficients'])


def test_context_pool_not_subtracted_from_training_response(tmp_path,monkeypatch):
    original=v.fit_residual_decoder
    responses=[]
    def capture(*args,**kwargs):
        responses.append(np.array(args[3],copy=True))
        return original(*args,**kwargs)
    monkeypatch.setattr(v,'fit_residual_decoder',capture)
    first,second=flat_cache(),flat_cache()
    second['context_control']+=.5
    second['context'][:,:2]+=.5
    for label,arrays in [('first',first),('second',second)]:
        v.run_validation(arrays,fixture()['table'],'true',tmp_path/label,excluded_targets=(),provenance=provenance())
    np.testing.assert_array_equal(responses[0],responses[2])
    np.testing.assert_array_equal(responses[1],responses[3])


def test_unpaired_unequal_populations_have_separate_group_axes(tmp_path):
    arrays=flat_cache()
    anchors=[]
    anchor_groups=[]
    for group in range(4):
        block=arrays['control'][arrays['control_group']==group]
        anchors.append(np.concatenate([block+i/100 for i in range(4)]))
        anchor_groups.extend([group]*32)
    arrays['control']=np.concatenate(anchors)
    arrays['control_group']=np.asarray(anchor_groups)
    out=tmp_path/'unpaired'
    summary=v.run_validation(arrays,fixture()['table'],'true',out,excluded_targets=(),provenance=provenance())
    for fold in summary['outer_folds']:
        assert all(group['anchor_cells']==32 and group['treated_cells']==8 for group in fold['groups'])
        assert 'anchor_cells' not in fold['aggregate']
        assert 'treated_cells' not in fold['aggregate']
    with np.load(out/'predictions.npz',allow_pickle=False) as saved:
        for prefix in ('fold_0_','fold_1_'):
            assert len(saved[prefix+'group'])==len(saved[prefix+'control_cache_rows'])==64
            assert len(saved[prefix+'treated_group'])==len(saved[prefix+'treated_cache_rows'])==16
            np.testing.assert_array_equal(np.unique(saved[prefix+'group'],return_counts=True)[1],[32,32])
            np.testing.assert_array_equal(np.unique(saved[prefix+'treated_group'],return_counts=True)[1],[8,8])


def test_no_h1_selection_or_gain_grid_in_cli():
    assert v.policy()['hyperparameter_selection'] is False
    assert v.policy()['gain']==1
    assert v.policy()['ridge_lambda']==10


@pytest.mark.parametrize('host',['login02.anvil.rcac.purdue.edu','aida2','unapproved'])
def test_cli_fake_slurm_guard(monkeypatch,host):
    monkeypatch.setattr(v.socket,'gethostname',lambda:host)
    monkeypatch.setenv('SLURM_JOB_ID','123')
    args=['--arm','true','--output-dir','/tmp/not-created']
    for name in ('cache','target-features','feature-receipt','contract','source-manifest','diagnostic-contract'):
        args += ['--'+name,'/tmp/not-read','--'+name+'-sha256','a'*64]
    with pytest.raises(RuntimeError,match='head node'):
        v.main(args)
