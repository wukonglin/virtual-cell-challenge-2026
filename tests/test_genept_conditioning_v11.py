"""Synthetic-only pure feature tests: no archive, source expression, model or I/O."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import genept_conditioning_v11 as g


def table():
    return g.prepare_features(['F1','F2','F3','H1','H2','HM','OFF'],
        np.array([[1,2],[3,4],[0,0],[5,8],[9,10],[0,0],[100,200]],np.float32),
        np.array([1,1,0,1,1,0,1],np.uint8),
        source_vector_object_id=np.array([0,1,-1,2,3,-1,4],np.int32),expected_dimension=2)


def fitted(raw=None):
    return g.fit_transform(table() if raw is None else raw,source_id='source_A',
        fit_targets=['F1','F2','F3'],held_targets=['H1','H2','HM'],official_targets=['OFF'])


def source_features(raw=None):
    raw = table() if raw is None else raw
    return g.transform_features(raw,fitted(raw),target_ids=['F1','F2','F3','H1','H2','HM'])


def test_mean_and_one_scalar_rms_available_fit_rows_only():
    stats = fitted()
    assert stats.mean == (2.,3.) and stats.scalar_rms == 1.
    record = stats.to_dict()
    assert record['fit_targets'] == ['F1','F2','F3']
    assert record['available_fit_targets'] == ['F1','F2']
    assert record['available_fit_rows'] == 2 and record['rms_coordinate_count'] == 4
    features = source_features()
    np.testing.assert_array_equal(features.embeddings[:2],[[-1,-1],[1,1]])
    assert np.mean(features.embeddings[:2].astype(float)**2) == 1.
    assert not record['responses_used'] and not record['model_training_performed']


def test_rms_is_scalar_not_coordinatewise_standard_deviation():
    raw = g.prepare_features(['F1','F2'],np.array([[1,2],[3,8]],np.float64),[1,1],expected_dimension=2)
    stats = g.fit_transform(raw,source_id='s',fit_targets=raw.target_ids)
    assert stats.mean == (2.,5.)
    assert stats.scalar_rms == np.sqrt(5.)
    transformed = g.transform_features(raw,stats)
    np.testing.assert_allclose(transformed.embeddings[0],[-1/np.sqrt(5),-3/np.sqrt(5)])


def test_held_and_official_feature_values_never_influence_stats():
    original = table()
    values = original.embeddings.copy()
    values[[3,4,6]] = [[-9e20,7e20],[13,-19],[2e30,3e30]]
    changed = g.prepare_features(original.target_ids,values,original.present,expected_dimension=2)
    assert fitted(original).to_dict() == fitted(changed).to_dict()


def test_fitting_hash_independent_of_table_order_and_object_numbering():
    raw = table()
    order = list(range(len(raw.target_ids)))[::-1]
    renamed = raw.source_vector_object_id.copy()
    renamed[renamed >= 0] += 100
    reordered = g.prepare_features([raw.target_ids[i] for i in order],raw.embeddings[order],raw.present[order],
        source_vector_object_id=renamed[order],expected_dimension=2)
    assert fitted(raw).to_dict() == fitted(reordered).to_dict()


def test_transform_json_roundtrip_and_new_names_do_not_refit():
    stats = fitted()
    record = stats.to_dict()
    restored = g.GenePTTransform.from_dict(json.loads(json.dumps(record)))
    assert restored.to_dict() == record
    new = g.prepare_features(['NEW','MISSING'],np.array([[12,23],[0,0]],np.float32),[1,0],expected_dimension=2)
    transformed = g.transform_features(new,restored)
    np.testing.assert_array_equal(transformed.embeddings,[[10,20],[0,0]])
    assert transformed.present.tolist() == [1,0]
    assert stats.to_dict() == record and restored.to_dict() == record


def test_missing_rows_zero_after_centering_not_negative_mean():
    output = source_features()
    np.testing.assert_array_equal(output.embeddings[[2,5]],np.zeros((2,2)))
    assert output.present.tolist() == [1,1,0,1,1,0]
    arms = g.build_arms(output,fit_targets=['F1','F2','F3'],held_targets=['H1','H2','HM'],seed=7)
    for arm in arms.values():
        assert not arm.report['availability_is_null_perturbation_flag']
        np.testing.assert_array_equal(arm.present,output.present)
        np.testing.assert_array_equal(arm.embeddings[[2,5]],np.zeros((2,2)))


def test_input_arrays_not_modified_and_return_arrays_readonly():
    names = ['a','b']
    values = np.array([[1,2],[3,4]],np.float32)
    availability = np.ones(2,np.uint8)
    references = np.array([1,2],np.int32)
    raw = g.prepare_features(names,values,availability,source_vector_object_id=references,expected_dimension=2)
    values[:] = 99; availability[:] = 0; references[:] = 5
    np.testing.assert_array_equal(raw.embeddings,[[1,2],[3,4]])
    np.testing.assert_array_equal(raw.present,[1,1])
    for array in (raw.embeddings,raw.present,raw.source_vector_object_id):
        assert not array.flags.writeable
    arms = g.build_arms(source_features(),fit_targets=['F1','F2','F3'],held_targets=['H1','H2','HM'],seed=3)
    for arm in arms.values():
        assert not arm.embeddings.flags.writeable and not arm.present.flags.writeable


def test_production_dimension1536_preserved():
    values = np.vstack((np.ones(1536),np.full(1536,3))).astype(np.float32)
    raw = g.prepare_features(['a','b'],values,[1,1])
    stats = g.fit_transform(raw,source_id='s',fit_targets=['a','b'])
    output = g.transform_features(raw,stats)
    assert output.embeddings.shape == (2,1536)
    np.testing.assert_array_equal(output.embeddings,np.vstack((-np.ones(1536),np.ones(1536))))


def test_numpy_unicode_names_accepted_without_case_or_alias_changes():
    raw = g.prepare_features(np.asarray(['GeneA','GENEA']),np.array([[1,2],[3,4]],np.float32),[1,1],expected_dimension=2)
    assert raw.target_ids == ('GeneA','GENEA') and all(type(name) is str for name in raw.target_ids)


@pytest.mark.parametrize('names',[['a','a'],['a',' '],['a',' b'],['a','a\x00'],[],None,'ab'])
def test_bad_names_rejected(names):
    with pytest.raises(ValueError):
        g.prepare_features(names,np.ones((2,2),np.float32),[1,1],expected_dimension=2)


@pytest.mark.parametrize('values',[np.ones((3,2),np.float32),np.ones((2,3),np.float32),
    np.ones(2,np.float32),np.ones((2,2),np.int32),np.ones((2,2),np.float16),
    np.full((2,2),np.nan),np.full((2,2),np.inf),np.full((2,2),1e100)])
def test_bad_numeric_matrices_rejected(values):
    with pytest.raises(ValueError):
        g.prepare_features(['a','b'],values,[1,1],expected_dimension=2)


@pytest.mark.parametrize('present',[[1],[[1,1]],[1,2],[-1,1],[1.,1.],['1','1']])
def test_bad_availability_rejected(present):
    with pytest.raises(ValueError):
        g.prepare_features(['a','b'],np.ones((2,2),np.float32),present,expected_dimension=2)


def test_missing_nonzero_and_available_zero_rejected():
    with pytest.raises(ValueError,match='Unavailable'):
        g.prepare_features(['a'],np.ones((1,2),np.float32),[0],expected_dimension=2)
    with pytest.raises(ValueError,match='Available'):
        g.prepare_features(['a'],np.zeros((1,2),np.float32),[1],expected_dimension=2)


@pytest.mark.parametrize('refs',[[0],[0,-2],[0,2**32],[0.,1.],[True,False]])
def test_bad_source_object_ids_rejected(refs):
    with pytest.raises(ValueError):
        g.prepare_features(['a','b'],np.ones((2,2),np.float32),[1,1],source_vector_object_id=refs,expected_dimension=2)


def test_source_object_conflicting_values_rejected():
    with pytest.raises(ValueError,match='conflicting'):
        g.prepare_features(['a','b'],np.array([[1,2],[3,4]],np.float32),[1,1],
            source_vector_object_id=[9,9],expected_dimension=2)


@pytest.mark.parametrize('fit,held,official',[(['F1','H1'],['H1'],[]),(['F1','OFF'],[],['OFF']),
    (['F1','unknown'],[],[]),(['F1','F1'],[],[]),([],[],[])])
def test_role_overlap_and_bad_fitting_rosters_rejected(fit,held,official):
    with pytest.raises(ValueError):
        g.fit_transform(table(),source_id='s',fit_targets=fit,held_targets=held,official_targets=official)


def test_zero_scale_and_too_few_available_fits_rejected():
    raw = g.prepare_features(['a','b'],np.ones((2,2),np.float32),[1,1],expected_dimension=2)
    with pytest.raises(ValueError,match='scale'):
        g.fit_transform(raw,source_id='s',fit_targets=['a','b'])
    with pytest.raises(ValueError,match='two available'):
        g.fit_transform(table(),source_id='s',fit_targets=['F1','F3'])


@pytest.mark.parametrize('key,value',[('scalar_rms',0),('scalar_rms',float('nan')),('scalar_rms',-1),
    ('mean',[1,float('inf')]),('mean',[1]),('dimension',False),('responses_used',True),
    ('model_training_performed',True),('fit_targets_sha256','a'*64),('available_fit_rows',99),
    ('fit_feature_sha256','bad'),('fitting_scope','all targets')])
def test_transform_replay_rejects_corrupt_records(key,value):
    record = fitted().to_dict()
    record[key] = value
    with pytest.raises(ValueError):
        g.GenePTTransform.from_dict(record)


def test_extra_transform_record_fields_rejected():
    record = fitted().to_dict(); record['response_model'] = 'not allowed'
    with pytest.raises(ValueError,match='Exact'):
        g.GenePTTransform.from_dict(record)


def test_mismatched_dimension_unknown_names_and_overflow_rejected():
    raw = g.prepare_features(['a','b'],np.ones((2,3),np.float32),[1,1],expected_dimension=3)
    with pytest.raises(ValueError,match='dimension'):
        g.transform_features(raw,fitted())
    with pytest.raises(ValueError,match='Unknown'):
        g.transform_features(table(),fitted(),target_ids=['unknown'])
    tiny = replace(fitted(),scalar_rms=1e-100)
    with pytest.raises(ValueError,match='float32 range'):
        g.transform_features(table(),tiny)


def test_deterministic_derangement_within_roles_and_availability_and_singletons():
    features = source_features()
    kwargs = dict(fit_targets=['F1','F2','F3'],held_targets=['H1','H2','HM'],seed=123)
    first,second = g.build_arms(features,**kwargs),g.build_arms(features,**kwargs)
    assert first.keys() == {'correct','masked','shuffled'}
    for name in first:
        np.testing.assert_array_equal(first[name].embeddings,second[name].embeddings)
        np.testing.assert_array_equal(first[name].present,features.present)
        assert first[name].report == second[name].report
    assert first['shuffled'].donor_target_ids == ('F2','F1','F3','H2','H1','HM')
    assert first['shuffled'].report['unchanged_singletons'] == [
        {'role':'fit','present':0,'target':'F3'},{'role':'held','present':0,'target':'HM'}]
    np.testing.assert_array_equal(first['correct'].embeddings,features.embeddings)
    assert not first['masked'].embeddings.any()
    assert first['masked'].present.tolist() == features.present.tolist()


def test_deranges_missing_multiple_members_without_inventing_signal():
    features = source_features()
    names = features.target_ids+('HM2',)
    expanded = replace(features,target_ids=names,embeddings=np.vstack((features.embeddings,np.zeros((1,2),np.float32))),
        present=np.r_[features.present,np.uint8(0)])
    arm = g.build_arms(expanded,fit_targets=['F1','F2','F3'],held_targets=['H1','H2','HM','HM2'],seed=7)['shuffled']
    assert arm.donor_target_ids[-2:] == ('HM2','HM')
    assert not arm.embeddings[-2:].any()


def test_local_rng_does_not_advance_numpy_global_rng():
    np.random.seed(27)
    before = np.random.get_state()
    g.build_arms(source_features(),fit_targets=['F1','F2','F3'],held_targets=['H1','H2','HM'],seed=8)
    after = np.random.get_state()
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1],after[1])


def test_held_roster_cannot_change_fit_shuffle():
    ids = tuple(f'F{i}' for i in range(6))+tuple(f'H{i}' for i in range(4))
    features = g.TransformedGenePT(ids,np.arange(20,dtype=np.float32).reshape(10,2),np.ones(10,np.uint8),'s','a'*64,ids[:6])
    a = g.build_arms(features,fit_targets=ids[:6],held_targets=ids[6:],seed=44)['shuffled']
    short = replace(features,target_ids=ids[:8],embeddings=features.embeddings[:8],present=features.present[:8])
    b = g.build_arms(short,fit_targets=ids[:6],held_targets=ids[6:8],seed=44)['shuffled']
    assert a.donor_target_ids[:6] == b.donor_target_ids[:6]
    assert all(name != donor for name,donor in zip(a.target_ids,a.donor_target_ids))


@pytest.mark.parametrize('seed',[-1,2**64,True,1.5])
def test_bad_shuffle_seed_rejected(seed):
    with pytest.raises(ValueError):
        g.build_arms(source_features(),fit_targets=['F1','F2','F3'],held_targets=['H1','H2','HM'],seed=seed)


def test_arm_roles_must_partition_and_not_overlap():
    for fit,held in [(['F1','F2','F3'],['F1','H1','H2','HM']),(['F1','F2'],['H1','H2','HM'])]:
        with pytest.raises(ValueError,match='partition'):
            g.build_arms(source_features(),fit_targets=fit,held_targets=held,seed=3)


def test_disjoint_arm_relabeling_cannot_move_statistic_fit_targets_to_held():
    with pytest.raises(ValueError,match='statistic-fitting'):
        g.build_arms(source_features(),fit_targets=['F1','H1','F3'],held_targets=['F2','H2','HM'],seed=3)


def duplicate_table():
    return g.prepare_features(['Fdup','Hdup','OTHERdup','F1','F2','M1','M2','OFF'],
        np.array([[1,4],[1,4],[1,4],[3,2],[7,6],[0,0],[0,0],[12,14]],np.float32),
        [1,1,1,1,1,0,0,1],source_vector_object_id=[8,8,9,10,11,-1,-1,12],expected_dimension=2)


def test_cross_role_sharing_distinguishes_content_and_objects_without_alias_claim():
    raw = duplicate_table()
    report = g.audit_sharing(raw,{'fit':['Fdup','F1','F2','M1'],'held':['Hdup','OTHERdup','M2'],'official':['OFF']})
    assert report['requires_training_admission_review']
    assert report['exact_numeric_content_sharing'][0]['target_names'] == ['Fdup','Hdup','OTHERdup']
    assert report['source_object_sharing'][0]['target_names'] == ['Fdup','Hdup']
    assert not report['biological_alias_equivalence_established'] and not report['roles_changed']
    assert not report['labels_changed'] and not report['training_authorized']
    assert all('M1' not in group['target_names'] and 'M2' not in group['target_names']
        for group in report['exact_numeric_content_sharing'])


def test_identity_overlap_is_explicit_audit_not_silent_role_rewrite():
    report = g.audit_sharing(table(),{'fit':['F1','F2'],'held':['F1','H1']})
    assert report['identity_overlap'] == [{'target':'F1','roles':['fit','held']}]
    assert report['requires_training_admission_review'] and not report['roles_changed']


def test_no_sharing_and_unknown_objects_not_grouped():
    raw = table()
    report = g.audit_sharing(raw,{'fit':['F1','F2','F3'],'held':['H1','H2','HM']})
    assert not report['requires_training_admission_review'] and not report['warnings']


def test_quarantine_all_duplicate_members_nonmutating_with_hashed_audit():
    raw = duplicate_table()
    original = (raw.embeddings.copy(),raw.present.copy(),raw.source_vector_object_id.copy())
    output,report = g.quarantine_duplicate_features(raw,raw_artifact_sha256='b'*64)
    assert output.target_ids == raw.target_ids
    assert report['quarantined_target_names'] == ['Fdup','Hdup','OTHERdup']
    assert report['quarantined_count'] == 3 and report['available_before'] == 6 and report['available_after'] == 3
    np.testing.assert_array_equal(output.present,[0,0,0,1,1,0,0,1])
    assert not output.embeddings[:3].any() and output.source_vector_object_id[:3].tolist() == [-1,-1,-1]
    for before,after in zip(original,(raw.embeddings,raw.present,raw.source_vector_object_id)):
        np.testing.assert_array_equal(before,after)
    for key in ('original_table_modified','labels_changed','GO_features_modified','response_roles_changed',
                'biological_alias_equivalence_established','model_training_performed'):
        assert report[key] is False
    assert report['input_table_sha256'] != report['output_table_sha256']
    assert report['caller_supplied_raw_artifact_sha256'] == 'b'*64
    assert not report['artifact_hash_authenticated_by_this_module']
    assert len(report['duplicate_groups'][0]['canonical_binary64_sha256']) == 64


def test_quarantined_fit_rows_ignored_and_masks_preserved_through_arms():
    raw = duplicate_table()
    output,_ = g.quarantine_duplicate_features(raw)
    stats = g.fit_transform(output,source_id='s',fit_targets=['Fdup','F1','F2','M1'],
        held_targets=['Hdup','OTHERdup','M2'],official_targets=['OFF'])
    assert stats.available_fit_targets == ('F1','F2') and stats.mean == (5.,4.) and stats.scalar_rms == 2.
    ids = ['Fdup','F1','F2','M1','Hdup','OTHERdup','M2']
    values = g.transform_features(output,stats,target_ids=ids)
    arms = g.build_arms(values,fit_targets=ids[:4],held_targets=ids[4:],seed=9)
    for arm in arms.values():
        np.testing.assert_array_equal(arm.present,[0,1,1,0,0,0,0])
        assert not arm.embeddings[arm.present == 0].any()


def test_quarantine_is_idempotent_and_does_not_group_missing_zeros():
    first,_ = g.quarantine_duplicate_features(duplicate_table())
    second,report = g.quarantine_duplicate_features(first)
    np.testing.assert_array_equal(first.embeddings,second.embeddings)
    np.testing.assert_array_equal(first.present,second.present)
    assert report['quarantined_count'] == 0 and not report['duplicate_groups']
    assert report['input_table_sha256'] == report['output_table_sha256']


def test_signed_zero_duplicates_quarantined_without_approximate_matching():
    raw = g.prepare_features(['a','b','c'],np.array([[0.,1.],[-0.,1.],[0.,1.+2**-20]],np.float64),[1,1,1],expected_dimension=2)
    output,report = g.quarantine_duplicate_features(raw)
    assert report['quarantined_target_names'] == ['a','b'] and output.present.tolist() == [0,0,1]


def test_quarantine_invalid_artifact_hash_rejected():
    with pytest.raises(ValueError,match='SHA256'):
        g.quarantine_duplicate_features(table(),raw_artifact_sha256='invalid')
