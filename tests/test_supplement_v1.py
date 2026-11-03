import importlib.util
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('supplement',ROOT/'experiments/supplement_v1/run.py')
sup=importlib.util.module_from_spec(spec);spec.loader.exec_module(sup)


def test_repaired_fixed_round_reproduces_integer_target_and_risk():
    original,generate=sup.original_modules();st=generate(2001)
    learner=original.UnifiedLogisticLearner(l2=.45).fit(st['X_D'],st['y_D'])
    q,_=original.compute_calibrated_upper_risk(learner,st['X_D'],st['y_D'],st['X_C'],rng=np.random.default_rng(2101))
    B,_=original.compute_exogenous_budget(q,st['ead_C'],target_k=6)
    S,target,fallback=sup.fixed_repair(original,st,learner,q,B)
    assert target==4 and len(S)==4
    assert original.bernstein_envelope(q[S],st['ead_C'][S])<=B


def test_factorial_has_shared_family_and_matches_reference_scores():
    _,generate=sup.original_modules();st=generate(90001)
    args=[st[k] for k in ['X_D','y_D','X_O','X_C','ead_C','X_Q']]
    choices,learner,q,B,_,family,_,scores,_,cert=sup.common_selection(*args,90001)
    assert len(choices)==21
    worlds=sup.eng.sample_common_parameter_worlds(learner,n_worlds=20,rng=np.random.default_rng(90701))
    M=choices['Safe-MinRisk']
    for i,S in enumerate(family):
        ref=sup.eng.evaluate_common_worlds_surplus(S,M,st['X_D'],st['X_O'],st['X_C'],st['X_Q'],learner,worlds)
        assert scores['P1Q1L1G1'][i]==pytest.approx(ref['LCB_surplus'],abs=1e-12)
    for name,S in choices.items():
        assert sup.eng.bernstein_envelope(q[S],st['ead_C'][S])<=B+1e-10
        if name.startswith('Fixed-'):
            rule=np.round if name=='Fixed-round' else np.ceil
            assert len(S)==max(int(rule(.75*len(M))),1)


def test_pending_free_family_is_invariant_when_O_changes():
    _,generate=sup.original_modules();st=generate(90002)
    args=[st[k] for k in ['X_D','y_D','X_O','X_C','ead_C','X_Q']]
    a=sup.common_selection(*args,90002)
    args[2]=np.full_like(st['X_O'],17.)
    b=sup.common_selection(*args,90002)
    assert a[6]==b[6]
    for name,S in a[0].items():
        if name.startswith('P0'):
            np.testing.assert_array_equal(S,b[0][name])


def test_bridge_generated_truth_is_consistent_and_input_unchanged():
    rng=np.random.default_rng(82)
    data={'X':rng.normal(size=(1200,6)),'y':rng.integers(0,2,1200),'ead':rng.uniform(1,3,1200)}
    old=data['X'].copy();a=sup.realx_state(data,90003,'logistic')
    other=dict(data,y=1-data['y']);b=sup.realx_state(other,90003,'logistic')
    np.testing.assert_array_equal(old,data['X'])
    for key in a:np.testing.assert_array_equal(a[key],b[key])
    p=sup.expit(-.8+a['X_C']@np.array([.45,.65,-.5,.35,.2,-.15]))
    np.testing.assert_allclose(p,a['p_C_true'])
