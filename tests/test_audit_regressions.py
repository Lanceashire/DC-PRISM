import itertools
import json
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
import prism_engine as eng
import run_p3_p4 as runner
from validate_p4 import validate


def test_fixed_k_regression_feasible_set_previously_missed():
    q=np.array([.6682510512236002,.28751617523274503,.32273110468260996,.6802178317134333,
                .6354128939686561,.5362678482054509,.02219546465614694,.20125700955884446])
    e=np.array([1.585476923500609,.612679281946567,.9551873059375516,2.230788018497314,
                .5634333893936377,1.3998727859031108,2.5417847092199226,1.3272001361098322])
    B=2.855695454652162
    selected,risk=eng.solve_fixed_k_minrisk_subset(q,e,B,3)
    assert selected.tolist()==[1,2,4]
    assert risk <= B
    with pytest.raises(ValueError):
        eng.solve_fixed_k_minrisk_subset(q,e,0.001,3)


def test_fixed_k_small_pool_matches_independent_bruteforce():
    rng=np.random.default_rng(731)
    for _ in range(16):
        q=rng.uniform(.01,.95,9); e=rng.uniform(.1,3,9); k=4
        all_sets=list(itertools.combinations(range(9),k))
        values=[eng.bernstein_envelope(q[list(s)],e[list(s)]) for s in all_sets]
        optimum=min(values)
        selected,risk=eng.solve_fixed_k_minrisk_subset(q,e,optimum+1e-8,k)
        assert len(selected)==k and risk==pytest.approx(optimum)


def test_large_fixed_k_anchor_is_feasible_and_no_false_infeasibility():
    q=np.linspace(.01,.6,24); e=np.ones(24); anchor=np.arange(7)
    B=eng.bernstein_envelope(q[anchor],e[anchor])
    selected,risk=eng.solve_fixed_k_minrisk_subset(q,e,B,5,feasible_anchor=anchor)
    assert len(selected)==5 and risk<=B


def fake_credit():
    rng=np.random.default_rng(777)
    X=rng.normal(size=(3200,6)); e=rng.uniform(100,10000,3200)
    y=rng.binomial(1,1/(1+np.exp(-(.5*X[:,0]-.3*X[:,1]-1))))
    return X,y,e


def test_state_builder_hidden_y_and_test_X_do_not_change_candidates():
    X,y,e=fake_credit()
    a=runner.build_strictly_hidden_y_free_real_state(X,y,e,4001)
    hidden_y=1-y; hidden_y[a['D0_ids']]=y[a['D0_ids']]
    altered_X=X.copy();altered_X[a['T_ids']]=1000
    b=runner.build_strictly_hidden_y_free_real_state(altered_X,hidden_y,e,4001)
    assert a['preprocessing_hash']==b['preprocessing_hash']
    for ra,rb in zip(a['rounds'],b['rounds']):
        np.testing.assert_array_equal(ra['C_ids'],rb['C_ids'])
        np.testing.assert_array_equal(ra['O_ids'],rb['O_ids'])


def test_actual_dynamic_runner_first_round_and_delayed_release(tmp_path,monkeypatch):
    X,y,e=fake_credit()
    for sub in ['p4_real_labels','results/audit']:(tmp_path/sub).mkdir(parents=True)
    monkeypatch.setattr(runner,'PKG_ROOT',str(tmp_path))
    monkeypatch.setattr(runner,'AUDIT_DIR',str(tmp_path/'results/audit'))
    df,*_=runner.run_p4_real_labels_benchmark({'SyntheticFixture':{'X':X,'y':y,'ead':e,'role':'TEST_ONLY'}},seeds=[4001])
    assert np.ptp(df.loc[df['round']==1,'brier_t']) < 1e-12
    result=validate(tmp_path/'p4_real_labels/p4_trajectories_10rounds.csv')
    assert result['rows']==100
    # The validator must reject an otherwise plausible premature-label metric.
    bad=df.copy();bad.loc[bad.index[0],'brier_t']+=.01
    bad.to_csv(tmp_path/'bad.csv',index=False)
    with pytest.raises(AssertionError,match='Unreleased'):
        validate(tmp_path/'bad.csv')


def test_fixed_fraction_ceil_and_empty_selection():
    rng=np.random.default_rng(13)
    X=rng.normal(size=(60,3));y=rng.integers(0,2,60)
    learner=eng.UnifiedLogisticLearner().fit(X,y)
    C=rng.normal(size=(9,3));q=np.linspace(.02,.3,9);e=np.ones(9)
    B=eng.bernstein_envelope(q[:5],e[:5])+.001
    for eps in [.25,1.0]:
        S,M,_=eng.select_acquisition_policy('Fixed-Eps-Safe-PA',X,y,X[:5],C,e,X[10:30],B,
                                           learner=learner,q_C_precomputed=q,fixed_eps=eps)
        assert len(S)==int(np.ceil((1-eps)*len(M)))
