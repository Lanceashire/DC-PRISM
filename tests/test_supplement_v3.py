import importlib.util
import inspect
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'experiments/supplement_v3'))
import prism_engine as eng
import baselines as bl
import run_offline as ro


def _state(seed=99101):
    return ro.controlled_state(seed)['only']


def test_selectors_have_no_hidden_inputs():
    hidden = {'y_C', 'y_O', 'y_T', 'p_C_true', 'p_O_true', 'p_T_true', 'X_T'}
    for fn in [bl.random_feasible, bl.uncertainty_greedy, bl.bait_greedy, bl.targetvar_greedy,
               bl.package_family_scores, bl.choose_from_scores]:
        assert not hidden & set(inspect.signature(fn).parameters)


def test_new_baselines_feasible_and_label_invariant():
    st = _state()
    ctx = ro.sup2.reference_context(*[st[k] for k in ['X_D', 'y_D', 'X_C', 'ead_C', 'X_Q']], 99101)
    choices, sens, _ = ro.select_all(st, ctx, 99101)
    flipped = dict(st, y_C=1-st['y_C'], y_O=1-st['y_O'], y_T=1-st['y_T'], p_C_true=st['p_C_true'][::-1])
    choices2, sens2, _ = ro.select_all(flipped, ctx, 99101)
    for name, s in list(choices.items())+list(sens.items()):
        assert eng.bernstein_envelope(ctx['q'][s], st['ead_C'][s]) <= ctx['B']+1e-10
    assert all(np.array_equal(choices[k], choices2[k]) for k in choices)
    assert all(np.array_equal(sens[k], sens2[k]) for k in sens)
    assert np.array_equal(sens['default'], choices['DC-PRISM'])


def test_decomposition_ordering_and_crossfit():
    rng = np.random.default_rng(0)
    L = 0.2 + rng.normal(0, 0.01, size=(128, 6)); L[:, 3] -= 0.004
    post, plug, cf = ro.decomposition(L, anchor=0)
    assert post >= plug >= 0
    assert cf <= plug + 1e-12
    assert abs(cf-0.004) < 0.004


def test_greedy_stops_at_budget():
    st = _state(99102)
    learner = eng.UnifiedLogisticLearner(l2=.45).fit(st['X_D'], st['y_D'])
    q = np.full(len(st['X_C']), .1); B = 0.6
    s = bl.targetvar_greedy(learner, st['X_D'], st['X_O'], st['X_C'], st['X_Q'], q, st['ead_C'], B)
    assert eng.bernstein_envelope(q[s], st['ead_C'][s]) <= B+1e-12
