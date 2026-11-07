"""Check that the coverage intervention changes only the intended history."""
import importlib.util
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('coverage', ROOT/'experiments/supplement_v2/run.py')
run = importlib.util.module_from_spec(spec); spec.loader.exec_module(run)


def fake_data():
    rng = np.random.default_rng(41)
    return dict(X=rng.normal(size=(1500, 6)), ead=rng.uniform(.2, 3, 1500))


def test_history_intervention_preserves_all_shared_roles_and_truth():
    data = fake_data(); before = data['X'].copy()
    arms, meta = run.make_pair(data, 98001, 'nonlinear')
    broad, gap = arms['broad'], arms['gap']
    for key in broad:
        if key not in ['X_D', 'y_D']:
            np.testing.assert_array_equal(broad[key], gap[key])
    assert broad['X_D'].shape == gap['X_D'].shape == (80, 6)
    assert meta['gap_history_axis_mean'] < meta['broad_history_axis_mean']
    np.testing.assert_array_equal(data['X'], before)
    logits = -.8+broad['X_C']@np.array([.45,.65,-.5,.35,.2,-.15])
    logits += .5*np.sin(broad['X_C'][:,1])+.35*broad['X_C'][:,2]*broad['X_C'][:,3]
    np.testing.assert_allclose(run.expit(logits), broad['p_C_true'])


def test_visible_selection_is_risk_feasible_and_ignores_hidden_changes():
    arms, _ = run.make_pair(fake_data(), 98002, 'logistic')
    broad = arms['broad']
    ctx = run.reference_context(*[broad[k] for k in ['X_D','y_D','X_C','ead_C','X_Q']], 98002)
    frozen_hash = run.sup.digest([s.tolist() for s in ctx['family']])
    for arm, st in arms.items():
        args = [st[k] for k in ['X_D','y_D','X_O','X_C','X_Q']]
        a = run.select(*args, ctx, 98002)
        for key in ['y_O','y_C','y_T','p_C_true','p_T_true']:
            st[key] = 1-st[key]
        b = run.select(*[st[k] for k in ['X_D','y_D','X_O','X_C','X_Q']], ctx, 98002)
        assert set(a[0]) == set(run.POLICIES)
        for policy, s in a[0].items():
            np.testing.assert_array_equal(s, b[0][policy])
            assert run.eng.bernstein_envelope(ctx['q'][s], st['ead_C'][s]) <= ctx['B']+1e-10
            assert tuple(s) in {tuple(x) for x in ctx['family']}
        assert len(a[0]['Fixed-ceil']) == int(np.ceil(.75*len(ctx['M'])))
    assert run.sup.digest([s.tolist() for s in ctx['family']]) == frozen_hash


def test_pair_evaluation_has_shared_decision_problem_and_complete_family():
    rows, family = run.one_pair(fake_data(), 98003, 'nonlinear', 'engineering')
    assert len(rows) == 12
    for key in ['B','q_hash','budget_hash','family_hash','shared_hidden_hash','shared_covariate_hash']:
        assert len({r[key] for r in rows}) == 1
    assert min(r['expected_regret'] for r in rows) >= -1e-12
    assert {r['arm'] for r in family} == {'broad','gap'}
    for r in rows:
        f = next(f for f in family if f['arm'] == r['arm'] and f['family_index'] == r['family_index'])
        assert f['expected_brier'] == r['expected_brier']
