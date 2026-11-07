"""supplement-v3 offline studies A-C on fresh controlled and Real-X states.

A. Pre-outcome opportunity. In every state the true outcome probabilities are
   known, so hidden training labels (pending O and candidates C) are resampled
   R times from them. For a finite visible-information family F containing the
   anchor M_K this gives Monte Carlo estimates of
     H_post = E[L(M,Y) - min_S L(S,Y)]          (post-outcome headroom)
     H_pre  = E L(M,Y) - min_S E L(S,Y)         (pre-outcome headroom)
   H_pre is reported twice: the plug-in (upward biased by selection on noise)
   and a two-fold cross-fitted value (choose S* on one half of the resamples,
   evaluate it on the other; downward biased). foresight_lb = H_post - H_pre_plugin
   is therefore a conservative estimate of the foresight term.
B. Baselines. All policies, including four new ones, are scored on the same
   ex-ante endpoint E_Y[L_p(S;Y)] (test-label randomness integrated exactly,
   training-label randomness by the shared resamples).
C. One-at-a-time sensitivity of the package DC-PRISM constants.

Selectors only receive visible inputs. Every selection, including all
sensitivity variants, is locked before the evaluator draws any label.
"""
import argparse
import hashlib
import importlib.util
import json
from multiprocessing import Pool
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'experiments/supplement_v3'))
import prism_engine as eng
import baselines as bl


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT/rel)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


sup1 = _load('supplement_v1', 'experiments/supplement_v1/run.py')
sup2 = _load('supplement_v2', 'experiments/supplement_v2/run.py')
SOURCE_PATHS = ['experiments/supplement_v3/run_offline.py', 'experiments/supplement_v3/baselines.py',
                'experiments/supplement_v3/stats.py',
                'experiments/supplement_v1/run.py', 'experiments/supplement_v2/run.py', 'src/prism_engine.py',
                'archive/submitted_v2_2/src/run_p0_p1_p2.py']
W_TRUE = np.array([-0.95, 0.85, 1.75, -2.10, 2.20, 0.0, 0.0])
MAIN_POLICIES = ['Safe-MinRisk', 'DC-PRISM', 'Fisher-cost', 'Full', 'Fixed-ceil', 'PointValue',
                 'NoPending', 'Random-Feasible', 'Uncertainty', 'BAIT-Greedy', 'TargetVar-Greedy']
SENSITIVITY = {
    'default': {},
    'quantile=0.1': {'quantile': .1}, 'quantile=0.3': {'quantile': .3}, 'quantile=0.5': {'quantile': .5},
    'penalty=0': {'penalty': 0.}, 'penalty=0.11': {'penalty': .11}, 'penalty=0.44': {'penalty': .44},
    'gate=0': {'gate': 0.}, 'gate=0.04': {'gate': .04}, 'gate=0.16': {'gate': .16},
    'damping=0': {'damping': 0.}, 'damping=0.9': {'damping': .9},
    'scale=1.25': {'scale': 1.25}, 'scale=5': {'scale': 5.},
}
_STATE = {}


def _controlled_generators():
    path = ROOT/'archive/submitted_v2_2/src/run_p0_p1_p2.py'
    import ast
    tree = ast.parse(path.read_text())
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {'true_dgp_prob', 'generate_synthetic_credit_state'}]
    scope = {'np': np, 'expit': expit}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), str(path), 'exec'), scope)
    return scope['generate_synthetic_credit_state'], scope['true_dgp_prob']


def controlled_state(seed):
    generate, truth = _controlled_generators()
    st = generate(seed)
    assert np.array_equal(truth(st['X_C'], W_TRUE), st['p_C_true'])
    st['p_O_true'] = truth(st['X_O'], W_TRUE)
    return {'only': st}


def realx_pair(data, seed, mechanism):
    """supplement-v2 pair construction, additionally exposing true p for O."""
    arms, _ = sup2.make_pair(data, seed, mechanism)
    rng = np.random.default_rng(seed)
    ids = rng.permutation(len(data['X']))[:706]
    raw, eraw = data['X'][ids], data['ead'][ids]
    prep = eng.SplitBeforePreprocessor(tail_scale=1).fit(raw[:80], eraw[:80]); prep.tau_aux = np.zeros(2)
    X = prep.transform_X(raw)
    logits = -.8 + X @ np.array([.45, .65, -.5, .35, .2, -.15])
    if mechanism == 'nonlinear':
        logits += .5*np.sin(X[:, 1]) + .35*X[:, 2]*X[:, 3]
    p = expit(logits)
    for st in arms.values():
        assert np.array_equal(st['X_O'], X[240:262]) and np.array_equal(st['p_C_true'], p[262:286])
        st['p_O_true'] = p[240:262]
    return arms


def select_all(st, ctx, seed):
    """Visible inputs only. Returns main choices and sensitivity choices."""
    X_D, y_D, X_O, X_C, ead, X_Q = (st[k] for k in ['X_D', 'y_D', 'X_O', 'X_C', 'ead_C', 'X_Q'])
    q, B, M = ctx['q'], ctx['B'], ctx['M']
    common, learner, _, _ = sup2.select(X_D, y_D, X_O, X_C, X_Q, ctx, seed)
    worlds = eng.sample_common_parameter_worlds(learner, n_worlds=20, rng=np.random.default_rng(seed+700))
    choices = {'Safe-MinRisk': M.copy(), 'Full': common['Full'], 'Fixed-ceil': common['Fixed-ceil'],
               'PointValue': common['PointValue'], 'NoPending': common['NoPending']}
    for name, policy in [('DC-PRISM', 'DC-PRISM'), ('Fisher-cost', 'BAIT-Cost-Safe')]:
        s, m_k, _ = eng.select_acquisition_policy(policy, X_D, y_D, X_O, X_C, ead, X_Q, B, learner=learner,
                        q_C_precomputed=q, w_worlds_precomputed=worlds, rng=np.random.default_rng(seed+400))
        assert np.array_equal(np.sort(m_k), np.sort(M))
        choices[name] = np.sort(np.asarray(s, int))
    choices['Random-Feasible'] = bl.random_feasible(X_C, q, ead, B, np.random.default_rng(seed+900))
    choices['Uncertainty'] = bl.uncertainty_greedy(learner, X_C, q, ead, B)
    choices['BAIT-Greedy'] = bl.bait_greedy(learner, X_D, X_O, X_C, X_Q, q, ead, B)
    choices['TargetVar-Greedy'] = bl.targetvar_greedy(learner, X_D, X_O, X_C, X_Q, q, ead, B)
    sens, cache = {}, {}
    for name, cfg in SENSITIVITY.items():
        key = (cfg.get('damping', .45), cfg.get('scale', 2.5))
        if key not in cache:
            cache[key] = bl.package_family_scores(learner, X_D, X_O, X_C, X_Q, q, ead, B, M, worlds,
                                                  damping=key[0], scale=key[1])
        family, deltas = cache[key]
        s, _ = bl.choose_from_scores(family, deltas, M, quantile=cfg.get('quantile', .2),
                                     penalty=cfg.get('penalty', .22), gate=cfg.get('gate', .08))
        sens[name] = np.sort(np.asarray(s, int))
    assert np.array_equal(sens['default'], choices['DC-PRISM']), 'Sensitivity re-implementation must reproduce package DC-PRISM'
    for s in list(choices.values()) + list(sens.values()):
        assert len(set(s.tolist())) == len(s) and eng.bernstein_envelope(q[s], ead[s]) <= B + 1e-10
    return choices, sens, learner


def evaluate_sets(st, sets, learner, R, rng):
    """Post-lock evaluator. Returns (R x n_sets) expected-Brier matrix and realized values."""
    Y_O = rng.binomial(1, st['p_O_true'], size=(R, len(st['p_O_true'])))
    Y_C = rng.binomial(1, st['p_C_true'], size=(R, len(st['p_C_true'])))
    pT = st['p_T_true']; noise = float(np.mean(pT*(1-pT)))
    L = np.empty((R, len(sets))); realized = []
    for j, s in enumerate(sets):
        X = np.vstack([st['X_D'], st['X_O'], st['X_C'][s]])
        for r in range(R):
            y = np.concatenate([st['y_D'], Y_O[r], Y_C[r, s]])
            fit = eng.UnifiedLogisticLearner(l2=.45).fit(X, y, w_init=learner.w)
            L[r, j] = float(np.mean((fit.predict_proba(st['X_T'])-pT)**2)) + noise
        realized.append(sup1.evaluate(st, s, learner))
    return L, realized


def decomposition(L, anchor):
    R = L.shape[0]
    post = float(np.mean(L[:, anchor]-L.min(axis=1)))
    means = L.mean(axis=0)
    plug = float(means[anchor]-means.min())
    a, b = np.arange(0, R, 2), np.arange(1, R, 2)
    cf = 0.
    for fit_idx, ev_idx in [(a, b), (b, a)]:
        star = int(np.argmin(L[fit_idx].mean(axis=0)))
        cf += 0.5*float(L[ev_idx, anchor].mean()-L[ev_idx, star].mean())
    return post, plug, cf


def one_state(task):
    group, cohort, seed, arm, R, code = task
    if cohort == 'Controlled':
        st = controlled_state(seed)['only']
        ref = st
    else:
        data = _STATE['data'][_STATE['cohorts'][cohort]['dataset']]
        arms = realx_pair(data, seed, _STATE['cohorts'][cohort]['mechanism'])
        st, ref = arms[arm], arms['broad']
    ctx = sup2.reference_context(*[ref[k] for k in ['X_D', 'y_D', 'X_C', 'ead_C', 'X_Q']], seed)
    choices, sens, learner = select_all(st, ctx, seed)
    # ---- selections locked; evaluator may now touch hidden outcomes ----
    family = {tuple(s.tolist()): s for s in ctx['family']}
    for s in choices.values():
        family.setdefault(tuple(s.tolist()), s)
    fam_keys = sorted(family)
    extra = sorted({tuple(s.tolist()) for s in sens.values()} - set(fam_keys))
    keys = fam_keys + extra
    sets = [np.array(k, int) for k in keys]
    L, realized = evaluate_sets(st, sets, learner, R, np.random.default_rng([seed, code, 7777]))
    index = {k: i for i, k in enumerate(keys)}
    anchor = index[tuple(ctx['M'].tolist())]
    nF = len(fam_keys)
    post, plug, cf = decomposition(L[:, :nF], anchor)
    means, ses = L.mean(axis=0), L.std(axis=0, ddof=1)/np.sqrt(R)
    base = dict(group=group, cohort=cohort, seed=seed, arm=arm, K=len(ctx['M']), B=ctx['B'],
                family_size=nF, family_hash=sup1.digest([list(k) for k in fam_keys]),
                H_post=post, H_pre_plugin=plug, H_pre_crossfit=cf, foresight_lb=post-plug)
    rows = []
    for policy, s in choices.items():
        i = index[tuple(s.tolist())]
        d = L[:, anchor]-L[:, i]
        rows.append(dict(base, policy=policy, k=len(s), selected=json.dumps(s.tolist()),
                         envelope=float(eng.bernstein_envelope(ctx['q'][s], st['ead_C'][s])),
                         exante_loss=float(means[i]), exante_se=float(ses[i]),
                         exante_gain_vs_anchor=float(d.mean()),
                         selection_gap_crossfit=cf-float(d.mean()),
                         realized_brier=realized[i][0], realized_expected_brier=realized[i][1]))
    srows = []
    for name, s in sens.items():
        i = index[tuple(s.tolist())]
        srows.append(dict(group=group, seed=seed, variant=name, k=len(s), exante_loss=float(means[i]),
                          exante_gain_vs_anchor=float(np.mean(L[:, anchor]-L[:, i]))))
    return rows, srows


def _init(datafile, cohorts):
    archive = np.load(datafile, allow_pickle=False)
    _STATE['data'] = {d: {k: archive[d+'_'+k] for k in ['X', 'ead']} for d in {c['dataset'] for c in cohorts.values() if 'dataset' in c}}
    _STATE['cohorts'] = cohorts


def tasks_from(protocol):
    R = protocol['offline']['label_resamples']; out = []
    for cohort, cfg in protocol['offline']['cohorts'].items():
        for seed in range(cfg['seed_start'], cfg['seed_start']+cfg['n']):
            if cohort == 'Controlled':
                out.append(('Controlled', cohort, seed, 'only', R, 0))
            else:
                for code, arm in enumerate(['broad', 'gap'], start=1):
                    out.append((f'{cohort}/{arm}', cohort, seed, arm, R, code))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--protocol', type=Path, required=True); ap.add_argument('--freeze', type=Path, required=True)
    ap.add_argument('--data-dir', type=Path, required=True); ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=20)
    args = ap.parse_args()
    import stats as st_mod
    protocol = json.loads(args.protocol.read_text()); freeze = json.loads(args.freeze.read_text())
    assert hashlib.sha256(args.protocol.read_bytes()).hexdigest() == freeze['protocol_sha256']
    sources = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCE_PATHS}
    assert all(freeze['source_sha256'][p] == h for p, h in sources.items()), 'Frozen source changed'
    datafile = args.data_dir/'real_credit_datasets_authentic.npz'
    assert hashlib.sha256(datafile.read_bytes()).hexdigest() == protocol['data_sha256']
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    manifest = dict(status='RUNNING', study='offline A-C', protocol_sha256=freeze['protocol_sha256'], source_sha256=sources,
                    python=platform.python_version(), numpy=np.__version__)
    (out/'run_manifest.json').write_text(json.dumps(manifest, indent=2))
    started = time.time()
    try:
        tasks = tasks_from(protocol)
        cohorts = protocol['offline']['cohorts']
        with Pool(args.workers, initializer=_init, initargs=(str(datafile), cohorts)) as pool:
            results = pool.map(one_state, tasks, chunksize=1)
        rows = [r for a, _ in results for r in a]; srows = [r for _, b in results for r in b]
        df = pd.DataFrame(rows); sens = pd.DataFrame(srows)
        df.to_csv(out/'offline_policy_results.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
        sens.to_csv(out/'offline_sensitivity.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
        assert len(df) == len(tasks)*len(MAIN_POLICIES) and len(sens) == len(tasks)*len(SENSITIVITY)
        assert (df.envelope <= df.B+1e-10).all()
        assert (df.H_post >= df.H_pre_plugin-1e-12).all() and (df.H_pre_plugin >= -1e-12).all()
        st_mod.offline_statistics(df, sens, protocol, out)
        manifest.update(status='COMPLETE', states=len(tasks), policy_rows=len(df), sensitivity_rows=len(sens),
                        elapsed_seconds=time.time()-started)
    except BaseException as exc:
        manifest.update(status='FAILED', error=f'{type(exc).__name__}: {exc}'); raise
    finally:
        (out/'run_manifest.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
