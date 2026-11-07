"""supplement-v7: generality of the protocol's findings.

A (learner). The same states are scored with two downstream learners, the
  l2-logistic model used throughout and a small gradient-boosting model, on
  common label replicates: controlled worlds, semi-synthetic pairs (known p) and
  credit real-label round-1 states (boosting p_hat, as in supplement-v6).
B (domain). Real-label round-1 states and a ten-round delayed replay on a
  non-credit cohort (UCI Diabetes 130-US hospitals; adverse = readmission within
  30 days; exposure = days in hospital), engineered and natural streams.
C (analytic EVPI). For every state of A and B, the two-leader normal EVPI
  s*phi(D/s) - D*Phi(-D/s), with D and s the mean and s.d. of the loss
  difference between the best and second-best set, is recorded next to the
  measured EVPI, to test whether the floor is predicted by label noise between
  near-tied leaders.

Selections are made by the unchanged policies from visible inputs and locked
before any label, p_hat or replicate is used.
"""
import argparse
import hashlib
import importlib.util as ilu
import json
from multiprocessing import Pool
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'experiments/supplement_v3'))
import prism_engine as eng
import run_p3_p4 as runner


def _load(name, rel):
    spec = ilu.spec_from_file_location(name, ROOT/rel); mod = ilu.module_from_spec(spec)
    sys.modules[name] = mod; spec.loader.exec_module(mod); return mod


v6 = _load('v6_run', 'experiments/supplement_v6/run.py')
v5on = _load('v5_run_online', 'experiments/supplement_v5/run_online.py')
v3, v3on, v5 = v6.v3, v6.v3on, v6.v5
SOURCE_PATHS = ['experiments/supplement_v7/run.py', 'experiments/supplement_v7/stats_v7.py', 'scripts/prepare_diabetes.py',
                'experiments/supplement_v6/run.py', 'experiments/supplement_v5/run_online.py'] + \
               [p for p in v6.SOURCE_PATHS if p not in {'experiments/supplement_v6/run.py', 'experiments/supplement_v6/stats_v6.py'}]
_S = {}


# ------------------------------------------------------------------ downstream learners
class _Const:
    def __init__(self, p): self.p = p
    def predict_proba(self, X): return np.full(len(X), self.p)


class _GB:
    def __init__(self, m): self.m = m
    def predict_proba(self, X): return self.m.predict_proba(X)[:, 1]


def fit_learner(kind, X, y, w_init):
    if kind == 'logit':
        return eng.UnifiedLogisticLearner(l2=.45).fit(X, y, w_init=w_init)
    if y.min() == y.max():
        return _Const(float(np.clip(y.mean(), 1e-3, 1-1e-3)))
    m = HistGradientBoostingClassifier(max_iter=50, learning_rate=.1, max_leaf_nodes=8, min_samples_leaf=10,
                                       l2_regularization=1.0, early_stopping=False, random_state=0)
    return _GB(m.fit(X, y))


def loss_matrix(kind, st, sets, w_init, U_O, U_C, pO, pC, test):
    R = U_O.shape[0]; L = np.empty((R, len(sets)))
    YO = (U_O < pO).astype(int); YC = (U_C < pC).astype(int)
    for j, s in enumerate(sets):
        X = np.vstack([st['X_D'], st['X_O'], st['X_C'][s]])
        for r in range(R):
            y = np.concatenate([st['y_D'], YO[r], YC[r, s]])
            L[r, j] = test(fit_learner(kind, X, y, w_init).predict_proba(st['X_T']))
    return L


def two_leader(L, anchor):
    d = L[:, [anchor]]-L                      # gains over the anchor, R x |F|
    mu = d.mean(axis=0); order = np.argsort(-mu); b, r = order[0], order[1]
    D = d[:, b]-d[:, r]; delta, s = float(D.mean()), float(D.std(ddof=1))
    tl = s*norm.pdf(delta/s)-delta*norm.cdf(-delta/s) if s > 1e-12 else 0.
    sd_all = float(np.median(d.std(axis=0, ddof=1)))
    near = int(np.sum(mu >= mu[b]-2*np.maximum(d.std(axis=0, ddof=1), 1e-12)))
    # upper-side approximation: independent normals over all sets (ignores positive correlation, so overstates)
    sd = np.maximum(d.std(axis=0, ddof=1), 1e-12)
    Z = np.random.default_rng(12345).standard_normal((4000, len(mu)))
    ind = float(np.mean(np.max(mu+sd*Z, axis=1))-mu[b])
    return dict(tl_evpi=float(tl), ind_evpi=ind, tl_delta=delta, tl_s=s, median_gain_sd=sd_all, near_ties=near)


def score(rows, base, kind, L, anchor, idx):
    post, plug, cf = v3.decomposition(L, anchor)
    g = v6.gains(L, anchor, idx)
    rows.append(dict(base, learner=kind, H_post=post, H_pre_plugin=plug, H_pre_crossfit=cf, foresight=post-plug,
                     **two_leader(L, anchor), **{f'gain_{k}': v for k, v in g.items()}))


# ------------------------------------------------------------------ states
def offline_state(task):
    """Controlled or semi-synthetic state with known p (A)."""
    group, cohort, seed, arm, R, code = task
    if cohort == 'Controlled':
        st = v3.controlled_state(seed)['only']; ref = st
    else:
        c = v3._STATE['cohorts'][cohort]; arms = v3.realx_pair(v3._STATE['data'][c['dataset']], seed, c['mechanism'])
        st, ref = arms[arm], arms['broad']
    ctx = v5.reference_context(*[ref[k] for k in ['X_D', 'y_D', 'X_C', 'ead_C', 'X_Q']], seed)
    sets, idx, anchor, learner = v6.family_and_choices(st, ctx, seed)
    rr = np.random.default_rng([seed, code, 7001])
    U_O, U_C = rr.random((R, len(st['X_O']))), rr.random((R, len(st['X_C'])))
    pT = st['p_T_true']; noise = float(np.mean(pT*(1-pT)))
    test = lambda p: float(np.mean((p-pT)**2))+noise
    rows, base = [], dict(study='A', setting='controlled' if cohort == 'Controlled' else 'semi-synthetic', group=group,
                          seed=seed, family_size=len(sets))
    for kind in _S['learners']:
        score(rows, base, kind, loss_matrix(kind, st, sets, learner.w, U_O, U_C, st['p_O_true'], st['p_C_true'], test), anchor, idx)
    return rows


def real_state(task):
    """Round-1 real-label state with boosting p_hat (A credit, B diabetes)."""
    study, stream, dataset, seed, R = task
    X_raw, y_raw, ead_raw = v3on._DATA[dataset]
    part = (runner.build_strictly_hidden_y_free_real_state(X_raw, y_raw, ead_raw, seed=seed, n_rounds=10)
            if stream == 'engineered' else v3on.natural_state(X_raw, y_raw, ead_raw, seed))
    r1 = part['rounds'][0]
    st = dict(X_D=part['X_D0'], y_D=part['y_D0'], X_O=r1['X_O'], X_C=r1['X_C'], ead_C=r1['ead_C'], X_Q=part['X_Q'], X_T=part['X_T'])
    ctx = v5.reference_context(st['X_D'], st['y_D'], st['X_C'], st['ead_C'], st['X_Q'], seed)
    sets, idx, anchor, learner = v6.family_and_choices(st, ctx, seed)
    # ---- choices locked ----
    used = np.concatenate([part['D0_ids'], part['Q_ids'], part['T_ids'], r1['O_ids'], r1['C_ids']]).astype(int)
    mask = np.ones(len(y_raw), bool); mask[used] = False
    aux = np.random.default_rng([seed, 7003]).permutation(np.flatnonzero(mask))[:v6.AUX_CAP]
    f = v6.fit_phat('gbdt', X_raw[aux], y_raw[aux].astype(int), seed)
    pO, pC = f(X_raw[r1['O_ids']]), f(X_raw[r1['C_ids']])
    yT = part['y_T'].astype(float); test = lambda p: float(np.mean((p-yT)**2))
    rr = np.random.default_rng([seed, 7004])
    U_O, U_C = rr.random((R, len(r1['O_ids']))), rr.random((R, len(r1['C_ids'])))
    rows, base = [], dict(study=study, setting='real-label', group=f'{dataset}/{stream}', seed=seed, family_size=len(sets))
    yO, yC = r1['y_O'].astype(float), r1['y_C'].astype(float)
    for kind in _S['learners']:
        score(rows, base, kind, loss_matrix(kind, st, sets, learner.w, U_O, U_C, pO, pC, test), anchor, idx)
        Lr = loss_matrix(kind, st, sets, learner.w, np.zeros((1, len(yO))), np.zeros((1, len(yC))), yO, yC, test)
        rows[-1]['H_post_realized'] = float(Lr[0, anchor]-Lr[0].min())
    return rows


def replay_task(task):
    return v5on.replay(task)


def _init(datafile, extra_file, cohorts, learners):
    v3._init(datafile, cohorts)
    a = np.load(extra_file)
    dia = (a['Diabetes_X'].astype(float), a['Diabetes_y'].astype(int), a['Diabetes_ead'].astype(float))
    for mod in {id(v3on): v3on, id(v5on.v3on): v5on.v3on}.values():   # both loaded copies of the replay module
        mod._init(datafile); mod._DATA['Diabetes'] = dia
    _S['learners'] = learners


def dispatch(task):
    return offline_state(task[1:]) if task[0] == 'off' else real_state(task[1:]) if task[0] == 'real' else replay_task(task[1:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--protocol', type=Path, required=True); ap.add_argument('--freeze', type=Path, required=True)
    ap.add_argument('--data-dir', type=Path, required=True); ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=22); ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    sys.path.insert(0, str(ROOT/'experiments/supplement_v7')); import stats_v7
    protocol = json.loads(args.protocol.read_text()); freeze = json.loads(args.freeze.read_text())
    assert hashlib.sha256(args.protocol.read_bytes()).hexdigest() == freeze['protocol_sha256']
    sources = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCE_PATHS}
    if not args.smoke:
        assert all(freeze['source_sha256'][p] == h for p, h in sources.items()), 'Frozen source changed'
    datafile = args.data_dir/'real_credit_datasets_authentic.npz'; extra = args.data_dir/'diabetes_cohort.npz'
    assert hashlib.sha256(datafile.read_bytes()).hexdigest() == protocol['data_sha256']
    assert hashlib.sha256(extra.read_bytes()).hexdigest() == protocol['diabetes_npz_sha256']
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    import sklearn
    manifest = dict(status='RUNNING', smoke=args.smoke, protocol_sha256=freeze['protocol_sha256'], source_sha256=sources,
                    python=platform.python_version(), numpy=np.__version__, sklearn=sklearn.__version__)
    (out/'run_manifest.json').write_text(json.dumps(manifest, indent=2))
    started = time.time()
    try:
        A, B = protocol['study_A'], protocol['study_B']
        sm = args.smoke
        R = 8 if sm else A['label_resamples']
        off = [('off',)+t[:4]+(R,)+t[5:] for t in v3.tasks_from({'offline': {'cohorts': A['offline_cohorts'], 'label_resamples': R}})]
        if sm:
            off = [t for t in off if t[3] == A['offline_cohorts'][t[2]]['seed_start']]
        seedsA = [99001] if sm else range(A['real_seed_start'], A['real_seed_start']+A['real_seeds'])
        realA = [('real', 'A', st, d, s, R) for st in A['streams'] for d in A['datasets'] for s in seedsA]
        seedsB = [99001] if sm else range(B['seed_start'], B['seed_start']+B['n_seeds'])
        realB = [('real', 'B', st, 'Diabetes', s, 8 if sm else B['label_resamples']) for st in B['streams'] for s in seedsB]
        seedsH = [99001] if sm else range(B['replay_seed_start'], B['replay_seed_start']+B['replay_seeds'])
        rep = [('rep', st, 'Diabetes', s, v5on.POLICIES) for st in B['streams'] for s in seedsH]
        tasks = off+realA+realB+rep
        with Pool(args.workers, initializer=_init, initargs=(str(datafile), str(extra), A['offline_cohorts'], protocol['learners'])) as pool:
            res = pool.map(dispatch, tasks, chunksize=1)
        nst = len(off)+len(realA)+len(realB)
        states = pd.DataFrame([r for x in res[:nst] for r in x])
        traj = pd.DataFrame([r for x in res[nst:] for r in x])
        states.to_csv(out/'states.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
        per_seed = v5on.per_seed_metrics(traj)
        per_seed.to_csv(out/'replay_seed_metrics.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
        if not sm:
            stats_v7.run(states, per_seed, traj, protocol, out)
        manifest.update(status='COMPLETE', tasks=len(tasks), state_rows=len(states), replay_rows=len(traj),
                        elapsed_seconds=time.time()-started)
    except BaseException as exc:
        manifest.update(status='FAILED', error=f'{type(exc).__name__}: {exc}'); raise
    finally:
        (out/'run_manifest.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
