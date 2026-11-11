"""supplement-v13: full learning curves with three oracles (evaluation-set split versus outcome luck).

Benchmark-style pool-based acquisition without delay or risk: start from 68 labelled records, at each of
30 steps draw 40 fresh candidates and acquire exactly 10 (a constant risk score makes the envelope a pure
cardinality budget). Every method follows its own trajectory and accumulates the recorded labels of what it
acquires. Learner: l2-logistic. Every curve is scored on the same test set T with recorded labels.

  oracle-test   (CDALBench / BoSS style): greedy, uses the candidates' recorded labels, selects on T itself
  oracle-split  (Zhou et al. 2021 style): greedy, recorded candidate labels, selects on a separate validation set V
  oracle-exante : greedy on the mean V loss over R replicates of the candidate labels drawn from p_hat
                  (a gradient-boosting model fitted on records disjoint from T, V, history and stream);
                  it never sees the candidates' labels before acquiring them
  strategies    : Random, Uncertainty, BAIT-greedy, TargetVar-greedy, Fisher-cost (DC-PRISM omitted: it may acquire
                  fewer than b labels by design, which would break the common x-axis)

Accounting on T (Brier, lower is better; area under the curve = mean over the 30 post-acquisition steps):
  test reuse  = L(oracle-split) - L(oracle-test)
  outcome luck = L(oracle-exante) - L(oracle-split)
  remaining   = L(strategy) - L(oracle-exante)
Every oracle acquires exactly 10 labels per step (the best available set even if no candidate improves the
objective), so all curves share the same x-axis.
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

ROOT = Path(__file__).resolve().parents[2]
for p in ['src', 'experiments/supplement_v3', 'experiments/supplement_v12', 'experiments/supplement_v13']:
    sys.path.insert(0, str(ROOT/p))
import prism_engine as eng
import baselines as bl
import common as C


def _load(name, rel):
    spec = ilu.spec_from_file_location(name, ROOT/rel); mod = ilu.module_from_spec(spec)
    sys.modules[name] = mod; spec.loader.exec_module(mod); return mod


v11 = _load('v11_run', 'experiments/supplement_v11/run.py')
v6, v5 = v11.v6, v11.v5
SOURCE_PATHS = ['experiments/supplement_v13/run.py', 'experiments/supplement_v13/stats_v13.py', 'experiments/supplement_v12/common.py',
                'experiments/supplement_v3/baselines.py', 'experiments/supplement_v5/run_offline.py', 'experiments/supplement_v6/run.py',
                'src/prism_engine.py']
STRATS = ['Random', 'Uncertainty', 'BAIT-Greedy', 'TargetVar-Greedy', 'Fisher-cost']  # DC-PRISM omitted: it may acquire fewer than b labels by design
_S = {}


def _init(files):
    a = np.load(files[0]); b = np.load(files[1]); c = np.load(files[2]); D = {}
    for k in ['HMEQ', 'LendingClub', 'PPDai', 'UCI_Credit']:
        D[k] = (a[k+'_X'].astype(float), a[k+'_y'].astype(int), a[k+'_ead'].astype(float))
    D['Diabetes'] = (b['Diabetes_X'].astype(float), b['Diabetes_y'].astype(int), b['Diabetes_ead'].astype(float))
    for k in ['USPS', 'Spambase', 'BankMarketing']:
        D[k] = (c[k+'_X'].astype(float), c[k+'_y'].astype(int), c[k+'_ead'].astype(float))
    _S['data'] = D


def build(dataset, seed, P, v_mult):
    X_raw, y_raw, ead_raw = _S['data'][dataset]
    rng = np.random.default_rng(seed); n = len(X_raw); perm = rng.permutation(n)
    n_T = min(max(int(n*0.18), 500), 1000); n_Q = min(max(int(n*0.12), 350), 600); n_V = n_T*v_mult
    n_stream = P['steps']*P['pool']
    ids_T, ids_V, ids_Q = perm[:n_T], perm[n_T:n_T+n_V], perm[n_T+n_V:n_T+n_V+n_Q]
    rest = perm[n_T+n_V+n_Q:]
    ids_D0, ids_stream, ids_aux = rest[:68], rest[68:68+n_stream], rest[68+n_stream:][:20000]
    assert len(ids_stream) == n_stream and len(ids_aux) >= 500, (dataset, len(ids_aux))
    if dataset in C.NEW:
        mu = X_raw[ids_D0].mean(axis=0); sd = X_raw[ids_D0].std(axis=0); sd = np.where(sd < 1e-6, 1., sd)
        tf = lambda Z: np.clip((Z-mu)/sd, -4.5, 4.5)
    else:
        prep = eng.SplitBeforePreprocessor(tail_scale=1.85).fit(X_raw[ids_D0], ead_raw[ids_D0])
        prep.stds = np.maximum(prep.stds, 1e-3); prep.tau_aux = np.maximum(prep.tau_aux, np.array([1.35, 1.35]))
        tf = prep.transform_X
    f = v6.fit_phat('gbdt', X_raw[ids_aux], y_raw[ids_aux].astype(int), seed)
    S = dict(X_D0=tf(X_raw[ids_D0]), y_D0=y_raw[ids_D0].astype(int), X_T=tf(X_raw[ids_T]), y_T=y_raw[ids_T].astype(float),
             X_V=tf(X_raw[ids_V]), y_V=y_raw[ids_V].astype(float), X_Q=tf(X_raw[ids_Q]),
             X_S=tf(X_raw[ids_stream]), y_S=y_raw[ids_stream].astype(int), p_S=f(X_raw[ids_stream]), n_aux=len(ids_aux))
    return S


def fit(X, y, w):
    return eng.UnifiedLogisticLearner(l2=.45).fit(X, y, w_init=w)


def brier(m, X, y):
    return float(np.mean((m.predict_proba(X)-y)**2))


def greedy_exact_b(XD, yD, XC, YC, E, w, b):
    """Pick exactly b candidates greedily; objective = mean over the rows of YC (label replicates) of the Brier
    score on evaluation set E=(X, y) of the model fitted on history + picked candidates."""
    Xe, ye = E; S = []
    for _ in range(b):
        best, bv = None, np.inf
        for i in range(len(XC)):
            if i in S: continue
            T = S+[i]; X = np.vstack([XD, XC[T]]); v = 0.
            for yc in YC:
                v += brier(fit(X, np.concatenate([yD, yc[T]]), w), Xe, ye)
            v /= len(YC)
            if v < bv-1e-15: best, bv = i, v
        S.append(best)
    return S


def strategy_pick(name, XD, yD, XC, XQ, seed, t):
    st = dict(X_D=XD, y_D=yD, X_O=np.zeros((0, XD.shape[1])), X_C=XC, ead_C=np.ones(len(XC)), X_Q=XQ)
    ctx = C.constant_risk_context(v5, st, seed+t, target_k=_S['b'])
    q, B, ead = ctx['q'], ctx['B'], st['ead_C']
    learner = eng.UnifiedLogisticLearner(l2=.45).fit(XD, yD); XO = st['X_O']
    if name == 'Random':
        s = bl.random_feasible(XC, q, ead, B, np.random.default_rng([seed, t, 13001]))
    elif name == 'Uncertainty':
        s = bl.uncertainty_greedy(learner, XC, q, ead, B)
    elif name == 'BAIT-Greedy':
        s = bl.bait_greedy(learner, XD, XO, XC, XQ, q, ead, B)
    elif name == 'TargetVar-Greedy':
        s = bl.targetvar_greedy(learner, XD, XO, XC, XQ, q, ead, B)
    else:
        pol = 'DC-PRISM' if name == 'DC-PRISM' else 'BAIT-Cost-Safe'
        worlds = eng.sample_common_parameter_worlds(learner, n_worlds=20, rng=np.random.default_rng(seed+t+700))
        s, _, _ = eng.select_acquisition_policy(pol, XD, yD, XO, XC, ead, XQ, B, learner=learner, q_C_precomputed=q,
                                                w_worlds_precomputed=worlds, rng=np.random.default_rng(seed+t+400))
    s = [int(i) for i in np.sort(np.asarray(s, int))]
    assert len(s) == _S['b'], (name, len(s))
    return s


def trajectory(dataset, seed, P, v_mult):
    S = build(dataset, seed, P, v_mult); b, pool, R = P['batch'], P['pool'], P['R']; _S['b'] = b
    methods = ['oracle-test', 'oracle-split', 'oracle-exante'] + STRATS
    lab = {m: (S['X_D0'].copy(), S['y_D0'].copy()) for m in methods}
    w0 = fit(S['X_D0'], S['y_D0'], None).w
    rows = []; T, V = (S['X_T'], S['y_T']), (S['X_V'], S['y_V'])
    m0 = fit(S['X_D0'], S['y_D0'], w0)
    for m in methods:
        rows.append(dict(step=0, method=m, n_labels=len(S['y_D0']), brier_T=brier(m0, *T), brier_V=brier(m0, *V)))
    rr = np.random.default_rng([seed, v_mult, 13003])
    for t in range(P['steps']):
        sl = slice(t*pool, (t+1)*pool); XC, yC, pC = S['X_S'][sl], S['y_S'][sl], S['p_S'][sl]
        Yrep = (rr.random((R, pool)) < pC).astype(int)
        for m in methods:
            XD, yD = lab[m]; w = fit(XD, yD, w0).w
            if m == 'oracle-test':
                pick = greedy_exact_b(XD, yD, XC, yC[None, :], T, w, b)
            elif m == 'oracle-split':
                pick = greedy_exact_b(XD, yD, XC, yC[None, :], V, w, b)
            elif m == 'oracle-exante':
                pick = greedy_exact_b(XD, yD, XC, Yrep, V, w, b)
            else:
                pick = strategy_pick(m, XD, yD, XC, S['X_Q'], seed, t)
            XD = np.vstack([XD, XC[pick]]); yD = np.concatenate([yD, yC[pick]]); lab[m] = (XD, yD)
            mt = fit(XD, yD, w)
            rows.append(dict(step=t+1, method=m, n_labels=len(yD), brier_T=brier(mt, *T), brier_V=brier(mt, *V),
                             picked_pos_rate=float(yC[pick].mean()), picked_noise=float(np.mean(pC[pick]*(1-pC[pick])))))
    base = dict(dataset=dataset, seed=seed, v_mult=v_mult, noise_index=float(np.mean(S['p_S']*(1-S['p_S']))), n_aux=S['n_aux'])
    return [dict(base, **r) for r in rows]


def task(t):
    return trajectory(*t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--protocol', type=Path, required=True); ap.add_argument('--freeze', type=Path, required=True)
    ap.add_argument('--data-dir', type=Path, required=True); ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--part', choices=['MAIN', 'BIGV'], required=True)
    ap.add_argument('--workers', type=int, default=22); ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    import stats_v13
    protocol = json.loads(args.protocol.read_text()); freeze = json.loads(args.freeze.read_text())
    assert hashlib.sha256(args.protocol.read_bytes()).hexdigest() == freeze['protocol_sha256']
    sources = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCE_PATHS}
    if not args.smoke:
        assert all(freeze['source_sha256'][p] == h for p, h in sources.items()), 'Frozen source changed'
    names = ['real_credit_datasets_authentic.npz', 'diabetes_cohort.npz', 'benchmarks_v9.npz']
    keys = ['data_sha256', 'diabetes_npz_sha256', 'benchmarks_npz_sha256']
    files = [args.data_dir/n for n in names]
    for fp, k in zip(files, keys):
        assert hashlib.sha256(fp.read_bytes()).hexdigest() == protocol[k], fp
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    manifest = dict(status='RUNNING', part=args.part, smoke=args.smoke, protocol_sha256=freeze['protocol_sha256'], source_sha256=sources,
                    python=platform.python_version(), numpy=np.__version__)
    (out/'run_manifest.json').write_text(json.dumps(manifest, indent=2))
    started = time.time()
    try:
        P = dict(protocol['common']); Q = protocol[args.part]
        if args.smoke:
            P.update(protocol['smoke_overrides'])
        seeds = [99001] if args.smoke else list(range(Q['seed_start'], Q['seed_start']+Q['n_seeds']))
        ds = Q['datasets'][:2] if args.smoke else Q['datasets']
        tasks = [(d, s, P, Q['v_mult']) for d in ds for s in seeds]
        rows = []
        with Pool(args.workers, initializer=_init, initargs=([str(f) for f in files],)) as pool:
            for n, r in enumerate(pool.imap_unordered(task, tasks, chunksize=1)):
                rows += r
                (out/'progress.txt').write_text(f'{n+1}/{len(tasks)} trajectories, {time.time()-started:.0f}s\n')
        df = pd.DataFrame(rows).sort_values(['dataset', 'seed', 'method', 'step'], kind='stable').reset_index(drop=True)
        df.to_csv(out/'rows.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
        if not args.smoke:
            stats_v13.run(args.part, df, protocol, out)
        manifest.update(status='COMPLETE', tasks=len(tasks), rows=len(df), elapsed_seconds=time.time()-started)
    except BaseException as exc:
        manifest.update(status='FAILED', error=f'{type(exc).__name__}: {exc}'); raise
    finally:
        (out/'run_manifest.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
