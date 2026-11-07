"""supplement-v8: what a benchmark-style greedy oracle gap consists of.

Active-learning benchmarks (Werner et al. 2023/2024; Huseljic et al. 2026) bound
achievable performance with a greedy oracle: at each step it adds the candidate
whose *realized* label most improves loss on the evaluation set. Here the same
oracle runs under the risk envelope (it adds feasible candidates only, at most K)
and its reported gap over a reference is split into three parts:

  reported   = E_r[L_T1(ref;Y_r) - L_T1(O_post(Y_r);Y_r)]   oracle selects and is scored on T1
  ex-post    = E_r[L_T2(ref;Y_r) - L_T2(O_post(Y_r);Y_r)]   same oracle, scored on held-out T2
  deployable = cross-fitted ex-ante oracle: selects by the mean T1 loss over half
               the label replicates, scored on T2 over the other half
  test-reuse optimism = reported - ex-post
  aleatoric floor     = ex-post - deployable

Y_r are replicates of the hidden training labels (known p, or boosting p_hat on
disjoint records for real labels); test labels are one recorded or drawn
realization, split into halves T1/T2. Practical policies' gains are scored on T2.
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
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'experiments/supplement_v3'))
import prism_engine as eng
import run_p3_p4 as runner


def _load(name, rel):
    spec = ilu.spec_from_file_location(name, ROOT/rel); mod = ilu.module_from_spec(spec)
    sys.modules[name] = mod; spec.loader.exec_module(mod); return mod


v7 = _load('v7_run', 'experiments/supplement_v7/run.py')
v6, v3, v3on, v5 = v7.v6, v7.v3, v7.v3on, v7.v5
SOURCE_PATHS = ['experiments/supplement_v8/run.py', 'experiments/supplement_v8/stats_v8.py', 'experiments/supplement_v7/run.py'] + \
               [p for p in v7.SOURCE_PATHS if p not in {'experiments/supplement_v7/run.py', 'experiments/supplement_v7/stats_v7.py'}]
PRACTICAL = ['DC-PRISM', 'TargetVar-Greedy', 'BAIT-Greedy', 'Fisher-cost', 'Uncertainty', 'Random-Feasible']


def brier(p, y): return float(np.mean((p-y)**2))


def fit_pred(st, S, y_tr, w):
    X = np.vstack([st['X_D'], st['X_O'], st['X_C'][S]]) if len(S) else np.vstack([st['X_D'], st['X_O']])
    return eng.UnifiedLogisticLearner(l2=.45).fit(X, y_tr, w_init=w).predict_proba(st['X_T'])


def labels(st, S, YO, YC):
    return np.concatenate([st['y_D'], YO, YC[list(S)]]) if len(S) else np.concatenate([st['y_D'], YO])


def greedy_oracle(st, ctx, w, Ys, T1, yT):
    """Greedy over feasible candidates; objective = mean T1 Brier over the replicates in Ys (list of (YO,YC))."""
    q, ead, B, K = ctx['q'], st['ead_C'], ctx['B'], len(ctx['M'])
    S = []
    def obj(S_):
        return np.mean([brier(fit_pred(st, S_, labels(st, S_, YO, YC), w)[T1], yT[T1]) for YO, YC in Ys])
    cur = obj(S)
    while len(S) < K:
        best, bv = None, cur
        for i in range(len(st['X_C'])):
            if i in S: continue
            trial = S+[i]
            if eng.bernstein_envelope(q[trial], ead[trial]) > B+1e-12: continue
            v = obj(trial)
            if v < bv-1e-12: best, bv = i, v
        if best is None: break
        S.append(best); cur = bv
    return sorted(S)


def decompose(st, ctx, w, YO_all, YC_all, yT, choices, seed):
    nT = len(yT); perm = np.random.default_rng([seed, 8001]).permutation(nT); T1, T2 = perm[:nT//2], perm[nT//2:]
    R = len(YO_all); M = list(ctx['M'])
    loss = lambda S, r, T: brier(fit_pred(st, S, labels(st, S, YO_all[r], YC_all[r]), w)[T], yT[T])
    rep, post, post_k = [], [], []
    for r in range(R):
        O = greedy_oracle(st, ctx, w, [(YO_all[r], YC_all[r])], T1, yT)
        rep.append(loss(M, r, T1)-loss(O, r, T1)); post.append(loss(M, r, T2)-loss(O, r, T2)); post_k.append(len(O))
    halves = [np.arange(0, R, 2), np.arange(1, R, 2)]; dep, dep_k = [], []
    for a, b in [(0, 1), (1, 0)]:
        O = greedy_oracle(st, ctx, w, [(YO_all[r], YC_all[r]) for r in halves[a]], T1, yT)
        dep.append(np.mean([loss(M, r, T2)-loss(O, r, T2) for r in halves[b]])); dep_k.append(len(O))
    prac = {p: float(np.mean([loss(M, r, T2)-loss(list(s), r, T2) for r in range(R)])) for p, s in choices.items() if p in PRACTICAL}
    return dict(reported=float(np.mean(rep)), ex_post=float(np.mean(post)), deployable=float(np.mean(dep)),
                oracle_k_post=float(np.mean(post_k)), oracle_k_dep=float(np.mean(dep_k)),
                best_practical=max(prac.values()), best_practical_name=max(prac, key=prac.get),
                **{f'prac_{k}': v for k, v in prac.items()})


def offline_task(task):
    group, cohort, seed, arm, R, code = task
    if cohort == 'Controlled':
        st = v3.controlled_state(seed)['only']; ref = st
    else:
        c = v3._STATE['cohorts'][cohort]; arms = v3.realx_pair(v3._STATE['data'][c['dataset']], seed, c['mechanism'])
        st, ref = arms[arm], arms['broad']
    ctx = v5.reference_context(*[ref[k] for k in ['X_D', 'y_D', 'X_C', 'ead_C', 'X_Q']], seed)
    ch, learner, _ = v5.choose_all(st, ctx, seed, ablation=False)
    rr = np.random.default_rng([seed, code, 8002])
    YO = (rr.random((R, len(st['X_O']))) < st['p_O_true']).astype(int)
    YC = (rr.random((R, len(st['X_C']))) < st['p_C_true']).astype(int)
    out = decompose(st, ctx, learner.w, YO, YC, st['y_T'].astype(float), ch, seed)
    return dict(setting='controlled' if cohort == 'Controlled' else 'semi-synthetic', group=group, seed=seed, K=len(ctx['M']), **out)


def real_task(task):
    stream, dataset, seed, R = task
    X_raw, y_raw, ead_raw = v3on._DATA[dataset]
    part = (runner.build_strictly_hidden_y_free_real_state(X_raw, y_raw, ead_raw, seed=seed, n_rounds=10)
            if stream == 'engineered' else v3on.natural_state(X_raw, y_raw, ead_raw, seed))
    r1 = part['rounds'][0]
    st = dict(X_D=part['X_D0'], y_D=part['y_D0'], X_O=r1['X_O'], X_C=r1['X_C'], ead_C=r1['ead_C'], X_Q=part['X_Q'], X_T=part['X_T'])
    ctx = v5.reference_context(st['X_D'], st['y_D'], st['X_C'], st['ead_C'], st['X_Q'], seed)
    ch, learner, _ = v5.choose_all(st, ctx, seed, ablation=False)
    used = np.concatenate([part['D0_ids'], part['Q_ids'], part['T_ids'], r1['O_ids'], r1['C_ids']]).astype(int)
    mask = np.ones(len(y_raw), bool); mask[used] = False
    aux = np.random.default_rng([seed, 8003]).permutation(np.flatnonzero(mask))[:v6.AUX_CAP]
    f = v6.fit_phat('gbdt', X_raw[aux], y_raw[aux].astype(int), seed)
    rr = np.random.default_rng([seed, 8004])
    YO = (rr.random((R, len(r1['O_ids']))) < f(X_raw[r1['O_ids']])).astype(int)
    YC = (rr.random((R, len(r1['C_ids']))) < f(X_raw[r1['C_ids']])).astype(int)
    out = decompose(st, ctx, learner.w, YO, YC, part['y_T'].astype(float), ch, seed)
    return dict(setting='real-label', group=f'{dataset}/{stream}', seed=seed, K=len(ctx['M']), **out)


def dispatch(t):
    return offline_task(t[1:]) if t[0] == 'off' else real_task(t[1:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--protocol', type=Path, required=True); ap.add_argument('--freeze', type=Path, required=True)
    ap.add_argument('--data-dir', type=Path, required=True); ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=22); ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    sys.path.insert(0, str(ROOT/'experiments/supplement_v8')); import stats_v8
    protocol = json.loads(args.protocol.read_text()); freeze = json.loads(args.freeze.read_text())
    assert hashlib.sha256(args.protocol.read_bytes()).hexdigest() == freeze['protocol_sha256']
    sources = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCE_PATHS}
    if not args.smoke:
        assert all(freeze['source_sha256'][p] == h for p, h in sources.items()), 'Frozen source changed'
    datafile = args.data_dir/'real_credit_datasets_authentic.npz'; extra = args.data_dir/'diabetes_cohort.npz'
    assert hashlib.sha256(datafile.read_bytes()).hexdigest() == protocol['data_sha256']
    assert hashlib.sha256(extra.read_bytes()).hexdigest() == protocol['diabetes_npz_sha256']
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    manifest = dict(status='RUNNING', smoke=args.smoke, protocol_sha256=freeze['protocol_sha256'], source_sha256=sources,
                    python=platform.python_version(), numpy=np.__version__)
    (out/'run_manifest.json').write_text(json.dumps(manifest, indent=2))
    started = time.time()
    try:
        P = protocol['study']; R = 4 if args.smoke else P['label_resamples']
        off = [('off',)+t[:4]+(R,)+t[5:] for t in v3.tasks_from({'offline': {'cohorts': P['offline_cohorts'], 'label_resamples': R}})]
        if args.smoke:
            off = [t for t in off if t[3] == P['offline_cohorts'][t[2]]['seed_start']]
        seeds = [99001] if args.smoke else range(P['real_seed_start'], P['real_seed_start']+P['real_seeds'])
        real = [('real', st, d, s+(1000 if d == 'Diabetes' else 0), R) for st in P['streams'] for d in P['datasets'] for s in seeds]
        tasks = off+real
        with Pool(args.workers, initializer=v7._init, initargs=(str(datafile), str(extra), P['offline_cohorts'], ['logit'])) as pool:
            rows = pool.map(dispatch, tasks, chunksize=1)
        df = pd.DataFrame(rows); df.to_csv(out/'states.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
        if not args.smoke:
            stats_v8.run(df, protocol, out)
        manifest.update(status='COMPLETE', tasks=len(tasks), rows=len(df), elapsed_seconds=time.time()-started)
    except BaseException as exc:
        manifest.update(status='FAILED', error=f'{type(exc).__name__}: {exc}'); raise
    finally:
        (out/'run_manifest.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
