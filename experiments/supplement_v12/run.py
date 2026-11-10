"""supplement-v12: final revision experiments for the oracle-gap paper.

CORE   (E1, E3, E5)  natural-stream round-1 states of the eight cohorts, risk envelope, logistic learner.
       E1  plug-in / cross-fitted bracket on the greedy residual and on the family floor, with nested
           replicate counts R (greedy 8..64, family 8..128).
       E3  stronger oracle searches (greedy + add/remove/swap local search; local search with 10 starts),
           the same search used for the ex-post and the ex-ante oracle, R = 32.
       E5  outcome-model sensitivity on real labels: GBDT p_hat, its logistic recalibration at the point
           slope and at the bootstrap 2.5% / 97.5% slopes (dispersion only), and an MLP p_hat.
EXACT  (E4)  16-candidate pools: every feasible subset evaluated; exact, greedy, local and multi-start
           searches compared on identical replicates and test halves.
BENCH  (E2)  benchmark-style pool-based step: no delay (pending labelled), constant risk score so the budget
           is exactly six labels; history sizes 68, 128, 256, 512 (learning-curve positions); logistic
           learner on CPU; practical strategies scored on T1 (as reported) and T2 (held out).
BENCHMLP (E2) the same states with the one-hidden-layer MLP learner on GPU (greedy searches).
LATER  (E6)  risk-envelope states at history sizes 128, 256, 512 (logistic).

All selections use visible inputs only and are locked before p_hat, recorded labels or replicates are used.
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
sys.path.insert(0, str(ROOT/'experiments/supplement_v10'))
sys.path.insert(0, str(ROOT/'experiments/supplement_v12'))
import prism_engine as eng
import common as C


def _load(name, rel):
    spec = ilu.spec_from_file_location(name, ROOT/rel); mod = ilu.module_from_spec(spec)
    sys.modules[name] = mod; spec.loader.exec_module(mod); return mod


v11 = _load('v11_run', 'experiments/supplement_v11/run.py')
v8, v6, v5 = v11.v8, v11.v6, v11.v5
SOURCE_PATHS = ['experiments/supplement_v12/run.py', 'experiments/supplement_v12/common.py', 'experiments/supplement_v12/stats_v12.py',
                'experiments/supplement_v12/gpu_bench.py', 'experiments/supplement_v10/gpu_mlp.py', 'experiments/supplement_v11/run.py'] + \
               [p for p in v11.SOURCE_PATHS if p not in {'experiments/supplement_v11/run.py', 'experiments/supplement_v11/stats_v11.py'}]
PRACTICAL = v8.PRACTICAL
_S = {}


def _init(files):
    a = np.load(files[0]); b = np.load(files[1]); c = np.load(files[2]); u = np.load(files[3]); D = {}
    for k in ['HMEQ', 'LendingClub', 'PPDai', 'UCI_Credit']:
        D[k] = (a[k+'_X'].astype(float), a[k+'_y'].astype(int), a[k+'_ead'].astype(float))
    D['Diabetes'] = (b['Diabetes_X'].astype(float), b['Diabetes_y'].astype(int), b['Diabetes_ead'].astype(float))
    for k in ['USPS', 'Spambase', 'BankMarketing']:
        D[k] = (c[k+'_X'].astype(float), c[k+'_y'].astype(int), c[k+'_ead'].astype(float))
    D['USPSraw'] = (u['USPSraw_X'].astype(float), u['USPSraw_y'].astype(int), u['USPSraw_ead'].astype(float))
    _S['data'] = D


def halves_T(yT, seed):
    perm = np.random.default_rng([seed, 8001]).permutation(len(yT)); return perm[:len(yT)//2], perm[len(yT)//2:]


def risk_state(dataset, seed, n_C=24, n_hist=68):
    st, meta = C.natural_state(_S['data'], dataset, seed, n_C=n_C, n_hist=n_hist, pending=True)
    ctx = v5.reference_context(st['X_D'], st['y_D'], st['X_C'], st['ead_C'], st['X_Q'], seed)
    sets, idx, anchor, learner = v6.family_and_choices(st, ctx, seed)
    ch, _, _ = v5.choose_all(st, ctx, seed, ablation=False)
    return st, meta, ctx, sets, idx, anchor, learner, ch


def split_aux(meta):
    n_hold = len(meta['aux'])//5; return meta['aux'][n_hold:], meta['aux'][:n_hold]


def oracle_rows(base, out):
    rows = []
    prac = {k: v for k, v in out.items() if isinstance(k, str)}
    for R, v in out.items():
        if isinstance(R, int):
            rows.append(dict(base, R=R, **v, **prac))
    return rows


# ------------------------------------------------------------------ CORE (E1, E3, E5)
def task_core(t):
    dataset, seed, P = t
    st, meta, ctx, sets, idx, anchor, learner, ch = risk_state(dataset, seed)
    prac = {p: s for p, s in ch.items() if p in PRACTICAL}
    # ---- all selections locked ----
    fit, hold = split_aux(meta); X, y = meta['X_raw'], meta['y_raw']
    f = v6.fit_phat('gbdt', X[fit], y[fit].astype(int), seed)
    cal = C.calibration(f(X[hold]), y[hold].astype(float), P['n_boot'], seed)
    g = C.mlp_phat(X[fit], y[fit].astype(int), seed)
    base_p = (f(X[meta['O_ids']]), f(X[meta['C_ids']]))
    variants = {'gbdt': base_p,
                'recal_point': tuple(C.recalibrated(p, cal, cal['b']) for p in base_p),
                'recal_lo': tuple(C.recalibrated(p, cal, cal['b_lo']) for p in base_p),
                'recal_hi': tuple(C.recalibrated(p, cal, cal['b_hi']) for p in base_p),
                'mlp_phat': (g(X[meta['O_ids']]), g(X[meta['C_ids']]))}
    yT = meta['yT']; T1, T2 = halves_T(yT, seed)
    rr = np.random.default_rng([seed, 12011]); U_O, U_C = rr.random((P['R_family'], 14)), rr.random((P['R_family'], 24))
    fz = C.Feasible(ctx['q'], st['ead_C'], ctx['B'], len(ctx['M']), 24)
    rows = []; info = dict(part='CORE', dataset=dataset, seed=seed, K=len(ctx['M']), family_size=len(sets),
                           cal_a=cal['a'], cal_b=cal['b'], cal_b_lo=cal['b_lo'], cal_b_hi=cal['b_hi'], ece=cal['ece'])
    for vn, (pO, pC) in variants.items():
        YO, YC = (U_O < pO).astype(int), (U_C < pC).astype(int)
        base = dict(info, variant=vn, noise_index=float(np.mean(pC*(1-pC))))
        Rf = P['R_family_list'] if vn == 'gbdt' else [P['R_family']]
        L = C.loss_matrix_logit(st, sets, learner.w, YO, YC, yT)
        for R, v in C.family_decomposition(L, anchor, Rf).items():
            rows.append(dict(base, kind='family', method='family', R=R, **v))
        ev = C.LogitEval(st, YO, YC, yT, T1, T2, learner.w)
        if vn == 'gbdt':
            out = C.decompose(ev, C.make_search('greedy', ev, fz, seed), ctx['M'], P['R_greedy_list'], prac)
            rows += oracle_rows(dict(base, kind='oracle', method='greedy'), out)
            for m in ['local', 'restarts']:
                out = C.decompose(ev, C.make_search(m, ev, fz, seed), ctx['M'], [P['R_search']])
                rows += oracle_rows(dict(base, kind='oracle', method=m), out)
        else:
            out = C.decompose(ev, C.make_search('greedy', ev, fz, seed), ctx['M'], [P['R_search']])
            rows += oracle_rows(dict(base, kind='oracle', method='greedy'), out)
    rows[-1]['fits_last_variant'] = ev.fits
    return rows


# ------------------------------------------------------------------ EXACT (E4)
def task_exact(t):
    dataset, seed, n, P = t
    st, meta, ctx, sets, idx, anchor, learner, ch = risk_state(dataset, seed, n_C=n)
    prac = {p: s for p, s in ch.items() if p in PRACTICAL}
    fz = C.Feasible(ctx['q'], st['ead_C'], ctx['B'], len(ctx['M']), n); space = C.feasible_space(fz)
    # ---- all selections locked ----
    fit, _ = split_aux(meta); X, y = meta['X_raw'], meta['y_raw']
    f = v6.fit_phat('gbdt', X[fit], y[fit].astype(int), seed)
    pO, pC = f(X[meta['O_ids']]), f(X[meta['C_ids']]); R = P['R']
    rr = np.random.default_rng([seed, 12013]); YO = (rr.random((R, 14)) < pO).astype(int); YC = (rr.random((R, n)) < pC).astype(int)
    yT = meta['yT']; T1, T2 = halves_T(yT, seed)
    L1 = np.empty((R, len(space))); L2 = np.empty((R, len(space)))
    for j, S in enumerate(space):
        Sl = list(S); X = np.vstack([st['X_D'], st['X_O'], st['X_C'][Sl]])
        for r in range(R):
            yy = np.concatenate([st['y_D'], YO[r], YC[r, Sl]])
            p = eng.UnifiedLogisticLearner(l2=.45).fit(X, yy, w_init=learner.w).predict_proba(st['X_T'])
            L1[r, j] = float(np.mean((p[T1]-yT[T1])**2)); L2[r, j] = float(np.mean((p[T2]-yT[T2])**2))
    ev = C.MatrixEval(space, L1, L2); rows = []
    base = dict(part='EXACT', dataset=dataset, seed=seed, K=len(ctx['M']), n_space=len(space), pool=n, noise_index=float(np.mean(pC*(1-pC))))
    for m in ['exact', 'greedy', 'local', 'restarts']:
        out = C.decompose(ev, C.make_search(m, ev, fz, seed, space=space), ctx['M'], P['R_list'], prac)
        rows += oracle_rows(dict(base, method=m), out)
    return rows


# ------------------------------------------------------------------ BENCH (E2, logistic) and BENCHMLP prep
def bench_state(dataset, seed, n_hist):
    st, meta = C.natural_state(_S['data'], dataset, seed, n_C=24, n_hist=n_hist, pending=False)
    st['ead_C'] = np.ones(24)
    ctx = C.constant_risk_context(v5, st, seed)
    ch, learner, _ = v5.choose_all(st, ctx, seed, ablation=False)
    prac = {p: s for p, s in ch.items() if p in PRACTICAL}
    # ---- all selections locked ----
    fit, _ = split_aux(meta); X, y = meta['X_raw'], meta['y_raw']
    f = v6.fit_phat('gbdt', X[fit], y[fit].astype(int), seed)
    pC = f(X[meta['C_ids']])
    return st, meta, ctx, learner, prac, pC


def task_bench(t):
    dataset, seed, n_hist, P = t
    st, meta, ctx, learner, prac, pC = bench_state(dataset, seed, n_hist); R = P['R']
    rr = np.random.default_rng([seed, n_hist, 12015]); YC = (rr.random((R, 24)) < pC).astype(int); YO = np.zeros((R, 0), int)
    yT = meta['yT']; T1, T2 = halves_T(yT, seed)
    ev = C.LogitEval(st, YO, YC, yT, T1, T2, learner.w); fz = C.Feasible(ctx['q'], st['ead_C'], ctx['B'], len(ctx['M']), 24)
    out = C.decompose(ev, C.make_search('greedy', ev, fz, seed), ctx['M'], [R], prac)
    return oracle_rows(dict(part='BENCH', learner='logit', dataset=dataset, seed=seed, n_hist=n_hist, K=len(ctx['M']),
                            noise_index=float(np.mean(pC*(1-pC)))), out)


def prep_benchmlp(t):
    dataset, seed, n_hist, P = t
    st, meta, ctx, learner, prac, pC = bench_state(dataset, seed, n_hist); R = P['R']
    rr = np.random.default_rng([seed, n_hist, 12015]); YC = (rr.random((R, 24)) < pC).astype(int)
    yT = meta['yT']; T1, T2 = halves_T(yT, seed)
    return dict(dataset=dataset, seed=seed, n_hist=n_hist, st=st, q=ctx['q'], B=ctx['B'], M=tuple(sorted(ctx['M'].tolist())),
                prac={p: tuple(sorted(s.tolist())) for p, s in prac.items()}, YC=YC, yT=yT, T1=T1, T2=T2,
                noise_index=float(np.mean(pC*(1-pC))))


# ------------------------------------------------------------------ LATER (E6)
def task_later(t):
    dataset, seed, n_hist, P = t
    st, meta, ctx, sets, idx, anchor, learner, ch = risk_state(dataset, seed, n_hist=n_hist)
    prac = {p: s for p, s in ch.items() if p in PRACTICAL}
    # ---- all selections locked ----
    fit, _ = split_aux(meta); X, y = meta['X_raw'], meta['y_raw']
    f = v6.fit_phat('gbdt', X[fit], y[fit].astype(int), seed)
    pO, pC = f(X[meta['O_ids']]), f(X[meta['C_ids']])
    rr = np.random.default_rng([seed, n_hist, 12017]); U_O, U_C = rr.random((P['R_family'], 14)), rr.random((P['R_family'], 24))
    YO, YC = (U_O < pO).astype(int), (U_C < pC).astype(int)
    yT = meta['yT']; T1, T2 = halves_T(yT, seed)
    base = dict(part='LATER', dataset=dataset, seed=seed, n_hist=n_hist, K=len(ctx['M']), noise_index=float(np.mean(pC*(1-pC))))
    rows = []
    L = C.loss_matrix_logit(st, sets, learner.w, YO, YC, yT)
    for R, v in C.family_decomposition(L, anchor, [P['R_family']]).items():
        rows.append(dict(base, kind='family', R=R, **v))
    ev = C.LogitEval(st, YO, YC, yT, T1, T2, learner.w); fz = C.Feasible(ctx['q'], st['ead_C'], ctx['B'], len(ctx['M']), 24)
    out = C.decompose(ev, C.make_search('greedy', ev, fz, seed), ctx['M'], [P['R_greedy']], prac)
    rows += oracle_rows(dict(base, kind='oracle'), out)
    return rows


def dispatch(t):
    return {'CORE': task_core, 'EXACT': task_exact, 'BENCH': task_bench, 'LATER': task_later}[t[0]](t[1:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--protocol', type=Path, required=True); ap.add_argument('--freeze', type=Path, required=True)
    ap.add_argument('--data-dir', type=Path, required=True); ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--part', choices=['CORE', 'EXACT', 'BENCH', 'BENCHMLP', 'LATER'], required=True)
    ap.add_argument('--workers', type=int, default=22); ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    import stats_v12
    protocol = json.loads(args.protocol.read_text()); freeze = json.loads(args.freeze.read_text())
    assert hashlib.sha256(args.protocol.read_bytes()).hexdigest() == freeze['protocol_sha256']
    sources = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCE_PATHS}
    if not args.smoke:
        assert all(freeze['source_sha256'][p] == h for p, h in sources.items()), 'Frozen source changed'
    names = ['real_credit_datasets_authentic.npz', 'diabetes_cohort.npz', 'benchmarks_v9.npz', 'usps_raw.npz']
    keys = ['data_sha256', 'diabetes_npz_sha256', 'benchmarks_npz_sha256', 'usps_raw_npz_sha256']
    files = [args.data_dir/n for n in names]
    for fp, k in zip(files, keys):
        assert hashlib.sha256(fp.read_bytes()).hexdigest() == protocol[k], fp
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    manifest = dict(status='RUNNING', part=args.part, smoke=args.smoke, protocol_sha256=freeze['protocol_sha256'], source_sha256=sources,
                    python=platform.python_version(), numpy=np.__version__)
    (out/'run_manifest.json').write_text(json.dumps(manifest, indent=2))
    started = time.time()
    try:
        P = dict(protocol[args.part]); sm = args.smoke
        seeds = P.get('smoke_seeds', [99001]) if sm else list(range(P['seed_start'], P['seed_start']+P['n_seeds']))
        if sm:
            P.update(P.get('smoke_overrides', {}))
        ds = P['datasets'][:P.get('smoke_datasets', 2)] if sm else P['datasets']
        if args.part == 'CORE':
            tasks = [('CORE', d, s, P) for d in ds for s in seeds]
        elif args.part == 'EXACT':
            pools = {16: P['n_seeds'], 20: P['n_seeds_pool20']} if not sm else {16: 1}
            tasks = [('EXACT', d, s, n, P) for n, k in pools.items() for d in ds for s in (seeds[:k] if not sm else seeds)]
        elif args.part in ('BENCH', 'BENCHMLP'):
            hs = P['n_hist'][:2] if sm else P['n_hist']
            dl = [d for d in ds if d in P['learner_datasets']['logit' if args.part == 'BENCH' else 'mlp']]
            tasks = [(args.part, d, s, h, P) for d in dl for h in hs for s in seeds]
        else:
            hs = P['n_hist'][:1] if sm else P['n_hist']
            tasks = [('LATER', d, s, h, P) for d in ds for h in hs for s in seeds]
        rows = []
        with Pool(args.workers, initializer=_init, initargs=([str(f) for f in files],)) as pool:
            if args.part == 'BENCHMLP':
                import torch, gpu_mlp, gpu_bench
                manifest.update(torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0))
                for n, pk in enumerate(pool.imap(prep_benchmlp, [t[1:] for t in tasks], chunksize=1)):
                    rows += gpu_bench.state_rows(pk, gpu_mlp, C)
                    if n % 8 == 0:
                        (out/'progress.txt').write_text(f'{n+1}/{len(tasks)} states, {time.time()-started:.0f}s\n')
            else:
                for n, r in enumerate(pool.imap_unordered(dispatch, tasks, chunksize=1)):
                    rows += r
                    if n % 8 == 0:
                        (out/'progress.txt').write_text(f'{n+1}/{len(tasks)} tasks, {time.time()-started:.0f}s\n')
        df = pd.DataFrame(rows)
        sort = [c for c in ['dataset', 'n_hist', 'seed', 'variant', 'kind', 'method', 'R'] if c in df.columns]
        df = df.sort_values(sort, kind='stable').reset_index(drop=True)
        df.to_csv(out/'rows.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
        if not sm:
            stats_v12.run(args.part, df, protocol, out)
        manifest.update(status='COMPLETE', tasks=len(tasks), rows=len(df), elapsed_seconds=time.time()-started)
    except BaseException as exc:
        manifest.update(status='FAILED', error=f'{type(exc).__name__}: {exc}'); raise
    finally:
        (out/'run_manifest.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
