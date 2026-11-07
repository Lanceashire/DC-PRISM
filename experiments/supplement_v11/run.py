"""supplement-v11: revision studies answering the two strict reviews.

A  Valid floor bounds. For every state, from the replicate gain matrix d_S = L(M) - L(S):
     lower  F_lo  = mean (d_r - d_b)^+            (b, r: best and second-best set by mean gain)
     plug-in floor F = mean max_S d_S - max_S mean d_S
     upper  F_up  = mean max_S (d_S - mean d_S)
   F_lo <= F <= F_up holds exactly in sample for any distribution and correlation; population
   versions follow from E max <= max E and the Gaussian maximal inequality, reported as
   G = sigma_max sqrt(2 ln |F|). The superseded independent-normal value F_ind is recorded
   only to document its failure. Natural-stream states of eight cohorts, semi-synthetic and
   controlled states, logistic and boosting learners; plus pure-simulation stress tests.
B  Estimand alignment. Twelve-candidate pools: every feasible subset is evaluated, so the exact
   ex-post and ex-ante oracles over the whole feasible space can be compared with the greedy
   benchmark oracle and with the DC-PRISM family oracle on the same replicates and test halves.
C  p_hat robustness. (C1) semi-synthetic known-p states: floor, H_pre and H_post under p, p_hat and
   perturbed p_hat (logit temperature 0.5, 2; logit shift -0.5, +0.5). (C2) real-label states: the
   greedy benchmark decomposition computed from the recorded training labels (one realization)
   against its p_hat-replicate mean; p_hat calibration (ECE, calibration slope) on held-out records.
D  (D1) label-noise dose-response: recorded labels flipped with probability 0, 0.1, 0.2, 0.3 in
   three cohorts; (D2) candidate-pool size 24, 48, 96 in three cohorts.
E  Exact maximum-cardinality reference for the 88 supplement-v5 states whose cardinality was not
   certified (run by experiments/supplement_v11/exact_mk.py).

All selections use visible inputs and are locked before replicates, recorded labels or p_hat are used.
"""
import argparse
import hashlib
import importlib.util as ilu
import itertools
import json
import zlib
from multiprocessing import Pool
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'experiments/supplement_v3'))
import prism_engine as eng


def _load(name, rel):
    spec = ilu.spec_from_file_location(name, ROOT/rel); mod = ilu.module_from_spec(spec)
    sys.modules[name] = mod; spec.loader.exec_module(mod); return mod


v9 = _load('v9_run', 'experiments/supplement_v9/run.py')
v8, v7, v6, v3on, v5, v3 = v9.v8, v9.v7, v9.v6, v9.v3on, v9.v5, v9.v6.v3
SOURCE_PATHS = ['experiments/supplement_v11/run.py', 'experiments/supplement_v11/stats_v11.py', 'experiments/supplement_v9/run.py'] + \
               [p for p in v9.SOURCE_PATHS if p not in {'experiments/supplement_v9/run.py', 'experiments/supplement_v9/stats_v9.py'}]
NEW = {'USPS', 'Spambase', 'BankMarketing'}
_S = {}


# ------------------------------------------------------------------ bounds
def bounds(L, anchor):
    d = L[:, [anchor]]-L; mu = d.mean(axis=0); R, K = d.shape
    F = float(d.max(axis=1).mean()-mu.max())
    o = np.argsort(-mu); b, r = o[0], o[1]
    F_lo = float(np.maximum(0., d[:, r]-d[:, b]).mean())
    F_up = float((d-mu).max(axis=1).mean())
    sd = d.std(axis=0, ddof=1); G = float(sd.max()*np.sqrt(2*np.log(K)))
    D = d[:, b]-d[:, r]; s = D.std(ddof=1); De = D.mean()
    F_two_normal = float(s*norm.pdf(De/s)-De*norm.cdf(-De/s)) if s > 1e-12 else 0.
    Z = np.random.default_rng(12345).standard_normal((4000, K)); sdz = np.maximum(sd, 1e-12)
    F_ind = float(np.mean(np.max(mu+sdz*Z, axis=1))-mu.max())
    C = np.corrcoef(d[:, sd > 1e-12].T) if (sd > 1e-12).sum() > 1 else np.ones((1, 1))
    iu = np.triu_indices(len(C), 1)
    skew = float(((D-De)**3).mean()/max(s**3, 1e-18)) if s > 1e-12 else 0.
    return dict(F=F, F_lo=F_lo, F_up=F_up, G=G, F_two_normal=F_two_normal, F_ind=F_ind, n_sets=K,
                neg_corr_share=float((C[iu] < 0).mean()) if len(iu[0]) else 0., skew_D=skew)


# ------------------------------------------------------------------ states
def natural_state_n(dataset, seed, n_C, flip=0.):
    """Natural stream round 1 with n_C candidates (14 pending); optional label flips (noise dose)."""
    X_raw, y_raw, ead_raw = _S['data'][dataset]
    if flip > 0:
        y_raw = np.where(np.random.default_rng([zlib.crc32(dataset.encode()), 11001]).random(len(y_raw)) < flip, 1-y_raw, y_raw)
    rng = np.random.default_rng(seed); n = len(X_raw); perm = rng.permutation(n)
    n_T = min(max(int(n*0.18), 500), 1000); n_Q = min(max(int(n*0.12), 350), 600)
    ids_T, ids_Q, pool = perm[:n_T], perm[n_T:n_T+n_Q], perm[n_T+n_Q:]
    ids_D0 = pool[:68]; stream = rng.permutation(pool[600:]); o, c = stream[:14], stream[14:14+n_C]
    if dataset in NEW:
        mu = X_raw[ids_D0].mean(axis=0); sd = X_raw[ids_D0].std(axis=0); sd = np.where(sd < 1e-6, 1., sd)
        tf = lambda Z: np.clip((Z-mu)/sd, -4.5, 4.5); te = lambda e: np.ones(len(e))
    else:
        prep = eng.SplitBeforePreprocessor(tail_scale=1.85).fit(X_raw[ids_D0], ead_raw[ids_D0])
        prep.stds = np.maximum(prep.stds, 1e-3); prep.tau_aux = np.maximum(prep.tau_aux, np.array([1.35, 1.35]))
        tf, te = prep.transform_X, prep.transform_ead
    st = dict(X_D=tf(X_raw[ids_D0]), y_D=y_raw[ids_D0].astype(int), X_O=tf(X_raw[o]), X_C=tf(X_raw[c]), ead_C=te(ead_raw[c]),
              X_Q=tf(X_raw[ids_Q]), X_T=tf(X_raw[ids_T]))
    used = np.concatenate([ids_D0, ids_Q, ids_T, o, c]); mask = np.ones(n, bool); mask[used] = False
    aux = np.random.default_rng([seed, 11003]).permutation(np.flatnonzero(mask))[:v6.AUX_CAP]
    return st, dict(X_raw=X_raw, y_raw=y_raw, O_ids=o, C_ids=c, aux=aux, yT=y_raw[ids_T].astype(float),
                    yO=y_raw[o].astype(int), yC=y_raw[c].astype(int))


def phat_for(meta, seed):
    n_hold = len(meta['aux'])//5; hold, fit = meta['aux'][:n_hold], meta['aux'][n_hold:]
    f = v6.fit_phat('gbdt', meta['X_raw'][fit], meta['y_raw'][fit].astype(int), seed)
    ph, yh = f(meta['X_raw'][hold]), meta['y_raw'][hold].astype(float)
    bins = np.minimum((ph*10).astype(int), 9)
    ece = float(sum(abs(ph[bins == k].mean()-yh[bins == k].mean())*(bins == k).mean() for k in range(10) if (bins == k).any()))
    lg = logit(np.clip(ph, 1e-4, 1-1e-4))
    slope = float(eng.UnifiedLogisticLearner(l2=1e-6).fit(lg[:, None], yh.astype(int)).w[1])  # calibration slope of y on logit(p_hat)
    return f, dict(ece=ece, cal_slope=slope, holdout_brier=float(np.mean((ph-yh)**2)))


# ------------------------------------------------------------------ A: bounds on families
def task_a(t):
    kind, key, seed, R = t
    if kind == 'natural':
        st, meta = natural_state_n(key, seed, 24)
        ctx = v5.reference_context(st['X_D'], st['y_D'], st['X_C'], st['ead_C'], st['X_Q'], seed)
        sets, idx, anchor, learner = v6.family_and_choices(st, ctx, seed)
        f, cal = phat_for(meta, seed); pO, pC = f(meta['X_raw'][meta['O_ids']]), f(meta['X_raw'][meta['C_ids']])
        yT = meta['yT']; test = lambda p: float(np.mean((p-yT)**2)); group = f'{key}/natural'
    else:  # known p: semi-synthetic arm or controlled world
        if key == 'Controlled':
            st = v3.controlled_state(seed)['only']; ref = st; group = 'Controlled'
        else:
            c = v3._STATE['cohorts'][key]; arms = v3.realx_pair(v3._STATE['data'][c['dataset']], seed, c['mechanism'])
            arm = 'gap' if seed % 2 else 'broad'; st, ref = arms[arm], arms['broad']; group = f'{key}/{arm}'
        ctx = v5.reference_context(*[ref[k] for k in ['X_D', 'y_D', 'X_C', 'ead_C', 'X_Q']], seed)
        sets, idx, anchor, learner = v6.family_and_choices(st, ctx, seed)
        pO, pC = st['p_O_true'], st['p_C_true']; pT = st['p_T_true']; nz = float(np.mean(pT*(1-pT)))
        test = lambda p: float(np.mean((p-pT)**2))+nz; cal = {}
    rr = np.random.default_rng([seed, 11005]); U_O, U_C = rr.random((R, len(pO))), rr.random((R, len(pC)))
    rows = []
    for lr in ['logit', 'gbdt']:
        L = v7.loss_matrix(lr, st, sets, learner.w, U_O, U_C, pO, pC, test)
        rows.append(dict(part='A', kind=kind, group=group, seed=seed, learner=lr, noise_index=float(np.mean(pC*(1-pC))), **bounds(L, anchor), **cal))
    return rows


def stress_tests():
    """Pure simulation: correlation structures and marginals where the old F_ind fails; new bounds hold."""
    rng = np.random.default_rng(11007); rows = []
    for name, K in itertools.product(['independent', 'equicorrelated_0.8', 'antithetic_pairs', 'skewed_lognormal', 'heavy_t3'], [3, 10, 30]):
        for rep in range(200):
            R = 2000
            if name == 'independent': X = rng.standard_normal((R, K))
            elif name == 'equicorrelated_0.8': X = np.sqrt(.8)*rng.standard_normal((R, 1))+np.sqrt(.2)*rng.standard_normal((R, K))
            elif name == 'antithetic_pairs':
                Zh = rng.standard_normal((R, (K+1)//2)); X = np.concatenate([Zh, -Zh], axis=1)[:, :K]
            elif name == 'skewed_lognormal': X = np.exp(rng.standard_normal((R, K)))-np.exp(.5)
            else: X = rng.standard_t(3, (R, K))
            mu = rng.normal(0, .3, K); d = np.concatenate([np.zeros((R, 1)), mu+X], axis=1)
            b = bounds(-(d-d[:, [0]]), 0)   # bounds() expects losses with anchor column; losses = -gains
            rows.append(dict(part='stress', structure=name, K=K, rep=rep, **b))
    return rows


# ------------------------------------------------------------------ B: exact feasible space, 12 candidates
def task_b(t):
    dataset, seed, R = t
    st, meta = natural_state_n(dataset, seed, 12)
    ctx = v5.reference_context(st['X_D'], st['y_D'], st['X_C'], st['ead_C'], st['X_Q'], seed)
    sets_F, idx, anchor_F, learner = v6.family_and_choices(st, ctx, seed)
    ch, _, _ = v5.choose_all(st, ctx, seed, ablation=False)
    q, ead, B = ctx['q'], st['ead_C'], ctx['B']
    space = [()]
    for k in range(1, 13):
        for S in itertools.combinations(range(12), k):
            if eng.bernstein_envelope(q[list(S)], ead[list(S)]) <= B+1e-12: space.append(S)
    # ---- all selections locked ----
    f, cal = phat_for(meta, seed)
    pO, pC = f(meta['X_raw'][meta['O_ids']]), f(meta['X_raw'][meta['C_ids']])
    rr = np.random.default_rng([seed, 11009]); YO = (rr.random((R, 14)) < pO).astype(int); YC = (rr.random((R, 12)) < pC).astype(int)
    yT = meta['yT']; nT = len(yT); perm = np.random.default_rng([seed, 8001]).permutation(nT); T1, T2 = perm[:nT//2], perm[nT//2:]
    L1 = np.empty((R, len(space))); L2 = np.empty((R, len(space)))
    for j, S in enumerate(space):
        S = list(S); X = np.vstack([st['X_D'], st['X_O'], st['X_C'][S]]) if S else np.vstack([st['X_D'], st['X_O']])
        for r in range(R):
            y = np.concatenate([st['y_D'], YO[r], YC[r, S]]) if S else np.concatenate([st['y_D'], YO[r]])
            p = eng.UnifiedLogisticLearner(l2=.45).fit(X, y, w_init=learner.w).predict_proba(st['X_T'])
            L1[r, j] = float(np.mean((p[T1]-yT[T1])**2)); L2[r, j] = float(np.mean((p[T2]-yT[T2])**2))
    key = {tuple(sorted(S)): j for j, S in enumerate(space)}; a = key[tuple(sorted(ctx['M'].tolist()))]
    Fidx = [key[tuple(sorted(s.tolist()))] for s in sets_F]
    def oracle(cols):
        o1 = [cols[int(np.argmin(L1[r, cols]))] for r in range(R)]
        rep = float(np.mean([L1[r, a]-L1[r, o] for r, o in enumerate(o1)])); post = float(np.mean([L2[r, a]-L2[r, o] for r, o in enumerate(o1)]))
        h = [np.arange(0, R, 2), np.arange(1, R, 2)]; dep = []
        for x, y in [(0, 1), (1, 0)]:
            o = cols[int(np.argmin(L1[h[x]][:, cols].mean(axis=0)))]; dep.append(float(np.mean(L2[h[y], a]-L2[h[y], o])))
        return rep, post, float(np.mean(dep))
    all_cols = list(range(len(space)))
    ex = oracle(all_cols); fam = oracle(Fidx)
    gr = v8.decompose(st, ctx, learner.w, YO, YC, yT, ch, seed)
    prac = {p: float(np.mean(L2[:, a]-L2[:, key[tuple(sorted(s.tolist()))]])) for p, s in ch.items() if p in v8.PRACTICAL}
    return [dict(part='B', dataset=dataset, seed=seed, n_space=len(space), n_family=len(Fidx), K=len(ctx['M']),
                 exact_reported=ex[0], exact_expost=ex[1], exact_deployable=ex[2],
                 family_reported=fam[0], family_expost=fam[1], family_deployable=fam[2],
                 greedy_reported=gr['reported'], greedy_expost=gr['ex_post'], greedy_deployable=gr['deployable'],
                 **{f'prac_{k}': v for k, v in prac.items()})]


# ------------------------------------------------------------------ C: p_hat robustness
def task_c1(t):
    key, seed, R = t
    c = v3._STATE['cohorts'][key]; arms = v3.realx_pair(v3._STATE['data'][c['dataset']], seed, c['mechanism'])
    arm = 'gap' if seed % 2 else 'broad'; st, ref = arms[arm], arms['broad']
    ctx = v5.reference_context(*[ref[k] for k in ['X_D', 'y_D', 'X_C', 'ead_C', 'X_Q']], seed)
    sets, idx, anchor, learner = v6.family_and_choices(st, ctx, seed)
    data = v3._STATE['data'][c['dataset']]; rng = np.random.default_rng(seed); perm = rng.permutation(len(data['X'])); ids = perm[:706]
    prep = eng.SplitBeforePreprocessor(tail_scale=1).fit(data['X'][ids][:80], data['ead'][ids][:80]); prep.tau_aux = np.zeros(2)
    Xa = prep.transform_X(data['X'][perm[706:][:v6.AUX_CAP]])
    def truth(X):
        z = -.8+X @ np.array([.45, .65, -.5, .35, .2, -.15])
        return expit(z+(.5*np.sin(X[:, 1])+.35*X[:, 2]*X[:, 3] if c['mechanism'] == 'nonlinear' else 0.))
    ya = (np.random.default_rng([seed, 11011]).random(len(Xa)) < truth(Xa)).astype(int)
    f = v6.fit_phat('gbdt', Xa, ya, seed)
    pT = st['p_T_true']; nz = float(np.mean(pT*(1-pT))); test = lambda p: float(np.mean((p-pT)**2))+nz
    rr = np.random.default_rng([seed, 11013]); U_O, U_C = rr.random((R, len(st['X_O']))), rr.random((R, len(st['X_C'])))
    variants = {'truth': (st['p_O_true'], st['p_C_true'])}
    base = (f(st['X_O']), f(st['X_C']))
    for nm, fn in [('phat', lambda p: p), ('temp0.5', lambda p: expit(logit(p)/.5)), ('temp2', lambda p: expit(logit(p)/2)),
                   ('shift-0.5', lambda p: expit(logit(p)-.5)), ('shift+0.5', lambda p: expit(logit(p)+.5))]:
        variants[nm] = (fn(base[0]), fn(base[1]))
    rows = []
    for nm, (pO, pC) in variants.items():
        L = v7.loss_matrix('logit', st, sets, learner.w, U_O, U_C, pO, pC, test)
        post, plug, cf = v3.decomposition(L, anchor)
        rows.append(dict(part='C1', group=f'{key}/{arm}', seed=seed, variant=nm, H_post=post, H_pre_plugin=plug, H_pre_crossfit=cf, F=post-plug))
    return rows


def task_c2(t):
    dataset, seed, R = t
    st, meta = natural_state_n(dataset, seed, 24)
    ctx = v5.reference_context(st['X_D'], st['y_D'], st['X_C'], st['ead_C'], st['X_Q'], seed)
    ch, learner, _ = v5.choose_all(st, ctx, seed, ablation=False)
    # ---- selections locked ----
    f, cal = phat_for(meta, seed)
    pO, pC = f(meta['X_raw'][meta['O_ids']]), f(meta['X_raw'][meta['C_ids']])
    rr = np.random.default_rng([seed, 11015]); YO = (rr.random((R, 14)) < pO).astype(int); YC = (rr.random((R, 24)) < pC).astype(int)
    model = v8.decompose(st, ctx, learner.w, YO, YC, meta['yT'], ch, seed)
    real = v8.decompose(st, ctx, learner.w, meta['yO'][None, :], meta['yC'][None, :], meta['yT'], ch, seed)  # recorded labels: one realization
    return [dict(part='C2', dataset=dataset, seed=seed, model_reported=model['reported'], model_expost=model['ex_post'],
                 real_reported=real['reported'], real_expost=real['ex_post'], **cal)]


# ------------------------------------------------------------------ D: noise dose and pool size
def task_d(t):
    sub, dataset, seed, level, R = t
    st, meta = natural_state_n(dataset, seed, level if sub == 'pool' else 24, flip=level if sub == 'flip' else 0.)
    ctx = v5.reference_context(st['X_D'], st['y_D'], st['X_C'], st['ead_C'], st['X_Q'], seed)
    sets, idx, anchor, learner = v6.family_and_choices(st, ctx, seed)
    ch, _, _ = v5.choose_all(st, ctx, seed, ablation=False)
    # ---- selections locked ----
    f, cal = phat_for(meta, seed); pO, pC = f(meta['X_raw'][meta['O_ids']]), f(meta['X_raw'][meta['C_ids']])
    yT = meta['yT']; test = lambda p: float(np.mean((p-yT)**2))
    rr = np.random.default_rng([seed, 11017]); U_O, U_C = rr.random((R, len(pO))), rr.random((R, len(pC)))
    L = v7.loss_matrix('logit', st, sets, learner.w, U_O, U_C, pO, pC, test)
    YO = (U_O[:R//2] < pO).astype(int); YC = (U_C[:R//2] < pC).astype(int)
    g = v8.decompose(st, ctx, learner.w, YO, YC, yT, ch, seed)
    return [dict(part='D', sub=sub, dataset=dataset, seed=seed, level=level, noise_index=float(np.mean(pC*(1-pC))),
                 K=len(ctx['M']), **bounds(L, anchor), reported=g['reported'], ex_post=g['ex_post'], deployable=g['deployable'],
                 **{k: v for k, v in g.items() if k.startswith('prac_')})]


def _init(files, cohorts):
    v9._init(*files[:3]); _S['data'] = dict(v9._S['data']); v3._init(files[0], cohorts)


def dispatch(t):
    return {'A': task_a, 'B': task_b, 'C1': task_c1, 'C2': task_c2, 'D': task_d}[t[0]](t[1:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--protocol', type=Path, required=True); ap.add_argument('--freeze', type=Path, required=True)
    ap.add_argument('--data-dir', type=Path, required=True); ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--parts', default='A,B,C1,C2,D'); ap.add_argument('--workers', type=int, default=22); ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    sys.path.insert(0, str(ROOT/'experiments/supplement_v11')); import stats_v11
    protocol = json.loads(args.protocol.read_text()); freeze = json.loads(args.freeze.read_text())
    assert hashlib.sha256(args.protocol.read_bytes()).hexdigest() == freeze['protocol_sha256']
    sources = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCE_PATHS}
    if not args.smoke:
        assert all(freeze['source_sha256'][p] == h for p, h in sources.items()), 'Frozen source changed'
    names = ['real_credit_datasets_authentic.npz', 'diabetes_cohort.npz', 'benchmarks_v9.npz']
    files = [args.data_dir/n for n in names]
    for fp, k in zip(files, ['data_sha256', 'diabetes_npz_sha256', 'benchmarks_npz_sha256']):
        assert hashlib.sha256(fp.read_bytes()).hexdigest() == protocol[k], fp
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    manifest = dict(status='RUNNING', smoke=args.smoke, parts=args.parts, protocol_sha256=freeze['protocol_sha256'], source_sha256=sources,
                    python=platform.python_version(), numpy=np.__version__)
    (out/'run_manifest.json').write_text(json.dumps(manifest, indent=2))
    started = time.time()
    try:
        P = protocol; sm = args.smoke; tasks = []
        S1 = lambda n: [99001] if sm else list(range(P['seed_start'], P['seed_start']+n))
        parts = args.parts.split(',')
        if 'A' in parts:
            tasks += [('A', 'natural', d, s, 4 if sm else P['A']['R']) for d in P['datasets'] for s in S1(P['A']['n_natural'])]
            tasks += [('A', 'known', k, s, 4 if sm else P['A']['R']) for k in P['A']['known_groups'] for s in S1(P['A']['n_known'])]
        if 'B' in parts:
            tasks += [('B', d, s, 4 if sm else P['B']['R']) for d in P['datasets'] for s in S1(P['B']['n'])]
        if 'C1' in parts:
            tasks += [('C1', k, s, 4 if sm else P['C1']['R']) for k in P['C1']['groups'] for s in S1(P['C1']['n'])]
        if 'C2' in parts:
            tasks += [('C2', d, s, 4 if sm else P['C2']['R']) for d in P['datasets'] for s in S1(P['C2']['n'])]
        if 'D' in parts:
            tasks += [('D', 'flip', d, s, lv, 4 if sm else P['D']['R']) for d in P['D']['datasets'] for lv in P['D']['flip'] for s in S1(P['D']['n'])]
            tasks += [('D', 'pool', d, s, lv, 4 if sm else P['D']['R']) for d in P['D']['datasets'] for lv in P['D']['pool'] for s in S1(P['D']['n'])]
        cohorts = {k: v for k, v in P['semi_cohorts'].items()}
        with Pool(args.workers, initializer=_init, initargs=([str(f) for f in files], cohorts)) as pool:
            res = pool.map(dispatch, tasks, chunksize=1)
        df = pd.DataFrame([r for x in res for r in x])
        if 'A' in parts and not sm:
            df = pd.concat([df, pd.DataFrame(stress_tests())], ignore_index=True)
        df.to_csv(out/'rows.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
        if not sm:
            stats_v11.run(df, protocol, out)
        manifest.update(status='COMPLETE', tasks=len(tasks), rows=len(df), elapsed_seconds=time.time()-started)
    except BaseException as exc:
        manifest.update(status='FAILED', error=f'{type(exc).__name__}: {exc}'); raise
    finally:
        (out/'run_manifest.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
