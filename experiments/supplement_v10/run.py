"""supplement-v10: (A) neural downstream learner, (B) the risk budget as a lever on the floor.

A. The supplement-v9 analyses (family protocol with closed-form floor bracket and
   model-free realized check; benchmark-style greedy oracle split into test reuse,
   aleatoric floor and deployable gap) with a one-hidden-layer MLP as the
   downstream learner, trained in GPU batches with a shared initialization per
   state. Cohorts: the eight v9 cohorts plus raw-pixel USPS (256 features, the
   CDALBench representation).
B. Logistic learner (CPU), the same analyses at three risk budgets, sized for
   about 3, 6 and 12 lowest-risk labels, on the eight v9 cohorts. Registered
   prediction: the aleatoric floor increases with the budget.

All selections use visible inputs and are locked before p_hat, recorded labels or
replicates are touched.
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
import prism_engine as eng


def _load(name, rel):
    spec = ilu.spec_from_file_location(name, ROOT/rel); mod = ilu.module_from_spec(spec)
    sys.modules[name] = mod; spec.loader.exec_module(mod); return mod


v9 = _load('v9_run', 'experiments/supplement_v9/run.py')
v8, v7, v6, v3on, v5 = v9.v8, v9.v7, v9.v6, v9.v3on, v9.v5
SOURCE_PATHS = ['experiments/supplement_v10/run.py', 'experiments/supplement_v10/gpu_mlp.py', 'experiments/supplement_v10/stats_v10.py',
                'scripts/prepare_usps_raw.py', 'experiments/supplement_v9/run.py'] + \
               [p for p in v9.SOURCE_PATHS if p not in {'experiments/supplement_v9/run.py', 'experiments/supplement_v9/stats_v9.py'}]
NEW = v9.NEW | {'USPSraw'}
_S = {}


# ------------------------------------------------------------------ shared state construction (CPU)
def build(dataset, seed, target_k):
    X_raw, y_raw, ead_raw = _S['data'][dataset]
    part = v9.natural_state_plain(X_raw, y_raw, seed) if dataset in NEW else v3on.natural_state(X_raw, y_raw, ead_raw, seed)
    r1 = part['rounds'][0]
    st = dict(X_D=part['X_D0'], y_D=part['y_D0'], X_O=r1['X_O'], X_C=r1['X_C'], ead_C=r1['ead_C'], X_Q=part['X_Q'], X_T=part['X_T'])
    ctx = v5.reference_context(st['X_D'], st['y_D'], st['X_C'], st['ead_C'], st['X_Q'], seed, target_k=target_k)
    sets, idx, anchor, learner = v6.family_and_choices(st, ctx, seed)
    ch, _, _ = v5.choose_all(st, ctx, seed, ablation=False)
    # ---- selections locked ----
    used = np.concatenate([part['D0_ids'], part['Q_ids'], part['T_ids'], r1['O_ids'], r1['C_ids']]).astype(int)
    mask = np.ones(len(y_raw), bool); mask[used] = False
    aux = np.random.default_rng([seed, 10003]).permutation(np.flatnonzero(mask))[:v6.AUX_CAP]
    f = v6.fit_phat('gbdt', X_raw[aux], y_raw[aux].astype(int), seed)
    pO, pC = f(X_raw[r1['O_ids']]), f(X_raw[r1['C_ids']])
    return st, ctx, sets, idx, anchor, learner, ch, pO, pC, part['y_T'].astype(float), r1['y_O'].astype(float), r1['y_C'].astype(float)


# ------------------------------------------------------------------ A: MLP on GPU
def prep_a(t):
    dataset, seed, R1, R2 = t
    st, ctx, sets, idx, anchor, learner, ch, pO, pC, yT, yO, yC = build(dataset, seed, 6)
    rr = np.random.default_rng([seed, 10004])
    YO1 = (rr.random((R1, len(pO))) < pO).astype(int); YC1 = (rr.random((R1, len(pC))) < pC).astype(int)
    YO2 = (rr.random((R2, len(pO))) < pO).astype(int); YC2 = (rr.random((R2, len(pC))) < pC).astype(int)
    perm = np.random.default_rng([seed, 8001]).permutation(len(yT)); T1, T2 = perm[:len(yT)//2], perm[len(yT)//2:]
    return dict(dataset=dataset, seed=seed, st=st, q=ctx['q'], B=ctx['B'], M=list(ctx['M']), sets=[list(s) for s in sets], idx=idx,
                anchor=anchor, choices={k: list(v) for k, v in ch.items() if k in v8.PRACTICAL}, pC=pC, yT=yT, yO=yO, yC=yC,
                YO1=YO1, YC1=YC1, YO2=YO2, YC2=YC2, T1=T1, T2=T2, noise_index=float(np.mean(pC*(1-pC))))


def _train(st, S, YO, YC):
    X = np.vstack([st['X_D'], st['X_O'], st['X_C'][S]]) if len(S) else np.vstack([st['X_D'], st['X_O']])
    y = np.concatenate([st['y_D'], YO, YC[S]]) if len(S) else np.concatenate([st['y_D'], YO])
    return X, y


def gpu_state(pk, mlp):
    st, yT, seed = pk['st'], pk['yT'], pk['seed']
    pred = lambda jobs: mlp.predict_many([_train(st, S, YO, YC) for S, YO, YC in jobs], st['X_T'], seed)
    br = lambda P, T: ((P[:, T]-yT[T])**2).mean(axis=1)
    sets, R1 = pk['sets'], len(pk['YO1'])
    # family protocol
    P = pred([(S, pk['YO1'][r], pk['YC1'][r]) for r in range(R1) for S in sets])
    L = br(P, np.arange(len(yT))).reshape(R1, len(sets))
    rows = []
    v7.score(rows, dict(dataset=pk['dataset'], seed=seed, part='family', noise_index=pk['noise_index']), 'mlp', L, pk['anchor'], pk['idx'])
    Pr = pred([(S, pk['yO'], pk['yC']) for S in sets]); Lr = br(Pr, np.arange(len(yT)))
    rows[-1]['H_post_realized'] = float(Lr[pk['anchor']]-Lr.min())
    # greedy oracle
    q, ead, B, M, K, nC = pk['q'], st['ead_C'], pk['B'], pk['M'], len(pk['M']), len(pk['pC'])
    T1, T2, YO2, YC2 = pk['T1'], pk['T2'], pk['YO2'], pk['YC2']; R2 = len(YO2)
    feas = lambda S, i: i not in S and eng.bernstein_envelope(q[S+[i]], ead[S+[i]]) <= B+1e-12

    def greedy(reps):
        """Shared set if reps is a list of replicate indices evaluated jointly; returns final set."""
        S = []; cur = br(pred([(S, YO2[r], YC2[r]) for r in reps]), T1).mean()
        while len(S) < K:
            cand = [i for i in range(nC) if feas(S, i)]
            if not cand: break
            v = br(pred([(S+[i], YO2[r], YC2[r]) for i in cand for r in reps]), T1).reshape(len(cand), len(reps)).mean(axis=1)
            j = int(np.argmin(v))
            if v[j] >= cur-1e-12: break
            S, cur = S+[cand[j]], v[j]
        return S

    def greedy_each():
        """Independent ex-post oracle per replicate, batched across replicates."""
        Ss = [[] for _ in range(R2)]; cur = br(pred([([], YO2[r], YC2[r]) for r in range(R2)]), T1); active = list(range(R2))
        while active:
            jobs, owner = [], []
            for r in active:
                if len(Ss[r]) >= K: continue
                for i in range(nC):
                    if feas(Ss[r], i): jobs.append((Ss[r]+[i], YO2[r], YC2[r])); owner.append((r, i))
            if not jobs: break
            v = br(pred(jobs), T1); nxt = []
            for r in set(o[0] for o in owner):
                k = [n for n, o in enumerate(owner) if o[0] == r]; b = k[int(np.argmin(v[k]))]
                if v[b] < cur[r]-1e-12:
                    Ss[r] = Ss[r]+[owner[b][1]]; cur[r] = v[b]; nxt.append(r)
            active = nxt
        return Ss

    Opost = greedy_each()
    halves = [list(range(0, R2, 2)), list(range(1, R2, 2))]
    Odep = [greedy(halves[0]), greedy(halves[1])]
    jobs = [(M, YO2[r], YC2[r]) for r in range(R2)] + [(Opost[r], YO2[r], YC2[r]) for r in range(R2)]
    jobs += [(Odep[1-h], YO2[r], YC2[r]) for h in (0, 1) for r in halves[h]]
    pnames = list(pk['choices']); jobs += [(pk['choices'][p], YO2[r], YC2[r]) for p in pnames for r in range(R2)]
    P = pred(jobs)
    LM1, LM2 = br(P[:R2], T1), br(P[:R2], T2); LO1, LO2 = br(P[R2:2*R2], T1), br(P[R2:2*R2], T2)
    o = 2*R2; dep = []
    for h in (0, 1):
        n = len(halves[h]); dep.append(np.mean(LM2[halves[h]]-br(P[o:o+n], T2))); o += n
    prac = {}
    for p in pnames:
        prac[p] = float(np.mean(LM2-br(P[o:o+R2], T2))); o += R2
    rows.append(dict(dataset=pk['dataset'], seed=seed, part='oracle', learner='mlp', noise_index=pk['noise_index'],
                     reported=float(np.mean(LM1-LO1)), ex_post=float(np.mean(LM2-LO2)), deployable=float(np.mean(dep)),
                     oracle_k_post=float(np.mean([len(s) for s in Opost])), best_practical=max(prac.values()),
                     **{f'prac_{k}': v for k, v in prac.items()}))
    return rows


# ------------------------------------------------------------------ B: budget lever (CPU, logistic)
def task_b(t):
    dataset, seed, target_k, R1, R2 = t
    st, ctx, sets, idx, anchor, learner, ch, pO, pC, yT, yO, yC = build(dataset, seed, target_k)
    test = lambda p: float(np.mean((p-yT)**2))
    base = dict(dataset=dataset, seed=seed, target_k=target_k, K=len(ctx['M']), noise_index=float(np.mean(pC*(1-pC))),
                selected_label_var=float(np.mean(pC[ctx['M']]*(1-pC[ctx['M']]))) if len(ctx['M']) else np.nan)
    rows = []
    rr = np.random.default_rng([seed, 10005]); U_O, U_C = rr.random((R1, len(pO))), rr.random((R1, len(pC)))
    v7.score(rows, dict(base, part='family'), 'logit', v7.loss_matrix('logit', st, sets, learner.w, U_O, U_C, pO, pC, test), anchor, idx)
    rr = np.random.default_rng([seed, 10006])
    YO = (rr.random((R2, len(pO))) < pO).astype(int); YC = (rr.random((R2, len(pC))) < pC).astype(int)
    rows.append(dict(base, part='oracle', learner='logit', **v8.decompose(st, ctx, learner.w, YO, YC, yT, ch, seed)))
    return rows


def _init(files):
    v9._init(*files[:3]); _S['data'] = dict(v9._S['data'])
    u = np.load(files[3]); _S['data']['USPSraw'] = (u['USPSraw_X'].astype(float), u['USPSraw_y'].astype(int), u['USPSraw_ead'].astype(float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--protocol', type=Path, required=True); ap.add_argument('--freeze', type=Path, required=True)
    ap.add_argument('--data-dir', type=Path, required=True); ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--part', choices=['A', 'B'], required=True)
    ap.add_argument('--workers', type=int, default=22); ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    import stats_v10
    protocol = json.loads(args.protocol.read_text()); freeze = json.loads(args.freeze.read_text())
    assert hashlib.sha256(args.protocol.read_bytes()).hexdigest() == freeze['protocol_sha256']
    sources = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCE_PATHS}
    if not args.smoke:
        assert all(freeze['source_sha256'][p] == h for p, h in sources.items()), 'Frozen source changed'
    names = ['real_credit_datasets_authentic.npz', 'diabetes_cohort.npz', 'benchmarks_v9.npz', 'usps_raw.npz']
    keys = ['data_sha256', 'diabetes_npz_sha256', 'benchmarks_npz_sha256', 'usps_raw_npz_sha256']
    files = [args.data_dir/n for n in names]
    for fpath, key in zip(files, keys):
        assert hashlib.sha256(fpath.read_bytes()).hexdigest() == protocol[key], fpath
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    manifest = dict(status='RUNNING', part=args.part, smoke=args.smoke, protocol_sha256=freeze['protocol_sha256'], source_sha256=sources,
                    python=platform.python_version(), numpy=np.__version__)
    (out/'run_manifest.json').write_text(json.dumps(manifest, indent=2))
    started = time.time()
    try:
        P = protocol[f'part_{args.part}']
        seeds = [99001] if args.smoke else range(P['seed_start'], P['seed_start']+P['n_seeds'])
        if args.part == 'A':
            import torch, gpu_mlp
            manifest.update(torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0))
            R1, R2 = (4, 4) if args.smoke else (P['family_resamples'], P['oracle_resamples'])
            tasks = [(d, s, R1, R2) for d in P['datasets'] for s in seeds]
            with Pool(args.workers, initializer=_init, initargs=([str(f) for f in files],)) as pool:
                rows = []
                for n, pk in enumerate(pool.imap(prep_a, tasks, chunksize=1)):
                    rows += gpu_state(pk, gpu_mlp)
                    if n % 16 == 0:
                        (out/'progress.txt').write_text(f'{n+1}/{len(tasks)} states, {time.time()-started:.0f}s\n')
        else:
            R1, R2 = (4, 4) if args.smoke else (P['family_resamples'], P['oracle_resamples'])
            tasks = [(d, s, k, R1, R2) for d in P['datasets'] for k in P['target_k'] for s in seeds]
            with Pool(args.workers, initializer=_init, initargs=([str(f) for f in files],)) as pool:
                rows = [r for x in pool.map(task_b, tasks, chunksize=1) for r in x]
        df = pd.DataFrame(rows); df.to_csv(out/'states.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
        if not args.smoke:
            stats_v10.run(args.part, df, protocol, out)
        manifest.update(status='COMPLETE', tasks=len(tasks), rows=len(df), elapsed_seconds=time.time()-started)
    except BaseException as exc:
        manifest.update(status='FAILED', error=f'{type(exc).__name__}: {exc}'); raise
    finally:
        (out/'run_manifest.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
