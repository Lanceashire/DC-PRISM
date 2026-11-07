"""supplement-v9: the oracle-gap decomposition along a label-noise gradient.

Eight cohorts on the natural (random) stream, one pipeline: four credit cohorts and
the hospital-readmission cohort (original replay preprocessing), and three
lower-noise benchmark cohorts (USPS digits >= 5, used by CDALBench; Spambase;
Bank Marketing), for which preprocessing is plain z-scoring fitted on the initial
history (no credit-specific auxiliary-tail hinge) and exposure is 1.

Per round-1 state: (i) family protocol with logistic and boosting learners,
boosting p_hat on disjoint records, model-free realized hindsight check and the
closed-form floor bracket (supplement-v7 code); (ii) benchmark-style greedy oracle
gap split into test reuse, aleatoric floor and deployable gap (supplement-v8 code);
(iii) a label-noise index, the mean of p_hat(1-p_hat) over the candidates.
Registered prediction: the floor's share of the oracle gap is smaller on the
lower-noise cohorts and increases with the noise index across cohorts.
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


def _load(name, rel):
    spec = ilu.spec_from_file_location(name, ROOT/rel); mod = ilu.module_from_spec(spec)
    sys.modules[name] = mod; spec.loader.exec_module(mod); return mod


v8 = _load('v8_run', 'experiments/supplement_v8/run.py')
v7, v6, v3on, v5 = v8.v7, v8.v6, v8.v3on, v8.v5
SOURCE_PATHS = ['experiments/supplement_v9/run.py', 'experiments/supplement_v9/stats_v9.py', 'scripts/prepare_benchmarks.py',
                'experiments/supplement_v8/run.py'] + [p for p in v8.SOURCE_PATHS if p not in {'experiments/supplement_v8/run.py', 'experiments/supplement_v8/stats_v8.py'}]
NEW = {'USPS', 'Spambase', 'BankMarketing'}
_S = {}


def natural_state_plain(X_raw, y_raw, seed, n_rounds=10):
    """v3 natural stream with plain z-scoring fitted on D0 and unit exposure (new cohorts only)."""
    rng = np.random.default_rng(seed)
    n = len(X_raw); perm = rng.permutation(n)
    n_T = min(max(int(n*0.18), 500), 1000); n_Q = min(max(int(n*0.12), 350), 600)
    ids_T, ids_Q, pool = perm[:n_T], perm[n_T:n_T+n_Q], perm[n_T+n_Q:]
    ids_hist, ids_stream = pool[:600], pool[600:]
    ids_D0 = ids_hist[:68]
    mu = X_raw[ids_D0].mean(axis=0); sd = X_raw[ids_D0].std(axis=0); sd = np.where(sd < 1e-6, 1., sd)
    tf = lambda Z: np.clip((Z-mu)/sd, -4.5, 4.5)
    stream = rng.permutation(ids_stream); rounds = []
    for t in range(n_rounds):
        o = stream[t*38:t*38+14]; c = stream[t*38+14:(t+1)*38]
        rounds.append({'round': t+1, 'O_ids': o, 'X_O': tf(X_raw[o]), 'y_O': y_raw[o].astype(int),
                       'C_ids': c, 'X_C': tf(X_raw[c]), 'y_C': y_raw[c].astype(int), 'ead_C': np.ones(len(c))})
    return {'D0_ids': ids_D0, 'X_D0': tf(X_raw[ids_D0]), 'y_D0': y_raw[ids_D0].astype(int), 'Q_ids': ids_Q, 'X_Q': tf(X_raw[ids_Q]),
            'T_ids': ids_T, 'X_T': tf(X_raw[ids_T]), 'y_T': y_raw[ids_T].astype(int), 'rounds': rounds}


def task(t):
    dataset, seed, R1, R2 = t
    X_raw, y_raw, ead_raw = _S['data'][dataset]
    part = natural_state_plain(X_raw, y_raw, seed) if dataset in NEW else v3on.natural_state(X_raw, y_raw, ead_raw, seed)
    r1 = part['rounds'][0]
    st = dict(X_D=part['X_D0'], y_D=part['y_D0'], X_O=r1['X_O'], X_C=r1['X_C'], ead_C=r1['ead_C'], X_Q=part['X_Q'], X_T=part['X_T'])
    ctx = v5.reference_context(st['X_D'], st['y_D'], st['X_C'], st['ead_C'], st['X_Q'], seed)
    sets, idx, anchor, learner = v6.family_and_choices(st, ctx, seed)
    ch, _, _ = v5.choose_all(st, ctx, seed, ablation=False)
    # ---- all selections locked ----
    used = np.concatenate([part['D0_ids'], part['Q_ids'], part['T_ids'], r1['O_ids'], r1['C_ids']]).astype(int)
    mask = np.ones(len(y_raw), bool); mask[used] = False
    aux = np.random.default_rng([seed, 9003]).permutation(np.flatnonzero(mask))[:v6.AUX_CAP]
    f = v6.fit_phat('gbdt', X_raw[aux], y_raw[aux].astype(int), seed)
    pO, pC = f(X_raw[r1['O_ids']]), f(X_raw[r1['C_ids']])
    yT = part['y_T'].astype(float); test = lambda p: float(np.mean((p-yT)**2))
    base = dict(dataset=dataset, seed=seed, K=len(ctx['M']), family_size=len(sets), noise_index=float(np.mean(pC*(1-pC))),
                adverse_rate=float(y_raw.mean()))
    rows = []
    rr = np.random.default_rng([seed, 9004]); U_O, U_C = rr.random((R1, len(pO))), rr.random((R1, len(pC)))
    yO, yC = r1['y_O'].astype(float), r1['y_C'].astype(float)
    for kind in ['logit', 'gbdt']:
        v7.score(rows, dict(base, part='family'), kind, v7.loss_matrix(kind, st, sets, learner.w, U_O, U_C, pO, pC, test), anchor, idx)
        Lr = v7.loss_matrix(kind, st, sets, learner.w, np.zeros((1, len(yO))), np.zeros((1, len(yC))), yO, yC, test)
        rows[-1]['H_post_realized'] = float(Lr[0, anchor]-Lr[0].min())
    rr = np.random.default_rng([seed, 9005])
    YO = (rr.random((R2, len(pO))) < pO).astype(int); YC = (rr.random((R2, len(pC))) < pC).astype(int)
    rows.append(dict(base, part='oracle', learner='logit', **v8.decompose(st, ctx, learner.w, YO, YC, yT, ch, seed)))
    return rows


def _init(credit_file, extra_file, bench_file):
    a = np.load(credit_file); b = np.load(extra_file); c = np.load(bench_file); D = {}
    for k in ['HMEQ', 'LendingClub', 'PPDai', 'UCI_Credit']:
        D[k] = (a[k+'_X'].astype(float), a[k+'_y'].astype(int), a[k+'_ead'].astype(float))
    D['Diabetes'] = (b['Diabetes_X'].astype(float), b['Diabetes_y'].astype(int), b['Diabetes_ead'].astype(float))
    for k in NEW:
        D[k] = (c[k+'_X'].astype(float), c[k+'_y'].astype(int), c[k+'_ead'].astype(float))
    _S['data'] = D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--protocol', type=Path, required=True); ap.add_argument('--freeze', type=Path, required=True)
    ap.add_argument('--data-dir', type=Path, required=True); ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=22); ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    sys.path.insert(0, str(ROOT/'experiments/supplement_v9')); import stats_v9
    protocol = json.loads(args.protocol.read_text()); freeze = json.loads(args.freeze.read_text())
    assert hashlib.sha256(args.protocol.read_bytes()).hexdigest() == freeze['protocol_sha256']
    sources = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCE_PATHS}
    if not args.smoke:
        assert all(freeze['source_sha256'][p] == h for p, h in sources.items()), 'Frozen source changed'
    files = [args.data_dir/'real_credit_datasets_authentic.npz', args.data_dir/'diabetes_cohort.npz', args.data_dir/'benchmarks_v9.npz']
    for fpath, key in zip(files, ['data_sha256', 'diabetes_npz_sha256', 'benchmarks_npz_sha256']):
        assert hashlib.sha256(fpath.read_bytes()).hexdigest() == protocol[key], fpath
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    manifest = dict(status='RUNNING', smoke=args.smoke, protocol_sha256=freeze['protocol_sha256'], source_sha256=sources,
                    python=platform.python_version(), numpy=np.__version__)
    (out/'run_manifest.json').write_text(json.dumps(manifest, indent=2))
    started = time.time()
    try:
        P = protocol['study']
        seeds = [99001] if args.smoke else range(P['seed_start'], P['seed_start']+P['n_seeds'])
        R1, R2 = (4, 4) if args.smoke else (P['family_resamples'], P['oracle_resamples'])
        tasks = [(d, s, R1, R2) for d in P['datasets'] for s in seeds]
        with Pool(args.workers, initializer=_init, initargs=tuple(str(f) for f in files)) as pool:
            res = pool.map(task, tasks, chunksize=1)
        df = pd.DataFrame([r for x in res for r in x]); df.to_csv(out/'states.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
        if not args.smoke:
            stats_v9.run(df, protocol, out)
        manifest.update(status='COMPLETE', tasks=len(tasks), rows=len(df), elapsed_seconds=time.time()-started)
    except BaseException as exc:
        manifest.update(status='FAILED', error=f'{type(exc).__name__}: {exc}'); raise
    finally:
        (out/'run_manifest.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
