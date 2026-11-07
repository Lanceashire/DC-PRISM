"""supplement-v5 study H: power-driven delayed replay (review M5/M6).

Identical round logic to supplement-v3 study D (verified: the inherited loop
must reproduce archived v3 ALC-Brier on seeds 51001/51016 exactly), with
512 fresh split seeds per cohort and stream, chosen from the v3 paired-SD power
calculation, and three added per-round outputs: test ROC-AUC, test log-loss,
and the realised adverse exposure of the acquired set next to its envelope.
"""
import argparse
import hashlib
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
sys.path.insert(0, str(ROOT/'experiments/supplement_v5'))
import prism_engine as eng
import run_p3_p4 as runner
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location('v3_run_online', ROOT/'experiments/supplement_v3/run_online.py')
v3on = _ilu.module_from_spec(_spec); sys.modules['v3_run_online'] = v3on; _spec.loader.exec_module(v3on)

SOURCE_PATHS = ['experiments/supplement_v5/run_online.py', 'experiments/supplement_v5/stats_v5.py'] + v3on.SOURCE_PATHS
POLICIES = v3on.POLICIES
ALPHA = 0.10


def logloss(y, p):
    p = np.clip(p, 1e-6, 1-1e-6)
    return float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))


def replay(task):
    stream, dataset, seed, policies = task
    X_raw, y_raw, ead_raw = v3on._DATA[dataset]
    part = (runner.build_strictly_hidden_y_free_real_state(X_raw, y_raw, ead_raw, seed=seed, n_rounds=10)
            if stream == 'engineered' else v3on.natural_state(X_raw, y_raw, ead_raw, seed))
    X_Q, X_T, y_T = part['X_Q'], part['X_T'], part['y_T']
    ref = eng.UnifiedLogisticLearner(l2=0.45).fit(part['X_D0'], part['y_D0'])
    budgets = {}
    for r in part['rounds']:
        q_ref, _ = eng.compute_calibrated_upper_risk(ref, part['X_D0'], part['y_D0'], r['X_C'], delta_r=0.10,
                        risk_regime='Calibrated', rng=np.random.default_rng(seed*50+r['round']))
        budgets[r['round']] = eng.compute_exogenous_budget(q_ref, r['ead_C'], alpha=ALPHA, target_k=6)[0]
    rows = []
    for policy in policies:
        X_D, y_D = part['X_D0'].copy(), part['y_D0'].copy()
        seen = set(int(x) for x in part['D0_ids']); queue = []
        for r in part['rounds']:
            t = r['round']; B = budgets[t]
            keep = []
            for rel, X_b, y_b, ids_b in queue:
                if t >= rel:
                    X_D = np.vstack([X_D, X_b]); y_D = np.concatenate([y_D, y_b]); seen.update(int(i) for i in ids_b)
                else:
                    keep.append((rel, X_b, y_b, ids_b))
            queue = keep
            assert not (set(int(i) for i in r['C_ids']) & seen)
            X_O = np.vstack([r['X_O']]+[b[1] for b in queue]) if queue else r['X_O']
            learner = eng.UnifiedLogisticLearner(l2=0.45).fit(X_D, y_D)
            worlds = eng.sample_common_parameter_worlds(learner, n_worlds=16, rng=np.random.default_rng(seed*100+t))
            q, _ = eng.compute_calibrated_upper_risk(learner, X_D, y_D, r['X_C'], delta_r=0.10,
                        risk_regime='Calibrated', rng=np.random.default_rng(seed*200+t))
            t0 = time.perf_counter()
            s = v3on.select(policy, seed, t, learner, X_D, y_D, X_O, r, X_Q, B, q, worlds)
            sec = time.perf_counter()-t0
            env = float(eng.bernstein_envelope(q[s], r['ead_C'][s])) if len(s) else 0.
            assert len(set(s.tolist())) == len(s) and env <= B+1e-9
            p_T = learner.predict_proba(X_T)
            queue.append((t+2, r['X_O'], r['y_O'], r['O_ids']))
            if len(s):
                queue.append((t+1, r['X_C'][s], r['y_C'][s], r['C_ids'][s]))
            rows.append(dict(stream=stream, dataset=dataset, seed=seed, policy=policy, round=t,
                             brier_t=eng.brier_score(y_T, p_T), auc_t=float(eng.roc_auc_score_np(y_T, p_T)),
                             logloss_t=logloss(y_T, p_T), k=len(s), B=B, envelope=env,
                             realized_loss=float(np.sum(r['y_C'][s]*r['ead_C'][s])),
                             adverse_count=int(np.sum(r['y_C'][s])), seconds=sec, n_train=len(y_D)))
    return rows


def per_seed_metrics(traj):
    traj = traj.assign(env_ratio=np.where(traj.B > 0, traj.envelope/traj.B, 0.))
    return traj.sort_values('round').groupby(['stream', 'dataset', 'seed', 'policy'], as_index=False).agg(
        alc_brier=('brier_t', 'mean'), alc_auc=('auc_t', 'mean'), alc_logloss=('logloss_t', 'mean'),
        round10_brier=('brier_t', 'last'), labels_per_round=('k', 'mean'),
        exploration_loss=('realized_loss', 'sum'), adverse_labels=('adverse_count', 'sum'),
        max_envelope_ratio=('env_ratio', 'max'), select_seconds=('seconds', 'mean'))


def reproduce_check(datafile, workers):
    archived = pd.read_csv(ROOT/'results/supplement_v3/online_seed_metrics.csv')
    tasks = [(st, d, s, POLICIES) for st in ['engineered', 'natural'] for d in ['HMEQ', 'LendingClub', 'PPDai', 'UCI_Credit'] for s in (51001, 51016)]
    with Pool(workers, initializer=v3on._init, initargs=(datafile,)) as pool:
        rows = [r for res in pool.map(replay, tasks) for r in res]
    mine = per_seed_metrics(pd.DataFrame(rows))
    m = mine.merge(archived, on=['stream', 'dataset', 'seed', 'policy'], suffixes=('', '_v3'))
    assert len(m) == len(mine) == len(tasks)*len(POLICIES)
    worst = float(np.max(np.abs(m.alc_brier-m.alc_brier_v3)))
    return dict(trajectories=len(mine), max_abs_alc_difference=worst, status='PASS' if worst < 1e-12 else 'FAIL')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--protocol', type=Path, required=True); ap.add_argument('--freeze', type=Path, required=True)
    ap.add_argument('--data-dir', type=Path, required=True); ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=22); ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    import stats_v5
    protocol = json.loads(args.protocol.read_text()); freeze = json.loads(args.freeze.read_text())
    assert hashlib.sha256(args.protocol.read_bytes()).hexdigest() == freeze['protocol_sha256']
    sources = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCE_PATHS}
    if not args.smoke:
        assert all(freeze['source_sha256'][p] == h for p, h in sources.items()), 'Frozen source changed'
    datafile = str(args.data_dir/'real_credit_datasets_authentic.npz')
    assert hashlib.sha256(Path(datafile).read_bytes()).hexdigest() == protocol['data_sha256']
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    manifest = dict(status='RUNNING', study='H', smoke=args.smoke, protocol_sha256=freeze['protocol_sha256'],
                    source_sha256=sources, python=platform.python_version(), numpy=np.__version__)
    (out/'run_manifest.json').write_text(json.dumps(manifest, indent=2))
    started = time.time()
    try:
        check = reproduce_check(datafile, args.workers)
        manifest['reproduce_check'] = check
        assert check['status'] == 'PASS', check
        on = protocol['study_H']
        seeds = [99001, 99002] if args.smoke else list(range(on['seed_start'], on['seed_start']+on['n_seeds']))
        tasks = [(st, d, s, POLICIES) for st in on['streams'] for d in on['datasets'] for s in seeds]
        with Pool(args.workers, initializer=v3on._init, initargs=(datafile,)) as pool:
            rows = [r for res in pool.map(replay, tasks, chunksize=2) for r in res]
        traj = pd.DataFrame(rows)
        traj.to_csv(out/'study_H_trajectories.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
        per_seed = per_seed_metrics(traj)
        per_seed.to_csv(out/'study_H_seed_metrics.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
        first = traj[traj['round'] == 1].groupby(['stream', 'dataset', 'seed']).brier_t.nunique()
        assert (first == 1).all()
        if not args.smoke:
            stats_v5.run('H', (per_seed, traj), protocol, out)
        manifest.update(status='COMPLETE', trajectories=len(per_seed), rows=len(traj), elapsed_seconds=time.time()-started)
    except BaseException as exc:
        manifest.update(status='FAILED', error=f'{type(exc).__name__}: {exc}'); raise
    finally:
        (out/'run_manifest.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
