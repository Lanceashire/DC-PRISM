"""supplement-v3 study D: ten-round delayed replay with additional baselines,
on both the original engineered candidate stream and a natural (uniformly
random) stream.

The round logic replicates src/run_p3_p4.py exactly (budgets from the D0
reference model, S released after one round, O after two, Brier measured at
round start before current labels are visible). `--reproduce-check` verifies
that this loop reproduces the archived p4-confirmation-v1 ALC values for the
inherited policies before any new result is interpreted.

Engineered stream: the package builder, which ranks observable covariates to
create low-risk, pending-aligned, complementary and tail candidate groups and
a narrow initial history. Natural stream: same T/Q/history-window split sizes
and the same D0-fitted preprocessing, but D0 is the first 68 rows of the random
history window, and each round's 14 ordinary acquisitions and 24 candidates are
consecutive rows of a random permutation of the stream. No covariate ranking.
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
import prism_engine as eng
import run_p3_p4 as runner
import baselines as bl

SOURCE_PATHS = ['experiments/supplement_v3/run_online.py', 'experiments/supplement_v3/baselines.py',
                'experiments/supplement_v3/stats.py', 'src/prism_engine.py', 'src/run_p3_p4.py']
PACKAGE = {'OrdinaryOnly': 'OrdinaryOnly', 'Safe-MinRisk': 'Safe-MinRisk', 'Fixed-count': 'Fixed-Eps-Safe-PA',
           'Fisher-cost': 'BAIT-Cost-Safe', 'DC-PRISM': 'DC-PRISM'}
NEW = ['Random-Feasible', 'Uncertainty', 'BAIT-Greedy', 'TargetVar-Greedy']
POLICIES = list(PACKAGE) + NEW
ALPHA = 0.10
_DATA = {}


def natural_state(X_raw, y_raw, ead_raw, seed, n_rounds=10):
    rng = np.random.default_rng(seed)
    n = len(X_raw); perm = rng.permutation(n)
    n_T = min(max(int(n*0.18), 500), 1000); n_Q = min(max(int(n*0.12), 350), 600)
    ids_T, ids_Q, pool = perm[:n_T], perm[n_T:n_T+n_Q], perm[n_T+n_Q:]
    ids_hist, ids_stream = pool[:600], pool[600:]
    ids_D0 = ids_hist[:68]
    prep = eng.SplitBeforePreprocessor(tail_scale=1.85).fit(X_raw[ids_D0], ead_raw[ids_D0])
    prep.stds = np.maximum(prep.stds, 1e-3); prep.tau_aux = np.maximum(prep.tau_aux, np.array([1.35, 1.35]))
    stream = rng.permutation(ids_stream)
    assert len(stream) >= 38*n_rounds
    rounds = []
    for t in range(n_rounds):
        o = stream[t*38:t*38+14]; c = stream[t*38+14:(t+1)*38]
        rounds.append({'round': t+1, 'O_ids': o, 'X_O': prep.transform_X(X_raw[o]), 'y_O': y_raw[o].astype(int),
                       'C_ids': c, 'X_C': prep.transform_X(X_raw[c]), 'y_C': y_raw[c].astype(int),
                       'ead_C': prep.transform_ead(ead_raw[c])})
    return {'D0_ids': ids_D0, 'X_D0': prep.transform_X(X_raw[ids_D0]), 'y_D0': y_raw[ids_D0].astype(int),
            'Q_ids': ids_Q, 'X_Q': prep.transform_X(X_raw[ids_Q]),
            'T_ids': ids_T, 'X_T': prep.transform_X(X_raw[ids_T]), 'y_T': y_raw[ids_T].astype(int), 'rounds': rounds}


def select(policy, seed, t, learner, X_D, y_D, X_O, r, X_Q, B, q, worlds):
    """Visible inputs only: history labels, pending/candidate/target covariates, q, B."""
    X_C, ead = r['X_C'], r['ead_C']
    if policy in PACKAGE:
        s, _, _ = eng.select_acquisition_policy(PACKAGE[policy], X_D, y_D, X_O, X_C, ead, X_Q, B,
                    learner=learner, alpha=ALPHA, lambda_K=0.22, q_C_precomputed=q, w_worlds_precomputed=worlds,
                    exact_max_n=14, rng=np.random.default_rng(seed*300+t))
        return np.asarray(s, int)
    if policy == 'Random-Feasible':
        return bl.random_feasible(X_C, q, ead, B, np.random.default_rng(seed*400+t))
    if policy == 'Uncertainty':
        return bl.uncertainty_greedy(learner, X_C, q, ead, B)
    if policy == 'BAIT-Greedy':
        return bl.bait_greedy(learner, X_D, X_O, X_C, X_Q, q, ead, B)
    if policy == 'TargetVar-Greedy':
        return bl.targetvar_greedy(learner, X_D, X_O, X_C, X_Q, q, ead, B)
    raise ValueError(policy)


def replay(task):
    stream, dataset, seed, policies = task
    X_raw, y_raw, ead_raw = _DATA[dataset]
    part = (runner.build_strictly_hidden_y_free_real_state(X_raw, y_raw, ead_raw, seed=seed, n_rounds=10)
            if stream == 'engineered' else natural_state(X_raw, y_raw, ead_raw, seed))
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
            s = select(policy, seed, t, learner, X_D, y_D, X_O, r, X_Q, B, q, worlds)
            env = float(eng.bernstein_envelope(q[s], r['ead_C'][s])) if len(s) else 0.
            assert len(set(s.tolist())) == len(s) and env <= B+1e-9
            brier = eng.brier_score(y_T, learner.predict_proba(X_T))
            queue.append((t+2, r['X_O'], r['y_O'], r['O_ids']))
            if len(s):
                queue.append((t+1, r['X_C'][s], r['y_C'][s], r['C_ids'][s]))
            rows.append(dict(stream=stream, dataset=dataset, seed=seed, policy=policy, round=t, brier_t=brier,
                             k=len(s), B=B, envelope=env, realized_loss=float(np.sum(r['y_C'][s]*r['ead_C'][s])),
                             n_train=len(y_D)))
    return rows


def _init(datafile):
    a = np.load(datafile, allow_pickle=False)
    for d in ['HMEQ', 'LendingClub', 'PPDai', 'UCI_Credit']:
        _DATA[d] = (a[d+'_X'].astype(float), a[d+'_y'].astype(int), a[d+'_ead'].astype(float))


def per_seed_metrics(traj):
    traj = traj.assign(env_ratio=np.where(traj.B > 0, traj.envelope/traj.B, 0.))
    return traj.sort_values('round').groupby(['stream', 'dataset', 'seed', 'policy'], as_index=False).agg(
        alc_brier=('brier_t', 'mean'), round10_brier=('brier_t', 'last'), labels_per_round=('k', 'mean'),
        max_envelope_ratio=('env_ratio', 'max'))


def reproduce_check(datafile, workers):
    """Inherited policies on two archived p4-confirmation-v1 seeds must match exactly."""
    archived = pd.read_csv(ROOT/'results/p4_confirmation_v1/seed_metrics.csv')
    tasks = [('engineered', d, s, list(PACKAGE)) for d in ['HMEQ', 'LendingClub', 'PPDai', 'UCI_Credit'] for s in (21001, 21016)]
    with Pool(workers, initializer=_init, initargs=(datafile,)) as pool:
        rows = [r for res in pool.map(replay, tasks) for r in res]
    mine = per_seed_metrics(pd.DataFrame(rows))
    worst = 0.
    for _, m in mine.iterrows():
        a = archived[(archived.dataset == m.dataset) & (archived.seed == m.seed) & (archived.policy == PACKAGE[m.policy])]
        worst = max(worst, abs(float(a.alc_brier.iloc[0])-m.alc_brier))
    return dict(trajectories=len(mine), max_abs_alc_difference=worst, status='PASS' if worst < 1e-12 else 'FAIL')


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
    datafile = str(args.data_dir/'real_credit_datasets_authentic.npz')
    assert hashlib.sha256(Path(datafile).read_bytes()).hexdigest() == protocol['data_sha256']
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    manifest = dict(status='RUNNING', study='online D', protocol_sha256=freeze['protocol_sha256'], source_sha256=sources,
                    python=platform.python_version(), numpy=np.__version__)
    (out/'run_manifest.json').write_text(json.dumps(manifest, indent=2))
    started = time.time()
    try:
        check = reproduce_check(datafile, args.workers)
        manifest['reproduce_check'] = check
        assert check['status'] == 'PASS', check
        on = protocol['online']
        tasks = [(st, d, s, POLICIES) for st in on['streams'] for d in on['datasets'] for s in on['seeds']]
        with Pool(args.workers, initializer=_init, initargs=(datafile,)) as pool:
            rows = [r for res in pool.map(replay, tasks, chunksize=1) for r in res]
        traj = pd.DataFrame(rows)
        traj.to_csv(out/'online_trajectories.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
        per_seed = per_seed_metrics(traj)
        per_seed.to_csv(out/'online_seed_metrics.csv', index=False)
        assert len(traj) == len(tasks)*len(POLICIES)*10
        first = traj[traj['round'] == 1].groupby(['stream', 'dataset', 'seed']).brier_t.nunique()
        assert (first == 1).all(), 'Round-1 Brier must be identical across policies (no label visible yet)'
        st_mod.online_statistics(per_seed, protocol, out)
        manifest.update(status='COMPLETE', trajectories=len(per_seed), rows=len(traj), elapsed_seconds=time.time()-started)
    except BaseException as exc:
        manifest.update(status='FAILED', error=f'{type(exc).__name__}: {exc}'); raise
    finally:
        (out/'run_manifest.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
