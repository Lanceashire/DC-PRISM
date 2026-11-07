"""Independent checks for supplement-v3 results.

Public mode (no data): recomputes all registered statistics from the saved
rows with an independent implementation and checks structural invariants.
With --data-dir: additionally replays the first and last state of every
offline cohort and the first/last seed of every online stream x dataset, and
requires exact agreement of selections and losses.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'experiments/supplement_v3'))


def recompute(table, vectors, alpha):
    m = len(table); worst = 0.; holm = []
    for (_, row), x in zip(table.iterrows(), vectors):
        n = len(x); se = np.std(x, ddof=1)/np.sqrt(n); c = sps.t.isf(alpha/(2*m), n-1)
        worst = max(worst, abs(np.mean(x)-row['mean']), abs(np.mean(x)-c*se-row.simultaneous_low),
                    abs(np.mean(x)+c*se-row.simultaneous_high))
        holm.append(sps.ttest_1samp(x, 0).pvalue if np.std(x) > 1e-15 else 1.)
    order = np.argsort(holm); adj = np.empty(m); run = 0.
    for r, i in enumerate(order):
        run = max(run, min(1, holm[i]*(m-r))); adj[i] = run
    return max(worst, float(np.max(np.abs(adj-table.p_holm.to_numpy()))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', type=Path, required=True); ap.add_argument('--data-dir', type=Path)
    args = ap.parse_args(); res = args.results
    protocol = json.loads((ROOT/'configs/supplement_v3.json').read_text()); alpha = protocol['inference']['alpha']
    report = {}
    df = pd.read_csv(res/'offline_policy_results.csv.gz')
    report['offline_rows'] = len(df)
    report['offline_envelope_feasible'] = bool((df.envelope <= df.B+1e-10).all())
    report['offline_H_post_ge_H_pre'] = bool((df.H_post >= df.H_pre_plugin-1e-12).all())
    report['offline_one_family_per_state'] = bool(df.groupby(['group', 'seed']).family_hash.nunique().eq(1).all())
    prim = pd.read_csv(res/'offline_primary_statistics.csv'); vecs = []
    for _, r in prim.iterrows():
        g = df[df.group == r.group]
        if r.endpoint in ('H_pre_crossfit', 'foresight_lb'):
            vecs.append(g[g.policy == 'Safe-MinRisk'].sort_values('seed')[r.endpoint].to_numpy())
        else:
            comp = r.endpoint.replace('DC_gain_vs_', '')
            a = g[g.policy == comp].sort_values('seed').exante_loss.to_numpy()
            vecs.append(a-g[g.policy == 'DC-PRISM'].sort_values('seed').exante_loss.to_numpy())
    report['offline_primary_max_abs_error'] = recompute(prim, vecs, alpha)
    sec = pd.read_csv(res/'offline_baseline_statistics.csv'); vecs = []
    for _, r in sec.iterrows():
        g = df[df.group == r.group]
        vecs.append(g[g.policy == r.comparator].sort_values('seed').exante_loss.to_numpy()
                    - g[g.policy == 'DC-PRISM'].sort_values('seed').exante_loss.to_numpy())
    report['offline_baseline_max_abs_error'] = recompute(sec, vecs, alpha)
    ps = pd.read_csv(res/'online_seed_metrics.csv'); on = pd.read_csv(res/'online_primary_statistics.csv'); vecs = []
    for _, r in on.iterrows():
        g = ps[(ps.stream == r.stream) & (ps.dataset == r.dataset)].pivot(index='seed', columns='policy', values='alc_brier')
        vecs.append((g[r.comparator]-g['DC-PRISM']).to_numpy())
    report['online_primary_max_abs_error'] = recompute(on, vecs, alpha)
    report['online_envelope_ratio_max'] = float(ps.max_envelope_ratio.max())
    if args.data_dir:
        import run_offline as ro, run_online as rn
        cohorts = protocol['offline']['cohorts']
        ro._init(str(args.data_dir/'real_credit_datasets_authentic.npz'), cohorts)
        tasks = [t for t in ro.tasks_from(protocol)]
        firsts = {}
        for t in tasks:
            firsts.setdefault(t[0], []).append(t)
        replay = [x for v in firsts.values() for x in (v[0], v[-1])]
        worst = 0.; n = 0
        for t in replay:
            rows, _ = ro.one_state(t)
            for r in rows:
                s = df[(df.group == t[0]) & (df.seed == t[2]) & (df.policy == r['policy'])].iloc[0]
                assert s.selected == r['selected']
                worst = max(worst, abs(s.exante_loss-r['exante_loss']), abs(s.H_pre_crossfit-r['H_pre_crossfit'])); n += 1
        report['offline_replayed_states'] = len(replay); report['offline_replayed_rows'] = n
        report['offline_replay_max_abs_error'] = worst
        rn._init(str(args.data_dir/'real_credit_datasets_authentic.npz'))
        traj = pd.read_csv(res/'online_trajectories.csv.gz'); worst = 0.; n = 0
        for stream in protocol['online']['streams']:
            for ds in protocol['online']['datasets']:
                for seed in (protocol['online']['seeds'][0], protocol['online']['seeds'][-1]):
                    for r in rn.replay((stream, ds, seed, rn.POLICIES)):
                        s = traj[(traj.stream == stream) & (traj.dataset == ds) & (traj.seed == seed) &
                                 (traj.policy == r['policy']) & (traj['round'] == r['round'])].iloc[0]
                        assert s.k == r['k']; worst = max(worst, abs(s.brier_t-r['brier_t'])); n += 1
        report['online_replayed_rows'] = n; report['online_replay_max_abs_error'] = worst
    ok = (report['offline_envelope_feasible'] and report['offline_H_post_ge_H_pre'] and report['offline_one_family_per_state']
          and max(report['offline_primary_max_abs_error'], report['offline_baseline_max_abs_error'],
                  report['online_primary_max_abs_error']) < 1e-10 and report['online_envelope_ratio_max'] <= 1+1e-9
          and report.get('offline_replay_max_abs_error', 0) < 1e-12 and report.get('online_replay_max_abs_error', 0) < 1e-12)
    report['status'] = 'PASS' if ok else 'FAIL'
    print(json.dumps(report, indent=2))
    if args.data_dir is None:
        report['scope'] = 'public statistical recomputation; replay requires local data'
    name = 'validation.json' if args.data_dir else 'validation_public.json'
    (res/name).write_text(json.dumps(report, indent=2)+"\n")
    assert ok


if __name__ == '__main__':
    main()
