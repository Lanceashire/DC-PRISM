"""Registered summaries for external-v15 (see protocol.json predictions Q1-Q5).

Per trajectory and method: ALC over a window = mean test Brier over its steps. Components (positive = first curve worse):
  reuse = ALC(oracle-split) - ALC(oracle-test);  luck = ALC(oracle-exante) - ALC(oracle-split)
  short_<oracle>_<strategy> = ALC(strategy) - ALC(oracle)
Families per grid: F1 whole-curve reuse and luck per dataset; F2 early-window luck per dataset; F3 every strategy x oracle
per dataset (whole curve). Paired t over seeds, Bonferroni simultaneous intervals, Holm; material delta 0.0005.
Usage: python stats_v15.py --protocol protocol.json --runs <dir> --output <dir>
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t as tdist, ttest_1samp, spearmanr

ORACLES = ['oracle-test', 'oracle-split', 'oracle-exante']; STRATS = ['Random', 'Margin', 'Entropy', 'CoreSet', 'BADGE']


def decide(lo, hi, d):
    if lo > d: return 'material_positive'
    if hi < -d: return 'material_negative'
    if lo >= -d and hi <= d: return 'within_material_band'
    if lo > 0: return 'positive_below_material'
    if hi < 0: return 'negative_below_material'
    return 'indeterminate'


def family(recs, alpha, d):
    m = len(recs); rows = []
    for lab, x in recs:
        x = np.asarray(x, float); n = len(x); mu = x.mean(); se = x.std(ddof=1)/np.sqrt(n); c = tdist.ppf(1-alpha/(2*m), n-1)
        p = float(ttest_1samp(x, 0).pvalue) if se > 0 else 1.
        rows.append(dict(lab, n=n, mean=mu, sim_low=mu-c*se, sim_high=mu+c*se, p_raw=p, decision=decide(mu-c*se, mu+c*se, d)))
    df = pd.DataFrame(rows); o = np.argsort(df.p_raw.values); adj = np.empty(m); run = 0.
    for r, i in enumerate(o): run = max(run, min(1., df.p_raw.values[i]*(m-r))); adj[i] = run
    df['p_holm'] = adj; df['family_size'] = m; return df


def per_traj(df, lo, hi):
    x = df[(df.step >= lo) & (df.step <= hi)]
    a = x.pivot_table(index=['dataset', 'seed'], columns='method', values='brier_T', aggfunc='mean')
    o = pd.DataFrame(index=a.index); o['reuse'] = a['oracle-split']-a['oracle-test']; o['luck'] = a['oracle-exante']-a['oracle-split']
    for orc in ORACLES:
        for s in STRATS: o[f'short_{orc}_{s}'] = a[s]-a[orc]
    for m in a.columns: o[f'alc_{m}'] = a[m]
    return o


def summarize(t, noise):
    g = t.groupby('dataset').mean(); best = g[[f'alc_{s}' for s in STRATS]].idxmin(axis=1).str[4:]
    g['best_strategy'] = best
    g['lead_test'] = [g.loc[d, f'short_oracle-test_{b}'] for d, b in best.items()]
    g['remaining'] = [g.loc[d, f'short_oracle-exante_{b}'] for d, b in best.items()]
    for k in ['reuse', 'luck', 'remaining']: g[f'share_{k}'] = g[k]/g.lead_test
    g['noise_index'] = noise; return g


def run_grid(df, P, out, grid):
    a, d = P['inference']['alpha'], P['inference']['material_delta']; W = P['windows']
    noise = df.groupby('dataset').noise_index.mean(); T = {w: per_traj(df, *W[w]) for w in W}
    S = {w: summarize(T[w], noise) for w in W}
    for w in W: S[w].to_csv(out/f'{grid}_summary_{w}.csv')
    f1 = family([({'dataset': ds, 'endpoint': e}, x[e]) for ds, x in T['all'].groupby('dataset') for e in ['reuse', 'luck']], a, d)
    f2 = family([({'dataset': ds, 'endpoint': 'luck_early'}, x['luck']) for ds, x in T['early'].groupby('dataset')], a, d)
    f3 = family([({'dataset': ds, 'oracle': o, 'strategy': s}, x[f'short_{o}_{s}']) for ds, x in T['all'].groupby('dataset')
                 for o in ORACLES for s in STRATS], a, d)
    for n_, f in [('F1_whole_reuse_luck', f1), ('F2_early_luck', f2), ('F3_shortfalls', f3)]: f.to_csv(out/f'{grid}_{n_}.csv', index=False)
    allshort = f3.groupby(['dataset', 'oracle']).decision.apply(lambda z: bool((z == 'material_positive').all())).unstack()
    allshort.to_csv(out/f'{grid}_all_strategies_materially_short.csv')
    dec1 = f1.set_index(['dataset', 'endpoint']).decision; dec2 = f2.set_index('dataset').decision
    res = dict(grid=grid, datasets=len(noise), trajectories=int(T['all'].shape[0]),
               reuse_material=int(sum(dec1[(ds, 'reuse')] == 'material_positive' for ds in noise.index)),
               luck_whole_material=int(sum(dec1[(ds, 'luck')] == 'material_positive' for ds in noise.index)),
               luck_early_material=int((dec2 == 'material_positive').sum()),
               reuse_share_late_gt_early=int((S['late'].share_reuse > S['early'].share_reuse).sum()),
               all_short_test=int(allshort['oracle-test'].sum()), all_short_split=int(allshort['oracle-split'].sum()),
               all_short_exante=int(allshort['oracle-exante'].sum()),
               pairs_short_test=int((f3[f3.oracle == 'oracle-test'].decision == 'material_positive').sum()),
               pairs_short_split=int((f3[f3.oracle == 'oracle-split'].decision == 'material_positive').sum()),
               pairs_short_exante=int((f3[f3.oracle == 'oracle-exante'].decision == 'material_positive').sum()),
               pairs_total=int((f3.oracle == 'oracle-test').sum()),
               spearman_noise_luck_share_early=float(spearmanr(S['early'].noise_index, S['early'].share_luck).statistic),
               spearman_noise_luck_share_all=float(spearmanr(S['all'].noise_index, S['all'].share_luck).statistic))
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--protocol', type=Path, required=True); ap.add_argument('--runs', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True); args = ap.parse_args()
    P = json.loads(args.protocol.read_text()); args.output.mkdir(parents=True, exist_ok=True)
    df = pd.concat([pd.read_csv(f) for f in sorted(args.runs.glob('*__*__*__*.csv'))], ignore_index=True)
    df.to_csv(args.output/'rows.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    R = {}
    for grid, G in P['grids'].items():
        s0 = G['seed_start']; x = df[(df.seed >= s0) & (df.seed < s0+G['n_seeds']) & (df['head'] == G['head'])]
        if len(x): R[grid] = run_grid(x, P, args.output, grid)
    A = R.get('A', {}); pred = {}
    if A:
        pred['Q1'] = A['reuse_material'] >= 6; pred['Q2'] = A['reuse_share_late_gt_early'] >= 6
        pred['Q3'] = A['all_short_test'] >= 7 and A['all_short_exante'] <= 4; pred['Q4'] = A['luck_early_material'] >= 4
    for g in ['B', 'C']:
        if g in R: pred[f'Q5_{g}'] = R[g]['all_short_test'] > R[g]['all_short_exante']
    (args.output/'prediction_check.json').write_text(json.dumps(dict(predictions=pred, grids=R), indent=2))
    print(json.dumps(dict(predictions=pred, grids=R), indent=2))


if __name__ == '__main__':
    main()
