"""Registered summaries for supplement-v13 (learning curves with test-set, split and ex-ante oracles).

Per trajectory and method: ALC = mean test Brier over steps 1..30; early / late windows = steps 1-10 / 21-30.
Components (positive = the first-named curve is worse):
  reuse       = ALC(oracle-split)  - ALC(oracle-test)
  luck        = ALC(oracle-exante) - ALC(oracle-split)
  short_<o>_<s> = ALC(strategy s) - ALC(oracle o)
Inference: paired t over seeds, Bonferroni simultaneous intervals and Holm within each registered family
(supplement-v3 family_table), material delta 0.5e-3 Brier.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location('v3_stats_v13', ROOT/'experiments/supplement_v3/stats.py')
v3s = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(v3s)
ORACLES = ['oracle-test', 'oracle-split', 'oracle-exante']


def per_trajectory(df, lo, hi):
    x = df[(df.step >= lo) & (df.step <= hi)]
    a = x.pivot_table(index=['dataset', 'seed'], columns='method', values='brier_T', aggfunc='mean')
    strats = [c for c in a.columns if c not in ORACLES]
    out = pd.DataFrame(index=a.index)
    out['reuse'] = a['oracle-split']-a['oracle-test']; out['luck'] = a['oracle-exante']-a['oracle-split']
    for o in ORACLES:
        for s in strats:
            out[f'short_{o}_{s}'] = a[s]-a[o]
    for m in a.columns:
        out[f'alc_{m}'] = a[m]
    return out, strats


def run(part, df, P, out):
    inf = P['inference']; a, d, ld = inf['alpha'], inf['material_delta'], inf['legacy_delta']
    noise = df.groupby('dataset').noise_index.mean()
    win = {'all': (1, 30), 'early': (1, 10), 'late': (21, 30)}
    tabs = {}
    for w, (lo, hi) in win.items():
        t, strats = per_trajectory(df, lo, hi); tabs[w] = t
        g = t.groupby('dataset').mean()
        best = g[[f'alc_{s}' for s in strats]].idxmin(axis=1).str[4:]
        g['best_strategy'] = best
        g['lead_test'] = [g.loc[ds, f'short_oracle-test_{b}'] for ds, b in best.items()]
        g['lead_split'] = [g.loc[ds, f'short_oracle-split_{b}'] for ds, b in best.items()]
        g['remaining'] = [g.loc[ds, f'short_oracle-exante_{b}'] for ds, b in best.items()]
        g['share_reuse'] = g.reuse/g.lead_test; g['share_luck'] = g.luck/g.lead_test; g['share_remaining'] = g.remaining/g.lead_test
        g['luck_share_of_split_lead'] = g.luck/g.lead_split
        g['noise_index'] = noise
        g.to_csv(out/f'summary_{w}.csv')
    t = tabs['all']; pred = {}
    # primary family 1: luck and reuse per dataset (whole curve)
    recs = []
    for ds, x in t.groupby('dataset'):
        for e in ['luck', 'reuse']:
            recs.append(({'dataset': ds, 'endpoint': e}, x[e].to_numpy()))
    f1 = v3s.family_table(recs, a, d, ld); f1.to_csv(out/'primary_luck_reuse.csv', index=False)
    dec1 = f1.set_index(['dataset', 'endpoint']).decision
    g = pd.read_csv(out/'summary_all.csv').set_index('dataset')
    order = noise.sort_values(ascending=False).index.tolist()
    if part == 'MAIN':
        # primary family 2: every strategy against every oracle (whole curve)
        recs = []
        for ds, x in t.groupby('dataset'):
            for o in ORACLES:
                for s in strats:
                    recs.append(({'dataset': ds, 'oracle': o, 'strategy': s}, x[f'short_{o}_{s}'].to_numpy()))
        f2 = v3s.family_table(recs, a, d, ld); f2.to_csv(out/'primary_shortfall.csv', index=False)
        allshort = f2.groupby(['dataset', 'oracle']).decision.apply(lambda z: bool((z == 'material_positive').all())).unstack()
        allshort.to_csv(out/'all_strategies_materially_short.csv')
        top4 = order[:4]
        pred['P1_noisiest4'] = ','.join(top4)
        pred['P1_luck_material_in_noisiest4'] = int(sum(dec1[(ds, 'luck')] == 'material_positive' for ds in top4))
        pred['P1_luck_material_all'] = int(sum(dec1[(ds, 'luck')] == 'material_positive' for ds in noise.index))
        pred['P1_spearman_noise_luck_share_of_split_lead'] = float(spearmanr(g.noise_index, g.luck_share_of_split_lead).statistic)
        e = pd.read_csv(out/'summary_early.csv').set_index('dataset'); l = pd.read_csv(out/'summary_late.csv').set_index('dataset')
        pred['P2_reuse_share_late_gt_early_count'] = int((l.share_reuse > e.share_reuse).sum())
        for o in ORACLES:
            pred[f'P3_all_short_{o}_count'] = int(allshort[o].sum())
    else:
        pred['P4_luck_material_count'] = int(sum(dec1[(ds, 'luck')] == 'material_positive' for ds in noise.index))
        pred['P4_datasets'] = ','.join(noise.index)
    pd.Series(pred, dtype=object).to_csv(out/'prediction_check.csv')
