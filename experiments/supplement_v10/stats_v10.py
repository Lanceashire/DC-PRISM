"""Registered summaries for supplement-v10."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location('v3_stats_v10', ROOT/'experiments/supplement_v3/stats.py')
v3s = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(v3s)


def _ci(x):
    x = np.asarray(x, float); m = x.mean(); se = x.std(ddof=1)/np.sqrt(len(x)); c = t.ppf(.975, len(x)-1)
    return float(m), float(m-c*se), float(m+c*se)


def _oracle(ora):
    ora = ora.assign(test_reuse=ora.reported-ora.ex_post, floor=ora.ex_post-ora.deployable)
    pc = [c for c in ora.columns if c.startswith('prac_')]; rows = []
    for key, x in ora.groupby(['dataset'] + (['target_k'] if 'target_k' in ora else [])):
        key = key if isinstance(key, tuple) else (key,); best = x[pc].mean().idxmax()
        m, lo, hi = _ci(x.deployable-x[best]); lr = float((x.reported-x[best]).mean())
        rows.append(dict(zip(['dataset', 'target_k'], key), noise_index=float(x.noise_index.mean()), reported=x.reported.mean(),
                         test_reuse=x.test_reuse.mean(), floor=x.floor.mean(), deployable=x.deployable.mean(),
                         share_floor=x.floor.mean()/x.reported.mean(), best_fixed=best[5:], lead_reported=lr,
                         lead_deployable=m, lead_lo=lo, lead_hi=hi))
    return ora, pd.DataFrame(rows)


def run(part, df, protocol, out):
    inf = protocol['inference']; a, d, ld = inf['alpha'], inf['material_delta'], inf['legacy_delta']
    fam = df[df.part == 'family']; ora, osum = _oracle(df[df.part == 'oracle'].copy())
    by = ['dataset'] + (['target_k'] if part == 'B' else [])
    g = fam.groupby(by)[['noise_index', 'H_post', 'H_pre_plugin', 'H_pre_crossfit', 'foresight', 'tl_evpi', 'ind_evpi']
                        + (['H_post_realized'] if 'H_post_realized' in fam else []) + (['K', 'selected_label_var'] if part == 'B' else [])].mean()
    g['bracket_ok'] = (g.tl_evpi <= g.foresight) & (g.foresight <= g.ind_evpi)
    g.to_csv(out/'family_summary.csv'); osum.to_csv(out/'oracle_summary.csv', index=False)
    recs = []
    if part == 'A':
        for ds, x in fam.groupby('dataset'):
            m, lo, hi = _ci(x.H_post-x.H_post_realized); g.loc[ds, ['cal_diff', 'cal_lo', 'cal_hi']] = [m, lo, hi]
        g.to_csv(out/'family_summary.csv')
        for ds, x in ora.groupby('dataset'):
            recs.append(({'dataset': ds, 'endpoint': 'floor'}, x.sort_values('seed').floor.to_numpy()))
        o = osum.set_index('dataset')
        pred = dict(spearman_noise_vs_floor_share=float(spearmanr(o.noise_index, o.share_floor).statistic),
                    bracket_coverage=float(g.bracket_ok.mean()), calibration_intervals_containing_zero=float(((g.cal_lo <= 0) & (g.cal_hi >= 0)).mean()))
    else:
        ks = protocol['part_B']['target_k']; lo_k, hi_k = min(ks), max(ks)
        for ds, x in fam.groupby('dataset'):
            w = x.pivot(index='seed', columns='target_k', values='foresight')
            recs.append(({'dataset': ds, 'endpoint': f'floor_k{hi_k}_minus_k{lo_k}'}, (w[hi_k]-w[lo_k]).to_numpy()))
        gg = g.reset_index()
        mono = gg.groupby('dataset').apply(lambda x: bool(np.all(np.diff(x.sort_values('target_k').foresight) > 0)), include_groups=False)
        pooled = fam.pivot_table(index=['dataset', 'seed'], columns='target_k', values='foresight')
        m, lo, hi = _ci(pooled[hi_k]-pooled[lo_k])
        pred = dict(monotone_datasets=int(mono.sum()), datasets=len(mono), pooled_diff=m, pooled_lo=lo, pooled_hi=hi,
                    bracket_coverage=float(g.bracket_ok.mean()))
    p = v3s.family_table(recs, a, d, ld); assert len(p) == protocol[f'part_{part}']['primary_family_size']
    p.to_csv(out/'primary_statistics.csv', index=False)
    pd.Series(pred).to_csv(out/'prediction_check.csv')
