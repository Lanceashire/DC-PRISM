"""Registered summaries for supplement-v9 (noise gradient)."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location('v3_stats_v9', ROOT/'experiments/supplement_v3/stats.py')
v3s = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(v3s)


def _ci(x):
    x = np.asarray(x, float); m = x.mean(); se = x.std(ddof=1)/np.sqrt(len(x)); c = t.ppf(.975, len(x)-1)
    return float(m), float(m-c*se), float(m+c*se)


def run(df, protocol, out):
    inf = protocol['inference']; a, d, ld = inf['alpha'], inf['material_delta'], inf['legacy_delta']
    fam = df[df.part == 'family']; ora = df[df.part == 'oracle'].copy()
    g = fam.groupby(['dataset', 'learner'])[['noise_index', 'H_post', 'H_post_realized', 'H_pre_plugin', 'H_pre_crossfit',
                                              'foresight', 'tl_evpi', 'ind_evpi', 'gain_DC-PRISM', 'gain_TargetVar-Greedy',
                                              'gain_ProxyArgmax-F']].mean()
    g['bracket_ok'] = (g.tl_evpi <= g.foresight) & (g.foresight <= g.ind_evpi)
    g['deployable_share'] = g.H_pre_plugin/g.H_post
    cal = []
    for (ds, lr), x in fam.groupby(['dataset', 'learner']):
        m, lo, hi = _ci(x.H_post-x.H_post_realized); cal.append(dict(dataset=ds, learner=lr, diff=m, lo=lo, hi=hi))
    g = g.join(pd.DataFrame(cal).set_index(['dataset', 'learner']))
    g.to_csv(out/'family_summary.csv')
    ora['test_reuse'] = ora.reported-ora.ex_post; ora['floor'] = ora.ex_post-ora.deployable
    pc = [c for c in ora.columns if c.startswith('prac_')]
    rows, recs = [], []
    for ds, x in ora.groupby('dataset'):
        x = x.sort_values('seed'); best = x[pc].mean().idxmax()
        lr, ldp = (x.reported-x[best]).to_numpy(), (x.deployable-x[best]).to_numpy()
        m, lo, hi = _ci(ldp)
        rows.append(dict(dataset=ds, noise_index=float(x.noise_index.mean()), reported=x.reported.mean(), test_reuse=x.test_reuse.mean(),
                         floor=x.floor.mean(), deployable=x.deployable.mean(), share_floor=x.floor.mean()/x.reported.mean(),
                         share_reuse=x.test_reuse.mean()/x.reported.mean(), best_fixed=best[5:], lead_reported=lr.mean(),
                         lead_deployable=m, lead_deployable_lo=lo, lead_deployable_hi=hi,
                         artefact_share_of_lead=1-m/lr.mean() if lr.mean() > 0 else np.nan))
        for e in protocol['study']['primary_endpoints']:
            recs.append(({'dataset': ds, 'endpoint': e}, x[e].to_numpy()))
    o = pd.DataFrame(rows).set_index('dataset'); o.to_csv(out/'oracle_summary.csv')
    p = v3s.family_table(recs, a, d, ld); assert len(p) == protocol['study']['primary_family_size']
    p.to_csv(out/'primary_statistics.csv', index=False)
    lowset = list(o.noise_index.nsmallest(2).index)          # defined by the noise index, not by the floor
    low = o.loc[lowset].share_floor.mean(); high = o.drop(index=lowset).share_floor.mean()
    pred = dict(spearman_noise_vs_floor_share=float(spearmanr(o.noise_index, o.share_floor).statistic),
                spearman_noise_vs_floor_abs=float(spearmanr(o.noise_index, o.floor).statistic),
                low_noise_cohorts=','.join(lowset), low_noise_mean_floor_share=float(low), other_mean_floor_share=float(high),
                bracket_coverage=float(g.bracket_ok.mean()))
    pd.Series(pred).to_csv(out/'prediction_check.csv')
