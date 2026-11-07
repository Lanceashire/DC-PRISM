"""Registered summaries for supplement-v6 (studies V and R)."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location('v3_stats_v6', ROOT/'experiments/supplement_v3/stats.py')
v3s = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(v3s)
POL = ['DC-PRISM', 'TargetVar-Greedy', 'BAIT-Greedy', 'Fisher-cost', 'Uncertainty', 'Random-Feasible', 'PointValue', 'ProxyArgmax-F']


def _ci(x):
    x = np.asarray(x, float); n = len(x); m = x.mean(); se = x.std(ddof=1)/np.sqrt(n); c = t.ppf(.975, n-1)
    return m, m-c*se, m+c*se


def study_v(df, protocol, out):
    tol = protocol['study_V']['adequacy_tolerance']
    truth = df[df.model == 'truth'].set_index(['group', 'seed'])
    rows = []
    for model in [m for m in df.model.unique() if m != 'truth']:
        est = df[df.model == model].set_index(['group', 'seed'])
        for g in sorted(est.index.get_level_values(0).unique()):
            e, tr = est.loc[g].sort_index(), truth.loc[g].sort_index()
            for col in ['H_post', 'H_pre_crossfit', 'H_pre_plugin', 'foresight'] + [f'gain_{p}' for p in POL]:
                d = (e[col]-tr[col]).to_numpy(); m, lo, hi = _ci(d)
                rows.append(dict(model=model, group=g, quantity=col, truth_mean=float(tr[col].mean()), est_mean=float(e[col].mean()),
                                 bias=m, bias_ci_low=lo, bias_ci_high=hi, adequate=bool(lo > -tol and hi < tol),
                                 corr=float(np.corrcoef(e[col], tr[col])[0, 1]) if tr[col].std() > 0 else np.nan))
    pd.DataFrame(rows).to_csv(out/'study_V_bias.csv', index=False)
    df.groupby(['group', 'model']).mean(numeric_only=True).to_csv(out/'study_V_summary.csv')


def study_r(df, protocol, out):
    cfg = protocol['inference']; a, d, ld = cfg['alpha'], cfg['material_delta'], cfg['legacy_delta']
    prim_model = protocol['study_R']['primary_model']
    df.groupby(['stream', 'dataset', 'model']).mean(numeric_only=True).to_csv(out/'study_R_summary.csv')
    recs, sec, cal = [], [], []
    for (stream, ds), g in df.groupby(['stream', 'dataset']):
        e = g[g.model == prim_model].sort_values('seed'); r = g[g.model == 'realized'].sort_values('seed')
        lab = {'stream': stream, 'dataset': ds}
        recs.append((dict(lab, endpoint='H_pre_crossfit'), e.H_pre_crossfit.to_numpy()))
        recs.append((dict(lab, endpoint='DC_selection_gap'), (e.H_pre_crossfit-e['gain_DC-PRISM']).to_numpy()))
        recs.append((dict(lab, endpoint='DC_minus_TargetVar'), (e['gain_DC-PRISM']-e['gain_TargetVar-Greedy']).to_numpy()))
        recs.append((dict(lab, endpoint='DC_vs_Safe-MinRisk'), e['gain_DC-PRISM'].to_numpy()))
        sec.append((dict(lab, endpoint='realized_DC_minus_TargetVar'), (r['gain_DC-PRISM']-r['gain_TargetVar-Greedy']).to_numpy()))
        sec.append((dict(lab, endpoint='realized_DC_vs_Safe-MinRisk'), r['gain_DC-PRISM'].to_numpy()))
        for model in protocol['phat_models']:
            m_ = g[g.model == model].sort_values('seed')
            mm, lo, hi = _ci(m_.H_post.to_numpy()-r.H_post.to_numpy())
            cal.append(dict(lab, model=model, H_post_phat=float(m_.H_post.mean()), H_post_realized=float(r.H_post.mean()),
                            diff=mm, diff_ci_low=lo, diff_ci_high=hi, holdout_brier=float(m_.holdout_brier.mean()),
                            holdout_mean_p=float(m_.holdout_mean_p.mean()), holdout_mean_y=float(m_.holdout_mean_y.mean())))
    p = v3s.family_table(recs, a, d, ld); assert len(p) == protocol['study_R']['primary_family_size']
    p.to_csv(out/'study_R_primary_statistics.csv', index=False)
    v3s.family_table(sec, a, d, ld).to_csv(out/'study_R_realized_statistics.csv', index=False)
    pd.DataFrame(cal).to_csv(out/'study_R_calibration.csv', index=False)


def run(study, df, protocol, out):
    {'V': study_v, 'R': study_r}[study](df, protocol, out)
