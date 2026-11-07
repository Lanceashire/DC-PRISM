"""Registered summaries for supplement-v7."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location('v3_stats_v7', ROOT/'experiments/supplement_v3/stats.py')
v3s = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(v3s)


def _ci(x):
    x = np.asarray(x, float); n = len(x); m = x.mean(); se = x.std(ddof=1)/np.sqrt(n); c = t.ppf(.975, n-1)
    return float(m), float(m-c*se), float(m+c*se)


def run(states, per_seed, traj, protocol, out):
    inf = protocol['inference']; a, d, ld = inf['alpha'], inf['material_delta'], inf['legacy_delta']
    gcols = [c for c in states.columns if c.startswith('gain_')]
    summ = states.groupby(['study', 'setting', 'group', 'learner'])[
        ['H_post', 'H_pre_plugin', 'H_pre_crossfit', 'foresight', 'tl_evpi', 'ind_evpi', 'tl_delta', 'tl_s', 'median_gain_sd', 'near_ties', 'family_size']
        + gcols + (['H_post_realized'] if 'H_post_realized' in states else [])].mean()
    summ.to_csv(out/'summary.csv')
    # A: floor under both learners; learner agreement
    rows = []
    for learner, x in summ.groupby(level='learner'):
        rows.append(dict(learner=learner, groups=len(x), evpi_min=x.foresight.min(), evpi_max=x.foresight.max(),
                         hpre_min=x.H_pre_crossfit.min(), hpre_max=x.H_pre_crossfit.max()))
    pd.DataFrame(rows).to_csv(out/'floor_by_learner.csv', index=False)
    # A primary family: real-label credit groups under the boosting learner
    recs = []
    A = states[(states.study == 'A') & (states.setting == 'real-label') & (states.learner == 'gbdt')]
    for g, x in A.groupby('group'):
        x = x.sort_values('seed')
        recs.append(({'group': g, 'endpoint': 'DC_selection_gap_gbdt'}, (x.H_pre_crossfit-x['gain_DC-PRISM']).to_numpy()))
        recs.append(({'group': g, 'endpoint': 'DC_minus_TargetVar_gbdt'}, (x['gain_DC-PRISM']-x['gain_TargetVar-Greedy']).to_numpy()))
    p = v3s.family_table(recs, a, d, ld); assert len(p) == protocol['study_A']['primary_family_size']
    p.to_csv(out/'A_primary_statistics.csv', index=False)
    # B: diabetes protocol (logistic learner primary) + calibration
    recs, cal = [], []
    B = states[states.study == 'B']
    for (g, learner), x in B.groupby(['group', 'learner']):
        x = x.sort_values('seed')
        m, lo, hi = _ci(x.H_post-x.H_post_realized)
        cal.append(dict(group=g, learner=learner, H_post=float(x.H_post.mean()), H_post_realized=float(x.H_post_realized.mean()),
                        diff=m, diff_ci_low=lo, diff_ci_high=hi))
        if learner == protocol['study_B']['primary_learner']:
            for name, v in [('H_pre_crossfit', x.H_pre_crossfit), ('DC_selection_gap', x.H_pre_crossfit-x['gain_DC-PRISM']),
                            ('DC_minus_TargetVar', x['gain_DC-PRISM']-x['gain_TargetVar-Greedy']), ('DC_vs_Safe-MinRisk', x['gain_DC-PRISM'])]:
                recs.append(({'group': g, 'endpoint': name}, v.to_numpy()))
    p = v3s.family_table(recs, a, d, ld); assert len(p) == protocol['study_B']['primary_family_size']
    p.to_csv(out/'B_primary_statistics.csv', index=False)
    pd.DataFrame(cal).to_csv(out/'B_calibration.csv', index=False)
    # B replay
    recs = []
    for stream, x in per_seed.groupby('stream'):
        w = x.pivot(index='seed', columns='policy', values='alc_brier')
        for comp in protocol['study_B']['replay_competitors']:
            recs.append(({'stream': stream, 'comparator': comp}, (w[comp]-w['DC-PRISM']).to_numpy()))
    p = v3s.family_table(recs, a, d, ld); assert len(p) == protocol['study_B']['replay_family_size']
    p.to_csv(out/'B_replay_statistics.csv', index=False)
    per_seed.groupby(['stream', 'policy']).mean(numeric_only=True).to_csv(out/'B_replay_summary.csv')
    t2 = traj[traj.k > 0]
    t2.assign(exceed=t2.realized_loss > t2.envelope+1e-12).groupby('policy').exceed.agg(['sum', 'count']).to_csv(out/'B_replay_exceedance.csv')
    # C: analytic EVPI
    rows = []
    for learner, x in states.groupby('learner'):
        g = summ.xs(learner, level='learner')
        rho = spearmanr(x.tl_evpi, x.foresight).statistic
        r2 = np.corrcoef(g.tl_evpi, g.foresight)[0, 1]**2
        inside = float(np.mean((g.tl_evpi <= g.foresight) & (g.foresight <= g.ind_evpi)))
        rho_i = spearmanr(x.ind_evpi, x.foresight).statistic
        rows.append(dict(learner=learner, states=len(x), state_spearman=float(rho), state_spearman_ind=float(rho_i), group_bracket_coverage=inside, group_r2=float(r2),
                         group_ratio_median=float(np.median(g.tl_evpi/g.foresight)),
                         group_ratio_min=float((g.tl_evpi/g.foresight).min()), group_ratio_max=float((g.tl_evpi/g.foresight).max())))
    pd.DataFrame(rows).to_csv(out/'C_analytic_evpi.csv', index=False)
