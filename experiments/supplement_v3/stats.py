"""Pre-registered inference for supplement-v3.

Paired one-sample t on the independent unit (world, pair-arm or split seed),
Bonferroni simultaneous t intervals within each registered family, global Holm
within each family. A difference is `material` only when its simultaneous
interval clears the registered delta; `legacy_delta` (1e-4) is reported for
continuity with earlier batches.
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import t, ttest_1samp


def _holm(p):
    p = np.asarray(p, float); order = np.argsort(p); adj = np.empty(len(p)); run = 0.
    for rank, i in enumerate(order):
        run = max(run, min(1., p[i]*(len(p)-rank))); adj[i] = run
    return adj


def _decide(lo, hi, delta):
    if lo > delta: return 'material_positive'
    if hi < -delta: return 'material_negative'
    if lo >= -delta and hi <= delta: return 'within_material_band'
    if lo > 0: return 'positive_below_material'
    if hi < 0: return 'negative_below_material'
    return 'indeterminate'


def family_table(records, alpha, delta, legacy_delta):
    """records: list of (labels:dict, vector)."""
    m = len(records); rows = []
    for labels, x in records:
        x = np.asarray(x, float); n = len(x); mu = float(x.mean()); sd = float(x.std(ddof=1)); se = sd/np.sqrt(n)
        c = float(t.ppf(1-alpha/(2*m), n-1)); c95 = float(t.ppf(.975, n-1))
        lo, hi = mu-c*se, mu+c*se
        p = float(ttest_1samp(x, 0).pvalue) if sd > 1e-15 else (1. if abs(mu) < 1e-15 else 0.)
        rows.append(dict(labels, n=n, mean=mu, sd=sd, ci95_low=mu-c95*se, ci95_high=mu+c95*se,
                         simultaneous_low=lo, simultaneous_high=hi, p_raw=p,
                         decision=_decide(lo, hi, delta), decision_legacy=_decide(lo, hi, legacy_delta)))
    df = pd.DataFrame(rows); df['p_holm'] = _holm(df.p_raw); df['family_size'] = m
    return df


def offline_statistics(df, sens, protocol, out):
    cfg = protocol['inference']; a, d, ld = cfg['alpha'], cfg['material_delta'], cfg['legacy_delta']
    groups = sorted(df.group.unique(), key=lambda g: (g != 'Controlled', g))
    summary = df.groupby(['group', 'policy'], as_index=False).agg(
        states=('seed', 'nunique'), exante_loss=('exante_loss', 'mean'), k=('k', 'mean'),
        exante_gain_vs_anchor=('exante_gain_vs_anchor', 'mean'), realized_brier=('realized_brier', 'mean'),
        mean_mc_se=('exante_se', 'mean'))
    summary.to_csv(out/'offline_policy_summary.csv', index=False)
    dec = df[df.policy == 'Safe-MinRisk'].groupby('group', as_index=False)[
        ['H_post', 'H_pre_plugin', 'H_pre_crossfit', 'foresight_lb', 'family_size', 'K']].mean()
    gaps = df[df.policy.isin(['DC-PRISM', 'Full'])].pivot_table(index='group', columns='policy',
                                                               values='selection_gap_crossfit', aggfunc='mean')
    dec = dec.merge(gaps.add_prefix('selection_gap_crossfit_').reset_index(), on='group')
    dec.to_csv(out/'offline_decomposition_summary.csv', index=False)

    def vec(g, policy, col):
        return df[(df.group == g) & (df.policy == policy)].sort_values('seed')[col].to_numpy()

    primary = []
    for g in groups:
        primary.append(({'group': g, 'endpoint': 'H_pre_crossfit'}, vec(g, 'Safe-MinRisk', 'H_pre_crossfit')))
        primary.append(({'group': g, 'endpoint': 'foresight_lb'}, vec(g, 'Safe-MinRisk', 'foresight_lb')))
        for comp in protocol['offline']['primary_competitors']:
            x = vec(g, comp, 'exante_loss')-vec(g, 'DC-PRISM', 'exante_loss')
            primary.append(({'group': g, 'endpoint': f'DC_gain_vs_{comp}'}, x))
    p = family_table(primary, a, d, ld); assert len(p) == protocol['offline']['primary_family_size']
    p.to_csv(out/'offline_primary_statistics.csv', index=False)
    secondary = []
    for g in groups:
        for comp in protocol['offline']['secondary_competitors']:
            x = vec(g, comp, 'exante_loss')-vec(g, 'DC-PRISM', 'exante_loss')
            secondary.append(({'group': g, 'comparator': comp}, x))
    s = family_table(secondary, a, d, ld); assert len(s) == protocol['offline']['secondary_family_size']
    s.to_csv(out/'offline_baseline_statistics.csv', index=False)
    rows = []
    for g in groups:
        base = sens[(sens.group == g) & (sens.variant == 'default')].sort_values('seed')
        for v in sens.variant.unique():
            if v == 'default': continue
            cur = sens[(sens.group == g) & (sens.variant == v)].sort_values('seed')
            x = cur.exante_loss.to_numpy()-base.exante_loss.to_numpy(); n = len(x); se = x.std(ddof=1)/np.sqrt(n)
            c = float(t.ppf(.975, n-1))
            rows.append(dict(group=g, variant=v, n=n, loss_change_vs_default=float(x.mean()),
                             ci95_low=float(x.mean()-c*se), ci95_high=float(x.mean()+c*se),
                             mean_k=float(cur.k.mean()), default_k=float(base.k.mean())))
    pd.DataFrame(rows).to_csv(out/'offline_sensitivity_summary.csv', index=False)
    return p, s


def online_statistics(per_seed, protocol, out):
    cfg = protocol['inference']; a, d, ld = cfg['alpha'], cfg['material_delta'], cfg['legacy_delta']
    on = protocol['online']; recs = []
    for stream in on['streams']:
        for ds in on['datasets']:
            g = per_seed[(per_seed.stream == stream) & (per_seed.dataset == ds)].pivot(index='seed', columns='policy', values='alc_brier')
            assert len(g) == len(on['seeds'])
            for comp in on['primary_competitors']:
                recs.append(({'stream': stream, 'dataset': ds, 'comparator': comp}, (g[comp]-g['DC-PRISM']).to_numpy()))
    p = family_table(recs, a, d, ld); assert len(p) == on['primary_family_size']
    p.to_csv(out/'online_primary_statistics.csv', index=False)
    summ = per_seed.groupby(['stream', 'dataset', 'policy'], as_index=False).agg(
        seeds=('seed', 'nunique'), alc_brier=('alc_brier', 'mean'), alc_sd=('alc_brier', 'std'),
        labels_per_round=('labels_per_round', 'mean'), max_envelope_ratio=('max_envelope_ratio', 'max'))
    summ.to_csv(out/'online_summary.csv', index=False)
    return p
