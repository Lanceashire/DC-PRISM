"""Registered inference and summaries for supplement-v5 (studies E-H).

Uses the supplement-v3 family_table: paired one-sample t on the independent
unit, Bonferroni simultaneous intervals and global Holm within each registered
family, `material` only when the simultaneous interval clears the delta.
"""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location('v3_stats', ROOT/'experiments/supplement_v3/stats.py')
v3s = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(v3s)


def _order(groups):
    return sorted(groups, key=lambda g: (g != 'Controlled', g))


def _boot_ratio(num, den, n_boot=4000, seed=0):
    rng = np.random.default_rng(seed); n = len(num)
    idx = rng.integers(0, n, (n_boot, n))
    r = num[idx].mean(axis=1)/den[idx].mean(axis=1)
    return float(np.quantile(r, .025)), float(np.quantile(r, .975))


def study_e(df, protocol, out):
    rows = []
    for (g, fam, size), x in df.groupby(['group', 'family', 'size']):
        x = x.sort_values('seed')
        if len(x) < 8:
            continue
        post, plug, cf = x.H_post.to_numpy(), x.H_pre_plugin.to_numpy(), x.H_pre_crossfit.to_numpy()
        lo, hi = _boot_ratio(post-plug, post)
        rows.append(dict(group=g, family=fam, size=size, states=len(x), H_post=post.mean(), H_pre_plugin=plug.mean(),
                         H_pre_crossfit=cf.mean(), foresight_share=float(1-plug.mean()/post.mean()),
                         foresight_share_ci_low=lo, foresight_share_ci_high=hi,
                         pre_share_crossfit=float(cf.mean()/post.mean())))
    pd.DataFrame(rows).sort_values(['group', 'family', 'size']).to_csv(out/'study_E_summary.csv', index=False)
    # enlarged-family size per group (for matched comparisons)
    df[df.family == 'enlarged_full'].groupby('group')['size'].describe().to_csv(out/'study_E_enlarged_sizes.csv')


def study_f(df, protocol, out):
    cfg = protocol['inference']; a, d, ld = cfg['alpha'], cfg['material_delta'], cfg['legacy_delta']
    groups = _order(df.group.unique())

    def vec(g, pol):
        return df[(df.group == g) & (df.policy == pol)].sort_values('seed').exante_loss.to_numpy()
    prim = []
    for g in groups:
        for better, worse, name in protocol['study_F']['primary_contrasts']:
            prim.append(({'group': g, 'contrast': name}, vec(g, worse)-vec(g, better)))
    p = v3s.family_table(prim, a, d, ld); assert len(p) == protocol['study_F']['primary_family_size']
    p.to_csv(out/'study_F_primary_statistics.csv', index=False)
    sec = []
    for g in groups:
        for better, worse, name in protocol['study_F']['secondary_contrasts']:
            sec.append(({'group': g, 'contrast': name}, vec(g, worse)-vec(g, better)))
    s = v3s.family_table(sec, a, d, ld); assert len(s) == protocol['study_F']['secondary_family_size']
    s.to_csv(out/'study_F_secondary_statistics.csv', index=False)
    summ = df.groupby(['group', 'policy'], as_index=False).agg(
        states=('seed', 'nunique'), exante_loss=('exante_loss', 'mean'), gain_vs_anchor=('exante_gain_vs_anchor', 'mean'),
        k=('k', 'mean'), equals_anchor=('equals_anchor', 'mean'), seconds=('seconds', 'mean'),
        q_covers_p=('q_covers_p', 'mean'), exceed_prob_mean=('exceed_prob', 'mean'), exceed_prob_max=('exceed_prob', 'max'))
    summ.to_csv(out/'study_F_policy_summary.csv', index=False)
    dec = df[df.policy == 'Safe-MinRisk'].groupby('group', as_index=False)[
        ['H_post', 'H_pre_plugin', 'H_pre_crossfit', 'family_size', 'K', 'mk_gap']].mean()
    dec.to_csv(out/'study_F_decomposition.csv', index=False)


def study_g(df, protocol, out):
    cfg = protocol['inference']; a, d, ld = cfg['alpha'], cfg['material_delta'], cfg['legacy_delta']
    cells = list(protocol['study_G']['cells'])

    def vec(c, pol):
        return df[(df.cell == c) & (df.policy == pol)].sort_values('seed').exante_loss.to_numpy()
    prim = []
    for c in cells:
        for comp in protocol['study_G']['primary_competitors']:
            prim.append(({'cell': c, 'comparator': comp}, vec(c, comp)-vec(c, 'DC-PRISM')))
    p = v3s.family_table(prim, a, d, ld); assert len(p) == protocol['study_G']['primary_family_size']
    p.to_csv(out/'study_G_primary_statistics.csv', index=False)
    summ = df.groupby(['cell', 'policy'], as_index=False).agg(
        worlds=('seed', 'nunique'), exante_loss=('exante_loss', 'mean'), gain_vs_anchor=('exante_gain_vs_anchor', 'mean'),
        k=('k', 'mean'), equals_anchor=('equals_anchor', 'mean'), seconds=('seconds', 'mean'),
        exceed_prob_mean=('exceed_prob', 'mean'), q_covers_p=('q_covers_p', 'mean'))
    summ.to_csv(out/'study_G_policy_summary.csv', index=False)
    dec = df[df.policy == 'Safe-MinRisk'].groupby('cell', as_index=False).agg(
        H_post=('H_post', 'mean'), H_pre_plugin=('H_pre_plugin', 'mean'), H_pre_crossfit=('H_pre_crossfit', 'mean'),
        family_size=('family_size', 'mean'), K=('K', 'mean'), n_C=('n_C', 'mean'), mk_gap=('mk_gap', 'mean'))
    dec['foresight_share'] = 1-dec.H_pre_plugin/dec.H_post
    dec.to_csv(out/'study_G_decomposition.csv', index=False)


def study_h(data, protocol, out):
    per_seed, traj = data
    cfg = protocol['inference']; a, d, ld = cfg['alpha'], cfg['material_delta'], cfg['legacy_delta']
    on = protocol['study_H']; prim, auc = [], []
    for stream in on['streams']:
        for ds in on['datasets']:
            g = per_seed[(per_seed.stream == stream) & (per_seed.dataset == ds)]
            b = g.pivot(index='seed', columns='policy', values='alc_brier'); u = g.pivot(index='seed', columns='policy', values='alc_auc')
            assert len(b) == on['n_seeds']
            for comp in on['primary_competitors']:
                prim.append(({'stream': stream, 'dataset': ds, 'comparator': comp}, (b[comp]-b['DC-PRISM']).to_numpy()))
                auc.append(({'stream': stream, 'dataset': ds, 'comparator': comp}, (u['DC-PRISM']-u[comp]).to_numpy()))
    p = v3s.family_table(prim, a, d, ld); assert len(p) == on['primary_family_size']
    p.to_csv(out/'study_H_primary_statistics.csv', index=False)
    q = v3s.family_table(auc, a, on['auc_material_delta'], ld)
    q.to_csv(out/'study_H_auc_statistics.csv', index=False)
    summ = per_seed.groupby(['stream', 'dataset', 'policy'], as_index=False).agg(
        seeds=('seed', 'nunique'), alc_brier=('alc_brier', 'mean'), alc_sd=('alc_brier', 'std'), alc_auc=('alc_auc', 'mean'),
        alc_logloss=('alc_logloss', 'mean'), labels_per_round=('labels_per_round', 'mean'),
        exploration_loss=('exploration_loss', 'mean'), adverse_labels=('adverse_labels', 'mean'),
        select_seconds=('select_seconds', 'mean'), max_envelope_ratio=('max_envelope_ratio', 'max'))
    summ.to_csv(out/'study_H_summary.csv', index=False)
    t = traj[traj.k > 0].copy()
    rows = []
    for pol, x in t.groupby('policy'):
        n = len(x); e = int((x.realized_loss > x.envelope+1e-12).sum()); eb = int((x.realized_loss > x.B+1e-12).sum())
        up = float(beta.ppf(.95, e+1, n-e)) if e < n else 1.
        rows.append(dict(policy=pol, rounds_with_acquisition=n, exceed_envelope=e, exceed_budget=eb,
                         exceed_envelope_cp95_upper=up, mean_realized_over_envelope=float((x.realized_loss/x.envelope).mean())))
    pd.DataFrame(rows).to_csv(out/'study_H_exceedance.csv', index=False)


def run(study, df, protocol, out):
    {'E': study_e, 'F': study_f, 'G': study_g, 'H': study_h}[study](df, protocol, out)
