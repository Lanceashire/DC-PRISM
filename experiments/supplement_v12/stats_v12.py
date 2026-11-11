"""Registered summaries and prediction checks for supplement-v12.

Shares are ratios of dataset means (share = mean component / mean reported gap), as in supplement-v9.
Inference: paired one-sample t on independent states, Bonferroni simultaneous intervals and Holm within
each registered family (supplement-v3 family_table), material delta 0.5e-3 Brier.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location('v3_stats_v12', ROOT/'experiments/supplement_v3/stats.py')
v3s = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(v3s)


def _ci(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; n = len(x); m = x.mean()
    se = x.std(ddof=1)/np.sqrt(n) if n > 1 else np.nan; c = t.ppf(.975, n-1) if n > 1 else np.nan
    return float(m), float(m-c*se), float(m+c*se)


def _sp(a, b):
    return float(spearmanr(a, b).statistic)


def _components(x):
    x = x.assign(reuse=x.reported-x.hold, res_cf=x.hold-x.cf, res_plug=x.hold-x.plug)
    return x


def _summ(x):
    m = x[['noise_index', 'reported', 'hold', 'cf', 'plug', 'reuse', 'res_cf', 'res_plug']].mean()
    return dict(states=len(x), **m.to_dict(), share_reuse=m.reuse/m.reported, share_res_cf=m.res_cf/m.reported,
                share_res_plug=m.res_plug/m.reported, width=m.res_cf-m.res_plug)


def run(part, df, P, out):
    inf = P['inference']; a, d, ld = inf['alpha'], inf['material_delta'], inf['legacy_delta']
    pred = {}
    if part == 'CORE':
        o = _components(df[df.kind == 'oracle'].copy()); fam = df[df.kind == 'family'].copy()
        # ---------- E1 greedy bracket and R sweep
        g = o[(o.variant == 'gbdt') & (o.method == 'greedy')]
        tab = pd.DataFrame([dict(dataset=ds, R=R, **_summ(x)) for (ds, R), x in g.groupby(['dataset', 'R'])])
        tab.to_csv(out/'E1_greedy_bracket.csv', index=False)
        Rm = max(P['CORE']['R_greedy_list']); R0 = min(P['CORE']['R_greedy_list'])
        top = tab[tab.R == Rm].set_index('dataset'); low = tab[tab.R == R0].set_index('dataset')
        hi2 = list(top.noise_index.nlargest(2).index)
        pred['E1_plug_le_cf_all'] = bool((top.res_plug <= top.res_cf).all())
        pred['E1_width_shrinks_count'] = int((top.width < low.width).sum())
        pred['E1_spearman_share_cf'] = _sp(top.noise_index, top.share_res_cf)
        pred['E1_spearman_share_plug'] = _sp(top.noise_index, top.share_res_plug)
        pred['E1_high_noise_datasets'] = ','.join(hi2)
        pred['E1_high_noise_min_share_plug'] = float(top.loc[hi2].share_res_plug.min())
        recs = []
        for ds, x in g[g.R == Rm].groupby('dataset'):
            x = x.sort_values('seed')
            for e in ['res_plug', 'res_cf']:
                recs.append(({'dataset': ds, 'endpoint': e}, x[e].to_numpy()))
        v3s.family_table(recs, a, d, ld).to_csv(out/'E1_primary_statistics.csv', index=False)
        # family-level floor bracket
        fv = fam[fam.variant == 'gbdt']
        ft = fv.groupby(['dataset', 'R'])[['noise_index', 'H_post', 'H_pre_plugin', 'H_pre_cf', 'F_plugin', 'F_cf', 'F_lo', 'F_up']].mean()
        ft['share_F_plugin'] = ft.F_plugin/ft.H_post; ft['share_F_cf'] = ft.F_cf/ft.H_post
        ft.to_csv(out/'E1_family_bracket.csv')
        fm = ft.xs(max(P['CORE']['R_family_list']), level='R')
        pred['E1_family_plug_le_cf_all'] = bool((fm.F_plugin <= fm.F_cf+1e-12).all())
        pred['E1_family_Flo_le_Fplug_all'] = bool((fm.F_lo <= fm.F_plugin+1e-12).all())
        pred['E1_family_spearman_share_plug'] = _sp(fm.noise_index, fm.share_F_plugin)
        # ---------- E3 search strength
        Rs = P['CORE']['R_search']; s3 = o[(o.variant == 'gbdt') & (o.R == Rs)]
        rows = []
        for ds, x in s3.groupby('dataset'):
            gx = x[x.method == 'greedy'].set_index('seed').sort_index()
            for m in ['greedy', 'local', 'restarts']:
                mx = x[x.method == m].set_index('seed').loc[gx.index]
                r = dict(dataset=ds, method=m, **_summ(mx.reset_index()))
                for c in ['cf', 'hold', 'res_cf', 'res_plug']:
                    mm, lo, hi = _ci(mx[c]-gx[c]); r.update({f'{c}_minus_greedy': mm, f'{c}_minus_greedy_lo': lo, f'{c}_minus_greedy_hi': hi})
                r['cf_gain_over_greedy_rel_reported'] = r['cf_minus_greedy']/gx.reported.mean()
                rows.append(r)
        e3 = pd.DataFrame(rows); e3.to_csv(out/'E3_search_strength.csv', index=False)
        rs = e3[e3.method == 'restarts'].set_index('dataset')
        pred['E3_cf_rise_le_10pct_count'] = int((rs.cf_gain_over_greedy_rel_reported <= .10).sum())
        pooled = s3.groupby('method')[['cf', 'reported']].mean()
        pred['E3_cf_rise_le_10pct_pooled'] = float((pooled.loc['restarts', 'cf']-pooled.loc['greedy', 'cf'])/pooled.loc['greedy', 'reported'])
        pred['E3_spearman_share_cf_restarts'] = _sp(rs.noise_index, rs.share_res_cf)
        # ---------- E5 outcome-model sensitivity
        s5 = o[(o.method == 'greedy') & (o.R == Rs)]
        e5 = pd.DataFrame([dict(dataset=ds, variant=v, **_summ(x)) for (ds, v), x in s5.groupby(['dataset', 'variant'])])
        fam5 = fam[fam.R == P['CORE']['R_family']].groupby(['dataset', 'variant'])[['F_plugin', 'F_cf', 'H_post']].mean()
        e5 = e5.merge(fam5.reset_index(), on=['dataset', 'variant'])
        cal = df[(df.kind == 'oracle') & (df.variant == 'gbdt') & (df.method == 'greedy') & (df.R == Rs)].groupby('dataset')[
            ['cal_a', 'cal_b', 'cal_b_lo', 'cal_b_hi', 'ece']].mean()
        e5.to_csv(out/'E5_phat_sensitivity.csv', index=False); cal.to_csv(out/'E5_calibration.csv')
        base_rank = e5[e5.variant == 'gbdt'].set_index('dataset').noise_index
        hi2, lo2 = list(base_rank.nlargest(2).index), list(base_rank.nsmallest(2).index)
        for v, x in e5.groupby('variant'):
            x = x.set_index('dataset')
            pred[f'E5_spearman_share_cf_{v}'] = _sp(x.noise_index, x.share_res_cf)
            pred[f'E5_spearman_share_plug_{v}'] = _sp(x.noise_index, x.share_res_plug)
            pred[f'E5_high_minus_low_share_cf_{v}'] = float(x.loc[hi2].share_res_cf.mean()-x.loc[lo2].share_res_cf.mean())
    elif part == 'EXACT':
        x = _components(df.copy()); rows = []
        for (pool, ds, R), z in list(x.groupby(['pool', 'dataset', 'R'])):
            ex = z[z.method == 'exact'].set_index('seed').sort_index()
            for m in ['exact', 'greedy', 'local', 'restarts']:
                mz = z[z.method == m].set_index('seed').loc[ex.index]
                r = dict(pool=pool, dataset=ds, R=R, method=m, n_space=float(mz.n_space.mean()), **_summ(mz.reset_index()))
                for c in ['reuse', 'res_cf', 'res_plug', 'cf', 'plug']:
                    mm, lo, hi = _ci(mz[c]-ex[c]); r.update({f'{c}_minus_exact': mm, f'{c}_minus_exact_lo': lo, f'{c}_minus_exact_hi': hi})
                rows.append(r)
        e4 = pd.DataFrame(rows); e4.to_csv(out/'E4_exact_vs_search.csv', index=False)
        Rm = max(P['EXACT']['R_list']); thr = P['EXACT']['agreement_threshold']
        for pool in sorted(e4.pool.unique()):
            for m in ['greedy', 'local', 'restarts']:
                z = e4[(e4.pool == pool) & (e4.R == Rm) & (e4.method == m)]
                pred[f'E4_pool{pool}_{m}_max_abs_diff'] = float(z[['reuse_minus_exact', 'res_cf_minus_exact', 'cf_minus_exact']].abs().max().max())
                pred[f'E4_pool{pool}_{m}_all_within'] = bool(pred[f'E4_pool{pool}_{m}_max_abs_diff'] <= thr)
            pred[f'E4_pool{pool}_exact_share_res_cf'] = float(e4[(e4.pool == pool) & (e4.R == Rm) & (e4.method == 'exact')].pipe(
                lambda z: z.res_cf.mean()/z.reported.mean()))
    elif part in ('BENCH', 'BENCHMLP'):
        x = _components(df.copy()); pc = sorted({c[8:] for c in x.columns if c.startswith('prac_T2_')}); learner = x.learner.iloc[0]
        for p in pc:
            x[f'short_rep_{p}'] = x.reported-x[f'prac_T1_{p}']; x[f'short_cf_{p}'] = x.cf-x[f'prac_T2_{p}']
            x[f'short_plug_{p}'] = x.plug-x[f'prac_T2_{p}']
            x[f'frac_rep_{p}'] = x[f'prac_T1_{p}']; x[f'frac_cf_{p}'] = x[f'prac_T2_{p}']
        rows = []
        for (ds, h), z in x.groupby(['dataset', 'n_hist']):
            r = dict(learner=learner, dataset=ds, n_hist=h, **_summ(z))
            for p in pc:
                r[f'prac_T1_{p}'] = z[f'prac_T1_{p}'].mean(); r[f'prac_T2_{p}'] = z[f'prac_T2_{p}'].mean()
                r[f'closed_rep_{p}'] = r[f'prac_T1_{p}']/r['reported']; r[f'closed_cf_{p}'] = r[f'prac_T2_{p}']/r['cf'] if r['cf'] > 0 else np.nan
            rep_rank = sorted(pc, key=lambda p: -r[f'prac_T1_{p}']); cf_rank = sorted(pc, key=lambda p: -r[f'prac_T2_{p}'])
            r['top2_rep'] = ','.join(rep_rank[:2]); r['top2_cf'] = ','.join(cf_rank[:2]); r['top2_differs'] = set(rep_rank[:2]) != set(cf_rank[:2])
            rows.append(r)
        e2 = pd.DataFrame(rows); e2.to_csv(out/'E2_benchmark_summary.csv', index=False)
        h0 = P[part]['primary_n_hist']; z0 = x[x.n_hist == h0]; recs = []
        for ds, z in z0.groupby('dataset'):
            z = z.sort_values('seed')
            for p in pc:
                for e in ['short_rep', 'short_cf']:
                    recs.append(({'dataset': ds, 'strategy': p, 'endpoint': e}, z[f'{e}_{p}'].to_numpy()))
        fam = v3s.family_table(recs, a, d, ld); fam.to_csv(out/'E2_primary_statistics.csv', index=False)
        sec = []
        for (ds, h), z in x.groupby(['dataset', 'n_hist']):
            for p in pc:
                for e in ['short_rep', 'short_cf', 'short_plug']:
                    m, lo, hi = _ci(z[f'{e}_{p}']); sec.append(dict(dataset=ds, n_hist=h, strategy=p, endpoint=e, mean=m, ci95_low=lo, ci95_high=hi))
        pd.DataFrame(sec).to_csv(out/'E2_all_positions_pointwise.csv', index=False)
        noise = e2[e2.n_hist == h0].set_index('dataset').noise_index
        hi2 = list(noise.nlargest(2).index); pred[f'E2_{learner}_high_noise'] = ','.join(hi2)
        dec = fam.set_index(['dataset', 'strategy', 'endpoint']).decision
        for ds in noise.index:
            rep_all = all(dec[(ds, p, 'short_rep')] == 'material_positive' for p in pc)
            cf_all = all(dec[(ds, p, 'short_cf')] == 'material_positive' for p in pc)
            pred[f'E2_{learner}_{ds}_rep_all_material'] = bool(rep_all)
            pred[f'E2_{learner}_{ds}_cf_all_material'] = bool(cf_all)
            pred[f'E2_{learner}_{ds}_conclusion_changes'] = bool(rep_all and not cf_all)
        pred[f'E2_{learner}_prediction_high_noise_changes'] = bool(all(pred[f'E2_{learner}_{ds}_conclusion_changes'] for ds in hi2))
        if learner == 'mlp' and 'USPSraw' in noise.index:
            pred['E2_mlp_USPSraw_cf_all_material'] = pred['E2_mlp_USPSraw_cf_all_material']
        pred[f'E2_{learner}_top2_differs_any'] = bool(e2.top2_differs.any())
    elif part == 'LATER':
        o = _components(df[df.kind == 'oracle'].copy()); fam = df[df.kind == 'family']
        tab = pd.DataFrame([dict(dataset=ds, n_hist=h, **_summ(x)) for (ds, h), x in o.groupby(['dataset', 'n_hist'])])
        ft = fam.groupby(['dataset', 'n_hist'])[['H_post', 'H_pre_plugin', 'H_pre_cf', 'F_plugin', 'F_cf']].mean().reset_index()
        tab = tab.merge(ft, on=['dataset', 'n_hist']); tab.to_csv(out/'E6_later_positions.csv', index=False)
        for h, z in tab.groupby('n_hist'):
            pred[f'E6_spearman_share_cf_n{h}'] = _sp(z.noise_index, z.share_res_cf)
    pd.Series(pred, dtype=object).to_csv(out/'prediction_check.csv')
