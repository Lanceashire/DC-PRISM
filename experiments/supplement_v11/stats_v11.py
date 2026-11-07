"""Registered summaries for supplement-v11 (revision studies)."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t


def _ci(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; n = len(x); m = x.mean()
    se = x.std(ddof=1)/np.sqrt(n) if n > 1 else np.nan; c = t.ppf(.975, n-1) if n > 1 else np.nan
    return float(m), float(m-c*se), float(m+c*se), n


def _tost(x, delta):
    """Equivalence within +-delta: both one-sided 95% tests reject (90% CI inside the band)."""
    x = np.asarray(x, float); x = x[np.isfinite(x)]; n = len(x); m = x.mean(); se = x.std(ddof=1)/np.sqrt(n)
    lo, hi = m-t.ppf(.95, n-1)*se, m+t.ppf(.95, n-1)*se
    return dict(mean=float(m), ci90_low=float(lo), ci90_high=float(hi), equivalent=bool(lo > -delta and hi < delta), n=n)


def run(df, P, out):
    delta = P['inference']['material_delta']
    # ---------------- A
    a = df[df.part == 'A']
    if len(a):
        a = a.assign(lo_ok=a.F_lo <= a.F+1e-12, up_ok=a.F <= a.F_up+1e-12, gauss_ok=a.F <= a.G+1e-12, ind_fail=a.F_ind < a.F-1e-12,
                     width_ratio=a.F_up/a.F.where(a.F > 1e-9))
        g = a.groupby(['kind', 'group', 'learner']).agg(states=('seed', 'size'), F=('F', 'mean'), F_lo=('F_lo', 'mean'), F_up=('F_up', 'mean'),
                                                       G=('G', 'mean'), F_ind=('F_ind', 'mean'), F_two_normal=('F_two_normal', 'mean'),
                                                       lo_ok=('lo_ok', 'mean'), up_ok=('up_ok', 'mean'), gauss_ok=('gauss_ok', 'mean'),
                                                       ind_fail=('ind_fail', 'mean'), neg_corr_share=('neg_corr_share', 'mean'),
                                                       skew_D=('skew_D', 'median'), width_ratio=('width_ratio', 'median'))
        g.to_csv(out/'A_bounds_by_group.csv')
        s = df[df.part == 'stress']
        st = s.assign(up_ok=s.F <= s.F_up+1e-12, lo_ok=s.F_lo <= s.F+1e-12, gauss_ok=s.F <= s.G+1e-12, ind_fail=s.F_ind < s.F-1e-12)\
              .groupby(['structure', 'K'])[['F', 'F_lo', 'F_up', 'G', 'F_ind', 'lo_ok', 'up_ok', 'gauss_ok', 'ind_fail']].mean()
        st.to_csv(out/'A_stress.csv')
        pd.Series(dict(states=len(a), lo_ok=a.lo_ok.mean(), up_ok=a.up_ok.mean(), gauss_ok=a.gauss_ok.mean(), ind_fail_states=a.ind_fail.mean(),
                       group_mean_inside_lo_up=float(((g.F_lo <= g.F) & (g.F <= g.F_up)).mean()),
                       group_mean_inside_lo_G=float(((g.F_lo <= g.F) & (g.F <= g.G)).mean()))).to_csv(out/'A_summary.csv')
    # ---------------- B
    b = df[df.part == 'B']
    if len(b):
        pc = [c for c in b.columns if c.startswith('prac_')]
        rows = []
        for ds, x in list(b.groupby('dataset'))+[('ALL', b)]:
            r = dict(dataset=ds, states=len(x), n_space=x.n_space.mean(), n_family=x.n_family.mean())
            for o in ['exact', 'family', 'greedy']:
                rep, post, dep = x[f'{o}_reported'], x[f'{o}_expost'], x[f'{o}_deployable']
                r.update({f'{o}_reported': rep.mean(), f'{o}_reuse': (rep-post).mean(), f'{o}_floor': (post-dep).mean(), f'{o}_deployable': dep.mean()})
            best = x[pc].mean().idxmax(); r['best_fixed'] = best[5:]; r['best_fixed_gain'] = x[best].mean()
            r['greedy_deployable_lb'] = np.maximum(x.greedy_deployable, x[best]).mean()
            for k in ['floor', 'deployable']:
                gk = (x.greedy_expost-x.greedy_deployable) if k == 'floor' else x.greedy_deployable
                ek = (x.exact_expost-x.exact_deployable) if k == 'floor' else x.exact_deployable
                m, lo, hi, n = _ci(gk-ek); r.update({f'greedy_minus_exact_{k}': m, f'greedy_minus_exact_{k}_lo': lo, f'greedy_minus_exact_{k}_hi': hi})
            r['exact_floor_share'] = r['exact_floor']/r['exact_reported']; r['greedy_floor_share'] = r['greedy_floor']/r['greedy_reported']
            rows.append(r)
        pd.DataFrame(rows).to_csv(out/'B_exact_vs_greedy.csv', index=False)
    # ---------------- C1
    c1 = df[df.part == 'C1']
    if len(c1):
        tr = c1[c1.variant == 'truth'].set_index(['group', 'seed']); rows = []
        for v in [v for v in c1.variant.unique() if v != 'truth']:
            e = c1[c1.variant == v].set_index(['group', 'seed']).loc[tr.index]
            for q in ['F', 'H_pre_crossfit', 'H_post']:
                dlt = (e[q]-tr[q]).to_numpy(); m, lo, hi, n = _ci(dlt)
                rows.append(dict(variant=v, quantity=q, truth_mean=float(tr[q].mean()), est_mean=float(e[q].mean()), bias=m, bias_lo=lo, bias_hi=hi,
                                 rmse=float(np.sqrt(np.mean(dlt**2))), within_material=bool(lo > -delta and hi < delta), n=n))
        pd.DataFrame(rows).to_csv(out/'C1_phat_sensitivity.csv', index=False)
    # ---------------- C2
    c2 = df[df.part == 'C2']
    if len(c2):
        rows = []
        for ds, x in list(c2.groupby('dataset'))+[('ALL', c2)]:
            r = dict(dataset=ds, states=len(x), ece=x.ece.mean(), cal_slope=x.cal_slope.mean(), holdout_brier=x.holdout_brier.mean(),
                     model_reported=x.model_reported.mean(), real_reported=x.real_reported.mean(),
                     model_expost=x.model_expost.mean(), real_expost=x.real_expost.mean())
            for q in ['reported', 'expost']:
                tt = _tost((x[f'model_{q}']-x[f'real_{q}']).to_numpy(), delta)
                r.update({f'{q}_diff': tt['mean'], f'{q}_ci90_low': tt['ci90_low'], f'{q}_ci90_high': tt['ci90_high'], f'{q}_equivalent': tt['equivalent']})
            rows.append(r)
        pd.DataFrame(rows).to_csv(out/'C2_greedy_calibration.csv', index=False)
    # ---------------- D
    d = df[df.part == 'D']
    if len(d):
        d = d.assign(floor_greedy=d.ex_post-d.deployable, reuse=d.reported-d.ex_post)
        g = d.groupby(['sub', 'dataset', 'level']).agg(states=('seed', 'size'), noise_index=('noise_index', 'mean'), K=('K', 'mean'),
                                                      F_family=('F', 'mean'), F_lo=('F_lo', 'mean'), F_up=('F_up', 'mean'), G=('G', 'mean'),
                                                      reported=('reported', 'mean'), reuse=('reuse', 'mean'), floor_greedy=('floor_greedy', 'mean'),
                                                      deployable=('deployable', 'mean'), n_sets=('n_sets', 'mean'))
        g['floor_share'] = g.floor_greedy/g.reported
        g.to_csv(out/'D_dose_and_pool.csv')
        rows = []
        for (sub, ds), x in g.reset_index().groupby(['sub', 'dataset']):
            x = x.sort_values('level')
            rows.append(dict(sub=sub, dataset=ds, monotone_F=bool(np.all(np.diff(x.F_family) > 0)),
                             monotone_share=bool(np.all(np.diff(x.floor_share) > 0)),
                             spearman_level_share=float(spearmanr(x.level, x.floor_share).statistic)))
        pd.DataFrame(rows).to_csv(out/'D_monotonicity.csv', index=False)
