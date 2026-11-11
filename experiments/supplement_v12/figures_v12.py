"""Figures of the v12 revision, from committed result files only.
Usage: python experiments/supplement_v12/figures_v12.py <out_dir>
  fig_predict.pdf : (a) residual share vs noise index, plug-in..cross-fit bracket (v12 CORE, logistic) and neural learner (v10);
                    (b) residual share under injected label flips (v11-D); (c) greedy vs exhaustive-search residual,
                    12- (v11-B), 16- and 20-candidate pools (v12 EXACT).
  fig_bench.pdf   : benchmark-style step: shortfall of the best practical strategy behind the oracle, as reported
                    (oracle and strategy scored on the evaluation half used for selection) and after removing reuse
                    and luck (cross-fitted ex-ante oracle, held-out half), along the learning curve.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[2]/'results'
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.size': 7, 'axes.titlesize': 7, 'axes.labelsize': 7, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
                     'legend.fontsize': 6, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
NM = {'BankMarketing': 'Bank', 'UCI_Credit': 'UCI', 'Diabetes': 'Readm.', 'LendingClub': 'LC', 'Spambase': 'Spam', 'USPSraw': 'USPS raw',
      'USPS': 'USPS', 'HMEQ': 'HMEQ', 'PPDai': 'PPDai'}
PAL = {'USPSraw': '#1b9e77', 'USPS': '#54a24b', 'Spambase': '#72b7b2', 'Diabetes': '#b279a2', 'BankMarketing': '#e45756',
       'HMEQ': '#4c78a8', 'PPDai': '#9d755d', 'UCI_Credit': '#f58518', 'LendingClub': '#bab0ac'}

# ---------------------------------------------------------------- fig_predict
core = pd.read_csv(R/'supplement_v12/CORE/E1_greedy_bracket.csv'); core = core[core.R == core.R.max()].sort_values('noise_index')
x = pd.read_csv(R/'supplement_v10/partA/states.csv.gz'); x = x[x.part == 'oracle']
g10 = pd.DataFrame([(ds, y.noise_index.mean(), (y.ex_post-y.deployable).mean()/y.reported.mean()) for ds, y in x.groupby('dataset')],
                   columns=['ds', 'noise', 'share']).sort_values('noise')
Dd = pd.read_csv(R/'supplement_v11/D_dose_and_pool.csv'); Dd = Dd[Dd['sub'] == 'flip']
Bx = pd.read_csv(R/'supplement_v11/B_exact_vs_greedy.csv'); Bx = Bx[Bx.dataset != 'ALL']
Ex = pd.read_csv(R/'supplement_v12/EXACT/E4_exact_vs_search.csv'); Ex = Ex[(Ex.R == Ex.R.max()) & (Ex.method == 'greedy')]

fig, (a, b, c) = plt.subplots(1, 3, figsize=(4.8, 1.9))
a.vlines(core.noise_index, 100*core.share_res_plug, 100*core.share_res_cf, color='#4c78a8', lw=1.6, alpha=.75)
a.scatter(core.noise_index, 100*core.share_res_cf, color='#4c78a8', s=7, lw=0, marker='o', label='logistic (cross-fit end)')
a.scatter(core.noise_index, 100*core.share_res_plug, color='white', edgecolor='#4c78a8', s=7, lw=.6, marker='o', label='logistic (plug-in end)')
a.plot(g10.noise, 100*g10.share, color='#54a24b', lw=.6, alpha=.5)
a.scatter(g10.noise, 100*g10.share, color='#54a24b', marker='^', s=8, lw=0, label='neural (MLP)')
a.set_ylim(0, 100); a.set_xlabel('noise index'); a.set_ylabel('residual share of gap (%)')
a.set_title('(a) across cohorts', fontsize=6.5); a.legend(frameon=False, fontsize=4.6, loc='upper left', handletextpad=.1)
pal = {'HMEQ': '#4c78a8', 'BankMarketing': '#e45756', 'USPS': '#54a24b'}
for ds, y in Dd.groupby('dataset'):
    y = y.sort_values('level'); b.plot(100*y.level, 100*y.floor_share, marker='o', ms=2.5, lw=.9, color=pal[ds], label=NM.get(ds, ds))
b.set_ylim(0, 100); b.set_xticks([0, 10, 20, 30]); b.set_xlabel('labels flipped (%)'); b.set_ylabel('residual share of gap (%)')
b.set_title('(b) injected label noise', fontsize=6.5); b.legend(frameon=False, fontsize=5, loc='lower right', handletextpad=.2)
lo, hi = 0.4, 2.6; c.plot((lo, hi), (lo, hi), color='#999999', lw=.7, ls=':')
c.scatter(Bx.exact_floor*1e3, Bx.greedy_floor*1e3, s=9, color='#4c78a8', lw=0, marker='o', label='12 candidates')
for pool, mk, col in [(16, 's', '#f58518'), (20, '^', '#e45756')]:
    z = Ex[Ex.pool == pool]
    c.scatter((z.res_cf-z.res_cf_minus_exact)*1e3, z.res_cf*1e3, s=9, color=col, lw=0, marker=mk, label=f'{pool} candidates')
c.set_xlim(lo, hi); c.set_ylim(lo, hi); c.set_xlabel(r'exact residual ($10^{-3}$)'); c.set_ylabel(r'greedy residual ($10^{-3}$)')
c.set_title('(c) greedy vs exact search', fontsize=6.5); c.legend(frameon=False, fontsize=4.6, loc='upper left', handletextpad=.1)
for ax in (a, b): ax.grid(axis='y', color='#eeeeee', lw=.5)
fig.tight_layout(w_pad=.5); fig.savefig(out/'fig_predict.pdf'); fig.savefig(out/'fig_predict.png', dpi=200)

# ---------------------------------------------------------------- fig_bench
panels = [('BENCH', 'logistic learner'), ('BENCHMLP', 'neural learner (MLP)')]
fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.1), sharey=True)
for ax, (part, title) in zip(axes, panels):
    p = R/f'supplement_v12/{part}/E2_benchmark_summary.csv'
    if not p.exists():
        continue
    e = pd.read_csv(p); pc = [k[8:] for k in e.columns if k.startswith('prac_T2_')]
    for ds, z in e.groupby('dataset'):
        z = z.sort_values('n_hist')
        rep = (z.reported.to_numpy()[:, None]-z[[f'prac_T1_{q}' for q in pc]].to_numpy()).min(axis=1)
        cf = (z.cf.to_numpy()[:, None]-z[[f'prac_T2_{q}' for q in pc]].to_numpy()).min(axis=1)
        col = PAL.get(ds, '#555555')
        ax.plot(z.n_hist, np.maximum(1e3*rep, .03), color=col, lw=.8, ls='--', marker='o', ms=2.4, mfc='white', mew=.6)
        ax.plot(z.n_hist, np.maximum(1e3*cf, .03), color=col, lw=1.1, marker='o', ms=2.4, label=NM.get(ds, ds))
    ax.axhline(.5, color='#999999', lw=.6, ls=':')
    ax.set_xscale('log', base=2); ax.set_yscale('log'); ax.set_ylim(.03, 30); ax.set_xticks([68, 128, 256, 512]); ax.set_xticklabels(['68', '128', '256', '512'])
    ax.set_xlabel('labelled history (records)'); ax.set_title(title, fontsize=6.5); ax.grid(axis='y', color='#eeeeee', lw=.5)
axes[0].set_ylabel(r'shortfall of best strategy ($10^{-3}$)')
h, l = axes[1].get_legend_handles_labels()
h += [plt.Line2D([], [], color='#555555', ls='--', marker='o', mfc='white', ms=2.4, lw=.8), plt.Line2D([], [], color='#555555', lw=1.1, marker='o', ms=2.4)]
l += ['as reported', 'reuse and luck removed']
fig.legend(h, l, frameon=False, fontsize=5, loc='lower center', ncol=5, handletextpad=.3, columnspacing=.8, bbox_to_anchor=(.5, -.02))
fig.tight_layout(rect=(0, .12, 1, 1), w_pad=.6); fig.savefig(out/'fig_bench.pdf'); fig.savefig(out/'fig_bench.png', dpi=200)
