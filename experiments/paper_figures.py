"""Figures of the oracle-gap paper, from committed result files only.
Usage: python experiments/paper_figures.py <out_dir>
  fig_predict.pdf : (a) greedy-residual share of the benchmark-oracle gap vs label-noise index (logistic, neural);
                    (b) the same share under injected label flips; (c) greedy vs exhaustive-search residual, 12-candidate pools.
  fig_budget.pdf  : floor (and deployable gap) vs risk budget k, eight cohorts.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1]/'results'
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.size': 7, 'axes.titlesize': 7, 'axes.labelsize': 7, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 6, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
LEARN = {'logit': ('logistic', '#4c78a8', 'o'), 'gbdt': ('boosting', '#f58518', 's'), 'mlp': ('neural (MLP)', '#54a24b', '^')}

# ---------------------------------------------------------------- data
v7 = pd.read_csv(R/'supplement_v7/summary.csv')[['group', 'learner', 'foresight', 'tl_evpi', 'ind_evpi']]
v9 = pd.read_csv(R/'supplement_v9/family_summary.csv').rename(columns={'dataset': 'group'})
v10a = pd.read_csv(R/'supplement_v10/partA/family_summary.csv').rename(columns={'dataset': 'group'}).assign(learner='mlp')
v10b = pd.read_csv(R/'supplement_v10/partB/family_summary.csv').rename(columns={'dataset': 'group'}).assign(learner='logit')
cols = ['group', 'learner', 'foresight', 'tl_evpi', 'ind_evpi']
allf = pd.concat([v7[cols], v9[cols], v10a[cols], v10b[cols]], ignore_index=True)
assert len(allf) == 87

# ---------------------------------------------------------------- fig_predict: (a) noise gradient, (b) injected noise, (c) exact vs greedy
def grad(path):
    x = pd.read_csv(path); x = x[x.part == 'oracle']; rows = []
    for ds, y in x.groupby('dataset'):
        rows.append((ds, y.noise_index.mean(), (y.ex_post-y.deployable).mean()/y.reported.mean()))
    return pd.DataFrame(rows, columns=['ds', 'noise', 'share']).sort_values('noise')
g9, g10 = grad(R/'supplement_v9/states.csv.gz'), grad(R/'supplement_v10/partA/states.csv.gz')
Dd = pd.read_csv(R/'supplement_v11/D_dose_and_pool.csv'); Dd = Dd[Dd['sub'] == 'flip']
Bx = pd.read_csv(R/'supplement_v11/B_exact_vs_greedy.csv'); Bx = Bx[Bx.dataset != 'ALL']
fig, (a, b, c) = plt.subplots(1, 3, figsize=(4.8, 1.85))
for df, lr in [(g9, 'logit'), (g10, 'mlp')]:
    lab, col, mk = LEARN[lr]
    a.plot(df.noise, 100*df.share, color=col, lw=.7, alpha=.5); a.scatter(df.noise, 100*df.share, color=col, marker=mk, s=9, lw=0, label=lab)
a.set_ylim(0, 100); a.set_xlabel('noise index'); a.set_ylabel('residual share of gap (%)')
a.set_title('(a) across cohorts', fontsize=6.5); a.legend(frameon=False, fontsize=5, loc='upper left', handletextpad=.1)
pal = {'HMEQ': '#4c78a8', 'BankMarketing': '#e45756', 'USPS': '#54a24b'}
for ds, x in Dd.groupby('dataset'):
    x = x.sort_values('level'); b.plot(100*x.level, 100*x.floor_share, marker='o', ms=2.5, lw=.9, color=pal[ds], label={'BankMarketing': 'Bank'}.get(ds, ds))
b.set_ylim(0, 100); b.set_xticks([0, 10, 20, 30]); b.set_xlabel('labels flipped (%)'); b.set_ylabel('residual share of gap (%)')
b.set_title('(b) injected label noise', fontsize=6.5); b.legend(frameon=False, fontsize=5, loc='lower right', handletextpad=.2)
lim = (0.5, 2.0); c.plot(lim, lim, color='#999999', lw=.7, ls=':')
c.scatter(Bx.exact_floor*1e3, Bx.greedy_floor*1e3, s=10, color='#4c78a8', lw=0)
nm = {'BankMarketing': 'Bank', 'UCI_Credit': 'UCI', 'Diabetes': 'Readm.', 'LendingClub': 'LC', 'Spambase': 'Spam'}
for _, r in Bx.iterrows():
    off = {'Spambase': (-14, 3), 'PPDai': (-14, 3), 'UCI_Credit': (-10, 3)}.get(r.dataset, (3, -4))
    c.annotate(nm.get(r.dataset, r.dataset), (r.exact_floor*1e3, r.greedy_floor*1e3), fontsize=4.3, xytext=off, textcoords='offset points', color='#444444')
c.set_xlim(*lim); c.set_ylim(*lim); c.set_xlabel(r'exact-search residual ($10^{-3}$)'); c.set_ylabel(r'greedy residual ($10^{-3}$)')
c.set_title('(c) greedy vs exact search', fontsize=6.5)
for ax in (a, b): ax.grid(axis='y', color='#eeeeee', lw=.5)
fig.tight_layout(w_pad=.5); fig.savefig(out/'fig_predict.pdf'); fig.savefig(out/'fig_predict.png', dpi=200)

# ---------------------------------------------------------------- fig_budget
fig, (a, b) = plt.subplots(1, 2, figsize=(4.8, 2.35), sharex=True)
nm = {'BankMarketing': 'Bank', 'UCI_Credit': 'UCI Credit', 'Diabetes': 'Readmission', 'LendingClub': 'LendingClub'}
cmap = plt.cm.tab10
for j, (ds, x) in enumerate(v10b.groupby('group')):
    x = x.sort_values('target_k'); c = cmap(j); mk = 'osD^vPXh'[j]
    a.plot(x.target_k, x.foresight*1e3, color=c, marker=mk, ms=2.8, lw=0.9, label=nm.get(ds, ds))
    b.plot(x.target_k, x.H_pre_crossfit*1e3, color=c, marker=mk, ms=2.8, lw=0.9)
for ax, t, yl in [(a, '(a) floor rises with the budget (8/8 cohorts)', r'Aleatoric floor ($\times10^{-3}$)'),
                  (b, '(b) deployable gap', r'Deployable $H_{\mathrm{pre}}$ ($\times10^{-3}$)')]:
    ax.set_xscale('log', base=2); ax.set_xticks([3, 6, 12]); ax.set_xticklabels(['3', '6', '12'])
    ax.set_xlabel('Budget size $k$ (lowest-risk labels)'); ax.set_ylabel(yl); ax.set_title(t, fontsize=7)
    ax.grid(axis='y', color='#eeeeee', lw=.5)
b.set_yscale('log')
h, l = a.get_legend_handles_labels()
fig.tight_layout(rect=(0, 0.12, 1, 1))
fig.legend(h, l, frameon=False, fontsize=5.5, ncol=4, loc='lower center', bbox_to_anchor=(0.5, 0.0), handlelength=1.6, columnspacing=1.0)
fig.savefig(out/'fig_budget.pdf'); fig.savefig(out/'fig_budget.png', dpi=200)
print('figures written to', out)
