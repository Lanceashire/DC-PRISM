"""fig_curves.pdf: learning curves with three oracles (supplement-v13 MAIN), from committed results only.
(a) whole-curve lead of the test-set oracle over the best strategy, split into test reuse, outcome luck and the remainder;
(b) the three parts along the learning curve, mean over cohorts.
Usage: python experiments/supplement_v13/figures_v13.py <out_dir>"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[2]/'results/supplement_v13/MAIN'
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.size': 7, 'axes.labelsize': 7, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 6,
                     'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
NM = {'BankMarketing': 'Bank', 'UCI_Credit': 'UCI Credit', 'Diabetes': 'Readmission', 'LendingClub': 'LendingClub', 'Spambase': 'Spambase',
      'USPS': 'USPS', 'HMEQ': 'HMEQ', 'PPDai': 'PPDai'}
COL = {'reuse': '#9e9e9e', 'luck': '#f58518', 'rem': '#4c78a8'}
g = pd.read_csv(R/'summary_all.csv').sort_values('noise_index')
d = pd.read_csv(R/'rows.csv.gz'); d = d[d.step > 0]
fig, (a, b) = plt.subplots(1, 2, figsize=(4.8, 2.0), gridspec_kw={'width_ratios': [1.1, 1]})
y = np.arange(len(g)); r, l, m = 1e3*g.reuse.values, 1e3*g.luck.values, 1e3*g.remaining.values
a.barh(y, r, color=COL['reuse'], label='test reuse')
a.barh(y, np.maximum(l, 0), left=r, color=COL['luck'], label='outcome luck')
a.barh(y, np.maximum(m, 0), left=r+np.maximum(l, 0), color=COL['rem'], label='remaining')
a.set_yticks(y); a.set_yticklabels([f"{NM[k]} ({v:.2f})" for k, v in zip(g.dataset, g.noise_index)]); a.invert_yaxis()
a.set_xlabel(r'lead over best strategy, whole curve ($10^{-3}$)'); a.set_title('(a) cohorts (noise index)', fontsize=6.5)
a.legend(frameon=False, fontsize=5, loc='lower right')
p = d.pivot_table(index=['dataset', 'seed', 'step'], columns='method', values='brier_T')
best = dict(zip(g.dataset, g.best_strategy)); rows = []
for (ds, s, t), x in p.iterrows():
    rows.append((ds, t, x['oracle-split']-x['oracle-test'], x['oracle-exante']-x['oracle-split'], x[best[ds]]-x['oracle-exante']))
c = pd.DataFrame(rows, columns=['ds', 'step', 'reuse', 'luck', 'rem']).groupby(['ds', 'step']).mean().groupby('step').mean()*1e3
n = 68+10*c.index.values
for k, lab in [('reuse', 'test reuse'), ('luck', 'outcome luck'), ('rem', 'remaining')]:
    b.plot(n, c[k].rolling(3, center=True, min_periods=1).mean(), color=COL[k], lw=1.2, label=lab)
b.axhline(0, color='#bbbbbb', lw=.6); b.set_xlabel('labelled records'); b.set_ylabel(r'mean over cohorts ($10^{-3}$)')
b.set_title('(b) along the learning curve', fontsize=6.5); b.grid(axis='y', color='#eeeeee', lw=.5)
fig.tight_layout(w_pad=.8); fig.savefig(out/'fig_curves.pdf'); fig.savefig(out/'fig_curves.png', dpi=200)
