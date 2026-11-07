"""Main figure for the oracle-gap paper: (a) closed-form bracket vs measured aleatoric floor,
(b) aleatoric floor vs deployable gap across settings and learners."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False})
COL = {'controlled': '#222222', 'semi-synthetic': '#4c78a8', 'real-label': '#f58518'}
LAB = {'controlled': 'controlled', 'semi-synthetic': 'semi-synthetic (known $p$)', 'real-label': 'real labels ($\\hat p$)'}


def main(res, out):
    s = pd.read_csv(res/'summary.csv')
    s['setting2'] = np.where(s.group.str.startswith('Diabetes'), 'real-label', s.setting)
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={'width_ratios': [1.25, 1]})
    s = s.sort_values(['learner', 'foresight']).reset_index(drop=True)
    x = 0
    for learner, mk in [('logit', 'o'), ('gbdt', 's')]:
        g = s[s.learner == learner]
        for _, r in g.iterrows():
            a.vlines(x, r.tl_evpi*1e3, r.ind_evpi*1e3, color='#bbbbbb', lw=2.2, zorder=1)
            a.scatter(x, r.foresight*1e3, color=COL[r.setting2], marker=mk, s=14, zorder=3, edgecolor='white', linewidth=.3)
            x += 1
        x += 2
    a.set_yscale('log'); a.set_xticks([9, 9+21]); a.set_xticklabels(['logistic learner', 'boosting learner'])
    a.set_ylabel(r'Aleatoric floor ($\times10^{-3}$ Brier)')
    a.set_title('(a) measured floor (points) inside the closed-form bracket (bars)', fontsize=7.5)
    for k in ['controlled', 'semi-synthetic', 'real-label']:
        a.scatter([], [], color=COL[k], s=14, label=LAB[k])
    a.legend(frameon=False, fontsize=6, loc='upper left')
    for learner, mk in [('logit', 'o'), ('gbdt', 's')]:
        g = s[s.learner == learner]
        for k in COL:
            h = g[g.setting2 == k]
            b.scatter(h.H_pre_crossfit.clip(lower=1e-4)*1e3, h.foresight*1e3, color=COL[k], marker=mk, s=16,
                      edgecolor='white', linewidth=.3, label=f'{learner}' if k == 'controlled' else None, zorder=3)
    lim = (0.1, 30); b.plot(lim, lim, color='#999999', lw=.7, ls=':')
    b.set_xscale('log'); b.set_yscale('log'); b.set_xlim(*lim); b.set_ylim(.3, 30)
    b.set_xlabel(r'Deployable gap $H_{\mathrm{pre}}$ ($\times10^{-3}$)'); b.set_ylabel(r'Aleatoric floor ($\times10^{-3}$)')
    b.set_title('(b) floor vs deployable gap', fontsize=7.5)
    fig.tight_layout(); fig.savefig(out/'fig_floor_main.pdf'); fig.savefig(out/'fig_floor_main.png', dpi=200)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--results', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True); main(a.results, a.out)
