"""Figures for supplement-v5 from aggregate result tables only."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False})
COMPS = ['Safe-MinRisk', 'Fixed-count', 'Random-Feasible', 'Uncertainty', 'Fisher-cost', 'BAIT-Greedy', 'TargetVar-Greedy']
MARKERS = ['o', 's', 'D', '^', 'v', 'P', 'X']   # secondary encoding: identity is not colour-alone


def online(res, out):
    p = pd.read_csv(res/'study_H_primary_statistics.csv')
    colors = dict(zip(COMPS, plt.cm.tab10(np.arange(len(COMPS)))))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
    for ax, stream in zip(axes, ['engineered', 'natural']):
        g = p[p.stream == stream]
        for i, ds in enumerate(['HMEQ', 'LendingClub', 'PPDai', 'UCI_Credit']):
            for j, c in enumerate(COMPS):
                r = g[(g.dataset == ds) & (g.comparator == c)].iloc[0]; y = i + (j-3)*0.11
                ax.errorbar(r['mean']*1e3, y, xerr=[[(r['mean']-r.simultaneous_low)*1e3], [(r.simultaneous_high-r['mean'])*1e3]],
                            fmt=MARKERS[j], ms=3, lw=.8, color=colors[c], label=c if i == 0 else None)
        ax.axvline(0, color='#333333', lw=.6); ax.axvspan(-.5, .5, color='#eeeeee', zorder=0)
        ax.set_title(f'{stream} stream (512 seeds per cohort)', fontsize=8)
        ax.set_xlabel(r'Comparator $-$ DC-PRISM ALC-Brier ($\times10^{-3}$)')
    axes[0].set_yticks(range(4)); axes[0].set_yticklabels(['HMEQ', 'LC', 'PPDai', 'UCI']); axes[0].invert_yaxis()
    axes[1].legend(frameon=False, fontsize=6, loc='center left', bbox_to_anchor=(1, .5))
    fig.tight_layout(); fig.savefig(out/'supplement_v5_online.pdf'); fig.savefig(out/'supplement_v5_online.png', dpi=200)


def family_size(res, out):
    s = pd.read_csv(res/'study_E_summary.csv')
    r = s[s.family == 'random_subfamily']
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    styles = {'broad': dict(color='#4c78a8', ls='-'), 'gap': dict(color='#f58518', ls='--')}
    seen = set()
    for g, x in r.groupby('group'):
        x = x.sort_values('size')
        if g == 'Controlled':
            ax.plot(x['size'], 1-x.foresight_share, color='#222222', lw=2, marker='o', ms=3, label='controlled', zorder=3)
            continue
        arm = g.split('/')[1]
        lab = f'real covariates, {arm} history' if arm not in seen else None; seen.add(arm)
        ax.plot(x['size'], 1-x.foresight_share, lw=1, marker='.', ms=3, label=lab, **styles[arm])
    ax.set_xscale('log', base=2); ax.set_xticks([2, 4, 8, 16, 32, 64]); ax.set_xticklabels(['2', '4', '8', '16', '32', '64'])
    ax.set_ylim(0, 1); ax.set_xlabel(r'Family size $|\mathcal{F}|$ (random sub-families containing $M_K$)')
    ax.set_ylabel(r'Pre-outcome share $\widehat H_{\mathrm{pre}}/H_{\mathrm{post}}$')
    ax.grid(axis='y', color='#e5e5e5', lw=.5); ax.legend(frameon=False, fontsize=6.5, loc='center right')
    fig.tight_layout(); fig.savefig(out/'supplement_v5_family_size.pdf'); fig.savefig(out/'supplement_v5_family_size.png', dpi=200)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--results', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    online(a.results, a.out)
    if (a.results/'study_E_summary.csv').exists():
        family_size(a.results, a.out)
