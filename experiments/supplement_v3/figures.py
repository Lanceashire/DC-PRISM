"""Figures for supplement-v3 from archived result tables only."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LABEL = {'Controlled': 'Controlled'}
for d, short in [('HMEQ', 'HMEQ'), ('LendingClub', 'LC')]:
    for m, ms in [('logistic', 'lin'), ('nonlinear', 'nl')]:
        for arm in ['broad', 'gap']:
            LABEL[f'{d}_{m}/{arm}'] = f'{short}-{ms}\n{arm}'
plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False})


def opportunity(res, out):
    d = pd.read_csv(res/'offline_decomposition_summary.csv')
    pol = pd.read_csv(res/'offline_policy_summary.csv')
    gain = pol[pol.policy == 'DC-PRISM'].set_index('group').exante_gain_vs_anchor
    d = d.set_index('group').loc[list(LABEL)]
    captured = gain.loc[d.index].clip(lower=0).to_numpy()
    sel_gap = (d.H_pre_crossfit.to_numpy()-gain.loc[d.index].to_numpy()).clip(min=0)
    unresolved = (d.H_pre_plugin-d.H_pre_crossfit).clip(lower=0).to_numpy()
    fore = d.foresight_lb.to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={'width_ratios': [1, 8]})
    for ax, idx in [(axes[0], [0]), (axes[1], list(range(1, len(d))))]:
        x = np.arange(len(idx)); bottom = np.zeros(len(idx))
        for vals, name, col in [(captured, 'captured by DC-PRISM', '#2a6f97'),
                                (sel_gap, 'pre-outcome selection gap', '#a9d6e5'),
                                (unresolved, 'estimator bracket', '#dddddd'),
                                (fore, 'foresight (not deployable)', '#e76f51')]:
            v = vals[idx]*1e3; ax.bar(x, v, bottom=bottom, color=col, width=.7, label=name); bottom += v
        ax.set_xticks(x); ax.set_xticklabels([LABEL[g] for g in d.index[idx]], fontsize=6.5)
    axes[0].set_ylabel(r'Expected Brier ($\times10^{-3}$)')
    axes[1].set_ylim(0, 8.6); axes[1].legend(frameon=False, fontsize=7, loc='upper left', ncol=2)
    fig.tight_layout(); fig.savefig(out/'supplement_v3_opportunity.pdf'); fig.savefig(out/'supplement_v3_opportunity.png', dpi=200)


def online(res, out):
    p = pd.read_csv(res/'online_primary_statistics.csv')
    comps = ['Safe-MinRisk', 'Fixed-count', 'Random-Feasible', 'Uncertainty', 'Fisher-cost', 'BAIT-Greedy', 'TargetVar-Greedy']
    colors = dict(zip(comps, plt.cm.tab10(np.arange(len(comps)))))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
    for ax, stream in zip(axes, ['engineered', 'natural']):
        g = p[p.stream == stream]
        for i, ds in enumerate(['HMEQ', 'LendingClub', 'PPDai', 'UCI_Credit']):
            for j, c in enumerate(comps):
                r = g[(g.dataset == ds) & (g.comparator == c)].iloc[0]; y = i + (j-3)*0.11
                ax.errorbar(r['mean']*1e3, y, xerr=[[(r['mean']-r.simultaneous_low)*1e3], [(r.simultaneous_high-r['mean'])*1e3]],
                            fmt='o', ms=2.5, lw=.8, color=colors[c], label=c if i == 0 else None)
        ax.axvline(0, color='k', lw=.6); ax.axvspan(-.5, .5, color='#eeeeee', zorder=0)
        ax.set_title(f'{stream} stream', fontsize=8); ax.set_xlabel(r'Comparator $-$ DC-PRISM ALC-Brier ($\times10^{-3}$)')
    axes[0].set_yticks(range(4)); axes[0].set_yticklabels(['HMEQ', 'LC', 'PPDai', 'UCI'])
    axes[1].legend(frameon=False, fontsize=6, loc='center left', bbox_to_anchor=(1, .5))
    fig.tight_layout(); fig.savefig(out/'supplement_v3_online.pdf'); fig.savefig(out/'supplement_v3_online.png', dpi=200)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--results', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True); opportunity(a.results, a.out); online(a.results, a.out)
