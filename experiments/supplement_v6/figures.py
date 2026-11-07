"""Main figure: foresight floor across settings (a) and family-size stability (b)."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False})


def main(res, out):
    v3 = pd.read_csv(res/'supplement_v3/offline_decomposition_summary.csv')
    g = pd.read_csv(res/'supplement_v5/study_G_decomposition.csv')
    r = pd.read_csv(res/'supplement_v6/study_R_summary.csv'); r = r[r.model == 'gbdt']
    pts = []
    for _, x in v3.iterrows():
        kind = 'controlled (known $p$)' if x.group == 'Controlled' else 'real covariates, simulated outcomes (known $p$)'
        pts.append((kind, x.H_pre_crossfit, x.H_post-x.H_pre_plugin))
    for _, x in g.iterrows():
        pts.append(('controlled (known $p$)', x.H_pre_crossfit, x.H_post-x.H_pre_plugin))
    for _, x in r.iterrows():
        pts.append((f'real labels, {x.stream} stream ($\\hat p$)', x.H_pre_crossfit, x.foresight))
    p = pd.DataFrame(pts, columns=['kind', 'pre', 'fore'])
    style = {'controlled (known $p$)': dict(color='#222222', marker='o'),
             'real covariates, simulated outcomes (known $p$)': dict(color='#4c78a8', marker='s'),
             'real labels, engineered stream ($\\hat p$)': dict(color='#f58518', marker='^'),
             'real labels, natural stream ($\\hat p$)': dict(color='#54a24b', marker='v')}
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={'width_ratios': [1.15, 1]})
    lim = (0.15, 60)
    a.fill_between(lim, 1.0, 6.5, color='#eeeeee', zorder=0)
    a.plot(lim, lim, color='#999999', lw=.7, ls=':')
    a.text(28, 17, 'foresight = deployable', fontsize=6, color='#777777', rotation=38, ha='center')
    for k, x in p.groupby('kind', sort=False):
        a.scatter(x.pre*1e3, x.fore*1e3, s=16, label=k, edgecolor='white', linewidth=.4, zorder=3, **style[k])
    a.set_xscale('log'); a.set_yscale('log'); a.set_xlim(*lim); a.set_ylim(.3, 60)
    a.set_xlabel(r'Deployable headroom $H_{\mathrm{pre}}$ ($\times10^{-3}$ Brier)')
    a.set_ylabel(r'Foresight (EVPI) ($\times10^{-3}$)')
    a.set_title('(a) foresight is a floor; deployable headroom varies', fontsize=7.5)
    a.legend(frameon=False, fontsize=5.6, loc='upper left', handletextpad=.3)
    s = pd.read_csv(res/'supplement_v5/study_E_summary.csv'); s = s[s.family == 'random_subfamily']
    seen = set()
    for grp, x in s.groupby('group'):
        x = x.sort_values('size')
        if grp == 'Controlled':
            b.plot(x['size'], 1-x.foresight_share, color='#222222', lw=2, marker='o', ms=3, label='controlled', zorder=3); continue
        arm = grp.split('/')[1]; lab = f'real covariates, {arm} history' if arm not in seen else None; seen.add(arm)
        b.plot(x['size'], 1-x.foresight_share, lw=1, marker='.', ms=3, label=lab,
               color='#4c78a8' if arm == 'broad' else '#72b7b2', ls='-' if arm == 'broad' else '--')
    b.set_xscale('log', base=2); b.set_xticks([2, 4, 8, 16, 32, 64]); b.set_xticklabels(['2', '4', '8', '16', '32', '64'])
    b.set_ylim(0, 1); b.set_xlabel(r'Family size $|\mathcal{F}|$ (random sub-families with $M_K$)')
    b.set_ylabel(r'Deployable share $H_{\mathrm{pre}}/H_{\mathrm{post}}$')
    b.set_title('(b) shares do not depend on family size', fontsize=7.5)
    b.grid(axis='y', color='#e5e5e5', lw=.5); b.legend(frameon=False, fontsize=5.8, loc='upper right', bbox_to_anchor=(1, .72))
    fig.tight_layout(); fig.savefig(out/'fig_protocol_main.pdf'); fig.savefig(out/'fig_protocol_main.png', dpi=200)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--results', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True); main(a.results, a.out)
