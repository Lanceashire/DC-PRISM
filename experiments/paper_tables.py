"""Generate the LaTeX table bodies of the oracle-gap paper from committed result files only.
Usage: python experiments/paper_tables.py <out_dir>"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1]/'results'
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)


def f(v, n=1):
    return (f"{v:.{n}f}" if round(abs(v), n) > 0 else f"{0:.{n}f}").replace('-', '$-$')


# T1: greedy-oracle decomposition by setting (supplement-v8, logistic learner)
d = pd.read_csv(R/'supplement_v8/states.csv.gz')
d['cat'] = np.where(d.setting != 'real-label', d.setting, np.where(d.group.str.startswith('Diabetes'), 'diabetes',
                    np.where(d.group.str.endswith('engineered'), 'credit-engineered', 'credit-natural')))
pc = [c for c in d.columns if c.startswith('prac_')]
lab = {'controlled': 'Controlled worlds', 'semi-synthetic': 'Semi-synthetic (known $p$)', 'credit-engineered': 'Credit, engineered stream',
       'credit-natural': 'Credit, natural stream', 'diabetes': 'Readmission, both streams'}
rows = []
for cat in lab:
    x = d[d.cat == cat]; best = x[pc].mean().idxmax()
    dep = x.deployable          # cross-fitted greedy gain (no post-hoc max over strategies: that would add selection optimism)
    rows.append(f"{lab[cat]} & {len(x)} & {f(x.reported.mean()*1e3)} & {f((x.reported-x.ex_post).mean()*1e3)} & "
                f"{f((x.ex_post-dep).mean()*1e3)} & {f(dep.mean()*1e3)} & "
                f"{f((x.reported-x[best]).mean()*1e3)}$\\to${f((dep-x[best]).mean()*1e3)} \\\\")
(out/'t_known.tex').write_text('\n'.join(rows)+'\n')

# T2: noise gradient, logistic (v9) and MLP (v10 part A), with the cross-fitted greedy gain
def grad_summary(path):
    x = pd.read_csv(path); x = x[x.part == 'oracle']; pcs = [c for c in x.columns if c.startswith('prac_')]; out_ = {}
    for ds, y in x.groupby('dataset'):
        best = y[pcs].mean().idxmax(); dep = y.deployable; rep = y.reported.mean()
        out_[ds] = dict(noise=y.noise_index.mean(), rep=rep, reuse=(y.reported-y.ex_post).mean()/rep, floor=(y.ex_post-dep).mean()/rep,
                        lead_r=(y.reported-y[best]).mean(), lead_d=(dep-y[best]).mean())
    return out_
a = grad_summary(R/'supplement_v9/states.csv.gz'); b = grad_summary(R/'supplement_v10/partA/states.csv.gz')
order = ['USPSraw', 'Spambase', 'USPS', 'Diabetes', 'BankMarketing', 'HMEQ', 'PPDai', 'UCI_Credit', 'LendingClub']
name = {'USPSraw': 'USPS, raw pixels', 'Spambase': 'Spambase', 'USPS': 'USPS, 10 PCs', 'Diabetes': 'Readmission',
        'BankMarketing': 'Bank Marketing', 'HMEQ': 'HMEQ', 'PPDai': 'PPDai', 'UCI_Credit': 'UCI Credit', 'LendingClub': 'LendingClub'}
cell = lambda r: f"{f(r['rep']*1e3)} & {100*r['reuse']:.0f} & {100*r['floor']:.0f} & {f(r['lead_r']*1e3)}$\\to${f(r['lead_d']*1e3)}"
rows = []
for ds in order:
    ni = a[ds]['noise'] if ds in a else b[ds]['noise']
    left = cell(a[ds]) if ds in a else r"\multicolumn{4}{c}{(neural only)}"
    rows.append(f"{name[ds]} & {ni:.3f} & {left} & {cell(b[ds])} \\\\")
(out/'t_gradient.tex').write_text('\n'.join(rows)+'\n')

# T3: who closes the deployable gap, real-label credit and readmission states, logistic vs boosting learner (v7)
s = pd.read_csv(R/'supplement_v7/summary.csv'); s = s[s.setting == 'real-label']
c = ['H_pre_crossfit', 'gain_DC-PRISM', 'gain_TargetVar-Greedy', 'gain_ProxyArgmax-F']
w = s.pivot_table(index='group', columns='learner', values=c)*1e3
order = ['HMEQ/engineered', 'LendingClub/engineered', 'PPDai/engineered', 'UCI_Credit/engineered', 'HMEQ/natural',
         'LendingClub/natural', 'PPDai/natural', 'UCI_Credit/natural', 'Diabetes/engineered', 'Diabetes/natural']
nm = lambda g: g.replace('UCI_Credit', 'UCI').replace('LendingClub', 'LC').replace('Diabetes', 'Readm.').replace('/engineered', ', eng.').replace('/natural', ', nat.')
rows = [nm(g)+' & '+' & '.join(f(w.loc[g, (k, 'logit')], 2) for k in c)+' & '+' & '.join(f(w.loc[g, (k, 'gbdt')], 2) for k in c)+r' \\' for g in order]
(out/'t_who.tex').write_text('\n'.join(rows)+'\n')

# T4: delayed replay, comparator minus DC-PRISM ALC-Brier (v5 study H credit, v7 study B readmission)
h = pd.read_csv(R/'supplement_v5/study_H_primary_statistics.csv'); bb = pd.read_csv(R/'supplement_v7/B_replay_statistics.csv'); bb['dataset'] = 'Diabetes'
dd = pd.concat([h, bb])
cols = [('engineered', 'HMEQ'), ('engineered', 'LendingClub'), ('engineered', 'PPDai'), ('engineered', 'UCI_Credit'), ('natural', 'HMEQ'),
        ('natural', 'LendingClub'), ('natural', 'PPDai'), ('natural', 'UCI_Credit'), ('engineered', 'Diabetes'), ('natural', 'Diabetes')]
names = {'Safe-MinRisk': 'Safe-MinRisk', 'Fixed-count': 'Fixed count', 'Random-Feasible': 'Random', 'Uncertainty': 'Uncertainty',
         'Fisher-cost': 'Fisher-cost', 'BAIT-Greedy': 'BAIT-greedy', 'TargetVar-Greedy': 'TargetVar-greedy'}
rows = []
for k, n in names.items():
    cs = []
    for st, ds in cols:
        r = dd[(dd.stream == st) & (dd.dataset == ds) & (dd.comparator == k)].iloc[0]; v = f(r['mean']*1e3)
        cs.append(r'\textbf{'+v+'}' if r.decision.startswith('material') else v)
    rows.append(n+' & '+' & '.join(cs)+r' \\')
(out/'t_replay.tex').write_text('\n'.join(rows)+'\n')

# T5: budget lever (v10 part B), family floor by target_k
g = pd.read_csv(R/'supplement_v10/partB/family_summary.csv')
nm2 = {'BankMarketing': 'Bank Marketing', 'Diabetes': 'Readmission', 'UCI_Credit': 'UCI Credit'}
rows = []
for ds, x in g.groupby('dataset'):
    x = x.sort_values('target_k')
    rows.append(f"{nm2.get(ds, ds)} & "+' & '.join(f(v*1e3, 2) for v in x.foresight)+' & '+' & '.join(f(v*1e3, 2) for v in x.H_pre_crossfit)+r' \\')
(out/'t_budget.tex').write_text('\n'.join(rows)+'\n')
print('tables written to', out)
