"""supplement-v11 part E: exact maximum cardinality for the supplement-v5 states whose heuristic
M_K was not certified (bound exceeded the found cardinality by one).

For each such state the visible reference problem is rebuilt exactly as in supplement-v5 and a
branch-and-bound search decides whether any feasible set of size K+1 exists. The lower bound of
the envelope of any completion adds the smallest remaining mean terms and variance terms
separately and keeps the current maximum exposure (the bound used by the package solver).
Usage: python experiments/supplement_v11/exact_mk.py <data_dir> <out_csv>
"""
import importlib.util as ilu
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
import prism_engine as eng


def _load(name, rel):
    spec = ilu.spec_from_file_location(name, ROOT/rel); mod = ilu.module_from_spec(spec)
    sys.modules[name] = mod; spec.loader.exec_module(mod); return mod


v5 = _load('v5_run_offline_e', 'experiments/supplement_v5/run_offline.py')
v3 = v5.v3


def exists_size(q, e, B, k, alpha=0.10, time_limit=600):
    ell = np.log(1/alpha); v = np.where(q <= .5, q*(1-q), .25); mu = q*e; var = e**2*v
    order = np.argsort(mu+0.35*np.sqrt(var*ell)); n = len(q); t0 = time.time(); nodes = [0]
    def env(m, s2, mx): return m+np.sqrt(2*ell*s2)+ell/3*mx
    def dfs(start, chosen, m, s2, mx):
        nodes[0] += 1
        if time.time()-t0 > time_limit: raise TimeoutError
        need = k-len(chosen)
        if need == 0: return list(chosen)
        rest = order[start:]
        if len(rest) < need: return None
        lb = env(m+np.sort(mu[rest])[:need].sum(), s2+np.sort(var[rest])[:need].sum(), mx)
        if lb > B+1e-12: return None
        for j in range(start, n-need+1):
            i = order[j]; nm, ns, nx = m+mu[i], s2+var[i], max(mx, e[i])
            if env(nm, ns, nx) > B+1e-12: continue
            r = dfs(j+1, chosen+[i], nm, ns, nx)
            if r is not None: return r
        return None
    res = dfs(0, [], 0., 0., 0.)
    return res, nodes[0], time.time()-t0


def main():
    data_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
    R5 = ROOT/'results/supplement_v5'
    rows = []
    g = pd.read_csv(R5/'study_G_rows.csv.gz'); g = g[(g.policy == 'Safe-MinRisk') & (g.mk_gap > 0)]
    import json
    cells = json.loads((ROOT/'configs/supplement_v5.json').read_text())['study_G']['cells']
    for _, r in g.iterrows():
        cfg = cells[r.cell]
        st = v5.perturbed_state(int(r.seed), cfg['overlap'], cfg['coupling'], cfg['tail'], cfg['double'])
        ctx = v5.reference_context(*[st[k] for k in ['X_D', 'y_D', 'X_C', 'ead_C', 'X_Q']], int(r.seed), target_k=cfg['target_k'])
        assert len(ctx['M']) == r.K
        try:
            S, nodes, sec = exists_size(ctx['q'], st['ead_C'], ctx['B'], int(r.K)+1); status = 'exact'
        except TimeoutError:
            S, nodes, sec, status = None, -1, 600., 'timeout'
        rows.append(dict(study='G', cell=r.cell, seed=int(r.seed), K=int(r.K), larger_set_exists=S is not None, status=status, nodes=nodes, seconds=sec))
    f = pd.read_csv(R5/'study_F_rows.csv.gz'); f = f[(f.policy == 'Safe-MinRisk') & (f.mk_gap > 0)]
    if len(f):
        v3._init(str(data_dir/'real_credit_datasets_authentic.npz'), json.loads((ROOT/'configs/supplement_v5.json').read_text())['study_F']['cohorts'])
        for _, r in f.iterrows():
            cohort, arm = r.group.split('/')
            c = v3._STATE['cohorts'][cohort]; arms = v3.realx_pair(v3._STATE['data'][c['dataset']], int(r.seed), c['mechanism'])
            ref = arms['broad']
            ctx = v5.reference_context(*[ref[k] for k in ['X_D', 'y_D', 'X_C', 'ead_C', 'X_Q']], int(r.seed))
            S, nodes, sec = exists_size(ctx['q'], ref['ead_C'], ctx['B'], int(r.K)+1)
            rows.append(dict(study='F', cell=r.group, seed=int(r.seed), K=int(r.K), larger_set_exists=S is not None, status='exact', nodes=nodes, seconds=sec))
    df = pd.DataFrame(rows); df.to_csv(out, index=False)
    print(df.groupby(['study', 'status']).larger_set_exists.agg(['sum', 'count']))


if __name__ == '__main__':
    main()
