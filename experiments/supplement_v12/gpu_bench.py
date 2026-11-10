"""supplement-v12 E2 with the MLP learner: greedy benchmark oracles batched on one GPU.

Same accounting as common.decompose with the greedy search; the ex-post oracle runs one greedy per
replicate, batched across replicates; the ex-ante oracles (two cross-fitted folds and the plug-in) run
one greedy each on the mean T1 loss of their replicates. Every (set, replicate) model is trained once
(cache) with the shared per-state initialization of supplement-v10.
"""
import numpy as np


def state_rows(pk, mlp, C):
    st, YC, yT, T1, T2, seed = pk['st'], pk['YC'], pk['yT'], pk['T1'], pk['T2'], pk['seed']
    R, n = YC.shape; fz = C.Feasible(pk['q'], st['ead_C'], pk['B'], len(pk['M']), n)
    cache = {}

    def ensure(pairs):
        need = list(dict.fromkeys(p for p in pairs if p not in cache))
        if need:
            sets = []
            for S, r in need:
                Sl = list(S)
                X = np.vstack([st['X_D'], st['X_C'][Sl]]) if Sl else st['X_D']
                y = np.concatenate([st['y_D'], YC[r, Sl]]) if Sl else st['y_D']
                sets.append((X, y))
            P = mlp.predict_many(sets, st['X_T'], seed)
            for (S, r), p in zip(need, P):
                cache[(S, r)] = (float(np.mean((p[T1]-yT[T1])**2)), float(np.mean((p[T2]-yT[T2])**2)))

    def l(S, r, k): return cache[(S, r)][k]

    def greedy_joint(reps):
        S = (); ensure([(S, r) for r in reps]); cur = np.mean([l(S, r, 0) for r in reps])
        while len(S) < fz.K:
            cand = [C._t(S+(i,)) for i in range(n) if i not in S and fz(C._t(S+(i,)))]
            if not cand: break
            ensure([(c, r) for c in cand for r in reps])
            v = np.array([np.mean([l(c, r, 0) for r in reps]) for c in cand]); j = int(np.argmin(v))
            if v[j] >= cur-1e-12: break
            S, cur = cand[j], v[j]
        return S

    def greedy_each(reps):
        Ss = {r: () for r in reps}; ensure([((), r) for r in reps]); cur = {r: l((), r, 0) for r in reps}; active = list(reps)
        while active:
            jobs = []
            for r in active:
                if len(Ss[r]) >= fz.K: continue
                for i in range(n):
                    if i not in Ss[r] and fz(C._t(Ss[r]+(i,))): jobs.append((C._t(Ss[r]+(i,)), r))
            if not jobs: break
            ensure(jobs); nxt = []
            for r in active:
                mine = [S for S, rr in jobs if rr == r]
                if not mine: continue
                v = [l(S, r, 0) for S in mine]; j = int(np.argmin(v))
                if v[j] < cur[r]-1e-12:
                    Ss[r], cur[r] = mine[j], v[j]; nxt.append(r)
            active = nxt
        return Ss

    reps = list(range(R)); M = pk['M']
    post = greedy_each(reps)
    halves = [reps[0::2], reps[1::2]]; Oh = [greedy_joint(halves[0]), greedy_joint(halves[1])]; Op = greedy_joint(reps)
    ensure([(M, r) for r in reps]+[(post[r], r) for r in reps]+[(Oh[1-h], r) for h in (0, 1) for r in halves[h]]
           + [(Op, r) for r in reps]+[(s, r) for s in pk['prac'].values() for r in reps])
    LM1 = np.array([l(M, r, 0) for r in reps]); LM2 = np.array([l(M, r, 1) for r in reps])
    rep = float(np.mean([LM1[r]-l(post[r], r, 0) for r in reps])); hold = float(np.mean([LM2[r]-l(post[r], r, 1) for r in reps]))
    cf = float(np.mean([np.mean([LM2[r]-l(Oh[1-h], r, 1) for r in halves[h]]) for h in (0, 1)]))
    plug = float(np.mean([LM2[r]-l(Op, r, 1) for r in reps]))
    row = dict(part='BENCHMLP', learner='mlp', dataset=pk['dataset'], seed=seed, n_hist=pk['n_hist'], K=len(M), noise_index=pk['noise_index'],
               R=R, reported=rep, hold=hold, cf=cf, plug=plug, k_post=float(np.mean([len(post[r]) for r in reps])), k_plug=len(Op))
    for p, s in pk['prac'].items():
        row[f'prac_T2_{p}'] = float(np.mean([LM2[r]-l(s, r, 1) for r in reps]))
        row[f'prac_T1_{p}'] = float(np.mean([LM1[r]-l(s, r, 0) for r in reps]))
    return [row]
