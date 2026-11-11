"""supplement-v12 shared pieces: states, cached evaluators, oracle searches, decompositions.

Notation follows the paper. For a replicate r of the hidden training labels and a feasible set S,
L1(S, r) / L2(S, r) is the Brier score on test half T1 / T2 of the downstream learner fitted on
history + pending + S. M is the cardinality reference. For an oracle search method m:

  ex-post oracle   O_r   = m(objective = L1(., r))                        one search per replicate
  reported               = mean_r L1(M, r) - L1(O_r, r)                     selected and scored on T1
  hold                   = mean_r L2(M, r) - L2(O_r, r)                     same sets scored on T2
  cross-fitted ex-ante   = two folds: select by mean_{r in fold A} L1(., r), score on T2 over fold B
  plug-in ex-ante        = select by mean_{r in all} L1(., r), score on T2 over the same replicates

  test reuse = reported - hold
  residual (cross-fit end) = hold - cross-fitted      (upper-end estimate of the floor's benchmark analogue)
  residual (plug-in end)   = hold - plug-in           (lower-end estimate)

Plug-in maximisation over noisy means overstates the ex-ante value and cross-fitting understates it
(Smith and Winkler 2006), so the two residuals bracket the residual one would obtain with exact
ex-ante values, up to search error and the finite test half.
"""
import itertools
import zlib

import numpy as np
from scipy.special import expit, logit

import prism_engine as eng

NEW = {'USPS', 'Spambase', 'BankMarketing', 'USPSraw'}
TOL = 1e-12


# ------------------------------------------------------------------ states
def natural_state(D, dataset, seed, n_C=24, n_hist=68, pending=True, flip=0.):
    """Natural-stream state (supplement-v11 construction) with a history of n_hist labelled records.

    The preprocessing is fitted on the first 68 history records for every n_hist, so that states with
    the same seed differ only by the additional history. With pending=False the 14 pending records are
    labelled and join the history (benchmark-style pool-based step without delay).
    """
    X_raw, y_raw, ead_raw = D[dataset]
    if flip > 0:
        y_raw = np.where(np.random.default_rng([zlib.crc32(dataset.encode()), 11001]).random(len(y_raw)) < flip, 1-y_raw, y_raw)
    rng = np.random.default_rng(seed); n = len(X_raw); perm = rng.permutation(n)
    n_T = min(max(int(n*0.18), 500), 1000); n_Q = min(max(int(n*0.12), 350), 600)
    ids_T, ids_Q, pool = perm[:n_T], perm[n_T:n_T+n_Q], perm[n_T+n_Q:]
    assert n_hist <= 600
    ids_D0, ids_D = pool[:68], pool[:n_hist]
    stream = rng.permutation(pool[600:]); o, c = stream[:14], stream[14:14+n_C]
    if dataset in NEW:
        mu = X_raw[ids_D0].mean(axis=0); sd = X_raw[ids_D0].std(axis=0); sd = np.where(sd < 1e-6, 1., sd)
        tf = lambda Z: np.clip((Z-mu)/sd, -4.5, 4.5); te = lambda e: np.ones(len(e))
    else:
        prep = eng.SplitBeforePreprocessor(tail_scale=1.85).fit(X_raw[ids_D0], ead_raw[ids_D0])
        prep.stds = np.maximum(prep.stds, 1e-3); prep.tau_aux = np.maximum(prep.tau_aux, np.array([1.35, 1.35]))
        tf, te = prep.transform_X, prep.transform_ead
    if pending:
        hist, pend = ids_D, o
    else:
        hist, pend = np.concatenate([ids_D, o]), o[:0]
    d = X_raw.shape[1]
    st = dict(X_D=tf(X_raw[hist]), y_D=y_raw[hist].astype(int), X_O=tf(X_raw[pend]) if len(pend) else np.zeros((0, d)),
              X_C=tf(X_raw[c]), ead_C=te(ead_raw[c]), X_Q=tf(X_raw[ids_Q]), X_T=tf(X_raw[ids_T]))
    used = np.concatenate([pool[:600], ids_Q, ids_T, o, c]); mask = np.ones(n, bool); mask[used] = False
    aux = np.random.default_rng([seed, 12003]).permutation(np.flatnonzero(mask))[:20000]
    return st, dict(X_raw=X_raw, y_raw=y_raw, O_ids=pend, C_ids=c, aux=aux, yT=y_raw[ids_T].astype(float),
                    yO=y_raw[pend].astype(int), yC=y_raw[c].astype(int))


def constant_risk_context(v5, st, seed, target_k=6, q0=.5):
    """Benchmark-style cardinality budget: a constant risk score makes the envelope a function of |S| only.

    With q = 0.5 and unit exposure, B = 1.03 x envelope(6) admits exactly six labels. Everything else
    (reference M, candidate family) is built as in supplement-v5's reference_context.
    """
    X_D, y_D, X_C, ead, X_Q = st['X_D'], st['y_D'], st['X_C'], st['ead_C'], st['X_Q']
    assert np.all(ead == 1.)
    learner = eng.UnifiedLogisticLearner(l2=.45).fit(X_D, y_D)
    q = np.full(len(X_C), q0)
    B, _ = eng.compute_exogenous_budget(q, ead, target_k=target_k)
    M, cert = eng.solve_safe_minrisk_exact_or_bounded(q, ead, B)
    assert len(M) == target_k and eng.bernstein_envelope(q[:target_k+1], ead[:target_k+1]) > B
    family, *_ = eng.generate_candidate_family(X_D, None, X_C, X_Q, q, ead, B, M, learner)
    cands = {tuple(np.sort(s)): np.sort(s) for s in family if len(s) <= len(M)}
    for k in range(len(M)+1):
        s = np.sort(M[:k]); cands[tuple(s)] = s
    return dict(q=q, B=B, M=M, family=[cands[k] for k in sorted(cands)], certificate=cert, t_mk=0.)


# ------------------------------------------------------------------ outcome models
def calibration(ph, yh, n_boot, seed):
    """Logistic recalibration y ~ a + b logit(p_hat) on held-out rows; bootstrap percentile band for b."""
    lg = logit(np.clip(ph, 1e-4, 1-1e-4)); m = float(lg.mean())
    fit = lambda idx: eng.UnifiedLogisticLearner(l2=1e-6).fit((lg[idx]-m)[:, None], yh[idx].astype(int)).w
    w = fit(np.arange(len(lg))); rng = np.random.default_rng([seed, 12005])
    bs = np.array([fit(rng.integers(0, len(lg), len(lg)))[1] for _ in range(n_boot)])
    bins = np.minimum((ph*10).astype(int), 9)
    ece = float(sum(abs(ph[bins == k].mean()-yh[bins == k].mean())*(bins == k).mean() for k in range(10) if (bins == k).any()))
    return dict(a=float(w[0]), b=float(w[1]), b_lo=float(np.quantile(bs, .025)), b_hi=float(np.quantile(bs, .975)), m=m, ece=ece)


def recalibrated(p, cal, b):
    """Recalibrate with slope b around the held-out mean logit (the mean level is kept at the point fit)."""
    return np.clip(expit(cal['a']+b*(logit(np.clip(p, 1e-4, 1-1e-4))-cal['m'])), 1e-4, 1-1e-4)


def mlp_phat(X, y, seed):
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    m = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                      MLPClassifier(hidden_layer_sizes=(32,), alpha=1e-3, max_iter=300, early_stopping=True,
                                    validation_fraction=.15, random_state=seed % (2**31)))
    m.fit(X, y)
    return lambda Z: np.clip(m.predict_proba(Z)[:, 1], 1e-4, 1-1e-4)


# ------------------------------------------------------------------ evaluators
class LogitEval:
    """Cached L1/L2 for (set, replicate) under the l2-logistic downstream learner (CPU)."""

    def __init__(self, st, YO, YC, yT, T1, T2, w_init):
        self.st, self.YO, self.YC, self.yT, self.T1, self.T2, self.w = st, YO, YC, yT, T1, T2, w_init
        self.cache = {}; self.fits = 0

    def L(self, S, r):
        k = (S, r)
        v = self.cache.get(k)
        if v is None:
            st = self.st; Sl = list(S)
            X = np.vstack([st['X_D'], st['X_O'], st['X_C'][Sl]]) if Sl else np.vstack([st['X_D'], st['X_O']])
            y = np.concatenate([st['y_D'], self.YO[r], self.YC[r, Sl]]) if Sl else np.concatenate([st['y_D'], self.YO[r]])
            p = eng.UnifiedLogisticLearner(l2=.45).fit(X, y, w_init=self.w).predict_proba(st['X_T'])
            v = (float(np.mean((p[self.T1]-self.yT[self.T1])**2)), float(np.mean((p[self.T2]-self.yT[self.T2])**2)))
            self.cache[k] = v; self.fits += 1
        return v

    def obj(self, S, reps):
        return float(np.mean([self.L(S, r)[0] for r in reps]))

    def objs(self, sets, reps):
        return np.array([self.obj(S, reps) for S in sets])

    def l2(self, S, reps):
        return np.array([self.L(S, r)[1] for r in reps])

    def l1(self, S, reps):
        return np.array([self.L(S, r)[0] for r in reps])


class MatrixEval:
    """Lookup evaluator over a fully evaluated feasible space (exact study)."""

    def __init__(self, space, L1, L2):
        self.key = {S: j for j, S in enumerate(space)}; self.L1, self.L2 = L1, L2; self.fits = 0

    def obj(self, S, reps):
        return float(self.L1[reps, self.key[S]].mean())

    def objs(self, sets, reps):
        return self.L1[np.ix_(reps, [self.key[S] for S in sets])].mean(axis=0)

    def l2(self, S, reps):
        return self.L2[reps, self.key[S]]

    def l1(self, S, reps):
        return self.L1[reps, self.key[S]]


# ------------------------------------------------------------------ searches over feasible sets
class Feasible:
    def __init__(self, q, ead, B, K, n):
        self.q, self.ead, self.B, self.K, self.n = q, ead, B, K, n; self._c = {}

    def __call__(self, S):
        v = self._c.get(S)
        if v is None:
            v = len(S) <= self.K and (len(S) == 0 or eng.bernstein_envelope(self.q[list(S)], self.ead[list(S)]) <= self.B+TOL)
            self._c[S] = v
        return v


def _t(S):
    return tuple(sorted(S))


def greedy(ev, fz, reps):
    """Benchmark greedy oracle (supplement-v8): add the feasible candidate that most lowers the T1 objective."""
    S = (); cur = ev.obj(S, reps)
    while len(S) < fz.K:
        cand = [_t(S+(i,)) for i in range(fz.n) if i not in S and fz(_t(S+(i,)))]
        if not cand: break
        v = ev.objs(cand, reps); j = int(np.argmin(v))
        if v[j] >= cur-TOL: break
        S, cur = cand[j], float(v[j])
    return S


def neighbours(S, fz):
    out = []
    ins = [i for i in range(fz.n) if i not in S]
    for i in ins:
        T = _t(S+(i,))
        if fz(T): out.append(T)
    for j in S:
        R = tuple(x for x in S if x != j); out.append(R)
        for i in ins:
            T = _t(R+(i,))
            if fz(T): out.append(T)
    return out


def local(ev, fz, reps, S0, max_iter=60):
    """Best-improvement local search with add / remove / swap moves from S0."""
    S, cur = S0, ev.obj(S0, reps)
    for _ in range(max_iter):
        nb = neighbours(S, fz)
        if not nb: break
        v = ev.objs(nb, reps); j = int(np.argmin(v))
        if v[j] >= cur-TOL: break
        S, cur = nb[j], float(v[j])
    return S, cur


def random_start(fz, rng):
    k = int(rng.integers(1, fz.K+1)); S = ()
    for i in rng.permutation(fz.n):
        T = _t(S+(int(i),))
        if fz(T): S = T
        if len(S) == k: break
    return S


def restarts(ev, fz, reps, rng, n_starts=10):
    """Local search from the greedy solution and from n_starts-1 random feasible sets; best objective."""
    starts = [greedy(ev, fz, reps)] + [random_start(fz, rng) for _ in range(n_starts-1)]
    best, bv = None, np.inf
    for s in starts:
        S, v = local(ev, fz, reps, s)
        if v < bv-TOL: best, bv = S, v
    return best


def exact(ev, space, reps):
    return space[int(np.argmin(ev.objs(space, reps)))]


def make_search(name, ev, fz, seed, space=None):
    if name == 'greedy':
        return lambda reps: greedy(ev, fz, reps)
    if name == 'local':
        return lambda reps: local(ev, fz, reps, greedy(ev, fz, reps))[0]
    if name == 'restarts':
        return lambda reps: restarts(ev, fz, reps, np.random.default_rng([seed, 12007, zlib.crc32(str(list(reps)).encode())]))
    if name == 'exact':
        return lambda reps: exact(ev, space, reps)
    raise ValueError(name)


# ------------------------------------------------------------------ decompositions
def decompose(ev, search, M, R_list, practical=None):
    """Greedy-benchmark accounting for every R in R_list (nested replicate prefixes, one ex-post search per replicate)."""
    Rmax = max(R_list); post = {}
    for r in range(Rmax):
        post[r] = search([r])
    M = _t(M); out = {}
    for R in R_list:
        reps = list(range(R)); LM1, LM2 = ev.l1(M, reps), ev.l2(M, reps)
        rep = np.mean([LM1[r]-ev.l1(post[r], [r])[0] for r in reps])
        hold = np.mean([LM2[r]-ev.l2(post[r], [r])[0] for r in reps])
        halves = [reps[0::2], reps[1::2]]; cf = []
        for a, b in [(0, 1), (1, 0)]:
            O = search(halves[a]); cf.append(float(np.mean(ev.l2(M, halves[b])-ev.l2(O, halves[b]))))
        O = search(reps); plug = float(np.mean(LM2-ev.l2(O, reps)))
        out[R] = dict(reported=float(rep), hold=float(hold), cf=float(np.mean(cf)), plug=plug,
                      k_post=float(np.mean([len(post[r]) for r in reps])), k_plug=len(O))
    if practical is not None:
        reps = list(range(Rmax)); LM1, LM2 = ev.l1(M, reps), ev.l2(M, reps)
        for p, s in practical.items():
            s = _t(s); out[f'prac_T2_{p}'] = float(np.mean(LM2-ev.l2(s, reps))); out[f'prac_T1_{p}'] = float(np.mean(LM1-ev.l1(s, reps)))
    return out


def family_decomposition(L, anchor, R_list):
    """Fixed-family protocol on nested replicate prefixes: H_post, plug-in and cross-fitted H_pre, F at both ends."""
    out = {}
    for R in R_list:
        X = L[:R]; post = float(np.mean(X[:, anchor]-X.min(axis=1))); mu = X.mean(axis=0); plug = float(mu[anchor]-mu.min())
        a, b = np.arange(0, R, 2), np.arange(1, R, 2); cf = 0.
        for f, e in [(a, b), (b, a)]:
            s = int(np.argmin(X[f].mean(axis=0))); cf += .5*float(X[e, anchor].mean()-X[e, s].mean())
        d = X[:, [anchor]]-X; mu_d = d.mean(axis=0); o = np.argsort(-mu_d)
        F_lo = float(np.maximum(0., d[:, o[1]]-d[:, o[0]]).mean()) if len(o) > 1 else 0.
        out[R] = dict(H_post=post, H_pre_plugin=plug, H_pre_cf=cf, F_plugin=post-plug, F_cf=post-cf, F_lo=F_lo,
                      F_up=float((d-mu_d).max(axis=1).mean()))
    return out


def loss_matrix_logit(st, sets, w, YO, YC, yT):
    L = np.empty((len(YO), len(sets)))
    for j, s in enumerate(sets):
        s = list(s); X = np.vstack([st['X_D'], st['X_O'], st['X_C'][s]])
        for r in range(len(YO)):
            y = np.concatenate([st['y_D'], YO[r], YC[r, s]])
            p = eng.UnifiedLogisticLearner(l2=.45).fit(X, y, w_init=w).predict_proba(st['X_T'])
            L[r, j] = float(np.mean((p-yT)**2))
    return L


def feasible_space(fz):
    space = [()]
    for k in range(1, fz.K+1):
        found = False
        for S in itertools.combinations(range(fz.n), k):
            if fz(S): space.append(S); found = True
        if not found: break
    return space
