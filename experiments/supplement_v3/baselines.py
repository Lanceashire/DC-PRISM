"""Additional risk-feasible acquisition baselines for supplement-v3.

Every selector receives visible information only: history covariates/labels,
pending covariates (no labels), candidate covariates, exposures, risk scores
q, the numerical budget B and target covariates. None of them accepts hidden
candidate, pending or test outcomes, or true probabilities.

All selectors pack candidates under the same Bernstein envelope used by every
other method (R_alpha(S) <= B). They keep adding while some feasible candidate
remains, so they generally select more labels than DC-PRISM; label count is
reported alongside loss.
"""
import numpy as np
from scipy.special import expit

import prism_engine as eng

ALPHA = 0.10
TOL = 1e-12


def _feasible(trial, q, ead, B):
    return eng.bernstein_envelope(q[trial], ead[trial], alpha=ALPHA) <= B + TOL


def _pack_in_order(order, q, ead, B):
    chosen = []
    for i in order:
        trial = np.array(chosen + [int(i)], dtype=int)
        if _feasible(trial, q, ead, B):
            chosen.append(int(i))
    return np.sort(np.array(chosen, dtype=int))


def random_feasible(X_C, q, ead, B, rng):
    """Uniformly random order, greedy feasible packing."""
    return _pack_in_order(rng.permutation(len(X_C)), q, ead, B)


def uncertainty_greedy(learner, X_C, q, ead, B):
    """Classical uncertainty sampling: largest p(1-p) first, feasible packing."""
    p = learner.predict_proba(X_C)
    order = np.lexsort((np.arange(len(X_C)), -(p*(1-p))))
    return _pack_in_order(order, q, ead, B)


def _information_parts(learner, X_D, X_O, X_C, X_Q):
    w = learner.w
    H, _, G = eng.build_pending_fisher_and_target(X_D, X_O, X_Q, w, l2=learner.l2)
    Xb_C = np.column_stack([np.ones(len(X_C)), X_C])
    pc = expit(Xb_C @ w)
    wc = np.clip(pc*(1-pc), 1e-6, None)
    Xb_Q = np.column_stack([np.ones(len(X_Q)), X_Q])
    pq = expit(Xb_Q @ w)
    wq = np.clip(pq*(1-pq), 1e-6, None)
    F_pool = (Xb_Q*wq[:, None]).T @ Xb_Q / max(len(X_Q), 1)
    return H, G, F_pool, Xb_C, wc


def _forward_greedy(target, H, Xb_C, wc, q, ead, B):
    """Forward greedy minimisation of tr(target @ (H + A_S)^{-1}).

    Rank-one Sherman-Morrison updates; at every step the feasible candidate with
    the largest objective reduction is added. Stops when no feasible candidate
    remains or no candidate reduces the objective.
    """
    Hinv = np.linalg.pinv(H, rcond=1e-12)
    chosen, remaining = [], list(range(len(Xb_C)))
    while remaining:
        best, best_gain = None, 0.0
        for i in remaining:
            trial = np.array(chosen + [i], dtype=int)
            if not _feasible(trial, q, ead, B):
                continue
            x = Xb_C[i]; u = Hinv @ x
            gain = wc[i]*float(u @ target @ u)/(1.0 + wc[i]*float(x @ u))
            if gain > best_gain + 1e-15:
                best, best_gain = i, gain
        if best is None:
            break
        x = Xb_C[best]; u = Hinv @ x
        Hinv = Hinv - (wc[best]/(1.0 + wc[best]*float(x @ u)))*np.outer(u, u)
        chosen.append(best); remaining.remove(best)
    return np.sort(np.array(chosen, dtype=int))


def bait_greedy(learner, X_D, X_O, X_C, X_Q, q, ead, B):
    """BAIT-style forward greedy (Ash et al., 2021) under the risk envelope.

    Objective tr(F_pool (H_{D+O} + A_S)^{-1}) with F_pool the mean Fisher
    information of the unlabeled target pool, as in BAIT. Forward step only
    (no oversample-then-prune backward step); pending covariates enter H.
    """
    H, _, F_pool, Xb_C, wc = _information_parts(learner, X_D, X_O, X_C, X_Q)
    return _forward_greedy(F_pool, H, Xb_C, wc, q, ead, B)


def targetvar_greedy(learner, X_D, X_O, X_C, X_Q, q, ead, B):
    """Target-predictive-variance greedy (EPIG-style Laplace approximation).

    Uses DC-PRISM's own target-weighted matrix G_Q (Brier-sensitivity weights)
    and pending-aware H_{D+O}, but none of DC-PRISM's reference comparison,
    parameter-draw quantile, quantity penalty or gate. This isolates what the
    surrounding decision rule adds to the shared information proxy.
    """
    H, G, _, Xb_C, wc = _information_parts(learner, X_D, X_O, X_C, X_Q)
    return _forward_greedy(G, H, Xb_C, wc, q, ead, B)


def package_family_scores(learner, X_D, X_O, X_C, X_Q, q, ead, B, M, worlds,
                          damping=0.45, scale=2.5):
    """Re-implementation of package DC-PRISM scoring with exposed constants.

    Returns the package candidate family and the per-draw normalised deltas so
    that quantile/penalty/gate variants can be applied without refitting.
    """
    family, *_ = eng.generate_candidate_family(X_D, X_O, X_C, X_Q, q, ead, B, M, learner, alpha=ALPHA)
    Xb = np.column_stack([np.ones(len(X_C)), X_C])
    draws = worlds[1:] if len(worlds) > 1 else worlds
    deltas = np.empty((len(draws), len(family)))
    for m, w in enumerate(draws):
        H, Hi, G = eng.build_pending_fisher_and_target(X_D, X_O, X_Q, w, l2=learner.l2)
        p = expit(Xb @ w)
        wc = np.clip(p*(1-p)*np.exp(-damping*(Xb @ (w-learner.w))**2), 1e-6, None)
        a = max(eng.pending_aware_value(M, Xb, wc, H, Hi, G), 1e-6)
        for j, s in enumerate(family):
            deltas[m, j] = np.tanh((eng.pending_aware_value(s, Xb, wc, H, Hi, G)-a)/(scale*a))
    return family, deltas


def choose_from_scores(family, deltas, M, quantile=0.2, penalty=0.22, gate=0.08):
    """Package decision rule: first strict maximiser of score; gate to M_K."""
    K = len(M)
    best_score, best = -1e9, None
    for j, s in enumerate(family):
        score = float(np.quantile(deltas[:, j], quantile)) - penalty*max(K-len(s), 0)/float(max(K, 1))
        if score > best_score:
            best_score, best = score, j
    return (family[best] if best_score > gate else M), best_score
