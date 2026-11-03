"""
Drop-in Upgrade Module for Lanceashire/LexiRiskLabel (src/phase0_engine.py)
===========================================================================
Implements Chance-Constrained & Fisher-UCB Lexicographic Risk-Budgeted Exploration (CC-LRBE-FAVE)
and Horvitz-Thompson Inverse Propensity Weighting (IPW) for Selective-Label Retraining.

Key Upgrades over Original src/phase0_engine.py:
  1. Zero-extra-matrix-cost extraction of Fisher epistemic logit standard deviation `logit_std`
     directly from `Hinv` inside `fave_value()`.
  2. Online Selective-Label Logit Bias Calibrator (`estimate_online_logit_shift`) to eliminate
     the O(K*) downward optimism bias on rejected applicants.
  3. Second-Order Chance-Constrained MILP (`optimize_cc_lrbe`) that enforces a high-probability
     (1 - alpha) VaR/Bernstein budget envelope to control O(sqrt(K*)) Bernoulli realization variance.
  4. Bounded Horvitz-Thompson Inverse Propensity Weighting (`compute_ipw_weights`) so acquired
     rejected samples effectively update the classifier decision boundary.
"""

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.special import expit

FAVE_GAMMA = 1.0
FAVE_DELTA = 0.5


def _ridge_inverse(A):
    return np.linalg.pinv(A, rcond=1e-12)


def fave_value_with_epistemic_std(
    raw_X,
    visible,
    rejected,
    reference,
    pre,
    model,
    gamma=FAVE_GAMMA,
    delta=FAVE_DELTA,
):
    """
    Drop-in replacement for `fave_value` in `src/phase0_engine.py`.
    Returns:
      - fave_val: 1D array of Frozen FAVE-v2.1 information values for rejected pool
      - logit_std: 1D array of local Fisher epistemic standard errors sqrt(r_i^T H^{-1} r_i)
    """
    visible = np.asarray(visible, dtype=int)
    rejected = np.asarray(rejected, dtype=int)
    reference = np.asarray(reference, dtype=int)
    if len(rejected) == 0:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)

    Zv = pre.transform(raw_X.iloc[visible]) if hasattr(raw_X, "iloc") else pre.transform(raw_X[visible])
    Zr = pre.transform(raw_X.iloc[rejected]) if hasattr(raw_X, "iloc") else pre.transform(raw_X[rejected])
    Zf = pre.transform(raw_X.iloc[reference]) if hasattr(raw_X, "iloc") else pre.transform(raw_X[reference])

    pv = np.clip(model.predict_proba(Zv)[:, 1], 1e-9, 1 - 1e-9)
    pr = np.clip(model.predict_proba(Zr)[:, 1], 1e-9, 1 - 1e-9)
    pf = np.clip(model.predict_proba(Zf)[:, 1], 1e-9, 1 - 1e-9)

    V = np.column_stack([np.ones(len(Zv)), Zv])
    R = np.column_stack([np.ones(len(Zr)), Zr])
    F = np.column_stack([np.ones(len(Zf)), Zf])

    wv = np.clip(pv * (1 - pv), 1e-6, None)
    wi = np.clip(pr * (1 - pr) * (pr ** float(delta)), 1e-10, None)

    Vw = V * np.sqrt(wv)[:, None]
    H = np.einsum("ni,nj->ij", Vw, Vw)
    ridge = max(float(np.trace(H) / H.shape[0]) * 1e-6, 1e-10)
    Hinv = _ridge_inverse(H + ridge * np.eye(H.shape[0]))

    G = np.einsum("n,ni,nj->ij", pf ** float(gamma), F, F) / max(len(F), 1)
    HG = np.einsum("ij,jk,kl->il", Hinv, G, Hinv)

    lev = np.einsum("ij,jk,ik->i", R, HG, R)
    value = wi * lev / (1 + wi * lev)
    fave_val = np.maximum(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0), 0.0)

    # Zero-extra-cost epistemic variance in logit space reusing Hinv
    logit_var = np.einsum("ij,jk,ik->i", R, Hinv, R)
    logit_std = np.sqrt(np.maximum(logit_var, 0.0))
    return fave_val, logit_std


def estimate_online_logit_shift(explored_pred_logits, explored_true_y, prior_delta=0.25, max_delta=0.85):
    """
    Estimates the online selective-label log-odds calibration offset delta_cal
    from historical explored rejected applicants.
    """
    if len(explored_true_y) < 8:
        return float(prior_delta)
    hist_y = np.asarray(explored_true_y, dtype=float)
    hist_logits = np.asarray(explored_pred_logits, dtype=float)
    emp_rate = np.clip(np.mean(hist_y) + 0.25 / np.sqrt(len(hist_y)), 1e-4, 0.99)
    pred_rate = np.clip(np.mean(expit(hist_logits)), 1e-4, 0.99)
    delta_cal = np.log(emp_rate / (1.0 - emp_rate)) - np.log(pred_rate / (1.0 - pred_rate))
    return float(np.clip(delta_cal, 0.0, max_delta))


def optimize_cc_lrbe(
    p_hat,
    ead,
    logit_std,
    score,
    budget,
    delta_cal=0.25,
    beta_ucb=0.65,
    z_alpha=1.05,
):
    """
    Two-Stage Chance-Constrained Lexicographic MILP (CC-LRBE):
      Stage 1: Maximize safe acquisition count K_safe* subject to:
               sum_i (p_ucb_i * ead_i * z_i) + z_alpha * sqrt(sum_i p_ucb_i*(1-p_ucb_i)*ead_i^2 * z_i) <= budget
      Stage 2: At fixed sum_i z_i = K_safe*, maximize sum_i (score_i * z_i) subject to the linearized
               second-order chance constraint.
    """
    n = len(p_hat)
    if n == 0 or budget <= 0:
        return np.array([], int), 0

    logit_p = np.log(np.clip(p_hat, 1e-9, 1 - 1e-9) / np.clip(1.0 - p_hat, 1e-9, 1 - 1e-9))
    p_ucb = expit(logit_p + delta_cal + beta_ucb * logit_std)

    mean_cost = p_ucb * ead
    var_cost = p_ucb * (1.0 - p_ucb) * (ead ** 2)

    # Stage 1: Exact monotonic search over risk-ordered prefix for maximum feasible K_safe*
    order_by_risk = np.argsort(mean_cost + 0.35 * z_alpha * np.sqrt(var_cost))
    k_safe = 0
    for k_try in range(1, n + 1):
        idx_try = order_by_risk[:k_try]
        if np.sum(mean_cost[idx_try]) + z_alpha * np.sqrt(np.sum(var_cost[idx_try])) <= budget:
            k_safe = k_try
        else:
            break

    if k_safe == 0:
        return np.array([], int), 0

    # Stage 2: First-order SOCP tangent linearization at K_safe* solved via MILP
    ref_idx = order_by_risk[:k_safe]
    sigma_k = max(np.sqrt(np.sum(var_cost[ref_idx])), 1e-6)
    eff_unit_cost = mean_cost + (z_alpha / (2.0 * sigma_k)) * var_cost
    eff_budget = budget - 0.5 * z_alpha * sigma_k

    integ = np.ones(n, int)
    bnd = Bounds(np.zeros(n), np.ones(n))
    con_b = LinearConstraint(eff_unit_cost.reshape(1, -1), -np.inf, eff_budget)
    con_k = LinearConstraint(np.ones((1, n)), k_safe, k_safe)

    eff_score = np.asarray(score, float) * (1.0 + 0.6 * logit_std)
    r2 = milp(c=-eff_score, integrality=integ, bounds=bnd, constraints=[con_b, con_k])
    chosen = np.flatnonzero(np.rint(r2.x) > 0.5) if (r2.success and r2.x is not None) else order_by_risk[:k_safe]

    # Exact nonlinear chance-constraint verification guard
    while len(chosen) > 1 and (np.sum(mean_cost[chosen]) + z_alpha * np.sqrt(np.sum(var_cost[chosen])) > budget):
        k_safe -= 1
        con_k = LinearConstraint(np.ones((1, n)), k_safe, k_safe)
        r2 = milp(c=-eff_score, integrality=integ, bounds=bnd, constraints=[con_b, con_k])
        chosen = np.flatnonzero(np.rint(r2.x) > 0.5) if (r2.success and r2.x is not None) else order_by_risk[:k_safe]

    return chosen, len(chosen)


def compute_ipw_weights(n_rejected_pool, k_acquired, w_min=1.8, w_max=5.0):
    """
    Computes clipped Horvitz-Thompson inverse acquisition propensity weight for
    explored rejected samples during classifier retraining.
    """
    if k_acquired <= 0:
        return 1.0
    return float(np.clip(n_rejected_pool / float(k_acquired), w_min, w_max))
