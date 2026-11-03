"""
DC-PRISM Core Engine (integration-v1; see docs/CODE_AUDIT.zh.md)
===========================================================
Implements:
  1. P0-R1: Exact unscaled Bernstein Risk Envelope (ell = log(1 / alpha)).
  2. Empirical Wilson/Laplace upper-risk scores (no simultaneous coverage certificate).
  3. P0-R3: Exact 2^n enumeration for n <= 16 and M(S)-stratified multi-weight
            Safe-MinRisk solver with lower/upper bounds and optimality gap.
  4. P0-R4 & P0-R5: Common-Random-Worlds full-covariance Laplace quantile LCB_beta
            with explicit quantity penalty:
            Surplus(S; M_K) = LCB_beta(Delta V(S; M_K)) - lambda_K * (K_safe - |S|) / K_safe.
  5. External Baseline: BAIT-Cost-Safe (Fisher trace optimization under R_alpha(S) <= B_t).
"""
import hashlib
import json
import itertools
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

class UnifiedLogisticLearner:
    """
    Strictly identical L2-regularized Logistic Regression used across ALL policies.
    """
    def __init__(self, l2=0.45):
        self.l2 = float(l2)
        self.w = None
        self.H_inv = None

    def fit(self, X, y, w_init=None):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        n, d = X.shape
        Xb = np.column_stack([np.ones(n), X])

        def loss_and_grad(w):
            p = expit(Xb @ w)
            nll = -np.sum(y * np.log(np.clip(p, 1e-12, 1.0)) + (1.0 - y) * np.log(np.clip(1.0 - p, 1e-12, 1.0)))
            reg = 0.5 * self.l2 * np.sum(w[1:] ** 2)
            grad = Xb.T @ (p - y)
            grad[1:] += self.l2 * w[1:]
            return nll + reg, grad

        x0 = np.zeros(d + 1) if w_init is None else np.asarray(w_init, float)
        res = minimize(loss_and_grad, x0, jac=True, method="BFGS", options={"maxiter": 55, "gtol": 1e-5})
        self.w = res.x.copy()
        p_fit = np.clip(expit(Xb @ self.w), 1e-6, 1.0 - 1e-6)
        wv = p_fit * (1.0 - p_fit)
        H = (Xb * wv[:, None]).T @ Xb
        reg_mat = self.l2 * np.eye(d + 1)
        reg_mat[0, 0] = 1e-4
        self.H_inv = np.linalg.pinv(H + reg_mat, rcond=1e-12)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, float)
        Xb = np.column_stack([np.ones(len(X)), X])
        return np.clip(expit(Xb @ self.w), 1e-9, 1.0 - 1e-9)

    def get_hash(self):
        if self.w is None:
            return "unfitted"
        return hashlib.sha256(np.round(self.w, 6).tobytes()).hexdigest()[:16]


def brier_score(y_true, p_pred):
    y_true = np.asarray(y_true, float)
    p_pred = np.asarray(p_pred, float)
    return float(np.mean((y_true - p_pred) ** 2))


def bernstein_components(q_sel, ead_sel, alpha=0.10):
    """
    Exact unscaled Bernstein Risk Envelope components (P0-R1):
      ell     = log(1 / alpha)
      mu_U(S) = sum_{i in S} q_i * E_i
      V_U(S)  = sum_{i in S} E_i^2 * [q_i(1-q_i) if q_i <= 0.5 else 0.25]
      M(S)    = max_{i in S} E_i
      R_alpha(S) = mu_U(S) + sqrt(2 * V_U(S) * ell) + (M(S) / 3.0) * ell
    """
    q_sel = np.asarray(q_sel, float)
    ead_sel = np.asarray(ead_sel, float)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if q_sel.shape != ead_sel.shape or q_sel.ndim != 1:
        raise ValueError("q and exposure must be matching one-dimensional arrays")
    if not np.isfinite(q_sel).all() or not np.isfinite(ead_sel).all() or np.any((q_sel < 0)|(q_sel > 1)) or np.any(ead_sel < 0):
        raise ValueError("Finite probabilities in [0,1] and nonnegative exposures required")
    if len(q_sel) == 0:
        return {"mu_U": 0.0, "V_U": 0.0, "M": 0.0, "ell": float(np.log(1.0 / alpha)), "risk_envelope": 0.0}
    ell = float(np.log(1.0 / max(float(alpha), 1e-12)))
    mu_u = float(np.sum(q_sel * ead_sel))
    v_i = np.where(q_sel <= 0.5, q_sel * (1.0 - q_sel), 0.25)
    v_u = float(np.sum((ead_sel ** 2) * v_i))
    m_s = float(np.max(ead_sel))
    r_env = mu_u + float(np.sqrt(2.0 * v_u * ell)) + (m_s / 3.0) * ell
    return {"mu_U": mu_u, "V_U": v_u, "M": m_s, "ell": ell, "risk_envelope": float(r_env)}


def bernstein_envelope(q_sel, ead_sel, alpha=0.10):
    return bernstein_components(q_sel, ead_sel, alpha=alpha)["risk_envelope"]


def compute_calibrated_upper_risk(
    learner, X_D, y_D, X_C, delta_r=0.10, risk_regime="Calibrated", n_draws=40, rng=None
):
    """Empirical upper-risk score using visible training labels only.

    Uses an in-sample upper-half Wilson shift and coordinatewise Laplace draw
    quantiles at the legacy effective size 8. This is not out-of-fold calibration
    and does not establish simultaneous coverage over a data-adaptive selected
    subset. Retained numerics support legacy comparisons; see docs/THEORY.md.
    """
    if rng is None:
        rng = np.random.default_rng(2026)
    X_D = np.asarray(X_D, float)
    y_D = np.asarray(y_D, float)
    X_C = np.asarray(X_C, float)

    p_D = learner.predict_proba(X_D)
    Xb_C = np.column_stack([np.ones(len(X_C)), X_C])
    p_C_hat = learner.predict_proba(X_C)

    if risk_regime == "Optimistic_Bias":
        # Uncalibrated optimistic risk estimate (underestimates tail risk under selective labels)
        q_opt = np.clip(p_C_hat * 0.58, 1e-6, 0.999)
        return q_opt, {"delta_cal": 0.0, "regime": "Optimistic_Bias"}

    # One-sided Wilson upper bound calibration on upper half of D_t
    mask_hi = p_D >= np.percentile(p_D, 50)
    n_hi = max(int(np.sum(mask_hi)), 1)
    emp_hi = float(np.mean(y_D[mask_hi]))
    prd_hi = float(np.clip(np.mean(p_D[mask_hi]), 1e-4, 0.99))
    z_wilson = 1.645
    wilson_upper = (emp_hi + (z_wilson**2)/(2*n_hi) + z_wilson * np.sqrt((emp_hi*(1-emp_hi) + (z_wilson**2)/(4*n_hi))/n_hi)) / (1 + (z_wilson**2)/n_hi)
    wilson_upper = float(np.clip(wilson_upper, prd_hi, 0.98))
    delta_cal = float(np.clip(np.log(wilson_upper / (1.0 - wilson_upper)) - np.log(prd_hi / (1.0 - prd_hi)), 0.15, 1.15))

    # Full-covariance Laplace posterior draws on logit space
    cov = 0.5 * (learner.H_inv + learner.H_inv.T)
    cov_reg = cov + 1e-8 * np.eye(len(learner.w))
    try:
        L_chol = np.linalg.cholesky(cov_reg)
        z_mat = rng.normal(0.0, 1.0, size=(n_draws, len(learner.w)))
        w_draws = learner.w[None, :] + z_mat @ L_chol.T
    except Exception:
        std_d = np.sqrt(np.maximum(np.diag(cov_reg), 1e-8))
        w_draws = learner.w[None, :] + rng.normal(0.0, 1.0, size=(n_draws, len(learner.w))) * std_d[None, :]

    logits_draws = w_draws @ Xb_C.T + delta_cal
    # Simultaneous quantile across selected set size m_eff ~ 10: level = 1 - delta_r / 10
    q_level = float(np.clip(1.0 - (delta_r / 8.0), 0.85, 0.99))
    logit_ucb = np.quantile(logits_draws, q_level, axis=0)
    q_cal = np.clip(expit(logit_ucb), 1e-6, 0.999)

    if risk_regime == "Conservative_Bias":
        q_cons = np.clip(expit(logit_ucb + 0.55), 1e-6, 0.999)
        return q_cons, {"delta_cal": delta_cal + 0.55, "regime": "Conservative_Bias"}

    return q_cal, {"delta_cal": delta_cal, "regime": "Calibrated"}


def solve_safe_minrisk_exact_or_bounded(q_C, ead_C, B_t, alpha=0.10, exact_max_n=16):
    """
    P0-R3: Solves Safe-MinRisk (M_K):
      max_{S subseteq C_t} |S|  s.t.  R_alpha(S) <= B_t
      breaking ties by minimum R_alpha(S).
    - For n <= exact_max_n (e.g. n <= 16 in P0): performs exact 2^n enumeration
      (100% guaranteed global optimum, optimality_gap = 0).
    - For n > exact_max_n: performs heuristic M(S)-stratified search across all distinct
      exposure ceilings m in E_C and variance weights, plus computes a rigorous
      analytical upper bound K_safe_upper and optimality gap.
    """
    q_C = np.asarray(q_C, float)
    ead_C = np.asarray(ead_C, float)
    n = len(q_C)
    if n == 0 or B_t <= 0:
        return np.array([], int), {
            "k_safe_lower": 0, "k_safe_upper": 0, "optimality_gap": 0,
            "solver_status": "EMPTY_OR_ZERO_BUDGET", "risk_envelope": 0.0
        }

    ell = float(np.log(1.0 / max(float(alpha), 1e-12)))

    if n <= exact_max_n:
        max_k = 0
        min_env = np.inf
        best_S = np.array([], int)
        for mask in range(1, 1 << n):
            S_idx = np.flatnonzero([mask & (1 << i) for i in range(n)])
            k_cur = len(S_idx)
            if k_cur < max_k:
                continue
            env = bernstein_envelope(q_C[S_idx], ead_C[S_idx], alpha=alpha)
            if env <= B_t + 1e-12:
                if k_cur > max_k or (k_cur == max_k and env < min_env):
                    max_k = k_cur
                    min_env = env
                    best_S = S_idx
        if max_k == 0:
            min_env = 0.0
        return np.sort(best_S), {
            "k_safe_lower": int(max_k),
            "k_safe_upper": int(max_k),
            "optimality_gap": 0,
            "solver_status": "EXACT_ENUMERATION",
            "risk_envelope": float(min_env),
        }

    # For n > exact_max_n:
    # 1. Compute rigorous upper bound K_safe_upper by stratifying on M(S) = m
    v_i = np.where(q_C <= 0.5, q_C * (1.0 - q_C), 0.25)
    mu_i = q_C * ead_C
    var_i = (ead_C ** 2) * v_i

    unique_m = np.unique(ead_C)
    best_S = np.array([], int)
    max_k_found = 0
    min_env_found = np.inf
    k_upper_global = 0

    for m_cap in unique_m:
        rem_budget = B_t - (m_cap / 3.0) * ell
        if rem_budget <= 0:
            continue
        elig = np.flatnonzero(ead_C <= m_cap + 1e-12)
        if len(elig) <= max_k_found:
            continue

        # Upper bound at ceiling m_cap: sort mu_i and var_i independently among eligible items
        mu_sorted = np.cumsum(np.sort(mu_i[elig]))
        var_sorted = np.cumsum(np.sort(var_i[elig]))
        min_possible_lhs = mu_sorted + np.sqrt(2.0 * var_sorted * ell)
        k_up_m = int(np.sum(min_possible_lhs <= rem_budget + 1e-9))
        if k_up_m > k_upper_global:
            k_upper_global = k_up_m

        # Multi-weight convex combination search over (mu_i + gamma * sqrt(var_i))
        for gamma in (0.0, 0.15, 0.35, 0.70, 1.40):
            score_g = mu_i[elig] + gamma * np.sqrt(var_i[elig] * ell)
            ord_g = elig[np.argsort(score_g, kind="stable")]
            # Binary/prefix search for largest feasible prefix
            for k_try in range(min(len(ord_g), k_up_m), max_k_found - 1, -1):
                if k_try == 0:
                    break
                cand = ord_g[:k_try]
                env_c = bernstein_envelope(q_C[cand], ead_C[cand], alpha=alpha)
                if env_c <= B_t + 1e-12:
                    if k_try > max_k_found or (k_try == max_k_found and env_c < min_env_found):
                        max_k_found = k_try
                        min_env_found = env_c
                        best_S = cand
                    break

    # 1->1 swap refinement at max_k_found to minimize R_alpha(M_K)
    if 0 < max_k_found < n:
        in_s = np.zeros(n, dtype=bool)
        in_s[best_S] = True
        out_s = np.flatnonzero(~in_s)
        for u in out_s[np.argsort(mu_i[out_s], kind="stable")[:min(len(out_s), 15)]]:
            worst_pos = int(np.argmax(mu_i[best_S] + 0.35 * np.sqrt(var_i[best_S] * ell)))
            trial = best_S.copy()
            trial[worst_pos] = u
            env_t = bernstein_envelope(q_C[trial], ead_C[trial], alpha=alpha)
            if env_t < min_env_found:
                min_env_found = env_t
                best_S = trial

    if max_k_found == 0:
        min_env_found = 0.0
    gap = max(int(k_upper_global - max_k_found), 0)
    # The bound certifies cardinality only, not the minimum-risk tie break.
    status = "CARDINALITY_CERTIFIED_RISK_HEURISTIC" if gap == 0 else "BOUNDED_M_STRATIFIED"
    return np.sort(best_S), {
        "k_safe_lower": int(max_k_found),
        "k_safe_upper": int(max(k_upper_global, max_k_found)),
        "optimality_gap": int(gap),
        "solver_status": status,
        "risk_envelope": float(min_env_found),
    }


def build_pending_fisher_and_target(X_D, X_O, X_Q, w_vec, l2=0.45):
    """
    Constructs H_{D+O}(w) = l2*I + I(D; w) + E[I(O; w)] and G_Q(w)
    using ZERO labels from O or Q.
    """
    X_D = np.asarray(X_D, float)
    X_Q = np.asarray(X_Q, float)
    d_plus_1 = len(w_vec)

    Xb_D = np.column_stack([np.ones(len(X_D)), X_D])
    p_D = expit(Xb_D @ w_vec)
    w_D = np.clip(p_D * (1.0 - p_D), 1e-6, None)
    I_D = (Xb_D * w_D[:, None]).T @ Xb_D

    if X_O is not None and len(X_O) > 0:
        X_O = np.asarray(X_O, float)
        Xb_O = np.column_stack([np.ones(len(X_O)), X_O])
        p_O = expit(Xb_O @ w_vec)
        w_O = np.clip(p_O * (1.0 - p_O), 1e-6, None)
        I_O = (Xb_O * w_O[:, None]).T @ Xb_O
    else:
        I_O = np.zeros((d_plus_1, d_plus_1))

    reg = l2 * np.eye(d_plus_1)
    reg[0, 0] = 1e-4
    H_DO = reg + I_D + I_O
    H_DO_inv = np.linalg.pinv(H_DO, rcond=1e-12)

    Xb_Q = np.column_stack([np.ones(len(X_Q)), X_Q])
    p_Q = expit(Xb_Q @ w_vec)
    w_Q = np.clip(p_Q * (1.0 - p_Q), 1e-6, None)
    G_Q = ((Xb_Q * (w_Q[:, None] ** 2)).T @ Xb_Q) / max(len(X_Q), 1)
    return H_DO, H_DO_inv, G_Q


def pending_aware_value(S_idx, Xb_C, w_C, H_DO, H_DO_inv, G_Q):
    """Exact matrix-inverse Pending-Aware Brier reduction proxy V(S)."""
    if len(S_idx) == 0:
        return 0.0
    X_s = Xb_C[S_idx]
    w_s = w_C[S_idx]
    A_S = (X_s * w_s[:, None]).T @ X_s
    H_post_inv = np.linalg.pinv(H_DO + A_S, rcond=1e-12)
    val = float(np.einsum("ij,ji->", G_Q, H_DO_inv - H_post_inv))
    return max(val, 0.0)


def sample_common_parameter_worlds(learner, n_worlds=20, rng=None):
    """
    P0-R5: Generates M common parameter worlds {theta^(m)}_{m=1}^M from the
    full-covariance Laplace posterior N(w_hat, H_D^{-1}) on D_t.
    """
    if rng is None:
        rng = np.random.default_rng(2026)
    cov = 0.5 * (learner.H_inv + learner.H_inv.T)
    cov_reg = cov + 1e-8 * np.eye(len(learner.w))
    try:
        L_chol = np.linalg.cholesky(cov_reg)
        z = rng.normal(0.0, 1.0, size=(n_worlds, len(learner.w)))
        draws = learner.w[None, :] + z @ L_chol.T
    except Exception:
        std_d = np.sqrt(np.maximum(np.diag(cov_reg), 1e-8))
        draws = learner.w[None, :] + rng.normal(0.0, 1.0, size=(n_worlds, len(learner.w))) * std_d[None, :]
    return np.vstack([learner.w[None, :], draws])


def evaluate_common_worlds_surplus(
    S_idx, M_K_idx, X_D, X_O, X_C, X_Q, learner, w_worlds,
    beta_val=0.20, lambda_K=0.22
):
    """
    P0-R4 & P0-R5:
    In each common parameter world m = 1..M drawn from full-covariance Laplace posterior:
      Evaluates second-order curvature-damped Fisher reduction (for conservative LCB_beta)
      and optimistic curvature-bonus Fisher reduction (for optimistic UCB_{1-beta}),
      normalized by the safe anchor utility scale:
        Delta V_m(S; M_K) = tanh((V_m(S) - V_m(M_K)) / max(2.0 * V_m(M_K), 1e-5))
      Subtracts explicit quantity penalty:
        qty_pen = lambda_K * (|M_K| - |S|) / max(|M_K|, 1)
    """
    Xb_C = np.column_stack([np.ones(len(X_C)), X_C])
    k_sac_ratio = max(len(M_K_idx) - len(S_idx), 0) / float(max(len(M_K_idx), 1))
    qty_pen = float(lambda_K * k_sac_ratio)
    w_hat = learner.w

    deltas_lcb = []
    deltas_ucb = []
    draws_iter = w_worlds[1:] if len(w_worlds) > 1 else w_worlds
    for w_m in draws_iter:
        H_m, H_inv_m, G_m = build_pending_fisher_and_target(X_D, X_O, X_Q, w_m, l2=learner.l2)
        p_m = expit(Xb_C @ w_m)
        shift_sq = (Xb_C @ (w_m - w_hat)) ** 2
        w_C_con = np.clip(p_m * (1.0 - p_m) * np.exp(-0.45 * shift_sq), 1e-6, None)
        w_C_opt = np.clip(p_m * (1.0 - p_m) * (1.0 + 0.65 * shift_sq), 1e-6, None)

        v_s_con = pending_aware_value(S_idx, Xb_C, w_C_con, H_m, H_inv_m, G_m)
        v_mk_con = max(pending_aware_value(M_K_idx, Xb_C, w_C_con, H_m, H_inv_m, G_m), 1e-6)
        deltas_lcb.append(np.tanh((v_s_con - v_mk_con) / (2.5 * v_mk_con)))

        v_s_opt = pending_aware_value(S_idx, Xb_C, w_C_opt, H_m, H_inv_m, G_m)
        v_mk_opt = max(pending_aware_value(M_K_idx, Xb_C, w_C_opt, H_m, H_inv_m, G_m), 1e-6)
        deltas_ucb.append(np.tanh((v_s_opt - v_mk_opt) / (2.5 * v_mk_opt)))

    deltas_lcb = np.asarray(deltas_lcb, float)
    deltas_ucb = np.asarray(deltas_ucb, float)
    lcb_raw = float(np.quantile(deltas_lcb, beta_val))
    ucb_raw = float(np.quantile(deltas_ucb, 1.0 - beta_val))
    mean_raw = float(np.mean(deltas_lcb))
    return {
        "LCB_surplus": lcb_raw - qty_pen,
        "UCB_surplus": ucb_raw - qty_pen,
        "Mean_surplus": mean_raw - qty_pen,
        "lcb_raw": lcb_raw,
        "ucb_raw": ucb_raw,
        "mean_raw": mean_raw,
        "qty_penalty": qty_pen,
        "std_raw": float(np.std(deltas_lcb, ddof=1)) if len(deltas_lcb) > 1 else 0.0,
    }


def generate_candidate_family(X_D, X_O_eff, X_C, X_Q, q_C, ead_C, B_t, M_K, learner, alpha=0.10):
    """
    Stage 3 & 4 Candidate Family Generator:
    Generates diverse, Bernstein-feasible candidate subsets across the cardinality
    spectrum k in {1, ..., K_safe} by:
      (a) Marginal utility-to-marginal-Bernstein-risk greedy trajectory;
      (b) Pure utility top-k prefix filtered for Bernstein feasibility;
      (c) Hybrid prefix-replacement sets (retaining m safest items from M_K and
          filling remaining budget B_t with highest-value tail items).
    """
    n_C = len(X_C)
    k_safe = len(M_K)
    H_DO, H_DO_inv, G_Q = build_pending_fisher_and_target(X_D, X_O_eff, X_Q, learner.w, l2=learner.l2)
    Xb_C = np.column_stack([np.ones(n_C), X_C])
    p_C = learner.predict_proba(X_C)
    w_C = np.clip(p_C * (1.0 - p_C), 1e-6, None)

    M_mat = H_DO_inv @ G_Q @ H_DO_inv
    q_quad = np.maximum(np.einsum("ij,jk,ik->i", Xb_C, M_mat, Xb_C), 0.0)
    h_quad = np.maximum(np.einsum("ij,jk,ik->i", Xb_C, H_DO_inv, Xb_C), 0.0)
    v_single = (w_C * q_quad) / (1.0 + w_C * h_quad)

    cand_dict = {tuple(np.sort(M_K).tolist()): M_K}

    # (a) Marginal Delta V_i / Delta R_i greedy sequence
    S_cur = []
    rem = set(range(n_C))
    H_cur_inv = H_DO_inv.copy()
    cur_r = 0.0
    for _ in range(k_safe):
        if not rem:
            break
        rem_arr = np.array(sorted(list(rem)), dtype=int)
        M_cur = H_cur_inv @ G_Q @ H_cur_inv
        X_rem = Xb_C[rem_arr]
        w_rem = w_C[rem_arr]
        q_r = np.maximum(np.einsum("ij,jk,ik->i", X_rem, M_cur, X_rem), 0.0)
        h_r = np.maximum(np.einsum("ij,jk,ik->i", X_rem, H_cur_inv, X_rem), 0.0)
        delta_v = (w_rem * q_r) / (1.0 + w_rem * h_r)

        unit_r_rem = q_C[rem_arr] * ead_C[rem_arr]
        top_pos = np.argsort(-delta_v / np.maximum(unit_r_rem, 1e-8), kind="stable")[:18]
        best_i, best_ratio = None, -1.0
        for pos in top_pos:
            cand_i = rem_arr[pos]
            trial = np.array(S_cur + [cand_i], dtype=int)
            r_new = bernstein_envelope(q_C[trial], ead_C[trial], alpha=alpha)
            if r_new <= B_t + 1e-12:
                d_r = max(r_new - cur_r, 1e-6)
                ratio = delta_v[pos] / d_r
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_i = cand_i
        if best_i is None:
            break
        S_cur.append(best_i)
        rem.remove(best_i)
        cur_r = bernstein_envelope(q_C[S_cur], ead_C[S_cur], alpha=alpha)
        x_i = Xb_C[best_i]
        w_i = w_C[best_i]
        u = H_cur_inv @ x_i
        H_cur_inv = H_cur_inv - (w_i / (1.0 + w_i * float(x_i @ u))) * np.outer(u, u)
        arr_s = np.sort(np.array(S_cur, dtype=int))
        cand_dict[tuple(arr_s.tolist())] = arr_s

    # (b) Pure high-value greedy packing (maximizes raw V_i regardless of unit cost)
    val_order = np.argsort(-v_single, kind="stable")
    S_val = []
    for idx_i in val_order:
        trial = np.array(S_val + [idx_i], dtype=int)
        if bernstein_envelope(q_C[trial], ead_C[trial], alpha=alpha) <= B_t + 1e-12:
            S_val.append(idx_i)
            arr_v = np.sort(np.array(S_val, dtype=int))
            cand_dict[tuple(arr_v.tolist())] = arr_v

    # (c) Hybrid exchange sets: keep m_keep items from M_K and pack high-v_single items
    for frac in (0.25, 0.50, 0.70, 0.85):
        m_keep = max(int(np.round(frac * k_safe)), 1)
        mk_by_v = M_K[np.argsort(-v_single[M_K], kind="stable")[:m_keep]]
        S_hyb = list(mk_by_v)
        for idx_i in val_order:
            if idx_i in S_hyb:
                continue
            trial = np.array(S_hyb + [idx_i], dtype=int)
            if bernstein_envelope(q_C[trial], ead_C[trial], alpha=alpha) <= B_t + 1e-12:
                S_hyb.append(idx_i)
        arr_h = np.sort(np.array(S_hyb, dtype=int))
        cand_dict[tuple(arr_h.tolist())] = arr_h

    # (d) Curvature-reliable LCB candidate ranking (damps directions with excessive posterior logit variance)
    logit_var = np.maximum(np.einsum("ij,jk,ik->i", Xb_C, learner.H_inv, Xb_C), 0.0)
    v_single_lcb = v_single * np.exp(-0.45 * logit_var)
    unit_r_all = np.maximum(q_C * ead_C + 0.25 * ead_C * np.sqrt(np.maximum(q_C * (1.0 - q_C), 1e-4)), 1e-6)
    lcb_order = np.argsort(-v_single_lcb / unit_r_all, kind="stable")

    S_lcb_pure = []
    for idx_i in lcb_order:
        trial = np.array(S_lcb_pure + [idx_i], dtype=int)
        if bernstein_envelope(q_C[trial], ead_C[trial], alpha=alpha) <= B_t + 1e-12:
            S_lcb_pure.append(idx_i)
            arr_lp = np.sort(np.array(S_lcb_pure, dtype=int))
            cand_dict[tuple(arr_lp.tolist())] = arr_lp

    for frac in (0.40, 0.60, 0.75, 0.85):
        m_keep = max(int(np.round(frac * k_safe)), 1)
        mk_by_lcb = M_K[np.argsort(q_C[M_K] * ead_C[M_K], kind="stable")[:m_keep]]
        S_hyb_lcb = list(mk_by_lcb)
        for idx_i in lcb_order:
            if idx_i in S_hyb_lcb:
                continue
            trial = np.array(S_hyb_lcb + [idx_i], dtype=int)
            if bernstein_envelope(q_C[trial], ead_C[trial], alpha=alpha) <= B_t + 1e-12:
                S_hyb_lcb.append(idx_i)
        arr_hl = np.sort(np.array(S_hyb_lcb, dtype=int))
        cand_dict[tuple(arr_hl.tolist())] = arr_hl

    # (e) Optimistic UCB candidate ranking (favors high posterior logit variance directions)
    v_single_ucb = v_single * (1.0 + 0.85 * logit_var)
    ucb_order = np.argsort(-v_single_ucb / unit_r_all, kind="stable")
    S_ucb_pure = []
    for idx_i in ucb_order:
        trial = np.array(S_ucb_pure + [idx_i], dtype=int)
        if bernstein_envelope(q_C[trial], ead_C[trial], alpha=alpha) <= B_t + 1e-12:
            S_ucb_pure.append(idx_i)
            arr_up = np.sort(np.array(S_ucb_pure, dtype=int))
            cand_dict[tuple(arr_up.tolist())] = arr_up

    return list(cand_dict.values()), v_single, H_DO, H_DO_inv, G_Q


def select_acquisition_policy(
    policy_name, X_D, y_D, X_O, X_C, ead_C, X_Q, B_t,
    learner=None, alpha=0.10, delta_r=0.10, beta_val=0.20,
    lambda_K=0.22, delta_gate=0.08, fixed_eps=0.25,
    risk_regime="Calibrated", q_C_precomputed=None, w_worlds_precomputed=None,
    exact_max_n=16, rng=None
):
    """
    Unified dispatcher for all acquisition policies under identical learner,
    identical exogenous B_t, and identical Bernstein risk envelope:
      1. 'OrdinaryOnly'
      2. 'Orig-MinRisk' (Point risk p_hat * E_i <= B_t)
      3. 'Orig-PA'      (Point risk + Pending-aware utility)
      4. 'Safe-MinRisk' (M_K under exact Bernstein envelope R_alpha(S) <= B_t)
      5. 'Safe-PA-NoGate' (Maximizes point V(S) under R_alpha(S) <= B_t, lambda_K=0, no gate)
      6. 'Fixed-Eps-Safe-PA' (Fixed cardinality k = round((1-fixed_eps)*K_safe) under R_alpha(S) <= B_t)
      7. 'D-Only-Safe-PA' (Ignores pending O_t, uses H_D only + Value-LCB gate)
      8. 'PRISM-ValueUCB' (Uses optimistic UCB_{1-beta_val} instead of LCB_{beta_val})
      9. 'BAIT-Cost-Safe' (External baseline: maximizes Fisher trace tr(H_{D+O}^{-1} - (H_{D+O}+A_S)^{-1}) under R_alpha(S) <= B_t)
      10. 'DC-PRISM' (Full method: Risk-UCB + Pending-aware H_{D+O} + Value-LCB + lambda_K quantity penalty + fallback to M_K)
    """
    if rng is None:
        rng = np.random.default_rng(2026)
    if learner is None:
        learner = UnifiedLogisticLearner(l2=2.5).fit(X_D, y_D)

    n_C = len(X_C)
    if policy_name == "OrdinaryOnly" or n_C == 0 or B_t <= 0:
        return np.array([], int), np.array([], int), {
            "k_safe": 0, "eps_star": 1.0, "surplus_score": 0.0,
            "gate_triggered": False, "solver_status": "ORDINARY_ONLY"
        }

    # Point risk for Orig-MinRisk and Orig-PA
    p_C_hat = learner.predict_proba(X_C)
    if policy_name == "Orig-MinRisk":
        cost_pt = p_C_hat * ead_C
        ord_pt = np.argsort(cost_pt, kind="stable")
        k_pt = int(np.searchsorted(np.cumsum(cost_pt[ord_pt]), B_t, side="right"))
        S_pt = np.sort(ord_pt[:k_pt])
        return S_pt, S_pt, {"k_safe": k_pt, "eps_star": 0.0, "surplus_score": 0.0, "gate_triggered": False, "solver_status": "POINT_MINRISK"}

    if policy_name == "Orig-PA":
        H_DO, H_DO_inv, G_Q = build_pending_fisher_and_target(X_D, X_O, X_Q, learner.w, l2=learner.l2)
        Xb_C = np.column_stack([np.ones(n_C), X_C])
        w_C = np.clip(p_C_hat * (1.0 - p_C_hat), 1e-6, None)
        M_mat = H_DO_inv @ G_Q @ H_DO_inv
        v_s = (w_C * np.maximum(np.einsum("ij,jk,ik->i", Xb_C, M_mat, Xb_C), 0.0)) / (1.0 + w_C * np.maximum(np.einsum("ij,jk,ik->i", Xb_C, H_DO_inv, Xb_C), 0.0))
        cost_pt = p_C_hat * ead_C
        ord_pa = np.argsort(-v_s / np.maximum(cost_pt, 1e-8), kind="stable")
        S_pa, cur_c = [], 0.0
        for idx_i in ord_pa:
            if cur_c + cost_pt[idx_i] <= B_t + 1e-12:
                S_pa.append(idx_i)
                cur_c += cost_pt[idx_i]
        S_pa = np.sort(np.array(S_pa, dtype=int))
        return S_pa, S_pa, {"k_safe": len(S_pa), "eps_star": 0.0, "surplus_score": 0.0, "gate_triggered": True, "solver_status": "POINT_PA"}

    # All Safe methods share the EXACT SAME q_C and EXACT SAME Safe-MinRisk M_K!
    if q_C_precomputed is not None:
        q_C = np.asarray(q_C_precomputed, float)
    else:
        q_C, _ = compute_calibrated_upper_risk(learner, X_D, y_D, X_C, delta_r=delta_r, risk_regime=risk_regime, rng=rng)

    M_K, mk_info = solve_safe_minrisk_exact_or_bounded(q_C, ead_C, B_t, alpha=alpha, exact_max_n=exact_max_n)
    k_safe = len(M_K)
    if k_safe == 0 or policy_name == "Safe-MinRisk":
        return M_K, M_K, {
            "k_safe": k_safe, "eps_star": 0.0, "surplus_score": 0.0,
            "gate_triggered": False, "solver_status": mk_info["solver_status"],
            "k_safe_upper": mk_info["k_safe_upper"], "optimality_gap": mk_info["optimality_gap"]
        }

    if policy_name == "BAIT-Cost-Safe":
        # External baseline: maximizes unweighted Fisher parameter trace (G_Q = I) under R_alpha(S) <= B_t
        H_DO, H_DO_inv, _ = build_pending_fisher_and_target(X_D, X_O, X_Q, learner.w, l2=learner.l2)
        G_eye = np.eye(len(learner.w))
        Xb_C = np.column_stack([np.ones(n_C), X_C])
        w_C = np.clip(p_C_hat * (1.0 - p_C_hat), 1e-6, None)
        M_eye = H_DO_inv @ H_DO_inv
        v_bait = (w_C * np.maximum(np.einsum("ij,jk,ik->i", Xb_C, M_eye, Xb_C), 0.0)) / (1.0 + w_C * np.maximum(np.einsum("ij,jk,ik->i", Xb_C, H_DO_inv, Xb_C), 0.0))
        unit_r = q_C * ead_C + 0.20 * ead_C * np.sqrt(np.maximum(q_C * (1.0 - q_C), 1e-4))
        ord_b = np.argsort(-v_bait / np.maximum(unit_r, 1e-8), kind="stable")
        S_b = []
        for idx_i in ord_b:
            trial = np.array(S_b + [idx_i], dtype=int)
            if bernstein_envelope(q_C[trial], ead_C[trial], alpha=alpha) <= B_t + 1e-12:
                S_b.append(idx_i)
        S_b = np.sort(np.array(S_b, dtype=int))
        eps_b = 1.0 - len(S_b) / float(max(k_safe, 1))
        return S_b, M_K, {"k_safe": k_safe, "eps_star": eps_b, "surplus_score": 0.0, "gate_triggered": True, "solver_status": mk_info["solver_status"]}

    use_pending = (policy_name != "D-Only-Safe-PA")
    X_O_eff = X_O if use_pending else None

    cand_sets, v_single, H_DO, H_DO_inv, G_Q = generate_candidate_family(
        X_D, X_O_eff, X_C, X_Q, q_C, ead_C, B_t, M_K, learner, alpha=alpha
    )
    Xb_C = np.column_stack([np.ones(n_C), X_C])
    w_C = np.clip(p_C_hat * (1.0 - p_C_hat), 1e-6, None)

    if policy_name == "Fixed-Eps-Safe-PA":
        if not 0.0 <= fixed_eps <= 1.0:
            raise ValueError("fixed_eps must be in [0, 1]")
        k_tar = int(np.ceil((1.0 - fixed_eps) * k_safe))
        if k_tar == 0:
            return np.array([], int), M_K, {"k_safe": k_safe, "eps_star": 1.0, "surplus_score": 0.0, "gate_triggered": True, "solver_status": mk_info["solver_status"]}
        unit_r = q_C * ead_C + 0.20 * ead_C * np.sqrt(np.maximum(q_C * (1.0 - q_C), 1e-4))
        ord_f = np.argsort(-v_single / np.maximum(unit_r, 1e-8), kind="stable")
        S_f = []
        for idx_i in ord_f:
            trial = np.array(S_f + [idx_i], dtype=int)
            if bernstein_envelope(q_C[trial], ead_C[trial], alpha=alpha) <= B_t + 1e-12:
                S_f.append(idx_i)
                if len(S_f) == k_tar:
                    break
        if len(S_f) < k_tar:
            for idx_i in M_K:
                if idx_i not in S_f:
                    trial = np.array(S_f + [idx_i], dtype=int)
                    if bernstein_envelope(q_C[trial], ead_C[trial], alpha=alpha) <= B_t + 1e-12:
                        S_f.append(idx_i)
                        if len(S_f) == k_tar:
                            break
        if len(S_f) < k_tar:
            # A greedy partial set may block every remaining anchor item.
            # Restart from the feasible anchor rather than silently miss k_tar.
            S_f = M_K[:k_tar].tolist()
        S_f = np.sort(np.array(S_f, dtype=int))
        eps_f = 1.0 - len(S_f) / float(max(k_safe, 1))
        return S_f, M_K, {"k_safe": k_safe, "eps_star": eps_f, "surplus_score": 0.0, "gate_triggered": True, "solver_status": mk_info["solver_status"]}

    if policy_name == "Safe-PA-NoGate":
        # Pure point V(S) maximization without quantity penalty or LCB gate
        best_s = max(cand_sets, key=lambda s: pending_aware_value(s, Xb_C, w_C, H_DO, H_DO_inv, G_Q))
        eps_ng = 1.0 - len(best_s) / float(max(k_safe, 1))
        return best_s, M_K, {"k_safe": k_safe, "eps_star": eps_ng, "surplus_score": 0.0, "gate_triggered": True, "solver_status": mk_info["solver_status"]}

    # Common parameter worlds for DC-PRISM, D-Only-Safe-PA, and PRISM-ValueUCB
    w_worlds = w_worlds_precomputed if w_worlds_precomputed is not None else sample_common_parameter_worlds(learner, n_worlds=20, rng=rng)

    # Precompute world Fisher & target matrices and anchor M_K values ONCE across the M worlds (20x speedup!)
    draws_iter = w_worlds[1:] if len(w_worlds) > 1 else w_worlds
    w_hat = learner.w
    world_cache = []
    for w_m in draws_iter:
        H_m, H_inv_m, G_m = build_pending_fisher_and_target(X_D, X_O_eff, X_Q, w_m, l2=learner.l2)
        p_m = expit(Xb_C @ w_m)
        shift_sq = (Xb_C @ (w_m - w_hat)) ** 2
        w_C_con = np.clip(p_m * (1.0 - p_m) * np.exp(-0.45 * shift_sq), 1e-6, None)
        w_C_opt = np.clip(p_m * (1.0 - p_m) * (1.0 + 0.65 * shift_sq), 1e-6, None)
        v_mk_con = max(pending_aware_value(M_K, Xb_C, w_C_con, H_m, H_inv_m, G_m), 1e-6)
        v_mk_opt = max(pending_aware_value(M_K, Xb_C, w_C_opt, H_m, H_inv_m, G_m), 1e-6)
        world_cache.append((H_m, H_inv_m, G_m, w_C_con, w_C_opt, v_mk_con, v_mk_opt))

    best_score = -1e9
    best_cand = M_K.copy()
    best_stats = None

    for s_cand in cand_sets:
        k_sac_ratio = max(len(M_K) - len(s_cand), 0) / float(max(len(M_K), 1))
        qty_pen = float(lambda_K * k_sac_ratio)
        deltas_lcb = []
        deltas_ucb = []
        for H_m, H_inv_m, G_m, w_C_con, w_C_opt, v_mk_con, v_mk_opt in world_cache:
            if policy_name == "PRISM-ValueUCB":
                v_s_opt = pending_aware_value(s_cand, Xb_C, w_C_opt, H_m, H_inv_m, G_m)
                deltas_ucb.append(np.tanh((v_s_opt - v_mk_opt) / (2.5 * v_mk_opt)))
            else:
                v_s_con = pending_aware_value(s_cand, Xb_C, w_C_con, H_m, H_inv_m, G_m)
                deltas_lcb.append(np.tanh((v_s_con - v_mk_con) / (2.5 * v_mk_con)))
        if policy_name == "PRISM-ValueUCB":
            ucb_raw = float(np.quantile(deltas_ucb, 1.0 - beta_val))
            lcb_raw = 0.0
            score = ucb_raw - qty_pen
        else:
            lcb_raw = float(np.quantile(deltas_lcb, beta_val))
            ucb_raw = 0.0
            score = lcb_raw - qty_pen
        if score > best_score:
            best_score = score
            best_cand = s_cand
            best_stats = {"lcb_raw": lcb_raw, "ucb_raw": ucb_raw, "qty_penalty": qty_pen}

    if best_score > delta_gate:
        S_star = best_cand
        triggered = True
    else:
        S_star = M_K
        triggered = False

    eps_star = 1.0 - len(S_star) / float(max(k_safe, 1))
    return S_star, M_K, {
        "k_safe": k_safe,
        "eps_star": float(eps_star),
        "surplus_score": float(best_score),
        "lcb_raw": float(best_stats["lcb_raw"]) if best_stats else 0.0,
        "ucb_raw": float(best_stats["ucb_raw"]) if best_stats else 0.0,
        "qty_penalty": float(best_stats["qty_penalty"]) if best_stats else 0.0,
        "gate_triggered": bool(triggered),
        "solver_status": mk_info["solver_status"],
        "k_safe_upper": mk_info["k_safe_upper"],
        "optimality_gap": mk_info["optimality_gap"],
    }


def compute_exogenous_budget(q_C, ead_C, alpha=0.10, target_k=5):
    """
    Computes an exogenous, policy-independent budget B_t for a state from the
    target_k lowest-risk items in C_t, returning (B_t, budget_hash).
    """
    q_C = np.asarray(q_C, float)
    ead_C = np.asarray(ead_C, float)
    n = len(q_C)
    if n == 0:
        return 0.0, "empty"
    k_eff = min(max(int(target_k), 1), n)
    ell = float(np.log(1.0 / max(float(alpha), 1e-12)))
    v_i = np.where(q_C <= 0.5, q_C * (1.0 - q_C), 0.25)
    score = q_C * ead_C + 0.35 * ead_C * np.sqrt(v_i * ell)
    idx_ref = np.argsort(score, kind="stable")[:k_eff]
    B_t = float(bernstein_envelope(q_C[idx_ref], ead_C[idx_ref], alpha=alpha) * 1.03)
    b_hash = hashlib.sha256(f"{B_t:.8f}_{alpha:.4f}_{n}".encode()).hexdigest()[:16]
    return B_t, b_hash


def roc_auc_score_np(y_true, p_pred):
    """Pure NumPy/SciPy Mann-Whitney U formulation of ROC-AUC."""
    y_true = np.asarray(y_true, int)
    p_pred = np.asarray(p_pred, float)
    pos = p_pred[y_true == 1]
    neg = p_pred[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    diff = pos[:, None] - neg[None, :]
    auc = (np.sum(diff > 0) + 0.5 * np.sum(diff == 0)) / float(len(pos) * len(neg))
    return float(auc)


def recall_at_threshold_np(y_true, p_pred, thr=0.30):
    """Recall on positive (default) class at probability threshold thr."""
    y_true = np.asarray(y_true, int)
    p_pred = np.asarray(p_pred, float)
    pos_mask = y_true == 1
    if np.sum(pos_mask) == 0:
        return 0.0
    return float(np.mean(p_pred[pos_mask] >= thr))


def ece_score_np(y_true, p_pred, n_bins=10):
    """Expected Calibration Error (ECE) across n_bins equal-width probability bins."""
    y_true = np.asarray(y_true, float)
    p_pred = np.asarray(p_pred, float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    if n == 0:
        return 0.0
    for b_idx in range(n_bins):
        if b_idx == n_bins - 1:
            mask = (p_pred >= bins[b_idx]) & (p_pred <= bins[b_idx + 1])
        else:
            mask = (p_pred >= bins[b_idx]) & (p_pred < bins[b_idx + 1])
        if np.any(mask):
            ece += (np.sum(mask) / float(n)) * abs(float(np.mean(p_pred[mask])) - float(np.mean(y_true[mask])))
    return float(ece)


def solve_fixed_k_minrisk_subset(q_C, ead_C, B_t, k_target, alpha=0.10, exact_max_n=16, feasible_anchor=None):
    """
    Exact minimum-risk feasible k-subset for n <= exact_max_n.
    For larger pools return the best feasible search candidate, with no global
    optimality claim. A supplied feasible anchor guarantees a k-prefix fallback.
    Raise ValueError if the search finds no feasible k-subset.
    """
    q_C = np.asarray(q_C, float)
    ead_C = np.asarray(ead_C, float)
    n = len(q_C)
    k_target = int(k_target)
    if k_target <= 0 or n == 0:
        return np.array([], dtype=int), 0.0
    if k_target > n:
        raise ValueError("k_target exceeds the candidate count")

    if n <= exact_max_n:
        candidates = (np.array(s, dtype=int) for s in itertools.combinations(range(n), k_target))
    else:
        ell = float(np.log(1.0 / float(alpha)))
        v_i = np.where(q_C <= 0.5, q_C * (1.0 - q_C), 0.25)
        candidates = [np.sort(np.argsort(q_C * ead_C + gamma * ead_C * np.sqrt(v_i * ell), kind="stable")[:k_target])
                      for gamma in (0.0, 0.20, 0.40, 0.80, 1.40)]
        if feasible_anchor is not None:
            anchor = np.asarray(feasible_anchor, dtype=int)
            if len(anchor) >= k_target:
                order = np.argsort(q_C[anchor] * ead_C[anchor], kind="stable")
                candidates.append(np.sort(anchor[order[:k_target]]))

    best_S, min_env = None, np.inf
    for cand in candidates:
        env = bernstein_envelope(q_C[cand], ead_C[cand], alpha=alpha)
        if env <= B_t + 1e-12 and env < min_env - 1e-12:
            best_S, min_env = cand, env
    if best_S is None:
        raise ValueError("No feasible fixed-cardinality subset found; not a proof of infeasibility for a large pool")
    return best_S, float(min_env)


def compute_q_c_j_decomposition(
    S_sel, M_K, q_C, ead_C, B_t, X_base, y_base, X_C, y_C, X_T, y_T, w_init=None, alpha=0.10, l2=0.45
):
    """
    Section 12: Computes the exact Quantity-Composition Decomposition (Q_k, C_k, J)
    from the EXACT SAME state, same learner, same outcome release, and same evaluation set:
      - M_K: Safe-MinRisk anchor set of size K = |M_K|
      - S  : Policy selected set of size k = |S_sel|
      - M_k: Minimum-risk safe subset of size k = |S_sel| from the same state
      - Q_k = R(M_k) - R(M_K)   (Quantity Loss from reducing cardinality K -> k)
      - C_k = R(M_k) - R(S)     (Composition Gain from replacing M_k with informative S at fixed k)
      - J   = C_k - Q_k = R(M_K) - R(S)  (Net Brier Gain over Safe-MinRisk)
    """
    K_val = len(M_K)
    k_val = len(S_sel)
    eps_star = 1.0 - (k_val / float(max(K_val, 1)))

    if k_val == K_val:
        M_k = np.asarray(M_K, int)
        env_Mk_small = bernstein_envelope(q_C[M_k], ead_C[M_k], alpha=alpha)
    else:
        M_k, env_Mk_small = solve_fixed_k_minrisk_subset(q_C, ead_C, B_t, k_target=k_val, alpha=alpha, feasible_anchor=M_K)
    env_MK = bernstein_envelope(q_C[M_K], ead_C[M_K], alpha=alpha) if K_val > 0 else 0.0
    env_S = bernstein_envelope(q_C[S_sel], ead_C[S_sel], alpha=alpha) if k_val > 0 else 0.0

    def eval_subset_brier(sub_idx):
        if len(sub_idx) > 0:
            X_tr = np.vstack([X_base, X_C[sub_idx]])
            y_tr = np.concatenate([y_base, y_C[sub_idx]])
        else:
            X_tr = X_base
            y_tr = y_base
        pl = UnifiedLogisticLearner(l2=l2).fit(X_tr, y_tr, w_init=w_init)
        return brier_score(y_T, pl.predict_proba(X_T))

    r_MK = eval_subset_brier(M_K)
    if np.array_equal(np.sort(S_sel), np.sort(M_K)):
        r_S = r_MK
        r_Mk_small = r_MK
    else:
        r_S = eval_subset_brier(S_sel)
        r_Mk_small = eval_subset_brier(M_k) if k_val < K_val else r_MK

    Q_k = float(r_Mk_small - r_MK)
    C_k = float(r_Mk_small - r_S)
    J_net = float(C_k - Q_k)  # Strictly equals r_MK - r_S

    return {
        "K": int(K_val),
        "k": int(k_val),
        "eps_star": float(eps_star),
        "brier_M_K": float(r_MK),
        "brier_M_k": float(r_Mk_small),
        "brier_S": float(r_S),
        "Q_k": Q_k,
        "C_k": C_k,
        "J": J_net,
        "risk_M_K": float(env_MK),
        "risk_M_k": float(env_Mk_small),
        "risk_S": float(env_S),
        "M_k_optimality": "EXACT" if len(q_C) <= 16 else "FEASIBLE_HEURISTIC",
        "evaluation_role": "offline_post_settlement_counterfactual",
    }


class SplitBeforePreprocessor:
    """
    P4-R2 & Section 16: Built-in leak-free preprocessor fitted strictly on Historical_D0 covariates
    and exposures (ZERO hidden y used!). Applies standard Z-score scaling to core features (0..3)
    and deadzone outlier thresholding to auxiliary features (4..5) beyond the 90th percentile of D_0.
    """
    def __init__(self, tail_scale=1.35):
        self.tail_scale = float(tail_scale)
        self.medians = None
        self.means = None
        self.stds = None
        self.tau_aux = None
        self.ead_p5 = None
        self.ead_p95 = None
        self.ead_mean = None

    def fit(self, X_raw, ead_raw):
        X_raw = np.asarray(X_raw, float)
        ead_raw = np.asarray(ead_raw, float)
        self.medians = np.nanmedian(X_raw, axis=0)
        self.medians = np.where(np.isnan(self.medians), 0.0, self.medians)
        X_imp = np.where(np.isnan(X_raw), self.medians[None, :], X_raw)
        self.means = np.mean(X_imp, axis=0)
        self.stds = np.std(X_imp, axis=0)
        self.stds = np.where(self.stds < 1e-6, 1.0, self.stds)
        Z_d0 = np.clip((X_imp - self.means[None, :]) / self.stds[None, :], -4.5, 4.5)
        if Z_d0.shape[1] >= 6:
            self.tau_aux = np.percentile(np.abs(Z_d0[:, 4:6]), 90, axis=0)
        else:
            self.tau_aux = np.zeros(2)

        ead_clean = np.where(np.isnan(ead_raw) | (ead_raw <= 0), np.nanmedian(ead_raw), ead_raw)
        self.ead_p5 = float(np.percentile(ead_clean, 5))
        self.ead_p95 = float(np.percentile(ead_clean, 95))
        ead_win = np.clip(ead_clean, self.ead_p5, self.ead_p95)
        self.ead_mean = max(float(np.mean(ead_win)), 1e-6)
        return self

    def transform_X(self, X_raw):
        X_raw = np.asarray(X_raw, float)
        X_imp = np.where(np.isnan(X_raw), self.medians[None, :], X_raw)
        Z = np.clip((X_imp - self.means[None, :]) / self.stds[None, :], -4.5, 4.5)
        if Z.shape[1] >= 6 and self.tau_aux is not None:
            Z[:, 4:6] = self.tail_scale * np.sign(Z[:, 4:6]) * np.maximum(np.abs(Z[:, 4:6]) - self.tau_aux[None, :], 0.0)
        return Z

    def transform_ead(self, ead_raw):
        ead_raw = np.asarray(ead_raw, float)
        ead_clean = np.where(np.isnan(ead_raw) | (ead_raw <= 0), self.ead_p5, ead_raw)
        ead_win = np.clip(ead_clean, self.ead_p5, self.ead_p95)
        return ead_win / self.ead_mean


class BoostedTreeTransferLearner:
    """
    Step 6 Non-Linear Transfer Learner: Gradient Boosted Decision Stump/Tree Ensemble (GBDT)
    built directly into prism_engine.py for standalone reproducibility.
    """
    def __init__(self, n_estimators=25, lr=0.18):
        self.n_estimators = n_estimators
        self.lr = lr
        self.base_logit = 0.0
        self.trees = []

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        p_bar = float(np.clip(np.mean(y), 0.02, 0.98))
        self.base_logit = float(np.log(p_bar / (1.0 - p_bar)))
        f = np.full(len(y), self.base_logit, dtype=float)
        self.trees = []

        n, d = X.shape
        for _ in range(self.n_estimators):
            p = expit(f)
            residual = y - p
            best_gain = -1.0
            best_tree = None
            for j in range(d):
                col = X[:, j]
                for q in (25, 50, 75):
                    thr = float(np.percentile(col, q))
                    left = col <= thr
                    right = ~left
                    if np.sum(left) < 3 or np.sum(right) < 3:
                        continue
                    w_l = np.sum(p[left] * (1.0 - p[left])) + 1.5
                    w_r = np.sum(p[right] * (1.0 - p[right])) + 1.5
                    v_l = float(np.sum(residual[left]) / w_l)
                    v_r = float(np.sum(residual[right]) / w_r)
                    gain = (np.sum(residual[left]) ** 2) / w_l + (np.sum(residual[right]) ** 2) / w_r
                    if gain > best_gain:
                        best_gain = gain
                        best_tree = (j, thr, v_l, v_r)
            if best_tree is None:
                break
            j, thr, v_l, v_r = best_tree
            f += self.lr * np.where(X[:, j] <= thr, v_l, v_r)
            self.trees.append(best_tree)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, float)
        f = np.full(len(X), self.base_logit, dtype=float)
        for j, thr, v_l, v_r in self.trees:
            f += self.lr * np.where(X[:, j] <= thr, v_l, v_r)
        return np.clip(expit(f), 1e-6, 1.0 - 1e-6)
