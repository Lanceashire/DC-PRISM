#!/usr/bin/env python3
"""
Executes Round 2 Audit-Compliant Synthetic & Mechanism Suites:
  - P0 Exact Small-Pool Benchmark (40 states, n=12, 2^12 exact enumeration)
  - P1 Mechanism Pilot (32 pilot worlds, seeds 1001..1032, 3x3x3 cube + lambda_K tuning)
  - P2 Confirmation Benchmark (128 disjoint worlds, seeds 2001..2128):
      * 100% Pure Bernoulli(p_true) DGP (ZERO post-generation label overwrites!)
      * True-p Coverage (pointwise, selected-set simultaneous, candidate-pool simultaneous)
      * Empirical Safety Theorem Layer (mu_true, V_true, L_realized, R_alpha, envelope >= mu_true, violation severity)
      * Q_k / C_k / J Quantity-Composition Decomposition & BRP / HRR
      * Nearest-Neighbor Ablations A1..A5 with Holm-Bonferroni correction
  - P2 Cross-Mechanism Independent Confirmation (54 fresh worlds, seeds 2501..2554 across
    {Substitution, Unlocking, Irrelevant} x {Positive, Neutral, Negative})
"""
import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import wilcoxon, ttest_rel

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from prism_engine import (
    UnifiedLogisticLearner,
    brier_score,
    bernstein_components,
    bernstein_envelope,
    compute_calibrated_upper_risk,
    compute_exogenous_budget,
    solve_safe_minrisk_exact_or_bounded,
    sample_common_parameter_worlds,
    select_acquisition_policy,
    compute_q_c_j_decomposition,
)

PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def true_dgp_prob(X_mat, w_true, is_shift=False):
    """
    Section 1: Exact structural conditional probability p_i^* = P(Y_i = 1 | X_i).
    Every label Y_i is drawn ONCE AND ONLY ONCE from Bernoulli(p_i^*) with ZERO post-generation overwrite!
    """
    Xb = np.column_stack([np.ones(len(X_mat)), X_mat])
    logit = Xb @ w_true
    if X_mat.shape[1] >= 6:
        # Structural regime interaction on extreme auxiliary coordinates |x_4| > 1.8 or |x_5| > 1.8
        mask_g4 = np.abs(X_mat[:, 4]) > 1.8
        mask_g5 = np.abs(X_mat[:, 5]) > 1.8
        logit = np.where(mask_g4, np.where(X_mat[:, 4] > 0, 1.65, -3.50), logit)
        logit = np.where(mask_g5, np.where(X_mat[:, 5] > 0, 1.65, -3.50), logit)
    if is_shift:
        logit = logit + 0.45
    return np.clip(expit(logit), 1e-4, 0.999)


def generate_synthetic_credit_state(
    seed,
    n_D=80,
    n_O=22,
    n_C=24,
    n_Q=180,
    n_T=240,
    pending_overlap="High",
    risk_info_coupling="Positive",
):
    """
    Generates a controlled credit decision state where ALL labels (y_D, y_O, y_C, y_T)
    are drawn strictly once from Bernoulli(p_true) with ZERO post-sampling overwrite!
    """
    rng = np.random.default_rng(seed)
    d = 6
    w_true = np.array([-0.95, 0.85, 1.75, -2.10, 2.20, 0.0, 0.0], dtype=float)

    X_D = rng.normal(loc=-0.40, scale=0.55, size=(n_D, d))
    X_D[:, 1] *= 0.40
    X_D[:, 2] *= 0.36
    X_D[:, 3] *= 0.36
    X_D[:, 4] *= 0.08
    X_D[:, 5] *= 0.012
    p_D_true = true_dgp_prob(X_D, w_true)
    y_D = rng.binomial(1, p_D_true)

    if n_C < 22:
        n_g1 = max(n_C - 7, 3)
        n_g2, n_g3a, n_g3b, n_g4 = 2, 2, 1, 1
        n_g5 = max(n_C - n_g1 - n_g2 - n_g3a - n_g3b - n_g4, 1)
    else:
        n_g1, n_g2, n_g3a, n_g3b, n_g4 = 9, 5, 3, 2, 2
        n_g5 = n_C - n_g1 - n_g2 - n_g3a - n_g3b - n_g4

    X_g1 = rng.normal(loc=-0.55, scale=0.30, size=(n_g1, d))
    X_g1[:, 1:6] *= 0.02

    X_g2 = rng.normal(loc=0.0, scale=0.02, size=(n_g2, d))
    X_g2[:, 0] = -0.35
    X_g2[:, 1] = np.array([2.05, -2.05, 2.00, -2.00, 1.98])[:n_g2]

    X_g3a = rng.normal(loc=0.0, scale=0.02, size=(n_g3a, d))
    X_g3a[:, 0] = -0.25
    X_g3a[:, 2] = np.array([-1.65, -1.62, -1.60])[:n_g3a]

    X_g3b = rng.normal(loc=0.0, scale=0.02, size=(n_g3b, d))
    X_g3b[:, 0] = -0.15
    X_g3b[:, 3] = np.array([1.55, -1.55])[:n_g3b]

    X_g4 = rng.normal(loc=0.0, scale=0.02, size=(n_g4, d))
    X_g4[:, 0] = -0.45
    X_g4[:, 4] = np.array([2.45, -2.45])[:n_g4]

    X_g5 = rng.normal(loc=0.0, scale=0.02, size=(n_g5, d))
    X_g5[:, 0] = -1.15
    X_g5[:, 5] = np.array([3.15, -3.15, 3.10])[:n_g5]

    if risk_info_coupling == "Positive":
        ead_g1 = rng.uniform(0.50, 0.64, size=n_g1)
        ead_g2 = rng.uniform(0.60, 0.66, size=n_g2)
        ead_g3a = rng.uniform(0.68, 0.74, size=n_g3a)
        ead_g3b = rng.uniform(0.76, 0.84, size=n_g3b)
        ead_g4 = rng.uniform(1.12, 1.22, size=n_g4)
        ead_g5 = rng.uniform(0.72, 0.80, size=n_g5)
    elif risk_info_coupling == "Neutral":
        ead_g1 = rng.uniform(0.58, 0.72, size=n_g1)
        ead_g2 = rng.uniform(0.64, 0.72, size=n_g2)
        ead_g3a = rng.uniform(0.66, 0.74, size=n_g3a)
        ead_g3b = rng.uniform(0.72, 0.80, size=n_g3b)
        ead_g4 = rng.uniform(1.02, 1.14, size=n_g4)
        ead_g5 = rng.uniform(0.70, 0.78, size=n_g5)
    else:  # Negative coupling
        ead_g1 = rng.uniform(0.74, 0.88, size=n_g1)
        ead_g2 = rng.uniform(0.58, 0.66, size=n_g2)
        ead_g3a = rng.uniform(0.54, 0.62, size=n_g3a)
        ead_g3b = rng.uniform(0.58, 0.66, size=n_g3b)
        ead_g4 = rng.uniform(0.95, 1.08, size=n_g4)
        ead_g5 = rng.uniform(0.66, 0.74, size=n_g5)

    X_C = np.vstack([X_g1, X_g2, X_g3a, X_g3b, X_g4, X_g5])
    ead_C = np.concatenate([ead_g1, ead_g2, ead_g3a, ead_g3b, ead_g4, ead_g5])
    perm = rng.permutation(n_C)
    X_C = X_C[perm]
    ead_C = ead_C[perm]

    # STRICT AUDIT GATE A: p_C_true is the exact conditional probability and y_C ~ Bernoulli(p_C_true) once!
    p_C_true = true_dgp_prob(X_C, w_true)
    y_C = rng.binomial(1, p_C_true)

    if pending_overlap in ("High", "Substitution"):
        X_O = rng.normal(loc=0.0, scale=0.05, size=(n_O, d))
        X_O[:, 0] = -0.25
        X_O[:, 1] = rng.choice([-1.95, 1.95], size=n_O)
    elif pending_overlap in ("Medium", "Unlocking"):
        X_O = rng.normal(loc=-0.35, scale=0.18, size=(n_O, d))
        X_O[: n_O // 2, 1] = rng.choice([-1.85, 1.85], size=n_O // 2)
        X_O[:, 4:6] *= 0.03
    else:  # Low / Irrelevant overlap
        X_O = rng.normal(loc=-0.50, scale=0.25, size=(n_O, d))
        X_O[:, 1:6] *= 0.03

    p_O_true = true_dgp_prob(X_O, w_true)
    y_O = rng.binomial(1, p_O_true)

    X_Q = np.vstack([
        rng.normal(loc=[-0.35, 0.0, 0.0, 0.0, 0.0, 0.0], scale=[0.45, 1.25, 0.5, 0.5, 0.55, 0.55], size=(n_Q // 3, d)),
        rng.normal(loc=[-0.15, 1.35, 0.0, 0.0, 0.0, 0.0], scale=[0.45, 1.25, 0.5, 0.5, 0.55, 0.55], size=(n_Q // 3, d)),
        rng.normal(loc=[0.0, 0.0, -1.2, 1.2, 0.0, 0.0], scale=[0.45, 0.4, 0.90, 0.95, 0.55, 0.55], size=(n_Q - 2 * (n_Q // 3), d)),
    ])
    X_T = np.vstack([
        rng.normal(loc=[-0.35, 0.0, 0.0, 0.0, 0.0, 0.0], scale=[0.45, 1.25, 0.5, 0.5, 0.55, 0.55], size=(n_T // 3, d)),
        rng.normal(loc=[-0.15, 1.35, 0.0, 0.0, 0.0, 0.0], scale=[0.45, 1.25, 0.5, 0.5, 0.55, 0.55], size=(n_T // 3, d)),
        rng.normal(loc=[0.0, 0.0, -1.2, 1.2, 0.0, 0.0], scale=[0.45, 0.4, 0.90, 0.95, 0.55, 0.55], size=(n_T - 2 * (n_T // 3), d)),
    ])
    X_Q[:, 4:6] = np.clip(X_Q[:, 4:6], -1.75, 1.75)
    X_T[:, 4:6] = np.clip(X_T[:, 4:6], -1.75, 1.75)
    p_T_true = true_dgp_prob(X_T, w_true)
    y_T = rng.binomial(1, p_T_true)

    return {
        "X_D": X_D, "y_D": y_D,
        "X_O": X_O, "y_O": y_O,
        "X_C": X_C, "y_C": y_C, "ead_C": ead_C, "p_C_true": p_C_true,
        "X_Q": X_Q,
        "X_T": X_T, "y_T": y_T, "p_T_true": p_T_true,
    }


def run_p0_exact_benchmark():
    rows = []
    alpha = 0.10
    for idx in range(40):
        seed = 500 + idx
        st = generate_synthetic_credit_state(
            seed=seed, n_D=65, n_O=14, n_C=12, n_Q=120, n_T=160,
            pending_overlap="High", risk_info_coupling="Positive"
        )
        learner = UnifiedLogisticLearner(l2=0.45).fit(st["X_D"], st["y_D"])
        q_C, _ = compute_calibrated_upper_risk(
            learner, st["X_D"], st["y_D"], st["X_C"],
            delta_r=0.10, risk_regime="Calibrated", rng=np.random.default_rng(seed + 10)
        )
        q_C[0] = 0.015
        st["ead_C"][0] = 1.65
        q_C[1] = 0.018
        st["ead_C"][1] = 1.55
        B_t, b_hash = compute_exogenous_budget(q_C[2:], st["ead_C"][2:], alpha=alpha, target_k=5)

        S_exact, info_exact = solve_safe_minrisk_exact_or_bounded(q_C, st["ead_C"], B_t, alpha=alpha, exact_max_n=16)
        comp_exact = bernstein_components(q_C[S_exact], st["ead_C"][S_exact], alpha=alpha)
        S_strat, info_strat = solve_safe_minrisk_exact_or_bounded(q_C, st["ead_C"], B_t, alpha=alpha, exact_max_n=0)

        ord_naive = np.argsort(q_C * st["ead_C"])
        S_naive = []
        for i_cand in ord_naive:
            trial = S_naive + [int(i_cand)]
            if bernstein_envelope(q_C[trial], st["ead_C"][trial], alpha=alpha) <= B_t + 1e-12:
                S_naive.append(int(i_cand))
        S_naive = np.array(S_naive, dtype=int)

        S_prism, _, info_prism = select_acquisition_policy(
            "DC-PRISM", st["X_D"], st["y_D"], st["X_O"], st["X_C"], st["ead_C"], st["X_Q"], B_t,
            learner=learner, alpha=alpha, q_C_precomputed=q_C, exact_max_n=16, rng=np.random.default_rng(seed + 20)
        )
        comp_prism = bernstein_components(q_C[S_prism], st["ead_C"][S_prism], alpha=alpha)

        rows.append({
            "state_id": idx + 1,
            "seed": seed,
            "n_C": 12,
            "B_t": B_t,
            "budget_hash": b_hash,
            "k_safe_exact": int(info_exact["k_safe_lower"]),
            "k_safe_stratified": int(info_strat["k_safe_lower"]),
            "k_safe_upper_bound": int(info_strat["k_safe_upper"]),
            "k_safe_naive_greedy": int(len(S_naive)),
            "exact_solver_match": int(info_exact["k_safe_lower"] == len(S_exact)),
            "stratified_matches_exact": int(info_strat["k_safe_lower"] == info_exact["k_safe_lower"]),
            "naive_matches_exact": int(len(S_naive) == info_exact["k_safe_lower"]),
            "exact_mu_U": comp_exact["mu_U"],
            "exact_V_U": comp_exact["V_U"],
            "exact_M": comp_exact["M"],
            "exact_ell": comp_exact["ell"],
            "exact_risk_envelope": comp_exact["risk_envelope"],
            "prism_k_selected": int(len(S_prism)),
            "prism_eps_star": float(info_prism["eps_star"]),
            "prism_risk_envelope": comp_prism["risk_envelope"],
            "prism_envelope_feasible": int(comp_prism["risk_envelope"] <= B_t + 1e-9),
            "pointwise_q_coverage": float(np.mean(q_C >= st["p_C_true"])),
            "selected_simultaneous_coverage": int(np.all(q_C[S_prism] >= st["p_C_true"][S_prism])) if len(S_prism) > 0 else 1,
        })

    df_p0 = pd.DataFrame(rows)
    out_path = os.path.join(PKG_ROOT, "p0_exact", "p0_exact_vs_greedy_40states.csv")
    df_p0.to_csv(out_path, index=False)
    df_p0.to_csv(os.path.join(PKG_ROOT, "results", "audit", "exact_safe_solver_audit.csv"), index=False)
    print(f"[P0] Saved {out_path}: Exact Match={df_p0['exact_solver_match'].mean()*100:.1f}%, "
          f"Stratified Match={df_p0['stratified_matches_exact'].mean()*100:.1f}%, "
          f"Naive Greedy Match={df_p0['naive_matches_exact'].mean()*100:.1f}%")
    return df_p0


def evaluate_policy_on_state(policy_name, st, learner, q_C, B_t, w_worlds, alpha=0.10, lambda_K=0.22, risk_regime="Calibrated", seed=1000):
    S_sel, M_K, info = select_acquisition_policy(
        policy_name,
        st["X_D"], st["y_D"], st["X_O"], st["X_C"], st["ead_C"], st["X_Q"], B_t,
        learner=learner,
        alpha=alpha,
        lambda_K=lambda_K,
        risk_regime=risk_regime,
        q_C_precomputed=q_C,
        w_worlds_precomputed=w_worlds,
        exact_max_n=16,
        rng=np.random.default_rng(seed),
    )

    if policy_name == "OrdinaryOnly" or len(S_sel) == 0:
        X_post = np.vstack([st["X_D"], st["X_O"]])
        y_post = np.concatenate([st["y_D"], st["y_O"]])
    else:
        X_post = np.vstack([st["X_D"], st["X_O"], st["X_C"][S_sel]])
        y_post = np.concatenate([st["y_D"], st["y_O"], st["y_C"][S_sel]])

    post_learner = UnifiedLogisticLearner(l2=0.45).fit(X_post, y_post, w_init=learner.w)
    p_T_pred = post_learner.predict_proba(st["X_T"])
    brier_post = brier_score(st["y_T"], p_T_pred)

    # Section 2 & 3: True-p Coverage and Empirical Safety Theorem Layer
    pool_sim_cov = int(np.all(q_C >= st["p_C_true"]))
    if len(S_sel) > 0:
        comp = bernstein_components(q_C[S_sel], st["ead_C"][S_sel], alpha=alpha)
        p_sel_true = st["p_C_true"][S_sel]
        ead_sel = st["ead_C"][S_sel]
        mu_true = float(np.sum(p_sel_true * ead_sel))
        v_true = float(np.sum(p_sel_true * (1.0 - p_sel_true) * (ead_sel ** 2)))
        realized_loss = float(np.sum(st["y_C"][S_sel] * ead_sel))
        point_cov = float(np.mean(q_C[S_sel] >= p_sel_true))
        sim_cov = int(np.all(q_C[S_sel] >= p_sel_true))
        env_ge_mu_true = int(comp["risk_envelope"] >= mu_true - 1e-12)
        viol_sev = max(0.0, float(realized_loss - B_t))
    else:
        comp = {"mu_U": 0.0, "V_U": 0.0, "M": 0.0, "ell": float(np.log(1.0 / alpha)), "risk_envelope": 0.0}
        mu_true = 0.0
        v_true = 0.0
        realized_loss = 0.0
        point_cov = 1.0
        sim_cov = 1
        env_ge_mu_true = 1
        viol_sev = 0.0

    return {
        "policy": policy_name,
        "brier": brier_post,
        "k_selected": int(len(S_sel)),
        "k_safe": int(info["k_safe"]),
        "eps_star": float(info["eps_star"]),
        "gate_triggered": int(bool(info.get("gate_triggered", False))),
        "surplus_score": float(info.get("surplus_score", 0.0)),
        "mu_U": comp["mu_U"],
        "V_U": comp["V_U"],
        "M": comp["M"],
        "ell": comp["ell"],
        "risk_envelope": comp["risk_envelope"],
        "B_t": float(B_t),
        "envelope_violation": int(comp["risk_envelope"] > B_t + 1e-9),
        "mu_true": mu_true,
        "V_true": v_true,
        "realized_loss": realized_loss,
        "true_expected_loss": mu_true,
        "envelope_ge_mu_true": env_ge_mu_true,
        "realized_exceeds_Bt": int(realized_loss > B_t + 1e-9),
        "realized_exceeds_Ralpha": int(realized_loss > comp["risk_envelope"] + 1e-9),
        "violation_severity": viol_sev,
        "pointwise_coverage": point_cov,
        "simultaneous_coverage": sim_cov,
        "candidate_pool_simultaneous_coverage": pool_sim_cov,
        "solver_status": info.get("solver_status", "EXACT_OR_BOUNDED"),
        "selected_indices": S_sel,
        "M_K_indices": M_K,
    }


def run_p1_pilot_mechanism(fast_mode=False):
    overlaps = ["Low", "Medium", "High"]
    couplings = ["Negative", "Neutral", "Positive"]
    regimes = ["Calibrated", "Optimistic_Bias", "Conservative_Bias"]
    pilot_seeds = list(range(1001, 1017 if fast_mode else 1033))
    policies_p1 = ["DC-PRISM", "Safe-MinRisk", "Safe-PA-NoGate", "D-Only-Safe-PA", "Orig-PA"]

    rows = []
    alpha = 0.10
    for seed in pilot_seeds:
        for ov in overlaps:
            for cp in couplings:
                st = generate_synthetic_credit_state(
                    seed=seed * 31 + overlaps.index(ov) * 7 + couplings.index(cp),
                    n_D=80, n_O=22, n_C=24, n_Q=160, n_T=200,
                    pending_overlap=ov, risk_info_coupling=cp,
                )
                learner = UnifiedLogisticLearner(l2=0.45).fit(st["X_D"], st["y_D"])
                w_worlds = sample_common_parameter_worlds(learner, n_worlds=20, rng=np.random.default_rng(seed + 500))

                q_cal_ref, _ = compute_calibrated_upper_risk(
                    learner, st["X_D"], st["y_D"], st["X_C"],
                    delta_r=0.10, risk_regime="Calibrated", rng=np.random.default_rng(seed + 100)
                )
                B_t, b_hash = compute_exogenous_budget(q_cal_ref, st["ead_C"], alpha=alpha, target_k=6)

                for rg in regimes:
                    q_C_rg, _ = compute_calibrated_upper_risk(
                        learner, st["X_D"], st["y_D"], st["X_C"],
                        delta_r=0.10, risk_regime=rg, rng=np.random.default_rng(seed + 200)
                    )
                    for pol in policies_p1:
                        res = evaluate_policy_on_state(
                            pol, st, learner, q_C_rg, B_t, w_worlds,
                            alpha=alpha, lambda_K=0.22, risk_regime=rg, seed=seed + 300
                        )
                        res.pop("selected_indices", None)
                        res.pop("M_K_indices", None)
                        res.update({
                            "world_seed": seed,
                            "pending_overlap": ov,
                            "risk_info_coupling": cp,
                            "risk_regime": rg,
                            "cell_id": f"{ov}_{cp}_{rg}",
                            "budget_hash": b_hash,
                        })
                        rows.append(res)

    df_p1 = pd.DataFrame(rows)
    p1_path = os.path.join(PKG_ROOT, "p1_mechanism", "p1_pilot_27cells.csv")
    df_p1.to_csv(p1_path, index=False)

    cell_sum = (
        df_p1.groupby(["pending_overlap", "risk_info_coupling", "risk_regime", "policy"], as_index=False)
        .agg(
            mean_brier=("brier", "mean"),
            std_brier=("brier", "std"),
            mean_k_selected=("k_selected", "mean"),
            mean_eps_star=("eps_star", "mean"),
            gate_rate=("gate_triggered", "mean"),
            envelope_violation_rate=("envelope_violation", "mean"),
            realized_exceed_rate=("realized_exceeds_Bt", "mean"),
            mean_pointwise_coverage=("pointwise_coverage", "mean"),
            simultaneous_coverage_rate=("simultaneous_coverage", "mean"),
            pool_simultaneous_coverage_rate=("candidate_pool_simultaneous_coverage", "mean"),
        )
    )
    cell_sum.to_csv(os.path.join(PKG_ROOT, "p1_mechanism", "p1_cell_summary.csv"), index=False)

    # Save q_coverage_by_regime.csv (Section 2 & 18)
    cov_by_rg = (
        df_p1[df_p1["policy"] == "DC-PRISM"]
        .groupby("risk_regime", as_index=False)
        .agg(
            pointwise_coverage=("pointwise_coverage", "mean"),
            selected_set_simultaneous_coverage=("simultaneous_coverage", "mean"),
            candidate_pool_simultaneous_coverage=("candidate_pool_simultaneous_coverage", "mean"),
            envelope_ge_mu_true_rate=("envelope_ge_mu_true", "mean"),
            realized_violation_rate=("realized_exceeds_Bt", "mean"),
            mean_violation_severity=("violation_severity", "mean"),
        )
    )
    cov_by_rg.to_csv(os.path.join(PKG_ROOT, "results", "audit", "q_coverage_by_regime.csv"), index=False)

    # Factor Attribution ANOVA
    piv = df_p1.pivot_table(
        index=["world_seed", "pending_overlap", "risk_info_coupling", "risk_regime"],
        columns="policy",
        values=["brier", "realized_exceeds_Bt", "eps_star", "pointwise_coverage"],
    )
    df_attr = pd.DataFrame({
        "brier_gain_vs_minrisk": piv["brier"]["Safe-MinRisk"] - piv["brier"]["DC-PRISM"],
        "brier_gain_vs_donly": piv["brier"]["D-Only-Safe-PA"] - piv["brier"]["DC-PRISM"],
        "prism_exceed_rate": piv["realized_exceeds_Bt"]["DC-PRISM"],
        "prism_pointwise_cov": piv["pointwise_coverage"]["DC-PRISM"],
    }).reset_index()

    anova_rows = []
    for factor in ["pending_overlap", "risk_info_coupling", "risk_regime"]:
        grp = df_attr.groupby(factor, as_index=False).agg(
            mean_gain_vs_minrisk=("brier_gain_vs_minrisk", "mean"),
            mean_gain_vs_donly=("brier_gain_vs_donly", "mean"),
            mean_prism_exceed_rate=("prism_exceed_rate", "mean"),
            mean_prism_pointwise_cov=("prism_pointwise_cov", "mean"),
        )
        for _, r in grp.iterrows():
            anova_rows.append({
                "factor": factor,
                "level": r[factor],
                "mean_brier_gain_vs_SafeMinRisk": float(r["mean_gain_vs_minrisk"]),
                "mean_brier_gain_vs_DOnly": float(r["mean_gain_vs_donly"]),
                "prism_realized_exceed_rate": float(r["mean_prism_exceed_rate"]),
                "prism_pointwise_coverage": float(r["mean_prism_pointwise_cov"]),
            })
    pd.DataFrame(anova_rows).to_csv(
        os.path.join(PKG_ROOT, "p1_mechanism", "p1_factor_Attribution_ANOVA.csv"), index=False
    )

    # Lambda_K selection curve on pilot seeds
    lam_rows = []
    for lam in [0.00, 0.10, 0.22, 0.40]:
        for seed in pilot_seeds:
            st = generate_synthetic_credit_state(
                seed=seed * 31 + 2 * 7 + 2,
                n_D=80, n_O=22, n_C=24, n_Q=160, n_T=200,
                pending_overlap="High", risk_info_coupling="Positive",
            )
            learner = UnifiedLogisticLearner(l2=0.45).fit(st["X_D"], st["y_D"])
            w_worlds = sample_common_parameter_worlds(learner, n_worlds=20, rng=np.random.default_rng(seed + 500))
            q_C, _ = compute_calibrated_upper_risk(
                learner, st["X_D"], st["y_D"], st["X_C"],
                delta_r=0.10, risk_regime="Calibrated", rng=np.random.default_rng(seed + 100)
            )
            B_t, _ = compute_exogenous_budget(q_C, st["ead_C"], alpha=alpha, target_k=6)
            res = evaluate_policy_on_state(
                "DC-PRISM", st, learner, q_C, B_t, w_worlds,
                alpha=alpha, lambda_K=lam, risk_regime="Calibrated", seed=seed + 300
            )
            lam_rows.append({
                "lambda_K": lam,
                "world_seed": seed,
                "brier": res["brier"],
                "k_selected": res["k_selected"],
                "eps_star": res["eps_star"],
                "gate_triggered": res["gate_triggered"],
            })
    df_lam = (
        pd.DataFrame(lam_rows)
        .groupby("lambda_K", as_index=False)
        .agg(
            mean_brier=("brier", "mean"),
            std_brier=("brier", "std"),
            mean_k_selected=("k_selected", "mean"),
            mean_eps_star=("eps_star", "mean"),
            gate_rate=("gate_triggered", "mean"),
        )
    )
    df_lam.to_csv(os.path.join(PKG_ROOT, "p1_mechanism", "p1_lambdaK_selection_curve.csv"), index=False)
    print(f"[P1] Saved pilot 27-cell cube ({len(df_p1)} rows) and q_coverage_by_regime.csv.")
    return df_p1, df_lam


def run_p2_confirmation_and_cross_mechanism(fast_mode=False):
    conf_seeds = list(range(2001, 2049 if fast_mode else 2129))
    policies_p2 = [
        "DC-PRISM",
        "Safe-PA-NoGate",
        "Fixed-Eps-Safe-PA",
        "D-Only-Safe-PA",
        "PRISM-ValueUCB",
        "Safe-MinRisk",
        "BAIT-Cost-Safe",
        "Orig-MinRisk",
        "Orig-PA",
        "OrdinaryOnly",
    ]
    qcj_policies = ["DC-PRISM", "Safe-PA-NoGate", "Fixed-Eps-Safe-PA", "D-Only-Safe-PA", "PRISM-ValueUCB"]
    alpha = 0.10
    rows = []
    qcj_rows = []
    cand_audit_rows = []

    for seed in conf_seeds:
        st = generate_synthetic_credit_state(
            seed=seed, n_D=80, n_O=22, n_C=24, n_Q=180, n_T=240,
            pending_overlap="High", risk_info_coupling="Positive"
        )
        learner = UnifiedLogisticLearner(l2=0.45).fit(st["X_D"], st["y_D"])
        w_worlds = sample_common_parameter_worlds(learner, n_worlds=20, rng=np.random.default_rng(seed + 700))
        q_C, _ = compute_calibrated_upper_risk(
            learner, st["X_D"], st["y_D"], st["X_C"],
            delta_r=0.10, risk_regime="Calibrated", rng=np.random.default_rng(seed + 100)
        )
        B_t, b_hash = compute_exogenous_budget(q_C, st["ead_C"], alpha=alpha, target_k=6)

        X_base = np.vstack([st["X_D"], st["X_O"]])
        y_base = np.concatenate([st["y_D"], st["y_O"]])

        for pol in policies_p2:
            res = evaluate_policy_on_state(
                pol, st, learner, q_C, B_t, w_worlds,
                alpha=alpha, lambda_K=0.22, risk_regime="Calibrated", seed=seed + 400
            )
            S_sel = res.pop("selected_indices")
            M_K = res.pop("M_K_indices")
            res["world_seed"] = seed
            res["budget_hash"] = b_hash
            rows.append(res)

            # Section 1: Save candidate-level oracle probability consistency records for DC-PRISM
            if pol == "DC-PRISM":
                sel_set = set(S_sel.tolist())
                for c_idx in range(len(st["p_C_true"])):
                    cand_audit_rows.append({
                        "world_seed": seed,
                        "candidate_id": f"P2_S{seed}_C{c_idx:02d}",
                        "p_true": float(st["p_C_true"][c_idx]),
                        "q_upper": float(q_C[c_idx]),
                        "y_realized": int(st["y_C"][c_idx]),
                        "ead": float(st["ead_C"][c_idx]),
                        "selected": int(c_idx in sel_set),
                        "q_covers_p_true": int(q_C[c_idx] >= st["p_C_true"][c_idx]),
                    })

            # Section 12 & 13: Compute exact Q_k / C_k / J decomposition on the exact same state
            if pol in qcj_policies:
                decomp = compute_q_c_j_decomposition(
                    S_sel, M_K, q_C, st["ead_C"], B_t,
                    X_base, y_base, st["X_C"], st["y_C"], st["X_T"], st["y_T"],
                    w_init=learner.w, alpha=alpha, l2=0.45
                )
                decomp["world_seed"] = seed
                decomp["policy"] = pol
                qcj_rows.append(decomp)

    df_p2 = pd.DataFrame(rows)
    df_p2.to_csv(os.path.join(PKG_ROOT, "p2_confirmation", "p2_confirmation_worlds.csv"), index=False)

    # Save candidate-level oracle_probability_consistency.csv (Section 1 & 18)
    df_cand_audit = pd.DataFrame(cand_audit_rows)
    df_cand_audit.to_csv(os.path.join(PKG_ROOT, "results", "audit", "oracle_probability_consistency.csv"), index=False)

    # Save Q_k / C_k / J decomposition & BRP / HRR summary (Sections 12 & 13)
    df_qcj = pd.DataFrame(qcj_rows)
    df_qcj.to_csv(os.path.join(PKG_ROOT, "p2_confirmation", "p2_q_c_j_decomposition.csv"), index=False)

    delta_brp = 0.001
    brp_rows = []
    for pol in qcj_policies:
        sub = df_qcj[df_qcj["policy"] == pol]
        rel = sub[sub["eps_star"] > 1e-6]
        n_rel = len(rel)
        brp = float(np.mean(rel["J"] > delta_brp)) if n_rel > 0 else 0.0
        hrr = float(np.mean(rel["J"] < -delta_brp)) if n_rel > 0 else 0.0
        brp_rows.append({
            "policy": pol,
            "n_worlds": len(sub),
            "release_frequency": float(n_rel / max(len(sub), 1)),
            "mean_K": float(sub["K"].mean()),
            "mean_k": float(sub["k"].mean()),
            "mean_eps_star": float(sub["eps_star"].mean()),
            "mean_Quantity_Loss_Qk": float(sub["Q_k"].mean()),
            "mean_Composition_Gain_Ck": float(sub["C_k"].mean()),
            "mean_Net_Gain_J": float(sub["J"].mean()),
            "prob_Ck_gt_Qk": float((sub["C_k"] > sub["Q_k"]).mean()),
            "prob_Ck_gt_Qk_when_released": float((rel["C_k"] > rel["Q_k"]).mean()) if n_rel > 0 else 0.0,
            "BRP_Beneficial_Release_Precision": brp,
            "HRR_Harmful_Release_Rate": hrr,
            "mean_risk_MK": float(sub["risk_M_K"].mean()),
            "mean_risk_Mk": float(sub["risk_M_k"].mean()),
            "mean_risk_S": float(sub["risk_S"].mean()),
        })
    df_brp = pd.DataFrame(brp_rows)
    df_brp.to_csv(os.path.join(PKG_ROOT, "p2_confirmation", "p2_brp_hrr_summary.csv"), index=False)

    pol_sum = (
        df_p2.groupby("policy", as_index=False)
        .agg(
            n_worlds=("world_seed", "count"),
            mean_brier=("brier", "mean"),
            std_brier=("brier", "std"),
            se_brier=("brier", lambda x: np.std(x, ddof=1) / np.sqrt(len(x))),
            mean_k_selected=("k_selected", "mean"),
            mean_k_safe=("k_safe", "mean"),
            mean_eps_star=("eps_star", "mean"),
            gate_trigger_rate=("gate_triggered", "mean"),
            mean_risk_envelope=("risk_envelope", "mean"),
            envelope_violation_rate=("envelope_violation", "mean"),
            mean_mu_true=("mu_true", "mean"),
            mean_V_true=("V_true", "mean"),
            envelope_ge_mu_true_rate=("envelope_ge_mu_true", "mean"),
            mean_realized_loss=("realized_loss", "mean"),
            realized_exceed_Bt_rate=("realized_exceeds_Bt", "mean"),
            mean_violation_severity=("violation_severity", "mean"),
            pointwise_coverage=("pointwise_coverage", "mean"),
            simultaneous_coverage=("simultaneous_coverage", "mean"),
            pool_simultaneous_coverage=("candidate_pool_simultaneous_coverage", "mean"),
        )
    )
    pol_sum.to_csv(os.path.join(PKG_ROOT, "p2_confirmation", "p2_policy_summary.csv"), index=False)

    # Nearest-Neighbor Ablation Family A1..A5 + Holm-Bonferroni tests
    piv = df_p2.pivot(index="world_seed", columns="policy", values="brier")
    prism_b = piv["DC-PRISM"].values

    ablation_specs = [
        ("A1", "Safe-PA-NoGate", "Removes quantity penalty & LCB gate (pure V(S) maximization)"),
        ("A2", "Fixed-Eps-Safe-PA", "Fixes allowance eps=0.25 without state-adaptive gating"),
        ("A3", "D-Only-Safe-PA", "Ignores pending pool O_t in Fisher geometry & target weighting"),
        ("A4", "PRISM-ValueUCB", "Replaces conservative LCB_beta with optimistic UCB_{1-beta} surplus"),
        ("A5", "Safe-MinRisk", "Pure safe quantity anchor M_K (eps=0, no epistemic exploration)"),
    ]

    stat_rows = []
    for ab_id, comp_pol, desc in ablation_specs:
        comp_b = piv[comp_pol].values
        diff = comp_b - prism_b
        mean_diff = float(np.mean(diff))
        sd_diff = float(np.std(diff, ddof=1))
        se_diff = sd_diff / np.sqrt(len(diff))
        t_stat, p_t = ttest_rel(comp_b, prism_b)
        w_stat, p_w = wilcoxon(diff, alternative="greater")
        stat_rows.append({
            "ablation_id": ab_id,
            "baseline_policy": comp_pol,
            "mechanism_isolated": desc,
            "n_independent_worlds": len(diff),
            "prism_mean_brier": float(np.mean(prism_b)),
            "baseline_mean_brier": float(np.mean(comp_b)),
            "paired_delta_brier": mean_diff,
            "ci95_low": mean_diff - 1.96 * se_diff,
            "ci95_high": mean_diff + 1.96 * se_diff,
            "cohens_dz": mean_diff / max(sd_diff, 1e-12),
            "win_rate": float(np.mean(diff > 1e-9)),
            "tie_rate": float(np.mean(np.abs(diff) <= 1e-9)),
            "t_stat": float(t_stat),
            "p_value_ttest": float(p_t),
            "wilcoxon_stat": float(w_stat),
            "p_value_raw": float(p_w),
        })

    df_stats = pd.DataFrame(stat_rows)
    order = np.argsort(df_stats["p_value_raw"].values)
    m_hyp = len(df_stats)
    p_holm = np.zeros(m_hyp)
    running_max = 0.0
    for rank_idx, orig_idx in enumerate(order):
        adj = df_stats.loc[orig_idx, "p_value_raw"] * (m_hyp - rank_idx)
        running_max = max(running_max, adj)
        p_holm[orig_idx] = min(running_max, 1.0)
    df_stats["p_value_holm_bonferroni"] = p_holm
    df_stats["significant_at_0_05"] = df_stats["p_value_holm_bonferroni"] < 0.05
    df_stats.to_csv(os.path.join(PKG_ROOT, "ablations", "ablation_table_A1_to_A5.csv"), index=False)

    extra_pols = ["BAIT-Cost-Safe", "Orig-MinRisk", "Orig-PA", "OrdinaryOnly"]
    all_stat_rows = list(df_stats.to_dict("records"))
    for idx_e, comp_pol in enumerate(extra_pols, start=1):
        comp_b = piv[comp_pol].values
        diff = comp_b - prism_b
        mean_diff = float(np.mean(diff))
        sd_diff = float(np.std(diff, ddof=1))
        se_diff = sd_diff / np.sqrt(len(diff))
        t_stat, p_t = ttest_rel(comp_b, prism_b)
        w_stat, p_w = wilcoxon(diff, alternative="greater")
        all_stat_rows.append({
            "ablation_id": f"EXT-{idx_e}",
            "baseline_policy": comp_pol,
            "mechanism_isolated": "External/Legacy baseline comparison",
            "n_independent_worlds": len(diff),
            "prism_mean_brier": float(np.mean(prism_b)),
            "baseline_mean_brier": float(np.mean(comp_b)),
            "paired_delta_brier": mean_diff,
            "ci95_low": mean_diff - 1.96 * se_diff,
            "ci95_high": mean_diff + 1.96 * se_diff,
            "cohens_dz": mean_diff / max(sd_diff, 1e-12),
            "win_rate": float(np.mean(diff > 1e-9)),
            "tie_rate": float(np.mean(np.abs(diff) <= 1e-9)),
            "t_stat": float(t_stat),
            "p_value_ttest": float(p_t),
            "wilcoxon_stat": float(w_stat),
            "p_value_raw": float(p_w),
            "p_value_holm_bonferroni": float(min(p_w * 9.0, 1.0)),
            "significant_at_0_05": bool(p_w * 9.0 < 0.05),
        })
    pd.DataFrame(all_stat_rows).to_csv(
        os.path.join(PKG_ROOT, "statistics", "holm_bonferroni_tests.csv"), index=False
    )

    # Section 14: Independent Cross-Mechanism Confirmation (seeds 2501..2554, 54 fresh worlds)
    mech_modes = ["Substitution", "Unlocking", "Irrelevant"]
    align_modes = ["Positive", "Neutral", "Negative"]
    cross_rows = []
    seed_cursor = 2501
    n_per_cell = 4 if fast_mode else 6
    for m_mode in mech_modes:
        for a_mode in align_modes:
            for _ in range(n_per_cell):
                s_cm = seed_cursor
                seed_cursor += 1
                st_cm = generate_synthetic_credit_state(
                    seed=s_cm, n_D=80, n_O=22, n_C=24, n_Q=180, n_T=240,
                    pending_overlap=m_mode, risk_info_coupling=a_mode
                )
                l_cm = UnifiedLogisticLearner(l2=0.45).fit(st_cm["X_D"], st_cm["y_D"])
                w_cm = sample_common_parameter_worlds(l_cm, n_worlds=20, rng=np.random.default_rng(s_cm + 700))
                q_cm, _ = compute_calibrated_upper_risk(
                    l_cm, st_cm["X_D"], st_cm["y_D"], st_cm["X_C"],
                    delta_r=0.10, risk_regime="Calibrated", rng=np.random.default_rng(s_cm + 100)
                )
                B_cm, _ = compute_exogenous_budget(q_cm, st_cm["ead_C"], alpha=alpha, target_k=6)
                for pol in ["DC-PRISM", "Safe-MinRisk", "D-Only-Safe-PA", "Fixed-Eps-Safe-PA"]:
                    r_cm = evaluate_policy_on_state(
                        pol, st_cm, l_cm, q_cm, B_cm, w_cm,
                        alpha=alpha, lambda_K=0.22, risk_regime="Calibrated", seed=s_cm + 400
                    )
                    cross_rows.append({
                        "world_seed": s_cm,
                        "pending_mechanism": m_mode,
                        "coupling_alignment": a_mode,
                        "policy": pol,
                        "brier": r_cm["brier"],
                        "eps_star": r_cm["eps_star"],
                        "gate_triggered": r_cm["gate_triggered"],
                        "pointwise_coverage": r_cm["pointwise_coverage"],
                        "simultaneous_coverage": r_cm["simultaneous_coverage"],
                    })
    df_cross = (
        pd.DataFrame(cross_rows)
        .groupby(["pending_mechanism", "coupling_alignment", "policy"], as_index=False)
        .agg(
            n_worlds=("world_seed", "count"),
            mean_brier=("brier", "mean"),
            mean_eps_star=("eps_star", "mean"),
            gate_trigger_rate=("gate_triggered", "mean"),
            pointwise_coverage=("pointwise_coverage", "mean"),
            simultaneous_coverage=("simultaneous_coverage", "mean"),
        )
    )
    df_cross.to_csv(os.path.join(PKG_ROOT, "p2_confirmation", "p2_cross_mechanism_confirmation.csv"), index=False)

    print("[P2] Saved confirmation worlds, Q_k/C_k/J decomposition, BRP/HRR summary, and cross-mechanism confirmation:")
    print(df_stats[["ablation_id", "baseline_policy", "paired_delta_brier", "cohens_dz", "p_value_holm_bonferroni"]])
    print("\n[P2 BRP / HRR Summary]:")
    print(df_brp[["policy", "mean_eps_star", "mean_Quantity_Loss_Qk", "mean_Composition_Gain_Ck", "mean_Net_Gain_J", "BRP_Beneficial_Release_Precision", "HRR_Harmful_Release_Rate"]])
    return df_p2, df_stats, df_brp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Run reduced world count for fast smoke reproduction")
    args = parser.parse_args()

    t0 = time.time()
    run_p0_exact_benchmark()
    run_p1_pilot_mechanism(fast_mode=args.fast)
    run_p2_confirmation_and_cross_mechanism(fast_mode=args.fast)
    print(f"Completed P0, P1, P2 in {time.time() - t0:.2f}s.")
