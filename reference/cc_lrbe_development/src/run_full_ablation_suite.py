import os
import zipfile
import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds, minimize
from scipy.special import expit
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Set up directories
BASE_DIR = "/working_dir/c_2154c32685b76347"
INTER_DIR = os.path.join(BASE_DIR, "artifacts/file_generation/ttl=63d/intermediate")
OUT_DIR = os.path.join(BASE_DIR, "artifacts/file_generation/ttl=63d/output")
PKG_STAGE = os.path.join(INTER_DIR, "LexiRiskLabel_CC_LRBE_Upgrade_Package")

for sub in ["", "src", "results", "figures", "paper", "reports"]:
    os.makedirs(os.path.join(PKG_STAGE, sub), exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# ==============================================================================
# 1. Core Simulation & Experiment Execution (10 Seeds x 5 Budgets x 8 Methods)
# ==============================================================================
FAVE_GAMMA = 1.0
FAVE_DELTA = 0.5

def _ridge_inverse(A):
    return np.linalg.pinv(A, rcond=1e-12)

def optimize_milp(cost, score, budget):
    n = len(cost)
    if n == 0 or budget <= 0:
        return np.array([], int), 0
    integ = np.ones(n, int)
    bnd = Bounds(np.zeros(n), np.ones(n))
    con = LinearConstraint(cost.reshape(1, -1), -np.inf, budget)
    r1 = milp(c=-np.ones(n), integrality=integ, bounds=bnd, constraints=con)
    if not r1.success or r1.x is None:
        return np.array([], int), 0
    k = int(np.rint(r1.x).sum())
    if k == 0:
        return np.array([], int), 0
    r2 = milp(
        c=-np.asarray(score, float),
        integrality=integ,
        bounds=bnd,
        constraints=[con, LinearConstraint(np.ones((1, n)), k, k)],
    )
    if not r2.success or r2.x is None:
        return np.flatnonzero(np.rint(r1.x) > 0.5), k
    return np.flatnonzero(np.rint(r2.x) > 0.5), k

def policy_score(name, p, cost):
    if name == "CountOnly":
        return -1e-9 * np.arange(len(p))
    if name == "CountOnly-MinRisk":
        return -cost
    if name == "LeastRisky":
        return -p
    if name == "MostRisky":
        return p
    if name == "Uncertainty":
        return p * (1 - p)
    if name == "RiskTilted-Uncertainty":
        return p * (1 - p) * np.sqrt(np.clip(p, 1e-9, 1))
    raise ValueError(name)

def fave_value_and_leverage(Zv, Zr, Zf, pv, pr, pf, gamma=FAVE_GAMMA, delta=FAVE_DELTA):
    if len(Zr) == 0:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)
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
    logit_var = np.einsum("ij,jk,ik->i", R, Hinv, R)
    return fave_val, np.sqrt(np.maximum(logit_var, 0.0))

class LogisticClassifier:
    def __init__(self, l2=1.0):
        self.l2 = l2
        self.w = None
    def fit(self, X, y, sample_weight=None):
        n, d = X.shape
        Xb = np.column_stack([np.ones(n), X])
        sw = np.ones(n) if sample_weight is None else np.asarray(sample_weight, float)
        def loss_and_grad(w):
            p = expit(Xb @ w)
            nll = -np.sum(sw * (y * np.log(np.clip(p, 1e-12, 1.0)) + (1 - y) * np.log(np.clip(1 - p, 1e-12, 1.0))))
            reg = 0.5 * self.l2 * np.sum(w[1:] ** 2)
            grad = Xb.T @ (sw * (p - y))
            grad[1:] += self.l2 * w[1:]
            return nll + reg, grad
        self.w = minimize(loss_and_grad, np.zeros(d + 1), jac=True, method="BFGS").x
        return self
    def predict_proba(self, X):
        p = np.clip(expit(np.column_stack([np.ones(len(X)), X]) @ self.w), 1e-9, 1 - 1e-9)
        return np.column_stack([1 - p, p])

def compute_metrics(y_true, p_pred, threshold=0.25):
    y_hat = (p_pred >= threshold).astype(int)
    tp = np.sum((y_true == 1) & (y_hat == 1))
    fn = np.sum((y_true == 1) & (y_hat == 0))
    recall = tp / max(tp + fn, 1)
    brier = np.mean((y_true - p_pred) ** 2)
    pos, neg = p_pred[y_true == 1], p_pred[y_true == 0]
    auc = np.mean(pos[:, None] > neg[None, :]) + 0.5 * np.mean(pos[:, None] == neg[None, :])
    return float(recall), float(auc), float(brier)

def generate_selective_credit_dataset(seed, n_init=600, n_rounds=5, n_pool_per_round=250, n_test=800):
    rng = np.random.default_rng(seed)
    n_total = n_init + n_rounds * n_pool_per_round + n_test
    X = rng.normal(0, 1, size=(n_total, 8))
    ead = np.clip(10.0 * np.exp(0.35 * X[:, 0] + rng.normal(0, 0.25, size=n_total)), 3.0, 45.0)
    w_true = np.array([0.45, 0.70, 0.55, -0.40, 0.35, -0.30, 0.25, 0.20])
    linear_logit = -1.35 + X @ w_true
    tail_shift = 0.55 * np.maximum(X[:, 1] + 0.5 * X[:, 2] - 0.2, 0.0)
    p_true = expit(linear_logit + tail_shift)
    y = rng.binomial(1, p_true)
    hist_score = linear_logit + rng.normal(0, 0.35, size=n_total)
    init_approved_mask = hist_score[:n_init] < np.percentile(hist_score[:n_init], 55)
    return {
        "X_init": X[:n_init], "y_init": y[:n_init], "init_vis_mask": init_approved_mask,
        "rounds": [{"X": X[n_init + r*n_pool_per_round : n_init + (r+1)*n_pool_per_round],
                    "y": y[n_init + r*n_pool_per_round : n_init + (r+1)*n_pool_per_round],
                    "ead": ead[n_init + r*n_pool_per_round : n_init + (r+1)*n_pool_per_round]} for r in range(n_rounds)],
        "X_test": X[n_init + n_rounds*n_pool_per_round :], "y_test": y[n_init + n_rounds*n_pool_per_round :],
    }

def optimize_chance_constrained_milp(p_eff, ead, score, budget, z_alpha=1.05):
    n = len(p_eff)
    mean_cost = p_eff * ead
    var_cost = p_eff * (1.0 - p_eff) * (ead ** 2)
    order_by_risk = np.argsort(mean_cost + z_alpha * np.sqrt(var_cost) * 0.35)
    k_safe = 0
    for k_try in range(1, n + 1):
        idx_try = order_by_risk[:k_try]
        if np.sum(mean_cost[idx_try]) + z_alpha * np.sqrt(np.sum(var_cost[idx_try])) <= budget:
            k_safe = k_try
        else:
            break
    if k_safe == 0:
        return np.array([], int), 0
    ref_idx = order_by_risk[:k_safe]
    sigma_k = max(np.sqrt(np.sum(var_cost[ref_idx])), 1e-6)
    eff_unit_cost = mean_cost + (z_alpha / (2.0 * sigma_k)) * var_cost
    eff_budget = budget - 0.5 * z_alpha * sigma_k
    integ, bnd = np.ones(n, int), Bounds(np.zeros(n), np.ones(n))
    con_b = LinearConstraint(eff_unit_cost.reshape(1, -1), -np.inf, eff_budget)
    con_k = LinearConstraint(np.ones((1, n)), k_safe, k_safe)
    r2 = milp(c=-np.asarray(score, float), integrality=integ, bounds=bnd, constraints=[con_b, con_k])
    chosen = np.flatnonzero(np.rint(r2.x) > 0.5) if (r2.success and r2.x is not None) else order_by_risk[:k_safe]
    while len(chosen) > 1 and (np.sum(mean_cost[chosen]) + z_alpha * np.sqrt(np.sum(var_cost[chosen])) > budget):
        k_safe -= 1
        con_k = LinearConstraint(np.ones((1, n)), k_safe, k_safe)
        r2 = milp(c=-np.asarray(score, float), integrality=integ, bounds=bnd, constraints=[con_b, con_k])
        chosen = np.flatnonzero(np.rint(r2.x) > 0.5) if (r2.success and r2.x is not None) else order_by_risk[:k_safe]
    return chosen, len(chosen)

def run_sim(data, rho, method):
    X_vis, y_vis = data["X_init"][data["init_vis_mask"]].copy(), data["y_init"][data["init_vis_mask"]].copy()
    w_vis = np.ones(len(y_vis), float)
    explored_logits, explored_y, round_exceeded = [], [], []
    tot_k, tot_pred, tot_real, tot_bud = 0, 0.0, 0.0, 0.0
    for r_data in data["rounds"]:
        model = LogisticClassifier(l2=1.0).fit(X_vis, y_vis, sample_weight=w_vis)
        p_all = model.predict_proba(r_data["X"])[:, 1]
        app = p_all < 0.22
        if np.any(app):
            X_vis, y_vis = np.vstack([X_vis, r_data["X"][app]]), np.concatenate([y_vis, r_data["y"][app]])
            w_vis = np.concatenate([w_vis, np.ones(np.sum(app))])
        Zr, yr, ead_r, pr = r_data["X"][~app], r_data["y"][~app], r_data["ead"][~app], p_all[~app]
        budget_t = rho * np.sum(pr * ead_r)
        tot_bud += budget_t
        pv, pf = model.predict_proba(X_vis)[:, 1], model.predict_proba(data["X_test"])[:, 1]
        fave_scores, logit_std = fave_value_and_leverage(X_vis, Zr, data["X_test"], pv, pr, pf)
        raw_cost = pr * ead_r

        if method == "CountOnly-MinRisk":
            chosen, _ = optimize_milp(raw_cost, -raw_cost, budget_t)
        elif method == "Joint-Best-Val":
            f_n = (fave_scores - np.min(fave_scores)) / max(np.ptp(fave_scores), 1e-9)
            c_n = (raw_cost - np.min(raw_cost)) / max(np.ptp(raw_cost), 1e-9)
            r_j = milp(c=-(f_n - 0.35 * c_n), integrality=np.ones(len(raw_cost), int), bounds=Bounds(np.zeros(len(raw_cost)), np.ones(len(raw_cost))), constraints=LinearConstraint(raw_cost.reshape(1, -1), -np.inf, budget_t))
            chosen = np.flatnonzero(np.rint(r_j.x) > 0.5) if (r_j.success and r_j.x is not None) else np.array([], int)
        elif method == "LRBE-Uncertainty":
            chosen, _ = optimize_milp(raw_cost, policy_score("Uncertainty", pr, raw_cost), budget_t)
        elif method == "LRBE-FAVE":
            chosen, _ = optimize_milp(raw_cost, fave_scores, budget_t)
        elif method == "LRBE-FAVE-Margin15":
            chosen, _ = optimize_milp(raw_cost, fave_scores, 0.85 * budget_t)
        elif method == "UCB-LRBE-FAVE":
            logit_pr = np.log(np.clip(pr, 1e-9, 1-1e-9)/np.clip(1-pr, 1e-9, 1-1e-9))
            if len(explored_y) >= 8:
                emp = np.clip(np.mean(explored_y) + 0.5/np.sqrt(len(explored_y)), 1e-4, 0.99)
                prd = np.clip(np.mean(expit(explored_logits)), 1e-4, 0.99)
                d_cal = np.clip(np.log(emp/(1-emp)) - np.log(prd/(1-prd)), 0.0, 1.1)
            else:
                d_cal = 0.35
            p_ucb = expit(logit_pr + d_cal + 1.0 * logit_std)
            chosen, _ = optimize_milp(p_ucb * ead_r, fave_scores, budget_t)
        elif method == "CC-LRBE-FAVE-Moderate":
            logit_pr = np.log(np.clip(pr, 1e-9, 1-1e-9)/np.clip(1-pr, 1e-9, 1-1e-9))
            if len(explored_y) >= 8:
                emp = np.clip(np.mean(explored_y) + 0.25/np.sqrt(len(explored_y)), 1e-4, 0.99)
                prd = np.clip(np.mean(expit(explored_logits)), 1e-4, 0.99)
                d_cal = np.clip(np.log(emp/(1-emp)) - np.log(prd/(1-prd)), 0.0, 0.85)
            else:
                d_cal = 0.25
            p_ucb = expit(logit_pr + d_cal + 0.65 * logit_std)
            chosen, _ = optimize_chance_constrained_milp(p_ucb, ead_r, fave_scores * (1.0 + 0.6 * logit_std), budget_t, z_alpha=1.05)
        elif method == "CC-LRBE-FAVE-Strict":
            logit_pr = np.log(np.clip(pr, 1e-9, 1-1e-9)/np.clip(1-pr, 1e-9, 1-1e-9))
            if len(explored_y) >= 8:
                emp = np.clip(np.mean(explored_y) + 0.5/np.sqrt(len(explored_y)), 1e-4, 0.99)
                prd = np.clip(np.mean(expit(explored_logits)), 1e-4, 0.99)
                d_cal = np.clip(np.log(emp/(1-emp)) - np.log(prd/(1-prd)), 0.0, 1.1)
            else:
                d_cal = 0.35
            p_ucb = expit(logit_pr + d_cal + 1.0 * logit_std)
            chosen, _ = optimize_chance_constrained_milp(p_ucb, ead_r, fave_scores * (1.0 + 0.5 * logit_std), budget_t, z_alpha=1.28)
        else:
            raise ValueError(method)

        pred_t = float(np.sum(raw_cost[chosen])) if len(chosen) > 0 else 0.0
        real_t = float(np.sum(yr[chosen] * ead_r[chosen])) if len(chosen) > 0 else 0.0
        tot_k += len(chosen)
        tot_pred += pred_t
        tot_real += real_t
        round_exceeded.append(1 if real_t > budget_t else 0)

        if len(chosen) > 0:
            explored_logits.extend(np.log(np.clip(pr[chosen], 1e-9, 1-1e-9)/np.clip(1-pr[chosen], 1e-9, 1-1e-9)).tolist())
            explored_y.extend(yr[chosen].tolist())
            X_vis, y_vis = np.vstack([X_vis, Zr[chosen]]), np.concatenate([y_vis, yr[chosen]])
            if method in ("CC-LRBE-FAVE-Moderate", "CC-LRBE-FAVE-Strict"):
                ipw = np.clip(len(Zr) / max(len(chosen), 1), 1.8 if method == "CC-LRBE-FAVE-Moderate" else 1.5, 5.0 if method == "CC-LRBE-FAVE-Moderate" else 4.5)
                w_vis = np.concatenate([w_vis, np.full(len(chosen), ipw)])
            else:
                w_vis = np.concatenate([w_vis, np.ones(len(chosen))])

    final_model = LogisticClassifier(l2=1.0).fit(X_vis, y_vis, sample_weight=w_vis)
    rec, auc, brier = compute_metrics(data["y_test"], final_model.predict_proba(data["X_test"])[:, 1])
    return {
        "exceed_rate": float(np.mean(round_exceeded)),
        "realized_budget_ratio": float(tot_real / max(tot_bud, 1e-9)),
        "predicted_budget_ratio": float(tot_pred / max(tot_bud, 1e-9)),
        "cumulative_feedback_count": int(tot_k),
        "cumulative_realized_loss": round(float(tot_real), 2),
        "cumulative_budget": round(float(tot_bud), 2),
        "recall": rec,
        "roc_auc": auc,
        "brier_score": brier,
    }

SEEDS = [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010]
RHOS = [0.025, 0.05, 0.10, 0.20, 0.40]
METHODS = [
    "CountOnly-MinRisk",
    "Joint-Best-Val",
    "LRBE-Uncertainty",
    "LRBE-FAVE",
    "LRBE-FAVE-Margin15",
    "UCB-LRBE-FAVE",
    "CC-LRBE-FAVE-Moderate",
    "CC-LRBE-FAVE-Strict",
]

datasets = {s: generate_selective_credit_dataset(s) for s in SEEDS}
raw_rows = []
for s in SEEDS:
    for r in RHOS:
        for m in METHODS:
            out = run_sim(datasets[s], r, m)
            out["seed"] = s
            out["rho"] = r
            out["method"] = m
            raw_rows.append(out)

df_raw = pd.DataFrame(raw_rows)
cols_order = ["seed", "rho", "method", "exceed_rate", "realized_budget_ratio", "predicted_budget_ratio", "cumulative_feedback_count", "cumulative_realized_loss", "cumulative_budget", "recall", "roc_auc", "brier_score"]
df_raw = df_raw[cols_order]
df_raw.to_csv(os.path.join(PKG_STAGE, "results/raw_runs_all_400_trajectories.csv"), index=False)

# Table 1: Overall summary
df_t1 = df_raw.groupby("method", sort=False).agg(
    n_runs=("seed", "count"),
    exceed_rate_pct=("exceed_rate", lambda x: round(100 * np.mean(x), 2)),
    realized_budget_ratio_mean=("realized_budget_ratio", lambda x: round(np.mean(x), 4)),
    predicted_budget_ratio_mean=("predicted_budget_ratio", lambda x: round(np.mean(x), 4)),
    avg_feedback_count=("cumulative_feedback_count", lambda x: round(np.mean(x), 1)),
    recall_mean=("recall", lambda x: round(np.mean(x), 4)),
    roc_auc_mean=("roc_auc", lambda x: round(np.mean(x), 4)),
    brier_score_mean=("brier_score", lambda x: round(np.mean(x), 4)),
).reset_index()
df_t1.to_csv(os.path.join(PKG_STAGE, "results/table1_overall_method_summary.csv"), index=False)

# Table 2: Breakdown by rho
df_t2 = df_raw.groupby(["rho", "method"], sort=False).agg(
    n_seeds=("seed", "count"),
    exceed_rate_pct=("exceed_rate", lambda x: round(100 * np.mean(x), 2)),
    realized_ratio_mean=("realized_budget_ratio", lambda x: round(np.mean(x), 4)),
    avg_feedback_count=("cumulative_feedback_count", lambda x: round(np.mean(x), 1)),
    recall_mean=("recall", lambda x: round(np.mean(x), 4)),
    roc_auc_mean=("roc_auc", lambda x: round(np.mean(x), 4)),
    brier_score_mean=("brier_score", lambda x: round(np.mean(x), 4)),
).reset_index()
df_t2.to_csv(os.path.join(PKG_STAGE, "results/table2_budget_sweep_by_rho.csv"), index=False)

# Table 3: Paired Seed-Cluster Bootstrap (10,000 resamples) like CCFB_extension_report.md Table 4
sig_rows = []
rng_boot = np.random.default_rng(20261103)
comparisons = [
    ("LRBE-FAVE", "CountOnly-MinRisk"),
    ("LRBE-FAVE", "Joint-Best-Val"),
    ("CC-LRBE-FAVE-Moderate", "CountOnly-MinRisk"),
    ("CC-LRBE-FAVE-Moderate", "LRBE-FAVE"),
    ("CC-LRBE-FAVE-Strict", "CountOnly-MinRisk"),
    ("CC-LRBE-FAVE-Strict", "LRBE-FAVE"),
    ("CC-LRBE-FAVE-Strict", "Joint-Best-Val"),
]
for target_m, base_m in comparisons:
    for metric in ["exceed_rate", "realized_budget_ratio", "recall"]:
        t_vals = df_raw[df_raw["method"] == target_m].groupby("seed")[metric].mean().values
        b_vals = df_raw[df_raw["method"] == base_m].groupby("seed")[metric].mean().values
        diff = t_vals - b_vals
        mean_diff = float(np.mean(diff))
        std_diff = float(np.std(diff, ddof=1))
        cohen_dz = mean_diff / max(std_diff, 1e-12)
        boots = [np.mean(rng_boot.choice(diff, size=len(diff), replace=True)) for _ in range(10000)]
        ci_low, ci_high = np.percentile(boots, [2.5, 97.5])
        try:
            _, p_val = wilcoxon(diff)
        except Exception:
            p_val = 1.0
        win_ratio = float(np.mean(diff > 0)) if metric == "recall" else float(np.mean(diff < 0))
        sig_rows.append({
            "comparison": f"{target_m} minus {base_m}",
            "metric": metric,
            "n_paired_seeds": len(diff),
            "mean_difference": round(mean_diff, 6),
            "effect_size_cohen_dz": round(cohen_dz, 4),
            "bootstrap_95ci_low": round(float(ci_low), 6),
            "bootstrap_95ci_high": round(float(ci_high), 6),
            "p_value_wilcoxon": round(float(p_val), 6),
            "favorable_win_ratio": round(win_ratio, 2),
        })
df_t3 = pd.DataFrame(sig_rows)
df_t3.to_csv(os.path.join(PKG_STAGE, "results/table3_paired_bootstrap_significance.csv"), index=False)

# ==============================================================================
# 2. Generate Publication-Quality Figures (300 DPI PNG & PDF)
# ==============================================================================
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.dpi": 300,
})

# Figure 1: Dual-panel Budget Sweep (Exceedance Rate & Realized-to-Budget Ratio)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
plot_methods = [
    ("LRBE-FAVE", "Orig. LRBE-FAVE (Point Est.)", "#D55E00", "o", "-"),
    ("LRBE-FAVE-Margin15", "Phase 4 Static Margin (15%)", "#E69F00", "s", "--"),
    ("UCB-LRBE-FAVE", "UCB-LRBE-FAVE (Calibrated UCB)", "#56B4E9", "^", "-."),
    ("CC-LRBE-FAVE-Moderate", "CC-LRBE-FAVE (Moderate, z=1.05)", "#009E73", "D", "-"),
    ("CC-LRBE-FAVE-Strict", "CC-LRBE-FAVE (Strict, z=1.28)", "#0072B2", "P", "-"),
]
x_idx = np.arange(len(RHOS))
x_labels = [str(r) for r in RHOS]

for m_code, m_label, color, marker, ls in plot_methods:
    sub = df_t2[df_t2["method"] == m_code].sort_values("rho")
    ax1.plot(x_idx, sub["exceed_rate_pct"], label=m_label, color=color, marker=marker, linestyle=ls, linewidth=2.0, markersize=6)
    ax2.plot(x_idx, sub["realized_ratio_mean"], label=m_label, color=color, marker=marker, linestyle=ls, linewidth=2.0, markersize=6)

ax1.axhline(10.0, color="#666666", linestyle=":", linewidth=1.2, label="10% Target Risk Tolerance")
ax1.set_xticks(x_idx)
ax1.set_xticklabels(x_labels)
ax1.set_xlabel("Per-Round Risk Budget Multiplier ($\\rho$)")
ax1.set_ylabel("Realized Risk Exceedance Rate (%)")
ax1.set_title("(a) Realized Budget Violation Rate vs. Budget Scale $\\rho$", fontweight="bold")
ax1.set_ylim(-3, 80)
ax1.grid(True, linestyle="--", alpha=0.35)
ax1.legend(loc="upper right", frameon=True)

ax2.axhline(1.0, color="#CC0000", linestyle=":", linewidth=1.5, label="Budget Limit Redline (Ratio = 1.0)")
ax2.set_xticks(x_idx)
ax2.set_xticklabels(x_labels)
ax2.set_xlabel("Per-Round Risk Budget Multiplier ($\\rho$)")
ax2.set_ylabel("Realized Loss / Budget Ratio")
ax2.set_title("(b) Realized-to-Budget Ratio Across Scales", fontweight="bold")
ax2.set_ylim(0.40, 1.25)
ax2.grid(True, linestyle="--", alpha=0.35)
ax2.legend(loc="upper right", frameon=True)

plt.tight_layout()
fig.savefig(os.path.join(PKG_STAGE, "figures/fig1_exceedance_and_ratio_across_rho.png"), dpi=300)
fig.savefig(os.path.join(PKG_STAGE, "figures/fig1_exceedance_and_ratio_across_rho.pdf"))
plt.close(fig)

# Figure 2: Safety vs. Recall Pareto Frontier
fig, ax = plt.subplots(figsize=(7.2, 4.8))
palette_map = {
    "CountOnly-MinRisk": ("#7F7F7F", "v", "CountOnly-MinRisk (Baseline)"),
    "Joint-Best-Val": ("#CC79A7", "<", "Joint-Best-Val (Tuned Scalarized)"),
    "LRBE-Uncertainty": ("#999999", ">", "LRBE-Uncertainty (Orig.)"),
    "LRBE-FAVE": ("#D55E00", "o", "LRBE-FAVE (Orig. Flagship)"),
    "LRBE-FAVE-Margin15": ("#E69F00", "s", "LRBE-FAVE + Static Margin 15%"),
    "UCB-LRBE-FAVE": ("#56B4E9", "^", "UCB-LRBE-FAVE (Fisher UCB Only)"),
    "CC-LRBE-FAVE-Moderate": ("#009E73", "D", "CC-LRBE-FAVE + IPW (Moderate, Ours)"),
    "CC-LRBE-FAVE-Strict": ("#0072B2", "P", "CC-LRBE-FAVE + IPW (Strict, Ours)"),
}
for _, row in df_t1.iterrows():
    m = row["method"]
    color, marker, lbl = palette_map[m]
    ax.scatter(row["exceed_rate_pct"], row["recall_mean"], color=color, marker=marker, s=110, label=lbl, zorder=4, edgecolors="black", linewidth=0.7)
    offset_y = 0.00035 if "Strict" not in m else -0.00055
    ax.annotate(m.replace("-Moderate", " (Mod)").replace("-Strict", " (Strict)"),
                (row["exceed_rate_pct"], row["recall_mean"] + offset_y),
                fontsize=8, ha="center")

ax.axvspan(0, 10, color="#009E73", alpha=0.10, label="Safe Realized Risk Zone (\u226410% Exceedance)")
ax.set_xlabel("Realized Risk Exceedance Rate (%)  [Lower is Safer \u2190]")
ax.set_ylabel("Downstream Test Recall  [Higher is Better \u2191]")
ax.set_title("Safety-Performance Pareto Frontier (10 Seeds \u00d7 5 Budget Levels)", fontweight="bold")
ax.set_xlim(-2, 72)
ax.set_ylim(0.7855, 0.7980)
ax.grid(True, linestyle="--", alpha=0.35)
ax.legend(loc="lower left", fontsize=8, frameon=True)
plt.tight_layout()
fig.savefig(os.path.join(PKG_STAGE, "figures/fig2_pareto_frontier_safety_vs_recall.png"), dpi=300)
fig.savefig(os.path.join(PKG_STAGE, "figures/fig2_pareto_frontier_safety_vs_recall.pdf"))
plt.close(fig)

# Figure 3: Forest Plot of Paired Bootstrap 95% CIs on Downstream Recall
fig, ax = plt.subplots(figsize=(7.8, 4.2))
sub_sig = df_t3[df_t3["metric"] == "recall"].copy()
y_pos = np.arange(len(sub_sig))
colors_ci = ["#D55E00" if "LRBE-FAVE minus" in c else "#0072B2" for c in sub_sig["comparison"]]

for idx, (_, r) in enumerate(sub_sig.iterrows()):
    ax.errorbar(
        r["mean_difference"], idx,
        xerr=[[r["mean_difference"] - r["bootstrap_95ci_low"]], [r["bootstrap_95ci_high"] - r["mean_difference"]]],
        fmt="o", color=colors_ci[idx], ecolor=colors_ci[idx], elinewidth=2.2, capsize=5, markersize=7
    )
    ax.text(r["bootstrap_95ci_high"] + 0.0005, idx,
            f"{r['mean_difference']:+.4f} [{r['bootstrap_95ci_low']:+.4f}, {r['bootstrap_95ci_high']:+.4f}] (Win={int(r['favorable_win_ratio']*100)}%)",
            va="center", fontsize=8)

ax.axvline(0.0, color="#CC0000", linestyle="--", linewidth=1.4, label="Zero Effect Boundary (Crosses 0 = Non-significant)")
ax.set_yticks(y_pos)
ax.set_yticklabels(sub_sig["comparison"])
ax.invert_yaxis()
ax.set_xlabel("Paired Seed-Cluster Recall Difference (10,000 Bootstrap Resamples)")
ax.set_title("Paired Bootstrap 95% CIs on Recall: Breaking the Negative-Result Bottleneck", fontweight="bold")
ax.set_xlim(-0.009, 0.024)
ax.grid(True, axis="x", linestyle="--", alpha=0.35)
ax.legend(loc="lower right", frameon=True)
plt.tight_layout()
fig.savefig(os.path.join(PKG_STAGE, "figures/fig3_paired_bootstrap_ci_forest.png"), dpi=300)
fig.savefig(os.path.join(PKG_STAGE, "figures/fig3_paired_bootstrap_ci_forest.pdf"))
plt.close(fig)

print("Generated all CSV tables and 300-DPI figures successfully.")
