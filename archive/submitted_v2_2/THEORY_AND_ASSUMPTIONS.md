# THEORETICAL GUARANTEES, ASSUMPTIONS, AND SCOPE LIMITATIONS

## 1. Conditional Safety Guarantee (Proposition 1 — Route A)

Let $C_t$ denote the candidate loan pool at decision round $t$ with known exposure proxies $\{E_i\}_{i \in C_t}$ ($E_i > 0$), and let $Y_i \in \{0, 1\}$ denote the binary default indicator with true conditional default probability $p_i^* = \mathbb{P}(Y_i = 1 \mid x_i, \mathcal{H}_t)$.

### Assumptions
- **Assumption A1 (Simultaneous Upper-Risk Coverage on Selected Set)**: For a given confidence tolerance $\delta_r \in (0, 1)$, the calibrated upper-risk estimator $q_i$ constructed from $D_t$ satisfies simultaneous upper coverage on the selected subset $S_t \subseteq C_t$:
  $$\mathbb{P}\Big(\mathcal{E}_{\text{cov}}(S_t)\Big) \ge 1 - \delta_r, \quad \text{where } \mathcal{E}_{\text{cov}}(S_t) := \big\{\forall i \in S_t,\; p_i^* \le q_i\big\}.$$
- **Assumption A2 (Conditional Independence of Realized Defaults)**: Conditional on the covariates $\{x_i\}_{i \in S_t}$ and regime state at round $t$, the realized binary default indicators $\{Y_i\}_{i \in S_t}$ are independent Bernoulli random variables with parameters $\{p_i^*\}_{i \in S_t}$.

### Proposition 1 (Conditional Upper-Confidence Tail Risk Bound)
Under Assumptions A1 and A2, for any subset $S_t \subseteq C_t$ satisfying the exact unscaled Bernstein Risk Envelope constraint
$$R_\alpha(S_t) := \mu_U(S_t) + \sqrt{2 V_U(S_t)\log(1/\alpha)} + \frac{M(S_t)}{3}\log(1/\alpha) \le B_t,$$
where $\mu_U(S_t) = \sum_{i \in S_t} E_i q_i$, $V_U(S_t) = \sum_{i \in S_t} E_i^2 \phi(q_i)$ with $\phi(q) = q(1-q)\mathbb{I}(q \le 0.5) + 0.25\mathbb{I}(q > 0.5)$, and $M(S_t) = \max_{i \in S_t} E_i$, the realized portfolio default loss $L(S_t) = \sum_{i \in S_t} E_i Y_i$ satisfies:
1. **Conditional Tail Bound on $\mathcal{E}_{\text{cov}}(S_t)$**:
   $$\mathbb{P}\big(L(S_t) > B_t \;\big|\; \mathcal{E}_{\text{cov}}(S_t)\big) \le \mathbb{P}\big(L(S_t) > R_\alpha(S_t) \;\big|\; \mathcal{E}_{\text{cov}}(S_t)\big) \le \alpha.$$
2. **Unconditional Total-Probability Tail Bound**:
   $$\mathbb{P}\big(L(S_t) > B_t\big) \le \alpha + \delta_r.$$

### Proof Sketch
On the event $\mathcal{E}_{\text{cov}}(S_t)$, $p_i^* \le q_i$ for all $i \in S_t$. Consequently, the true expected loss satisfies $\mu^*(S_t) = \sum_{i \in S_t} E_i p_i^* \le \sum_{i \in S_t} E_i q_i = \mu_U(S_t)$, and the true conditional variance satisfies $\text{Var}(L(S_t)) = \sum_{i \in S_t} E_i^2 p_i^*(1-p_i^*) \le \sum_{i \in S_t} E_i^2 \phi(q_i) = V_U(S_t)$ because $p(1-p)$ is strictly increasing on $[0, 0.5]$ and bounded above by $0.25$ on $(0.5, 1]$. Since each centered summand $Z_i = E_i(Y_i - p_i^*)$ satisfies $Z_i \le E_i \le M(S_t)$ almost surely under Assumption A2, Bernstein's inequality yields $\mathbb{P}(L(S_t) - \mu^*(S_t) > t \mid \mathcal{E}_{\text{cov}}(S_t)) \le \exp\big(-\frac{t^2 / 2}{V_U(S_t) + M(S_t)t/3}\big)$. Setting the right-hand side equal to $\alpha$ and solving the quadratic equation in $t$ with $\ell = \log(1/\alpha)$ gives $t_\alpha \le \sqrt{2V_U(S_t)\ell} + \frac{M(S_t)}{3}\ell$. Combining $\mu^*(S_t) \le \mu_U(S_t)$ and $R_\alpha(S_t) \le B_t$ yields (1), and the union bound with $\mathbb{P}(\neg \mathcal{E}_{\text{cov}}(S_t)) \le \delta_r$ yields (2). $\square$

---

## 2. Finite-Family Oracle ($S_{\mathcal{F}}^*$) & P3 World D Conditional-$z$ Diagnostics

1. **Finite-Family Oracle Definition ($S_{\mathcal{F}}^*$)**:
   - Because post-settlement Brier loss $R(S) = \text{Brier}(\hat{\theta}(D_t \cup O_t \cup S); T_{\text{test}})$ is a non-modular set function coupled through logistic regression optimization, exhaustive $2^{|C_t|}$ subset enumeration is intractable for $|C_t| = 24$.
   - Accordingly, Diagnostic 3 in P3 is defined as the **Finite-Family Oracle**:
     $$S_{\mathcal{F}}^* = \arg\min_{S \in \mathcal{F}_t:\; R_\alpha^*(S) \le B_t} R(S),$$
     where $\mathcal{F}_t$ is the structured candidate family generated across all cardinality levels $k \in \{1, \dots, K_{\text{safe}}\}$ (`finite_family_oracle_brier` in `p3_realx/p3_oracle_risk_diagnostics.csv`).
2. **Separating Macro-Shock Drift from Correlation in `World_D_Correlated_Defaults`**:
   - In `World_D_Correlated_Defaults`, candidate defaults are generated under a shared macroeconomic shock $z_{\text{macro}} \sim \mathcal{N}(0, 0.65^2)$ with true conditional probability $p_i^*(z_{\text{macro}}) = \mathbb{P}(Y_i = 1 \mid X_i, z_{\text{macro}})$, while the pre-shock marginal probability is $p_{i,\text{marginal}} = \mathbb{P}(Y_i = 1 \mid X_i, z_{\text{macro}}=0)$.
   - Evaluating coverage only against $p_{i,\text{marginal}}$ would conflate correlation with unmodeled macro-shock drift. In `p3_realx/p3_world_d_conditional_vs_marginal_audit.csv` and `p3_realx/p3_oracle_risk_diagnostics.csv`, all 7 Oracle Diagnostics are aligned to the true data-generating $p_i^*(z_{\text{macro}})$ (`pointwise_coverage = 94.0%`, `simultaneous_coverage = 86.0%`), alongside the marginal pre-shock reference (`97.3%` / `92.0%`).

---

## 3. Exposure Proxy & Operational Risk Envelope Scope

- In public credit benchmark datasets where accounting Loss-Given-Default ($\text{LGD}$) and post-default recovery cashflows are not reported, $E_i$ is defined as a standardized **Principal / Credit-Limit Exposure Proxy**:
  - `LendingClub`: `loan_amnt` (funded/requested loan principal);
  - `HMEQ`: `LOAN` (requested home-equity loan amount);
  - `UCI_Credit`: `LIMIT_BAL` (credit card limit proxy);
  - `PPDai`: `credit_limit` (approved credit line proxy).
- Consequently, $R_\alpha(S_t) \le B_t$ is interpreted as an **Operational Upper-Confidence Risk Envelope under Proxy Exposure $E_i$**, rather than a regulatory Basel IRB accounting capital calculation.
