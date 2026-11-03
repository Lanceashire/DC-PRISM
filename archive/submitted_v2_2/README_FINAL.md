# DC-PRISM (DASFAA 2027 Final Clean-Room Revision Package — Frozen Release)

**Title**: *DC-PRISM: Pending-Aware Safe Active Learning with Adaptive Cardinality Allowance for Delayed Credit Risk Assessment*  
**Target Venue**: DASFAA 2027 Research Track (CCF-B Formal Submission Package)

---

## 1. Scientific Scope & Final Audit Closure Summary

This package provides the complete, frozen experimental and theoretical artifacts for `DC-PRISM`, organized around a four-stage progressive validation hierarchy (`P0` $\to$ `P1` $\to$ `P2` $\to$ `P3` $\to$ `P4`):

1. **Frozen Core Algorithm & P2 Mechanism Confirmation (`p2_confirmation/`, $N = 128$ Worlds + $54$ Cross-Mechanism Worlds)**:
   - **Pure `Bernoulli(p_true)` DGP**: Every synthetic and Real-X candidate label is drawn strictly once from $Y_i \sim \text{Bernoulli}(p_i^*)$ (`0%` post-generation label overwrites, verified in `results/audit/oracle_probability_consistency.csv`).
   - **All 5 Nearest-Neighbor Ablations (`A1`–`A5`)**: `DC-PRISM` (`mean_brier = 0.123585`) significantly outperforms `Safe-PA-NoGate` (`0.126487`, $p_{\text{Holm}} = 2.66 \times 10^{-11}$), `Fixed-Eps-Safe-PA` (`0.133493`, $p_{\text{Holm}} = 8.48 \times 10^{-16}$), `D-Only-Safe-PA` (`0.130662`, $p_{\text{Holm}} = 2.55 \times 10^{-11}$), `PRISM-ValueUCB` (`0.128829`, $p_{\text{Holm}} = 2.24 \times 10^{-10}$), and `Safe-MinRisk` (`0.140034`, $p_{\text{Holm}} = 6.32 \times 10^{-18}$).
   - **Cross-Mechanism Heterogeneity (`p2_cross_mechanism_confirmation.csv`)**: In `Substitution + Positive` coupling, releasing cardinality ($k < K_{\text{safe}}$) yields a large positive gain (`+0.0156 Brier`), whereas in `Irrelevant + Negative` coupling, safe min-risk points are already informative and `DC-PRISM`'s conservative gate automatically restricts $\varepsilon^*$ (`mean_eps_star = 0.083`), matching `Safe-MinRisk` within `0.0005` Brier while avoiding the severe degradation of fixed-release baselines (`Fixed-Eps-Safe-PA` loses `0.0044` Brier).
   - **Exact Quantity-Composition Decomposition ($Q_k / C_k / J$) & $\mathbb{P}(C_k > Q_k)$**: Across the 128 P2 worlds (`p2_confirmation/p2_brp_hrr_summary.csv`), `DC-PRISM` achieves $\mathbb{P}(C_k > Q_k) = 78.1\%$ (`80.6%` when releasing), **Beneficial Release Precision (`BRP = 77.4%`)**, and the lowest **Harmful Release Rate (`HRR = 16.9%`)** among all cardinality-releasing policies (vs `Fixed-Eps-Safe-PA`: $\mathbb{P}(C_k > Q_k) = 37.5\%$, `BRP = 35.2%`, `HRR = 54.7%`).

2. **P3 Real-X Semi-Synthetic Benchmark (`p3_realx/`, `Finite-Family Oracle` & World D Conditional-$z$ Alignment)**:
   - Evaluates `HMEQ-X` and `LendingClub-X` across 4 worlds (`World_A_Model_Correct`, `World_B_Misspecified`, `World_C_Risk_Shift`, `World_D_Correlated_Defaults`).
   - **Finite-Family Oracle ($S_{\mathcal{F}}^*$)**: Because finding the global optimum over all $2^{|C_t|}$ subsets of a post-settlement non-modular Brier objective is combinatorially intractable for large pools, Diagnostic 3 is explicitly defined as the **Finite-Family Oracle**:
     $$S_{\mathcal{F}}^* = \arg\min_{S \in \mathcal{F}_t:\; R_\alpha^*(S) \le B_t} R(S),$$
     where $\mathcal{F}_t$ is the structured candidate family generated across all cardinality levels $k$ (`finite_family_oracle_brier`).
   - **World D Conditional vs Marginal Probability Separation (`p3_world_d_conditional_vs_marginal_audit.csv`)**: In `World_D_Correlated_Defaults`, a shared macroeconomic shock $z_{\text{macro}} \sim \mathcal{N}(0, 0.65^2)$ induces correlated defaults with true conditional probability $p_i^*(z_{\text{macro}}) = \mathbb{P}(Y_i = 1 \mid X_i, z_{\text{macro}})$. We explicitly log $z_{\text{macro}}$, pre-shock marginal $p_{i,\text{marginal}}$, and conditional $p_i^*(z_{\text{macro}})$, aligning all 7 Oracle Diagnostics to $p_i^*(z_{\text{macro}})$ (`pointwise_coverage = 94.0%`, `simultaneous_coverage = 86.0%` vs marginal `97.3%` / `92.0%`), separating macro-shock drift from correlation effects.

3. **Final P4 Clean-Room Real-Dataset Benchmark (`p4_real_labels/`, Option A Strictly Inductive $D_0$-Only Preprocessing, 16 Seeds `4001–4016`)**:
   - **Option A Strictly Inductive $D_0$-Only Preprocessing**: `SplitBeforePreprocessor` is fitted **100% strictly on `Historical_D0`** ($68$ historical loans selected from an initial historical window prior to and strictly disjoint from the future candidate stream), with a pure numerical stability floor `prep.stds = np.maximum(prep.stds, 1e-3)` and zero reference to future pool covariate statistics.
   - **100% Hidden-Label-Free State Construction & Mutation Invariance**: Flipping or permuting 100% of hidden labels ($y_O, y_C, y_T$) leaves all cohort IDs, preprocessed covariates, budgets $B_t$, upper-risk scores $q_C$, and Round 1 selections 100% identical (`results/audit/hidden_label_mutation_test.json`).
   - **True Delayed Ordinary Feedback Queue ($L = 2$ Rounds) & 0.0% Duplicate Exploration**: Ordinary loans $O_t$ enter `pending_queue` with `release_round = t + 2` (`results/audit/pending_release_audit.csv`), and every applicant ID is queried at most once (`0.0%` duplicate rate in `results/audit/duplicate_acquisition_audit.csv`).
   - **Unified 16 Seeds (`4001–4016`) × 10 Rounds × 4 Datasets × 10 Policies & Honest Real-Data Heterogeneity**:
     - Across the 4-dataset equal-weight meta-summary (`p4_real_labels/p4_cross_dataset_meta_summary.csv`), `DC-PRISM` achieves lower mean 10-round `ALC-Brier` (`0.161909`) and `Round-10 Brier` (`0.157090`) than `Safe-MinRisk` (`0.162800` / `0.160997`, winning Round-10 Brier on all 4 out of 4 datasets), `Safe-PA-NoGate` (`0.162185` / `0.157971`), `D-Only-Safe-PA` (`0.162146` / `0.157779`), and `OrdinaryOnly` (`0.165595` / `0.164224`).
     - At the individual dataset level (`p4_real_labels/p4_dataset_level_paired_statistics.csv`, $N=16$ seeds per dataset), real-data performance is honestly regime-dependent: on `HMEQ` and `PPDai`, `DC-PRISM` outperforms both `Safe-MinRisk` and `Safe-PA-NoGate`; on `LendingClub`, `DC-PRISM` outperforms `Safe-MinRisk`, `D-Only-Safe-PA`, and `OrdinaryOnly` while aggressive un-gated release (`Safe-PA-NoGate`) attains slightly lower Brier at the cost of higher $\varepsilon^*$ (`0.588` vs `0.388`); on noisy `UCI_Credit`, `DC-PRISM` triggers its conservative gate (`mean_eps_star = 0.033`), protecting against over-release and improving Round-10 Brier over `Safe-MinRisk`.

4. **Data Provenance & Exposure Proxy Definitions (`scripts/prepare_lendingclub.py` & `data_manifest/`)**:
   - Documents the exact LendingClub cohort filtering chain: legacy repo registry reference (`16,566` pre-maturity rows) / raw uploaded 2015 sample (`LC_Loan_sample_2015.csv`, `11,910` rows) $\to$ retaining strictly resolved loans (`Fully Paid: 8,425` + `Charged Off: 2,199` = `10,624` rows, default rate `20.70%`) and excluding `1,286` unresolved loans (`Current: 1,227`, `Late (31-120 days): 32`, `In Grace Period: 22`, `Late (16-30 days): 5`).
   - Explicitly defines `loan_amnt` (`LendingClub`), `LOAN` (`HMEQ`), `LIMIT_BAL` (`UCI_Credit`), and `credit_limit` (`PPDai`) as normalized **Exposure Proxies ($E_i$)** rather than accounting $\text{LGD} \times \text{EAD}$ (`data_manifest/cohort_manifest.csv`, `data_manifest/feature_manifest.csv`, `data_manifest/dataset_hash.csv`, `data_manifest/DC_PRISM_REALDATA_REGISTRY.csv`).

---

## 2. Reproducibility Entrypoints

> **Note**: Runtime depends on hardware; fast mode is intended for verification rather than full statistical reproduction.

```bash
# Fast verification mode (runs all 7 unit tests + provenance manifests + fast P0-P4 check + table/figure build)
bash reproduce_fast.sh

# Full statistical reproduction from scratch (128 P2 worlds + 54 cross-mechanism worlds + 25 P3 seeds + 16 P4 seeds)
bash reproduce_full.sh
```
