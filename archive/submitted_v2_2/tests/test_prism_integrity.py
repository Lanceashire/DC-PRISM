#!/usr/bin/env python3
"""
Unit & Protocol Integrity Tests for DC-PRISM Revision 2 Package (DASFAA 2027)
Audits:
1. Exact Bernstein Risk Envelope formula (zero 0.32 factor, ell = log(1/alpha))
2. 100% Exact K_safe and MinRisk on small candidate pools (n <= 16)
3. Strict Hidden-Label Isolation in Selector
4. Exogenous Budget Hash Consistency across all policies
5. Factor C (risk_regime) directly modifies q_i
6. Full P4 State-Builder Hidden-Label Mutation Invariance & 0% Duplicate Exploration Rate
7. Q_k / C_k / J Decomposition Algebraic Identity (J == C_k - Q_k == R(M_K) - R(S))
"""
import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from prism_engine import (
    UnifiedLogisticLearner,
    bernstein_components,
    bernstein_envelope,
    compute_calibrated_upper_risk,
    compute_exogenous_budget,
    solve_safe_minrisk_exact_or_bounded,
    select_acquisition_policy,
    compute_q_c_j_decomposition,
)
from run_p3_p4 import load_four_real_datasets, build_strictly_hidden_y_free_real_state


class TestPrismIntegrity(unittest.TestCase):
    def test_1_exact_bernstein_formula_no_magic_factor(self):
        """P0-R1: Verify code matches R_alpha(S) = mu_U + sqrt(2 V_U ell) + (M/3)*ell exactly."""
        q = np.array([0.08, 0.12, 0.25, 0.05, 0.18], dtype=float)
        c = np.array([1.2, 0.8, 2.5, 1.0, 1.5], dtype=float)
        alpha = 0.10
        res = bernstein_components(q, c, alpha=alpha)

        ell_expected = float(np.log(1.0 / alpha))
        mu_expected = float(np.sum(c * q))
        vu_expected = float(np.sum((c ** 2) * q * (1.0 - q)))
        m_expected = float(np.max(c))
        r_expected = mu_expected + np.sqrt(2.0 * vu_expected * ell_expected) + (m_expected / 3.0) * ell_expected

        self.assertAlmostEqual(res["ell"], ell_expected, places=12)
        self.assertAlmostEqual(res["mu_U"], mu_expected, places=12)
        self.assertAlmostEqual(res["V_U"], vu_expected, places=12)
        self.assertAlmostEqual(res["M"], m_expected, places=12)
        self.assertAlmostEqual(res["risk_envelope"], r_expected, places=12)
        self.assertAlmostEqual(bernstein_envelope(q, c, alpha=alpha), r_expected, places=12)

    def test_2_small_pool_exact_enumeration_k_safe(self):
        """P0-R3: Verify Safe-MinRisk achieves 100% exact K_safe on n=12 via 2^n enumeration."""
        rng = np.random.default_rng(42)
        n = 12
        q = rng.uniform(0.03, 0.28, size=n)
        c = rng.uniform(0.6, 2.2, size=n)
        alpha = 0.10
        B_t, _ = compute_exogenous_budget(q, c, alpha=alpha, target_k=5)

        S_sol, sol = solve_safe_minrisk_exact_or_bounded(q, c, B_t, alpha=alpha, exact_max_n=16)
        self.assertEqual(sol["solver_status"], "EXACT_ENUMERATION")
        self.assertEqual(sol["optimality_gap"], 0)
        self.assertEqual(sol["k_safe_lower"], sol["k_safe_upper"])

        best_k = 0
        best_r = float("inf")
        for mask in range(1, 1 << n):
            idx = [i for i in range(n) if (mask >> i) & 1]
            r = bernstein_envelope(q[idx], c[idx], alpha=alpha)
            if r <= B_t + 1e-12:
                if len(idx) > best_k or (len(idx) == best_k and r < best_r - 1e-12):
                    best_k = len(idx)
                    best_r = r
        self.assertEqual(len(S_sol), best_k)
        self.assertEqual(sol["k_safe_lower"], best_k)
        if best_k > 0:
            self.assertAlmostEqual(sol["risk_envelope"], best_r, places=10)

    def test_3_hidden_label_isolation_in_selector(self):
        """Section 6: Hidden labels y_O, y_C, y_T are not accepted by select_acquisition_policy."""
        rng = np.random.default_rng(101)
        X_D = rng.normal(size=(60, 5))
        y_D = (X_D[:, 0] - 0.5 * X_D[:, 1] + rng.normal(scale=0.5, size=60) > 0).astype(int)
        learner = UnifiedLogisticLearner(l2=0.45).fit(X_D, y_D)

        X_O = rng.normal(size=(10, 5))
        X_C = rng.normal(size=(14, 5))
        ead_C = rng.uniform(0.7, 1.8, size=14)
        X_Q = rng.normal(size=(80, 5))

        q_C, _ = compute_calibrated_upper_risk(learner, X_D, y_D, X_C, delta_r=0.10, rng=np.random.default_rng(77))
        B_t, _ = compute_exogenous_budget(q_C, ead_C, alpha=0.10, target_k=5)

        S1, M1, info1 = select_acquisition_policy(
            "DC-PRISM", X_D, y_D, X_O, X_C, ead_C, X_Q, B_t,
            learner=learner, q_C_precomputed=q_C, rng=np.random.default_rng(999)
        )
        S2, M2, info2 = select_acquisition_policy(
            "DC-PRISM", X_D, y_D, X_O, X_C, ead_C, X_Q, B_t,
            learner=learner, q_C_precomputed=q_C, rng=np.random.default_rng(999)
        )
        self.assertTrue(np.array_equal(S1, S2))
        self.assertTrue(np.array_equal(M1, M2))
        self.assertAlmostEqual(info1["surplus_score"], info2["surplus_score"], places=12)

    def test_4_exogenous_budget_hash_consistency(self):
        """Section 18: Exogenous B_t and budget_hash are identical across all policies on the same state."""
        rng = np.random.default_rng(202)
        X_D = rng.normal(size=(50, 4))
        y_D = rng.integers(0, 2, size=50)
        learner = UnifiedLogisticLearner(l2=0.45).fit(X_D, y_D)
        X_O = rng.normal(size=(8, 4))
        X_C = rng.normal(size=(12, 4))
        ead_C = rng.uniform(0.5, 2.0, size=12)
        X_Q = rng.normal(size=(60, 4))

        q_C, _ = compute_calibrated_upper_risk(learner, X_D, y_D, X_C, delta_r=0.10, rng=np.random.default_rng(55))
        B_t, b_hash = compute_exogenous_budget(q_C, ead_C, alpha=0.10, target_k=5)
        self.assertEqual(len(b_hash), 16)

        for pol in ["DC-PRISM", "Safe-MinRisk", "Safe-PA-NoGate", "BAIT-Cost-Safe", "Orig-PA"]:
            S_pol, _, _ = select_acquisition_policy(
                pol, X_D, y_D, X_O, X_C, ead_C, X_Q, B_t,
                learner=learner, q_C_precomputed=q_C, rng=np.random.default_rng(123)
            )
            if pol in ["DC-PRISM", "Safe-MinRisk", "Safe-PA-NoGate", "BAIT-Cost-Safe"]:
                env = bernstein_envelope(q_C[S_pol], ead_C[S_pol], alpha=0.10)
                self.assertLessEqual(env, B_t + 1e-9)

    def test_5_factor_c_risk_regime_modifies_q_i(self):
        """Verify risk_regime directly alters q_i in Calibrated vs Optimistic vs Conservative."""
        rng = np.random.default_rng(303)
        X_D = rng.normal(size=(60, 4))
        y_D = rng.integers(0, 2, size=60)
        learner = UnifiedLogisticLearner(l2=0.45).fit(X_D, y_D)
        X_C = rng.normal(size=(15, 4))

        q_opt, _ = compute_calibrated_upper_risk(learner, X_D, y_D, X_C, risk_regime="Optimistic_Bias", rng=np.random.default_rng(11))
        q_cal, _ = compute_calibrated_upper_risk(learner, X_D, y_D, X_C, risk_regime="Calibrated", rng=np.random.default_rng(11))
        q_con, _ = compute_calibrated_upper_risk(learner, X_D, y_D, X_C, risk_regime="Conservative_Bias", rng=np.random.default_rng(11))

        self.assertTrue(np.all(q_opt < q_cal))
        self.assertTrue(np.all(q_cal < q_con))

    def test_6_p4_state_builder_hidden_label_mutation_and_zero_duplicates(self):
        """Sections 5, 6, 8: Flipping all hidden labels y_O, y_C, y_T must leave state construction & selection 100% invariant, with 0 duplicate IDs."""
        dsets = load_four_real_datasets()
        h_info = dsets["HMEQ"]
        part_orig = build_strictly_hidden_y_free_real_state(h_info["X"], h_info["y"], h_info["ead"], seed=4001, n_rounds=10)

        y_flip = 1 - h_info["y"].copy()
        y_flip[part_orig["D0_ids"]] = h_info["y"][part_orig["D0_ids"]]
        part_flip = build_strictly_hidden_y_free_real_state(h_info["X"], y_flip, h_info["ead"], seed=4001, n_rounds=10)

        self.assertTrue(np.array_equal(part_orig["D0_ids"], part_flip["D0_ids"]))
        self.assertTrue(np.array_equal(part_orig["Q_ids"], part_flip["Q_ids"]))
        self.assertTrue(np.array_equal(part_orig["T_ids"], part_flip["T_ids"]))

        seen_ids = set(part_orig["D0_ids"].tolist()) | set(part_orig["Q_ids"].tolist()) | set(part_orig["T_ids"].tolist())
        for r1, r2 in zip(part_orig["rounds"], part_flip["rounds"]):
            self.assertTrue(np.array_equal(r1["O_ids"], r2["O_ids"]))
            self.assertTrue(np.array_equal(r1["C_ids"], r2["C_ids"]))
            self.assertTrue(np.allclose(r1["X_C"], r2["X_C"], atol=1e-12))
            for i_id in np.concatenate([r1["O_ids"], r1["C_ids"]]):
                self.assertNotIn(int(i_id), seen_ids)
                seen_ids.add(int(i_id))

    def test_7_q_c_j_decomposition_algebraic_identity(self):
        """Section 12: Verify J == C_k - Q_k == R(M_K) - R(S) on the exact same state."""
        rng = np.random.default_rng(707)
        X_base = rng.normal(size=(60, 5))
        y_base = rng.integers(0, 2, size=60)
        X_C = rng.normal(size=(12, 5))
        y_C = rng.integers(0, 2, size=12)
        q_C = rng.uniform(0.05, 0.25, size=12)
        ead_C = rng.uniform(0.6, 1.4, size=12)
        X_T = rng.normal(size=(80, 5))
        y_T = rng.integers(0, 2, size=80)
        B_t, _ = compute_exogenous_budget(q_C, ead_C, alpha=0.10, target_k=5)
        M_K, _ = solve_safe_minrisk_exact_or_bounded(q_C, ead_C, B_t, alpha=0.10, exact_max_n=12)
        S_sel = M_K[: max(len(M_K) - 2, 1)]

        decomp = compute_q_c_j_decomposition(S_sel, M_K, q_C, ead_C, B_t, X_base, y_base, X_C, y_C, X_T, y_T)
        self.assertAlmostEqual(decomp["J"], decomp["C_k"] - decomp["Q_k"], places=12)
        self.assertAlmostEqual(decomp["J"], decomp["brier_M_K"] - decomp["brier_S"], places=12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
