#!/usr/bin/env python3
"""
Master Reproducibility Entrypoint for DC-PRISM Final Package (DASFAA 2027)
Supports:
  - python3 src/run_all.py --fast   (Fast verification mode)
  - python3 src/run_all.py          (Full 128-world P2 + 25-seed P3 + 16-seed P4 reproduction)
Note: Runtime depends on hardware; fast mode is intended for verification rather than full statistical reproduction.
"""
import os
import sys
import time
import shutil
import tempfile
import argparse
import subprocess

SRC_DIR = os.path.abspath(os.path.dirname(__file__))
PKG_ROOT = os.path.abspath(os.path.join(SRC_DIR, ".."))


def run_step(desc, cmd):
    print(f"\n============================================================")
    print(f"[DC-PRISM Master Runner] {desc}")
    print(f"============================================================")
    t0 = time.time()
    subprocess.run(cmd, cwd=PKG_ROOT, check=True)
    print(f"[Done in {time.time() - t0:.2f}s] {desc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Run fast verification mode (preserves frozen full statistical tables)")
    args = parser.parse_args()

    t_start = time.time()
    run_step("Step 1/5: Unit & Protocol Integrity Tests (7 Tests)", [sys.executable, os.path.join("tests", "test_prism_integrity.py")])
    run_step("Step 2/5: Data Provenance & Cohort/Feature/Hash Manifests", [sys.executable, os.path.join("scripts", "prepare_lendingclub.py")])

    if args.fast:
        # In fast verification mode, back up the frozen full statistical tables, execute fast verification
        # of P0-P4 to verify every script and audit check end-to-end, then restore the full statistical tables
        # before regenerating paper tables and figures.
        backup_dirs = ["p0_exact", "p1_mechanism", "p2_confirmation", "p3_realx", "p4_real_labels", "ablations", "statistics", "results"]
        with tempfile.TemporaryDirectory() as tmp_bak:
            for d in backup_dirs:
                src_d = os.path.join(PKG_ROOT, d)
                if os.path.exists(src_d):
                    shutil.copytree(src_d, os.path.join(tmp_bak, d))
            run_step("Step 3/5: Fast Verification of P0 Exact, P1 Pilot, P2 Confirmation, Q/C/J & BRP/HRR", [sys.executable, os.path.join("src", "run_p0_p1_p2.py"), "--fast"])
            run_step("Step 4/5: Fast Verification of P3 Real-X (Finite-Family Oracle) & P4 Option-A Real Labels", [sys.executable, os.path.join("src", "run_p3_p4.py"), "--fast"])
            for d in backup_dirs:
                bak_d = os.path.join(tmp_bak, d)
                dst_d = os.path.join(PKG_ROOT, d)
                if os.path.exists(bak_d):
                    if os.path.exists(dst_d):
                        shutil.rmtree(dst_d)
                    shutil.copytree(bak_d, dst_d)
    else:
        run_step("Step 3/5: Full P0 Exact, P1 Pilot, P2 Confirmation (128 Worlds), Q/C/J & BRP/HRR", [sys.executable, os.path.join("src", "run_p0_p1_p2.py")])
        run_step("Step 4/5: Full P3 Real-X (25 Seeds) & P4 Option-A Real Labels (16 Seeds x 4 Datasets)", [sys.executable, os.path.join("src", "run_p3_p4.py")])

    run_step("Step 5/5: Figures, Paper Tables, and Audit Manifests", [sys.executable, os.path.join("src", "generate_figures_tables_and_docs.py")])
    print(f"\n[SUCCESS] DC-PRISM Suite completed in {time.time() - t_start:.2f}s.")
