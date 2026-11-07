# Fresh random-split confirmation of repaired P4

The active repaired pipeline runs all 10 unchanged methods on all four existing
real cohorts, seeds 21001–21016, for 10 rounds. The result has 6,400 state/method
rows and 640 trajectories. Feedback timing, budgets, preprocessing, empirical
risk model and method parameters are unchanged. These are fresh random splits
and streams of existing outcomes, not new independent institutions or verified
calendar-based forecasting cohorts.

The frozen primary outcome is seed-level online ALC-Brier. Three comparisons
per dataset (Safe-MinRisk, fixed epsilon, package Fisher-trace/cost adaptation)
form one family of 12 tests. The primary table has global Holm t-test p-values
and Bonferroni simultaneous t intervals, delta=0.0001. The inherited runner's
per-dataset Wilcoxon/bootstrap tables and round-10 outcomes are secondary.
The four-dataset average is descriptive; splits are not pooled as institutions.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/p4_confirmation_v1/run.py \
  --protocol configs/p4_confirmation_v1.json \
  --freeze results/p4_confirmation_v1/pre_run_freeze.json \
  --data-dir /your/local/data --output runs/p4-fresh-001
python experiments/p4_confirmation_v1/verify.py \
  --output runs/p4-fresh-001 --data-dir /your/local/data \
  --replay-output runs/p4-fresh-replay-001
```

Both output directories must be new. The verifier audits all release schedules,
first-round causality, shared budgets, duplicate acquisition, feasible envelope
and fixed-cardinality targets. It recomputes all primary statistics and replays
the first/last seeds for every dataset (800 rows), comparing all 41 non-timing
columns. Every logistic optimizer call is recorded.

Public results contain aggregate tables, seed-level metrics and checksums.

```bash
python experiments/p4_confirmation_v1/verify_summary.py results/p4_confirmation_v1
```

Individual true-credit acquisition/release identifiers and raw rows remain in
the local run directory. The strict trajectory verifier requires a full local
run; seed-level metrics and primary tables are sufficient to recheck primary
statistical calculations. Full deterministic replay requires local data.
