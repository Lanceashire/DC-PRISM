# Supplement v1

This study repairs the known fixed-epsilon defect, tests isolated rescoring
components, and probes unmodified real covariates with known artificial truth.
The study is a fixed-size diagnostic supplement, not a claim of adequate power,
a new external cohort validation, or a replacement for ten-round online P4.

The protocol in `configs/supplement_v1.json` was frozen before the formal run.
Original seeds 2001–2128 are repair reanalysis only. New simulation seeds are
6001–6128 (controlled), and 7001–7064, 7101–7164, 7201–7264, 7301–7364 (bridge).
No additional looks or outcome-dependent seed additions are used.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/supplement_v1/run.py \
  --protocol configs/supplement_v1.json --data-dir /your/local/data \
  --output runs/supplement-001
python experiments/supplement_v1/verify.py \
  --output runs/supplement-001 --data-dir /your/local/data
```

The verifier also accepts the compressed CSVs in `results/supplement_v1`.
It recomputes all primary mean differences and simultaneous intervals and replays
the first/last world in each cohort. The sampled optimizer audit is not an audit
of every fit.

Policies `PpQqLlGg` represent pending rescoring (p), lower versus upper parameter
draw quantile (q), quantity penalty (l), and a positive-surplus threshold (g).
The full common-family policy is `P1Q1L1G1`. All variants share q, B, draws and
a family generated without O. Quantile comparisons use identical conservative
curvature weights. Pending comparisons isolate rescoring, not the entire search
effect of the original pending-aware algorithm. `Package-DC-PRISM` separately
retains the active package's search as a secondary comparator.

`Fixed-ceil` maximizes point-value within the common family at ceil(.75*K).
It is a baseline comparison, not a one-switch causal attribution of cardinality:
its objective differs from the full policy. `Fixed-round` is secondary.
The original-world repair changes only fallback feasibility and retains round(.75*K).
It does not claim an optimal fixed-k information selector.

All methods lock choices before the evaluation service consumes hidden O/C/T
labels. Each candidate-family subset is fitted on D+O+S with the same learner;
known-p expected Brier and the family oracle are offline diagnostics only.
Candidate indices refer to synthetic simulation roles, not public borrower IDs.
Raw real-credit arrays and row identities are not redistributed.

The main family consists of 35 comparisons (5 cohorts x 7 contrasts), with global
Holm two-sided paired t p-values and Bonferroni simultaneous t intervals.
Delta=0.0001 is fixed. T intervals use a world-level normal approximation.
Seeds are independent simulations conditional on reused real X; they do not
create independent institutions. The bridge changes covariate distribution,
history coverage and DGP together, so it cannot identify a unique transfer-failure
cause. Negative and indeterminate comparisons are retained.
