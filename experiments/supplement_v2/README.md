# Frozen history-coverage intervention

This follow-up holds a reference decision problem fixed while changing the
distribution of 80 observed history rows. Four dataset/DGP cells each use 64
fresh simulation pairs: seeds 11001–11064, 12001–12064, 13001–13064,
14001–14064. The two arms share preprocessing, truth, labels, O/C/Q/T,
empirical q, numerical B, anchor and candidate family. These are 256 pairs,
not 512 independent replications or new external institutions.

The gap arm uses the 80 lowest feature-1 rows of a 240-row random reservoir.
The reference arm uses the first 80. Preprocessing and nuisance calibration
are fitted once on the reference arm. Consequently this isolates history
coverage in a fixed reference decision problem; it does not estimate the full
effect of retraining calibration, preprocessing and search after a shift.

Selections for both arms are locked before evaluating any hidden O/C/T
labels. The primary outcome is known-p expected Brier after O and S settle.
The finite-family oracle uses realized hidden training labels, so its headroom
is an offline diagnostic, not achievable deployable acquisition value.

All constants and 16 primary comparisons are frozen in
`configs/supplement_v2.json`. Each cell has four endpoints. Global Holm
p-values and Bonferroni simultaneous t intervals apply to this 16-test family;
normal-approximation assumptions and delta=0.0001 are explicit. No formal
worlds were added after looking at results. Other method variants are secondary.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/supplement_v2/run.py \
  --protocol configs/supplement_v2.json \
  --freeze results/supplement_v2/pre_run_freeze.json \
  --data-dir /your/local/data --output runs/coverage-001
python experiments/supplement_v2/verify.py \
  --output runs/coverage-001 --data-dir /your/local/data
```

The output directory must be new. The verifier accepts the archived compressed
results. It audits every family and choice, recomputes statistics independently,
and replays the first/last pair of every cell. All optimizer calls are recorded.
The simulated local indices are not source borrower identities. Original real
credit arrays are required locally and are not redistributed.
