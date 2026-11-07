# supplement-v3: pre-outcome opportunity, added baselines, natural stream

Frozen protocol: `configs/supplement_v3.json`; source and protocol hashes in
`results/supplement_v3/pre_run_freeze.json` were fixed before any study seed was
run. Execution smoke tests used seeds 99001–99005 only. One re-freeze occurred
after a syntax error in `stats.py` crashed the first launch before any
computation; no constant changed.

| Study | States | Question |
|---|---|---|
| A. Opportunity decomposition | 128 controlled worlds (31001–31128); 4 Real-X cells × 64 pairs × {broad, gap} (41001–44064) | How much of the post-outcome family headroom is achievable before outcomes arrive? |
| B. Baselines (ex-ante endpoint) | same states | DC-PRISM vs Random-Feasible, Uncertainty, BAIT-Greedy, TargetVar-Greedy, Fixed-ceil, Full, Safe-MinRisk, Fisher-cost |
| C. Sensitivity | same states | One-at-a-time changes of quantile, quantity penalty, gate, curvature damping, tanh scale |
| D. Online replay | 4 cohorts × 32 fresh seeds (51001–51032) × {engineered, natural} stream × 10 rounds × 9 policies | Do the new baselines change the real-label conclusions; does the engineered stream drive them? |

**Ex-ante endpoint.** In A–C true probabilities are known. Hidden pending and
candidate training labels are resampled 128 times per state; test-label noise is
integrated exactly. `H_pre` is reported as a plug-in value (biased upward by
choosing the best set on noisy averages) and a two-fold cross-fitted value
(biased downward). `foresight_lb = H_post − H_pre_plugin` is therefore
conservative.

**Natural stream.** Same T/Q/history-window sizes and the same D0-fitted
preprocessing as the package builder, but D0 is the first 68 rows of the random
history window and each round's 14 ordinary acquisitions and 24 candidates are
consecutive rows of a random permutation of the stream. There is no covariate
ranking into safe, pending-aligned, complementary or tail groups.

**New baselines** (`baselines.py`, visible inputs only, same envelope):
Random-Feasible (random order, feasible packing); Uncertainty (largest p̂(1−p̂)
first); BAIT-Greedy (forward greedy on tr(F_pool (H_{D+O}+A_S)^{-1}), forward step
only); TargetVar-Greedy (forward greedy on DC-PRISM's own target-weighted
proxy, without its reference comparison, quantile, penalty or gate).

```bash
OPENBLAS_NUM_THREADS=1 python experiments/supplement_v3/run_offline.py --protocol configs/supplement_v3.json \
  --freeze results/supplement_v3/pre_run_freeze.json --data-dir /your/local/data --output runs/v3-offline
OPENBLAS_NUM_THREADS=1 python experiments/supplement_v3/run_online.py --protocol configs/supplement_v3.json \
  --freeze results/supplement_v3/pre_run_freeze.json --data-dir /your/local/data --output runs/v3-online
python experiments/supplement_v3/verify.py --results results/supplement_v3 [--data-dir /your/local/data]
python experiments/supplement_v3/figures.py --results results/supplement_v3 --out figures
```

`run_online.py` first checks that its loop reproduces the archived
p4-confirmation-v1 ALC values of the inherited policies (seeds 21001/21016,
max difference 1.4e-16) and refuses to run otherwise. The verifier recomputes
all 146 registered statistics independently, and with data replays 18 offline
states (198 rows) and 1,440 online round records exactly.

Public files contain simulated-state local indices and aggregate metrics only;
no credit rows or borrower identifiers. Wall time: offline 97 s, online 7 s on
22 CPU workers.
