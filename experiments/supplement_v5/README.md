# supplement-v5: response to the strict review

Frozen protocol `configs/supplement_v5.json`; source/protocol hashes in
`results/supplement_v5/pre_run_freeze.json`, committed (f4daaf3) before any
study seed of F, G or H was executed. Smoke tests used seeds 99001–99002 only.
Study E re-uses the 640 supplement-v3 states and their exact label resamples
by design (diagnostic); its `v3_base` rows reproduce v3 H_post/H_pre exactly.

| Study | Review item | States | Question |
|---|---|---|---|
| E | M2 | 640 v3 states, family + 64 random feasible sets | Does the foresight share depend on family size or structure? |
| F | M3, M6, M7 | 128 controlled (61001–) + 4 Real-X cells × 64 pairs × {broad, gap} | Does the conservative layer add value over the point proxy on the same family, or over the greedy set? Exact exceedance probability; selection time. |
| G | M4, M7 | 12 generator cells × 64 worlds (70001–) | Do controlled conclusions survive pending overlap × coupling changes, no tail regime, 48-candidate pools, K=12? |
| H | M5, M6 | 4 cohorts × 2 streams × 512 seeds (81001–) × 9 policies | Power-driven replay; AUC, log-loss, exploration loss, envelope exceedance. |

```bash
OPENBLAS_NUM_THREADS=1 python experiments/supplement_v5/run_offline.py --protocol configs/supplement_v5.json \
  --freeze results/supplement_v5/pre_run_freeze.json --data-dir /your/local/data --output runs/v5-F --study F   # also E, G
OPENBLAS_NUM_THREADS=1 python experiments/supplement_v5/run_online.py --protocol configs/supplement_v5.json \
  --freeze results/supplement_v5/pre_run_freeze.json --data-dir /your/local/data --output runs/v5-H
python experiments/supplement_v5/figures.py --results results/supplement_v5 --out figures
```

`run_online.py` first reproduces archived v3 ALC-Brier for all nine policies on
seeds 51001/51016 (both streams; max difference 9.7e-17) and refuses to run otherwise.
Wall time on 22 CPU workers: E 352 s, F 245 s, G 327 s, H 116 s. The first E
launch was interrupted by a power loss (`runs/v5-E-interrupted-poweroff`, no
output) and re-run unchanged. Per-round H trajectories (18.6 MB) stay in local
`runs/v5-H/`; the per-seed metrics committed here suffice for every statistic.
