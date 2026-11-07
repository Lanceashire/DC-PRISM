# supplement-v4: localising the pre-outcome selection gap

Diagnostic on the 640 supplement-v3 states with identical label resamples
(pairing with archived v3 `H_pre` and anchor loss verified to 2.8e-16 inside the
run). Protocol `configs/supplement_v4.json`, frozen in
`results/supplement_v4/pre_run_freeze.json`; smoke tests used seeds 99001–99002.

On the same finite family F, the paper's target-weighted proxy V_theta(S) ranks
sets with theta_hat (visible-history fit, deployable) and theta_pop (soft-label
KL projection of the true p onto the logistic class; equals the true parameter
in logistic cells; not deployable). Registered endpoints per group:

- `parameter_effect` = gain(argmax V_theta_pop) − gain(argmax V_theta_hat)
- `approximation_gap` = H_pre_crossfit − gain(argmax V_theta_pop)

18 tests, Bonferroni simultaneous t, Holm, material delta 5e-4.

```bash
OPENBLAS_NUM_THREADS=1 python experiments/supplement_v4/run.py --protocol configs/supplement_v4.json \
  --freeze results/supplement_v4/pre_run_freeze.json --data-dir /your/local/data --output runs/v4
```
Wall time 92 s on 22 workers.
