# Method, solver and safety scope

DC-PRISM uses the visible history to fit a logistic model, incorporates expected
Fisher information from pending covariates, and scores candidate sets by a
target-weighted matrix-inverse reduction proxy. It compares feasible candidate
sets with a risk reference through empirical quantiles of common Laplace
parameter draws, a cardinality penalty and a fallback gate. This is a proxy-based
set search; it is not an exact theorem about downstream logistic Brier loss.

## Risk envelope

For conditionally independent Bernoulli losses with known exposures, define
`mu_U=sum(E*q)`, `V_U=sum(E^2*phi(q))`, where `phi(q)=q*(1-q)` for `q<=1/2`
and `1/4` otherwise. The implemented envelope is
`mu_U + sqrt(2*V_U*log(1/alpha)) + max(E)*log(1/alpha)/3`.

Conditional on a legally visible history for which every selected true probability
is at most its q, the upper mean and variance bounds support the envelope through
the one-sided sub-gamma moment bound
`log E exp(lambda*(L-mu)) <= V*lambda^2/[2*(1-lambda*M/3)]`.
Chernoff optimization gives the square-root-plus-linear threshold. The submitted
proof's claim that this threshold upper-bounds the positive root of the usual
rational Bernstein exponent is algebraically incorrect; use the moment bound
directly rather than that quadratic inversion. This is a proof repair, not a
numerical rescaling of the envelope.

If simultaneous coverage fails with probability at most delta, a per-round
union argument yields an `alpha+delta` bound. The implementation does not establish
that coverage premise in real data: its Wilson shift is in-sample, its draw
quantiles use an effective size of 8, and selected sets are adaptive. Empirical
coverage in controlled worlds is evidence within those generators, not a
distribution-free certificate. `Calibrated` is retained as a legacy regime name.

Exposures are D0-percentile-winsorized and normalized. Loss and safety outputs
refer to that proxy, not raw monetary default losses or regulatory capital.
Correlated-default World D is conditional on its latent shock for its conditional
probability diagnostics. The zero-shock reference is not the marginal integrated
probability over the shock distribution. Per-round risk control does not by
itself certify an entire ten-round trajectory.

## Search certificates and diagnostic references

Small pools use exact enumeration for the risk/count anchor. Larger pools use
stratified heuristic prefixes plus a cardinality upper bound. Closing that bound
certifies maximum count, not a globally minimal risk tie break at that count.
Status `CARDINALITY_CERTIFIED_RISK_HEURISTIC` makes this distinction explicit.

Fixed-k references are exact only for `n<=16`; larger references carry
`FEASIBLE_HEURISTIC`. The reference anchor supplies a feasible fallback. Quantity
and composition differences share a common state, so `J=C_k-Q_k` is an algebraic
identity even when reference optimality is heuristic. They remain offline
post-settlement counterfactual diagnostics, separate from delayed online Brier.
At k=K the reference is the same anchor for both loss and risk accounting.

## Baselines and attribution

`BAIT-Cost-Safe` denotes this package's static Fisher-trace/cost greedy adaptation;
it is not a certified faithful reproduction of published BAIT. `Orig-PA` denotes
the package's point-budget local adaptation, not the complete historical method.
`BoostedTreeTransferLearner` is a custom boosted stump ensemble, not XGBoost.

`Safe-PA-NoGate` simultaneously changes the objective, uncertainty treatment and
quantity penalty. `PRISM-ValueUCB` changes curvature weights as well as quantile
direction. These compare algorithm variants; they do not isolate one causal
component each. Empirical parameter-draw quantiles are not confidence bounds on
the real predictive benefit. Do not use significance of these comparisons as
proof of a uniquely identified Gate or pessimism effect.

## Statistical interpretation

The primary P4 metric averages ten round-level Brier values within each seed;
round 10 is secondary. Pairing occurs within dataset and seed. Bootstrap intervals
reported by the legacy routine are pointwise and conditional on this replay
design; Holm p-values are adjusted within each dataset/metric competitor family.
They do not provide a single simultaneous four-dataset/two-endpoint claim.
Seeds and rounds do not create independent institutions or new external cohorts.
