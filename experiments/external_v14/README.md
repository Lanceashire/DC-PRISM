# External v14: frozen vision/text encoders with nonlinear heads

This extends the three-oracle full-curve design of supplement v13 to two binary tasks:
CIFAR-10 cat/dog with MobileNetV3-Small, and AG News Business/Sci-Tech with MiniLM-L6.
The pretrained encoders are frozen. A 32-hidden-unit ReLU head is fitted from a shared
initialization for 60 full-batch Adam steps for every actual and hypothetical training set.
This is not end-to-end backbone fine-tuning or a full multiclass CDALBench reproduction.

Protocol: 12 trajectory seeds per task; 30 steps; 40 fresh candidates and exactly 10 labels
per step; 32 Bernoulli outcome-model replicates; initial history 68, final history 368.
Strategies: Random, Uncertainty and CoreSet. All three oracles use the same greedy search
and learner budget. PCA and the outcome GBDT are fitted on disjoint auxiliary data only.
Official test records do not enter training. oracle-test intentionally selects using test
labels, as required by the benchmark-style comparison. Other methods cannot use them.

The local freeze preceded formal launch. The GitHub archive was created after launch,
before inspection of any formal endpoints; it is NOT a prospective public preregistration.
The run uses fixed seeds without endpoint peeking or adaptive expansion.

## Reproduce

Use Python 3.12 and a CUDA GPU. The audited host was an RTX 5060 Laptop GPU (8 GiB),
with torch 2.11.0+cu128 and torchvision 0.26.0+cu128. Install these from the official
PyTorch cu128 wheel index, then the pinned analysis/model dependencies in requirements.txt.
Model and dataset revisions are fixed in assets_manifest.json.

From the repository root:

```bash
python experiments/external_v14/prepare.py
python experiments/external_v14/check_search.py
python experiments/external_v14/gpu_head.py
python experiments/external_v14/run.py --protocol experiments/external_v14/protocol.json --output runs/external_v14_smoke --smoke
python experiments/external_v14/replication_freeze.py --directory runs/external_v14_config
python experiments/external_v14/run.py --protocol runs/external_v14_config/protocol.json --output runs/external_v14
python experiments/external_v14/stats.py --protocol runs/external_v14_config/protocol.json --output runs/external_v14
```

The archived freeze verifies original source and feature hashes. Feature regeneration can
vary across GPU/library versions; prepare.py also refreshes extraction timing metadata.
The replication command preserves the original freeze and records actual regenerated
feature/metadata hashes in a separate configuration directory. Report this as a replication
rather than claiming a byte-identical rerun. Raw source records, pretrained weights and
feature caches are not in this repository.

## Interpretation

Primary families and hypotheses H1-H3 are in protocol.json. Simultaneous intervals use
paired trajectory-seed t statistics with Bonferroni correction; Holm adjusted p-values are
also saved. H3 requires an upper confidence bound at or below 0.0005 for at least one
strategy against ex-ante, not merely a non-significant shortfall.

Contrasts compare methods on their own acquired histories. Their arithmetic decomposition
is exact, but the outcome-luck contrast depends on the estimated outcome model and greedy
search, and includes subsequent trajectory changes. It is not a certified irreducible
population bound. Confidence intervals describe random trajectory seeds conditional on
these datasets, representations, splits and learner; there are only two binary tasks.
