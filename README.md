# DC-PRISM — label-acquisition experiments and oracle-gap evaluation

**2026-11-07 consolidated research snapshot / 研究工作汇总。** This repository now includes the local Claude and Codex work through supplement v11, both bilingual manuscript branches, saved experimental rows and trajectories, frozen protocols, code, and review records. The prior public head was `a35fc466b4cfdd4783310a3da39f1ea6e88edd8b` (supplement v1); the local Claude checkout was based on `f8a71bb` plus two uncommitted generator changes. Local historical commit IDs are recorded as provenance; the consolidated public commit does not recreate those historical Git objects.

## Start here / 阅读入口

- [中文完整工作说明](docs/WORK_SUMMARY_20261107.zh.md) · [English work summary](docs/WORK_SUMMARY_20261107.en.md)
- [Manuscript versions, PDF links and scientific status / 论文版本与状态](paper/README.md)
- [Current review and remaining problems / 当前审查与待解决问题](docs/REVIEW_STATUS_20261107.md)
- [Experiment manifest index](provenance/experiment_status_index_20261107.json) · [Unique run evidence and duplicate mapping](provenance/runtime_output_index_20261107.json)
- [Publication file hashes](provenance/publication_manifest_20261107.json) · [Local verification](provenance/publication_checks_20261107.json)

研究主题由风险预算、延迟选择性反馈下的 DC-PRISM 策略研究，发展为 oracle 差距的评估研究：区分隐藏结果的事后信息价值、评估集复用，以及指定事前选择器的收益。完整实验同时保留正向、负向和未确定的结果。固定族的完全信息价值与贪心基准的残差具有不同含义；当前证据尚不能把所有贪心残差解释成不可消除的偶然性下限。

The central question is how much of an acquisition oracle's lead represents attainable improvement. For a fixed feasible family and information set, perfect-information value is distinct from test-set reuse and practical selector performance. The project includes a conservative policy, DC-PRISM, whose observed performance is heterogeneous. This snapshot is a research archive and working manuscript collection; the scientific status is described in the review notes.

## Results / 结果

| Stage | Evidence and purpose |
|---|---|
| repair-v1, supplement-v1 | Original implementation audit, protocol repair and frozen follow-up; [audit](docs/CODE_AUDIT.zh.md), [v1 report](docs/SUPPLEMENT_V1_RESULTS.zh.md) |
| supplement-v2, P4 confirmation | Shared-family coverage diagnostic and fresh delayed-replay confirmation; [report](docs/CONTINUATION_RESULTS.zh.md) |
| supplement-v3–v4 | Additional baselines, pre/post-outcome opportunity, natural replay and proxy/parameter diagnosis; [v3](docs/SUPPLEMENT_V3_RESULTS.zh.md), [v4](docs/SUPPLEMENT_V4_RESULTS.zh.md) |
| supplement-v5 | Candidate-family enlargement, offline and large fresh replay; [results](results/supplement_v5), [protocol](configs/supplement_v5.json). The archived H replay includes 36,864 trajectories / 368,640 round rows. |
| supplement-v6 | Probability-model recovery and recorded-outcome diagnostics; [results](results/supplement_v6) |
| supplement-v7 | Learner dependence, calibration, replay and analytic approximations; [results](results/supplement_v7) |
| supplement-v8 | Greedy benchmark accounting: reuse, residual and cross-fitted gain; [results](results/supplement_v8) |
| supplement-v9 | Eight-cohort noise-gradient diagnostics; [results](results/supplement_v9) |
| supplement-v10 | MLP learner and budget sensitivity; [results](results/supplement_v10) |
| supplement-v11 | Distribution-free sample inequalities, 12-candidate exact search, probability sensitivity, label flips, pool size and exact cardinality reference; [results](results/supplement_v11) |
| Codex nested diagnostic | [Interrupted 38/128-state study](archive/codex_nested_opportunity_v1_incomplete), stored separately with `INCOMPLETE_INTERRUPTED` status |

Later corrections supersede broad interpretations in earlier reports. In particular, the old independent-normal construction is not a universal upper bound; empirical inequality checks do not establish population confidence coverage; cross-fitted greedy gains are the performance of that selector, not the optimal deployable value. See the current review before quoting historical headlines.

## Code, data and reproduction / 代码、数据与复核

- `src/`, `tests/`: repaired acquisition/replay implementation and regression checks.
- `experiments/`, `configs/`: frozen study runners, inference, plotting and configuration.
- `results/`: committed saved evidence, with study-specific manifests and freezes.
- `paper/`: latest Claude draft, separate Codex-reviewed snapshot, historical versions and reviews.
- `archive/`: submitted originals, distinct historical work, incomplete/stopped records.
- `deliverables/`: dated bilingual PDFs, source packages and handoff notes.

Use Python 3.12 and the appropriate study instructions. The core pinned dependencies are in `requirements-lock.txt`; study-specific learners may require additional dependencies visible in their imports. The published integrity check below uses only the Python standard library:

```bash
python scripts/verify_publication_20261107.py
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
python src/run_all.py --suite checks
```

Detailed execution instructions: [v1](experiments/supplement_v1/README.md), [v2](experiments/supplement_v2/README.md), [P4](experiments/p4_confirmation_v1/README.md), [v3](experiments/supplement_v3/README.md), [v4](experiments/supplement_v4/README.md), [v5](experiments/supplement_v5/README.md). Later runners expose `--help`; use their frozen `configs/` and disjoint new output directories. Code/row verification does not imply that all experiments were rerun at publication time.

Original third-party input datasets, local environments and account material are outside this publication. Saved derived experimental evidence is included. Data sources and acquisition boundaries are in [data/README.md](data/README.md) and `scripts/prepare_*.py`; the exposure score is an operational constraint unless its additional probability assumptions hold. The earlier repository introduction remains [archived here](docs/README_BEFORE_CONSOLIDATION_20261107.zh.md).
