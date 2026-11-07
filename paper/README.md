# Manuscripts and version status / 中英文论文版本

| Version | English | 中文 | Status |
|---|---|---|---|
| Latest local Claude writing, 2026-11-06 | [PDF](latest_claude_draft_20261106/en/main.pdf) · [source](latest_claude_draft_20261106/en/main.tex) | [PDF](latest_claude_draft_20261106/zh/main.pdf) · [source](latest_claude_draft_20261106/zh/main.tex) | Later than Codex review; headline and framework claims require review |
| Codex reviewed snapshot, 2026-11-06 | [PDF](codex_reviewed_20261106/en/main.pdf) · [source](codex_reviewed_20261106/en/main.tex) | [PDF](codex_reviewed_20261106/zh/main.pdf) · [source](codex_reviewed_20261106/zh/main.tex) | Conservative revision with explicit limits; residual caption inconsistencies remain |

`history/` preserves earlier Claude manuscript versions, reviews, responses, the pre-review backup and the separate v3 Codex copies. `review_records_20261106/` contains Codex audit records and editing/plotting scripts. Dated final-delivery files from the chat are in [`../deliverables/`](../deliverables/); these can predate the latest local writing. PDFs are working research drafts.

The latest Claude paper has the title “The Oracle Gap Has a Floor: Separating Luck from Headroom in Evaluating Label Acquisition.” The Codex review snapshot has the earlier subtitle “Separating Aleatoric and Deployable Headroom.” The source files and figures accompany their own PDF; use each version's complete language directory when compiling. LLNCS class and bibliography style files are retained with their original notices. Compilation used Tectonic and locally available CJK fonts; readers need a compatible TeX toolchain.

The two uncommitted generator changes in `experiments/paper_tables.py` and `paper_figures.py` were included at publication. They remove the post-hoc per-state maximum from cross-fitted-greedy summaries and revise the diagnostic figure. They must not be used to overwrite the separate Codex snapshot's two-panel diagnostic without updating its caption. No claim of anonymous submission compliance or acceptance is made.

Before quoting a percentage as an irreducible floor, read the [current scientific review](../docs/REVIEW_STATUS_20261107.md). The current archive deliberately preserves research history, including superseded interpretations.
