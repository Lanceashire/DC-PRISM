This directory preserves submitted artifacts as historical evidence.
`submitted_v2_2` is selected by file attributes, ZIP member times, changelog,
code differences and actual result coverage, not by download filename suffix.
Its original README and theory claims are retained verbatim for traceability;
they are not the current audit conclusions. Read `docs/CODE_AUDIT.zh.md` first.

The original source is not patched here. The archived fast runner restores old
tables after reduced runs, and the archived P4 Brier uses current selected labels
before their queue release time. Do not treat successful archived fast execution
as independent regeneration. Use the active `src/run_all.py` for corrected runs.

Only aggregate real-data outputs and synthetic experiment records are included.
The raw NPZ, per-applicant real-data trajectories and ID trace are kept local.
