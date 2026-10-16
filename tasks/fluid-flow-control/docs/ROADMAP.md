# Remaining research and acceptance work

Current task: Research v2, 39 cases, unchanged ~454 GB source data;
`hydrogym-policy/v1` delivery and `hydrogym-physical-metrics/v1` reporting.

## Ready in this PR

- Frozen public-HF download manifest, offline/local reuse and integrity checks.
- All 39 executable native profiles and fixed physical primary metrics.
- One-package resident inference and independent full-episode evaluator.
- Native Harbor separate verifier producing labeled numeric physical metrics.
- Curated runnable specialist source and 39 completed independent baseline results.
- A shared PPO training/resume/publication path across native backends.
- Regression tests, clean-image construction and integration evidence in VALIDATION.md.

## Explicit remaining work

1. Done (2026-09-19): Cylinder_3D_Re3900's 4,000-step specialist run and its paired
   independent evaluations finished; the result is appended to the verified table.
2. Measure a competitive **jointly trained** checkpoint across all 39 cases.
   The per-case specialist table is not evidence of this.
3. Run formal AutoResearch sessions and compare successive shared packages.
4. Extend training/repeat seeds where needed for quality; single-initialization
   evaluation does not establish statistical confidence or full-paper matching.

No new data selection, private split, reference-weight bank, mixed-unit aggregate
or task-specific agent harness is introduced by this PR. A future aggregate
requires an explicit versioned metric decision; current Harbor output is a vector
of the user's fixed physical metrics. Session duration is a runner concern.
