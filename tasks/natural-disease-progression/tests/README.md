# Verifier and regression tests

`test.sh` is the Harbor entry point. It runs `main_verifier.py`, which executes
untrusted `infer.sh` as uid 10001 with only `public/test` mounted, kills remaining
uid-10001 processes, and seals `predictions.parquet` with a run ID and content
hashes. `grader.py` is a separate labels-only service. It mounts exactly
`trusted/test`, the fixed train-only normalization parquet, and the same public
test input; it rejects a validation manifest or non-test boundaries. The grader
writes its result through the sealed volume, then main copies `report.json` and
`reward.json` to `/logs/verifier`.

Verifier main includes the pinned runtime dependencies used by the task image.
Package presence does not determine architecture compliance: the grader remains
an objective scorer, while the submitted neural model, training artifacts, and
actual `infer.sh` path are reviewed separately. The grader executes only trusted
scoring code against the sealed prediction file.

The Harbor reward is the disease-first macro time-averaged AUROC. Valid
predictions therefore receive a value in `(0,1]` that changes with model quality.
Missing, malformed, timed-out, or failed submissions receive zero. Trusted data,
mount, and scorer failures are infrastructure errors and produce no reward.
There is no trajectory field or sampler budget.

`test_risk_protocol.py` covers split reproducibility, competing death and censoring, feature leakage, direct-grid enforcement, probability constraints, disease-first/time-first aggregation, single-class handling, frozen denominator diagnostics, fixed-cohort scoring, and risk-only output. `test_longitudinal_v2.py` locks keyed label alignment, missing-value handling, event-time gaps, and diagnosis-only tensor compaction. `smoke_verifier.py` builds a tiny MIMIC-shaped fixture through the complete two-phase path. Patient bootstrap remains optional and disabled by default.

Unit tests are deliberately separate from the Harbor score path:

```bash
docker build -f tests/Dockerfile --target unit-tests -t natural-disease-progression-tests .
docker run --rm natural-disease-progression-tests
```

For an end-to-end replay, set `TASK_PRIVATE_DIR` to the private tree produced by
`stage_runtime_data.py`, place a legal package at `/workspace/submission` through
Harbor, and start both services from `tests/docker-compose.yaml`. The main service
cannot mount `trusted/test`, its parent, `working.duckdb`, or raw MIMIC.
