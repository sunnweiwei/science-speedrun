# Operator-side replay of the official-weight reference (maintainers)

Moved out of the agent image from environment/references/README.md: commands and resources for the
frozen official-weight / clean-MIST comparison, which the agent image cannot run.

## From `environment/references/README.md`

## Compute and runnable replay

These are **operator-side** commands for the frozen official-weight/MIST
comparison. The agent image now contains native GLACIER source/runtime and
train/validation-only MAGMa supervision for the scratch recipe in
`/workspace/baseline/starter/glacier/README.md`. It does not contain reference weights or
the external MIST runtime. These reproduction commands do not authorize
pretrained initialization. The completed scratch score is recorded above;
matched-H100 training/replay remains a separate acceptance item.

The construction replay used eight parallel P100 shards. The initial large
feature-cache jobs produced usable predictions for roughly 94--98% of every
shard before a 64 GiB host-memory cgroup OOM; answer-free set-difference jobs
completed the remainder with batch size 2, two workers, a 64-entry feature
cache, prefetch factor 1, and about 5.1--5.3 GiB process RSS. Total construction
cost was approximately 33.1 P100 GPU-hours including retained partial
work, diagnostics, and one cancelled insurance retry. This is inference, not
GLACIER training. A single-H100 six-hour replay has not been directly timed and
must not be inferred from the parallel P100 wall clock.

The verifier's six-hour envelope includes scoring: `main_verifier.py` defaults
to a five-hour model-inference timeout, followed by up to one hour waiting for
the trusted grader. Full package replay must be measured against both limits.

Prepare and verify the immutable operator-side assets:

```bash
bash environment/references/scripts/prepare_glacier_reference_assets.sh ASSET_ROOT
(cd ASSET_ROOT/ms-pred && uv sync --extra cu124)
```

The formal runner requires the independently trained clean MIST weight and
checks its digest rather than silently substituting the weaker released weight:

```bash
export MASS_SPEC_GLACIER_PYTHON=ASSET_ROOT/ms-pred/.venv/bin/python
export MASS_SPEC_MIST_CHECKPOINT=/absolute/path/to/clean/spectra_encoder.pt
bash environment/references/scripts/run_glacier_reference.sh \
  INPUT_ROOT OUTPUT_ROOT ASSET_ROOT
```

The runner uses `MASS_SPEC_GLACIER_PYTHON` for both task adapters by default;
set `MASS_SPEC_REFERENCE_PYTHON` only if the MIST branch needs a separate
compatible interpreter.

`run_glacier_reference.sh` defaults to H100-oriented batch size 16, two loader
workers, feature-cache budget 64, and prefetch factor 1. All are overridable by
`MASS_SPEC_GLACIER_*` variables. The runner consumes only public contract input
and emits `OUTPUT_ROOT/rankings.jsonl`, one complete candidate permutation per
query. `resume_glacier_candidates.py` and the scorer's `--exclude-ranking`
and `--query-id-file` options support answer-free recovery and query-disjoint
parallel scoring; `merge_partial_rankings.py` rejects overlap, unknown queries,
or incomplete permutations.
