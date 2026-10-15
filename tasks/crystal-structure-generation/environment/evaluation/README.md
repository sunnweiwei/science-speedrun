# Exact development evaluation

Development uses the same pinned scorer, model identifiers, hull data, reference
set, relaxation settings, and thresholds as final tests. It is not a proxy.

Generate a visible request batch, then run:

```bash
bash /workspace/evaluation/score.sh \
  /workspace/evaluation/examples/input \
  /workspace/research/output \
  /workspace/research/score
```

The command first relaxes each candidate with frozen UMA `uma-s-1` (`omat`),
then runs the unchanged ORB + MACE + UMA LeMat scorer on the relaxed geometry.
It writes `reward.json`, `report.json` (`crystalgen-report/v2`), and a
`relaxation/` directory containing the exported CIFs and per-sample diagnostics.
Use a fresh score directory for each attempt; existing relaxation output is never
silently overwritten or reused. The headline key is
`unconditional_sun_metasun`, the same SUN + MetaSUN ordering used by the public
leaderboard. Only unconditional quality metrics are emitted. The supplied
example requests 100 unconditional structures; agents can also create a
2,500-sample development request for a final-sized comparison.

The evaluator reads the prepared scorer assets at `/workspace/data/scorer-assets` (also reachable
as `/scorer-assets`), read-only. Those files are tools
for evaluation only. They may not be copied into the submission, used as model
initialization, or converted into a training/distillation dataset.

Evaluation inputs here are visible development requests. Final generation uses
a fresh operator-held seed and 2,500 samples, with the same public contract and
scorer. There is no hidden conditional target grid or test-structure answer set.
Final verification also seals outputs before invoking this shared scoring
implementation. Small development batches have higher statistical uncertainty;
uniqueness also depends on batch size.
Fréchet distance requires at least two valid structures. The checked-in
`lemat-small-batch.patch` reports this diagnostic as unavailable below that
size instead of passing undefined covariance into LAPACK. It does not change
SUN/MetaSUN or metric calculations for batches with two or more valid structures.

## Relaxation alone

```bash
bash /workspace/evaluation/relax.sh \
  /workspace/evaluation/examples/input \
  /workspace/research/output \
  /workspace/research/relaxation
```

Inspect `relaxation/generated/` (CIFs and manifest), `summary.json`, and
`samples/` (input/output hashes, convergence, steps, energies and timings).
The script uses the visible CUDA GPUs, up to eight workers. Set
`CRYSTALGEN_LEMAT_GPU_WORKERS=1` to use one worker for a small development check.
It runs offline using the same prepared scorer assets; no new
credentials or assets are needed after setup.

The fixed recipe is BFGSLineSearch with FrechetCellFilter, optimizing atom
positions and the cell while keeping species fixed; force threshold 0.02 eV/Å,
at most 200 steps and 300 seconds per candidate. A failed refinement preserves
the original CIF. A finite but unconverged final iterate is retained and flagged.
Neither case drops a requested sample. CUDA, model-loading or filesystem failures
stop the run rather than being reported as poor model quality.

For an explicit development diagnostic on **already-relaxed** outputs:

```bash
/opt/lemat/.venv/bin/python /workspace/evaluation/score.py \
  --input /workspace/evaluation/examples/input \
  --generated /workspace/research/relaxation/generated \
  --output /workspace/research/relaxed-score \
  --skip-relaxation
```

This flag skips only the task-side pre-relaxation and labels the report as a
development diagnostic. It is not exposed by the final grader. Normal `score.sh`
always performs relaxation, so pass it raw generated outputs. The unchanged
LeMat scorer also computes its separate 50-step relaxation diagnostics; those
do not replace the incoming-geometry energies used for headline SUN/MetaSUN.

In final tests the grader owns this fixed step. Agents may optimize the generator
and its self-contained sampling/geometry code, but cannot change final relaxer
settings or access the scoring potentials during isolated generation. Use this
helper to study what survives the same fixed postprocessing during development;
do not copy its pretrained weights or results into training data.

The task applies the small audited patch in `runtime/lemat-eight-gpu.patch` to
the pinned scorer. It assigns each of eight upstream preprocessing workers to a
different visible GPU while leaving one global benchmark batch, thresholds,
models, and metric code unchanged. Keeping the batch global matters because
uniqueness is a batch-level metric; independently scoring eight shards would
change the answer. The patch was checked for score parity against a one-worker run.
