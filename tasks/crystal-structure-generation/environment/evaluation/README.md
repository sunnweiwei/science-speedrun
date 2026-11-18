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

The grader first relaxes a separate copy of each submitted crystal with the frozen UMA
`uma-s-1` (`omat`), BFGSLineSearch and FrechetCellFilter, up to 200 steps at 0.02 eV/Å and
300 seconds per crystal. It retains finite unconverged iterates and uses the original
candidate when refinement fails. It then scores the resulting geometry with ORB-v3, MACE-MP
and UMA. Mean energy above hull defines stable (≤0 eV/atom) and metastable (0–0.1 eV/atom)
bins, with stable structures excluded from the metastable bin. The primary score (`score` and `reward`)
is `(SUN + 0.5 * MetaSUN) / N_expected`, the equal mean of strict SUN yield `SUN / N_expected` and
stable-or-metastable SUN yield `(SUN + MetaSUN) / N_expected` relative to LeMat-Bulk.

For equal compositions, CrystalGen applies MatterGen's v1.0.3 ordered-disordered matcher.
The additional dilute-parent guard identifies a known lattice decorated with a substituent
occupying ≤25% of the host-plus-guest sites. It requires relative radius difference ≤0.3 and
electronegativity difference ≤1.0. Supercells, symmetry copies and dilute variants of a known
compound such as GdDy7Cu8 from DyCu are covered by these comparisons.

The optional `--as-submitted` evaluation scores the raw lattice and coordinates. It also
measures displacement from the raw geometry under all three potentials on 250 prespecified
slots (50 steps, 0.02 eV/Å), with coverage and a 95% structure-bootstrap interval.
This result is recorded separately from the primary UMA-200 reward.

The official report uses `crystalgen-report/v4-uma200-hull-aligned`. The as-submitted sensitivity uses `crystalgen-report/v3-submitted-geometry`.

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
  --as-submitted
```

This flag scores the supplied coordinates directly under an explicitly identified sensitivity
protocol. On raw inputs it measures displacement in up to 250 prespecified slots. The trusted
grader supports the same option for archived comparisons. Normal `score.sh` takes raw generated
outputs and performs the fixed UMA-200 relaxation. Optional 50-step displacements
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
