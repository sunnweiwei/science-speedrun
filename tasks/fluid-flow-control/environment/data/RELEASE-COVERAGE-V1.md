# HydroGym public coverage release v1

The current resource membership is **hydrogym-public-coverage-v1**, frozen
2026-09-07: **40 cases (21 two-dimensional, 19 three-dimensional), 576 files,
486,282,110,078 bytes (486.282 GB)**. The hard source cap is 500,000,000,000 bytes.
It expands and preserves all 28 cases in the first tested speed-v1 release.

The [manifest](release.json), [reviewed selection](selection-coverage-v1.json)
and [measured evidence](cost-evidence-coverage-v1.json) are authoritative.
The [complete cost/decision table](COST-RESULTS-20260907.md) lists every included
case and all 22 excluded complete resources. This is our practical cost/value
selection, not the complete paper benchmark or a proof of global optimality
across every possible solver configuration.

Execution status is **data-release-not-cfd-certified**. The resource/cost audit
is finished; full physical evaluator acceptance, a reviewed executable
`suite.json`, non-MAIA policy adapters and competitive baseline scores are not.
No model score or completed full-suite episode replay is claimed here.

## Coverage

| Family | Cases | Source GB | H100s per measured environment | Warm wall s/control |
| --- | ---: | ---: | --- | --- |
| 2D: cavity, circular/square cylinders, pinball, gust/no-gust wings, Kolmogorov, coarse/fine nozzles | 21 | 1.360 | 1 | 0.092–3.455 |
| 3D spectral channel | 1 | 0.009 | 1 | 0.104 |
| 3D pinball Re30/75/100/150 | 4 | 33.369 | 1 | 10.126–15.947 |
| 3D sphere Re300 | 1 | 19.965 | 1 | 28.752 |
| 3D cube Re3700 | 1 | 46.191 | 2 | 34.626 |
| 3D circular cylinders Re200/Re3900 | 2 | 64.389 | 1 / 2 | 42.610 / 88.976 |
| 3D square cylinder Re200 | 1 | 25.941 | 1 | 65.216 |
| 3D NACA0012 gust/no-gust wings | 9 | 295.057 | 1 or 2; see exact-case table | 5.560–41.762 |
| **Total** | **40** | **486.282** | | |

The six gust-wing variants are Re100/AOA20, Re100/AOA40, Re1000/AOA20,
Re1000/AOA40, Re10000/AOA12 and Re50000/AOA12. The three no-gust variants are
Re100/AOA20, Re100/AOA40 and Re1000/AOA20. Both higher-Re gust cases were
successfully measured on two distinct H100s, at 38.954 and 35.869 s/control.
Their cold initialization is approximately 33 minutes, separately recorded
from steady interaction and approximately one-minute second resets.

We deliberately retain the canonical 3D cylinder Re3900 despite its higher
cost. Its native episode extrapolates to about five wall-hours on two H100s;
this is a scientific-coverage tradeoff, not a claim that all cases are cheap.
No universal 30-second/control cutoff is used.

Cube Re300 also passed (43.435 s/control on one H100), but retaining it along
with all selected cases would total **500.372 GB**, exceeding the hard cap.
The higher-Re cube is retained instead. The remaining 13.718 GB is not filled
with unresolved configurations just to use disk space.

## What the cost tests establish

Each included case completed native reset/control/reset and clean process
exit, with at least two post-warmup samples. MAIA/JAX-Fluids exclude the first
control; JAX excludes the first two. Source grids, restarts, dt and native
actuation intervals were not shortened. Raw samples, cold/reset times,
effective profiles, immutable container IDs and raw-log hashes are retained.

An allocated second GPU is not enough for MAIA: each solver rank must be
explicitly bound to its own CUDA UUID. Incorrectly bound earlier attempts
were invalidated; all accepted two-GPU results have distinct verified rank
receipts and process-to-device checks. The reusable native launcher now
performs explicit rank binding.

The fine 2D and coarse 3D nozzle JSONs omit the flag enabling their supplied
restart. The [runtime-only binding](native_resource_bindings.py) enables that
unchanged source restart, without interpolation or editing any source byte.
The corrected fine 2D nozzle shows nonzero flow, changing fields and signed
thrust response. The coarse 3D nozzle passes too, but its measured 234.35
s/control on two H100s implies approximately 16 wall-hours per native episode;
it is excluded for cost. Its eight-actuator diagnostic is explicitly an
organizer choice within the supported range, not a recovered paper setup.

Kolmogorov uses the pinned official `minimize_tke` example (Re200, 64²,
float64, alpha=1), not the differing source YAML or paper Re250 setup.
Channel uses pinned class defaults (72³, 24 actions, dt=0.0002), not its
differing asset YAML. MAIA sensors are diagnostic coordinates, not certified
paper or evaluation sensors. These boundaries remain explicit.

The exact-case table estimates native episode integration by multiplying a
short measured rate by the native horizon. **Those episodes were not run.**
One episode per selected case sums to about **43.34 GPU-hours** before startup,
reset, learner/inference work and contention. Eight GPUs do not guarantee a
two-hour full benchmark or training run. Repeated samples within a short
trajectory are not independent full episodes or confidence intervals.

## Training and scoring

Training and evaluation use exactly the same selected environment identities
and supplied initial-state pools. This is known-task multi-task learning,
not an unseen-task or private-restart split. Interactive training generates
experience; the source resource count is not an offline transition budget.

The unchanged [hydrogym-policy/v1 contract](../contract.md) requires one
jointly trained shared checkpoint across every selected case. No model or
learned-normalizer updates are permitted during evaluation. Packing
separately trained experts into one file is not a shared model.

A separately reviewed `suite.json` must bind the exact release hash and all
40 cases, with accepted sensors, rewards, horizons, initialization and solver
launch settings. The evaluator must not silently drop an unsupported case.
Raw-return replay removes only the reference-anchor requirement, not full
physical acceptance. Historical paper curves are not scores for this release.

## Data budget and materialization

The 500 GB cap conservatively sums all named selected source files, without
compression or duplicate-file discounts. Training and evaluation share one
copy. Container images, caches, generated trajectories, policy checkpoints,
logs, solver scratch and preserved old archives are outside this source cap.

From the task repository root:

```bash
python3 fluid-flow-control/environment/data/public_release.py --check
python3 fluid-flow-control/environment/data/materialize.py /large-disk/hydrogym-public-coverage-v1
python3 fluid-flow-control/environment/data/materialize.py /large-disk/hydrogym-public-coverage-v1 \
  --from-local /large-disk/existing-hydrogym
```

The first materializer call prints a plan and downloads nothing. Local reuse
hard-links only the selected files on the same filesystem and verifies every
Git/LFS checksum. On a fresh machine use `--download` instead of `--from-local`;
the existing `hf` interface fetches only selected directories at pinned resource
revision `cf5ef6748443ae72e9647f398487e98d421a7243`. No full 1.09 TB archive
is needed. `--verify-only` works offline.

The local target is
`/mnt/workspace/hydrogym/releases/hydrogym-public-coverage-v1`.
Mount this selected root read-only at `/data` through `TASK_DATA_DIR`.
Both original files and hard-linked views must remain immutable; generated
work belongs elsewhere. Materialization never invents an unvalidated suite.

Manifest SHA256:
`365e9531671ae4eb24491e880f7e6c1d9768013b5da2bb389df9342525ca1896`.

Local materialization completed on 2026-09-07 at 22:22 UTC. All **576 files**
passed full offline Git/LFS hash verification. The view contains exactly the
40 selected directories and named source files; every file is a hard link to
the unchanged source archive. The receipt binds the manifest hash above and
explicitly records `benchmark_validated: false`; no `suite.json` was created.
Both historical manifest identities and data-root manifests remain unchanged.

## Historical versions

- First tested 28 cases / 138.269 GB: [manifest](release-speed-v1.json)
  and [specification](RELEASE-SPEED-V1.md).
- Earlier storage-only 47 cases / 488.771 GB: [manifest](release-500gb-v1.json)
  and [specification](RELEASE-500GB-V1.md). Its paper evidence stays bound there.

Historical manifests, data roots and unrelated specialist experiments are
preserved. To materialize an old version, pass its explicit `--release` path
and use a distinct destination. No external upload/push or new agent session
was performed by this audit.
