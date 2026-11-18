# Independent single-checkpoint verifier

`test.sh` invokes `verify_single.py`, using only the snapshotted submission,
private image/suite, operator-pinned runtime and fresh scratch. It uses the current single-checkpoint pipeline without a historical campaign
ledger or Slurm submission interface.

Set `TASK_DATA_DIR`, `TASK_PRIVATE_DIR` and `TASK_VERIFIER_SCRATCH` to disjoint
materialized/fresh directories. Harbor supplies the submission snapshot,
verifier logs, GPU allocation and network policy. For standalone Docker, bind
`/workspace/submission` read-only, `/logs/verifier` writable, and expose the
GPUs (eight in task.toml; the pipeline uses every exposed GPU) in addition to
these Compose resources. Run `/tests/test.sh`.

Prediction runs under bubblewrap with a mount allowlist, separate PID/network
namespaces and a clean environment. Only trusted code/runtime, image, staged
weights, devices and scratch are present. Package Python/native extensions are
ignored. The private suite is read only after prediction completes. There is
no non-isolated fallback when namespace setup fails. Deployment must permit
unprivileged user namespaces; runtime-specific NVIDIA mounts may need operator
configuration. Under Docker, the Compose file relaxes seccomp/AppArmor, unmasks
/proc and adds SYS_ADMIN for the namespace, and `test.sh` detaches the NVIDIA
runtime's /proc/driver/nvidia mounts.

`reward.json` is emitted only after complete fresh execution and scoring, or as
zero for invalid package data. Infrastructure and timeout failures produce no
reward; `status.json` explains the failure. Resource caps live in `limits.json`
and task.toml; the measured full run is below.

## Execution evidence

The formal route ran end to end under Docker on four H100s (2026-10-17/19) with
an agent package whose FFN follows the baseline recipe (legacy step 6750 plus
one production update) and whose SENSE/SHAPE were trained for minutes. Stage
times, in order: seeds 10 min, FFN 4 h 46 min (53.2 M calls, 16 workers under
MPS, `norm32`), stitch 22 min, context 23 min, SENSE 67 min (443,824 centers),
FFN-pair 0.5 min, SHAPE 0.5 min, finalize 2 min, scoring 2 min. Reward:
nERL 2.2e-5 (4,103 merge and 4,389 split errors, 47% omitted nodes; all three
methods equal). The run exposed four failures that are now fixed: shards
failed when a window reached the FFN call cap, 298 M sub-10-voxel fragments
exceeded the context label bound, the namespace lacked /etc/gitconfig, and the
frozen suite's common.py pin predated the track rename. After the third fix the
run resumed from stitching in the same namespace, and scoring ran alone after
the fourth; FFN was not repeated. Seven complete windows computed with `norm32`
on H100s match the eager backend bit for bit.
See [the run record](../docs/validation/docker-formal-run.md).

Earlier, native GPU reconstruction and nERL scoring have run, including the historical
full-volume comparison and newer single/eight-window FFN evaluations. See the
[results](../README.md#results-and-validation). These used native experiment
runners, not this exact shell entry.

Beyond inference and scoring, the `test.sh` → `verify_single.py` route adds
container-marker checks, fixed mount/scratch paths, data-only staging,
bubblewrap prediction isolation and the final reward/status interface.
`verify_single.py` rejects ordinary native execution before inference; this is
why a successful native experiment does not constitute running this wrapper.

Tests cover package hash/path rejection, exclusion of submitted code, native
allocation enforcement, process timeout cleanup, data visibility/materialization
and failure-output semantics; the real-namespace test runs inside the verifier
image. The Docker run above used the pinned FPS binary on H100s; other GPU
architectures need their own FPS ABI check.

The prepared public release passed anonymous download and per-file hash checks;
rebuilding it from upstream data is a separate operation. See
[validation scope and receipts](../docs/READINESS.md).

## Scientific & Literature Justification of Verifier Design

> **[STATUS: HOLD — Standalone Metric Redesign Pending]**
> Per SSI Benchmark V4 governance (§5.2), `fruit-fly-neuron-reconstruction` (`#06`) is currently **ON HOLD** pending a dedicated connectomics working-group review of its evaluation volume scale and sparse skeleton metric formulation. All verifier code, scoring logic, and evaluation volumes are **100% frozen** in V4 with zero metric or volume changes until the standalone connectomics redesign is finalized.

### 1. Primary Metric & Upstream Paper Provenance
- **Upstream Pipeline & Metric Definition**: Directly inherits the three-stage automated connectomics pipeline (**FFN** flood-filling segmentation -> **SENSE** merge-candidate proposal -> **SHAPE** point-cloud agglomeration) and **local sparse normalized Expected Run Length (nERL)** metric of **PATHFINDER** (*bioRxiv 2025.05.16.654254*) on 16 nm isotropic electron microscopy of the male *Drosophila* central nervous system (**MaleCNS**), building on **Flood-Filling Networks (FFN)** (*Januszewski et al., "High-precision automated reconstruction of neurons with flood-filling networks", Nature Methods 2018, 10.1038/s41592-018-0049-4*) and **Local Shape Descriptors (LSD)** (*Sheridan et al., "Local shape descriptors for neuron segmentation", Nature Methods 2023, 10.1038/s41592-022-01711-z*).
- **Why `[STATUS: HOLD — Standalone Metric Redesign Pending]`**: Independent V3/V4 audit identified that on a single small `8.2 µm` held-out micro-subvolume, sparse skeleton `SHAPE nERL` exhibits pathological finite-volume boundary behavior (where an image-blind $12^3$ regular cube grid can achieve higher local sparse `nERL` than learned neurite reconstructions by avoiding catastrophic false-merge truncation within short neurite chord segments). Because resolving this requires changing either the held-out physical volume extent/multi-region tiling or switching/combining topological metrics (e.g., Variation of Information VOI split/merge decomposition, edge accuracy, or merge-free ERL across larger neurite arbors), the metric and test volume are strictly frozen under `[STATUS: HOLD]` rather than applying ad-hoc patches.

### 2. Why Additional Ad-Hoc Constraints Are Intentionally Omitted
- **Zero Unapproved Ad-Hoc Patches While Under Hold**: Rather than introducing arbitrary heuristic filters (such as ad-hoc segment-count floors, voxel-size penalties, or post-hoc merge caps) that have no basis in *Januszewski et al. (2018)*, *Sheridan et al. (2023)*, or *PATHFINDER (2025)* and could distort downstream proofreading cost trade-offs, the existing `verify_single.py` single-checkpoint Bubblewrap verifier and container isolation boundary (`environment/Dockerfile` `development` target excludes `tests/`) are preserved verbatim until the connectomics redesign is ratified.
- **Data-Only Model Package Execution**: `verify_single.py` executes only trusted operator-pinned FFN/SENSE/SHAPE inference code on submitted weight tensors inside an isolated Bubblewrap namespace (`network` disabled, private reference suite unmounted during prediction), structurally preventing executable-code side channels without altering scientific scoring.

### 3. Diagnostic Signals Retained in `report.json`
- **Topological Proofreading Error Decomposition**: Alongside primary `SHAPE nERL` (`reward.json`), the evaluation outputs explicit topological error counts — **split errors**, **merge errors**, **omitted skeleton nodes / omission fraction**, and **voxel/skeleton model coverage** — across pre-agglomeration FFN, FFN-pair, and SHAPE-merged stages, plus stage-by-stage wall-clock timings and FFN call counts in `status.json`.
