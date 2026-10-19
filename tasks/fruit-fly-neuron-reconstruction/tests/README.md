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
