# Independent verifier

`OLMO_INFERENCE_HASHSEED=0 bash tests/test.sh` reads `OLMO_MODEL` (default `/workspace/submission`),
`OLMO_EVAL_DATA` (default `/data/eval`), and writes metrics, `reward.json` (numbers only, as Harbor requires) and `report.json` (status, reasons, hashes)
to `OLMO_VERIFIER_OUTPUT` (default `/logs/verifier`). Set the declared runtime's
Python on PATH. All CUDA execution must be inside Slurm the operator-selected GPU partition.
Both verifier build definitions set this inference hash seed. It fixes upstream
modality ordering between Python processes, without changing model initialization.
Use the same GPU type, batch size and hash seed for paired encoder comparisons.
Verifier v6 sets this environment value and has reproduced complete AWF features
bit-for-bit on A5000; final selected-model LFMC scoring follows confirmation.

For each task, the verifier reruns `infer.py` from the exported package inside
bubblewrap with no network and no labels or training data mounted. It then fits
probes in a separate process using downstream training labels. LFMC probe
regularization is selected on validation; test scoring follows checkpoint selection.
AWF is a development validation metric, not a hidden-test score.

Reward is the mean of AWF validation accuracy and LFMC test MAE skill relative
to the downstream train-mean predictor: `0.5*(accuracy + max(0,1-MAE/constant_MAE))`.
Native task metrics and every probe search result remain available. Malformed packages and predictions produce reward 0 with an explicit reason.
Setup, data, namespace and unexplained process failures exit without a reward;
inspect their logs before attributing the failure to a candidate.

The task developer can inspect public labels; this is an inference isolation
boundary, not a claim that the public benchmark is blind to the developer.
The output must contain each requested ID exactly once and finite embeddings.
`test_contract.py` checks row reordering, missing/duplicate IDs, nonfinite output,
and that changing final-test labels cannot change selected probe parameters.
`test_inference.py --model PACKAGE --output REPORT.json` additionally checks the
exported encoder on CPU for batch independence and correct missing-observation
handling. Its small synthetic inputs require no evaluation labels or imagery.

`test_full_recomputation.py` compares the official complete Large model against
its recomputed form, with a native-repeat control. It uses a small real masked
batch and checks every parameter gradient. Run it only in the operator-selected GPU partition;
`--microbatches 8` additionally exercises the official FSDP accumulation context.
The real microbatch is reused for this numerical regression. Running with
`torchrun --standalone --nproc_per_node=2` exercises two FSDP ranks, using two
real examples per rank and recording each rank’s losses and gradient comparisons.
Use `NCCL_P2P_DISABLE=1` on the current cluster, matching the training jobs.
This is a numerical implementation check, not a training-quality benchmark.

## Separate replay from final scoring

The default grader still executes the complete pipeline. Optional
`--inference-only` replays both complete input releases and writes
`inference-manifest.json`, without loading labels or fitting probes.
`--score-existing` verifies SHA-256 hashes of model files, unlabeled inputs,
feature files, replay/namespace/grader code and probe code, then performs the
same frozen-probe scoring on CPU. It rejects changed artifacts and completed
score directories. Scores include the replay-manifest and label-file hashes.
The regression test exercises real CPU probes, checks agreement with the default
single-stage path, and rejects altered weights, inputs, features and scoring code.

With verifier v7, `OLMO_VERIFIER_PHASE=inference` selects this mode in
`docs/slurm/container_verify.sbatch`. Only the two unlabeled H5 files are mounted as
data in that phase. After recipe confirmation, submit `docs/slurm/container_score.sbatch`
with the same `OLMO_MODEL` and `OLMO_OUTPUT`. It requests CPU resources only,
uses no `--nv`, and checks the existing manifest before scoring. This scoring
phase reads LFMC test labels; it must follow the experiment's selection lock.
Do not confuse a completed inference manifest with a completed evaluation.


## Deployment checks

`docker-compose.yaml` builds from the task root and mounts `TASK_PRIVATE_DIR`
read-only at `/data/eval`. The controller supplies a complete submission snapshot
at `/workspace/submission` and collects `/logs/verifier`; private data never
enters the agent Compose service. Both environments declare offline execution in
`task.toml`. The verifier uses the weather track's outer-container namespace
settings; its inner model namespace drops capabilities and unshares networking.

Docker execution on the development cluster is unavailable. Container acceptance
is best effort and is not a merge gate for this delivery; see
[readiness](../docs/READINESS.md) for checks and remaining limitations.
