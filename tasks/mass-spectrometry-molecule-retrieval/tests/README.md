# Isolated verifier

The `main` container receives the relocated submission and the public verifier
view: spectra, metadata, candidate IDs, and candidate SMILES. It validates the
package, runs `infer.sh` as UID 10001 with cleared supplementary groups and
`no_new_privs`, without network access, kills remaining
UID-10001 processes, and seals the ranking file with content hashes.

The `grader` sidecar receives the same public input, the sealed snapshot, and a
separate private-label directory. It never mounts, imports, or executes the
submission. Named volumes carry only the immutable ranking snapshot and final
reports between services.

`reward.json` contains only the canonical normalized `score`. `report.json`
retains raw Hit@1/5/20 and MCES@1, metric directions, score bounds/version,
unclipped score and saturation. The score is a linear transform of Hit@1
between the fixed DeepSets and GLACIER/MIST reference anchors, clipped to
[0, 1]; it is not a weighted combination of the scientific metrics. Missing rankings
retain the full official test denominator. A malformed/unknown/duplicate row
invalidates the ranking file, and missing queries receive misses and MCES 100.

Invalid packages, inference timeouts and model-process failures produce a
numeric failure reward (`score: 0`; report metrics: zero hit rates, MCES 100) and a typed `report.json` with
`model_quality_measured: false`. Input/setup/grader faults produce an
`infrastructure_error` report without a reward. The grader publishes JSON
atomically and signals errors to the waiting main service. The trusted log
directory is owned by the verifier and is not writable by inference UID 10001;
the sealed volume is accessible only to the trusted verifier/grader. Stale
`reward.json` and `reward.txt` files are removed before replay.

Run lightweight source and synthetic-data tests from the repository root:

```bash
bash mass-spectrometry-molecule-retrieval/tests/run_unit_tests.sh
```

Check deployment networking in a Python environment containing the campaign's
Harbor and agent-runner installations (also requires PyYAML and Python 3.11+):

```bash
python mass-spectrometry-molecule-retrieval/tests/check_runner_network.py
# If runner is a source checkout rather than installed in this environment:
python mass-spectrometry-molecule-retrieval/tests/check_runner_network.py \
  --runner-source /absolute/agent-runner
```

This uses actual runner staging and Harbor service-selection code with temporary
task copies. It checks allowlist/no-network at 0, 1 and 8 GPUs, both offline
verifier services, and reproduces the old explicit-Compose-network conflict.
It starts no containers and makes no API calls. On the deployment host, also
check permitted API access and rejected non-allowlisted access inside the agent
container, plus offline inference/grading; a passing source check does not
replace those runtime checks. Start development through agent-runner, since
bare Compose does not apply the task TOML or Harbor's egress policy.

Run a full replay after materializing data and training a package:

This command requires Docker with NVIDIA Container Toolkit and an H100. The
main service requests one GPU, 16 CPUs and 64 GiB RAM; the CPU-only grader has
its own 16 CPU / 16 GiB ceiling. Enforce the 50 GiB writable replay quota and
six-hour overall deadline in the deployment. Compose declares GPU counts but
does not check GPU models or enforce disk quotas. Start every independent
attempt with fresh named volumes (use a distinct Compose project name).

For the native scratch package, new publications default to inference batch 64
with two workers. The [PR 27 reviewer](https://github.com/T0-RSI/ai4sci-tasks/pull/27#issuecomment-5750791449)
reported full 17,556-query replay in 3.4–3.9 hours on one H100 at this batch,
versus about five hours at batch 8. Operators can republish with a positive
`--inference-batch-size` override for less GPU memory. Record the package's
actual setting and measure the complete current Docker/HF replay; reviewer
timing alone does not accept the current image, inputs, isolation or quotas.

```bash
export TASK_DATA_DIR=/absolute/public-root
export TASK_PRIVATE_DIR=/absolute/private-root
export MASSSPECGYM_SUBMISSION_DIR=/absolute/submission
export MASSSPECGYM_REPLAY_PROJECT=massspecgym-check-UNIQUE_ATTEMPT_ID
replay_compose=(docker compose --project-name "$MASSSPECGYM_REPLAY_PROJECT" \
  -f mass-spectrometry-molecule-retrieval/tests/docker-compose.yaml \
  -f mass-spectrometry-molecule-retrieval/tests/docker-compose.local.yaml)
"${replay_compose[@]}" up -d --build --wait
"${replay_compose[@]}" exec -T main /tests/test.sh
"${replay_compose[@]}" cp main:/logs/verifier /absolute/new-replay-results
"${replay_compose[@]}" down --volumes
```

The local override mounts `MASSSPECGYM_SUBMISSION_DIR` for this manual command
only and keeps `main` alive for explicit verifier execution. Do not use
`--abort-on-container-exit`: the grader normally exits before main copies its
reward, so that option can stop a successful replay before collection. Preserve
logs before removing this attempt's volumes, including on verifier failure.
During a native Harbor run, `artifacts = ["/workspace/submission"]` moves
the package into the separate verifier environment at the same path. Keeping
that path free of a read-only Compose bind is required for Harbor's artifact
upload to succeed.

The native model uses `/opt/glacier/bin/python` in both images. A fresh host
runtime and rootless namespace replay can be checked on a cluster without
Docker using `replay_local.py --submission PACKAGE --input PUBLIC_INPUT
--output NEW_RUN_DIR --runtime ROOT/glacier [--gpu]`. It binds only the selected
runtime/CPython, system libraries, the read-only package and public inputs;
network, host processes, home, training data and private labels are absent.
This is additional portability/isolation evidence, not Docker or H100 deadline
acceptance. `replay.json` records that distinction explicitly.

For the real native CPU recovery regression, first prepare a small 64/16-row
MAGMa view using `baseline/starter/glacier/prepare.py --limit-train 64
--limit-validation 16`, then run with the native interpreter:

```bash
/opt/glacier/bin/python mass-spectrometry-molecule-retrieval/tests/check_training_recovery.py \
  --data-root /absolute/small-prepared-view --fallback /absolute/trained-fallback \
  --output /absolute/new-recovery-check
```

It compares continuous and resumed four-step training, including parameter
tensors, data order, RNG, optimizer-step count and the native MAGMa schedule.
These small checks validate the workflow and do not create a scientific score.
