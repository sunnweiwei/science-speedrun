# Isolated verifier

The `main` container receives the relocated submission and the public verifier
view: spectra, metadata, candidate IDs, and candidate SMILES, below the root-only
`/verifier-input`. It validates the package and executes one of two replay
protocols:

1. **Persistent Single-Spectrum Streaming (`massspecgym-retrieval-stream/v2`, default when `serve.sh` or `predict_server.py` is present)**:
   Launches **1** persistent unprivileged (`UID 10001`, cleared supplementary
   groups, `no_new_privs`, `network_mode: none`) worker via `subprocess.Popen`
   (never `os.fork()` after CUDA initialization). Dispatches **1 anonymized
   spectrum + its 1 shuffled candidate SMILES pool at a time**, blocking until a
   valid full `ranking` permutation (`{"query_id", "ranking"}` only) is returned
   before sending the next query. Immediately after Pass 1 finishes, all
   `UID 10001` processes are `SIGKILL`ed and `/tmp` + System V IPC state is
   purged; up to 128 queries (`min(128, N)`), stratified to prioritize later
   occurrences in repeated/shared/isomeric candidate pools and shuffled with a
   private seed, are re-evaluated on a fresh cold-start worker to verify **exact
   full `ranking` list equality** (`StatefulQueryLeakageError` -> `reward: 0.0`,
   `score: 0.0` on any mismatch).
2. **Legacy Per-Pool Batch Fallback (`massspecgym-per-spectrum-batches/v2`, when `serve.sh` / `predict_server.py` is absent)**:
   Splits queries into batches with at most one spectrum per candidate pool
   (373 invocations on the full frozen test), writing a fresh input root per
   batch, running `infer.sh` as `UID 10001` with a fresh output and home
   directory, and killing `UID 10001` processes and purging `/tmp` + System V
   IPC state between batches.

`main` seals the merged `rankings.jsonl` with content hashes (`main/stream.json`
or `main/batches.json` records protocol metadata and timings; inference logs are
kept root-only in `main/logs/`).

The `grader` sidecar receives the same public input, the sealed snapshot, and a
separate private-label directory. It never mounts, imports, or executes the
submission. Named volumes carry only the immutable ranking snapshot and final
reports between services.

`reward.json` contains the raw per-spectrum Hit@1 under both `reward` and
`score`. `report.json` provides `score`, `reward`, Hit@1/5/20, MCES@1,
`protocol_mode` (`evaluation_protocol`), `statelessness_spot_check` (in streaming
mode), and batch metadata (in legacy fallback mode), plus the historical
reference-normalized score as a diagnostic. Missing rankings retain the full
official test denominator. A malformed/unknown/duplicate row or stateful leakage
invalidates the output (`reward: 0.0`, `score: 0.0`).

Invalid packages, inference timeouts and model-process failures produce a
numeric failure reward (`reward: 0`; report metrics: zero hit rates, MCES 100) and a typed `report.json` with
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
overall deadline from `task.toml` in the deployment. Compose declares GPU counts but
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

## Scientific & Literature Justification of Verifier Design

### 1. Primary Metric & Upstream Paper Provenance

- **Upstream Benchmark**: *MassSpecGym: A benchmark for the discovery and identification of molecules from mass spectra* (Bushuiev et al., NeurIPS 2024 Datasets and Benchmarks Track, MassSpecGym v1.5).
- **Primary Metric (`score` / `reward`)**: Mean per-spectrum top-1 retrieval accuracy (`Hit@1`, bounded in `[0, 1]`) across the full held-out test denominator of 17,556 experimental MS/MS spectra, matching `ssi_index_v3.massspec(hit_at_1)` identically (`score = Hit@1`).
- **Evaluation Unit**: Following Bushuiev et al. (2024), molecular retrieval from tandem mass spectrometry is defined strictly as a **single-spectrum, independent query-conditional** ranking task: given one experimental MS/MS fragmentation spectrum $(m/z, I)$, its measured precursor ion mass-to-charge ratio (`precursor_mz`), ionization adduct, instrument metadata, and a candidate pool of up to 256 constitutional/mass-window isomers, the model ranks the candidates for that spectrum alone without access to future spectra or cross-query test pool statistics.

### 2. Why Additional Ad-Hoc Constraints Are Intentionally Omitted

- **Byte-Identical Preservation of `precursor_mz` (Zero Synthetic Noise)**: In real high-resolution Orbitrap and Q-TOF tandem mass spectrometry, precursor $m/z$ is an intrinsic physical measurement used to compute neutral-loss channels ($\Delta m/z = \text{precursor\_mz} - m/z_i$), precursor mass error, and subformula assignments in chemically grounded forward simulators and fragmentation models (including MIST and GLACIER). Injecting synthetic jitter into `precursor_mz` would corrupt genuine physical neutral-loss features and deviate from the upstream MassSpecGym v1.5 specification. All experimental spectra, `precursor_mz` values, metadata, and candidate SMILES are therefore preserved 100% untouched.
- **Protocol Enforcement Instead of Metric Distortion**: Rather than altering inputs or inventing non-standard composite penalties, transductive multi-query shortcuts are addressed structurally at the execution boundary via `massspecgym-retrieval-stream/v2` (with legacy 373-batch isolation fallback `massspecgym-per-spectrum-batches/v2` when `serve.sh` is absent):
  1. **Blocking 1-at-a-Time Dispatch**: The worker sees strictly one anonymized spectrum and one shuffled candidate SMILES pool (`candidate_pool_id` stripped) per step and cannot observe query $i+1$ before committing a valid full permutation `{"query_id", "ranking"}` for query $i$ (eliminating lookahead while amortizing model load to a single persistent worker instead of 373 cold restarts).
  2. **128-Query Repeated-Pool Cold-Start Full-`ranking` Verification**: After Pass 1, all `UID 10001` processes, `/tmp` files, and System V IPC objects are destroyed, and up to 128 queries—stratified to prioritize later occurrences in shared/repeated/isomeric candidate pools and shuffled under a private seed—are re-evaluated on a fresh cold-start worker. Exact full `ranking` list equality (`ranking_pass2 == ranking_pass1`) is required (`StatefulQueryLeakageError` -> `score = 0.0`). Accurately scoped, this guarantee **blocks all future-query lookahead and verifies targeted state-independence across arrival order, candidate permutation, and repeated-pool history** using only the contracted `{"query_id", "ranking"}` output (no `scores` field required).

### 3. Diagnostic Signals Retained in report.json

While `reward.json` locks `score` and `reward` strictly to raw `Hit@1`, `report.json` retains full scientific and operational diagnostics:
- `metrics.hit_rate_at_5` (`Hit@5`) and `metrics.hit_rate_at_20` (`Hit@20`): top-$k$ candidate recall over the 17,556 test queries.
- `metrics.mces_at_1` (`MCES@1`): mean MyopicMCES molecular graph-edit distance between the top-1 predicted molecule and ground-truth structure (with fixed-denominator penalty 100.0 on missing/invalid predictions).
- `protocol_mode` (`evaluation_protocol`): records whether replay ran under `"massspecgym-retrieval-stream/v2"` or `"massspecgym-per-spectrum-batches/v2"`.
- `statelessness_spot_check`: records `checked_queries`, `repeated_pool_queries`, `mismatched_queries`, and `passed` for streaming verification.
- `reference_comparison` and `benchmark_score`: comparison against the frozen `0.4939052175894281` reference anchor and historical normalized diagnostic.
